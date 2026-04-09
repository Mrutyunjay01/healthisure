"""
Healthisure baseline inference script.

Runs an LLM agent (via OpenAI client) through all three tasks by connecting
to a running Healthisure server via the OpenEnv WebSocket client protocol.

Auto-start priority (if the server is not already running at ENV_URL):
  1. docker compose up -d  (if Docker is available and docker-compose.yml exists)
  2. uvicorn subprocess    (fallback for environments without Docker)

Set ENV_URL to connect to an already-running server (local or HF Space) and
the auto-start logic is skipped entirely.

Output format (per OpenEnv hackathon guidelines):
  [START] task=<task_name> env=healthisure model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
  API_BASE_URL  — LLM API endpoint   (default: https://router.huggingface.co/v1)
  MODEL_NAME    — LLM model name     (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      — API token          (required)
  ENV_URL       — Env server base URL (default: http://localhost:8000)
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Load .env if present (graceful — works without python-dotenv installed too)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:8000")
# Optional: name/tag of a locally-built Docker image to run when the server
# isn't already up and docker compose is not available (e.g. "healthisure:latest").
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required (set it in .env or export it)")

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Server lifecycle — auto-start via docker compose or uvicorn subprocess
# ---------------------------------------------------------------------------

_server_proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
_started_with_compose: bool = False
_docker_run_container_id: Optional[str] = None


def _server_is_up(url: str, timeout: float = 2.0) -> bool:
    """Return True if the /health endpoint responds OK."""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _docker_compose_available() -> bool:
    """Return True if `docker compose` CLI is available."""
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _wait_for_server(url: str, attempts: int = 30, interval: float = 0.5) -> bool:
    """Poll /health until the server is up or we exhaust attempts."""
    for _ in range(attempts):
        time.sleep(interval)
        if _server_is_up(url):
            return True
    return False


def ensure_server(url: str = ENV_URL) -> None:
    """Start the server if it isn't already running at *url*.

    Tries docker compose first; falls back to a uvicorn subprocess.
    """
    global _server_proc, _started_with_compose

    if _server_is_up(url):
        return  # already up — user-managed server or HF Space

    # Give a Docker-managed or slow-starting server up to 30 s to respond
    # before deciding to auto-start a new one.  uvicorn + openenv + Gradio
    # can take 10–20 s to bind the port even though the container is running.
    print(
        f"[INFO] Server not immediately available at {url} — waiting up to 30 s for it to come up…",
        file=sys.stderr,
        flush=True,
    )
    if _wait_for_server(url, attempts=30, interval=1.0):
        print(f"[INFO] Server is up at {url}.", file=sys.stderr, flush=True)
        return

    print(f"[INFO] Server not detected at {url} — starting automatically…", file=sys.stderr, flush=True)

    if LOCAL_IMAGE_NAME:
        _start_with_docker_run(LOCAL_IMAGE_NAME, url)
    elif _docker_compose_available():
        _start_with_compose(url)
    else:
        _start_with_uvicorn(url)


def _start_with_docker_run(image_name: str, url: str) -> None:
    """Start the server by running a local Docker image with ``docker run``."""
    global _docker_run_container_id

    # Parse host port from url (default 8000)
    port = "8000"
    try:
        parsed_port = url.rsplit(":", 1)[-1].split("/")[0]
        if parsed_port.isdigit():
            port = parsed_port
    except Exception:
        pass

    print(f"[INFO] Starting local Docker image '{image_name}' on port {port}…", file=sys.stderr, flush=True)

    cmd = [
        "docker", "run", "--rm", "-d",
        "-p", f"{port}:{port}",
        "-e", f"PORT={port}",
    ]
    # Forward relevant env vars into the container
    for var in ("HF_TOKEN", "API_BASE_URL", "MODEL_NAME", "ENABLE_WEB_INTERFACE"):
        val = os.getenv(var)
        if val:
            cmd += ["-e", f"{var}={val}"]
    cmd.append(image_name)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        _docker_run_container_id = result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"docker run {image_name} failed — see output above."
        ) from exc

    if not _wait_for_server(url, attempts=40, interval=1.0):
        if _docker_run_container_id:
            subprocess.run(["docker", "stop", _docker_run_container_id], check=False)
            _docker_run_container_id = None
        raise RuntimeError(
            f"Server from image '{image_name}' did not become healthy within 40 s."
        )

    print(f"[INFO] Server is up (docker run / {image_name}).", file=sys.stderr, flush=True)


def _start_with_compose(url: str) -> None:
    """Start the server service via docker compose."""
    global _started_with_compose

    print("[INFO] Using docker compose to start the server…", file=sys.stderr, flush=True)
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d", "--build", "server"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("docker compose up failed — see output above.") from exc

    _started_with_compose = True

    if not _wait_for_server(url, attempts=40, interval=1.0):
        subprocess.run(["docker", "compose", "stop", "server"], check=False)
        _started_with_compose = False
        raise RuntimeError(f"Server at {url} did not become healthy within 40 s after docker compose up.")

    print("[INFO] Server is up (docker compose).", file=sys.stderr, flush=True)


def _start_with_uvicorn(url: str) -> None:
    """Start the server as a local uvicorn subprocess."""
    global _server_proc

    print("[INFO] Using uvicorn subprocess to start the server…", file=sys.stderr, flush=True)

    # Parse port from URL
    port = "8000"
    try:
        parsed_port = url.rsplit(":", 1)[-1].split("/")[0]
        if parsed_port.isdigit():
            port = parsed_port
    except Exception:
        pass

    _server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not _wait_for_server(url, attempts=30, interval=0.5):
        _server_proc.terminate()
        _server_proc = None
        raise RuntimeError(f"Server at {url} did not start within 15 s. Check server logs.")

    print("[INFO] Server is up (uvicorn).", file=sys.stderr, flush=True)


def stop_server() -> None:
    """Stop the auto-started server (docker compose, docker run, or uvicorn subprocess)."""
    global _server_proc, _started_with_compose, _docker_run_container_id

    if _started_with_compose:
        print("[INFO] Stopping docker compose server…", file=sys.stderr, flush=True)
        subprocess.run(["docker", "compose", "stop", "server"], check=False)
        _started_with_compose = False

    if _docker_run_container_id:
        print(f"[INFO] Stopping docker container {_docker_run_container_id[:12]}…", file=sys.stderr, flush=True)
        subprocess.run(["docker", "stop", _docker_run_container_id], check=False)
        _docker_run_container_id = None

    if _server_proc is not None:
        _server_proc.terminate()
        _server_proc = None


# ---------------------------------------------------------------------------
# OpenEnv typed client
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import HealthisureEnvClient  # noqa: E402
from models import HealthisureAction  # noqa: E402

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI-assisted health insurance support specialist.
Your job is to resolve member support cases by calling the available actions one at a time.

Available actions and their required parameters (use EXACT parameter names):
  lookup_member(member_id: str)
  lookup_plan_benefits(plan_id: str, cpt_code: str)
  check_claim_status(claim_id: str)
  decode_denial_code(code: str)
  check_prior_auth_required(cpt_code: str, plan_id: str)
  check_deductible_status(member_id: str)
  apply_cost_share(amount: float, plan_id: str, deductible_met: bool, cpt_code: str = null, member_id: str = null)
  draft_appeal_letter(claim_id: str, reason: str, citation: str)
  draft_dispute_letter(provider_id: str, claim_id: str, reason: str)
  escalate_case(member_id: str, reason: str, priority: str)
  file_corrected_claim(member_id: str, claim_id: str, secondary_insurer_id: str)
  send_member_response(message: str)

IMPORTANT — Parameter formats:
- plan_id is an internal code like "GOLD-001", "SILVER-001", "SILVER-002", "BRONZE-001".
  It is shown in brackets after the plan name in lookup_member output: e.g. "Silver Choice Plan [plan_id=SILVER-001]".
  NEVER use the plan name as plan_id.
- member_id is a code like "M001", "M002", "M003", "M004", "M005".
- claim_id is a code like "CLM-001", "CLM-002", etc.
- cpt_code is a numeric string like "70553", "27447", "99213".

Response format — output EXACTLY this JSON (no extra text):
{
  "action_name": "<action_name>",
  "parameters": { <key>: <value>, ... }
}

Rules:
- Call send_member_response as your FINAL action when you have fully resolved the case.
- Always look up the member first, then gather information before responding.
- For prior auth cases: check_claim_status → decode_denial_code → check_prior_auth_required → (draft_appeal_letter if erroneous) → send_member_response.
- For COB cases: look up member (note secondary_insurer_id), check claim, look up both plans, apply cost share, file corrected claim (use secondary_insurer_id from member lookup), draft dispute letter, escalate, then respond.
- Only output valid JSON. No markdown, no explanation outside the JSON.
"""


def build_user_prompt(
    task_description: str,
    step_count: int,
    step_budget: int,
    last_result: Optional[str],
    member_context: Optional[str],
    history: List[Dict[str, Any]],
) -> str:
    parts = [f"=== CASE ===\n{task_description}"]

    if member_context:
        parts.append(f"\n=== INFORMATION GATHERED SO FAR ===\n{member_context}")

    if history:
        recent = history[-3:]
        hist_str = "\n".join(
            f"  Step {h['step']}: {h['action']}({json.dumps(h['parameters'])}) → reward {h['step_reward']:+.2f}"
            for h in recent
        )
        parts.append(f"\n=== RECENT ACTIONS ===\n{hist_str}")

    if last_result:
        parts.append(f"\n=== LAST RESULT ===\n{last_result[:800]}")

    parts.append(
        f"\n=== STATUS ===\n"
        f"Step {step_count} of {step_budget} (budget remaining: {step_budget - step_count})\n"
        "Now output your next action as JSON."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------


def choose_action(
    task_description: str,
    step_count: int,
    step_budget: int,
    last_result: Optional[str],
    member_context: Optional[str],
    history: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], str]:
    """Call the LLM; return (action_name, parameters, display_str)."""
    user_msg = build_user_prompt(
        task_description, step_count, step_budget, last_result, member_context, history
    )
    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        err_msg = str(e)
        if "model_not_supported" in err_msg or "not supported" in err_msg.lower():
            print(
                f"\n[ERROR] Model '{MODEL_NAME}' is not available on {API_BASE_URL}.\n"
                "  → Update MODEL_NAME in your .env file.\n"
                "  → Reliable HF serverless models: Qwen/Qwen2.5-72B-Instruct, "
                "meta-llama/Llama-3.3-70B-Instruct\n",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1) from e
        raw_err = err_msg[:120].replace("\n", " ")
        return "send_member_response", {"message": f"LLM error: {raw_err}"}, f"llm_error({raw_err})"

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return "send_member_response", {"message": "Could not parse LLM response."}, raw[:80]

    try:
        parsed = json.loads(json_match.group())
        action_name = parsed.get("action_name", "send_member_response")
        parameters = parsed.get("parameters", {})
        raw_str = f"{action_name}({json.dumps(parameters, separators=(',', ':'))})"
        return action_name, parameters, raw_str
    except json.JSONDecodeError:
        return "send_member_response", {"message": "JSON parse error."}, raw[:80]


# ---------------------------------------------------------------------------
# Run one task episode (via server client)
# ---------------------------------------------------------------------------


def run_task(task_name: str) -> None:
    """Run a full episode for the given task and print OpenEnv-format output."""
    print(f"[START] task={task_name} env=healthisure model={MODEL_NAME}", flush=True)

    rewards: List[float] = []
    history: List[Dict[str, Any]] = []
    total_steps = 0
    success = False
    final_score = 0.0

    try:
        with HealthisureEnvClient(base_url=ENV_URL).sync() as env:
            result = env.reset(task_name=task_name)
            obs = result.observation

            for step_num in range(1, obs.step_budget + 2):
                if obs.done:
                    break

                action_name, parameters, action_str = choose_action(
                    task_description=obs.task_description,
                    step_count=obs.step_count,
                    step_budget=obs.step_budget,
                    last_result=obs.last_action_result,
                    member_context=obs.member_context,
                    history=history,
                )

                action = HealthisureAction(action_name=action_name, parameters=parameters)
                result = env.step(action)
                obs = result.observation

                step_reward = round(result.reward or 0.0, 2)
                rewards.append(step_reward)
                total_steps = step_num

                error_str = obs.error if obs.error else "null"
                done_str = "true" if obs.done else "false"

                print(
                    f"[STEP] step={step_num} action={action_str} "
                    f"reward={step_reward:.2f} done={done_str} error={error_str}",
                    flush=True,
                )

                history.append({
                    "step": step_num,
                    "action": action_name,
                    "parameters": parameters,
                    "step_reward": step_reward,
                })

            final_score = min(max(obs.cumulative_reward, 0.0), 1.0)
            success = obs.done and final_score > 0

    except Exception as e:
        print(f"[ERROR] Episode failed: {e}", file=sys.stderr, flush=True)

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={total_steps} score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ensure_server(ENV_URL)
    try:
        for task in ["task1", "task2", "task3"]:
            run_task(task)
            print(flush=True)
    finally:
        stop_server()

