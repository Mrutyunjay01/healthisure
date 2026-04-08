---
title: Healthisure
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Healthisure

A fully **OpenEnv-compliant** reinforcement learning environment that simulates an AI-powered health insurance support specialist. Built for the Meta OpenEnv Hackathon.

---

## Overview

Healthisure places an RL agent in the role of a health insurance support specialist handling real-world member queries: benefit verification, prior authorization disputes, claim reviews, and multi-party coordination of benefits (COB). The environment is **stateful and reactive** — each agent action changes the episode state, triggers consequences, and earns a shaped reward signal.

Key properties:

- **Three tasks** of increasing difficulty (eligibility → prior auth → COB dispute)
- **12 action types** for information retrieval, document drafting, and case resolution
- **Shaped rewards** on every step — not just at episode end
- **Deterministic grading** — no LLM in the reward loop
- **OpenEnv SDK** (`openenv-core`) compliant; deployable to Hugging Face Spaces

---

## Environment Details

| Property | Value |
|---|---|
| Spec Version | 1 |
| Runtime | FastAPI (uvicorn) |
| Port | 7860 (Hugging Face Spaces default) |
| Tasks | 3 (task1, task2, task3) |
| Step budgets | 10 / 15 / 20 |
| Action space | 12 discrete named actions |
| Reward range | [−0.35, 1.0] per episode |

---

## Three Tasks

### Task 1 — Benefit Verification & Eligibility (`task1`, budget=10)

**Difficulty:** Easy

The member asks about the cost of a procedure. The agent must:

1. Look up the member and their plan
2. Check deductible status
3. Determine if prior authorization is required
4. Calculate the member's cost share
5. Deliver a clear, accurate response

**Scenarios (3):** MRI on Gold plan, knee replacement on Silver (deductible met), office visit on Bronze

**Max score:** 1.0

---

### Task 2 — Prior Auth & Claim Status Resolution (`task2`, budget=15)

**Difficulty:** Medium

A claim has been denied. The agent must investigate whether the denial is correct or erroneous and take appropriate action (explain the correct denial, or draft an appeal letter with regulatory citations).

**Scenarios (3):** CO-4 correct denial, CO-4 erroneous denial (PA was submitted but not recorded), CO-50 mental health parity violation (ACA/MHPAEA citation required)

**Max score:** 1.0

---

### Task 3 — Multi-Party COB Dispute (`task3`, budget=20)

**Difficulty:** Hard

An ER claim has a coordination of benefits issue: primary paid partial, secondary hasn't responded, and the provider is threatening collections. The agent must:

- Determine COB order
- Apply ACA ER parity rules
- Invoke a deadline exception (provider delay = excused)
- File a corrected claim with the secondary insurer
- Draft a collections dispute letter
- Escalate the case internally

**Scenarios (2):** Standard COB dispute, COB + preventive care complication

**Max score:** 1.0

---

## Action Space

| Action | Parameters | Description |
|---|---|---|
| `lookup_member` | `member_id` | Fetch member profile and plan |
| `lookup_plan_benefits` | `plan_id, cpt_code` | Get coverage and cost-share details |
| `check_claim_status` | `claim_id` | Get claim status and denial codes |
| `decode_denial_code` | `code` | Explain a denial reason code |
| `check_prior_auth_required` | `cpt_code, plan_id` | Check if PA is needed |
| `check_deductible_status` | `member_id` | Get deductible met/remaining |
| `apply_cost_share` | `amount, plan_id, deductible_met, [cpt_code], [member_id]` | Calculate member's out-of-pocket |
| `draft_appeal_letter` | `claim_id, reason, citation` | Generate regulatory appeal |
| `draft_dispute_letter` | `provider_id, claim_id, reason` | Draft collections stop letter |
| `escalate_case` | `member_id, reason, priority` | Flag for internal escalation |
| `file_corrected_claim` | `member_id, claim_id, secondary_insurer_id` | File with secondary insurer |
| `send_member_response` | `message` | Final response (terminal action) |

---

## Reward Structure

| Event | Reward |
|---|---|
| Correct member/plan lookup | +0.05–0.10 |
| Correct deductible/benefit lookup | +0.10–0.15 |
| PA flag correctly raised | +0.10–0.15 |
| Correct denial decode | +0.10 |
| Correct cost-share calculation | +0.15–0.20 |
| Valid appeal letter with citation | +0.20 |
| Full resolution (send_member_response) | +0.20–0.30 |
| Incorrect COB order | −0.20 |
| Missed PA requirement | −0.15 |
| Hallucinated figure | −0.20 |
| Exceeded step budget | −0.10 |

---

## Setup

### Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose (recommended for running the server)

### Install dependencies

```bash
uv sync
```

### Configure environment variables

```bash
cp .env.example .env   # then edit .env to add your HF_TOKEN
```

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(required)* | Hugging Face API token |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | LLM model name |
| `ENV_URL` | `http://localhost:8000` | Healthisure server URL used by inference.py |
| `LOCAL_IMAGE_NAME` | *(optional)* | Local Docker image tag to auto-run when compose is unavailable (e.g. `healthisure:latest`) |

---

## Running the Server

### Option A — Docker Compose (recommended)

```bash
docker compose up --build          # start server (Ctrl+C to stop)
docker compose up --build -d       # start in background
docker compose logs -f server      # tail logs
docker compose down                # stop and remove containers
```

The server is available at **http://localhost:8000** (mapped from container port 7860).  
The Gradio web UI is at **http://localhost:8000/web** (enabled by default in compose).

### Option B — uv (local development, no Docker)

```bash
# Plain server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

# With Gradio web UI at /web
ENABLE_WEB_INTERFACE=true uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## Running Inference

`inference.py` connects to the server via `ENV_URL` using the OpenEnv WebSocket client.
**It auto-starts the server if not already running** — trying `docker compose` first, then falling back to a local uvicorn subprocess:

```bash
uv run python inference.py
```

To connect to an already-running server, just set `ENV_URL`:

```bash
# Local docker compose (default)
ENV_URL=http://localhost:8000 uv run python inference.py

# Remote HF Space
ENV_URL=https://your-space.hf.space uv run python inference.py
```

---

## Docker (without Compose)

```bash
# Build
docker build -t healthisure .

# Run (maps host port 8000 → container port 8000)
docker run -p 8000:8000 --env-file .env healthisure
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check → `{"status":"healthy"}` |
| `/reset` | POST | Start a new episode. Query param: `task_name` |
| `/step` | POST | Execute an action |
| `/state` | GET | Current episode state |
| `/schema` | GET | Action and observation JSON schemas |
| `/ws` | WebSocket | Persistent stateful session (used by `HealthisureEnvClient`) |
| `/web` | GET | Gradio playground UI (enabled via `ENABLE_WEB_INTERFACE=true`) |

> **Note:** Use `/ws` (WebSocket) for multi-step episodes. The HTTP `/reset` and `/step` endpoints create a new environment instance per request and cannot maintain state across calls.

### Client-Server Architecture

`inference.py` uses `HealthisureEnvClient` from `client.py`, which wraps the OpenEnv `EnvClient` over WebSocket:

```python
from client import HealthisureEnvClient

with HealthisureEnvClient(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_name="task1")
    obs = result.observation
    # ... episode loop ...
    result = env.step(action)
```

### Example: Reset via HTTP

```bash
curl -X POST "http://localhost:8000/reset?task_name=task1"
```

### Example: Step via HTTP

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_name": "lookup_member", "parameters": {"member_id": "M001"}}'
```

---

## Project Structure

```sh
meta-openenv-26/
├── models.py             # Shared: HealthisureAction + HealthisureObservation (Pydantic)
├── client.py             # Shared: HealthisureEnvClient (OpenEnv WebSocket client)
├── inference.py          # Baseline LLM agent (uses client.py)
├── Dockerfile            # Container definition (port 7860 for HF Spaces)
├── docker-compose.yml    # Compose: server on localhost:8000, Gradio at /web
├── openenv.yaml          # OpenEnv metadata
├── pyproject.toml        # uv project manifest
├── .env.example          # Environment variable template
├── README.md
├── docs/                 # Architecture and planning docs
└── server/
    ├── app.py            # FastAPI app (openenv-core create_app + Gradio at /web)
    ├── environment.py    # HealthisureEnvironment core
    ├── tasks/            # Task 1, 2, 3 scenario definitions + BaseTask
    ├── actions/          # 12 action handler implementations
    ├── graders/          # Deterministic step-level reward graders + BaseGrader
    └── data/             # Simulated JSON data store
```

---

## Baseline Performance

Initial baseline scores using `gpt-5.4-mini` (deterministic, temperature=0):

| Task | Reward | Success |
|---|---|---|
| task1 | ~0.55 | ✓ |
| task2 | ~0.40–0.60 | varies |
| task3 | ~0.30–0.50 | varies |

*Scores vary by scenario selected at reset (random).*

---

## Hugging Face Spaces Deployment

This environment is ready to deploy as a Docker-based HF Space:

1. **Create a new Space** with SDK = "Docker"
2. **Add the `openenv` topic tag** (required by the hackathon)
3. **Set Space secrets** (Settings → Variables and secrets):
   - `HF_TOKEN` — your Hugging Face token
   - `API_BASE_URL` — LLM endpoint (default: `https://router.huggingface.co/v1`)
   - `MODEL_NAME` — LLM model (default: `Qwen/Qwen2.5-72B-Instruct`)
4. **Push the code** — the Dockerfile will build automatically
5. **Ensure the Space is in "Running" state** before submission

The server listens on port **7860** (HF Spaces default). The Gradio web interface is available at `/web` for interactive testing.

To run inference against the deployed Space:

```bash
ENV_URL=https://<your-username>-<space-name>.hf.space uv run python inference.py
```

---

## Design Notes

- **No live databases or external APIs** — all data is served from `server/data/*.json`
- **Deterministic grading** — rewards are computed by rule-based graders, not LLMs
- **Stateful episodes** — use the WebSocket endpoint `/ws` (via `HealthisureEnvClient`) for multi-step episodes; HTTP `/reset` and `/step` are stateless
- **OpenEnv-core SDK** — uses `openenv-core[core]>=0.2.2` (not the PyPI `openenv` package)
- **Client-server separation** — `inference.py` never imports the environment directly; it only communicates via HTTP/WebSocket through `HealthisureEnvClient`
