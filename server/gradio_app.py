"""
Custom Gradio UI for the Healthisure.

Replaces the default openenv-core Gradio playground with a domain-specific
interface featuring:
  - Action dropdown (12 valid actions)
  - Dynamic parameter fields (show/hide based on selected action)
  - Per-step reward history table
  - Task selector with budget-aware episode controls
  - Live observation panels (task description, member context)

Design is forward-compatible with:
  - Proper reward function inspection (reward_history stored as structured dicts)
  - GRPO-like rollout collection (reward_history mirrors a rollout buffer)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import gradio as gr
import pandas as pd

# ---------------------------------------------------------------------------
# Action / parameter specification
# ---------------------------------------------------------------------------

VALID_ACTIONS: List[str] = [
    "lookup_member",
    "lookup_plan_benefits",
    "check_claim_status",
    "decode_denial_code",
    "check_prior_auth_required",
    "check_deductible_status",
    "apply_cost_share",
    "draft_appeal_letter",
    "draft_dispute_letter",
    "escalate_case",
    "file_corrected_claim",
    "send_member_response",
]

# Ordered list of all unique parameter field names used across actions.
# The order here determines the positional mapping in Gradio component lists.
ALL_PARAM_FIELDS: List[str] = [
    "member_id",
    "claim_id",
    "provider_id",
    "code",
    "secondary_insurer_id",
    "plan_id",
    "cpt_code",
    "amount",
    "deductible_met",
    "priority",
    "citation",
    "reason",
    "message",
]

# Fields that are active (visible) for each action.
# "optional" fields are still shown but labeled as optional.
ACTION_PARAM_FIELDS: Dict[str, List[str]] = {
    "lookup_member": ["member_id"],
    "lookup_plan_benefits": ["plan_id", "cpt_code"],
    "check_claim_status": ["claim_id"],
    "decode_denial_code": ["code"],
    "check_prior_auth_required": ["cpt_code", "plan_id"],
    "check_deductible_status": ["member_id"],
    # apply_cost_share: cpt_code and member_id are optional but shown
    "apply_cost_share": ["amount", "plan_id", "deductible_met", "cpt_code", "member_id"],
    "draft_appeal_letter": ["claim_id", "reason", "citation"],
    "draft_dispute_letter": ["provider_id", "claim_id", "reason"],
    "escalate_case": ["member_id", "reason", "priority"],
    "file_corrected_claim": ["member_id", "claim_id", "secondary_insurer_id"],
    "send_member_response": ["message"],
}

TASK_OPTIONS: List[str] = ["task1", "task2", "task3"]

TASK_LABELS: Dict[str, str] = {
    "task1": "Task 1 – Eligibility & Cost Share  (budget: 10 steps)",
    "task2": "Task 2 – Prior Auth & Denial  (budget: 15 steps)",
    "task3": "Task 3 – COB & Dispute  (budget: 20 steps)",
}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ── Layout ── */
.hc-header { text-align: center; padding: 8px 0 4px 0; }
.hc-status-bar { font-size: 0.92rem; }

/* ── Status pill colours ── */
.status-not-started { color: #888; }
.status-running     { color: #2563eb; font-weight: 600; }
.status-done-ok     { color: #16a34a; font-weight: 600; }
.status-done-fail   { color: #dc2626; font-weight: 600; }

/* ── Param group ── */
.param-group {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px;
    margin-top: 4px;
    background: #f8fafc;
}

/* ── Reward table ── */
.reward-table th { background: #f1f5f9 !important; }

/* ── Action result box ── */
.result-box textarea {
    font-family: monospace;
    font-size: 0.85rem;
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMPTY_STATE: Dict[str, Any] = {
    "started": False,
    "reward_history": [],   # List[{step, action, params, reward, cumulative, done}]
    "step_budget": 10,
    "step_count": 0,
    "cumulative_reward": 0.0,
}


def _history_to_rows(history: List[Dict]) -> List[List]:
    """Convert reward_history dicts to Dataframe rows."""
    return [
        [
            h["step"],
            h["action"],
            round(h["reward"], 4),
            round(h["cumulative"], 4),
            h["done"],
        ]
        for h in history
    ]


def _history_to_df(history: List[Dict]) -> pd.DataFrame:
    """Convert reward_history to a tidy DataFrame for LinePlot.

    Returns a long-format DataFrame with columns [Step, Value, Series]
    so both per-step reward and cumulative reward can be plotted as two lines.
    """
    if not history:
        return pd.DataFrame(columns=["Step", "Value", "Series"])
    rows = []
    for h in history:
        rows.append({"Step": h["step"], "Value": round(h["reward"], 4), "Series": "Step Reward"})
        rows.append({"Step": h["step"], "Value": round(h["cumulative"], 4), "Series": "Cumulative"})
    return pd.DataFrame(rows)


def _status_md(state: Dict) -> str:
    if not state["started"]:
        return "**Status:** ⚪ Not started — select a task and click **Reset Episode**"
    step = state["step_count"]
    budget = state["step_budget"]
    cumr = round(state["cumulative_reward"], 4)
    bar_filled = int((step / max(budget, 1)) * 10)
    bar = "█" * bar_filled + "░" * (10 - bar_filled)
    return (
        f"**Step:** {step} / {budget}  `{bar}`  "
        f"**Cumulative Reward:** `{cumr:+.4f}`"
    )


def _parse_observation(data: Dict) -> Dict[str, Any]:
    """Flatten step/reset response into easy-to-use dict."""
    obs = data.get("observation", {})
    return {
        "task_description": obs.get("task_description", ""),
        "member_context": obs.get("member_context") or "",
        "last_action_result": obs.get("last_action_result") or "",
        "step_count": obs.get("step_count", 0),
        "step_budget": obs.get("step_budget", 10),
        "cumulative_reward": obs.get("cumulative_reward", 0.0),
        "done": obs.get("done", False),
        "error": obs.get("error"),
        "task_name": obs.get("task_name", ""),
        "reward": data.get("reward", 0.0),
    }


# ---------------------------------------------------------------------------
# Gradio builder — called once at server startup
# ---------------------------------------------------------------------------

def build_healthisure_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str = "Healthisure",
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    """
    Custom Gradio UI builder for the Healthisure environment.

    This function is passed as `gradio_builder` to `create_app()` and is invoked
    once during server startup. The returned `gr.Blocks` is mounted at /web.

    Args:
        web_manager:   WebInterfaceManager — provides async reset_environment /
                       step_environment / get_state methods.
        action_fields: Field metadata list from openenv-core (not used directly;
                       we define our own typed fields).
        metadata:      EnvironmentMetadata (name, description, readme_content).
        is_chat_env:   Unused — Healthisure is always a structured-action env.
        title:         Display title for the Gradio app.
        quick_start_md: Quick-start markdown from openenv-core (shown in About).

    Returns:
        gr.Blocks instance to be mounted at /web.
    """

    # ── Event: update parameter field visibility ────────────────────────────

    def update_param_visibility(action: str):
        active = ACTION_PARAM_FIELDS.get(action, [])
        return [gr.update(visible=(f in active)) for f in ALL_PARAM_FIELDS]

    # ── Event: reset episode ────────────────────────────────────────────────

    async def reset_episode(task: str, state: Dict):
        try:
            data = await web_manager.reset_environment(
                reset_kwargs={"task_name": task}
            )
        except Exception as exc:
            err = f"❌ Reset failed: {exc}"
            new_state = {**EMPTY_STATE}
            return (
                err, "", "", _status_md(new_state),
                _history_to_rows([]), _history_to_df([]), new_state,
            )

        obs = _parse_observation(data)
        new_state: Dict[str, Any] = {
            "started": True,
            "reward_history": [],
            "step_budget": obs["step_budget"],
            "step_count": 0,
            "cumulative_reward": 0.0,
        }
        return (
            obs["task_description"],
            obs["member_context"],
            "Environment reset. Take your first action ▶",
            _status_md(new_state),
            _history_to_rows([]),
            _history_to_df([]),
            new_state,
        )

    # ── Event: execute action ───────────────────────────────────────────────

    async def execute_action(
        action: str,
        # positional args matching ALL_PARAM_FIELDS order:
        member_id: str,
        claim_id: str,
        provider_id: str,
        code: str,
        secondary_insurer_id: str,
        plan_id: str,
        cpt_code: str,
        amount: float,
        deductible_met: bool,
        priority: str,
        citation: str,
        reason: str,
        message: str,
        state: Dict,
    ):
        if not state.get("started"):
            return (
                state.get("_task_desc", ""),
                state.get("_member_ctx", ""),
                "⚠️  Please reset the environment first.",
                _status_md(state),
                _history_to_rows(state.get("reward_history", [])),
                _history_to_df(state.get("reward_history", [])),
                state,
            )

        # Build parameters dict — only include fields for the selected action
        all_values: Dict[str, Any] = {
            "member_id": member_id,
            "claim_id": claim_id,
            "provider_id": provider_id,
            "code": code,
            "secondary_insurer_id": secondary_insurer_id,
            "plan_id": plan_id,
            "cpt_code": cpt_code,
            "amount": amount,
            "deductible_met": deductible_met,
            "priority": priority,
            "citation": citation,
            "reason": reason,
            "message": message,
        }
        active_fields = ACTION_PARAM_FIELDS.get(action, [])
        params = {k: v for k, v in all_values.items() if k in active_fields}

        # Strip empty optional strings so handlers receive clean input
        params = {
            k: v for k, v in params.items()
            if v not in (None, "", 0.0) or k in ["deductible_met", "amount"]
        }

        action_data = {"action_name": action, "parameters": params}

        try:
            data = await web_manager.step_environment(action_data)
        except Exception as exc:
            result_text = f"❌ Step failed: {exc}"
            return (
                state.get("_task_desc", ""),
                state.get("_member_ctx", ""),
                result_text,
                _status_md(state),
                _history_to_rows(state.get("reward_history", [])),
                _history_to_df(state.get("reward_history", [])),
                state,
            )

        obs = _parse_observation(data)
        step_reward = obs["reward"]
        new_cumulative = obs["cumulative_reward"]

        # Build reward history entry (GRPO-ready format)
        history_entry: Dict[str, Any] = {
            "step": obs["step_count"],
            "action": action,
            "params": params,
            "reward": step_reward,
            "cumulative": new_cumulative,
            "done": obs["done"],
            # Snapshot for future reward function / training use
            "obs_snapshot": {
                "task_name": obs["task_name"],
                "step_count": obs["step_count"],
                "step_budget": obs["step_budget"],
                "member_context_len": len(obs["member_context"]),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        new_history = state.get("reward_history", []) + [history_entry]

        new_state: Dict[str, Any] = {
            "started": not obs["done"],
            "reward_history": new_history,
            "step_budget": obs["step_budget"],
            "step_count": obs["step_count"],
            "cumulative_reward": new_cumulative,
            "_task_desc": obs["task_description"],
            "_member_ctx": obs["member_context"],
        }

        # Result display
        result_parts = []
        if obs["error"]:
            result_parts.append(f"⚠️  Error: {obs['error']}")
        if obs["last_action_result"]:
            result_parts.append(obs["last_action_result"])
        if obs["done"]:
            emoji = "✅" if new_cumulative > 0 else "❌"
            result_parts.append(
                f"\n{emoji} Episode complete — "
                f"Total reward: {new_cumulative:+.4f} over {obs['step_count']} steps."
            )
        result_text = "\n\n".join(result_parts) or "(no result returned)"

        return (
            obs["task_description"],
            obs["member_context"],
            result_text,
            _status_md(new_state),
            _history_to_rows(new_history),
            _history_to_df(new_history),
            new_state,
        )

    # ── Build the Gradio Blocks layout ──────────────────────────────────────

    with gr.Blocks(title="🏥 Healthisure") as demo:
        # Inject custom CSS via HTML element (Gradio-6 compatible)
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # ── Header ──────────────────────────────────────────────────────────
        gr.Markdown(
            "# 🏥 Healthisure\n"
            "An OpenEnv reinforcement-learning environment for health insurance "
            "claim handling — eligibility, prior auth, denials, COB & disputes.",
            elem_classes="hc-header",
        )

        # ── Episode state (server-side, per Gradio session) ─────────────────
        episode_state = gr.State(dict(EMPTY_STATE))

        # ── Episode controls ─────────────────────────────────────────────────
        with gr.Row():
            task_selector = gr.Dropdown(
                choices=[(TASK_LABELS[k], k) for k in TASK_OPTIONS],
                value="task1",
                label="Task",
                scale=3,
            )
            reset_btn = gr.Button("▶ Reset Episode", variant="primary", scale=1)

        status_bar = gr.Markdown(
            _status_md(EMPTY_STATE), elem_classes="hc-status-bar"
        )

        # ── Observation panels ───────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                task_desc_box = gr.Textbox(
                    label="📋 Task Description",
                    lines=8,
                    interactive=False,
                    placeholder="Reset the environment to load the case…",
                )
            with gr.Column(scale=2):
                member_ctx_box = gr.Textbox(
                    label="👤 Member Context",
                    lines=8,
                    interactive=False,
                    placeholder="Accumulated data from lookup actions…",
                )

        # ── Action panel ─────────────────────────────────────────────────────
        with gr.Group(elem_classes="param-group"):
            gr.Markdown("### ⚡ Take Action")

            action_dropdown = gr.Dropdown(
                choices=VALID_ACTIONS,
                value=VALID_ACTIONS[0],
                label="Action",
            )

            # Pre-create ALL parameter fields; visibility toggled on action change
            with gr.Row():
                with gr.Column():
                    f_member_id = gr.Textbox(
                        label="member_id",
                        placeholder="e.g. M001",
                        visible=True,
                    )
                    f_claim_id = gr.Textbox(
                        label="claim_id",
                        placeholder="e.g. CLM-001",
                        visible=False,
                    )
                    f_provider_id = gr.Textbox(
                        label="provider_id",
                        placeholder="e.g. PRV-001",
                        visible=False,
                    )
                    f_code = gr.Textbox(
                        label="code  (denial code)",
                        placeholder="e.g. CO-4",
                        visible=False,
                    )
                    f_secondary_insurer_id = gr.Textbox(
                        label="secondary_insurer_id",
                        placeholder="e.g. SILVER-001",
                        visible=False,
                    )

                with gr.Column():
                    f_plan_id = gr.Textbox(
                        label="plan_id",
                        placeholder="e.g. GOLD-001",
                        visible=False,
                    )
                    f_cpt_code = gr.Textbox(
                        label="cpt_code",
                        placeholder="e.g. 70553",
                        visible=False,
                    )
                    f_amount = gr.Number(
                        label="amount  ($)",
                        value=0.0,
                        visible=False,
                    )
                    f_deductible_met = gr.Checkbox(
                        label="deductible_met",
                        value=False,
                        visible=False,
                    )
                    f_priority = gr.Dropdown(
                        choices=["high", "medium", "low"],
                        value="medium",
                        label="priority",
                        visible=False,
                    )
                    f_citation = gr.Textbox(
                        label="citation",
                        placeholder="e.g. 29 CFR §2560.503-1",
                        visible=False,
                    )

                with gr.Column(scale=2):
                    f_reason = gr.Textbox(
                        label="reason",
                        lines=3,
                        placeholder="Describe the reason for this action…",
                        visible=False,
                    )
                    f_message = gr.Textbox(
                        label="message  (to member)",
                        lines=5,
                        placeholder="Type your final response to the member…",
                        visible=False,
                    )

            execute_btn = gr.Button("Execute Action ▶", variant="primary")

        # ── Action result ────────────────────────────────────────────────────
        result_box = gr.Textbox(
            label="📤 Action Result",
            lines=6,
            interactive=False,
            placeholder="Result of the last action will appear here…",
            elem_classes="result-box",
        )

        # ── Reward history: table + chart side-by-side ───────────────────────
        with gr.Row():
            reward_history_df = gr.Dataframe(
                headers=["Step", "Action", "Reward", "Cumulative", "Done"],
                datatype=["number", "str", "number", "number", "bool"],
                label="📋 Reward History",
                interactive=False,
                wrap=True,
                elem_classes="reward-table",
                scale=2,
            )
            reward_plot = gr.LinePlot(
                value=_history_to_df([]),
                x="Step",
                y="Value",
                color="Series",
                title="Reward over Steps",
                x_title="Step",
                y_title="Reward",
                color_title="Series",
                label="📈 Reward Chart",
                scale=3,
            )

        # ────────────────────────────────────────────────────────────────────
        # Wire up param field list — order MUST match ALL_PARAM_FIELDS
        # ────────────────────────────────────────────────────────────────────
        param_components = [
            f_member_id,
            f_claim_id,
            f_provider_id,
            f_code,
            f_secondary_insurer_id,
            f_plan_id,
            f_cpt_code,
            f_amount,
            f_deductible_met,
            f_priority,
            f_citation,
            f_reason,
            f_message,
        ]

        # Shared outputs for both reset and step
        shared_outputs = [
            task_desc_box,
            member_ctx_box,
            result_box,
            status_bar,
            reward_history_df,
            reward_plot,
            episode_state,
        ]

        # ── Events ───────────────────────────────────────────────────────────

        # Action dropdown → show/hide param fields
        action_dropdown.change(
            fn=update_param_visibility,
            inputs=[action_dropdown],
            outputs=param_components,
        )

        # Reset button
        reset_btn.click(
            fn=reset_episode,
            inputs=[task_selector, episode_state],
            outputs=shared_outputs,
        )

        # Execute button — all param fields + state as inputs
        execute_btn.click(
            fn=execute_action,
            inputs=[action_dropdown] + param_components + [episode_state],
            outputs=shared_outputs,
        )

    return demo
