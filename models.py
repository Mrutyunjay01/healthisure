"""
Pydantic data models for Healthisure.

HealthisureAction  — what the agent sends on each step
HealthisureObservation — what the environment returns to the agent
"""

from typing import Any, Dict, List, Optional, Union

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

VALID_ACTIONS = [
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


class HealthisureAction(Action):
    """
    Action taken by the agent in the Healthisure environment.

    The agent selects one of 12 named actions and provides the required
    parameters as a JSON-compatible dict.

    Available actions and their required parameters:
      lookup_member(member_id)
      lookup_plan_benefits(plan_id, cpt_code)
      check_claim_status(claim_id)
      decode_denial_code(code)
      check_prior_auth_required(cpt_code, plan_id)
      check_deductible_status(member_id)
      apply_cost_share(amount, plan_id, deductible_met)
      draft_appeal_letter(claim_id, reason, citation)
      draft_dispute_letter(provider_id, claim_id, reason)
      escalate_case(member_id, reason, priority)
      file_corrected_claim(member_id, claim_id, secondary_insurer_id)
      send_member_response(message)
    """

    action_name: str = Field(
        ...,
        description=(
            "Name of the action to execute. Must be one of the 12 defined actions: "
            + ", ".join(VALID_ACTIONS)
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            'Action-specific parameters as a JSON dict. '
            'Example: {"member_id": "M001"}  |  {"plan_id": "GOLD-001", "cpt_code": "70553"}'
        ),
    )

    @field_validator("parameters", mode="before")
    @classmethod
    def _parse_parameters(cls, v: Any) -> Any:
        """Accept a JSON string for the parameters field (e.g. from Gradio text input)."""
        if isinstance(v, str):
            import json
            v = v.strip()
            if not v:
                return {}
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return v


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class HealthisureObservation(Observation):
    """
    Observation returned by the Healthisure environment after reset() or step().

    Contains the task description, result of the last action, current step
    tracking, reward information, and contextual member data accumulated
    through lookups.
    """

    task_description: str = Field(
        ...,
        description="Full description of the support case the agent must resolve.",
    )
    available_actions: List[str] = Field(
        default_factory=lambda: VALID_ACTIONS,
        description="List of action names the agent may call.",
    )
    last_action_result: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable result of the previous action. "
            "None on the initial reset observation."
        ),
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken so far in this episode.",
    )
    step_budget: int = Field(
        ...,
        description="Maximum number of steps allowed for this task.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Accumulated reward so far in this episode.",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the last action was invalid or malformed.",
    )
    member_context: Optional[str] = Field(
        default=None,
        description=(
            "Formatted string of member/plan/claim information discovered "
            "so far through lookup actions."
        ),
    )
    task_name: str = Field(
        default="",
        description="Identifier of the active task (task1, task2, task3).",
    )
