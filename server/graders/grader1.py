"""
Grader for Task 1 — Benefit Verification & Eligibility Check.

Scores each step of the agent's trajectory and determines whether the
episode is fully resolved.
"""

from typing import Any, Dict

from .base import BaseGrader


class Grader1(BaseGrader):
    """
    Scoring rubric (max ≈ 1.0 per episode):
      lookup_member called correctly          → +0.05
      check_deductible_status correct         → +0.15
      lookup_plan_benefits correct            → +0.10
      check_prior_auth_required correct       → +0.10 (+ penalty if PA missed)
      apply_cost_share with correct result    → +0.20
      send_member_response (resolution)       → +0.10 + full_resolution bonus +0.30
    """

    def grade_step(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        action_result: Dict[str, Any],
        episode_state: Dict[str, Any],
    ) -> float:
        if not action_result.get("success"):
            return 0.0

        gold = episode_state.get("gold_standard", {})
        flags = episode_state.setdefault("grader_flags", {})
        reward = 0.0

        if action_name == "lookup_member":
            if not flags.get("member_looked_up"):
                flags["member_looked_up"] = True
                reward += self.REWARD_MEMBER_LOOKUP

        elif action_name == "check_deductible_status":
            if not flags.get("deductible_checked"):
                flags["deductible_checked"] = True
                # Check if deductible_remaining is correct
                result_remaining = action_result.get("deductible_remaining")
                gold_remaining = gold.get("deductible_remaining")
                if result_remaining is not None and gold_remaining is not None:
                    if abs(result_remaining - gold_remaining) < 1.0:
                        reward += self.REWARD_COST_SHARE_CORRECT  # +0.15

        elif action_name == "lookup_plan_benefits":
            if not flags.get("plan_benefits_looked_up"):
                flags["plan_benefits_looked_up"] = True
                if action_result.get("covered") == gold.get("covered", True):
                    reward += self.REWARD_CPT_RESOLUTION  # +0.10

        elif action_name == "check_prior_auth_required":
            if not flags.get("pa_checked"):
                flags["pa_checked"] = True
                pa_result = action_result.get("pa_required")
                pa_gold = gold.get("pa_required", False)
                if pa_result == pa_gold:
                    reward += self.REWARD_PA_FLAG  # +0.10
                elif pa_gold and not pa_result:
                    # Agent missed a required PA check
                    reward += self.PENALTY_MISSED_PA  # -0.15

        elif action_name == "apply_cost_share":
            if not flags.get("cost_share_applied"):
                flags["cost_share_applied"] = True
                member_cost = action_result.get("member_cost")
                gold_cost = gold.get("member_cost_estimate")
                if member_cost is not None and gold_cost is not None:
                    if abs(member_cost - gold_cost) <= 1.0:
                        reward += self.REWARD_COST_SHARE_CORRECT  # +0.15 for correct math
                    # Give partial credit even for imprecise calculations
                    elif abs(member_cost - gold_cost) <= gold_cost * 0.10:
                        reward += self.REWARD_COST_SHARE_CORRECT * 0.5

        elif action_name == "send_member_response":
            flags["response_sent"] = True
            message = parameters.get("message", "")
            # Check response mentions key elements
            has_pa_mention = (
                "prior auth" in message.lower()
                or "authorization" in message.lower()
                or "pa" in message.lower()
            )
            gold_pa = gold.get("pa_must_be_flagged", False)
            if gold_pa and not has_pa_mention:
                reward += self.PENALTY_MISSED_PA  # -0.15 for not mentioning PA
            else:
                reward += 0.10  # partial credit for sending a response

        return reward

    def is_resolved(self, episode_state: Dict[str, Any]) -> bool:
        flags = episode_state.get("grader_flags", {})
        gold = episode_state.get("gold_standard", {})
        required = gold.get("required_lookups", [])
        # All required lookups must have been done and response sent
        lookup_flag_map = {
            "lookup_member": "member_looked_up",
            "check_deductible_status": "deductible_checked",
            "lookup_plan_benefits": "plan_benefits_looked_up",
            "check_prior_auth_required": "pa_checked",
            "apply_cost_share": "cost_share_applied",
        }
        all_done = all(flags.get(lookup_flag_map.get(r, r), False) for r in required)
        return all_done and flags.get("response_sent", False)
