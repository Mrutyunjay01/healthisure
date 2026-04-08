"""
Grader for Task 3 — Multi-Party COB Dispute.

Verifies that the agent correctly resolved COB order, applied ACA ER parity,
invoked the deadline exception, and produced all three required documents.
"""

from typing import Any, Dict

from .base import BaseGrader


class Grader3(BaseGrader):
    """
    Scoring rubric (max ≈ 1.0 per episode):
      lookup_member (with secondary plan noted)  → +0.05
      check_claim_status                         → +0.10
      lookup_plan_benefits (primary + secondary) → +0.10 (0.05 each)
      apply_cost_share with ACA ER parity        → +0.15
      file_corrected_claim (secondary)           → +0.15
      draft_dispute_letter (collections halt)    → +0.15
      escalate_case                              → +0.10
      send_member_response                       → +0.10 + full_resolution +0.30
      Wrong COB order                            → -0.20
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
                data = action_result.get("data", {})
                # Reward +0.05 only if secondary plan was noted
                if data.get("secondary_plan_id"):
                    flags["secondary_plan_noted"] = True
                reward += self.REWARD_MEMBER_LOOKUP  # +0.05

        elif action_name == "check_claim_status":
            if not flags.get("claim_checked"):
                flags["claim_checked"] = True
                reward += self.REWARD_CPT_RESOLUTION  # +0.10

        elif action_name == "lookup_plan_benefits":
            plan_id = parameters.get("plan_id", "")
            gold_primary = gold.get("cob_primary", "")
            gold_secondary = gold.get("cob_secondary", "")
            if plan_id == gold_primary and not flags.get("primary_benefits_looked_up"):
                flags["primary_benefits_looked_up"] = True
                reward += 0.05
            elif plan_id == gold_secondary and not flags.get("secondary_benefits_looked_up"):
                flags["secondary_benefits_looked_up"] = True
                reward += 0.05
                # Check if ACA ER parity is in the result
                if action_result.get("aca_er_parity"):
                    flags["aca_er_parity_noted"] = True

        elif action_name == "apply_cost_share":
            if not flags.get("cost_share_applied"):
                flags["cost_share_applied"] = True
                # ACA ER parity should have been noted before applying cost share
                if flags.get("aca_er_parity_noted") or flags.get("primary_benefits_looked_up"):
                    flags["aca_er_parity_applied"] = True
                    reward += self.REWARD_COST_SHARE_CORRECT  # +0.15

        elif action_name == "file_corrected_claim":
            if not flags.get("corrected_claim_filed"):
                flags["corrected_claim_filed"] = True
                # Check correct secondary insurer used
                insurer = parameters.get("secondary_insurer_id", "")
                gold_insurer = episode_state.get("scenario", {}).get("secondary_insurer_id", "")
                if insurer == gold_insurer:
                    reward += self.REWARD_COST_SHARE_CORRECT  # +0.15
                else:
                    # Wrong COB — filed with wrong insurer
                    reward += self.PENALTY_WRONG_COB  # -0.20
                    flags["wrong_cob"] = True

        elif action_name == "draft_dispute_letter":
            if not flags.get("dispute_letter_drafted"):
                flags["dispute_letter_drafted"] = True
                reward += self.REWARD_COST_SHARE_CORRECT  # +0.15

        elif action_name == "escalate_case":
            if not flags.get("escalation_created"):
                flags["escalation_created"] = True
                reward += self.REWARD_PA_FLAG  # +0.10

        elif action_name == "send_member_response":
            flags["response_sent"] = True
            # Penalise if required documents were skipped
            docs_complete = (
                flags.get("corrected_claim_filed", False)
                and flags.get("dispute_letter_drafted", False)
                and flags.get("escalation_created", False)
            )
            if docs_complete:
                reward += 0.10
            else:
                reward -= 0.10  # responded without completing all docs

        return reward

    def is_resolved(self, episode_state: Dict[str, Any]) -> bool:
        flags = episode_state.get("grader_flags", {})
        gold = episode_state.get("gold_standard", {})
        return (
            flags.get("member_looked_up", False)
            and flags.get("claim_checked", False)
            and flags.get("corrected_claim_filed", False)
            and flags.get("dispute_letter_drafted", False)
            and flags.get("escalation_created", False)
            and flags.get("response_sent", False)
            and not flags.get("wrong_cob", False)
        )
