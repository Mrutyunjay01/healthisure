"""
Grader for Task 2 — Prior Authorization & Claim Status Resolution.

Determines whether the agent correctly diagnosed the denial (correct vs erroneous)
and took the appropriate action (explain denial vs draft appeal letter).
"""

from typing import Any, Dict

from .base import BaseGrader


class Grader2(BaseGrader):
    """
    Scoring rubric (max ≈ 1.0 per episode):
      lookup_member                          → +0.05
      check_claim_status                     → +0.10
      decode_denial_code                     → +0.10
      check_prior_auth_required              → +0.10
      draft_appeal_letter (when required)    → +0.20
      send_member_response                   → +0.10 + full_resolution +0.30
      Missing appeal when required           → -0.20
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
                reward += self.REWARD_MEMBER_LOOKUP  # +0.05

        elif action_name == "check_claim_status":
            if not flags.get("claim_checked"):
                flags["claim_checked"] = True
                # Verify agent looked up the correct claim
                claim_data = action_result.get("data", {})
                if claim_data.get("claim_id") == gold.get("claim_id"):
                    reward += self.REWARD_CPT_RESOLUTION  # +0.10

        elif action_name == "decode_denial_code":
            if not flags.get("denial_decoded"):
                flags["denial_decoded"] = True
                decoded = action_result.get("code")
                if decoded == gold.get("denial_code"):
                    reward += self.REWARD_DENIAL_DECODE  # +0.10

        elif action_name == "check_prior_auth_required":
            if not flags.get("pa_checked"):
                flags["pa_checked"] = True
                pa_result = action_result.get("pa_required")
                pa_gold = gold.get("pa_required")
                if pa_result == pa_gold:
                    reward += self.REWARD_PA_FLAG  # +0.10

        elif action_name == "draft_appeal_letter":
            if not flags.get("appeal_drafted"):
                flags["appeal_drafted"] = True
                appeal_required = gold.get("appeal_required", False)
                if appeal_required:
                    # Check if citation is present (MHPAEA, PA reference, etc.)
                    citation = parameters.get("citation", "")
                    required_citation = gold.get("appeal_citation_required", "")
                    if required_citation and required_citation.lower()[:6] in citation.lower():
                        reward += self.REWARD_APPEAL_LETTER  # +0.20
                    else:
                        reward += self.REWARD_APPEAL_LETTER * 0.5  # partial credit
                else:
                    # Appeal drafted when not needed — minor penalty
                    reward -= 0.05

        elif action_name == "send_member_response":
            flags["response_sent"] = True
            appeal_required = gold.get("appeal_required", False)
            if appeal_required and not flags.get("appeal_drafted"):
                # Sent response without drafting appeal — major miss
                reward += self.PENALTY_HALLUCINATION  # -0.20
            else:
                reward += 0.10

        return reward

    def is_resolved(self, episode_state: Dict[str, Any]) -> bool:
        flags = episode_state.get("grader_flags", {})
        gold = episode_state.get("gold_standard", {})
        appeal_required = gold.get("appeal_required", False)
        base = (
            flags.get("member_looked_up", False)
            and flags.get("claim_checked", False)
            and flags.get("denial_decoded", False)
            and flags.get("response_sent", False)
        )
        if appeal_required:
            return base and flags.get("appeal_drafted", False)
        return base
