"""
Task 3 — Hard: Multi-Party Claim Dispute with Coordination of Benefits

The agent must handle a complex scenario involving two insurance plans,
an out-of-network ER visit, a missed filing deadline due to provider error,
and a collections threat. The agent must produce three documents and escalate.

Step budget: 20
"""

from typing import Any, Dict, List

from .base import BaseTask


class Task3CobDispute(BaseTask):
    task_name = "task3"
    step_budget = 20

    @property
    def scenarios(self) -> List[Dict[str, Any]]:
        return [
            {
                "scenario_id": 0,
                "member_id": "M004",
                "claim_id": "CLM-003",
                "primary_plan_id": "GOLD-001",
                "secondary_plan_id": "SILVER-002",
                "secondary_insurer_id": "INS-SPOUSE-001",
                "provider_id": "PRV-303",
                "task_description": (
                    "URGENT — Robert Kim (Member ID: M004) is calling in distress. He visited an "
                    "emergency room on November 15, 2023 (Claim ID: CLM-003) and received treatment "
                    "for a hip fracture. The ER provider (Valley Emergency Group, PRV-303) is "
                    "OUT-OF-NETWORK.\n\n"
                    "Here is the situation:\n"
                    "  • His employer plan (GOLD-001) is his PRIMARY insurer and paid $1,200 of "
                    "    the $1,800 bill.\n"
                    "  • He also has SECONDARY coverage through his spouse's employer plan "
                    "    (SILVER-002, Insurer ID: INS-SPOUSE-001), but no claim was ever filed "
                    "    with the secondary insurer.\n"
                    "  • The standard 90-day filing deadline has passed, but the delay was caused "
                    "    by the provider sending the bill to the wrong address — the corrected "
                    "    billing statement arrived on February 20, 2024.\n"
                    "  • The provider is now threatening to send the $600 balance to collections "
                    "    by April 15, 2024.\n\n"
                    "You must:\n"
                    "  1. Determine the correct COB order (who is primary/secondary).\n"
                    "  2. Apply ACA ER parity — in-network rates must apply regardless of the "
                    "     provider's network status.\n"
                    "  3. Invoke the deadline exception due to the provider billing error.\n"
                    "  4. File the corrected claim with the secondary insurer.\n"
                    "  5. Draft a dispute letter to the provider halting collections.\n"
                    "  6. Create an internal escalation note flagging compliance risk.\n"
                    "  7. Send a clear, empathetic response to the member.\n\n"
                    "This case must be fully resolved before the collections deadline."
                ),
                "gold_standard": {
                    "cob_primary": "GOLD-001",
                    "cob_secondary": "SILVER-002",
                    "cob_order_correct": True,
                    "aca_er_parity_applied": True,
                    "deadline_exception_invoked": True,
                    "delay_reason": "provider_billing_error",
                    "secondary_claim_filed": True,
                    "dispute_letter_drafted": True,
                    "escalation_created": True,
                    "required_steps": [
                        "lookup_member",
                        "check_claim_status",
                        "lookup_plan_benefits",
                        "apply_cost_share",
                        "file_corrected_claim",
                        "draft_dispute_letter",
                        "escalate_case",
                        "send_member_response",
                    ],
                    "documents_required": [
                        "file_corrected_claim",
                        "draft_dispute_letter",
                        "escalate_case",
                    ],
                    "terminal_action": "send_member_response",
                },
            },
            {
                "scenario_id": 1,
                "member_id": "M004",
                "claim_id": "CLM-003",
                "primary_plan_id": "GOLD-001",
                "secondary_plan_id": "SILVER-002",
                "secondary_insurer_id": "INS-SPOUSE-001",
                "provider_id": "PRV-303",
                "task_description": (
                    "URGENT — Robert Kim (Member ID: M004) has received a second demand notice "
                    "from Valley Emergency Group (PRV-303) regarding his November 2023 ER visit "
                    "(Claim ID: CLM-003). The ER was out-of-network.\n\n"
                    "New context from the member:\n"
                    "  • His employer plan (GOLD-001) is PRIMARY. His wife's employer plan "
                    "    (SILVER-002, Insurer ID: INS-SPOUSE-001) is SECONDARY.\n"
                    "  • The primary insurer applied out-of-network rates, but Robert believes "
                    "    ACA ER parity rules require in-network rates to apply even for out-of-network "
                    "    ER visits.\n"
                    "  • The secondary claim was never filed due to the provider's billing error "
                    "    (incorrect mailing address).\n"
                    "  • The provider is demanding immediate payment and threatening collections.\n\n"
                    "Fully resolve the case: verify COB order, apply ACA ER parity, invoke the "
                    "deadline exception, file with secondary, draft a dispute letter to stop "
                    "collections, escalate internally, and communicate the outcome to the member."
                ),
                "gold_standard": {
                    "cob_primary": "GOLD-001",
                    "cob_secondary": "SILVER-002",
                    "cob_order_correct": True,
                    "aca_er_parity_applied": True,
                    "deadline_exception_invoked": True,
                    "delay_reason": "provider_billing_error",
                    "secondary_claim_filed": True,
                    "dispute_letter_drafted": True,
                    "escalation_created": True,
                    "required_steps": [
                        "lookup_member",
                        "check_claim_status",
                        "lookup_plan_benefits",
                        "apply_cost_share",
                        "file_corrected_claim",
                        "draft_dispute_letter",
                        "escalate_case",
                        "send_member_response",
                    ],
                    "documents_required": [
                        "file_corrected_claim",
                        "draft_dispute_letter",
                        "escalate_case",
                    ],
                    "terminal_action": "send_member_response",
                },
            },
        ]

    def get_initial_observation_text(self, scenario: Dict[str, Any]) -> str:
        return scenario["task_description"]
