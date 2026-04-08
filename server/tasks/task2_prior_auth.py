"""
Task 2 — Medium: Prior Authorization & Claim Status Resolution

The agent must investigate a denied claim, decode the denial reason,
determine if the denial is correct or erroneous, and either explain the
correct denial or draft a regulatory-grounded appeal letter.

Step budget: 15
"""

from typing import Any, Dict, List

from .base import BaseTask


class Task2PriorAuth(BaseTask):
    task_name = "task2"
    step_budget = 15

    @property
    def scenarios(self) -> List[Dict[str, Any]]:
        return [
            {
                "scenario_id": 0,
                "member_id": "M001",
                "claim_id": "CLM-001",
                "plan_id": "GOLD-001",
                "task_description": (
                    "Member Sarah Chen (ID: M001) is upset. Her claim for an MRI (Claim ID: CLM-001) "
                    "was denied and she received a denial code CO-4. She does not understand why it was "
                    "denied and wants an explanation. Her doctor has submitted supporting clinical notes.\n\n"
                    "Investigate: Look up the claim, decode the denial reason, check whether prior "
                    "authorization was required and whether it was obtained, determine if the denial "
                    "is correct or erroneous, then provide the appropriate resolution:\n"
                    "  - If correct: explain clearly and guide the member on next steps.\n"
                    "  - If erroneous: draft a formal appeal letter with regulatory citation.\n"
                    "Finally, send a clear, empathetic response to the member."
                ),
                "gold_standard": {
                    "claim_id": "CLM-001",
                    "denial_code": "CO-4",
                    "pa_required": True,
                    "pa_obtained": False,
                    "denial_is_correct": True,
                    "appeal_required": False,
                    "required_steps": [
                        "lookup_member",
                        "check_claim_status",
                        "decode_denial_code",
                        "check_prior_auth_required",
                        "send_member_response",
                    ],
                    "appeal_citation_required": None,
                    "terminal_action": "send_member_response",
                },
            },
            {
                "scenario_id": 1,
                "member_id": "M005",
                "claim_id": "CLM-002",
                "plan_id": "SILVER-001",
                "task_description": (
                    "Member Linda Patel (ID: M005) received a denial (CO-4) for her knee replacement "
                    "surgery claim (Claim ID: CLM-002). She insists her doctor obtained prior "
                    "authorization before the procedure. She is frustrated and wants this resolved.\n\n"
                    "Investigate: Look up the claim and member details, decode the denial, verify "
                    "whether prior authorization was actually obtained, and determine if the denial "
                    "is correct or erroneous.\n"
                    "  - If correct: explain clearly.\n"
                    "  - If erroneous: draft a formal appeal letter citing the PA reference and "
                    "    plan language, then send an empathetic response to the member."
                ),
                "gold_standard": {
                    "claim_id": "CLM-002",
                    "denial_code": "CO-4",
                    "pa_required": True,
                    "pa_obtained": True,
                    "pa_reference": "PA-2024-77821",
                    "denial_is_correct": False,
                    "appeal_required": True,
                    "required_steps": [
                        "lookup_member",
                        "check_claim_status",
                        "decode_denial_code",
                        "check_prior_auth_required",
                        "draft_appeal_letter",
                        "send_member_response",
                    ],
                    "appeal_citation_required": "PA-2024-77821",
                    "terminal_action": "send_member_response",
                },
            },
            {
                "scenario_id": 2,
                "member_id": "M002",
                "claim_id": "CLM-004",
                "plan_id": "SILVER-001",
                "task_description": (
                    "Member James Williams (ID: M002) received a denial (CO-50, 'not medically "
                    "necessary') for his psychotherapy sessions (Claim ID: CLM-004). He has been "
                    "seeing this therapist for major depression and feels this denial is unfair — "
                    "the same insurer would cover comparable medical services without question.\n\n"
                    "Investigate: Look up the member and claim, decode the denial reason, check if "
                    "this denial may violate the Mental Health Parity and Addiction Equity Act (MHPAEA). "
                    "If a parity violation is identified, draft an appeal letter citing ACA MHPAEA, "
                    "then send an empathetic explanation to the member."
                ),
                "gold_standard": {
                    "claim_id": "CLM-004",
                    "denial_code": "CO-50",
                    "pa_required": False,
                    "pa_obtained": False,
                    "denial_is_correct": False,
                    "mhpaea_violation": True,
                    "appeal_required": True,
                    "required_steps": [
                        "lookup_member",
                        "check_claim_status",
                        "decode_denial_code",
                        "draft_appeal_letter",
                        "send_member_response",
                    ],
                    "appeal_citation_required": "ACA MHPAEA / 29 U.S.C. § 1185a",
                    "terminal_action": "send_member_response",
                },
            },
        ]

    def get_initial_observation_text(self, scenario: Dict[str, Any]) -> str:
        return scenario["task_description"]
