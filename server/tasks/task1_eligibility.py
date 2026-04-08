"""
Task 1 — Easy: Benefit Verification & Eligibility Check

The agent must look up a member's plan, check deductible status,
verify CPT code coverage, check if PA is required, calculate cost share,
and send a clear response to the member.

Step budget: 10
"""

from typing import Any, Dict, List

from .base import BaseTask


class Task1Eligibility(BaseTask):
    task_name = "task1"
    step_budget = 10

    @property
    def scenarios(self) -> List[Dict[str, Any]]:
        return [
            {
                "scenario_id": 0,
                "member_id": "M001",
                "cpt_code": "70553",
                "plan_id": "GOLD-001",
                "task_description": (
                    "Member Sarah Chen (ID: M001) is calling to ask whether an MRI of the brain "
                    "with contrast (CPT 70553) is covered under her plan, and what her out-of-pocket "
                    "cost will be. She has not had the procedure yet and wants to know before scheduling. "
                    "The billed amount is estimated at $2,500.00.\n\n"
                    "Resolve the inquiry: confirm coverage, flag any prior authorization requirement, "
                    "calculate her cost share, and provide a clear answer to the member."
                ),
                "gold_standard": {
                    "covered": True,
                    "pa_required": True,
                    "deductible_remaining": 1000.00,
                    "member_cost_estimate": 1000.00 + (1500.00 * 0.20),
                    "member_cost_after_deductible": round(1000.00 + (1500.00 * 0.20), 2),
                    "cost_share_type": "coinsurance",
                    "required_lookups": [
                        "lookup_member",
                        "check_deductible_status",
                        "lookup_plan_benefits",
                        "check_prior_auth_required",
                        "apply_cost_share",
                    ],
                    "terminal_action": "send_member_response",
                    "pa_must_be_flagged": True,
                },
            },
            {
                "scenario_id": 1,
                "member_id": "M002",
                "cpt_code": "27447",
                "plan_id": "SILVER-001",
                "task_description": (
                    "Member James Williams (ID: M002) is calling to ask about the cost of total knee "
                    "replacement surgery (CPT 27447) under his Silver Choice plan. He has already met "
                    "his full annual deductible. The estimated procedure cost is $25,000.00.\n\n"
                    "Confirm coverage, check prior authorization requirements, calculate his cost share, "
                    "and communicate the result clearly to him."
                ),
                "gold_standard": {
                    "covered": True,
                    "pa_required": True,
                    "deductible_remaining": 0.00,
                    "deductible_fully_met": True,
                    "member_cost_estimate": round(25000.00 * 0.30, 2),
                    "cost_share_type": "coinsurance",
                    "required_lookups": [
                        "lookup_member",
                        "check_deductible_status",
                        "lookup_plan_benefits",
                        "check_prior_auth_required",
                        "apply_cost_share",
                    ],
                    "terminal_action": "send_member_response",
                    "pa_must_be_flagged": True,
                },
            },
            {
                "scenario_id": 2,
                "member_id": "M003",
                "cpt_code": "99213",
                "plan_id": "BRONZE-001",
                "task_description": (
                    "Member Maria Gonzalez (ID: M003) is calling to ask about the cost of an office "
                    "visit with her specialist (CPT 99213 — established patient, moderate complexity) "
                    "under her Bronze Saver plan. She has not met any of her deductible yet. "
                    "The billed amount is $150.00.\n\n"
                    "Confirm coverage, check if prior authorization is needed, calculate her cost share, "
                    "and respond to the member with a clear explanation."
                ),
                "gold_standard": {
                    "covered": True,
                    "pa_required": False,
                    "deductible_remaining": 6000.00,
                    "deductible_fully_met": False,
                    "member_cost_estimate": 100.00,
                    "cost_share_type": "copay",
                    "required_lookups": [
                        "lookup_member",
                        "check_deductible_status",
                        "lookup_plan_benefits",
                        "check_prior_auth_required",
                        "apply_cost_share",
                    ],
                    "terminal_action": "send_member_response",
                    "pa_must_be_flagged": False,
                },
            },
        ]

    def get_initial_observation_text(self, scenario: Dict[str, Any]) -> str:
        return scenario["task_description"]
