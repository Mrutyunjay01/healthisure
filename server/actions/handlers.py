"""
Action handlers for all 12 Healthisure actions.

Each handler reads from the JSON data store and returns a structured result
dict containing a human-readable `message` and any structured data the
grader or environment needs.
"""

import json
import os
from typing import Any, Dict

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load(filename: str) -> Dict[str, Any]:
    path = os.path.join(_DATA_DIR, filename)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Lazy-loaded data store
# ---------------------------------------------------------------------------

class _DataStore:
    _cache: Dict[str, Any] = {}

    @classmethod
    def get(cls, name: str) -> Dict[str, Any]:
        if name not in cls._cache:
            cls._cache[name] = _load(name)
        return cls._cache[name]


def members() -> Dict:    return _DataStore.get("members.json")
def plans() -> Dict:      return _DataStore.get("plans.json")
def claims() -> Dict:     return _DataStore.get("claims.json")
def cpt_codes() -> Dict:  return _DataStore.get("cpt_codes.json")
def icd_codes() -> Dict:  return _DataStore.get("icd_codes.json")
def denial_codes() -> Dict: return _DataStore.get("denial_codes.json")
def providers() -> Dict:  return _DataStore.get("providers.json")
def regulatory() -> Dict: return _DataStore.get("regulatory.json")


# ---------------------------------------------------------------------------
# Individual action handlers
# ---------------------------------------------------------------------------

def lookup_member(member_id: str) -> Dict[str, Any]:
    """Fetch member profile and current plan summary."""
    db = members()
    member = db.get(member_id)
    if not member:
        return {
            "success": False,
            "error": f"Member '{member_id}' not found.",
            "message": f"No member record found for ID '{member_id}'. Please verify the member ID.",
        }
    plan_db = plans()
    plan = plan_db.get(member["plan_id"], {})
    return {
        "success": True,
        "member_id": member_id,
        "data": member,
        "message": (
            f"Member found: {member['name']} (DOB: {member['dob']})\n"
            f"Plan: {plan.get('plan_name', member['plan_id'])} [plan_id={member['plan_id']}] ({plan.get('tier','')}) | "
            f"Employer: {member['employer']}\n"
            f"Deductible: ${member['deductible_met']:.2f} met of ${member['deductible_total']:.2f}\n"
            f"OOP Spent: ${member['oop_spent']:.2f} of ${member['oop_max']:.2f} max\n"
            f"Secondary Plan: {member.get('secondary_plan_id') or 'None'}\n"
            f"Status: {member['status'].upper()}"
        ),
    }


def lookup_plan_benefits(plan_id: str, cpt_code: str) -> Dict[str, Any]:
    """Get coverage details and cost-share rules for a specific procedure."""
    plan_db = plans()
    plan = plan_db.get(plan_id)
    if not plan:
        return {
            "success": False,
            "error": f"Plan '{plan_id}' not found.",
            "message": f"No plan record found for ID '{plan_id}'.",
        }
    cpt_db = cpt_codes()
    cpt = cpt_db.get(cpt_code)
    if not cpt:
        return {
            "success": False,
            "error": f"CPT code '{cpt_code}' not found in database.",
            "message": f"CPT code '{cpt_code}' is not recognized. Please verify the procedure code.",
        }
    coverage = plan.get("cpt_coverage", {}).get(cpt_code)
    if not coverage:
        return {
            "success": True,
            "plan_id": plan_id,
            "cpt_code": cpt_code,
            "covered": False,
            "message": (
                f"CPT {cpt_code} ({cpt['description']}) is NOT covered under "
                f"plan {plan['plan_name']} ({plan['tier']} tier)."
            ),
        }
    pa_required = coverage.get("pa_required", False)
    cost_type = coverage.get("cost_share_type", "coinsurance")
    aca_preventive = coverage.get("aca_preventive", False)
    aca_er = coverage.get("aca_er_parity", False)

    cost_detail = ""
    if aca_preventive:
        cost_detail = "PREVENTIVE CARE: $0.00 member cost (ACA mandate — 100% covered in-network)"
    elif cost_type == "copay":
        cost_detail = f"Copay: ${coverage.get('copay', plan.get('copay_specialist', 0)):.2f} per visit"
    elif cost_type == "coinsurance":
        cost_detail = (
            f"Coinsurance: {int(plan['coinsurance_in_network']*100)}% member responsibility "
            f"(after deductible of ${plan['deductible_individual']:.2f} is met)"
        )
    elif cost_type == "copay_then_coinsurance":
        cost_detail = (
            f"ER Copay: ${coverage.get('copay', plan.get('copay_er', 0)):.2f}, "
            f"then {int(plan['coinsurance_in_network']*100)}% coinsurance after deductible"
        )

    return {
        "success": True,
        "plan_id": plan_id,
        "cpt_code": cpt_code,
        "covered": True,
        "pa_required": pa_required,
        "cost_share_type": cost_type,
        "coinsurance_rate": plan.get("coinsurance_in_network"),
        "aca_preventive": aca_preventive,
        "aca_er_parity": aca_er,
        "data": coverage,
        "message": (
            f"CPT {cpt_code} ({cpt['description']}) under {plan['plan_name']} ({plan['tier']}):\n"
            f"  Covered: YES\n"
            f"  Prior Auth Required: {'YES ⚠️' if pa_required else 'No'}\n"
            f"  Cost Share: {cost_detail}\n"
            + (f"  ACA ER Parity applies: in-network rates apply regardless of network status\n" if aca_er else "")
            + (f"  ACA Preventive Mandate: No member cost share\n" if aca_preventive else "")
        ),
    }


def check_claim_status(claim_id: str) -> Dict[str, Any]:
    """Retrieve the status, denial code, and details of a claim."""
    claim_db = claims()
    claim = claim_db.get(claim_id)
    if not claim:
        return {
            "success": False,
            "error": f"Claim '{claim_id}' not found.",
            "message": f"No claim found with ID '{claim_id}'.",
        }
    denial_info = f" | Denial Code: {claim['denial_code']}" if claim.get("denial_code") else ""
    pa_info = (
        f" | PA Reference: {claim.get('pa_reference', 'N/A')}"
        if claim.get("pa_obtained")
        else " | PA Obtained: No"
    )
    return {
        "success": True,
        "claim_id": claim_id,
        "data": claim,
        "message": (
            f"Claim {claim_id}:\n"
            f"  Member: {claim['member_id']} | Provider: {claim['provider_id']}\n"
            f"  CPT: {claim['cpt_code']} | ICD-10: {claim.get('icd10_code','N/A')}\n"
            f"  Service Date: {claim['service_date']} | Filed: {claim['filed_date']}\n"
            f"  Amount Billed: ${claim['amount_billed']:.2f}\n"
            f"  Status: {claim['status'].upper()}{denial_info}\n"
            f"  In-Network: {'Yes' if claim.get('in_network') else 'No (Out-of-Network)'}"
            f"{pa_info}"
        ),
    }


def decode_denial_code(code: str) -> Dict[str, Any]:
    """Explain a claim denial reason code."""
    dc_db = denial_codes()
    dc = dc_db.get(code)
    if not dc:
        return {
            "success": False,
            "error": f"Denial code '{code}' not recognized.",
            "message": f"Denial code '{code}' is not in the reference database.",
        }
    return {
        "success": True,
        "code": code,
        "data": dc,
        "message": (
            f"Denial Code {code}: {dc['short_description']}\n"
            f"Category: {dc['category']}\n"
            f"Explanation: {dc['full_description']}\n"
            f"Member Action: {dc['member_action']}\n"
            f"Appeal Eligible: {'Yes' if dc.get('appeal_eligible') else 'No'}"
        ),
    }


def check_prior_auth_required(cpt_code: str, plan_id: str) -> Dict[str, Any]:
    """Determine if prior authorization is required for a procedure on a given plan."""
    plan_db = plans()
    plan = plan_db.get(plan_id)
    if not plan:
        return {
            "success": False,
            "error": f"Plan '{plan_id}' not found.",
            "message": f"Plan '{plan_id}' not found.",
        }
    cpt_db = cpt_codes()
    cpt = cpt_db.get(cpt_code)
    if not cpt:
        return {
            "success": False,
            "error": f"CPT code '{cpt_code}' not recognized.",
            "message": f"CPT '{cpt_code}' not found in database.",
        }
    coverage = plan.get("cpt_coverage", {}).get(cpt_code, {})
    pa_required = coverage.get("pa_required", False)
    return {
        "success": True,
        "cpt_code": cpt_code,
        "plan_id": plan_id,
        "pa_required": pa_required,
        "message": (
            f"Prior Authorization for CPT {cpt_code} ({cpt['description']}) "
            f"on plan {plan['plan_name']} ({plan['tier']}):\n"
            f"  PA REQUIRED: {'YES ⚠️ — must be obtained before service' if pa_required else 'No — PA not required'}"
        ),
    }


def check_deductible_status(member_id: str) -> Dict[str, Any]:
    """Get current deductible and OOP status for a member."""
    member_db = members()
    member = member_db.get(member_id)
    if not member:
        return {
            "success": False,
            "error": f"Member '{member_id}' not found.",
            "message": f"Member '{member_id}' not found.",
        }
    deductible_met = member["deductible_met"]
    deductible_total = member["deductible_total"]
    deductible_remaining = max(0.0, deductible_total - deductible_met)
    oop_remaining = max(0.0, member["oop_max"] - member["oop_spent"])
    fully_met = deductible_remaining == 0.0
    return {
        "success": True,
        "member_id": member_id,
        "deductible_met": deductible_met,
        "deductible_total": deductible_total,
        "deductible_remaining": deductible_remaining,
        "deductible_fully_met": fully_met,
        "oop_spent": member["oop_spent"],
        "oop_max": member["oop_max"],
        "oop_remaining": oop_remaining,
        "message": (
            f"Deductible Status for {member['name']} ({member_id}):\n"
            f"  Deductible: ${deductible_met:.2f} met of ${deductible_total:.2f} "
            f"({'FULLY MET ✓' if fully_met else f'${deductible_remaining:.2f} remaining'})\n"
            f"  Out-of-Pocket: ${member['oop_spent']:.2f} spent of ${member['oop_max']:.2f} max "
            f"(${oop_remaining:.2f} remaining)"
        ),
    }


def apply_cost_share(
    amount: float, plan_id: str, deductible_met: bool,
    cpt_code: str = None, member_id: str = None
) -> Dict[str, Any]:
    """
    Calculate the member's cost share for a given billed amount.

    If cpt_code is provided, uses the plan's cost_share_type for that procedure.
    Otherwise uses standard coinsurance after deductible.
    """
    plan_db = plans()
    plan = plan_db.get(plan_id)
    if not plan:
        return {
            "success": False,
            "error": f"Plan '{plan_id}' not found.",
            "message": f"Plan '{plan_id}' not found.",
        }

    amount = float(amount)
    member_cost = 0.0
    plan_cost = 0.0
    calculation_notes = ""

    # Determine cost share type
    coverage = {}
    if cpt_code:
        coverage = plan.get("cpt_coverage", {}).get(cpt_code, {})

    cost_type = coverage.get("cost_share_type", "coinsurance") if coverage else "coinsurance"

    if coverage.get("aca_preventive"):
        member_cost = 0.0
        plan_cost = amount
        calculation_notes = "ACA preventive care: $0.00 member cost (covered 100%)"
    elif cost_type == "copay":
        copay = coverage.get("copay", plan.get("copay_specialist", 0))
        member_cost = min(copay, amount)
        plan_cost = max(0.0, amount - copay)
        calculation_notes = f"Copay: ${copay:.2f}"
    elif cost_type == "coinsurance":
        if not deductible_met:
            # Member pays up to remaining deductible first
            member_db = members()
            remaining_ded = 0.0
            if member_id and member_db.get(member_id):
                m = member_db[member_id]
                remaining_ded = max(0.0, m["deductible_total"] - m["deductible_met"])
            deductible_portion = min(amount, remaining_ded)
            after_deductible = max(0.0, amount - deductible_portion)
            coinsurance_portion = after_deductible * plan["coinsurance_in_network"]
            member_cost = deductible_portion + coinsurance_portion
            plan_cost = after_deductible * (1 - plan["coinsurance_in_network"])
            calculation_notes = (
                f"Deductible portion: ${deductible_portion:.2f} + "
                f"Coinsurance ({int(plan['coinsurance_in_network']*100)}% of ${after_deductible:.2f}): "
                f"${coinsurance_portion:.2f}"
            )
        else:
            coinsurance = plan["coinsurance_in_network"]
            member_cost = amount * coinsurance
            plan_cost = amount * (1 - coinsurance)
            calculation_notes = f"{int(coinsurance*100)}% coinsurance of ${amount:.2f} (deductible fully met)"
    elif cost_type == "copay_then_coinsurance":
        copay = coverage.get("copay", plan.get("copay_er", 0))
        member_cost = copay
        remaining_after_copay = max(0.0, amount - copay)
        if not deductible_met:
            member_cost += remaining_after_copay
            plan_cost = 0.0
            calculation_notes = f"ER Copay: ${copay:.2f} + remaining ${remaining_after_copay:.2f} applied to deductible"
        else:
            coinsurance_amount = remaining_after_copay * plan["coinsurance_in_network"]
            member_cost += coinsurance_amount
            plan_cost = remaining_after_copay * (1 - plan["coinsurance_in_network"])
            calculation_notes = (
                f"ER Copay: ${copay:.2f} + "
                f"{int(plan['coinsurance_in_network']*100)}% coinsurance of ${remaining_after_copay:.2f}: "
                f"${coinsurance_amount:.2f}"
            )
    else:
        coinsurance = plan["coinsurance_in_network"]
        member_cost = amount * coinsurance
        plan_cost = amount - member_cost
        calculation_notes = f"{int(coinsurance*100)}% coinsurance (default)"

    # Apply OOP max cap if member_id provided
    oop_cap_applied = False
    if member_id:
        member_db = members()
        m = member_db.get(member_id, {})
        oop_remaining = max(0.0, m.get("oop_max", float("inf")) - m.get("oop_spent", 0.0))
        if member_cost > oop_remaining:
            member_cost = oop_remaining
            oop_cap_applied = True

    member_cost = round(member_cost, 2)
    plan_cost = round(plan_cost, 2)

    return {
        "success": True,
        "amount": amount,
        "plan_id": plan_id,
        "deductible_met": deductible_met,
        "member_cost": member_cost,
        "plan_cost": plan_cost,
        "oop_cap_applied": oop_cap_applied,
        "calculation_notes": calculation_notes,
        "message": (
            f"Cost Share Calculation for ${amount:.2f} billed amount:\n"
            f"  Plan: {plan['plan_name']} ({plan['tier']})\n"
            f"  Deductible Met: {'Yes' if deductible_met else 'No'}\n"
            f"  Calculation: {calculation_notes}\n"
            f"  Member Responsibility: ${member_cost:.2f}"
            + (" (OOP max applied)" if oop_cap_applied else "") + "\n"
            f"  Plan Pays: ${plan_cost:.2f}"
        ),
    }


def draft_appeal_letter(claim_id: str, reason: str, citation: str) -> Dict[str, Any]:
    """Generate a regulatory-grounded appeal letter for a denied claim."""
    claim_db = claims()
    claim = claim_db.get(claim_id)
    if not claim:
        return {
            "success": False,
            "error": f"Claim '{claim_id}' not found.",
            "message": f"Claim '{claim_id}' not found.",
        }
    member_db = members()
    member = member_db.get(claim.get("member_id", ""), {})
    member_name = member.get("name", "Member")

    letter = (
        f"APPEAL LETTER — Claim {claim_id}\n"
        f"{'='*60}\n"
        f"Date: [Current Date]\n"
        f"Re: Appeal of Adverse Benefit Determination\n"
        f"Member: {member_name} | Member ID: {claim.get('member_id')}\n"
        f"Claim ID: {claim_id} | Date of Service: {claim.get('service_date')}\n"
        f"CPT Code: {claim.get('cpt_code')} | Denial Code: {claim.get('denial_code', 'N/A')}\n\n"
        f"Dear Appeals Department,\n\n"
        f"On behalf of {member_name}, I am writing to formally appeal the denial of the above-referenced claim.\n\n"
        f"REASON FOR APPEAL:\n{reason}\n\n"
        f"REGULATORY/PLAN CITATION:\n{citation}\n\n"
        f"REQUEST:\nWe respectfully request that you reverse this denial and process the claim for payment "
        f"in accordance with the member's plan benefits and applicable federal regulations.\n\n"
        f"Please provide a written response within 60 days as required by 29 CFR § 2560.503-1.\n\n"
        f"Sincerely,\n[Insurance Support Specialist]\nHealthFirst Insurance — Member Services"
    )
    return {
        "success": True,
        "claim_id": claim_id,
        "appeal_letter": letter,
        "reason": reason,
        "citation": citation,
        "message": f"Appeal letter drafted for claim {claim_id}.\n\n{letter}",
    }


def draft_dispute_letter(provider_id: str, claim_id: str, reason: str) -> Dict[str, Any]:
    """Draft a letter to a provider disputing a collections threat and halting collections."""
    provider_db = providers()
    provider = provider_db.get(provider_id)
    if not provider:
        return {
            "success": False,
            "error": f"Provider '{provider_id}' not found.",
            "message": f"Provider '{provider_id}' not found.",
        }
    claim_db = claims()
    claim = claim_db.get(claim_id, {})
    member_db = members()
    member = member_db.get(claim.get("member_id", ""), {})

    letter = (
        f"PROVIDER DISPUTE LETTER — Claim {claim_id}\n"
        f"{'='*60}\n"
        f"Date: [Current Date]\n"
        f"To: {provider['name']} (NPI: {provider['npi']})\n"
        f"Address: {provider['address']}\n\n"
        f"Re: Formal Dispute — Collections Hold Request\n"
        f"Member: {member.get('name', 'Member')} | Claim: {claim_id}\n"
        f"Date of Service: {claim.get('service_date', 'N/A')}\n\n"
        f"Dear Billing Department,\n\n"
        f"We are writing on behalf of our member to formally dispute the above-referenced balance "
        f"and to request an immediate hold on any collections activity while this matter is under review.\n\n"
        f"DISPUTE REASON:\n{reason}\n\n"
        f"Please cease all collections activity immediately. Under state insurance regulations, "
        f"members are protected from collections while an active insurance dispute is pending.\n\n"
        f"We will provide an updated Explanation of Benefits (EOB) within 30 business days.\n\n"
        f"Sincerely,\n[Insurance Support Specialist]\nHealthFirst Insurance — Provider Relations"
    )
    return {
        "success": True,
        "provider_id": provider_id,
        "claim_id": claim_id,
        "dispute_letter": letter,
        "reason": reason,
        "message": f"Provider dispute letter drafted for provider {provider_id}, claim {claim_id}.\n\n{letter}",
    }


def escalate_case(member_id: str, reason: str, priority: str) -> Dict[str, Any]:
    """Create an internal case escalation note flagging compliance or operational risk."""
    member_db = members()
    member = member_db.get(member_id)
    if not member:
        return {
            "success": False,
            "error": f"Member '{member_id}' not found.",
            "message": f"Member '{member_id}' not found.",
        }
    valid_priorities = ["low", "medium", "high", "critical"]
    priority = priority.lower()
    if priority not in valid_priorities:
        priority = "medium"

    escalation = (
        f"INTERNAL ESCALATION NOTE\n"
        f"{'='*60}\n"
        f"Priority: {priority.upper()}\n"
        f"Member ID: {member_id} | Member: {member['name']}\n"
        f"Date: [Current Date]\n\n"
        f"ESCALATION REASON:\n{reason}\n\n"
        f"ACTION REQUIRED: Supervisory review within "
        f"{'4 hours' if priority == 'critical' else '24 hours' if priority == 'high' else '72 hours'}.\n"
        f"Assigned to: Compliance & Quality Assurance Team"
    )
    return {
        "success": True,
        "member_id": member_id,
        "priority": priority,
        "escalation_note": escalation,
        "message": f"Case escalated for member {member_id} at priority {priority.upper()}.\n\n{escalation}",
    }


def file_corrected_claim(
    member_id: str, claim_id: str, secondary_insurer_id: str
) -> Dict[str, Any]:
    """File a corrected claim with the secondary insurer."""
    member_db = members()
    member = member_db.get(member_id)
    if not member:
        return {
            "success": False,
            "error": f"Member '{member_id}' not found.",
            "message": f"Member '{member_id}' not found.",
        }
    claim_db = claims()
    claim = claim_db.get(claim_id)
    if not claim:
        return {
            "success": False,
            "error": f"Claim '{claim_id}' not found.",
            "message": f"Claim '{claim_id}' not found.",
        }
    if member.get("secondary_insurer_id") != secondary_insurer_id:
        return {
            "success": False,
            "error": (
                f"Secondary insurer '{secondary_insurer_id}' does not match member's "
                f"secondary insurer on file: '{member.get('secondary_insurer_id')}'."
            ),
            "message": (
                f"Incorrect secondary insurer ID '{secondary_insurer_id}'. "
                f"Member {member_id} has secondary insurer: {member.get('secondary_insurer_id')}."
            ),
        }
    corrected_claim_ref = f"CC-{claim_id}-SEC"
    return {
        "success": True,
        "member_id": member_id,
        "original_claim_id": claim_id,
        "secondary_insurer_id": secondary_insurer_id,
        "corrected_claim_reference": corrected_claim_ref,
        "message": (
            f"Corrected claim filed with secondary insurer {secondary_insurer_id}.\n"
            f"  Member: {member['name']} ({member_id})\n"
            f"  Original Claim: {claim_id} | Service Date: {claim['service_date']}\n"
            f"  CPT: {claim['cpt_code']} | Amount Billed: ${claim['amount_billed']:.2f}\n"
            f"  Primary Paid: ${claim.get('amount_paid_primary', 0):.2f}\n"
            f"  Corrected Claim Reference: {corrected_claim_ref}\n"
            f"  Status: SUBMITTED — secondary insurer will process within 30 days."
        ),
    }


def send_member_response(message: str) -> Dict[str, Any]:
    """Send the final response to the member. This is the terminal action."""
    if not message or not message.strip():
        return {
            "success": False,
            "error": "Message cannot be empty.",
            "message": "Cannot send an empty response to the member.",
            "terminal": False,
        }
    return {
        "success": True,
        "member_message": message,
        "terminal": True,
        "message": f"Member response sent:\n\n{message}",
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class ActionHandler:
    """Dispatches a HealthisureAction to the appropriate handler function."""

    _HANDLERS = {
        "lookup_member": lambda p: lookup_member(**p),
        "lookup_plan_benefits": lambda p: lookup_plan_benefits(**p),
        "check_claim_status": lambda p: check_claim_status(**p),
        "decode_denial_code": lambda p: decode_denial_code(**p),
        "check_prior_auth_required": lambda p: check_prior_auth_required(**p),
        "check_deductible_status": lambda p: check_deductible_status(**p),
        "apply_cost_share": lambda p: apply_cost_share(**p),
        "draft_appeal_letter": lambda p: draft_appeal_letter(**p),
        "draft_dispute_letter": lambda p: draft_dispute_letter(**p),
        "escalate_case": lambda p: escalate_case(**p),
        "file_corrected_claim": lambda p: file_corrected_claim(**p),
        "send_member_response": lambda p: send_member_response(**p),
    }

    @classmethod
    def dispatch(cls, action_name: str, parameters: dict) -> Dict[str, Any]:
        handler = cls._HANDLERS.get(action_name)
        if handler is None:
            return {
                "success": False,
                "error": f"Unknown action: '{action_name}'",
                "message": (
                    f"Action '{action_name}' is not recognized. "
                    f"Valid actions: {', '.join(cls._HANDLERS.keys())}"
                ),
            }
        try:
            return handler(parameters)
        except TypeError as e:
            return {
                "success": False,
                "error": f"Invalid parameters for action '{action_name}': {e}",
                "message": f"Parameter error for '{action_name}': {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Unexpected error executing '{action_name}': {e}",
            }
