"""
Central reward configuration loaded from openenv.yaml.

Imported by server graders and the inference script so reward values
and per-task maximum scores are defined in exactly one place.
"""

from pathlib import Path
from typing import Dict

import yaml

_cfg = yaml.safe_load((Path(__file__).parent / "openenv.yaml").read_text())
_r: Dict = _cfg["rewards"]
_p: Dict = _r["penalties"]

REWARD_MEMBER_LOOKUP: float = _r["member_lookup"]
REWARD_CPT_RESOLUTION: float = _r["cpt_resolution"]
REWARD_COST_SHARE_CORRECT: float = _r["cost_share_correct"]
REWARD_PA_FLAG: float = _r["pa_flag"]
REWARD_DENIAL_DECODE: float = _r["denial_decode"]
REWARD_APPEAL_LETTER: float = _r["appeal_letter"]
REWARD_FULL_RESOLUTION: float = _r["full_resolution"]

PENALTY_HALLUCINATION: float = _p["hallucination"]
PENALTY_MISSED_PA: float = _p["missed_pa"]
PENALTY_WRONG_COB: float = _p["wrong_cob"]
PENALTY_STEP_BUDGET: float = _p["step_budget"]

TASK_MAX_SCORES: Dict[str, float] = {
    name: info["max_score"] for name, info in _r["tasks"].items()
}
