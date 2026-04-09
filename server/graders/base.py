"""
Abstract base class for Healthisure graders.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import config


class BaseGrader(ABC):
    """Abstract base for deterministic step-by-step graders."""

    REWARD_MEMBER_LOOKUP = config.REWARD_MEMBER_LOOKUP
    REWARD_CPT_RESOLUTION = config.REWARD_CPT_RESOLUTION
    REWARD_COST_SHARE_CORRECT = config.REWARD_COST_SHARE_CORRECT
    REWARD_PA_FLAG = config.REWARD_PA_FLAG
    REWARD_DENIAL_DECODE = config.REWARD_DENIAL_DECODE
    REWARD_APPEAL_LETTER = config.REWARD_APPEAL_LETTER
    REWARD_FULL_RESOLUTION = config.REWARD_FULL_RESOLUTION

    PENALTY_HALLUCINATION = config.PENALTY_HALLUCINATION
    PENALTY_MISSED_PA = config.PENALTY_MISSED_PA
    PENALTY_WRONG_COB = config.PENALTY_WRONG_COB
    PENALTY_STEP_BUDGET = config.PENALTY_STEP_BUDGET

    @abstractmethod
    def grade_step(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        action_result: Dict[str, Any],
        episode_state: Dict[str, Any],
    ) -> float:
        """Return incremental reward for this step (may be negative)."""

    @abstractmethod
    def is_resolved(self, episode_state: Dict[str, Any]) -> bool:
        """Return True when the task has been fully and correctly resolved."""
