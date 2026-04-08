"""
Abstract base class for Healthisure graders.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseGrader(ABC):
    """
    Abstract base for deterministic step-by-step graders.

    Reward constants shared across all graders.
    """

    REWARD_MEMBER_LOOKUP = 0.05
    REWARD_CPT_RESOLUTION = 0.10
    REWARD_COST_SHARE_CORRECT = 0.15
    REWARD_PA_FLAG = 0.10
    REWARD_DENIAL_DECODE = 0.10
    REWARD_APPEAL_LETTER = 0.20
    REWARD_FULL_RESOLUTION = 0.30

    PENALTY_HALLUCINATION = -0.20
    PENALTY_MISSED_PA = -0.15
    PENALTY_WRONG_COB = -0.20
    PENALTY_STEP_BUDGET = -0.10

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
