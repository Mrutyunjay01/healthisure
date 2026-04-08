"""
Abstract base class for tasks in Healthisure.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import random


class BaseTask(ABC):
    """
    Abstract base for the three environment tasks.

    Each concrete task defines a list of scenarios (pre-built cases),
    a step budget, and a gold-standard resolution per scenario used
    by the grader for deterministic scoring.
    """

    task_name: str = ""
    step_budget: int = 10

    @property
    @abstractmethod
    def scenarios(self) -> List[Dict[str, Any]]:
        """
        List of scenario dicts. Each scenario contains at minimum:
          - scenario_id: int
          - task_description: str  (shown to the agent)
          - member_id: str
          - gold_standard: dict   (expected resolution, used by grader)
        """

    def get_scenario(self, scenario_id: Optional[int] = None) -> Dict[str, Any]:
        """Return a specific or random scenario."""
        if scenario_id is not None:
            return self.scenarios[scenario_id % len(self.scenarios)]
        return random.choice(self.scenarios)

    @abstractmethod
    def get_initial_observation_text(self, scenario: Dict[str, Any]) -> str:
        """Return the task description string shown to the agent on reset."""
