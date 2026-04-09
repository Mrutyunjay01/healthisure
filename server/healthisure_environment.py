"""
HealthisureEnvironment — core OpenEnv environment class.

Implements the Environment interface from openenv-core, wiring together
the task definitions, action handlers, and graders into a fully compliant
step / reset / state loop.
"""

from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .actions.handlers import ActionHandler
from .graders.grader1 import Grader1
from .graders.grader2 import Grader2
from .graders.grader3 import Grader3
from .tasks.task1_eligibility import Task1Eligibility
from .tasks.task2_prior_auth import Task2PriorAuth
from .tasks.task3_cob_dispute import Task3CobDispute

from models import HealthisureAction, HealthisureObservation

# ---------------------------------------------------------------------------
# Task & grader registry
# ---------------------------------------------------------------------------

_TASKS = {
    "task1": Task1Eligibility(),
    "task2": Task2PriorAuth(),
    "task3": Task3CobDispute(),
}

_GRADERS = {
    "task1": Grader1(),
    "task2": Grader2(),
    "task3": Grader3(),
}

_DEFAULT_TASK = "task1"

REWARD_FULL_RESOLUTION = 0.30
PENALTY_STEP_BUDGET = -0.10


class HealthisureEnvironment(Environment):
    """
    Stateful RL environment simulating a health insurance support specialist.

    The agent receives a case description on reset(), then iteratively calls
    actions (lookup_member, check_claim_status, draft_appeal_letter, …) until
    it sends a final member response or exhausts the step budget.

    Supports three tasks of increasing difficulty:
      task1 — Benefit Verification & Eligibility       (step budget: 10)
      task2 — Prior Auth & Claim Status Resolution     (step budget: 15)
      task3 — Multi-Party COB Dispute                  (step budget: 20)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._episode_id: str = str(uuid4())
        self._state: State = State(episode_id=self._episode_id, step_count=0)
        self._episode: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_name: Optional[str] = None,
        scenario_id: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> HealthisureObservation:
        """
        Start a new episode.

        Args:
            task_name: One of "task1", "task2", "task3". Defaults to "task1".
            scenario_id: Index of the scenario within the task. Random if None.
            episode_id: Optional custom episode ID.

        Returns:
            Initial HealthisureObservation with the task description.
        """
        task_name = task_name or _DEFAULT_TASK
        if task_name not in _TASKS:
            task_name = _DEFAULT_TASK

        task = _TASKS[task_name]
        scenario = task.get_scenario(scenario_id)

        self._episode_id = episode_id or str(uuid4())
        self._state = State(episode_id=self._episode_id, step_count=0)

        self._episode = {
            "episode_id": self._episode_id,
            "task_name": task_name,
            "scenario": scenario,
            "step_count": 0,
            "step_budget": task.step_budget,
            "done": False,
            "cumulative_reward": 0.0,
            "gold_standard": scenario["gold_standard"],
            "grader_flags": {},
            "action_history": [],
            "member_context": None,
        }

        return HealthisureObservation(
            task_description=task.get_initial_observation_text(scenario),
            available_actions=list(ActionHandler._HANDLERS.keys()),
            last_action_result=None,
            step_count=0,
            step_budget=task.step_budget,
            cumulative_reward=0.0,
            reward=0.0,
            done=False,
            error=None,
            member_context=None,
            task_name=task_name,
        )

    def step(self, action: HealthisureAction) -> HealthisureObservation:
        """
        Execute one action and return the resulting observation, reward, done flag.

        The grader evaluates the action for incremental reward. When
        send_member_response is called (or the step budget is exceeded), the
        episode ends with a full-resolution bonus (if earned) or a budget penalty.
        """
        ep = self._episode

        # Guard: already done
        if ep.get("done", False):
            return self._build_obs(
                last_result="Episode already ended. Call reset() to start a new episode.",
                error="Episode already done.",
            )

        # Validate action name
        if action.action_name not in ActionHandler._HANDLERS:
            return self._build_obs(
                last_result=None,
                error=f"Unknown action '{action.action_name}'. Valid actions: {list(ActionHandler._HANDLERS.keys())}",
            )

        # Execute action
        result = ActionHandler.dispatch(action.action_name, action.parameters)

        # Update member context for informational lookups
        self._update_member_context(action.action_name, result)

        # Grade the step
        grader = _GRADERS[ep["task_name"]]
        step_reward = grader.grade_step(
            action_name=action.action_name,
            parameters=action.parameters,
            action_result=result,
            episode_state=ep,
        )
        ep["cumulative_reward"] = round(ep["cumulative_reward"] + step_reward, 4)
        ep["step_count"] += 1
        self._state = State(episode_id=self._episode_id, step_count=ep["step_count"])

        # Record history
        ep["action_history"].append({
            "step": ep["step_count"],
            "action": action.action_name,
            "parameters": action.parameters,
            "result_summary": result.get("message", "")[:200],
            "step_reward": step_reward,
        })

        # Determine done
        done = False
        terminal_reason = None

        if action.action_name == "send_member_response" and result.get("success"):
            done = True
            terminal_reason = "response_sent"
            # Full resolution bonus
            if grader.is_resolved(ep):
                ep["cumulative_reward"] = round(ep["cumulative_reward"] + REWARD_FULL_RESOLUTION, 4)

        elif ep["step_count"] >= ep["step_budget"]:
            done = True
            terminal_reason = "budget_exceeded"
            ep["cumulative_reward"] = round(ep["cumulative_reward"] + PENALTY_STEP_BUDGET, 4)

        # Clamp final score to (0.001, 0.999) — OpenEnv requires strictly
        # between 0 and 1; raw rewards can underflow to 0.0 (all failures)
        # or overflow above 1.0 (Task 3 perfect path reaches ~1.20).
        if done:
            ep["cumulative_reward"] = round(
                max(0.001, min(0.999, ep["cumulative_reward"])), 4
            )

        ep["done"] = done

        error_msg = result.get("error") if not result.get("success") else None

        # Use the clamped cumulative reward as the terminal reward so that the
        # OpenEnv framework (which reads observation.reward, not cumulative_reward)
        # always receives a value strictly within (0, 1).
        reported_reward = ep["cumulative_reward"] if done else step_reward

        return self._build_obs(
            last_result=result.get("message", ""),
            error=error_msg,
            done=done,
            step_reward=reported_reward,
        )

    @property
    def state(self) -> State:
        """Return the current episode state (for /state endpoint and Gradio UI)."""
        ep = self._episode
        return State(
            episode_id=ep.get("episode_id"),
            step_count=ep.get("step_count", 0),
            # extra fields (State allows extra="allow")
            task_name=ep.get("task_name"),
            scenario_id=ep.get("scenario", {}).get("scenario_id"),
            step_budget=ep.get("step_budget", 10),
            done=ep.get("done", False),
            cumulative_reward=ep.get("cumulative_reward", 0.0),
            grader_flags=ep.get("grader_flags", {}),
            action_history=ep.get("action_history", []),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        last_result: Optional[str],
        error: Optional[str] = None,
        done: bool = False,
        step_reward: Optional[float] = None,
    ) -> HealthisureObservation:
        ep = self._episode
        scenario = ep.get("scenario", {})
        task_name = ep.get("task_name", "")
        task = _TASKS.get(task_name)
        task_desc = (
            task.get_initial_observation_text(scenario)
            if task and scenario
            else "No active episode. Call reset() first."
        )
        return HealthisureObservation(
            task_description=task_desc,
            available_actions=list(ActionHandler._HANDLERS.keys()),
            last_action_result=last_result,
            step_count=ep.get("step_count", 0),
            step_budget=ep.get("step_budget", 10),
            cumulative_reward=ep.get("cumulative_reward", 0.0),
            reward=step_reward if step_reward is not None else 0.0,
            done=done or ep.get("done", False),
            error=error,
            member_context=ep.get("member_context"),
            task_name=task_name,
        )

    def _update_member_context(self, action_name: str, result: Dict[str, Any]) -> None:
        """Accumulate discovered member/claim info into the context string."""
        ep = self._episode
        if not result.get("success"):
            return
        ctx_lines = ep.get("member_context") or ""
        msg = result.get("message", "")
        if action_name in {
            "lookup_member",
            "check_claim_status",
            "check_deductible_status",
            "lookup_plan_benefits",
            "check_prior_auth_required",
        }:
            if msg and msg not in ctx_lines:
                ep["member_context"] = (ctx_lines + "\n\n" + msg).strip()
