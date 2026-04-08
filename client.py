"""
HealthisureEnvClient — typed OpenEnv client for Healthisure.

Subclasses openenv-core's EnvClient so inference scripts and external agents
can talk to the server over WebSocket without importing environment internals.

Sync usage (inference scripts):
    with HealthisureEnvClient(base_url="http://localhost:8000").sync() as env:
        result = env.reset(task_name="task1")
        result = env.step({"action_name": "lookup_member", "parameters": {"member_id": "M001"}})

Async usage:
    async with HealthisureEnvClient(base_url="http://localhost:8000") as env:
        result = await env.reset(task_name="task1")
        result = await env.step(HealthisureAction(...))
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import HealthisureAction, HealthisureObservation


class HealthisureEnvClient(
    EnvClient[HealthisureAction, HealthisureObservation, Dict[str, Any]]
):
    """
    Typed WebSocket client for Healthisure.

    Connects to a running Healthisure server and exposes reset() / step() /
    state() with full type information.  Use the .sync() wrapper for synchronous
    inference scripts.
    """

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _step_payload(self, action: HealthisureAction) -> Dict[str, Any]:
        """Serialize HealthisureAction to the wire format expected by the server."""
        if isinstance(action, dict):
            return action
        if hasattr(action, "model_dump"):
            return action.model_dump()
        return {"action_name": action.action_name, "parameters": action.parameters}

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[HealthisureObservation]:
        """Deserialize the server StepResponse into a typed StepResult."""
        obs_data = payload.get("observation", {})
        # The server serializes `done` and `reward` at the top level (not inside
        # the observation dict) via serialize_observation(). Pull them from there.
        top_done = payload.get("done", False)
        top_reward = payload.get("reward")

        if isinstance(obs_data, dict):
            try:
                obs = HealthisureObservation(**obs_data, done=top_done)
            except Exception:
                obs = HealthisureObservation(
                    task_description=obs_data.get("task_description", ""),
                    available_actions=obs_data.get("available_actions", []),
                    last_action_result=obs_data.get("last_action_result"),
                    step_count=obs_data.get("step_count", 0),
                    step_budget=obs_data.get("step_budget", 10),
                    cumulative_reward=obs_data.get("cumulative_reward", 0.0),
                    done=top_done,
                    error=obs_data.get("error"),
                    member_context=obs_data.get("member_context"),
                    task_name=obs_data.get("task_name", ""),
                )
        else:
            obs = HealthisureObservation(
                task_description="",
                available_actions=[],
                step_count=0,
                step_budget=10,
                cumulative_reward=0.0,
                done=top_done,
                task_name="",
            )

        return StepResult(
            observation=obs,
            reward=top_reward,
            done=top_done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return the raw state dict from the server."""
        return payload
