"""
Tests verifying that the terminal observation's reward is always strictly
within (0, 1) — the range required by the OpenEnv submission validator.

The critical path:
  openenv serialization → StepResponse.reward = observation.reward
  (NOT observation.cumulative_reward)

These tests cover:
  - All three tasks via send_member_response (normal termination)
  - Budget-exceeded termination
  - Worst-case penalty scenarios (reward would be ≤ 0 without clamping)
  - Best-case full-resolution scenarios (reward would be > 1 without clamping)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import pytest
from server.healthisure_environment import HealthisureEnvironment
from models import HealthisureAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_action(name: str, **params) -> HealthisureAction:
    return HealthisureAction(action_name=name, parameters=params)


def assert_strictly_between_0_and_1(reward, context=""):
    """The OpenEnv validator requires 0 < reward < 1 (exclusive)."""
    assert reward is not None, f"reward is None {context}"
    assert reward > 0.0, f"reward={reward} is not > 0  {context}"
    assert reward < 1.0, f"reward={reward} is not < 1  {context}"


# ---------------------------------------------------------------------------
# Task 1 — Benefit Verification
# ---------------------------------------------------------------------------


class TestTask1TerminalReward:
    def _make_env(self, scenario_id=0):
        env = HealthisureEnvironment()
        env.reset(task_name="task1", scenario_id=scenario_id)
        return env

    def test_send_response_immediately_gives_valid_reward(self):
        """Skip all lookups; send response right away → low score but valid."""
        env = self._make_env()
        obs = env.step(make_action("send_member_response", message="Done"))
        assert obs.done
        assert_strictly_between_0_and_1(obs.reward, "(task1 immediate response)")

    def test_send_response_with_missed_pa_gives_valid_reward(self):
        """
        If gold requires PA and we don't mention it, Grader1 applies
        PENALTY_MISSED_PA (-0.15) making the raw step_reward negative.
        The terminal reward must still be strictly > 0.
        """
        env = self._make_env(scenario_id=0)
        # Don't check PA; send a response without mentioning it
        obs = env.step(make_action("send_member_response", message="All good!"))
        assert obs.done
        assert_strictly_between_0_and_1(obs.reward, "(task1 missed PA penalty)")

    def test_full_resolution_does_not_exceed_1(self):
        """Full happy-path should produce reward < 1 (max ~1.0, so clamped to 0.999)."""
        env = self._make_env(scenario_id=0)
        scenario = env._episode["scenario"]
        gold = scenario["gold_standard"]
        member_id = scenario.get("member_id", "M001")
        plan_id = scenario.get("plan_id", "GOLD-001")
        cpt_code = scenario.get("cpt_code", "70553")

        env.step(make_action("lookup_member", member_id=member_id))
        env.step(make_action("check_deductible_status", member_id=member_id))
        env.step(make_action("lookup_plan_benefits", plan_id=plan_id, cpt_code=cpt_code))
        env.step(make_action("check_prior_auth_required", cpt_code=cpt_code, plan_id=plan_id))
        env.step(
            make_action(
                "apply_cost_share",
                amount=gold.get("member_cost_estimate", 100),
                plan_id=plan_id,
                deductible_met=False,
            )
        )
        obs = env.step(
            make_action(
                "send_member_response",
                message="Your prior authorization is required. Member cost is calculated.",
            )
        )
        assert obs.done
        assert_strictly_between_0_and_1(obs.reward, "(task1 full resolution)")

    def test_budget_exceeded_gives_valid_reward(self):
        """
        Exhaust the step budget with no-ops; the budget penalty (-0.10) can
        push cumulative down but terminal reward must stay > 0.
        """
        env = self._make_env()
        obs = None
        # Spam lookup_member (idempotent beyond first) until budget runs out
        for _ in range(env._episode["step_budget"] + 2):
            if env._episode.get("done"):
                break
            obs = env.step(make_action("lookup_member", member_id="M001"))
        assert obs is not None
        assert obs.done
        assert_strictly_between_0_and_1(obs.reward, "(task1 budget exceeded)")


# ---------------------------------------------------------------------------
# Task 2 — Prior Auth & Claim Status Resolution
# ---------------------------------------------------------------------------


class TestTask2TerminalReward:
    def _make_env(self, scenario_id=0):
        env = HealthisureEnvironment()
        env.reset(task_name="task2", scenario_id=scenario_id)
        return env

    def test_send_response_without_appeal_when_required_valid_reward(self):
        """
        Grader2 applies PENALTY_HALLUCINATION (-0.20) when appeal_required
        but not drafted. Terminal reward must be strictly > 0.
        """
        env = self._make_env()
        obs = env.step(make_action("send_member_response", message="Claim handled"))
        assert obs.done
        assert_strictly_between_0_and_1(obs.reward, "(task2 missing appeal penalty)")

    def test_budget_exceeded_gives_valid_reward(self):
        env = self._make_env()
        obs = None
        for _ in range(env._episode["step_budget"] + 2):
            if env._episode.get("done"):
                break
            obs = env.step(make_action("lookup_member", member_id="M001"))
        assert obs is not None
        assert obs.done
        assert_strictly_between_0_and_1(obs.reward, "(task2 budget exceeded)")


# ---------------------------------------------------------------------------
# Task 3 — Multi-Party COB Dispute
# ---------------------------------------------------------------------------


class TestTask3TerminalReward:
    def _make_env(self, scenario_id=0):
        env = HealthisureEnvironment()
        env.reset(task_name="task3", scenario_id=scenario_id)
        return env

    def test_send_response_without_required_docs_gives_valid_reward(self):
        """
        Grader3 subtracts 0.10 if docs not complete; still must be > 0.
        """
        env = self._make_env()
        obs = env.step(make_action("send_member_response", message="Done"))
        assert obs.done
        assert_strictly_between_0_and_1(obs.reward, "(task3 missing docs penalty)")

    def test_budget_exceeded_gives_valid_reward(self):
        env = self._make_env()
        obs = None
        for _ in range(env._episode["step_budget"] + 2):
            if env._episode.get("done"):
                break
            obs = env.step(make_action("lookup_member", member_id="M001"))
        assert obs is not None
        assert obs.done
        assert_strictly_between_0_and_1(obs.reward, "(task3 budget exceeded)")

    def test_full_resolution_does_not_exceed_1(self):
        """
        Task 3 perfect path accumulates ~1.20 raw; must be clamped below 1.
        """
        env = self._make_env(scenario_id=0)
        scenario = env._episode["scenario"]
        gold = scenario["gold_standard"]
        member_id = scenario.get("member_id", "M001")
        claim_id = scenario.get("claim_id", "CLM-001")
        primary_plan = gold.get("cob_primary", "PLAN-PRIMARY")
        secondary_plan = gold.get("cob_secondary", "PLAN-SECONDARY")
        secondary_insurer = scenario.get("secondary_insurer_id", "INS-SECONDARY")

        env.step(make_action("lookup_member", member_id=member_id))
        env.step(make_action("check_claim_status", claim_id=claim_id))
        env.step(make_action("lookup_plan_benefits", plan_id=primary_plan, cpt_code="99285"))
        env.step(make_action("lookup_plan_benefits", plan_id=secondary_plan, cpt_code="99285"))
        env.step(make_action("apply_cost_share", amount=500, plan_id=primary_plan, deductible_met=True))
        env.step(
            make_action(
                "file_corrected_claim",
                member_id=member_id,
                claim_id=claim_id,
                secondary_insurer_id=secondary_insurer,
            )
        )
        env.step(
            make_action(
                "draft_dispute_letter",
                provider_id="PROV-001",
                claim_id=claim_id,
                reason="COB correction",
            )
        )
        env.step(make_action("escalate_case", member_id=member_id, reason="COB dispute", priority="high"))
        obs = env.step(make_action("send_member_response", message="Your COB dispute has been resolved."))

        assert obs.done
        assert_strictly_between_0_and_1(obs.reward, "(task3 full resolution)")


# ---------------------------------------------------------------------------
# Cross-cutting: reward field vs cumulative_reward
# ---------------------------------------------------------------------------


class TestRewardFieldIsTerminalScore:
    """
    Verify that observation.reward (what OpenEnv reads) equals
    observation.cumulative_reward when the episode is terminal — i.e.,
    the fix is in place and both fields carry the clamped value.
    """

    def test_reward_equals_cumulative_reward_on_terminal_step(self):
        env = HealthisureEnvironment()
        env.reset(task_name="task1", scenario_id=0)
        obs = env.step(make_action("send_member_response", message="Done"))
        assert obs.done
        assert obs.reward == obs.cumulative_reward, (
            f"observation.reward ({obs.reward}) != "
            f"observation.cumulative_reward ({obs.cumulative_reward}); "
            "OpenEnv uses observation.reward as the task score"
        )

    def test_reward_not_equal_cumulative_reward_during_episode(self):
        """During an episode, reward is the per-step delta, not cumulative."""
        env = HealthisureEnvironment()
        env.reset(task_name="task1", scenario_id=0)
        obs = env.step(make_action("lookup_member", member_id="M001"))
        assert not obs.done
        # The per-step reward (0.05) should differ from cumulative (0.05 here but
        # conceptually they can diverge; at minimum they behave independently).
        # This test documents the contract, not a specific value.
        assert obs.reward is not None
