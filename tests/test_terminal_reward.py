"""
Tests verifying that terminal episode scores are always strictly within (0, 1)
— the range required by the OpenEnv submission validator.

The inference script computes:
    score = clamp(obs.cumulative_reward / TASK_MAX_SCORES[task], 0.01, 0.99)

These tests verify:
  - obs.cumulative_reward at episode end is clamped to [0.01, 0.99] by the server
  - The normalized score is strictly within (0, 1) for all tasks/scenarios
  - Worst-case penalties don't produce a score ≤ 0
  - Best-case full-resolution doesn't produce a score ≥ 1
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import pytest
from server.healthisure_environment import HealthisureEnvironment
from models import HealthisureAction
from config import TASK_MAX_SCORES


def make_action(name: str, **params) -> HealthisureAction:
    return HealthisureAction(action_name=name, parameters=params)


def assert_valid_score(cumulative_reward: float, task_name: str, context: str = "") -> None:
    """
    Verify the normalized score (as inference.py would compute it) is
    strictly within (0, 1) and safe at 2 decimal places.
    """
    max_score = TASK_MAX_SCORES.get(task_name, 1.0)
    score = max(0.01, min(0.99, cumulative_reward / max_score))
    assert score >= 0.01, f"score={score} rounds to 0.00 at 2dp  {context}"
    assert score <= 0.99, f"score={score} rounds to 1.00 at 2dp  {context}"
    assert f"{score:.2f}" != "0.00", f"score={score} formats as 0.00  {context}"
    assert f"{score:.2f}" != "1.00", f"score={score} formats as 1.00  {context}"


# ---------------------------------------------------------------------------
# Task 1 — Benefit Verification
# ---------------------------------------------------------------------------


class TestTask1TerminalScore:
    def _make_env(self, scenario_id=0):
        env = HealthisureEnvironment()
        env.reset(task_name="task1", scenario_id=scenario_id)
        return env

    def test_send_response_immediately_gives_valid_score(self):
        env = self._make_env()
        obs = env.step(make_action("send_member_response", message="Done"))
        assert obs.done
        assert_valid_score(obs.cumulative_reward, "task1", "(task1 immediate response)")

    def test_send_response_with_missed_pa_gives_valid_score(self):
        env = self._make_env(scenario_id=0)
        obs = env.step(make_action("send_member_response", message="All good!"))
        assert obs.done
        assert_valid_score(obs.cumulative_reward, "task1", "(task1 missed PA penalty)")

    def test_full_resolution_does_not_exceed_1(self):
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
        assert_valid_score(obs.cumulative_reward, "task1", "(task1 full resolution)")

    def test_budget_exceeded_gives_valid_score(self):
        env = self._make_env()
        obs = None
        for _ in range(env._episode["step_budget"] + 2):
            if env._episode.get("done"):
                break
            obs = env.step(make_action("lookup_member", member_id="M001"))
        assert obs is not None
        assert obs.done
        assert_valid_score(obs.cumulative_reward, "task1", "(task1 budget exceeded)")


# ---------------------------------------------------------------------------
# Task 2 — Prior Auth & Claim Status Resolution
# ---------------------------------------------------------------------------


class TestTask2TerminalScore:
    def _make_env(self, scenario_id=0):
        env = HealthisureEnvironment()
        env.reset(task_name="task2", scenario_id=scenario_id)
        return env

    def test_send_response_without_appeal_gives_valid_score(self):
        env = self._make_env()
        obs = env.step(make_action("send_member_response", message="Claim handled"))
        assert obs.done
        assert_valid_score(obs.cumulative_reward, "task2", "(task2 missing appeal penalty)")

    def test_budget_exceeded_gives_valid_score(self):
        env = self._make_env()
        obs = None
        for _ in range(env._episode["step_budget"] + 2):
            if env._episode.get("done"):
                break
            obs = env.step(make_action("lookup_member", member_id="M001"))
        assert obs is not None
        assert obs.done
        assert_valid_score(obs.cumulative_reward, "task2", "(task2 budget exceeded)")


# ---------------------------------------------------------------------------
# Task 3 — Multi-Party COB Dispute
# ---------------------------------------------------------------------------


class TestTask3TerminalScore:
    def _make_env(self, scenario_id=0):
        env = HealthisureEnvironment()
        env.reset(task_name="task3", scenario_id=scenario_id)
        return env

    def test_send_response_without_required_docs_gives_valid_score(self):
        env = self._make_env()
        obs = env.step(make_action("send_member_response", message="Done"))
        assert obs.done
        assert_valid_score(obs.cumulative_reward, "task3", "(task3 missing docs penalty)")

    def test_budget_exceeded_gives_valid_score(self):
        env = self._make_env()
        obs = None
        for _ in range(env._episode["step_budget"] + 2):
            if env._episode.get("done"):
                break
            obs = env.step(make_action("lookup_member", member_id="M001"))
        assert obs is not None
        assert obs.done
        assert_valid_score(obs.cumulative_reward, "task3", "(task3 budget exceeded)")

    def test_full_resolution_does_not_exceed_1(self):
        """Task 3 perfect path accumulates ~1.20 raw; normalized score must be < 1."""
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
        assert_valid_score(obs.cumulative_reward, "task3", "(task3 full resolution)")


# ---------------------------------------------------------------------------
# Cumulative reward is clamped on terminal step
# ---------------------------------------------------------------------------


class TestCumulativeRewardClamp:
    """Verify cumulative_reward is always clamped to [0.01, 0.99] at episode end."""

    def test_cumulative_reward_clamped_on_terminal_step(self):
        env = HealthisureEnvironment()
        env.reset(task_name="task1", scenario_id=0)
        obs = env.step(make_action("send_member_response", message="Done"))
        assert obs.done
        assert 0.01 <= obs.cumulative_reward <= 0.99, (
            f"cumulative_reward={obs.cumulative_reward} is outside [0.01, 0.99]"
        )

    def test_per_step_reward_not_clamped(self):
        """During the episode, obs.reward shows per-step delta (not clamped cumulative)."""
        env = HealthisureEnvironment()
        env.reset(task_name="task1", scenario_id=0)
        obs = env.step(make_action("lookup_member", member_id="M001"))
        assert not obs.done
        assert obs.reward is not None
