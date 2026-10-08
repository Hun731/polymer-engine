"""Open-ended campaigns: what stops them, and what merely pauses them.

The 96-hour box is gone. These tests pin the replacement -- a campaign runs until
science, resources or a person says otherwise -- and, just as importantly, pin the
distinction between a stop that is a conclusion and one that is an interruption.
"""

from __future__ import annotations

import pytest

from polymer_engine.orchestrator.stop_policy import (
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    CampaignStatus,
    StopPolicy,
)

HEALTHY = {"disk_free_gb": 3000.0, "queue_depth": 5, "n_validated": 2,
           "consecutive_failures": 0}


class TestOpenEnded:
    @pytest.mark.parametrize("hours", [1.0, 96.0, 500.0, 10_000.0])
    def test_time_alone_never_stops_an_open_ended_campaign(self, hours: float) -> None:
        """Regression for the removed 96-hour limit."""
        decision = StopPolicy().evaluate(elapsed_hours=hours, **HEALTHY)
        assert decision.status is CampaignStatus.RUNNING
        assert decision.should_continue is True

    def test_open_ended_is_the_default(self) -> None:
        policy = StopPolicy()
        assert policy.duration_hours is None
        assert policy.as_dict()["open_ended"] is True

    def test_a_deliberate_time_box_still_works(self) -> None:
        """Removing the assumption is not the same as removing the capability."""
        policy = StopPolicy(duration_hours=96.0)
        assert policy.evaluate(elapsed_hours=95.0, **HEALTHY).should_continue is True
        expired = policy.evaluate(elapsed_hours=97.0, **HEALTHY)
        assert expired.status is CampaignStatus.COMPLETED
        assert "budget is spent" in expired.reason


class TestStopReasons:
    def test_low_disk_blocks_on_resources_not_science(self) -> None:
        decision = StopPolicy().evaluate(
            elapsed_hours=1.0, **{**HEALTHY, "disk_free_gb": 10.0})
        assert decision.status is CampaignStatus.RESOURCE_BLOCKED
        assert decision.status.resumable is True
        assert decision.status.terminal is False

    def test_repeated_failure_is_systemic_not_unlucky(self) -> None:
        decision = StopPolicy().evaluate(
            elapsed_hours=1.0,
            **{**HEALTHY, "consecutive_failures": DEFAULT_MAX_CONSECUTIVE_FAILURES})
        assert decision.status is CampaignStatus.FAILED
        assert "machine or configuration" in decision.reason

    def test_one_failure_short_of_the_limit_keeps_going(self) -> None:
        decision = StopPolicy().evaluate(
            elapsed_hours=1.0,
            **{**HEALTHY, "consecutive_failures": DEFAULT_MAX_CONSECUTIVE_FAILURES - 1})
        assert decision.should_continue is True

    def test_an_empty_queue_completes_rather_than_fails(self) -> None:
        """Out of work is a conclusion, not an error."""
        decision = StopPolicy().evaluate(
            elapsed_hours=1.0, **{**HEALTHY, "queue_depth": 0})
        assert decision.status is CampaignStatus.COMPLETED

    def test_needing_a_person_waits_rather_than_terminating(self) -> None:
        decision = StopPolicy().evaluate(
            elapsed_hours=1.0, awaiting_human=True, **HEALTHY)
        assert decision.status is CampaignStatus.WAITING_FOR_INPUT
        assert decision.status.resumable is True

    def test_reaching_the_validated_target_completes(self) -> None:
        decision = StopPolicy(target_validated=2).evaluate(
            elapsed_hours=1.0, **HEALTHY)
        assert decision.status is CampaignStatus.COMPLETED
        assert "target" in decision.reason

    def test_a_saturated_objective_completes(self) -> None:
        """Nothing more to learn is a scientific stop, not a timeout."""
        policy = StopPolicy(saturation_tolerance=0.01, saturation_window=5)
        decision = policy.evaluate(
            elapsed_hours=1.0,
            recent_scores=[0.900, 0.901, 0.9005, 0.9002, 0.9001], **HEALTHY)
        assert decision.status is CampaignStatus.COMPLETED
        assert "saturated" in decision.reason

    def test_an_improving_model_is_not_saturated(self) -> None:
        policy = StopPolicy(saturation_tolerance=0.01, saturation_window=5)
        decision = policy.evaluate(
            elapsed_hours=1.0,
            recent_scores=[0.10, 0.30, 0.55, 0.70, 0.88], **HEALTHY)
        assert decision.should_continue is True

    def test_too_few_scores_is_not_saturation(self) -> None:
        policy = StopPolicy(saturation_tolerance=0.01, saturation_window=5)
        assert policy.evaluate(elapsed_hours=1.0, recent_scores=[0.9, 0.9],
                               **HEALTHY).should_continue is True


class TestSeverityOrdering:
    def test_resource_safety_outranks_everything(self) -> None:
        """A full disk must not be reported as a scientific conclusion."""
        decision = StopPolicy(target_validated=1).evaluate(
            elapsed_hours=1.0,
            **{**HEALTHY, "disk_free_gb": 1.0, "queue_depth": 0,
               "consecutive_failures": 99})
        assert decision.status is CampaignStatus.RESOURCE_BLOCKED

    def test_systemic_failure_outranks_a_scientific_conclusion(self) -> None:
        decision = StopPolicy(target_validated=1).evaluate(
            elapsed_hours=1.0, **{**HEALTHY, "consecutive_failures": 5})
        assert decision.status is CampaignStatus.FAILED


class TestStatusSemantics:
    def test_resumable_and_terminal_are_disjoint(self) -> None:
        for status in CampaignStatus:
            assert not (status.resumable and status.terminal), status

    def test_the_states_the_spec_requires_all_exist(self) -> None:
        expected = {"RUNNING", "PAUSED", "WAITING_FOR_INPUT", "RESOURCE_BLOCKED",
                    "SCIENTIFICALLY_BLOCKED", "COMPLETED", "FAILED"}
        assert {s.value for s in CampaignStatus} == expected

    def test_a_transient_problem_is_resumable(self) -> None:
        """A paused campaign costs an idle machine; a terminated one costs the run."""
        assert CampaignStatus.PAUSED.resumable is True
        assert CampaignStatus.PAUSED.terminal is False
