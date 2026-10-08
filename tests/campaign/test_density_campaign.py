"""The campaign's judgement and its durability.

Most of these are refusals. A 96-hour autonomous run has nobody watching it, so the
only thing standing between a broken simulation and a reported density is the code
below.
"""

from __future__ import annotations

import json
import math

import pytest

from polymer_engine.orchestrator.density_campaign import (
    CampaignState,
    CandidateSpec,
    ExperimentResult,
    ReplicaResult,
    analyse_replica_density,
    combine_replica_densities,
    requeue_unfinished,
)


def replica(index: int, density: float | None, stderr: float | None = 1.0,
            succeeded: bool = True) -> ReplicaResult:
    return ReplicaResult(
        replica=index, seed=1000 + index, directory=f"r{index}",
        succeeded=succeeded, density_kg_m3=density, density_stderr=stderr,
    )


class TestReplicaCombination:
    """Replicas are the independent units; frames are not."""

    def test_three_agreeing_replicas_pass(self) -> None:
        out = combine_replica_densities(
            [replica(1, 830.0), replica(2, 832.0), replica(3, 831.0)],
            chi_square_max=4.0, required=3,
        )
        assert out["usable"] is True
        assert out["status"] == "PASS"
        assert out["mean"] == pytest.approx(831.0)
        assert out["n_replicas"] == 3

    def test_the_uncertainty_is_the_error_of_the_replica_means(self) -> None:
        """Not the frame-level error, which would be pseudoreplication."""
        values = [830.0, 832.0, 831.0]
        out = combine_replica_densities(
            [replica(i + 1, v) for i, v in enumerate(values)], chi_square_max=4.0, required=3
        )
        mean = sum(values) / 3
        variance = sum((v - mean) ** 2 for v in values) / 2
        assert out["stderr"] == pytest.approx(math.sqrt(variance / 3))

    def test_two_replicas_where_three_are_required_is_inconclusive(self) -> None:
        out = combine_replica_densities(
            [replica(1, 830.0), replica(2, 832.0)], chi_square_max=4.0, required=3
        )
        assert out["usable"] is False
        assert out["status"] == "INCONCLUSIVE"
        assert "reproducibility is not demonstrated" in out["reason"]

    def test_disagreeing_replicas_fail_on_reduced_chi_square(self) -> None:
        """Precise replicas that disagree are worse than imprecise ones that agree."""
        out = combine_replica_densities(
            [replica(1, 800.0, 0.5), replica(2, 860.0, 0.5), replica(3, 830.0, 0.5)],
            chi_square_max=4.0, required=3,
        )
        assert out["usable"] is False
        assert out["status"] == "FAIL"
        assert out["chi_square"] > 4.0
        assert "disagree beyond their own uncertainties" in out["reason"]

    def test_a_failed_replica_does_not_count_toward_the_requirement(self) -> None:
        out = combine_replica_densities(
            [replica(1, 830.0), replica(2, 831.0), replica(3, None, None, succeeded=False)],
            chi_square_max=4.0, required=3,
        )
        assert out["usable"] is False
        assert out["status"] == "INCONCLUSIVE"

    def test_no_replicas_at_all_is_inconclusive_not_an_error(self) -> None:
        out = combine_replica_densities([], chi_square_max=4.0, required=3)
        assert out["usable"] is False
        assert out["status"] == "INCONCLUSIVE"


class TestReplicaStatistics:
    def test_a_short_series_is_refused(self) -> None:
        out = analyse_replica_density([830.0] * 5, min_effective_samples=20.0)
        assert out["usable"] is False
        assert "frames" in out["reason"]

    def test_a_drifting_series_is_refused_however_many_frames(self) -> None:
        """Still-compressing NPT looks like plenty of data and is not equilibrated."""
        drifting = [700.0 + 0.15 * i for i in range(2000)]
        out = analyse_replica_density(drifting, min_effective_samples=20.0,
                                      max_drift_fraction=0.02)
        assert out["usable"] is False
        assert "drift" in out["reason"]

    def test_a_correlated_series_reports_fewer_effective_samples_than_frames(self) -> None:
        import numpy as np

        rng = np.random.default_rng(7)
        value, series = 830.0, []
        for _ in range(4000):                      # AR(1): strongly correlated
            value = 830.0 + 0.95 * (value - 830.0) + rng.normal(0.0, 1.0)
            series.append(value)
        out = analyse_replica_density(series, min_effective_samples=20.0)
        assert out["effective_samples"] < out["n_frames"]
        assert out["statistical_inefficiency"] > 1.0

    def test_a_well_sampled_series_passes(self) -> None:
        import numpy as np

        rng = np.random.default_rng(3)
        series = list(830.0 + rng.normal(0.0, 2.0, 3000))
        out = analyse_replica_density(series, min_effective_samples=20.0)
        assert out["usable"] is True
        assert out["mean"] == pytest.approx(830.0, abs=1.0)


class TestDurability:
    """The filesystem is the source of truth; no conversation is required to resume."""

    def test_state_round_trips_through_disk(self, tmp_path) -> None:
        state = CampaignState(tmp_path)
        state.queue.append(CandidateSpec(name="polyethylene", repeat_unit_smiles="*CC*",
                                         origin="dataset", design_reason="calibration"))
        result = ExperimentResult(candidate="polypropylene", started_at="2026-01-01T00:00:00Z",
                                  density_kg_m3=900.0, scientifically_usable=True,
                                  gate_status="PASS")
        result.replicas.append(replica(1, 900.0))
        state.results["polypropylene"] = result
        state.completed.append("polypropylene")
        state.iteration = 5
        state.save()

        resumed = CampaignState.load(tmp_path)
        assert resumed is not None
        assert resumed.iteration == 5
        assert resumed.completed == ["polypropylene"]
        assert [c.name for c in resumed.queue] == ["polyethylene"]
        assert resumed.results["polypropylene"].density_kg_m3 == 900.0
        assert resumed.results["polypropylene"].replicas[0].seed == 1001

    def test_loading_an_absent_campaign_returns_none(self, tmp_path) -> None:
        assert CampaignState.load(tmp_path) is None

    def test_the_checkpoint_write_is_atomic(self, tmp_path) -> None:
        """A killed write must not truncate the only record of the campaign."""
        state = CampaignState(tmp_path)
        state.iteration = 3
        path = state.save()
        assert path.is_file()
        assert not (tmp_path / "campaign_state.tmp").exists()
        json.loads(path.read_text())          # complete and parseable

    def test_decisions_are_appended_not_overwritten(self, tmp_path) -> None:
        state = CampaignState(tmp_path)
        state.record_decision({"iteration": 1, "action": "simulate"})
        state.record_decision({"iteration": 2, "action": "judged"})
        lines = (tmp_path / "decision_log.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        assert state.decisions == 2
        assert all("recorded_at" in json.loads(line) for line in lines)

    def test_elapsed_and_remaining_are_consistent(self, tmp_path) -> None:
        state = CampaignState(tmp_path)
        assert state.elapsed_hours() >= 0.0
        assert state.remaining_hours(96.0) <= 96.0
        assert state.remaining_hours(0.0) == 0.0


class TestSeedDiscipline:
    def test_replica_seeds_are_distinct_and_reproducible(self) -> None:
        """Identical seeds would make replica agreement a tautology."""
        def seeds_for(name: str) -> list[int]:
            return [100003 + (sum(ord(c) for c in name) * 31) % 90000 + r * 7919
                    for r in (1, 2, 3)]

        first = seeds_for("polyethylene")
        assert len(set(first)) == 3
        assert first == seeds_for("polyethylene")            # reproducible
        assert set(first).isdisjoint(seeds_for("polypropylene"))

    def test_a_new_campaign_is_resumable_before_any_candidate_finishes(self, tmp_path) -> None:
        """Regression for BUG-001.

        The checkpoint used to be written only after a candidate was judged. The first
        candidate takes tens of minutes, so a crash in that window lost the queue, the
        iteration counter and the seeded candidate list. A campaign that never crashes
        never reveals this, which is why it needs a test rather than a run.
        """
        state = CampaignState(tmp_path)
        state.queue.extend([
            CandidateSpec(name="polyethylene", repeat_unit_smiles="*CC*",
                          origin="dataset", design_reason="Stage A"),
            CandidateSpec(name="polypropylene", repeat_unit_smiles="*CC(C)*",
                          origin="dataset", design_reason="Stage A"),
        ])
        state.save()                       # before any candidate has run

        resumed = CampaignState.load(tmp_path)
        assert resumed is not None, "a freshly seeded campaign must be resumable"
        assert [c.name for c in resumed.queue] == ["polyethylene", "polypropylene"]
        assert resumed.results == {}

    def test_a_partial_candidate_survives_a_crash(self, tmp_path) -> None:
        """One finished replica must not be lost because the next one died."""
        state = CampaignState(tmp_path)
        partial = ExperimentResult(candidate="polyethylene", started_at="2026-01-01T00:00:00Z")
        partial.replicas.append(replica(1, 829.7))
        state.results["polyethylene"] = partial
        state.save()

        resumed = CampaignState.load(tmp_path)
        assert resumed is not None
        recovered = resumed.results["polyethylene"]
        assert len(recovered.replicas) == 1
        assert recovered.replicas[0].density_kg_m3 == pytest.approx(829.7)
        assert recovered.scientifically_usable is False    # one replica is not a result


class TestResumeRetry:
    """Regression for BUG-003: a crash used to cost a whole candidate."""

    @staticmethod
    def candidates() -> dict[str, CandidateSpec]:
        return {
            "polyethylene": CandidateSpec(name="polyethylene", repeat_unit_smiles="*CC*",
                                          origin="dataset", design_reason="Stage A"),
            "polypropylene": CandidateSpec(name="polypropylene", repeat_unit_smiles="*CC(C)*",
                                           origin="dataset", design_reason="Stage A"),
        }

    @staticmethod
    def partial(name: str, n_usable: int) -> ExperimentResult:
        result = ExperimentResult(candidate=name, started_at="2026-01-01T00:00:00Z",
                                  gate_status="INCONCLUSIVE", scientifically_usable=False)
        for index in range(1, n_usable + 1):
            result.replicas.append(replica(index, 830.0))
        return result

    def test_an_interrupted_candidate_goes_back_on_the_queue(self, tmp_path) -> None:
        state = CampaignState(tmp_path)
        state.results["polyethylene"] = self.partial("polyethylene", 1)
        restored = requeue_unfinished(state, self.candidates(),
                                      required_replicas=3, max_attempts=2)
        assert restored == ["polyethylene"]
        assert [c.name for c in state.queue] == ["polyethylene"]

    def test_a_validated_candidate_is_not_re_run(self, tmp_path) -> None:
        state = CampaignState(tmp_path)
        done = self.partial("polyethylene", 3)
        done.scientifically_usable = True
        done.gate_status = "PASS"
        state.results["polyethylene"] = done
        assert requeue_unfinished(state, self.candidates(),
                                  required_replicas=3, max_attempts=2) == []

    def test_a_candidate_with_enough_replicas_is_not_re_run(self, tmp_path) -> None:
        """It failed for a scientific reason, not an interruption."""
        state = CampaignState(tmp_path)
        state.results["polyethylene"] = self.partial("polyethylene", 3)
        assert requeue_unfinished(state, self.candidates(),
                                  required_replicas=3, max_attempts=2) == []

    def test_retries_are_bounded(self, tmp_path) -> None:
        """A candidate that cannot converge must not loop forever."""
        state = CampaignState(tmp_path)
        state.results["polyethylene"] = self.partial("polyethylene", 1)
        state.attempts["polyethylene"] = 2
        assert requeue_unfinished(state, self.candidates(),
                                  required_replicas=3, max_attempts=2) == []

    def test_a_candidate_already_queued_is_not_duplicated(self, tmp_path) -> None:
        state = CampaignState(tmp_path)
        state.results["polyethylene"] = self.partial("polyethylene", 1)
        state.queue.append(self.candidates()["polyethylene"])
        assert requeue_unfinished(state, self.candidates(),
                                  required_replicas=3, max_attempts=2) == []
        assert len(state.queue) == 1

    def test_attempt_counts_survive_a_restart(self, tmp_path) -> None:
        state = CampaignState(tmp_path)
        state.attempts["polyethylene"] = 1
        state.save()
        resumed = CampaignState.load(tmp_path)
        assert resumed is not None
        assert resumed.attempts["polyethylene"] == 1
