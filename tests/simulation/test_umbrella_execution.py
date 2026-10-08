"""Umbrella execution: the justification gate, adaptive refinement, and PMF trust.

Windows are sampled from the analytic biased distribution for a known harmonic PMF,
so the recovered free-energy surface has a right answer to check against.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from polymer_engine.core.config import UmbrellaDefaults
from polymer_engine.core.errors import InsufficientDataError
from polymer_engine.core.models import Determination, GateStatus
from polymer_engine.core.provenance import ProvenanceGraph
from polymer_engine.core.units import kT
from polymer_engine.simulation.umbrella import ReactionCoordinate
from polymer_engine.simulation.umbrella_execution import (
    UmbrellaCampaign,
    UmbrellaJustification,
    UmbrellaStatus,
    WindowExecution,
    read_colvar,
)

TEMPERATURE = 300.0
TRUE_K = 200.0       # underlying harmonic PMF stiffness, kJ/mol/nm^2
TRUE_MINIMUM = 1.0   # nm

COORDINATE = ReactionCoordinate(
    name="d", kind="com_distance", units="nm",
    justification="Interchain centre-of-mass separation probes cohesion.",
    group_a="1-100", group_b="101-200",
)


def complete_justification(**overrides) -> UmbrellaJustification:
    base = {
        "question": "How strongly do two chains associate in the melt?",
        "reaction_coordinate": "centre-of-mass distance between two chains",
        "physical_interpretation": "separation of the two chain centres of mass",
        "expected_observable": "depth of the PMF minimum relative to the plateau",
        "reason_for_method": "the association free energy requires sampling the barrier region",
        "starting_state": "contact pair equilibrated at 0.6 nm separation",
        "endpoint_definition": "the PMF plateau beyond 1.5 nm",
        "author": "test",
    }
    base.update(overrides)
    return UmbrellaJustification(**base)


def analytic_runner(seed: int = 0, n_samples: int = 8000, fail_indices: set[int] | None = None):
    """Sample each window from the exact biased distribution for the known PMF.

    p_k(x) ~ exp(-beta[0.5*K*(x-x0)^2 + 0.5*k*(x-x_k)^2]) is Gaussian, so the windows
    are exact and any error in the recovered PMF is the analysis, not the sampling.
    """
    rng = np.random.default_rng(seed)
    beta = 1.0 / kT(TEMPERATURE)
    failures = fail_indices or set()

    def run(window, directory: Path) -> WindowExecution:
        if window.index in failures:
            return WindowExecution(succeeded=False, error="simulated window failure")
        k = window.force_constant
        mean = (TRUE_K * TRUE_MINIMUM + k * window.center) / (TRUE_K + k)
        sd = math.sqrt(1.0 / (beta * (TRUE_K + k)))
        values = rng.normal(mean, sd, n_samples)
        path = Path(directory) / "COLVAR"
        path.write_text(
            "#! FIELDS time d\n"
            + "\n".join(f"{i * 0.5:.2f} {v:.6f}" for i, v in enumerate(values)),
            encoding="utf-8",
        )
        return WindowExecution(succeeded=True, colvar_path=path)

    return run


# ==========================================================================
# The justification gate
# ==========================================================================
class TestJustification:
    def test_a_complete_justification_is_accepted(self) -> None:
        assert complete_justification().complete is True
        assert complete_justification().missing_fields() == []

    @pytest.mark.parametrize(
        "field",
        ["question", "reaction_coordinate", "physical_interpretation",
         "expected_observable", "reason_for_method", "starting_state", "endpoint_definition"],
    )
    def test_every_field_is_required(self, field: str) -> None:
        justification = complete_justification(**{field: ""})
        assert justification.complete is False
        assert field in justification.missing_fields()

    def test_no_justification_means_no_windows_run(self, tmp_path: Path) -> None:
        """The planner can always propose umbrella sampling; that is not a reason to run it."""
        campaign = UmbrellaCampaign(COORDINATE, temperature_k=TEMPERATURE)
        result = campaign.run(
            minimum=0.6, maximum=1.4, root=tmp_path,
            justification=None, runner=analytic_runner(),
        )
        assert result.status is UmbrellaStatus.REQUIRES_EXPERT_DECISION
        assert result.runs == []
        assert result.trustworthy is False
        assert result.determination is Determination.REQUIRES_VALIDATION

    def test_an_incomplete_justification_also_refuses(self, tmp_path: Path) -> None:
        campaign = UmbrellaCampaign(COORDINATE, temperature_k=TEMPERATURE)
        result = campaign.run(
            minimum=0.6, maximum=1.4, root=tmp_path,
            justification=complete_justification(physical_interpretation=""),
            runner=analytic_runner(),
        )
        assert result.status is UmbrellaStatus.REQUIRES_EXPERT_DECISION
        assert "physical_interpretation" in result.report.gates[0].evidence["missing_fields"]

    def test_the_justification_is_written_alongside_the_windows(self, tmp_path: Path) -> None:
        campaign = UmbrellaCampaign(
            COORDINATE, UmbrellaDefaults(bootstrap_samples=20), temperature_k=TEMPERATURE
        )
        campaign.run(
            minimum=0.6, maximum=1.0, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(1),
        )
        assert (tmp_path / "justification.json").exists()


# ==========================================================================
# Execution
# ==========================================================================
class TestExecution:
    def test_a_well_spaced_campaign_recovers_the_analytic_pmf(self, tmp_path: Path) -> None:
        campaign = UmbrellaCampaign(
            COORDINATE, UmbrellaDefaults(spacing_nm=0.08, bootstrap_samples=40),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.4, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(2),
        )
        assert result.status is UmbrellaStatus.COMPLETED
        assert result.trustworthy is True

        mask = np.isfinite(result.pmf.pmf) & result.pmf.well_sampled
        x = result.pmf.coordinate[mask]
        recovered = result.pmf.pmf[mask] - result.pmf.pmf[mask].min()
        analytic = 0.5 * TRUE_K * (x - TRUE_MINIMUM) ** 2
        analytic -= analytic.min()
        assert np.abs(recovered - analytic).max() < 1.5

    def test_the_pmf_minimum_is_at_the_true_minimum(self, tmp_path: Path) -> None:
        campaign = UmbrellaCampaign(
            COORDINATE, UmbrellaDefaults(spacing_nm=0.08, bootstrap_samples=20),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.4, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(3),
        )
        mask = np.isfinite(result.pmf.pmf) & result.pmf.well_sampled
        location = result.pmf.coordinate[mask][np.argmin(result.pmf.pmf[mask])]
        assert location == pytest.approx(TRUE_MINIMUM, abs=0.08)

    def test_without_a_runner_inputs_are_written_but_nothing_runs(self, tmp_path: Path) -> None:
        campaign = UmbrellaCampaign(COORDINATE, temperature_k=TEMPERATURE)
        result = campaign.run(
            minimum=0.6, maximum=1.4, root=tmp_path,
            justification=complete_justification(), runner=None,
        )
        assert result.status is UmbrellaStatus.NOT_EXECUTED
        assert result.trustworthy is False
        assert (tmp_path / "umbrella_plan.json").exists()
        assert list(tmp_path.glob("window_*"))

    def test_a_failing_runner_does_not_abort_the_campaign(self, tmp_path: Path) -> None:
        def exploding(window, directory):
            raise RuntimeError("mdrun segfaulted")

        campaign = UmbrellaCampaign(COORDINATE, temperature_k=TEMPERATURE)
        result = campaign.run(
            minimum=0.6, maximum=1.0, root=tmp_path,
            justification=complete_justification(), runner=exploding,
        )
        assert result.status is UmbrellaStatus.FAILED
        assert all(not r.succeeded for r in result.runs)
        assert "segfaulted" in result.runs[0].error

    def test_partial_window_failure_is_recorded(self, tmp_path: Path) -> None:
        campaign = UmbrellaCampaign(
            COORDINATE, UmbrellaDefaults(spacing_nm=0.08, bootstrap_samples=20, max_adaptive_rounds=0),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.4, root=tmp_path,
            justification=complete_justification(),
            runner=analytic_runner(4, fail_indices={3}),
        )
        failed = [r for r in result.runs if not r.succeeded]
        assert len(failed) == 1
        assert failed[0].index == 3

    def test_equilibration_is_trimmed_from_each_window(self, tmp_path: Path) -> None:
        campaign = UmbrellaCampaign(
            COORDINATE,
            UmbrellaDefaults(spacing_nm=0.08, bootstrap_samples=20, equilibration_fraction=0.25),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.0, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(5, n_samples=4000),
        )
        successful = [r for r in result.runs if r.succeeded]
        assert successful
        assert successful[0].n_discarded == 1000
        assert successful[0].n_samples == 3000


# ==========================================================================
# Adaptive refinement
# ==========================================================================
class TestAdaptiveRefinement:
    def test_gaps_trigger_window_insertion(self, tmp_path: Path) -> None:
        """A gap between adjacent windows leaves the free energy across it undetermined."""
        campaign = UmbrellaCampaign(
            COORDINATE,
            UmbrellaDefaults(spacing_nm=0.30, bootstrap_samples=20, max_adaptive_rounds=3, min_windows=2),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.5, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(6),
        )
        assert result.rounds > 1
        assert result.inserted_centers
        assert result.overlap.sufficient is True
        assert result.status is UmbrellaStatus.COMPLETED

    def test_inserted_windows_are_at_gap_midpoints(self, tmp_path: Path) -> None:
        campaign = UmbrellaCampaign(
            COORDINATE,
            UmbrellaDefaults(spacing_nm=0.30, bootstrap_samples=20, max_adaptive_rounds=1, min_windows=2),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.5, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(7),
        )
        # Original centres 0.6, 0.9, 1.2, 1.5 -> midpoints 0.75, 1.05, 1.35.
        assert {round(c, 2) for c in result.inserted_centers} == {0.75, 1.05, 1.35}

    def test_existing_windows_are_not_rerun(self, tmp_path: Path) -> None:
        """Refinement runs only the new windows."""
        campaign = UmbrellaCampaign(
            COORDINATE,
            UmbrellaDefaults(spacing_nm=0.30, bootstrap_samples=20, max_adaptive_rounds=1, min_windows=2),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.5, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(8),
        )
        centres = [round(r.center, 3) for r in result.runs]
        assert len(centres) == len(set(centres)), "a window was run twice"

    def test_refinement_is_bounded(self, tmp_path: Path) -> None:
        """An unbounded loop on a hopeless coordinate would insert windows forever."""
        campaign = UmbrellaCampaign(
            COORDINATE,
            UmbrellaDefaults(spacing_nm=0.8, bootstrap_samples=20, max_adaptive_rounds=1, min_windows=2),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=2.2, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(9),
        )
        assert result.status is UmbrellaStatus.REFINEMENT_EXHAUSTED
        assert result.trustworthy is False
        assert result.rounds <= 2

    def test_an_exhausted_refinement_reports_the_failing_gate(self, tmp_path: Path) -> None:
        campaign = UmbrellaCampaign(
            COORDINATE,
            UmbrellaDefaults(spacing_nm=0.8, bootstrap_samples=20, max_adaptive_rounds=0, min_windows=2),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=2.2, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(10),
        )
        overlap_gate = next(g for g in result.report.gates if g.gate == "umbrella:window_overlap")
        assert overlap_gate.status is GateStatus.FAIL


# ==========================================================================
# PMF interpretation
# ==========================================================================
class TestPmfInterpretation:
    def test_a_trustworthy_pmf_reports_a_barrier(self, tmp_path: Path) -> None:
        campaign = UmbrellaCampaign(
            COORDINATE, UmbrellaDefaults(spacing_nm=0.08, bootstrap_samples=40),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.4, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(11),
        )
        barrier = result.pmf.barrier()
        assert barrier.value is not None
        assert barrier.units == "kJ/mol"

    def test_an_untrustworthy_pmf_refuses_a_barrier(self, tmp_path: Path) -> None:
        campaign = UmbrellaCampaign(
            COORDINATE,
            UmbrellaDefaults(spacing_nm=0.8, bootstrap_samples=20, max_adaptive_rounds=0, min_windows=2),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=2.2, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(12),
        )
        assert result.pmf.barrier().value is None

    def test_provenance_records_the_pmf(self, tmp_path: Path) -> None:
        graph = ProvenanceGraph()
        campaign = UmbrellaCampaign(
            COORDINATE, UmbrellaDefaults(spacing_nm=0.08, bootstrap_samples=20),
            temperature_k=TEMPERATURE, graph=graph,
        )
        campaign.run(
            minimum=0.6, maximum=1.2, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(13),
        )
        artifacts = [a for a in graph if a.kind == "pmf"]
        assert artifacts
        assert artifacts[0].units["energy"] == "kJ/mol"
        assert artifacts[0].parameters["temperature_k"] == TEMPERATURE

    def test_the_result_serialises_completely(self, tmp_path: Path) -> None:
        import json

        campaign = UmbrellaCampaign(
            COORDINATE, UmbrellaDefaults(spacing_nm=0.08, bootstrap_samples=20),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.2, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(14),
        )
        payload = result.as_dict()
        assert json.dumps(payload)
        assert payload["justification"]["complete"] is True
        assert payload["pmf"]["trustworthy"] is True


# ==========================================================================
# COLVAR parsing
# ==========================================================================
class TestColvarReading:
    def test_reads_the_cv_column(self, tmp_path: Path) -> None:
        path = tmp_path / "COLVAR"
        path.write_text("#! FIELDS time d bias\n0.0 1.23 0.5\n0.5 1.25 0.4\n")
        assert read_colvar(path).tolist() == pytest.approx([1.23, 1.25])

    def test_comments_and_blanks_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "COLVAR"
        path.write_text("#! FIELDS time d\n# a comment\n\n0.0 1.0\n0.5 1.1\n")
        assert read_colvar(path).size == 2

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InsufficientDataError):
            read_colvar(tmp_path / "absent")

    def test_a_file_with_no_data_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "COLVAR"
        path.write_text("#! FIELDS time d\n# nothing else\n")
        with pytest.raises(InsufficientDataError):
            read_colvar(path)


class TestStartingStructures:
    def test_missing_starting_structures_warn_but_do_not_block(self, tmp_path: Path) -> None:
        """Absent starting structures cost equilibration time; they do not invalidate a PMF."""
        campaign = UmbrellaCampaign(
            COORDINATE, UmbrellaDefaults(spacing_nm=0.08, bootstrap_samples=20),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.0, root=tmp_path,
            justification=complete_justification(), runner=analytic_runner(20),
        )
        gate = next(g for g in result.report.gates if g.gate == "umbrella:starting_structures")
        assert gate.status is GateStatus.WARN
        assert result.trustworthy is True, "a warning must not block an otherwise sound PMF"

    def test_supplied_structures_are_copied_into_each_window(self, tmp_path: Path) -> None:
        source = tmp_path / "structures"
        source.mkdir()
        available = {}
        for centre in (0.60, 0.68, 0.76, 0.84, 0.92, 1.00):
            path = source / f"eq_{centre:.2f}.gro"
            path.write_text(f"equilibrated at {centre}\n1\n    1POL C1 1 0 0 0\n2 2 2\n")
            available[centre] = str(path)

        campaign = UmbrellaCampaign(
            COORDINATE, UmbrellaDefaults(spacing_nm=0.08, bootstrap_samples=20),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.0, root=tmp_path / "run",
            justification=complete_justification(), runner=analytic_runner(21),
            starting_structures=available,
        )
        gate = next(g for g in result.report.gates if g.gate == "umbrella:starting_structures")
        assert gate.status is GateStatus.PASS
        assert all(r.starting_structure for r in result.runs)
        assert (Path(result.runs[0].directory) / "start.gro").exists()

    def test_distant_starting_structures_are_warned_about(self, tmp_path: Path) -> None:
        """Starting far from the restraint centre wastes equilibration and can trap a basin."""
        source = tmp_path / "structures"
        source.mkdir()
        far = source / "eq_far.gro"
        far.write_text("far from every window\n1\n    1POL C1 1 0 0 0\n2 2 2\n")

        campaign = UmbrellaCampaign(
            COORDINATE, UmbrellaDefaults(spacing_nm=0.08, bootstrap_samples=20),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.0, root=tmp_path / "run",
            justification=complete_justification(), runner=analytic_runner(22),
            starting_structures={5.0: str(far)},
        )
        gate = next(g for g in result.report.gates if g.gate == "umbrella:starting_structures")
        assert gate.status is GateStatus.WARN

    def test_a_missing_structure_file_fails_that_window(self, tmp_path: Path) -> None:
        campaign = UmbrellaCampaign(
            COORDINATE, UmbrellaDefaults(spacing_nm=0.08, bootstrap_samples=20, max_adaptive_rounds=0),
            temperature_k=TEMPERATURE,
        )
        result = campaign.run(
            minimum=0.6, maximum=1.0, root=tmp_path / "run",
            justification=complete_justification(), runner=analytic_runner(23),
            starting_structures={0.60: str(tmp_path / "absent.gro")},
        )
        failed = [r for r in result.runs if r.error and "not found" in r.error]
        assert failed
