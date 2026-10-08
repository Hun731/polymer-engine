"""Convergence gates, replica agreement, and umbrella/PMF analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from polymer_engine.analysis.convergence import (
    analyse_series,
    build_convergence_report,
    combine_replicas,
    convergence_gates,
    replica_agreement_gate,
)
from polymer_engine.analysis.free_energy import (
    WindowSamples,
    build_pmf,
    diagnose_overlap,
    overlap_matrix,
    pmf_gates,
    trim_equilibration,
    wham,
)
from polymer_engine.core.config import AnalysisDefaults, UmbrellaDefaults
from polymer_engine.core.errors import InsufficientDataError, ParameterValidationError
from polymer_engine.core.models import Determination, GateStatus, Measurement
from polymer_engine.core.units import kT
from polymer_engine.simulation.umbrella import (
    ReactionCoordinate,
    insert_windows,
    plan_windows,
    plumed_input,
    recommend_force_constant,
    restraint_sigma,
    write_windows,
)

TEMPERATURE = 300.0


def gate(gates, name: str):
    return next(g for g in gates if g.gate == name)


# ==========================================================================
# Convergence
# ==========================================================================
class TestSeriesConvergence:
    def test_equilibrated_series_passes(self) -> None:
        rng = np.random.default_rng(1)
        values = 1000.0 + rng.normal(0, 2.0, 5000)
        analysis = analyse_series(values, name="density", units="kg/m^3")
        gates = convergence_gates(analysis)
        assert all(g.status is GateStatus.PASS for g in gates), [g.message for g in gates]

    def test_drifting_series_fails_the_drift_gate(self) -> None:
        t = np.arange(5000, dtype=float)
        values = 1000.0 + 0.02 * t + np.random.default_rng(2).normal(0, 1.0, t.size)
        analysis = analyse_series(values, name="density", units="kg/m^3", times_ps=t * 10)
        assert gate(convergence_gates(analysis), "density:drift").status is GateStatus.FAIL

    def test_drift_direction_is_reported(self) -> None:
        t = np.arange(3000, dtype=float) * 10.0
        values = 1000.0 - 0.001 * t
        analysis = analyse_series(values, name="density", units="kg/m^3", times_ps=t)
        assert analysis.drift_per_ns is not None
        assert analysis.drift_per_ns < 0

    def test_highly_correlated_series_fails_the_sample_count_gate(self) -> None:
        """A long trajectory that has barely decorrelated has few real samples."""
        rng = np.random.default_rng(3)
        x = np.zeros(4000)
        for i in range(1, x.size):
            x[i] = 0.999 * x[i - 1] + rng.normal(0, 0.05)
        analysis = analyse_series(x + 1000.0, name="density", units="kg/m^3")
        assert gate(convergence_gates(analysis), "density:effective_samples").status is GateStatus.FAIL

    def test_short_series_gives_inconclusive_not_pass(self) -> None:
        analysis = analyse_series([1.0, 1.1, 0.9], name="x")
        statuses = {g.status for g in convergence_gates(analysis)}
        assert GateStatus.PASS not in statuses or GateStatus.INCONCLUSIVE in statuses

    def test_equilibration_transient_is_removed(self) -> None:
        t = np.arange(6000, dtype=float)
        values = 1000.0 - 100.0 * np.exp(-t / 500.0) + np.random.default_rng(4).normal(0, 1.0, t.size)
        analysis = analyse_series(values, name="density", units="kg/m^3")
        assert analysis.equilibration.n_discarded > 200
        assert analysis.production.value == pytest.approx(1000.0, abs=2.0)

    def test_fixed_fraction_mode_is_honoured(self) -> None:
        values = np.random.default_rng(5).normal(1000, 1, 1000)
        analysis = analyse_series(
            values,
            name="density",
            units="kg/m^3",
            defaults=AnalysisDefaults(equilibration_detection="fraction", discard_fraction=0.3),
        )
        assert analysis.equilibration.n_discarded == 300


# ==========================================================================
# Replica agreement
# ==========================================================================
def replica(value: float, uncertainty: float | None = 1.0) -> Measurement:
    return Measurement(name="density", value=value, uncertainty=uncertainty, units="kg/m^3")


class TestReplicaAgreement:
    def test_agreeing_replicas_pass(self) -> None:
        agreement = combine_replicas([replica(1000.2), replica(999.4), replica(1000.9)])
        assert replica_agreement_gate(agreement).status is GateStatus.PASS

    def test_disagreeing_replicas_fail(self) -> None:
        agreement = combine_replicas([replica(1000.0), replica(1040.0), replica(970.0)])
        result = replica_agreement_gate(agreement)
        assert result.status is GateStatus.FAIL
        assert result.value > 4.0

    def test_a_single_replica_is_inconclusive_not_pass(self) -> None:
        """One run cannot demonstrate reproducibility, however precise it is."""
        agreement = combine_replicas([replica(1000.0, uncertainty=0.001)])
        assert agreement.determination is Determination.INSUFFICIENT_DATA
        assert replica_agreement_gate(agreement).status is GateStatus.INCONCLUSIVE

    def test_two_replicas_when_three_are_required_is_inconclusive(self) -> None:
        agreement = combine_replicas([replica(1000.0), replica(1000.1)])
        assert replica_agreement_gate(agreement, required_replicas=3).status is GateStatus.INCONCLUSIVE

    def test_no_replicas_fails(self) -> None:
        agreement = combine_replicas([])
        assert agreement.combined.determination is Determination.INSUFFICIENT_DATA
        assert replica_agreement_gate(agreement).status is GateStatus.FAIL

    def test_unknown_measurements_are_excluded(self) -> None:
        agreement = combine_replicas(
            [replica(1000.0), Measurement.unknown("density", units="kg/m^3"), replica(1000.5)]
        )
        assert agreement.n_replicas == 2

    def test_combined_uncertainty_uses_replica_spread_not_frame_count(self) -> None:
        """Using the within-replica error here would be pseudoreplication."""
        agreement = combine_replicas([replica(1000.0, 0.001), replica(1010.0, 0.001), replica(990.0, 0.001)])
        assert agreement.combined.uncertainty > 1.0
        assert agreement.combined.effective_samples == 3.0

    def test_missing_per_replica_uncertainties_give_inconclusive(self) -> None:
        agreement = combine_replicas([replica(1000.0, None), replica(1001.0, None), replica(999.0, None)])
        assert replica_agreement_gate(agreement).status is GateStatus.INCONCLUSIVE

    def test_report_aggregation_is_worst_case(self) -> None:
        good = analyse_series(np.random.default_rng(6).normal(1000, 1, 4000), name="density", units="kg/m^3")
        report = build_convergence_report(
            {"density": good}, {"density": combine_replicas([replica(1000.0)])}
        )
        assert report.status is GateStatus.INCONCLUSIVE
        assert report.promotable is False


# ==========================================================================
# Umbrella planning
# ==========================================================================
COORD = ReactionCoordinate(
    name="d",
    kind="com_distance",
    units="nm",
    justification="Interchain centre-of-mass separation probes cohesive interactions.",
    group_a="1-100",
    group_b="101-200",
)


class TestWindowPlanning:
    def test_sigma_matches_the_analytic_restraint_width(self) -> None:
        k = 1000.0
        assert restraint_sigma(k, TEMPERATURE) == pytest.approx(math.sqrt(kT(TEMPERATURE) / k))

    def test_default_spacing_is_expected_to_overlap(self) -> None:
        plan = plan_windows(COORD, minimum=0.4, maximum=1.6, temperature_k=TEMPERATURE)
        assert plan.expected_to_overlap
        assert plan.warnings == []

    def test_too_wide_a_spacing_is_warned_about(self) -> None:
        plan = plan_windows(
            COORD, minimum=0.4, maximum=1.6, temperature_k=TEMPERATURE,
            defaults=UmbrellaDefaults(spacing_nm=0.5),
        )
        assert plan.expected_to_overlap is False
        assert any("not expected to overlap" in w for w in plan.warnings)

    def test_strict_mode_refuses_a_non_overlapping_plan(self) -> None:
        with pytest.raises(ParameterValidationError, match="not expected to overlap"):
            plan_windows(
                COORD, minimum=0.4, maximum=1.6, temperature_k=TEMPERATURE,
                defaults=UmbrellaDefaults(spacing_nm=0.5), strict=True,
            )

    def test_inverted_range_is_rejected(self) -> None:
        with pytest.raises(ParameterValidationError, match="increasing"):
            plan_windows(COORD, minimum=1.6, maximum=0.4, temperature_k=TEMPERATURE)

    def test_endpoint_is_always_covered(self) -> None:
        plan = plan_windows(
            COORD, minimum=0.0, maximum=1.0, temperature_k=TEMPERATURE,
            defaults=UmbrellaDefaults(spacing_nm=0.3),
        )
        assert plan.centers[-1] == pytest.approx(1.0)

    def test_windows_are_monotonic_and_unique(self) -> None:
        centers = plan_windows(COORD, minimum=0.4, maximum=1.6, temperature_k=TEMPERATURE).centers
        assert centers == sorted(centers)
        assert len(set(centers)) == len(centers)

    def test_force_constant_recommendation_round_trips(self) -> None:
        k = recommend_force_constant(0.08, TEMPERATURE, sigma_factor=2.0)
        assert 2.0 * restraint_sigma(k, TEMPERATURE) == pytest.approx(0.08)

    def test_adaptive_insertion_fills_gaps(self) -> None:
        plan = plan_windows(
            COORD, minimum=0.4, maximum=1.2, temperature_k=TEMPERATURE,
            defaults=UmbrellaDefaults(spacing_nm=0.4),
        )
        refined = insert_windows(plan, [(0.4, 0.8)])
        assert 0.6 in refined.centers
        assert refined.n_windows == plan.n_windows + 1
        assert refined.centers == sorted(refined.centers)

    def test_plumed_input_contains_the_restraint(self) -> None:
        plan = plan_windows(COORD, minimum=0.4, maximum=0.8, temperature_k=TEMPERATURE)
        text = plumed_input(COORD, plan.windows[1])
        assert "RESTRAINT" in text
        assert f"AT={plan.windows[1].center:.6f}" in text
        assert "COM ATOMS=1-100" in text
        assert "UNITS LENGTH=nm ENERGY=kj/mol" in text

    def test_torsion_needs_four_atoms(self) -> None:
        bad = ReactionCoordinate(name="phi", kind="torsion", units="deg", justification="x", atoms="1,2,3")
        with pytest.raises(ParameterValidationError, match="four atoms"):
            bad.plumed_definition()

    def test_writing_windows_creates_one_directory_each(self, tmp_path) -> None:
        plan = plan_windows(COORD, minimum=0.4, maximum=0.8, temperature_k=TEMPERATURE)
        written = write_windows(plan, tmp_path / "umbrella")
        assert len(written) == plan.n_windows
        assert (tmp_path / "umbrella" / "umbrella_plan.json").exists()
        assert all(p.exists() for p in written)


# ==========================================================================
# Overlap and PMF
# ==========================================================================
def harmonic_windows(
    centers, *, k: float = 1000.0, n: int = 6000, true_k: float = 200.0, x0: float = 1.0, seed: int = 0
) -> list[WindowSamples]:
    """Windows sampled exactly from the analytic biased distribution.

    Underlying PMF is ``0.5 * true_k * (x - x0)^2``, so WHAM has a known right answer.
    """
    beta = 1.0 / kT(TEMPERATURE)
    rng = np.random.default_rng(seed)
    out = []
    for i, c in enumerate(centers):
        mean = (true_k * x0 + k * c) / (true_k + k)
        sd = math.sqrt(1.0 / (beta * (true_k + k)))
        out.append(WindowSamples(i, float(c), k, rng.normal(mean, sd, n), "nm"))
    return out


class TestOverlap:
    def test_well_spaced_windows_overlap(self) -> None:
        diagnostic = diagnose_overlap(harmonic_windows(np.arange(0.7, 1.35, 0.08), seed=1))
        assert diagnostic.sufficient
        assert diagnostic.gaps == []
        assert diagnostic.min_pair_overlap > 0.1

    def test_far_apart_windows_do_not_overlap(self) -> None:
        diagnostic = diagnose_overlap(harmonic_windows([0.6, 1.0, 1.4], seed=2))
        assert diagnostic.sufficient is False
        assert len(diagnostic.gaps) == 2

    def test_a_missing_middle_window_creates_one_gap(self) -> None:
        centers = [0.70, 0.78, 0.86, 1.10, 1.18]
        diagnostic = diagnose_overlap(harmonic_windows(centers, seed=3))
        assert len(diagnostic.gaps) == 1
        assert diagnostic.gaps[0] == pytest.approx((0.86, 1.10))

    def test_empty_window_is_reported(self) -> None:
        windows = harmonic_windows([0.8, 0.88, 0.96], seed=4)
        windows[1] = WindowSamples(1, 0.88, 1000.0, np.array([]), "nm")
        diagnostic = diagnose_overlap(windows)
        assert diagnostic.empty_windows == [1]
        assert diagnostic.sufficient is False

    def test_duplicate_windows_overlap_completely(self) -> None:
        windows = harmonic_windows([0.9, 0.9], seed=5)
        assert diagnose_overlap(windows).min_pair_overlap > 0.9

    def test_overlap_matrix_diagonal_is_one(self) -> None:
        matrix = overlap_matrix(harmonic_windows([0.8, 0.88, 0.96], seed=6))
        assert np.allclose(np.diag(matrix), 1.0, atol=1e-9)

    def test_overlap_matrix_is_symmetric(self) -> None:
        matrix = overlap_matrix(harmonic_windows([0.8, 0.88, 0.96], seed=7))
        assert np.allclose(matrix, matrix.T)


class TestPmf:
    def test_wham_recovers_a_known_harmonic_pmf(self) -> None:
        windows = harmonic_windows(np.arange(0.6, 1.45, 0.08), n=20_000, seed=11)
        result = build_pmf(windows, temperature_k=TEMPERATURE, bootstrap_samples=40, seed=1)
        assert result.trustworthy, result.problems
        mask = np.isfinite(result.pmf) & result.well_sampled
        x = result.coordinate[mask]
        analytic = 0.5 * 200.0 * (x - 1.0) ** 2
        analytic -= analytic.min()
        computed = result.pmf[mask] - result.pmf[mask].min()
        assert np.abs(computed - analytic).max() < 1.5

    def test_pmf_sign_convention_puts_the_minimum_at_the_stable_state(self) -> None:
        windows = harmonic_windows(np.arange(0.6, 1.45, 0.08), n=20_000, seed=12)
        result = build_pmf(windows, temperature_k=TEMPERATURE, bootstrap_samples=0)
        mask = np.isfinite(result.pmf) & result.well_sampled
        assert result.coordinate[mask][np.argmin(result.pmf[mask])] == pytest.approx(1.0, abs=0.1)

    def test_poorly_overlapped_windows_are_not_trustworthy(self) -> None:
        result = build_pmf(harmonic_windows([0.6, 1.0, 1.4], n=8000, seed=13), temperature_k=TEMPERATURE, bootstrap_samples=0)
        assert result.trustworthy is False
        assert any("overlap" in p for p in result.problems)
        assert pmf_gates(result).status is GateStatus.FAIL

    def test_untrustworthy_pmf_refuses_to_report_a_barrier(self) -> None:
        """The engine must be able to say 'do not trust this PMF yet'."""
        result = build_pmf(harmonic_windows([0.6, 1.0, 1.4], n=8000, seed=14), temperature_k=TEMPERATURE, bootstrap_samples=0)
        barrier = result.barrier()
        assert barrier.value is None
        assert barrier.determination is not Determination.KNOWN

    def test_short_windows_fail_the_convergence_check(self) -> None:
        windows = harmonic_windows(np.arange(0.7, 1.35, 0.08), n=60, seed=15)
        result = build_pmf(windows, temperature_k=TEMPERATURE, bootstrap_samples=0)
        assert result.trustworthy is False

    def test_a_missing_window_is_caught(self) -> None:
        result = build_pmf(
            harmonic_windows([0.70, 0.78, 0.86, 1.10, 1.18], n=8000, seed=16),
            temperature_k=TEMPERATURE, bootstrap_samples=0,
        )
        assert result.trustworthy is False
        assert result.overlap.gaps

    def test_duplicated_window_does_not_break_wham(self) -> None:
        centers = [0.8, 0.88, 0.88, 0.96, 1.04]
        result = build_pmf(harmonic_windows(centers, n=8000, seed=17), temperature_k=TEMPERATURE, bootstrap_samples=0)
        assert np.isfinite(result.pmf[result.well_sampled]).any()

    def test_non_monotonic_window_order_is_handled(self) -> None:
        windows = harmonic_windows([1.04, 0.80, 0.96, 0.88], n=8000, seed=18)
        diagnostic = diagnose_overlap(windows)
        assert diagnostic.centers == sorted(diagnostic.centers)

    def test_bootstrap_produces_an_uncertainty_band(self) -> None:
        windows = harmonic_windows(np.arange(0.7, 1.35, 0.08), n=8000, seed=19)
        result = build_pmf(windows, temperature_k=TEMPERATURE, bootstrap_samples=40, seed=2)
        assert result.uncertainty is not None
        assert np.nanmax(result.uncertainty[result.well_sampled]) > 0

    def test_fewer_than_two_windows_is_rejected(self) -> None:
        with pytest.raises(InsufficientDataError):
            build_pmf(harmonic_windows([1.0], n=100, seed=20), temperature_k=TEMPERATURE)

    def test_wham_is_deterministic(self) -> None:
        windows = harmonic_windows(np.arange(0.8, 1.25, 0.08), n=5000, seed=21)
        a = wham(windows, temperature_k=TEMPERATURE)
        b = wham(windows, temperature_k=TEMPERATURE)
        assert np.allclose(a.pmf, b.pmf, equal_nan=True)

    def test_trim_equilibration_drops_the_leading_fraction(self) -> None:
        values, discarded = trim_equilibration(np.arange(100.0), 0.2)
        assert discarded == 20
        assert values[0] == 20.0

    def test_trim_equilibration_rejects_a_bad_fraction(self) -> None:
        with pytest.raises(InsufficientDataError):
            trim_equilibration(np.arange(10.0), 1.0)


class TestWhamConvergenceBudget:
    """The iteration cap must be generous for well-posed problems and fail fast otherwise.

    Measured on this implementation: ~250 iterations for well-overlapped windows and
    ~4,500 for sparse-but-overlapping ones. The cap exists to stop disjoint windows,
    which have no solution to converge to, from running indefinitely.
    """

    def test_well_overlapped_windows_converge_quickly(self) -> None:
        result = wham(harmonic_windows(np.arange(0.6, 1.45, 0.08), n=8000, seed=30), temperature_k=TEMPERATURE)
        assert result.converged is True
        assert result.iterations < 2000

    def test_sparse_but_overlapping_windows_still_converge(self) -> None:
        from polymer_engine.analysis.free_energy import DEFAULT_WHAM_MAX_ITERATIONS

        result = wham(harmonic_windows(np.arange(0.6, 1.6, 0.3), n=8000, seed=31), temperature_k=TEMPERATURE)
        assert result.converged is True
        assert result.iterations < DEFAULT_WHAM_MAX_ITERATIONS

    def test_wham_converging_is_not_evidence_the_pmf_is_right(self) -> None:
        """Disjoint windows can converge trivially, because the bins decouple.

        Each bin is then occupied by exactly one window, the self-consistent equations
        have nothing to reconcile, and WHAM reports success in a handful of iterations.
        The resulting PMF is meaningless. Only the overlap diagnostic catches this, which
        is why convergence alone never promotes a PMF.
        """
        windows = harmonic_windows(np.arange(0.6, 2.3, 0.8), n=4000, seed=32)
        result = wham(windows, temperature_k=TEMPERATURE)
        assert result.converged is True, "this fixture is chosen to converge trivially"

        pmf = build_pmf(windows, temperature_k=TEMPERATURE, bootstrap_samples=0)
        assert pmf.trustworthy is False
        assert any("overlap" in problem for problem in pmf.problems)
        assert pmf_gates(pmf).status is GateStatus.FAIL

    def test_the_vectorised_solver_places_the_minimum_correctly(self) -> None:
        """Guards the vectorised inner loop against a regression in the maths."""
        windows = harmonic_windows(np.arange(0.7, 1.35, 0.08), n=6000, seed=33)
        result = wham(windows, temperature_k=TEMPERATURE)
        assert result.free_energies[0] == pytest.approx(0.0, abs=1e-12)

        # The biased free energy is lowest for the window whose restraint sits at the
        # true PMF minimum (1.0 nm), because that window needs the least biasing work.
        centres = np.array([w.center for w in windows])
        assert centres[int(np.argmin(result.free_energies))] == pytest.approx(1.0, abs=0.1)
