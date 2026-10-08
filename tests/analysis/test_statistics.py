"""Statistics, verified against analytic results and stressed on edge cases."""

from __future__ import annotations

import math

import numpy as np
import pytest

from polymer_engine.analysis.statistics import (
    autocorrelation,
    block_average,
    blocking_curve,
    bootstrap_ci,
    describe,
    detect_equilibration,
    drop_non_finite,
    effective_sample_size,
    integrated_autocorrelation_time,
    partial_correlation,
    pearson,
    permutation_test,
    spearman,
    statistical_inefficiency,
    welch_t_test,
)
from polymer_engine.core.errors import InsufficientDataError
from polymer_engine.core.models import Determination


def ar1(n: int, phi: float, seed: int = 0) -> np.ndarray:
    """AR(1) series with known tau_int = phi / (1 - phi) and unit variance."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    scale = math.sqrt(1.0 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, scale)
    return x


# ==========================================================================
# Autocorrelation -- checked against AR(1) theory
# ==========================================================================
class TestAutocorrelation:
    def test_iid_series_has_no_correlation(self) -> None:
        x = np.random.default_rng(1).normal(size=50_000)
        assert statistical_inefficiency(x) == pytest.approx(1.0, abs=0.15)

    @pytest.mark.parametrize("phi,expected_tau", [(0.5, 1.0), (0.8, 4.0), (0.9, 9.0)])
    def test_ar1_autocorrelation_time_matches_theory(self, phi: float, expected_tau: float) -> None:
        tau = integrated_autocorrelation_time(ar1(200_000, phi, seed=2))
        assert tau == pytest.approx(expected_tau, rel=0.15)

    def test_effective_sample_size_is_far_below_frame_count(self) -> None:
        x = ar1(100_000, 0.9, seed=3)
        ess = effective_sample_size(x)
        assert ess < x.size / 10
        assert ess == pytest.approx(x.size / 19.0, rel=0.2)

    def test_correlated_standard_error_exceeds_the_naive_one(self) -> None:
        """This is the pseudoreplication bug: ignoring correlation understates the error."""
        x = ar1(100_000, 0.9, seed=4)
        correlated = describe(x, account_for_correlation=True)
        naive = describe(x, account_for_correlation=False)
        assert correlated.uncertainty > 3 * naive.uncertainty

    def test_constant_series_has_defined_autocorrelation(self) -> None:
        rho = autocorrelation(np.full(100, 5.0))
        assert rho[0] == 1.0
        assert np.all(rho[1:] == 0.0)

    def test_statistical_inefficiency_never_below_one(self) -> None:
        assert statistical_inefficiency(np.random.default_rng(5).normal(size=100)) >= 1.0


# ==========================================================================
# describe() edge cases
# ==========================================================================
class TestDescribe:
    def test_empty_series_is_insufficient_data(self) -> None:
        m = describe([], name="x")
        assert m.determination is Determination.INSUFFICIENT_DATA
        assert m.value is None

    def test_single_sample_reports_no_uncertainty_rather_than_zero(self) -> None:
        m = describe([42.0], name="x")
        assert m.value == 42.0
        assert m.uncertainty is None, "a single sample has no estimable uncertainty"

    def test_constant_series_has_zero_uncertainty(self) -> None:
        m = describe(np.full(100, 7.0), name="x")
        assert m.value == 7.0
        assert m.uncertainty == pytest.approx(0.0)

    def test_nan_input_is_rejected_loudly(self) -> None:
        with pytest.raises(InsufficientDataError, match="non-finite"):
            describe([1.0, float("nan"), 3.0])

    def test_inf_input_is_rejected_loudly(self) -> None:
        with pytest.raises(InsufficientDataError):
            describe([1.0, float("inf"), 3.0])

    def test_dropping_non_finite_is_explicit(self) -> None:
        cleaned, dropped = drop_non_finite([1.0, float("nan"), 3.0, float("inf")])
        assert cleaned.tolist() == [1.0, 3.0]
        assert dropped == 2

    def test_units_are_carried_through(self) -> None:
        assert describe([1.0, 2.0, 3.0], name="density", units="kg/m^3").units == "kg/m^3"

    def test_mean_is_correct(self) -> None:
        assert describe([1.0, 2.0, 3.0, 4.0]).value == pytest.approx(2.5)


# ==========================================================================
# Blocking and bootstrap
# ==========================================================================
class TestBlocking:
    def test_block_average_matches_the_overall_mean(self) -> None:
        x = np.arange(100, dtype=float)
        blocks, measurement = block_average(x, n_blocks=5)
        assert blocks.size == 5
        assert measurement.value == pytest.approx(x.mean())

    def test_too_few_samples_is_rejected(self) -> None:
        with pytest.raises(InsufficientDataError):
            block_average([1.0, 2.0], n_blocks=5)

    def test_one_block_is_rejected(self) -> None:
        with pytest.raises(InsufficientDataError):
            block_average(np.arange(100.0), n_blocks=1)

    def test_blocking_curve_grows_toward_the_true_error(self) -> None:
        """For correlated data the naive error is too small; blocking exposes that."""
        curve = blocking_curve(ar1(65_536, 0.9, seed=6))
        assert curve[-1][1] > 2 * curve[0][1]


class TestBootstrap:
    def test_interval_brackets_the_mean_for_iid_data(self) -> None:
        x = np.random.default_rng(7).normal(loc=5.0, scale=1.0, size=2000)
        estimate, lower, upper = bootstrap_ci(x, n_resamples=400, seed=1)
        assert lower < estimate < upper
        assert lower < 5.0 < upper

    def test_block_bootstrap_widens_the_interval_for_correlated_data(self) -> None:
        x = ar1(20_000, 0.9, seed=8)
        _, lo_iid, hi_iid = bootstrap_ci(x, n_resamples=300, block_size=1, seed=1)
        _, lo_blk, hi_blk = bootstrap_ci(x, n_resamples=300, seed=1)
        assert (hi_blk - lo_blk) > 2 * (hi_iid - lo_iid)

    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        x = np.random.default_rng(9).normal(size=500)
        assert bootstrap_ci(x, n_resamples=100, seed=42) == bootstrap_ci(x, n_resamples=100, seed=42)

    def test_one_sample_is_rejected(self) -> None:
        with pytest.raises(InsufficientDataError):
            bootstrap_ci([1.0])

    def test_invalid_confidence_level_is_rejected(self) -> None:
        with pytest.raises(InsufficientDataError):
            bootstrap_ci([1.0, 2.0, 3.0], confidence_level=1.5)


# ==========================================================================
# Equilibration detection
# ==========================================================================
class TestEquilibration:
    def test_relaxing_series_has_its_transient_discarded(self) -> None:
        t = np.arange(4000, dtype=float)
        signal = 1000.0 - 80.0 * np.exp(-t / 400.0)
        noisy = signal + np.random.default_rng(10).normal(0, 1.0, t.size)
        result = detect_equilibration(noisy)
        assert result.start_index > 200, "the exponential transient must be excluded"
        assert result.n_retained > 1000

    def test_already_equilibrated_series_keeps_almost_everything(self) -> None:
        x = np.random.default_rng(11).normal(loc=1.0, scale=0.01, size=3000)
        assert detect_equilibration(x).discarded_fraction < 0.2

    def test_short_series_is_reported_as_such(self) -> None:
        result = detect_equilibration([1.0, 2.0, 3.0])
        assert result.start_index == 0
        assert "too short" in result.method


# ==========================================================================
# Correlation
# ==========================================================================
class TestCorrelation:
    def test_perfect_linear_relationship(self) -> None:
        x = np.arange(20, dtype=float)
        assert pearson(x, 2 * x + 1).statistic == pytest.approx(1.0)
        assert pearson(x, -3 * x).statistic == pytest.approx(-1.0)

    def test_spearman_captures_monotone_nonlinearity(self) -> None:
        x = np.arange(1, 21, dtype=float)
        y = x**3
        assert spearman(x, y).statistic == pytest.approx(1.0)
        assert pearson(x, y).statistic < 1.0

    def test_spearman_handles_ties(self) -> None:
        result = spearman([1, 1, 2, 3, 4, 5], [1, 1, 2, 3, 4, 5])
        assert result.statistic == pytest.approx(1.0)

    def test_constant_input_is_insufficient_data_not_nan(self) -> None:
        result = pearson([1.0] * 10, np.arange(10.0))
        assert result.statistic is None
        assert result.determination is Determination.INSUFFICIENT_DATA

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_too_few_pairs(self, n: int) -> None:
        result = pearson(list(range(n)), list(range(n)))
        assert result.determination is Determination.INSUFFICIENT_DATA

    def test_unequal_lengths_are_rejected(self) -> None:
        with pytest.raises(InsufficientDataError, match="same length"):
            pearson([1, 2, 3], [1, 2])

    def test_missing_values_are_pairwise_dropped(self) -> None:
        x = [1.0, 2.0, float("nan"), 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = pearson(x, y)
        assert result.n == 4
        assert result.statistic == pytest.approx(1.0)

    def test_partial_correlation_removes_a_confounder(self) -> None:
        """Two variables driven by a common cause correlate until it is controlled for."""
        rng = np.random.default_rng(12)
        z = rng.normal(size=400)
        x = z + rng.normal(scale=0.1, size=400)
        y = z + rng.normal(scale=0.1, size=400)
        assert pearson(x, y).statistic > 0.9
        assert abs(partial_correlation(x, y, z).statistic) < 0.3

    def test_partial_correlation_needs_enough_observations(self) -> None:
        result = partial_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [[1.0], [2.0], [3.0]])
        assert result.determination is Determination.INSUFFICIENT_DATA

    def test_permutation_test_finds_a_real_association(self) -> None:
        rng = np.random.default_rng(13)
        x = rng.normal(size=60)
        y = x + rng.normal(scale=0.3, size=60)
        assert permutation_test(x, y, n_permutations=500, seed=1).p_value < 0.01

    def test_permutation_test_is_not_fooled_by_noise(self) -> None:
        rng = np.random.default_rng(14)
        result = permutation_test(rng.normal(size=60), rng.normal(size=60), n_permutations=500, seed=1)
        assert result.p_value > 0.05

    def test_permutation_p_value_is_never_exactly_zero(self) -> None:
        x = np.arange(50, dtype=float)
        assert permutation_test(x, 2 * x, n_permutations=200, seed=1).p_value > 0.0


class TestWelch:
    def test_detects_a_shifted_mean(self) -> None:
        rng = np.random.default_rng(15)
        result = welch_t_test(rng.normal(0, 1, 200), rng.normal(2, 1, 150))
        assert abs(result.statistic) > 5

    def test_unequal_sample_sizes_are_fine(self) -> None:
        rng = np.random.default_rng(16)
        assert welch_t_test(rng.normal(size=500), rng.normal(size=12)).n == 512

    def test_too_few_values(self) -> None:
        assert welch_t_test([1.0], [2.0, 3.0]).determination is Determination.INSUFFICIENT_DATA

    def test_two_constant_samples(self) -> None:
        result = welch_t_test([1.0] * 5, [1.0] * 5)
        assert result.determination is Determination.INSUFFICIENT_DATA


class TestOscillatingAutocorrelation:
    """Regression for BUG-002: an aliased series reported N independent samples.

    A Parrinello-Rahman barostat at ``tau_p = 5 ps`` sampled every 2 ps aliases the
    volume oscillation, so the density autocorrelation alternates: near zero at odd
    lags, large at even lags. Truncating at the first individual non-positive lag then
    gives ``tau = 0`` and ``g = 1`` for a series that is strongly correlated.
    """

    @staticmethod
    def aliased_series(n: int = 8000, seed: int = 11) -> np.ndarray:
        """Slowly-correlated signal plus an alternating component, as MD produces."""
        rng = np.random.default_rng(seed)
        slow, values = 0.0, []
        for index in range(n):
            slow = 0.97 * slow + rng.normal(0.0, 1.0)
            values.append(slow + 4.0 * (-1.0) ** index)
        return np.asarray(values, dtype=float)

    def test_the_autocorrelation_really_does_oscillate(self) -> None:
        """Guards the premise: if this stops holding, the test below proves nothing."""
        rho = autocorrelation(self.aliased_series())
        assert rho[1] < rho[2], "lag 1 should be suppressed relative to lag 2"
        assert rho[2] > 0.3, "even lags should retain substantial correlation"

    def test_an_aliased_series_is_not_reported_as_independent(self) -> None:
        series = self.aliased_series()
        g = statistical_inefficiency(series)
        assert g > 5.0, f"aliased series reported g={g:.3f}; it is strongly correlated"

    def test_effective_samples_are_far_below_the_frame_count(self) -> None:
        series = self.aliased_series()
        effective = effective_sample_size(series)
        assert effective < series.size / 5, (
            f"{effective:.0f} effective samples from {series.size} frames is too generous "
            "for an aliased series"
        )
        assert effective >= 1.0

    def test_white_noise_is_still_uncorrelated(self) -> None:
        """The fix must not inflate correlation times across the board."""
        rng = np.random.default_rng(5)
        g = statistical_inefficiency(rng.normal(size=40000))
        assert g == pytest.approx(1.0, abs=0.2)

    def test_ar1_theory_is_unchanged(self) -> None:
        """phi=0.9 => tau=9, g=19. The pairing must not disturb the smooth case."""
        rng = np.random.default_rng(0)
        value, series = 0.0, []
        for _ in range(200000):
            value = 0.9 * value + rng.normal()
            series.append(value)
        assert integrated_autocorrelation_time(series) == pytest.approx(9.0, rel=0.15)

    def test_a_perfectly_antithetic_series_is_cheap_but_not_free(self) -> None:
        """An alternating series is de-aliased by pair averaging, at a factor-two cost.

        This test used to assert g == 1 exactly, which encoded the behaviour that hid a
        real defect: barostat ringing aliased to near-alternation also produced g = 1,
        and 50,001 correlated frames were counted as 50,001 independent samples. Paying
        one halving to remove a deterministic oscillation is the honest price.
        """
        x = np.array([1.0, -1.0] * 5000) + np.random.default_rng(3).normal(0, 0.01, 10000)
        g = statistical_inefficiency(x)
        assert 1.0 <= g <= 4.0, g
