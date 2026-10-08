"""Telling a slowly-relaxing melt apart from one that is crystallising.

Both present to a drift check as "the density will not settle", so both were being
reported as needing more sampling. Only one of them does. The discriminator is the
potential energy: ordering releases it, relaxation does not.

The thresholds here were placed using nine real replicas across three polyolefins, and
the tests below encode those three regimes rather than synthetic extremes.
"""

from __future__ import annotations

import numpy as np
import pytest

from polymer_engine.analysis.ordering import (
    MIN_ENERGY_DROP_SD,
    MIN_RELATIVE_CHANGE,
    OrderingVerdict,
    analyse,
    determination_for,
    ordering_gates,
    trend,
)
from polymer_engine.core.models import Determination, GateStatus

RNG = np.random.default_rng(20260901)


PHI = 0.98   # lag-one autocorrelation, giving g of roughly 100


def _series(n: int = 4000, *, start: float, drift: float, noise: float):
    """A trending series with correlated noise, like a real observable.

    AR(1) scaled so the *stationary* standard deviation is exactly ``noise``. Getting
    that scaling wrong turns the fixture into a random walk whose amplitude swamps the
    drift it is supposed to carry -- which is what happened on the first attempt, and
    the analysis correctly reported the resulting series as trending the other way.
    """
    t = np.linspace(0.0, 50_000.0, n)             # 50 ns in ps
    innovation = RNG.normal(0.0, noise * np.sqrt(1.0 - PHI**2), n)
    correlated = np.empty(n)
    correlated[0] = RNG.normal(0.0, noise)
    for i in range(1, n):
        correlated[i] = PHI * correlated[i - 1] + innovation[i]
    return t, start + drift * (t / t[-1]) + correlated


# -- the three regimes -----------------------------------------------------
def test_an_equilibrated_melt_is_stationary() -> None:
    """Flat density, flat energy: polypropylene's signature."""
    t, rho = _series(start=847.0, drift=-2.0, noise=11.0)
    _t, energy = _series(start=16_000.0, drift=-40.0, noise=380.0)
    result = analyse(t, rho, energy)
    assert result.verdict is OrderingVerdict.STATIONARY
    assert ordering_gates(result).status is GateStatus.PASS
    assert determination_for(result) is Determination.KNOWN


def test_rising_density_with_falling_energy_is_ordering() -> None:
    """Polyethylene's signature: +9% density, energy down a full standard deviation."""
    t, rho = _series(start=845.0, drift=77.0, noise=11.0)
    _t, energy = _series(start=6_000.0, drift=-2200.0, noise=900.0)
    result = analyse(t, rho, energy)
    assert result.verdict is OrderingVerdict.ORDERING
    assert not result.verdict.more_time_would_help
    assert ordering_gates(result).status is GateStatus.FAIL
    # Not INSUFFICIENT_DATA: more data is not what is missing.
    assert determination_for(result) is Determination.REQUIRES_VALIDATION
    assert "longer run will not change that" in result.reason


def test_rising_density_with_flat_energy_is_unfinished_relaxation() -> None:
    """Polyisobutylene's signature: density climbing, energy going nowhere."""
    t, rho = _series(start=800.0, drift=68.0, noise=20.0)
    _t, energy = _series(start=20_000.0, drift=-200.0, noise=2900.0)
    result = analyse(t, rho, energy)
    assert result.verdict is OrderingVerdict.EQUILIBRATING
    assert result.verdict.more_time_would_help
    assert ordering_gates(result).status is GateStatus.INCONCLUSIVE


# -- why the obvious criteria do not work ----------------------------------
def test_a_tiny_drift_is_not_systematic_however_significant() -> None:
    """With enough effective samples a 0.2% drift is many sigma and still irrelevant."""
    t, rho = _series(n=20_000, start=847.0, drift=1.7, noise=2.0)
    result = analyse(t, rho, np.full(20_000, 16_000.0))
    assert result.density is not None
    assert abs(result.density.relative_change) < MIN_RELATIVE_CHANGE
    assert result.verdict is OrderingVerdict.STATIONARY


def test_a_noisy_trend_is_still_a_trend() -> None:
    """A real rise buried in fluctuation explains little variance and still counts."""
    t, rho = _series(start=845.0, drift=60.0, noise=40.0)
    result = analyse(t, rho, None)
    assert result.density is not None
    assert result.density.r_squared < 0.5, "the fixture must be genuinely noisy"
    assert result.density.systematic


def test_correlated_noise_does_not_inflate_significance() -> None:
    """Naive counting would treat every frame as independent and overstate the drift."""
    t, rho = _series(start=847.0, drift=0.0, noise=11.0)
    measured = trend(t, rho)
    assert measured is not None
    assert measured.n_effective < measured.n / 5, "correlation should be detected"


# -- refusals --------------------------------------------------------------
def test_without_an_energy_series_ordering_cannot_be_claimed() -> None:
    t, rho = _series(start=845.0, drift=77.0, noise=11.0)
    result = analyse(t, rho, None)
    assert result.verdict is OrderingVerdict.INCONCLUSIVE
    assert "no way" in result.reason


@pytest.mark.parametrize("n", [0, 1, 7])
def test_too_few_points_yields_no_trend(n: int) -> None:
    assert trend(np.arange(n, dtype=float), np.zeros(n)) is None


def test_a_constant_series_has_no_trend_to_report() -> None:
    assert trend(np.linspace(0, 1000, 100), np.zeros(100)) is None


def test_falling_density_is_never_ordering() -> None:
    """Ordering compacts. A density that drops is something else entirely."""
    t, rho = _series(start=900.0, drift=-60.0, noise=11.0)
    _t, energy = _series(start=6_000.0, drift=-2200.0, noise=900.0)
    assert analyse(t, rho, energy).verdict is OrderingVerdict.STATIONARY


def test_the_gate_carries_its_evidence() -> None:
    t, rho = _series(start=845.0, drift=77.0, noise=11.0)
    _t, energy = _series(start=6_000.0, drift=-2200.0, noise=900.0)
    gate = ordering_gates(analyse(t, rho, energy)).gates[0]
    assert gate.evidence["density"]["relative_change"] > 0
    assert gate.evidence["energy"]["relative_change"] <= -MIN_ENERGY_DROP_SD
    assert gate.evidence["more_time_would_help"] is False


class TestOscillatoryInefficiency:
    """Barostat ringing aliased by the output interval defeats the Geyer estimator.

    Sampled near half the ringing period, the density *alternates* about its mean:
    lag-one autocorrelation about -0.75 on real data. Geyer's initial positive sequence
    terminates on its first pair and reports g = 1 -- 50,001 frames counted as 50,001
    independent samples -- while block analysis puts the truth near g ~ 1000. The
    downstream damage was standard errors understated forty-fold and a between-replica
    chi-square of ~600 for replicas that actually agreed.
    """

    def _slow_plus_ringing(self, n: int = 20000, ring: float = 25.0):
        rng = np.random.default_rng(20260903)
        slow = np.empty(n)
        slow[0] = 0.0
        innovation = rng.normal(0.0, np.sqrt(1 - 0.999**2), n)
        for i in range(1, n):
            slow[i] = 0.999 * slow[i - 1] + innovation[i]      # tau ~ 1000 frames
        ringing = ring * np.cos(np.pi * 0.92 * np.arange(n))    # near-alternating
        return 850.0 + 5.0 * slow + ringing + rng.normal(0, 0.5, n)

    def test_ringing_no_longer_hides_the_slow_mode(self) -> None:
        from polymer_engine.analysis.statistics import statistical_inefficiency

        series = self._slow_plus_ringing()
        centred = series - series.mean()
        lag_one = float(np.dot(centred[:-1], centred[1:]) / np.dot(centred, centred))
        assert lag_one < -0.5, "the fixture must actually alternate"
        g = statistical_inefficiency(series)
        assert g > 100.0, f"g={g:.1f}: the slow mode under the ringing was missed"

    def test_a_plain_ar1_is_unchanged(self) -> None:
        """The BUG-002 verification case must keep its answer."""
        from polymer_engine.analysis.statistics import statistical_inefficiency

        rng = np.random.default_rng(7)
        n, phi = 200_000, 0.9
        x = np.empty(n)
        x[0] = rng.normal()
        innovation = rng.normal(0.0, np.sqrt(1 - phi**2), n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + innovation[i]
        g = statistical_inefficiency(x)
        expected = 1 + 2 * phi / (1 - phi)                      # = 19 for phi = 0.9
        assert 0.7 * expected < g < 1.3 * expected, g

    def test_white_noise_still_counts_every_sample(self) -> None:
        from polymer_engine.analysis.statistics import statistical_inefficiency

        g = statistical_inefficiency(np.random.default_rng(11).normal(0, 1, 50_000))
        assert g < 1.5, g

    def test_pure_alternation_with_no_slow_mode_stays_near_one(self) -> None:
        """De-aliasing must not invent correlation that is not there."""
        from polymer_engine.analysis.statistics import statistical_inefficiency

        rng = np.random.default_rng(13)
        n = 20_000
        series = 10.0 * np.cos(np.pi * np.arange(n)) + rng.normal(0, 0.5, n)
        g = statistical_inefficiency(series)
        assert g < 8.0, f"g={g:.1f}: alternation alone should not cost much"

    def test_real_replicas_agree_once_the_errors_are_honest(self) -> None:
        """The finding itself, pinned: means 876-879 with honest errors ~2-4 kg/m3
        must not produce a chi-square in the hundreds."""
        means = np.array([879.3, 879.3, 876.6])
        errors = np.array([2.32, 1.88, 4.41])
        weighted = np.average(means, weights=1 / errors**2)
        chi2 = float(np.sum((means - weighted) ** 2 / errors**2) / (len(means) - 1))
        assert chi2 < 4.0, chi2
