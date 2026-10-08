"""Statistics for correlated time series.

The single most important thing in this module is that **MD frames are not
independent samples**.  A 100 ns trajectory written every 10 ps gives 10 000 frames,
but if the density autocorrelation time is 200 ps there are only about 250
independent measurements.  Reporting ``std/sqrt(10000)`` understates the uncertainty
by a factor of ~6 and turns a disagreement into a "significant" result.

Everything here therefore works in terms of the *statistical inefficiency* ``g``
(Chodera et al., JCTC 2007), with

    g = 1 + 2 * tau_int          effective_samples = N / g
    standard_error = sd / sqrt(effective_samples)

Edge cases return an honest ``Measurement`` with ``determination != KNOWN`` rather
than a number: with n = 1 there is no uncertainty to report, and saying ``0.0`` would
be a lie.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from polymer_engine.core.errors import InsufficientDataError
from polymer_engine.core.models import Determination, Measurement

#: Sokal's window factor: truncate the autocorrelation sum at W where W >= c * tau(W).
SOKAL_C = 6.0


def _clean(values: Sequence[float] | np.ndarray, *, name: str = "series") -> np.ndarray:
    """Return a 1-D float array, rejecting non-finite entries loudly."""
    array = np.asarray(values, dtype=float).ravel()
    if array.size and not np.all(np.isfinite(array)):
        n_bad = int(np.count_nonzero(~np.isfinite(array)))
        raise InsufficientDataError(
            f"{name} contains {n_bad} non-finite value(s); clean or mask them explicitly",
            n_bad=n_bad,
            n_total=int(array.size),
        )
    return array


def drop_non_finite(values: Sequence[float] | np.ndarray) -> tuple[np.ndarray, int]:
    """Explicitly remove non-finite entries, reporting how many were dropped."""
    array = np.asarray(values, dtype=float).ravel()
    mask = np.isfinite(array)
    return array[mask], int(np.count_nonzero(~mask))


# --------------------------------------------------------------------------
# Autocorrelation
# --------------------------------------------------------------------------
def autocorrelation(values: Sequence[float] | np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalised autocorrelation function, ``rho(0) = 1``.

    A constant series has no defined correlation structure; ``rho`` is returned as
    ``[1, 0, 0, ...]`` so downstream code sees "no correlation" rather than NaN.
    """
    x = _clean(values, name="autocorrelation input")
    n = x.size
    if n < 2:
        return np.ones(1)
    max_lag = min(max_lag if max_lag is not None else n - 1, n - 1)
    centred = x - x.mean()
    variance = float(np.dot(centred, centred) / n)
    if variance <= 0 or math.isclose(variance, 0.0, abs_tol=1e-300):
        rho = np.zeros(max_lag + 1)
        rho[0] = 1.0
        return rho
    # FFT-based estimate of the (biased) autocovariance.
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, size)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[: max_lag + 1]
    acov /= n
    return acov / acov[0]


def integrated_autocorrelation_time(values: Sequence[float] | np.ndarray) -> float:
    """Integrated autocorrelation time in units of samples.

    Uses Geyer's **initial positive sequence**: consecutive lags are summed in pairs,
    ``Gamma_k = rho(2k+1) + rho(2k+2)``, and the sum is truncated at the first pair that
    is not positive.  A Sokal-style window caps it as well, so the noisy tail of ``rho``
    never enters the estimate.

    Pairing is not a refinement -- it is what makes the estimator work on an oscillating
    autocorrelation function.  Truncating at the first *individual* non-positive lag
    fails badly on real MD data: a 20 ns polyethylene melt sampled every 2 ps against a
    Parrinello-Rahman barostat with ``tau_p = 5 ps`` aliases the volume oscillation, and
    its density autocorrelation comes out as::

        lag  0      1      2      3      4      5      6
        rho  1.000 -0.021 +0.825 +0.079 +0.683 +0.171 +0.567

    The even lags decay smoothly to 1/e near lag 13 (~26 ps), so the series is strongly
    correlated -- but ``rho(1)`` is very slightly negative, so the unpaired rule stops at
    the first lag and returns ``tau = 0``, i.e. ``g = 1``.  That reports 10,001 frames as
    10,001 independent samples and understates the uncertainty by roughly an order of
    magnitude.  Paired, ``Gamma_0 = -0.021 + 0.825 = +0.804`` and the sum proceeds
    correctly.

    Geyer's pairing is provably non-negative for a reversible Markov chain, which is why
    it is the right rule here rather than, say, taking absolute values.
    """
    x = _clean(values, name="autocorrelation input")
    if x.size < 2:
        return 0.0
    rho = autocorrelation(x)
    tau = 0.0
    lag = 1
    while lag + 1 < rho.size:
        pair = float(rho[lag]) + float(rho[lag + 1])
        if pair <= 0.0:
            break
        tau += pair
        lag += 2
        if lag >= SOKAL_C * (1.0 + 2.0 * tau):
            break
    # A final unpaired lag is only added when it is positive on its own.
    if lag < rho.size and lag + 1 >= rho.size and float(rho[lag]) > 0.0:
        tau += float(rho[lag])
    return float(max(tau, 0.0))


#: Halvings applied before giving up on de-aliasing an oscillatory series. Ten levels
#: is a factor of 1024 in sampling interval, far past any barostat period.
_MAX_PAIR_AVERAGING = 10


def statistical_inefficiency(values: Sequence[float] | np.ndarray) -> float:
    """``g = 1 + 2*tau_int``: how many samples one independent measurement costs.

    Always at least 1.0 -- a series cannot contain more information than it has points.

    **Oscillatory series are pre-averaged until the oscillation is gone.** A barostat
    rings the volume (and through it the density) at a period set by ``tau_p``; sampled
    near half that period the series *alternates* about its mean, and the lag-one
    autocorrelation comes out strongly negative. Geyer's initial-positive-sequence sum
    then terminates on its very first pair and reports ``g = 1`` -- fifty thousand
    frames counted as fifty thousand independent samples, while the slow physical modes
    hidden beneath the ringing carry a true inefficiency in the thousands.

    That is not a flaw in Geyer's estimator so much as a violated premise: its
    initial-sequence argument is for reversible chains, and an aliased deterministic
    oscillation is not one. Averaging consecutive pairs of frames halves the sampling
    rate and cancels the alternation; repeated until the lag-one autocorrelation is
    non-negative, the estimator sees the underlying process instead of the ringing, and
    the total inefficiency is the block factor times the inefficiency of the blocked
    series.

    Measured on the data that exposed this (three polyisobutylene replicas, lag-one
    autocorrelation about -0.75): reported g went from 1.0 to roughly 1000-3000, and
    the between-replica chi-square -- computed from standard errors that had been
    understated forty-fold -- fell from about 600 to order 1. The replicas agreed all
    along; the error bars were wrong.
    """
    x = _clean(np.asarray(values, dtype=float), name="series")
    if x.size < 4:
        return 1.0

    block = 1
    for _ in range(_MAX_PAIR_AVERAGING):
        if x.size < 8:
            break
        centred = x - x.mean()
        denominator = float(np.dot(centred, centred))
        if denominator <= 0.0:
            break
        lag_one = float(np.dot(centred[:-1], centred[1:])) / denominator
        # A margin, not zero: white noise's sample lag-one fluctuates around zero with
        # standard deviation 1/sqrt(n) and can land slightly negative by chance; a
        # zero threshold then pair-averages noise into shorter noise, whose lag-one is
        # again slightly negative, and the cascade multiplies g by the block factor for
        # nothing. Real barostat ringing sits near -0.7; noise sits within a few
        # thousandths. The margin only needs to separate those.
        if lag_one >= -max(0.05, 3.0 / np.sqrt(x.size)):
            break
        # Alternation about the mean: average consecutive pairs and try again.
        n = (x.size // 2) * 2
        x = x[:n].reshape(-1, 2).mean(axis=1)
        block *= 2

    return block * max(1.0, 1.0 + 2.0 * integrated_autocorrelation_time(x))


def effective_sample_size(values: Sequence[float] | np.ndarray) -> float:
    """Number of *independent* samples in a correlated series."""
    x = _clean(values, name="series")
    if x.size == 0:
        return 0.0
    return float(x.size) / statistical_inefficiency(x)


# --------------------------------------------------------------------------
# Location and spread
# --------------------------------------------------------------------------
def describe(
    values: Sequence[float] | np.ndarray,
    *,
    name: str = "value",
    units: str = "1",
    account_for_correlation: bool = True,
) -> Measurement:
    """Mean with a correlation-aware standard error.

    ``account_for_correlation=False`` gives the naive ``sd/sqrt(n)``; it exists only so
    the difference can be demonstrated in tests and should not be used for MD data.
    """
    x = _clean(values, name=name)
    n = int(x.size)
    if n == 0:
        return Measurement.unknown(name, units=units, reason="no samples", determination=Determination.INSUFFICIENT_DATA)
    if n == 1:
        return Measurement(
            name=name,
            value=float(x[0]),
            uncertainty=None,
            units=units,
            n_samples=1,
            effective_samples=1.0,
            method="single sample",
            notes="a single sample has no estimable uncertainty",
        )

    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if account_for_correlation:
        ess = effective_sample_size(x)
        method = "correlation-aware (Chodera statistical inefficiency)"
    else:
        ess = float(n)
        method = "naive iid standard error"
    stderr = sd / math.sqrt(ess) if ess > 0 else float("inf")
    return Measurement(
        name=name,
        value=mean,
        uncertainty=stderr,
        units=units,
        n_samples=n,
        effective_samples=ess,
        method=method,
    )


def block_average(
    values: Sequence[float] | np.ndarray, n_blocks: int = 5
) -> tuple[np.ndarray, Measurement]:
    """Split into ``n_blocks`` contiguous blocks and average each.

    Block means are far closer to independent than raw frames, which is what makes the
    spread of block means a usable error estimate.
    """
    x = _clean(values, name="series")
    if n_blocks < 2:
        raise InsufficientDataError("Block averaging needs at least 2 blocks", n_blocks=n_blocks)
    if x.size < n_blocks * 2:
        raise InsufficientDataError(
            "Not enough samples for the requested block count",
            n_samples=int(x.size),
            n_blocks=n_blocks,
            minimum=n_blocks * 2,
        )
    blocks = np.array([chunk.mean() for chunk in np.array_split(x, n_blocks)])
    stderr = float(blocks.std(ddof=1) / math.sqrt(n_blocks))
    return blocks, Measurement(
        name="block_mean",
        value=float(blocks.mean()),
        uncertainty=stderr,
        n_samples=int(x.size),
        effective_samples=float(n_blocks),
        method=f"block averaging with {n_blocks} blocks",
    )


def blocking_curve(values: Sequence[float] | np.ndarray) -> list[tuple[int, float]]:
    """Standard error as a function of block size (Flyvbjerg-Petersen blocking).

    The curve plateaus once blocks are longer than the correlation time; the plateau
    height is the true standard error.
    """
    x = _clean(values, name="series")
    out: list[tuple[int, float]] = []
    current = x.copy()
    while current.size >= 4:
        stderr = float(current.std(ddof=1) / math.sqrt(current.size))
        out.append((int(x.size // current.size), stderr))
        if current.size % 2:
            current = current[:-1]
        current = current.reshape(-1, 2).mean(axis=1)
    return out


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    block_size: int | None = None,
    seed: int = 20240101,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval.

    ``block_size`` selects the *moving block* bootstrap, which preserves serial
    correlation.  For time-series data, leave it as ``None`` and the block size is
    chosen from the estimated correlation time -- resampling correlated data as if it
    were iid produces intervals that are far too narrow.
    """
    x = _clean(values, name="series")
    if x.size < 2:
        raise InsufficientDataError("Bootstrap needs at least 2 samples", n_samples=int(x.size))
    if not 0 < confidence_level < 1:
        raise InsufficientDataError("confidence_level must be in (0, 1)", confidence_level=confidence_level)

    rng = np.random.default_rng(seed)
    if block_size is None:
        g = statistical_inefficiency(x)
        block_size = max(1, min(int(x.size // 2), math.ceil(g)))

    estimates = np.empty(n_resamples, dtype=float)
    if block_size <= 1:
        for i in range(n_resamples):
            estimates[i] = statistic(rng.choice(x, size=x.size, replace=True))
    else:
        n_blocks = math.ceil(x.size / block_size)
        max_start = x.size - block_size
        for i in range(n_resamples):
            starts = rng.integers(0, max_start + 1, size=n_blocks)
            sample = np.concatenate([x[s : s + block_size] for s in starts])[: x.size]
            estimates[i] = statistic(sample)

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.quantile(estimates, alpha))
    upper = float(np.quantile(estimates, 1.0 - alpha))
    return float(statistic(x)), lower, upper


# --------------------------------------------------------------------------
# Equilibration detection
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EquilibrationResult:
    start_index: int
    n_discarded: int
    n_retained: int
    effective_samples: float
    statistical_inefficiency: float
    method: str

    @property
    def discarded_fraction(self) -> float:
        total = self.n_discarded + self.n_retained
        return self.n_discarded / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "n_discarded": self.n_discarded,
            "n_retained": self.n_retained,
            "discarded_fraction": self.discarded_fraction,
            "effective_samples": self.effective_samples,
            "statistical_inefficiency": self.statistical_inefficiency,
            "method": self.method,
        }


def detect_equilibration(
    values: Sequence[float] | np.ndarray, *, max_scan_fraction: float = 0.5
) -> EquilibrationResult:
    """Choose the production window by maximising effective sample count.

    Chodera's rule: for each candidate start ``t0``, the retained series has
    ``(N - t0) / g(t0)`` effective samples; keep the ``t0`` that maximises it.  This
    trades bias (early, unequilibrated frames) against variance (throwing data away)
    rather than applying an arbitrary "discard the first 20%".
    """
    x = _clean(values, name="series")
    n = int(x.size)
    if n < 10:
        return EquilibrationResult(
            start_index=0,
            n_discarded=0,
            n_retained=n,
            effective_samples=float(n),
            statistical_inefficiency=1.0,
            method="series too short to detect equilibration",
        )

    best_start, best_ess, best_g = 0, -1.0, 1.0
    limit = max(1, int(n * max_scan_fraction))
    step = max(1, limit // 100)
    for start in range(0, limit, step):
        segment = x[start:]
        if segment.size < 10:
            break
        g = statistical_inefficiency(segment)
        ess = segment.size / g
        if ess > best_ess:
            best_start, best_ess, best_g = start, ess, g

    return EquilibrationResult(
        start_index=best_start,
        n_discarded=best_start,
        n_retained=n - best_start,
        effective_samples=float(best_ess),
        statistical_inefficiency=float(best_g),
        method="reverse-cumulative effective-sample maximisation (Chodera 2016)",
    )


# --------------------------------------------------------------------------
# Association
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CorrelationResult:
    statistic: float | None
    p_value: float | None
    n: int
    method: str
    determination: Determination = Determination.KNOWN
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n": self.n,
            "method": self.method,
            "determination": self.determination.value,
            "note": self.note,
        }


def _pair(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.size != b.size:
        raise InsufficientDataError("Paired series must be the same length", n_x=int(a.size), n_y=int(b.size))
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _undefined(method: str, n: int, note: str) -> CorrelationResult:
    return CorrelationResult(
        statistic=None, p_value=None, n=n, method=method,
        determination=Determination.INSUFFICIENT_DATA, note=note,
    )


def pearson(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> CorrelationResult:
    """Pearson correlation with a two-sided p-value."""
    a, b = _pair(x, y)
    n = int(a.size)
    if n < 3:
        return _undefined("pearson", n, "at least 3 complete pairs are required")
    if np.std(a) == 0 or np.std(b) == 0:
        return _undefined("pearson", n, "a constant input has no defined correlation")
    r = float(np.corrcoef(a, b)[0, 1])
    return CorrelationResult(statistic=r, p_value=_correlation_p_value(r, n), n=n, method="pearson")


def spearman(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> CorrelationResult:
    """Spearman rank correlation (average ranks for ties)."""
    a, b = _pair(x, y)
    n = int(a.size)
    if n < 3:
        return _undefined("spearman", n, "at least 3 complete pairs are required")
    ra, rb = _rank(a), _rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return _undefined("spearman", n, "a constant input has no defined rank correlation")
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return CorrelationResult(statistic=rho, p_value=_correlation_p_value(rho, n), n=n, method="spearman")


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    # Average ranks within tied groups.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index in np.flatnonzero(counts > 1):
        mask = inverse == index
        ranks[mask] = ranks[mask].mean()
    return ranks


def _correlation_p_value(r: float, n: int) -> float | None:
    """Two-sided p-value from the t transformation; ``None`` without SciPy."""
    if abs(r) >= 1.0:
        return 0.0
    t = abs(r) * math.sqrt((n - 2) / (1.0 - r * r))
    try:
        from scipy import stats
    except ImportError:
        return None
    return float(2.0 * stats.t.sf(t, df=n - 2))


def partial_correlation(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    covariates: Sequence[Sequence[float]] | np.ndarray,
) -> CorrelationResult:
    """Correlation of x and y after linearly removing ``covariates`` from both.

    This is the minimum defence against reporting a confounded association: two
    descriptors that both track molar mass will correlate with each other whether or
    not either drives the property.
    """
    a, b = _pair(x, y)
    z = np.atleast_2d(np.asarray(covariates, dtype=float))
    if z.shape[0] != a.size:
        z = z.T
    if z.shape[0] != a.size:
        raise InsufficientDataError(
            "Covariates must have one row per observation", n_obs=int(a.size), covariate_shape=z.shape
        )
    n = int(a.size)
    if n < z.shape[1] + 3:
        return _undefined("partial_pearson", n, "too few observations for the number of covariates")

    design = np.column_stack([np.ones(n), z])
    resid_a = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
    resid_b = b - design @ np.linalg.lstsq(design, b, rcond=None)[0]
    if np.std(resid_a) == 0 or np.std(resid_b) == 0:
        return _undefined("partial_pearson", n, "residuals are constant after removing covariates")
    r = float(np.corrcoef(resid_a, resid_b)[0, 1])
    dof = n - z.shape[1] - 2
    p = _correlation_p_value(r, dof + 2) if dof > 0 else None
    return CorrelationResult(
        statistic=r, p_value=p, n=n, method=f"partial_pearson (controlling {z.shape[1]} covariate(s))"
    )


def permutation_test(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_permutations: int = 10_000,
    seed: int = 20240101,
) -> CorrelationResult:
    """Two-sided permutation test, which assumes nothing about the distributions."""
    a, b = _pair(x, y)
    n = int(a.size)
    if n < 4:
        return _undefined("permutation", n, "at least 4 complete pairs are required")
    fn = statistic or (lambda u, v: float(np.corrcoef(u, v)[0, 1]))
    if np.std(a) == 0 or np.std(b) == 0:
        return _undefined("permutation", n, "a constant input has no defined association")

    observed = fn(a, b)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        if abs(fn(a, rng.permutation(b))) >= abs(observed):
            count += 1
    # +1 smoothing so an empirical p-value is never exactly zero.
    p = (count + 1) / (n_permutations + 1)
    return CorrelationResult(
        statistic=float(observed), p_value=float(p), n=n, method=f"permutation ({n_permutations} draws)"
    )


def welch_t_test(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> CorrelationResult:
    """Welch's t-test for two samples of unequal size and variance."""
    x = _clean(a, name="sample a")
    y = _clean(b, name="sample b")
    if x.size < 2 or y.size < 2:
        return _undefined("welch_t", int(min(x.size, y.size)), "each sample needs at least 2 values")
    va, vb = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    denominator = va + vb
    if denominator <= 0:
        return _undefined("welch_t", int(x.size + y.size), "both samples are constant")
    t = float((x.mean() - y.mean()) / math.sqrt(denominator))
    dof = denominator**2 / (va**2 / (x.size - 1) + vb**2 / (y.size - 1))
    try:
        from scipy import stats

        p: float | None = float(2.0 * stats.t.sf(abs(t), df=dof))
    except ImportError:
        p = None
    return CorrelationResult(statistic=t, p_value=p, n=int(x.size + y.size), method="welch_t")


__all__ = [
    "CorrelationResult",
    "EquilibrationResult",
    "autocorrelation",
    "block_average",
    "blocking_curve",
    "bootstrap_ci",
    "describe",
    "detect_equilibration",
    "drop_non_finite",
    "effective_sample_size",
    "integrated_autocorrelation_time",
    "partial_correlation",
    "pearson",
    "permutation_test",
    "spearman",
    "statistical_inefficiency",
    "welch_t_test",
]
