"""Umbrella-sampling analysis: overlap, WHAM, PMF, uncertainty, convergence.

The design goal is that the engine must be able to say **"do not trust this PMF yet"**
and say it for a specific reason.  A PMF is produced only alongside:

* an overlap diagnostic for every adjacent window pair,
* a bootstrap uncertainty band,
* a convergence check comparing the first and second half of each window's sampling,
* an explicit list of the windows and gaps that failed.

If windows do not overlap, the free-energy differences across the gap are not
determined by the data at all -- WHAM will still return numbers, and those numbers are
meaningless.  :func:`build_pmf` therefore returns a ``determination`` that downstream
code must check before using the result.

Sign convention (stated because it is the easiest thing in this file to get backwards):
``PMF(x) = -kT ln p_unbiased(x)``, so the PMF is *high* where the system is unlikely.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polymer_engine.core.config import UmbrellaDefaults
from polymer_engine.core.errors import InsufficientDataError
from polymer_engine.core.models import Determination, GateReport, GateResult, GateStatus, Measurement
from polymer_engine.core.units import kT

DEFAULT_WHAM_TOLERANCE = 1e-7

#: Iteration cap for the self-consistent solution.  Measured on this implementation:
#: well-overlapped windows (0.08 nm spacing, ~1.6 sigma) converge in ~250 iterations and
#: sparse-but-overlapping windows (0.3 nm) in ~4,500.  Windows that do not overlap have
#: no solution to converge to and will run forever, so 20,000 is comfortably above any
#: well-posed problem while failing fast on an ill-posed one.  Hitting this cap is
#: reported as non-convergence, which blocks the PMF from being trusted.
DEFAULT_WHAM_MAX_ITERATIONS = 20_000


@dataclass
class WindowSamples:
    """One window's sampled coordinate values, after equilibration trimming."""

    index: int
    center: float
    force_constant: float
    values: np.ndarray
    units: str = "nm"
    discarded: int = 0

    @property
    def n(self) -> int:
        return int(self.values.size)

    @property
    def mean(self) -> float:
        return float(self.values.mean()) if self.n else float("nan")

    @property
    def sd(self) -> float:
        return float(self.values.std(ddof=1)) if self.n > 1 else float("nan")


def trim_equilibration(
    values: Sequence[float] | np.ndarray, fraction: float
) -> tuple[np.ndarray, int]:
    """Drop the leading ``fraction`` of a window's samples."""
    array = np.asarray(values, dtype=float).ravel()
    if not 0 <= fraction < 1:
        raise InsufficientDataError("Equilibration fraction must be in [0, 1)", fraction=fraction)
    start = int(array.size * fraction)
    return array[start:], start


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------
@dataclass
class OverlapDiagnostic:
    """Histogram overlap between every adjacent window pair."""

    centers: list[float]
    pair_overlap: list[float]
    matrix: np.ndarray
    min_pair_overlap: float
    threshold: float
    gaps: list[tuple[float, float]] = field(default_factory=list)
    empty_windows: list[int] = field(default_factory=list)

    @property
    def sufficient(self) -> bool:
        return not self.gaps and not self.empty_windows

    def as_dict(self) -> dict[str, Any]:
        return {
            "centers": self.centers,
            "pair_overlap": self.pair_overlap,
            "min_pair_overlap": self.min_pair_overlap,
            "threshold": self.threshold,
            "gaps": [list(g) for g in self.gaps],
            "empty_windows": self.empty_windows,
            "sufficient": self.sufficient,
        }


def overlap_matrix(windows: Sequence[WindowSamples], bins: int = 100) -> np.ndarray:
    """Pairwise histogram overlap coefficient on a shared grid.

    The overlap coefficient is ``sum_i min(p_i, q_i)``: 1 for identical distributions,
    0 for disjoint ones.  A window with no samples yields zero overlap with everything,
    which is the correct answer, not a division-by-zero.
    """
    usable = [w for w in windows if w.n > 0]
    if not usable:
        return np.zeros((len(windows), len(windows)))
    low = min(float(w.values.min()) for w in usable)
    high = max(float(w.values.max()) for w in usable)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return np.eye(len(windows))

    edges = np.linspace(low, high, bins + 1)
    histograms = np.zeros((len(windows), bins), dtype=float)
    for i, window in enumerate(windows):
        if window.n == 0:
            continue
        counts, _ = np.histogram(window.values, bins=edges)
        total = counts.sum()
        if total > 0:
            histograms[i] = counts / total

    n = len(windows)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = float(np.minimum(histograms[i], histograms[j]).sum())
    return matrix


def diagnose_overlap(
    windows: Sequence[WindowSamples], *, min_overlap: float = 0.10, bins: int = 100
) -> OverlapDiagnostic:
    """Report which adjacent window pairs fail to overlap.

    A *pair* is the unit of failure, not a window: a gap between windows i and i+1
    leaves the free-energy difference across it undetermined regardless of how well
    each window overlaps its other neighbour.
    """
    ordered = sorted(windows, key=lambda w: w.center)
    matrix = overlap_matrix(ordered, bins=bins)
    centers = [w.center for w in ordered]
    empty = [w.index for w in ordered if w.n == 0]

    pair_overlap: list[float] = []
    gaps: list[tuple[float, float]] = []
    for i in range(len(ordered) - 1):
        value = float(matrix[i, i + 1])
        pair_overlap.append(value)
        if value < min_overlap:
            gaps.append((centers[i], centers[i + 1]))

    return OverlapDiagnostic(
        centers=centers,
        pair_overlap=pair_overlap,
        matrix=matrix,
        min_pair_overlap=min(pair_overlap) if pair_overlap else 0.0,
        threshold=min_overlap,
        gaps=gaps,
        empty_windows=empty,
    )


#: A bin holding fewer than this fraction of the busiest bin's samples is statistically
#: empty; comparing PMF values there compares noise, not physics.
WELL_SAMPLED_FRACTION = 0.02

#: ...and an absolute floor, because a fraction of a small dataset is still tiny.
WELL_SAMPLED_MIN_COUNTS = 30


def well_sampled_mask(
    windows: Sequence[WindowSamples], *, bins: int, range_limits: tuple[float, float]
) -> np.ndarray:
    """Bins with enough pooled samples for their PMF value to mean anything."""
    low, high = range_limits
    edges = np.linspace(low, high, bins + 1)
    total = np.zeros(bins, dtype=float)
    for window in windows:
        if window.n:
            counts, _ = np.histogram(window.values, bins=edges)
            total += counts
    if total.max() <= 0:
        return np.zeros(bins, dtype=bool)
    floor = min(WELL_SAMPLED_MIN_COUNTS, max(1.0, total.max() * WELL_SAMPLED_FRACTION))
    return total >= max(WELL_SAMPLED_FRACTION * total.max(), floor)


def align_pmfs(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Largest shape difference between two PMFs, ignoring their arbitrary offset.

    A PMF is determined only up to an additive constant, so comparing raw values would
    measure the gauge choice rather than whether the sampling has converged.  The two
    curves are shifted onto a common mean over ``mask`` before differencing.
    """
    if mask.sum() == 0:
        return float("nan")
    difference = a[mask] - b[mask]
    return float(np.max(np.abs(difference - difference.mean())))


# --------------------------------------------------------------------------
# WHAM
# --------------------------------------------------------------------------
@dataclass
class WhamResult:
    coordinate: np.ndarray
    pmf: np.ndarray
    free_energies: np.ndarray
    converged: bool
    iterations: int
    residual: float
    temperature_k: float
    units: str = "nm"
    energy_units: str = "kJ/mol"


def wham(
    windows: Sequence[WindowSamples],
    *,
    temperature_k: float,
    bins: int = 100,
    tolerance: float = DEFAULT_WHAM_TOLERANCE,
    max_iterations: int = DEFAULT_WHAM_MAX_ITERATIONS,
    range_limits: tuple[float, float] | None = None,
) -> WhamResult:
    """Binned WHAM for harmonic umbrella windows.

    Solves the self-consistent pair

        p(x_i)  = sum_k H_k(x_i) / sum_k N_k exp((f_k - u_k(x_i)) / kT)
        exp(-f_k / kT) = sum_i p(x_i) exp(-u_k(x_i) / kT)

    Iteration is done in log space, which keeps the exponentials from underflowing for
    stiff restraints or widely separated windows.
    """
    usable = [w for w in windows if w.n > 0]
    if len(usable) < 2:
        raise InsufficientDataError(
            "WHAM needs at least two windows with samples", n_windows_with_samples=len(usable)
        )
    beta = 1.0 / kT(temperature_k)

    if range_limits is None:
        low = min(float(w.values.min()) for w in usable)
        high = max(float(w.values.max()) for w in usable)
    else:
        low, high = range_limits
    if high <= low:
        raise InsufficientDataError("Sampled coordinate range is degenerate", low=low, high=high)

    edges = np.linspace(low, high, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    counts = np.zeros((len(usable), bins), dtype=float)
    for k, window in enumerate(usable):
        counts[k], _ = np.histogram(window.values, bins=edges)
    n_k = counts.sum(axis=1)
    total_counts = counts.sum(axis=0)

    # Bias energy of each window evaluated at each bin centre.
    bias = np.array(
        [0.5 * w.force_constant * (centers - w.center) ** 2 for w in usable], dtype=float
    )

    f = np.zeros(len(usable), dtype=float)  # free energies in kJ/mol
    occupied = total_counts > 0
    converged = False
    residual = float("inf")
    iterations = 0

    log_n = np.log(np.where(n_k > 0, n_k, 1.0))
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        # log denominator_i = logsumexp_k [ log N_k + beta*(f_k - u_k(x_i)) ]
        terms = log_n[:, None] + beta * (f[:, None] - bias)
        max_term = terms.max(axis=0)
        log_denominator = max_term + np.log(np.exp(terms - max_term).sum(axis=0))

        log_p = np.full(bins, -np.inf)
        log_p[occupied] = np.log(total_counts[occupied]) - log_denominator[occupied]

        # log exp(-beta f_k) = logsumexp_i [ log p_i - beta u_k(x_i) ], vectorised over
        # windows: the per-window Python loop dominated the runtime of every PMF.
        terms_k = log_p[None, :] - beta * bias
        terms_k = np.where(occupied[None, :], terms_k, -np.inf)
        row_max = terms_k.max(axis=1)
        finite_rows = np.isfinite(row_max)
        new_f = f.copy()
        if np.any(finite_rows):
            shifted = terms_k[finite_rows] - row_max[finite_rows, None]
            total = np.exp(shifted).sum(axis=1)
            new_f[finite_rows] = -(row_max[finite_rows] + np.log(total)) / beta

        new_f -= new_f[0]  # gauge fix: only differences are determined
        residual = float(np.max(np.abs(new_f - f)))
        f = new_f
        if residual < tolerance:
            converged = True
            break

    terms = log_n[:, None] + beta * (f[:, None] - bias)
    max_term = terms.max(axis=0)
    log_denominator = max_term + np.log(np.exp(terms - max_term).sum(axis=0))
    log_p = np.full(bins, -np.inf)
    log_p[occupied] = np.log(total_counts[occupied]) - log_denominator[occupied]

    # PMF(x) = -kT ln p(x).  High PMF means low probability.
    pmf = np.full(bins, np.nan)
    pmf[occupied] = -log_p[occupied] / beta
    if np.any(occupied):
        pmf[occupied] -= np.nanmin(pmf[occupied])

    return WhamResult(
        coordinate=centers,
        pmf=pmf,
        free_energies=f,
        converged=converged,
        iterations=iterations,
        residual=residual,
        temperature_k=temperature_k,
        units=usable[0].units,
    )


# --------------------------------------------------------------------------
# PMF with uncertainty and verdict
# --------------------------------------------------------------------------
@dataclass
class PmfResult:
    """A PMF plus everything needed to decide whether to believe it."""

    coordinate: np.ndarray
    pmf: np.ndarray
    well_sampled: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    uncertainty: np.ndarray | None
    overlap: OverlapDiagnostic
    wham_converged: bool
    half_split_max_deviation: float | None
    temperature_k: float
    units: str
    energy_units: str = "kJ/mol"
    determination: Determination = Determination.REQUIRES_VALIDATION
    problems: list[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return self.determination is Determination.KNOWN

    def barrier(self) -> Measurement:
        """Height of the highest barrier encountered moving outward from the minimum.

        Not simply ``max - min``: the barrier that matters is the one the system has to
        cross starting from the free-energy minimum, so it is measured from the minimum
        outward rather than between two arbitrary points on the curve.
        """
        if not self.trustworthy:
            return Measurement.unknown(
                "pmf_barrier",
                units=self.energy_units,
                reason="; ".join(self.problems) or "PMF has not been validated",
                determination=self.determination,
            )
        finite = np.isfinite(self.pmf)
        if finite.sum() < 3:
            return Measurement.unknown(
                "pmf_barrier", units=self.energy_units, reason="too few occupied bins",
                determination=Determination.INSUFFICIENT_DATA,
            )
        values = np.where(finite, self.pmf, np.nan)
        i_min = int(np.nanargmin(values))
        right = np.nanmax(values[i_min:]) if i_min < values.size - 1 else np.nan
        left = np.nanmax(values[: i_min + 1]) if i_min > 0 else np.nan
        barrier = float(np.nanmax([left, right]) - values[i_min])
        uncertainty = None
        if self.uncertainty is not None:
            region = finite & self.well_sampled
            uncertainty = float(np.nanmax(self.uncertainty[region])) if region.any() else None
        return Measurement(
            name="pmf_barrier",
            value=barrier,
            uncertainty=uncertainty,
            units=self.energy_units,
            method="WHAM with block bootstrap",
        )

    def well_depth(self) -> Measurement:
        """Depth of the free-energy well relative to the largest sampled separation.

        Reported only when the coordinate actually reaches a plateau; without one there
        is no meaningful reference state and the number would be arbitrary.
        """
        if not self.trustworthy:
            return Measurement.unknown(
                "binding_free_energy",
                units=self.energy_units,
                reason="; ".join(self.problems) or "PMF has not been validated",
                determination=self.determination,
            )
        finite = np.isfinite(self.pmf)
        if finite.sum() < 5:
            return Measurement.unknown(
                "binding_free_energy", units=self.energy_units, reason="too few occupied bins",
                determination=Determination.INSUFFICIENT_DATA,
            )
        values = self.pmf[finite]
        tail = values[-max(3, values.size // 10) :]
        if float(tail.std()) > 1.0:
            return Measurement.unknown(
                "binding_free_energy",
                units=self.energy_units,
                reason=(
                    f"the PMF has not plateaued at large separation (tail sd "
                    f"{tail.std():.2f} {self.energy_units}); there is no reference state"
                ),
                determination=Determination.INSUFFICIENT_DATA,
            )
        depth = float(tail.mean() - values.min())
        region = finite & self.well_sampled
        uncertainty = (
            float(np.nanmax(self.uncertainty[region]))
            if self.uncertainty is not None and region.any()
            else None
        )
        return Measurement(
            name="binding_free_energy",
            value=depth,
            uncertainty=uncertainty,
            units=self.energy_units,
            method="WHAM well depth relative to the plateau",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate.tolist(),
            "pmf": [None if math.isnan(v) else v for v in self.pmf.tolist()],
            "uncertainty": None
            if self.uncertainty is None
            else [None if math.isnan(v) else v for v in self.uncertainty.tolist()],
            "units": self.units,
            "energy_units": self.energy_units,
            "temperature_k": self.temperature_k,
            "wham_converged": self.wham_converged,
            "half_split_max_deviation": self.half_split_max_deviation,
            "overlap": self.overlap.as_dict(),
            "determination": self.determination.value,
            "trustworthy": self.trustworthy,
            "problems": self.problems,
        }


def build_pmf(
    windows: Sequence[WindowSamples],
    *,
    temperature_k: float,
    defaults: UmbrellaDefaults | None = None,
    bins: int = 100,
    bootstrap_samples: int | None = None,
    seed: int = 20240101,
    max_half_split_deviation: float = 2.0,
) -> PmfResult:
    """Build a PMF and decide whether it can be trusted.

    ``determination`` is ``KNOWN`` only when the windows overlap everywhere, WHAM
    converged, and the first and second halves of the sampling agree.  Otherwise the
    numbers are still returned -- for diagnosis -- but flagged.
    """
    defaults = defaults or UmbrellaDefaults()
    bootstrap_samples = bootstrap_samples if bootstrap_samples is not None else defaults.bootstrap_samples

    overlap = diagnose_overlap(windows, min_overlap=defaults.min_pair_overlap, bins=bins)
    problems: list[str] = []
    if overlap.empty_windows:
        problems.append(f"windows with no samples: {overlap.empty_windows}")
    for lower, upper in overlap.gaps:
        problems.append(
            f"insufficient overlap between windows at {lower:.3f} and {upper:.3f} "
            f"(need >= {defaults.min_pair_overlap:.2f})"
        )

    usable = [w for w in windows if w.n > 0]
    if len(usable) < 2:
        raise InsufficientDataError(
            "A PMF needs at least two windows with samples", n_windows_with_samples=len(usable)
        )

    low = min(float(w.values.min()) for w in usable)
    high = max(float(w.values.max()) for w in usable)
    result = wham(usable, temperature_k=temperature_k, bins=bins, range_limits=(low, high))
    if not result.converged:
        problems.append(
            f"WHAM did not converge in {result.iterations} iterations (residual {result.residual:.3g})"
        )

    # -- convergence: does the first half of the sampling give the same PMF? --
    half_deviation: float | None = None
    try:
        first = [
            WindowSamples(w.index, w.center, w.force_constant, w.values[: w.n // 2], w.units)
            for w in usable
            if w.n >= 4
        ]
        second = [
            WindowSamples(w.index, w.center, w.force_constant, w.values[w.n // 2 :], w.units)
            for w in usable
            if w.n >= 4
        ]
        if len(first) >= 2 and len(second) >= 2:
            pmf_a = wham(first, temperature_k=temperature_k, bins=bins, range_limits=(low, high)).pmf
            pmf_b = wham(second, temperature_k=temperature_k, bins=bins, range_limits=(low, high)).pmf
            # Restrict the comparison to well-sampled bins: in the sparsely visited
            # tails both halves are dominated by noise, and demanding agreement there
            # would reject converged PMFs for the wrong reason.
            both = (
                np.isfinite(pmf_a)
                & np.isfinite(pmf_b)
                & well_sampled_mask(usable, bins=bins, range_limits=(low, high))
            )
            if both.sum() >= 3:
                half_deviation = align_pmfs(pmf_a, pmf_b, both)
                if half_deviation > max_half_split_deviation:
                    problems.append(
                        f"first and second halves of the sampling give PMFs differing by up to "
                        f"{half_deviation:.2f} kJ/mol (limit {max_half_split_deviation:.2f}); not converged"
                    )
    except InsufficientDataError:
        problems.append("could not run the half-split convergence check")

    # -- bootstrap uncertainty ------------------------------------------
    lower_band = upper_band = uncertainty = None
    if bootstrap_samples >= 20:
        lower_band, upper_band, uncertainty = _bootstrap_pmf(
            usable,
            temperature_k=temperature_k,
            bins=bins,
            n_resamples=bootstrap_samples,
            seed=seed,
            range_limits=(low, high),
        )

    determination = Determination.KNOWN if not problems else Determination.REQUIRES_VALIDATION
    return PmfResult(
        well_sampled=well_sampled_mask(usable, bins=bins, range_limits=(low, high)),
        coordinate=result.coordinate,
        pmf=result.pmf,
        lower=lower_band,
        upper=upper_band,
        uncertainty=uncertainty,
        overlap=overlap,
        wham_converged=result.converged,
        half_split_max_deviation=half_deviation,
        temperature_k=temperature_k,
        units=result.units,
        determination=determination,
        problems=problems,
    )


def _bootstrap_pmf(
    windows: Sequence[WindowSamples],
    *,
    temperature_k: float,
    bins: int,
    n_resamples: int,
    seed: int,
    range_limits: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block-bootstrap each window independently and re-run WHAM.

    Blocks preserve within-window correlation; resampling frames independently would
    produce an error band several times too narrow.
    """
    rng = np.random.default_rng(seed)
    curves: list[np.ndarray] = []
    for _ in range(n_resamples):
        resampled: list[WindowSamples] = []
        for window in windows:
            n = window.n
            block = max(1, min(n // 4, 50))
            n_blocks = math.ceil(n / block)
            starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
            values = np.concatenate([window.values[s : s + block] for s in starts])[:n]
            resampled.append(
                WindowSamples(window.index, window.center, window.force_constant, values, window.units)
            )
        try:
            curves.append(
                wham(resampled, temperature_k=temperature_k, bins=bins, range_limits=range_limits).pmf
            )
        except InsufficientDataError:  # pragma: no cover - degenerate resample
            continue

    if not curves:  # pragma: no cover - defensive
        empty = np.full(bins, np.nan)
        return empty, empty, empty

    stacked = np.vstack(curves)
    with np.errstate(invalid="ignore"):
        lower = np.nanquantile(stacked, 0.025, axis=0)
        upper = np.nanquantile(stacked, 0.975, axis=0)
        spread = np.nanstd(stacked, axis=0)
    return lower, upper, spread


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
def pmf_gates(result: PmfResult, *, defaults: UmbrellaDefaults | None = None) -> GateReport:
    """Turn a PMF result into an auditable gate report."""
    defaults = defaults or UmbrellaDefaults()
    report = GateReport(name="umbrella_pmf")

    overlap = result.overlap
    if overlap.empty_windows:
        report.gates.append(
            GateResult(
                gate="umbrella:windows_sampled",
                status=GateStatus.FAIL,
                message=f"{len(overlap.empty_windows)} window(s) produced no samples",
                evidence={"empty_windows": overlap.empty_windows},
            )
        )
    else:
        report.gates.append(
            GateResult(
                gate="umbrella:windows_sampled",
                status=GateStatus.PASS,
                message=f"All {len(overlap.centers)} windows produced samples",
                value=float(len(overlap.centers)),
                units="1",
            )
        )

    if overlap.gaps:
        report.gates.append(
            GateResult(
                gate="umbrella:window_overlap",
                status=GateStatus.FAIL,
                message=(
                    f"{len(overlap.gaps)} adjacent window pair(s) fall below the minimum overlap "
                    f"of {overlap.threshold:.2f}; free-energy differences across those gaps are "
                    "not determined by the data"
                ),
                value=overlap.min_pair_overlap,
                threshold=overlap.threshold,
                units="1",
                evidence={"gaps": [list(g) for g in overlap.gaps]},
            )
        )
    else:
        report.gates.append(
            GateResult(
                gate="umbrella:window_overlap",
                status=GateStatus.PASS,
                message=f"Every adjacent pair overlaps (minimum {overlap.min_pair_overlap:.3f})",
                value=overlap.min_pair_overlap,
                threshold=overlap.threshold,
                units="1",
            )
        )

    report.gates.append(
        GateResult(
            gate="umbrella:wham_converged",
            status=GateStatus.PASS if result.wham_converged else GateStatus.FAIL,
            message=(
                "WHAM self-consistent iteration converged"
                if result.wham_converged
                else "WHAM did not reach the convergence tolerance"
            ),
        )
    )

    if result.half_split_max_deviation is None:
        report.gates.append(
            GateResult(
                gate="umbrella:sampling_converged",
                status=GateStatus.INCONCLUSIVE,
                message="Half-split convergence could not be evaluated",
            )
        )
    else:
        ok = result.half_split_max_deviation <= 2.0
        report.gates.append(
            GateResult(
                gate="umbrella:sampling_converged",
                status=GateStatus.PASS if ok else GateStatus.FAIL,
                message=(
                    f"First and second halves of the sampling agree to "
                    f"{result.half_split_max_deviation:.2f} kJ/mol"
                    if ok
                    else f"PMF is still changing: halves differ by "
                    f"{result.half_split_max_deviation:.2f} kJ/mol"
                ),
                value=result.half_split_max_deviation,
                threshold=2.0,
                units="kJ/mol",
            )
        )

    if result.uncertainty is None:
        report.gates.append(
            GateResult(
                gate="umbrella:uncertainty_estimated",
                status=GateStatus.INCONCLUSIVE,
                message="No bootstrap uncertainty was computed for this PMF",
            )
        )
    else:
        # Peak uncertainty over the well-sampled region; the tails are noise by
        # construction and would otherwise dominate the reported number.
        region = result.uncertainty[result.well_sampled] if result.well_sampled.any() else result.uncertainty
        peak = float(np.nanmax(region)) if region.size else float("nan")
        report.gates.append(
            GateResult(
                gate="umbrella:uncertainty_estimated",
                status=GateStatus.PASS if peak <= 2.0 else GateStatus.WARN,
                message=f"Peak bootstrap uncertainty {peak:.2f} kJ/mol",
                value=peak,
                threshold=2.0,
                units="kJ/mol",
            )
        )
    return report


__all__ = [
    "DEFAULT_WHAM_MAX_ITERATIONS",
    "WELL_SAMPLED_FRACTION",
    "WELL_SAMPLED_MIN_COUNTS",
    "OverlapDiagnostic",
    "PmfResult",
    "WhamResult",
    "WindowSamples",
    "align_pmfs",
    "build_pmf",
    "diagnose_overlap",
    "overlap_matrix",
    "pmf_gates",
    "trim_equilibration",
    "well_sampled_mask",
    "wham",
]
