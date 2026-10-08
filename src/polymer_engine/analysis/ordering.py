"""Telling a slowly-equilibrating melt apart from one that is crystallising.

Both look identical to a drift check: the density climbs and never settles, so the
observable is reported as "still drifting" and the run is written off as needing more
time. More time does not help, because they are not the same problem.

A melt that has not finished relaxing is heading *towards* a stationary state, and its
potential energy fluctuates about a mean while it gets there. A melt that is ordering is
heading *away* from the amorphous state altogether: density rises **and** potential
energy falls, because packing chains into register releases energy. That second
signature is what distinguishes them, and it costs nothing extra to look at -- the
energy is already in the ``.edr``.

The distinction matters because the two have opposite remedies. Slow equilibration wants
a longer run. Crystallisation wants a different question: at 300 K a C60 alkane sits
roughly 70 K below its melting point, so an amorphous melt is not its equilibrium state
and no amount of simulation will make it one.

This module reports the signature. It does not claim to have observed a crystal -- that
would need a structural order parameter, and a monotonic energy drop is consistent with
several kinds of ordering. What it establishes is that the system is *leaving* the
amorphous state rather than settling into it, which is enough to stop reporting the run
as under-sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination, GateReport, GateResult, GateStatus

logger = get_logger("analysis.ordering")

#: A trend has to be *both* big enough to matter and bigger than the noise, and neither
#: test alone is sufficient here.
#:
#: R-squared is the wrong criterion on its own: a real density rise buried in
#: fluctuations of 11 kg/m^3 explains only 13% of the variance, and a threshold on R^2
#: strict enough to exclude noise also excludes that. Statistical significance alone is
#: no better in the other direction: with 26000 effective samples a drift of 0.4% is
#: significant at 24 sigma while being physically irrelevant.
#:
#: So: the observable must move by at least this fraction of its mean across the window,
MIN_RELATIVE_CHANGE = 0.005
#: and that movement must exceed this many standard errors of the mean, computed with an
#: autocorrelation-corrected sample count rather than the raw frame count.
MIN_SIGNIFICANCE = 3.0

#: Potential energy must fall by at least this fraction of its own spread across the
#: window, and the fall must be significant. Relative to the spread because absolute
#: energies scale with system size and would otherwise need a per-system threshold.
#:
#: The value sits in a gap measured across three polyolefins, nine replicas: systems that
#: had equilibrated moved their energy by at most 0.19 sd, while the ones densifying with
#: an energy drop moved 0.48 to 0.99. Placing the cut between those is a choice made from
#: data rather than from theory, and it is worth re-checking on chemistry unlike these --
#: a borderline case should be read as "look at this run", not as a verdict.
MIN_ENERGY_DROP_SD = 0.25


class OrderingVerdict(str, Enum):
    """What the density and energy trends together say about the state."""

    #: Neither density nor energy is trending: the system is where it is going to stay.
    STATIONARY = "STATIONARY"
    #: Density rising while energy falls. The system is leaving the amorphous state.
    ORDERING = "ORDERING"
    #: Density still moving but energy is not: relaxation that has not finished.
    EQUILIBRATING = "EQUILIBRATING"
    #: Not enough data, or the series disagree in a way none of the above describes.
    INCONCLUSIVE = "INCONCLUSIVE"

    @property
    def more_time_would_help(self) -> bool:
        """Whether a longer run is the right response."""
        return self is OrderingVerdict.EQUILIBRATING


@dataclass
class Trend:
    """A least-squares trend in one series, with both tests applied."""

    slope_per_ns: float
    relative_slope_per_ns: float
    #: Total change across the window, as a fraction of the mean.
    relative_change: float
    #: That change in units of the standard error of the mean.
    significance: float
    r_squared: float
    mean: float
    std: float
    n: int
    n_effective: float

    @property
    def systematic(self) -> bool:
        """Large enough to matter, and larger than the noise."""
        return (abs(self.relative_change) >= MIN_RELATIVE_CHANGE
                and abs(self.significance) >= MIN_SIGNIFICANCE)

    def as_dict(self) -> dict[str, Any]:
        return {"slope_per_ns": self.slope_per_ns,
                "relative_slope_per_ns": self.relative_slope_per_ns,
                "relative_change": self.relative_change,
                "significance": self.significance, "systematic": self.systematic,
                "r_squared": self.r_squared, "mean": self.mean, "std": self.std,
                "n": self.n, "n_effective": self.n_effective}


def trend(times_ps: Any, values: Any, *, scale: str = "mean") -> Trend | None:
    """Least-squares slope per nanosecond, relative to the mean or the spread.

    Returns ``None`` rather than a slope when there is too little to fit, or when the
    series is constant -- a zero-variance series has no trend to report and dividing by
    its spread would invent one.
    """
    t = np.asarray(times_ps, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.size < 8 or t.size != y.size:
        return None
    finite = np.isfinite(t) & np.isfinite(y)
    t, y = t[finite], y[finite]
    if t.size < 8 or float(t[-1] - t[0]) <= 0:
        return None

    slope_per_ps, intercept = np.polyfit(t, y, 1)
    slope_per_ns = float(slope_per_ps) * 1000.0
    mean = float(y.mean())
    std = float(y.std(ddof=1)) if y.size > 1 else 0.0

    residual = y - (slope_per_ps * t + intercept)
    total = float(((y - mean) ** 2).sum())
    r_squared = 0.0 if total <= 0 else float(1.0 - (residual**2).sum() / total)

    denominator = abs(mean) if scale == "mean" else std
    if denominator <= 0:
        return None

    window_ns = float(t[-1] - t[0]) / 1000.0
    change = slope_per_ns * window_ns

    # Correlated frames are not independent samples, so the standard error uses the
    # effective count. Without this the significance of a drift in a well-sampled run is
    # overstated by the square root of the statistical inefficiency -- a factor of 50 in
    # the series this was built for.
    from polymer_engine.analysis.statistics import effective_sample_size

    try:
        n_eff = float(effective_sample_size(y))
    except Exception:  # noqa: BLE001 - a degenerate series is not a reason to fail here
        n_eff = float(y.size)
    n_eff = max(n_eff, 1.0)
    standard_error = std / np.sqrt(n_eff) if std > 0 else 0.0
    significance = float(change / standard_error) if standard_error > 0 else 0.0

    return Trend(slope_per_ns=slope_per_ns,
                 relative_slope_per_ns=slope_per_ns / denominator,
                 relative_change=float(change / denominator),
                 significance=significance, r_squared=r_squared,
                 mean=mean, std=std, n=int(t.size), n_effective=n_eff)


@dataclass
class OrderingAnalysis:
    """The verdict, and the two trends behind it."""

    verdict: OrderingVerdict
    density: Trend | None
    energy: Trend | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict.value, "reason": self.reason,
                "more_time_would_help": self.verdict.more_time_would_help,
                "density": self.density.as_dict() if self.density else None,
                "energy": self.energy.as_dict() if self.energy else None}


def analyse(
    times_ps: Any, density: Any, potential_energy: Any | None = None,
) -> OrderingAnalysis:
    """Classify what a density and energy series are doing together.

    Pass the **whole** production trajectory, not the post-equilibration window the
    sampling gates use. Those gates ask "can this window be averaged?", which is a
    question about the window. This asks "what is this trajectory doing?", and a system
    that did most of its ordering early looks stationary if the early part is discarded
    -- which is precisely the run that most needs the diagnosis.
    """
    rho = trend(times_ps, density, scale="mean")
    if rho is None:
        return OrderingAnalysis(OrderingVerdict.INCONCLUSIVE, None, None,
                                "too few density samples to fit a trend")
    energy = (trend(times_ps, potential_energy, scale="std")
              if potential_energy is not None else None)

    rising = rho.systematic and rho.relative_change > 0
    if not rising:
        return OrderingAnalysis(
            OrderingVerdict.STATIONARY, rho, energy,
            f"density is not systematically rising: {rho.relative_change * 100:+.2f}% "
            f"across the window at {rho.significance:+.1f} standard errors "
            f"(needs {MIN_RELATIVE_CHANGE * 100:g}% and "
            f"{MIN_SIGNIFICANCE:g} sigma)")

    if energy is None:
        return OrderingAnalysis(
            OrderingVerdict.INCONCLUSIVE, rho, energy,
            "density is rising, but without a potential-energy series there is no way "
            "to tell ordering from unfinished relaxation")

    falling = (energy.relative_change <= -MIN_ENERGY_DROP_SD
               and abs(energy.significance) >= MIN_SIGNIFICANCE)
    if falling:
        return OrderingAnalysis(
            OrderingVerdict.ORDERING, rho, energy,
            f"density rose {rho.relative_change * 100:+.2f}% while potential energy "
            f"fell {energy.relative_change:+.2f} standard deviations across the same "
            f"window; the system is leaving the amorphous state rather than settling "
            f"into it, and a longer run will not change that")
    return OrderingAnalysis(
        OrderingVerdict.EQUILIBRATING, rho, energy,
        f"density rose {rho.relative_change * 100:+.2f}% with potential energy flat "
        f"({energy.relative_change:+.2f} sd); this is relaxation that has not finished")


def ordering_gates(analysis: OrderingAnalysis) -> GateReport:
    """Turn the verdict into a gate.

    ``ORDERING`` is a **FAIL**, not an INCONCLUSIVE: it is a positive finding that the
    requested state is not the one being simulated, which is a different and more useful
    thing to report than an absence of data.
    """
    report = GateReport(name="ordering")
    status = {
        OrderingVerdict.STATIONARY: GateStatus.PASS,
        OrderingVerdict.EQUILIBRATING: GateStatus.INCONCLUSIVE,
        OrderingVerdict.ORDERING: GateStatus.FAIL,
        OrderingVerdict.INCONCLUSIVE: GateStatus.INCONCLUSIVE,
    }[analysis.verdict]
    report.gates.append(GateResult(
        gate="amorphous_state_retained", status=status, message=analysis.reason,
        value=analysis.density.relative_change if analysis.density else None,
        threshold=MIN_RELATIVE_CHANGE,
        evidence=analysis.as_dict(),
    ))
    return report


def determination_for(analysis: OrderingAnalysis) -> Determination:
    if analysis.verdict is OrderingVerdict.ORDERING:
        # Not INSUFFICIENT_DATA: more data is not what is missing.
        return Determination.REQUIRES_VALIDATION
    if analysis.verdict is OrderingVerdict.STATIONARY:
        return Determination.KNOWN
    return Determination.INSUFFICIENT_DATA


__all__ = [
    "MIN_ENERGY_DROP_SD", "MIN_RELATIVE_CHANGE", "MIN_SIGNIFICANCE",
    "OrderingAnalysis", "OrderingVerdict", "Trend", "analyse", "determination_for",
    "ordering_gates", "trend",
]
