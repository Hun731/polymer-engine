"""Transport and dynamical properties.

The organising principle: **do not assume normal diffusion.**  Polymer melts are
subdiffusive over the Rouse regime, a tracer at short times is ballistic, and a chain
in a glass may not diffuse at all on a simulated timescale.  Fitting a straight line to
an MSD and calling the slope a diffusion coefficient produces a number in every one of
those cases, and it is wrong in all but one.

:func:`classify_regime` therefore fits the scaling exponent ``alpha`` in
``MSD ~ t^alpha`` on log-log axes and reports the regime.  A diffusion coefficient is
issued only when ``alpha ~ 1``; otherwise the calculator says which regime it found and
declines.

Relaxation times come from autocorrelation functions and carry the same caution: a
correlation function that has not decayed over the simulated window has a relaxation
time *longer than the simulation*, which is a bound, not a measurement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from polymer_engine.core.config import AnalysisDefaults
from polymer_engine.core.models import Determination, GateReport, GateResult, GateStatus, Measurement
from polymer_engine.properties.base import (
    PropertyCalculator,
    PropertyClass,
    PropertyDefinition,
    PropertyResult,
    SamplingRequirement,
    UncertaintyMethod,
)


class DiffusionRegime(str, Enum):
    BALLISTIC = "ballistic"
    SUBDIFFUSIVE = "subdiffusive"
    DIFFUSIVE = "diffusive"
    SUPERDIFFUSIVE = "superdiffusive"
    UNDETERMINED = "undetermined"


#: How far ``alpha`` may stray from 1 and still count as normal diffusion.
DIFFUSIVE_TOLERANCE = 0.15
#: Above this, motion is ballistic rather than merely superdiffusive.
BALLISTIC_THRESHOLD = 1.7


@dataclass(frozen=True, slots=True)
class RegimeDiagnosis:
    """What kind of motion the MSD actually shows."""

    regime: DiffusionRegime
    alpha: float | None
    alpha_error: float | None
    r_squared: float | None
    fit_range_ps: tuple[float, float] | None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "alpha": self.alpha,
            "alpha_error": self.alpha_error,
            "r_squared": self.r_squared,
            "fit_range_ps": list(self.fit_range_ps) if self.fit_range_ps else None,
            "note": self.note,
        }


def classify_regime(
    lag_ps: Sequence[float] | np.ndarray,
    msd_nm2: Sequence[float] | np.ndarray,
    *,
    fit_fraction: tuple[float, float] = (0.2, 0.6),
) -> RegimeDiagnosis:
    """Determine the diffusion regime from the log-log slope of the MSD.

    The exponent, not the fit quality, is the decisive test.  A quadratic MSD fits a
    straight line with R^2 near 0.99 over a narrow window, so an R^2 test alone would
    report a diffusion coefficient for a particle moving at constant velocity.
    """
    lags = np.asarray(lag_ps, dtype=float).ravel()
    msd = np.asarray(msd_nm2, dtype=float).ravel()
    if lags.size != msd.size:
        return RegimeDiagnosis(
            DiffusionRegime.UNDETERMINED, None, None, None, None,
            "lag and MSD arrays have different lengths",
        )
    if lags.size < 10:
        return RegimeDiagnosis(
            DiffusionRegime.UNDETERMINED, None, None, None, None,
            f"only {lags.size} lag times; at least 10 are needed to fit an exponent",
        )

    low = int(lags.size * fit_fraction[0])
    high = int(lags.size * fit_fraction[1])
    if high - low < 5:
        return RegimeDiagnosis(
            DiffusionRegime.UNDETERMINED, None, None, None, None, "fit window is too narrow"
        )

    t = lags[low:high]
    y = msd[low:high]
    mask = (t > 0) & (y > 0) & np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 5:
        return RegimeDiagnosis(
            DiffusionRegime.UNDETERMINED, None, None, None, None,
            "too few positive MSD values in the fit window",
        )
    t, y = t[mask], y[mask]

    log_t, log_y = np.log(t), np.log(y)
    design = np.column_stack([np.ones_like(log_t), log_t])
    coefficients, *_ = np.linalg.lstsq(design, log_y, rcond=None)
    alpha = float(coefficients[1])

    predicted = design @ coefficients
    ss_res = float(((log_y - predicted) ** 2).sum())
    ss_tot = float(((log_y - log_y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    dof = max(1, log_t.size - 2)
    centred = log_t - log_t.mean()
    denominator = float((centred**2).sum())
    alpha_error = math.sqrt(ss_res / dof / denominator) if denominator > 0 else None

    if abs(alpha - 1.0) <= DIFFUSIVE_TOLERANCE:
        regime = DiffusionRegime.DIFFUSIVE
        note = "MSD grows linearly with time; the Einstein relation applies"
    elif alpha >= BALLISTIC_THRESHOLD:
        regime = DiffusionRegime.BALLISTIC
        note = "MSD grows roughly as t^2: the motion is directed, not diffusive"
    elif alpha > 1.0 + DIFFUSIVE_TOLERANCE:
        regime = DiffusionRegime.SUPERDIFFUSIVE
        note = "MSD grows faster than linearly; the diffusive regime has not been reached"
    else:
        regime = DiffusionRegime.SUBDIFFUSIVE
        note = (
            "MSD grows more slowly than linearly, as expected for a polymer melt in the "
            "Rouse or reptation regime; a diffusion coefficient from this window would be wrong"
        )

    return RegimeDiagnosis(
        regime=regime,
        alpha=alpha,
        alpha_error=alpha_error,
        r_squared=r_squared,
        fit_range_ps=(float(t[0]), float(t[-1])),
        note=note,
    )


class MeanSquaredDisplacement(PropertyCalculator):
    """The MSD itself, with its regime diagnosis attached."""

    definition = PropertyDefinition(
        name="mean_squared_displacement",
        property_class=PropertyClass.TRANSPORT,
        units="nm^2",
        observable="Time-averaged squared displacement of the selected atoms",
        estimator="All time origins, averaged over atoms",
        uncertainty_method=UncertaintyMethod.NONE,
        sampling=SamplingRequirement(
            min_effective_samples=10.0, min_replicas=3, min_simulation_ns=10.0,
            notes="Needs a window long enough to reach whichever regime is being studied.",
        ),
        description="MSD curve against lag time in ps.",
        caveats="Requires an unwrapped trajectory: periodic wrapping truncates displacements "
                "and flattens the MSD.",
    )

    def compute(
        self,
        lag_ps: Sequence[float] | np.ndarray,
        msd_nm2: Sequence[float] | np.ndarray,
        *,
        n_replicas: int = 1,
        provenance: dict[str, Any] | None = None,
    ) -> PropertyResult:
        lags = np.asarray(lag_ps, dtype=float).ravel()
        msd = np.asarray(msd_nm2, dtype=float).ravel()
        if lags.size < 4:
            return self.unknown("at least four lag times are required")

        diagnosis = classify_regime(lags, msd)
        report = GateReport(name=f"property:{self.definition.name}")
        report.gates.append(
            GateResult(
                gate=f"{self.definition.name}:regime",
                status=GateStatus.PASS if diagnosis.regime is not DiffusionRegime.UNDETERMINED
                else GateStatus.INCONCLUSIVE,
                message=f"{diagnosis.regime.value}: {diagnosis.note}",
                value=diagnosis.alpha,
                units="1",
                evidence=diagnosis.as_dict(),
            )
        )
        measurement = Measurement(
            name=self.definition.name,
            value=float(msd[-1]),
            units=self.definition.units,
            n_samples=int(lags.size),
            effective_samples=float(lags.size),
            method="MSD at the longest lag time",
        )
        report.gates.extend(
            self.sampling_gates(measurement, n_replicas=n_replicas, equilibration_shown=True)
        )
        return PropertyResult(
            definition=self.definition,
            measurement=measurement,
            report=report,
            n_replicas=n_replicas,
            provenance={"regime": diagnosis.as_dict(), **(provenance or {})},
            extra={"regime": diagnosis.regime.value, "alpha": diagnosis.alpha},
        )


class DiffusionCoefficient(PropertyCalculator):
    """Einstein diffusion coefficient, issued only for genuinely diffusive motion."""

    definition = PropertyDefinition(
        name="diffusion_coefficient",
        property_class=PropertyClass.TRANSPORT,
        units="nm^2/ps",
        observable="Long-time slope of the mean squared displacement",
        estimator="Einstein relation D = slope / (2 * dimensionality), fitted in the diffusive regime",
        uncertainty_method=UncertaintyMethod.FIT_COVARIANCE,
        sampling=SamplingRequirement(
            min_effective_samples=10.0,
            min_replicas=3,
            min_simulation_ns=50.0,
            notes="The diffusive regime must actually be reached; for a polymer melt that "
                  "can require hundreds of nanoseconds.",
        ),
        description="Self-diffusion coefficient.",
        caveats="Finite-size effects on D in a periodic box are substantial and are not "
                "corrected here; the Yeh-Hummer correction needs the shear viscosity.",
    )

    def compute(
        self,
        lag_ps: Sequence[float] | np.ndarray,
        msd_nm2: Sequence[float] | np.ndarray,
        *,
        dimensionality: int = 3,
        fit_fraction: tuple[float, float] = (0.2, 0.6),
        n_replicas: int = 1,
        simulation_ns: float | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> PropertyResult:
        lags = np.asarray(lag_ps, dtype=float).ravel()
        msd = np.asarray(msd_nm2, dtype=float).ravel()
        if dimensionality not in (1, 2, 3):
            return self.unknown("dimensionality must be 1, 2 or 3", determination=Determination.UNKNOWN)

        diagnosis = classify_regime(lags, msd, fit_fraction=fit_fraction)
        report = GateReport(name=f"property:{self.definition.name}")
        report.gates.append(
            GateResult(
                gate=f"{self.definition.name}:diffusive_regime",
                status=GateStatus.PASS if diagnosis.regime is DiffusionRegime.DIFFUSIVE
                else GateStatus.FAIL,
                message=f"{diagnosis.regime.value}: {diagnosis.note}",
                value=diagnosis.alpha,
                threshold=1.0,
                units="1",
                evidence=diagnosis.as_dict(),
            )
        )

        if diagnosis.regime is not DiffusionRegime.DIFFUSIVE:
            return PropertyResult(
                definition=self.definition,
                measurement=Measurement.unknown(
                    self.definition.name, units=self.definition.units,
                    reason=(
                        f"motion is {diagnosis.regime.value}"
                        + (f" (MSD ~ t^{diagnosis.alpha:.2f})" if diagnosis.alpha else "")
                        + "; the Einstein relation does not apply"
                    ),
                    determination=Determination.INSUFFICIENT_DATA,
                ),
                report=report,
                n_replicas=n_replicas,
                diagnostics=[diagnosis.note],
                provenance={"regime": diagnosis.as_dict(), **(provenance or {})},
                extra={"regime": diagnosis.regime.value, "alpha": diagnosis.alpha},
            )

        low = int(lags.size * fit_fraction[0])
        high = int(lags.size * fit_fraction[1])
        t, y = lags[low:high], msd[low:high]
        design = np.column_stack([np.ones_like(t), t])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        slope = float(coefficients[1])

        predicted = design @ coefficients
        ss_res = float(((y - predicted) ** 2).sum())
        dof = max(1, t.size - 2)
        centred = t - t.mean()
        denominator = float((centred**2).sum())
        slope_error = math.sqrt(ss_res / dof / denominator) if denominator > 0 else None

        if slope <= 0:
            report.gates.append(
                GateResult(
                    gate=f"{self.definition.name}:positive_slope",
                    status=GateStatus.FAIL,
                    message="the fitted MSD slope is not positive",
                    value=slope,
                )
            )
            return PropertyResult(
                definition=self.definition,
                measurement=Measurement.unknown(
                    self.definition.name, units=self.definition.units,
                    reason="non-positive MSD slope",
                    determination=Determination.INSUFFICIENT_DATA,
                ),
                report=report,
                n_replicas=n_replicas,
            )

        divisor = 2.0 * dimensionality
        measurement = Measurement(
            name=self.definition.name,
            value=slope / divisor,
            uncertainty=(slope_error / divisor) if slope_error else None,
            units=self.definition.units,
            n_samples=int(t.size),
            effective_samples=float(t.size),
            method=f"Einstein relation in {dimensionality}D; MSD ~ t^{diagnosis.alpha:.2f}",
            notes="Not corrected for periodic finite-size effects.",
        )
        report.gates.extend(
            self.sampling_gates(
                measurement, n_replicas=n_replicas, equilibration_shown=True,
                simulation_ns=simulation_ns,
            )
        )
        return PropertyResult(
            definition=self.definition,
            measurement=measurement,
            report=report,
            n_replicas=n_replicas,
            provenance={
                "regime": diagnosis.as_dict(),
                "slope_nm2_per_ps": slope,
                "dimensionality": dimensionality,
                **(provenance or {}),
            },
            extra={"regime": diagnosis.regime.value, "alpha": diagnosis.alpha},
        )


class RelaxationTime(PropertyCalculator):
    """Relaxation time from the decay of an autocorrelation function.

    When the correlation function has not decayed below ``exp(-1)`` within the
    simulated window, the relaxation time is *longer than the simulation*.  That is a
    lower bound, and it is reported as one rather than as a fitted value.
    """

    definition = PropertyDefinition(
        name="relaxation_time",
        property_class=PropertyClass.DYNAMICAL,
        units="ps",
        observable="Decay of a normalised autocorrelation function",
        estimator="Integral of the correlation function up to its first zero crossing",
        uncertainty_method=UncertaintyMethod.BLOCK_BOOTSTRAP,
        sampling=SamplingRequirement(
            min_effective_samples=10.0, min_replicas=3,
            notes="The simulation must be several relaxation times long for the integral to converge.",
        ),
        description="Characteristic relaxation time.",
        caveats="A correlation function that has not decayed gives a bound, not a value.",
    )

    def compute(
        self,
        times_ps: Sequence[float] | np.ndarray,
        correlation: Sequence[float] | np.ndarray,
        *,
        n_replicas: int = 1,
        provenance: dict[str, Any] | None = None,
    ) -> PropertyResult:
        t = np.asarray(times_ps, dtype=float).ravel()
        c = np.asarray(correlation, dtype=float).ravel()
        if t.size != c.size:
            return self.unknown("time and correlation arrays differ in length",
                                determination=Determination.UNKNOWN)
        if t.size < 5:
            return self.unknown("at least five points are needed")
        if not np.all(np.isfinite(c)):
            return self.unknown("correlation contains non-finite values",
                                determination=Determination.UNKNOWN)

        # Normalise so the integral has the units of time.
        if c[0] == 0:
            return self.unknown("correlation at zero lag is zero", determination=Determination.UNKNOWN)
        normalised = c / c[0]

        report = GateReport(name=f"property:{self.definition.name}")
        decayed = np.flatnonzero(normalised <= math.exp(-1.0))
        if decayed.size == 0:
            bound = float(t[-1])
            report.gates.append(
                GateResult(
                    gate=f"{self.definition.name}:decayed",
                    status=GateStatus.FAIL,
                    message=(
                        f"the correlation function has not decayed to 1/e within the simulated "
                        f"{bound:.1f} ps; the relaxation time exceeds the simulation length"
                    ),
                    value=float(normalised[-1]),
                    threshold=math.exp(-1.0),
                    units="1",
                )
            )
            return PropertyResult(
                definition=self.definition,
                measurement=Measurement.unknown(
                    self.definition.name, units="ps",
                    reason=f"relaxation time exceeds the {bound:.1f} ps simulated window",
                    determination=Determination.INSUFFICIENT_DATA,
                ),
                report=report,
                n_replicas=n_replicas,
                extra={"lower_bound_ps": bound},
            )

        # Integrate to the first zero crossing; beyond it the tail is noise.
        crossing = np.flatnonzero(normalised <= 0.0)
        cutoff = int(crossing[0]) if crossing.size else normalised.size
        tau = float(np.trapezoid(normalised[:cutoff], t[:cutoff]))

        report.gates.append(
            GateResult(
                gate=f"{self.definition.name}:decayed",
                status=GateStatus.PASS,
                message=f"correlation decayed below 1/e after {t[decayed[0]]:.2f} ps",
                value=float(t[decayed[0]]),
                units="ps",
            )
        )
        window_ratio = float(t[-1]) / tau if tau > 0 else 0.0
        report.gates.append(
            GateResult(
                gate=f"{self.definition.name}:window_length",
                status=GateStatus.PASS if window_ratio >= 5.0 else GateStatus.WARN,
                message=f"the window spans {window_ratio:.1f} relaxation times",
                value=window_ratio,
                threshold=5.0,
                units="1",
            )
        )
        measurement = Measurement(
            name=self.definition.name,
            value=tau,
            units="ps",
            n_samples=int(cutoff),
            effective_samples=float(cutoff),
            method="integral of the normalised correlation to its first zero crossing",
        )
        report.gates.extend(
            self.sampling_gates(measurement, n_replicas=n_replicas, equilibration_shown=True)
        )
        return PropertyResult(
            definition=self.definition,
            measurement=measurement,
            report=report,
            n_replicas=n_replicas,
            provenance={"points_integrated": int(cutoff), **(provenance or {})},
        )


def transport_calculators(defaults: AnalysisDefaults | None = None) -> list[PropertyCalculator]:
    return [
        MeanSquaredDisplacement(defaults), DiffusionCoefficient(defaults), RelaxationTime(defaults),
    ]


__all__ = [
    "BALLISTIC_THRESHOLD",
    "DIFFUSIVE_TOLERANCE",
    "DiffusionCoefficient",
    "DiffusionRegime",
    "MeanSquaredDisplacement",
    "RegimeDiagnosis",
    "RelaxationTime",
    "classify_regime",
    "transport_calculators",
]
