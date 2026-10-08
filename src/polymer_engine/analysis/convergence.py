"""Convergence and replica-agreement gates.

"GROMACS exited 0" is not a scientific result.  These gates ask the questions that
actually determine whether a number can be believed:

* Has the observable stopped drifting?
* Are there enough *independent* samples behind it?
* Do independent replicas agree, given their own uncertainties?

Every check produces a value, an uncertainty, a threshold, a verdict, and a
diagnostic message.  The verdict has four states and ``INCONCLUSIVE`` is a real one:
a single replica cannot demonstrate reproducibility, and reporting that as ``PASS``
would be the most consequential lie the engine could tell.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polymer_engine.analysis.statistics import (
    EquilibrationResult,
    describe,
    detect_equilibration,
    effective_sample_size,
    statistical_inefficiency,
)
from polymer_engine.core.config import AnalysisDefaults
from polymer_engine.core.models import Determination, GateReport, GateResult, GateStatus, Measurement


@dataclass
class SeriesAnalysis:
    """Everything we determined about one observable's time series."""

    name: str
    units: str
    n_raw: int
    equilibration: EquilibrationResult
    production: Measurement
    drift_fraction: float | None
    drift_per_ns: float | None
    first_half: Measurement
    second_half: Measurement

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "units": self.units,
            "n_raw": self.n_raw,
            "equilibration": self.equilibration.as_dict(),
            "production": self.production.model_dump(mode="json"),
            "drift_fraction": self.drift_fraction,
            "drift_per_ns": self.drift_per_ns,
            "first_half": self.first_half.model_dump(mode="json"),
            "second_half": self.second_half.model_dump(mode="json"),
        }


def analyse_series(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
    units: str = "1",
    times_ps: Sequence[float] | np.ndarray | None = None,
    defaults: AnalysisDefaults | None = None,
) -> SeriesAnalysis:
    """Detect equilibration, then characterise the production portion."""
    defaults = defaults or AnalysisDefaults()
    x = np.asarray(values, dtype=float).ravel()
    n_raw = int(x.size)

    if defaults.equilibration_detection == "fraction":
        start = int(n_raw * defaults.discard_fraction)
        equilibration = EquilibrationResult(
            start_index=start,
            n_discarded=start,
            n_retained=n_raw - start,
            effective_samples=effective_sample_size(x[start:]) if n_raw > start else 0.0,
            statistical_inefficiency=statistical_inefficiency(x[start:]) if n_raw > start else 1.0,
            method=f"fixed discard of {defaults.discard_fraction:.0%}",
        )
    else:
        equilibration = detect_equilibration(x)

    production_values = x[equilibration.start_index :]
    production = describe(production_values, name=name, units=units)

    drift_fraction: float | None = None
    drift_per_ns: float | None = None
    if production_values.size >= 4:
        halves = np.array_split(production_values, 2)
        first = describe(halves[0], name=f"{name}_first_half", units=units)
        second = describe(halves[1], name=f"{name}_second_half", units=units)
        if production.value not in (None, 0.0) and first.value is not None and second.value is not None:
            drift_fraction = abs(second.value - first.value) / abs(production.value)
        if times_ps is not None:
            t = np.asarray(times_ps, dtype=float).ravel()[equilibration.start_index :]
            if t.size == production_values.size and t.size >= 2:
                span_ns = (float(t[-1]) - float(t[0])) / 1000.0
                if span_ns > 0:
                    slope = float(np.polyfit(t, production_values, 1)[0])  # per ps
                    drift_per_ns = slope * 1000.0
    else:
        first = Measurement.unknown(
            f"{name}_first_half", units=units, reason="too few production samples",
            determination=Determination.INSUFFICIENT_DATA,
        )
        second = Measurement.unknown(
            f"{name}_second_half", units=units, reason="too few production samples",
            determination=Determination.INSUFFICIENT_DATA,
        )

    return SeriesAnalysis(
        name=name,
        units=units,
        n_raw=n_raw,
        equilibration=equilibration,
        production=production,
        drift_fraction=drift_fraction,
        drift_per_ns=drift_per_ns,
        first_half=first,
        second_half=second,
    )


def convergence_gates(
    analysis: SeriesAnalysis, *, defaults: AnalysisDefaults | None = None
) -> list[GateResult]:
    """Turn a series analysis into pass/warn/fail/inconclusive gates."""
    defaults = defaults or AnalysisDefaults()
    gates: list[GateResult] = []
    name = analysis.name

    # -- enough independent samples? -----------------------------------
    ess = analysis.production.effective_samples
    if ess is None:
        gates.append(
            GateResult(
                gate=f"{name}:effective_samples",
                status=GateStatus.INCONCLUSIVE,
                message="Effective sample size could not be determined",
            )
        )
    elif ess < defaults.min_effective_samples:
        gates.append(
            GateResult(
                gate=f"{name}:effective_samples",
                status=GateStatus.FAIL,
                message=(
                    f"Only {ess:.1f} effective samples after accounting for autocorrelation "
                    f"({analysis.production.n_samples} raw frames); need {defaults.min_effective_samples:.0f}"
                ),
                value=float(ess),
                threshold=defaults.min_effective_samples,
                units="1",
                evidence={"n_raw_frames": analysis.production.n_samples},
            )
        )
    else:
        gates.append(
            GateResult(
                gate=f"{name}:effective_samples",
                status=GateStatus.PASS,
                message=(
                    f"{ess:.1f} effective samples from {analysis.production.n_samples} frames "
                    f"(statistical inefficiency {analysis.equilibration.statistical_inefficiency:.1f})"
                ),
                value=float(ess),
                threshold=defaults.min_effective_samples,
                units="1",
            )
        )

    # -- relative precision --------------------------------------------
    relative = analysis.production.relative_uncertainty
    if relative is None:
        gates.append(
            GateResult(
                gate=f"{name}:relative_uncertainty",
                status=GateStatus.INCONCLUSIVE,
                message="Relative uncertainty is undefined (zero mean or no uncertainty estimate)",
            )
        )
    else:
        status = GateStatus.PASS if relative <= defaults.max_relative_stderr else GateStatus.WARN
        gates.append(
            GateResult(
                gate=f"{name}:relative_uncertainty",
                status=status,
                message=f"Relative standard error {relative:.3%} (threshold {defaults.max_relative_stderr:.3%})",
                value=float(relative),
                uncertainty=analysis.production.uncertainty,
                threshold=defaults.max_relative_stderr,
                units="1",
            )
        )

    # -- drift ----------------------------------------------------------
    if analysis.drift_fraction is None:
        gates.append(
            GateResult(
                gate=f"{name}:drift",
                status=GateStatus.INCONCLUSIVE,
                message="Drift could not be assessed (too few production samples)",
            )
        )
    else:
        # Drift only counts as real if it exceeds the noise on the two half-means.
        noise = _half_noise(analysis)
        significant = noise is not None and analysis.drift_fraction is not None and _drift_significant(analysis, noise)
        if analysis.drift_fraction <= defaults.max_drift_fraction:
            status = GateStatus.PASS
            message = f"No meaningful drift: half-to-half change {analysis.drift_fraction:.3%}"
        elif significant:
            status = GateStatus.FAIL
            message = (
                f"Observable is still drifting: half-to-half change {analysis.drift_fraction:.3%} "
                f"exceeds both the {defaults.max_drift_fraction:.3%} threshold and the statistical noise"
            )
        else:
            status = GateStatus.WARN
            message = (
                f"Half-to-half change {analysis.drift_fraction:.3%} exceeds the threshold but is within "
                "statistical noise; more sampling would settle it"
            )
        gates.append(
            GateResult(
                gate=f"{name}:drift",
                status=status,
                message=message,
                value=float(analysis.drift_fraction),
                threshold=defaults.max_drift_fraction,
                units="1",
                evidence={
                    "first_half": analysis.first_half.value,
                    "second_half": analysis.second_half.value,
                    "drift_per_ns": analysis.drift_per_ns,
                },
            )
        )

    # -- equilibration cost ---------------------------------------------
    if analysis.equilibration.discarded_fraction > 0.6:
        gates.append(
            GateResult(
                gate=f"{name}:equilibration",
                status=GateStatus.WARN,
                message=(
                    f"{analysis.equilibration.discarded_fraction:.0%} of the trajectory was discarded as "
                    "unequilibrated; the production window is short relative to the run"
                ),
                value=analysis.equilibration.discarded_fraction,
                threshold=0.6,
                units="1",
            )
        )
    else:
        gates.append(
            GateResult(
                gate=f"{name}:equilibration",
                status=GateStatus.PASS,
                message=(
                    f"Discarded {analysis.equilibration.n_discarded} of {analysis.n_raw} frames "
                    f"({analysis.equilibration.method})"
                ),
                value=analysis.equilibration.discarded_fraction,
                units="1",
            )
        )
    return gates


def _half_noise(analysis: SeriesAnalysis) -> float | None:
    a, b = analysis.first_half.uncertainty, analysis.second_half.uncertainty
    if a is None or b is None:
        return None
    return math.sqrt(a * a + b * b)


def _drift_significant(analysis: SeriesAnalysis, noise: float) -> bool:
    if analysis.first_half.value is None or analysis.second_half.value is None or noise <= 0:
        return False
    return abs(analysis.second_half.value - analysis.first_half.value) > 2.0 * noise


# --------------------------------------------------------------------------
# Replica agreement
# --------------------------------------------------------------------------
@dataclass
class ReplicaAgreement:
    """Whether independent replicas agree, given their own uncertainties."""

    n_replicas: int
    combined: Measurement
    per_replica: list[Measurement] = field(default_factory=list)
    reduced_chi_square: float | None = None
    between_replica_sd: float | None = None
    mean_within_uncertainty: float | None = None
    determination: Determination = Determination.KNOWN
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_replicas": self.n_replicas,
            "combined": self.combined.model_dump(mode="json"),
            "per_replica": [m.model_dump(mode="json") for m in self.per_replica],
            "reduced_chi_square": self.reduced_chi_square,
            "between_replica_sd": self.between_replica_sd,
            "mean_within_uncertainty": self.mean_within_uncertainty,
            "determination": self.determination.value,
            "note": self.note,
        }


def combine_replicas(
    replicas: Sequence[Measurement], *, name: str | None = None, units: str | None = None
) -> ReplicaAgreement:
    """Combine per-replica means into one estimate and test their consistency.

    The combined uncertainty is the **standard error of the replica means**, not the
    within-replica error.  Replicas are the independent experimental units here; using
    the frame-level error would be textbook pseudoreplication.
    """
    usable = [m for m in replicas if m.determination is Determination.KNOWN and m.value is not None]
    n = len(usable)
    label = name or (usable[0].name if usable else "value")
    unit = units or (usable[0].units if usable else "1")

    if n == 0:
        return ReplicaAgreement(
            n_replicas=0,
            combined=Measurement.unknown(
                label, units=unit, reason="no usable replica results",
                determination=Determination.INSUFFICIENT_DATA,
            ),
            determination=Determination.INSUFFICIENT_DATA,
            note="No replica produced a usable value.",
        )

    values = np.array([m.value for m in usable], dtype=float)

    if n == 1:
        # One replica can be measured but cannot demonstrate reproducibility.
        only = usable[0]
        return ReplicaAgreement(
            n_replicas=1,
            combined=Measurement(
                name=label,
                value=float(values[0]),
                uncertainty=only.uncertainty,
                units=unit,
                n_samples=only.n_samples,
                effective_samples=only.effective_samples,
                method="single replica (within-replica uncertainty only)",
                notes="Reproducibility across replicas is not demonstrated by a single run.",
            ),
            per_replica=list(usable),
            determination=Determination.INSUFFICIENT_DATA,
            note="A single replica cannot establish replica agreement.",
        )

    mean = float(values.mean())
    between_sd = float(values.std(ddof=1))
    stderr = between_sd / math.sqrt(n)

    within = [m.uncertainty for m in usable if m.uncertainty is not None]
    mean_within = float(np.mean(within)) if within else None

    reduced_chi2: float | None = None
    if len(within) == n and all(u > 0 for u in within):
        sigmas = np.array(within, dtype=float)
        weights = 1.0 / sigmas**2
        weighted_mean = float(np.sum(weights * values) / np.sum(weights))
        chi2 = float(np.sum(((values - weighted_mean) / sigmas) ** 2))
        reduced_chi2 = chi2 / (n - 1)

    return ReplicaAgreement(
        n_replicas=n,
        combined=Measurement(
            name=label,
            value=mean,
            uncertainty=stderr,
            units=unit,
            n_samples=n,
            effective_samples=float(n),
            method="mean of independent replicas (standard error of replica means)",
        ),
        per_replica=list(usable),
        reduced_chi_square=reduced_chi2,
        between_replica_sd=between_sd,
        mean_within_uncertainty=mean_within,
    )


def replica_agreement_gate(
    agreement: ReplicaAgreement,
    *,
    required_replicas: int = 3,
    max_reduced_chi_square: float = 4.0,
) -> GateResult:
    """Decide whether replicas agree well enough to trust the combined value."""
    name = f"{agreement.combined.name}:replica_agreement"

    if agreement.n_replicas == 0:
        return GateResult(
            gate=name,
            status=GateStatus.FAIL,
            message="No replica produced a usable value",
            value=0.0,
            threshold=float(required_replicas),
            units="1",
        )

    if agreement.n_replicas < required_replicas:
        return GateResult(
            gate=name,
            status=GateStatus.INCONCLUSIVE,
            message=(
                f"{agreement.n_replicas} replica(s) available but {required_replicas} are required; "
                "reproducibility is not demonstrated"
            ),
            value=float(agreement.n_replicas),
            threshold=float(required_replicas),
            units="1",
        )

    if agreement.reduced_chi_square is None:
        # Fall back to comparing spread against the combined standard error.
        return GateResult(
            gate=name,
            status=GateStatus.INCONCLUSIVE,
            message=(
                "Replicas produced values but not per-replica uncertainties, so their agreement "
                "cannot be tested statistically"
            ),
            value=agreement.between_replica_sd,
            units=agreement.combined.units,
            evidence={"values": [m.value for m in agreement.per_replica]},
        )

    if agreement.reduced_chi_square > max_reduced_chi_square:
        return GateResult(
            gate=name,
            status=GateStatus.FAIL,
            message=(
                f"Replicas disagree: reduced chi-square {agreement.reduced_chi_square:.2f} exceeds "
                f"{max_reduced_chi_square:.1f}, so the spread between replicas is far larger than "
                "their individual uncertainties"
            ),
            value=agreement.reduced_chi_square,
            threshold=max_reduced_chi_square,
            units="1",
            evidence={"values": [m.value for m in agreement.per_replica]},
        )

    return GateResult(
        gate=name,
        status=GateStatus.PASS,
        message=(
            f"{agreement.n_replicas} replicas agree (reduced chi-square "
            f"{agreement.reduced_chi_square:.2f})"
        ),
        value=agreement.reduced_chi_square,
        threshold=max_reduced_chi_square,
        units="1",
        evidence={"combined": agreement.combined.value, "between_replica_sd": agreement.between_replica_sd},
    )


def build_convergence_report(
    analyses: dict[str, SeriesAnalysis],
    agreements: dict[str, ReplicaAgreement] | None = None,
    *,
    defaults: AnalysisDefaults | None = None,
    required_replicas: int = 3,
) -> GateReport:
    """Assemble every convergence and agreement gate into one report."""
    report = GateReport(name="convergence")
    for analysis in analyses.values():
        report.gates.extend(convergence_gates(analysis, defaults=defaults))
    for agreement in (agreements or {}).values():
        report.gates.append(replica_agreement_gate(agreement, required_replicas=required_replicas))
    return report


__all__ = [
    "ReplicaAgreement",
    "SeriesAnalysis",
    "analyse_series",
    "build_convergence_report",
    "combine_replicas",
    "convergence_gates",
    "replica_agreement_gate",
]
