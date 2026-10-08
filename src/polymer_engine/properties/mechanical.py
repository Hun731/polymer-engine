"""Mechanical properties from deformation simulations.

This module is deliberately conservative about language, because atomistic deformation
results are routinely over-claimed.

What a non-equilibrium MD tensile simulation measures is the stress response of a
periodic, defect-free, nanometre-scale cell strained at ~10^7-10^9 s^-1.  A real
tensile test measures a macroscopic specimen containing voids, entanglement networks,
crystallites and surfaces, strained at ~10^-3 s^-1 -- eleven or more orders of
magnitude slower.  These are not the same measurement.

So every quantity here is labelled by what it *is*:

``elastic_modulus``
    The slope of the initial linear region of the simulated stress-strain curve.  A
    well-defined simulation observable.
``yield_stress_proxy`` / ``peak_stress``
    Features of the simulated curve.  ``_proxy`` is in the name because the simulated
    yield point is strain-rate dependent and systematically exceeds the experimental
    one.
``strain_hardening_slope``
    The post-yield slope of the simulated curve.

:class:`MechanicalInterpretation` records the strain rate and states explicitly whether
the result may be compared with experiment.  At MD strain rates the answer is no, and
the engine says so instead of leaving the reader to assume otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from polymer_engine.core.config import AnalysisDefaults
from polymer_engine.core.errors import ScientificError
from polymer_engine.core.models import Determination, GateReport, GateResult, GateStatus, Measurement
from polymer_engine.properties.base import (
    PropertyCalculator,
    PropertyClass,
    PropertyDefinition,
    PropertyResult,
    SamplingRequirement,
    UncertaintyMethod,
)

#: Strain rates above this are far outside any experimental regime.  Chosen because
#: laboratory tensile tests run near 1e-3 s^-1 and even split-Hopkinson bar impact
#: testing reaches only ~1e4 s^-1; anything faster has no experimental counterpart.
EXPERIMENTAL_STRAIN_RATE_CEILING = 1.0e4


class MetricKind(str, Enum):
    """What a mechanical number actually is."""

    #: A well-defined property of the simulated system.
    SIMULATION_OBSERVABLE = "simulation_observable"
    #: A simulated feature that stands in for an experimental concept but is not it.
    SIMULATION_PROXY = "simulation_proxy"
    #: Directly comparable with a laboratory measurement.
    EXPERIMENTAL_EQUIVALENT = "experimental_equivalent"


@dataclass(frozen=True, slots=True)
class MechanicalInterpretation:
    """Under what conditions a mechanical number was obtained, and what it means."""

    strain_rate_per_s: float | None
    metric_kind: MetricKind
    comparable_to_experiment: bool
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "strain_rate_per_s": self.strain_rate_per_s,
            "metric_kind": self.metric_kind.value,
            "comparable_to_experiment": self.comparable_to_experiment,
            "rationale": self.rationale,
        }


def interpret_strain_rate(strain_rate_per_s: float | None, metric_kind: MetricKind) -> MechanicalInterpretation:
    """Decide whether a result at this strain rate may be compared with experiment."""
    if strain_rate_per_s is None:
        return MechanicalInterpretation(
            strain_rate_per_s=None,
            metric_kind=metric_kind,
            comparable_to_experiment=False,
            rationale=(
                "The strain rate was not recorded, so the result cannot be placed relative "
                "to any experimental regime."
            ),
        )
    if strain_rate_per_s > EXPERIMENTAL_STRAIN_RATE_CEILING:
        return MechanicalInterpretation(
            strain_rate_per_s=strain_rate_per_s,
            metric_kind=metric_kind,
            comparable_to_experiment=False,
            rationale=(
                f"The strain rate is {strain_rate_per_s:.2e} s^-1, far above the ~1e-3 s^-1 of a "
                "laboratory tensile test and above even impact testing (~1e4 s^-1). Polymer "
                "mechanical response is strongly rate dependent, so this is a simulated "
                "observable and not an experimental equivalent."
            ),
        )
    return MechanicalInterpretation(
        strain_rate_per_s=strain_rate_per_s,
        metric_kind=metric_kind,
        comparable_to_experiment=metric_kind is MetricKind.EXPERIMENTAL_EQUIVALENT,
        rationale=(
            f"The strain rate {strain_rate_per_s:.2e} s^-1 is within reach of experiment, but "
            "finite system size and the absence of defects still limit comparability."
        ),
    )


@dataclass
class StressStrainCurve:
    """A simulated stress-strain response."""

    strain: np.ndarray
    stress_mpa: np.ndarray
    strain_rate_per_s: float | None = None
    temperature_k: float | None = None
    direction: str = "x"
    stress_uncertainty_mpa: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.strain = np.asarray(self.strain, dtype=float).ravel()
        self.stress_mpa = np.asarray(self.stress_mpa, dtype=float).ravel()
        if self.strain.size != self.stress_mpa.size:
            raise ScientificError(
                "Strain and stress arrays must be the same length",
                n_strain=int(self.strain.size), n_stress=int(self.stress_mpa.size),
            )
        if self.strain.size < 3:
            raise ScientificError("A stress-strain curve needs at least three points")
        if not np.all(np.isfinite(self.strain)) or not np.all(np.isfinite(self.stress_mpa)):
            raise ScientificError("Stress-strain curve contains non-finite values")
        if np.any(np.diff(self.strain) < 0):
            raise ScientificError("Strain must be non-decreasing along the curve")

    @property
    def max_strain(self) -> float:
        return float(self.strain.max())

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_points": int(self.strain.size),
            "max_strain": self.max_strain,
            "max_stress_mpa": float(self.stress_mpa.max()),
            "strain_rate_per_s": self.strain_rate_per_s,
            "temperature_k": self.temperature_k,
            "direction": self.direction,
        }


@dataclass
class MechanicalAnalysis:
    """Everything extracted from one stress-strain curve."""

    elastic_modulus: Measurement
    peak_stress: Measurement
    yield_stress_proxy: Measurement
    yield_strain: Measurement
    strain_hardening_slope: Measurement
    interpretation: MechanicalInterpretation
    report: GateReport
    linear_region: tuple[float, float] | None = None
    diagnostics: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "elastic_modulus": self.elastic_modulus.model_dump(mode="json"),
            "peak_stress": self.peak_stress.model_dump(mode="json"),
            "yield_stress_proxy": self.yield_stress_proxy.model_dump(mode="json"),
            "yield_strain": self.yield_strain.model_dump(mode="json"),
            "strain_hardening_slope": self.strain_hardening_slope.model_dump(mode="json"),
            "interpretation": self.interpretation.as_dict(),
            "linear_region": list(self.linear_region) if self.linear_region else None,
            "gate_status": self.report.status.value,
            "gates": [g.model_dump(mode="json") for g in self.report.gates],
            "diagnostics": self.diagnostics,
        }


def find_linear_region(
    strain: np.ndarray,
    stress: np.ndarray,
    *,
    max_strain: float = 0.02,
    min_points: int = 5,
    min_r_squared: float = 0.98,
) -> tuple[int, int, float] | None:
    """Locate the initial linear elastic region.

    Grows a window from the origin while the linear fit stays good.  A fixed
    "first 2% strain" rule is wrong for a stiff glassy polymer that has already
    yielded by then, and equally wrong for a soft elastomer still linear at 10%.
    """
    n = strain.size
    if n < min_points:
        return None
    limit = int(np.searchsorted(strain, max_strain, side="right"))
    limit = max(min_points, min(limit, n))

    best: tuple[int, int, float] | None = None
    for end in range(min_points, limit + 1):
        x, y = strain[:end], stress[:end]
        if np.allclose(x, x[0]):
            continue
        design = np.column_stack([np.ones_like(x), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        predicted = design @ coefficients
        ss_res = float(((y - predicted) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if r_squared >= min_r_squared:
            best = (0, end, r_squared)
    return best


class TensileAnalysis(PropertyCalculator):
    """Extracts mechanical metrics from a simulated stress-strain curve."""

    definition = PropertyDefinition(
        name="tensile_response",
        property_class=PropertyClass.MECHANICAL,
        units="MPa",
        observable="Stress response of a periodic cell under imposed uniaxial strain",
        estimator="Linear fit in the elastic region; curve features beyond it",
        uncertainty_method=UncertaintyMethod.FIT_COVARIANCE,
        sampling=SamplingRequirement(
            min_effective_samples=5.0,
            min_replicas=3,
            requires_equilibration=True,
            notes="Deformation must start from a properly equilibrated, density-converged cell; "
                  "mechanical response is highly sensitive to the starting configuration.",
        ),
        description="Simulated uniaxial tensile response.",
        caveats="MD strain rates exceed experimental ones by many orders of magnitude, and the "
                "cell is defect-free and nanometre-scale. These numbers describe the simulation.",
    )

    def compute(
        self,
        curve: StressStrainCurve,
        *,
        n_replicas: int = 1,
        elastic_max_strain: float = 0.02,
        provenance: dict[str, Any] | None = None,
    ) -> PropertyResult:
        analysis = self.analyse(curve, elastic_max_strain=elastic_max_strain, n_replicas=n_replicas)
        return PropertyResult(
            definition=self.definition,
            measurement=analysis.elastic_modulus,
            report=analysis.report,
            n_replicas=n_replicas,
            diagnostics=analysis.diagnostics,
            provenance={
                "curve": curve.as_dict(),
                "interpretation": analysis.interpretation.as_dict(),
                **(provenance or {}),
            },
            extra=analysis.as_dict(),
        )

    def analyse(
        self,
        curve: StressStrainCurve,
        *,
        elastic_max_strain: float = 0.02,
        n_replicas: int = 1,
    ) -> MechanicalAnalysis:
        report = GateReport(name="property:tensile_response")
        diagnostics: list[str] = []

        interpretation = interpret_strain_rate(
            curve.strain_rate_per_s, MetricKind.SIMULATION_OBSERVABLE
        )
        report.gates.append(
            GateResult(
                gate="mechanical:strain_rate",
                status=GateStatus.PASS if curve.strain_rate_per_s is not None else GateStatus.INCONCLUSIVE,
                message=interpretation.rationale,
                value=curve.strain_rate_per_s,
                units="1",
                evidence=interpretation.as_dict(),
            )
        )
        if not interpretation.comparable_to_experiment:
            diagnostics.append(
                "these values describe the simulation and are not experimental equivalents"
            )

        # -- elastic modulus -------------------------------------------
        region = find_linear_region(curve.strain, curve.stress_mpa, max_strain=elastic_max_strain)
        if region is None:
            report.gates.append(
                GateResult(
                    gate="mechanical:linear_region",
                    status=GateStatus.FAIL,
                    message=(
                        f"no linear region was found below {elastic_max_strain:.1%} strain; "
                        "an elastic modulus cannot be defined for this curve"
                    ),
                )
            )
            modulus = Measurement.unknown(
                "elastic_modulus", units="MPa",
                reason="no identifiable linear elastic region",
                determination=Determination.INSUFFICIENT_DATA,
            )
            linear_region = None
        else:
            start, end, r_squared = region
            x, y = curve.strain[start:end], curve.stress_mpa[start:end]
            design = np.column_stack([np.ones_like(x), x])
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            slope = float(coefficients[1])
            predicted = design @ coefficients
            ss_res = float(((y - predicted) ** 2).sum())
            dof = max(1, x.size - 2)
            centred = x - x.mean()
            denominator = float((centred**2).sum())
            slope_error = math.sqrt(ss_res / dof / denominator) if denominator > 0 else None

            linear_region = (float(x[0]), float(x[-1]))
            modulus = Measurement(
                name="elastic_modulus",
                value=slope,
                uncertainty=slope_error,
                units="MPa",
                n_samples=int(x.size),
                effective_samples=float(x.size),
                method=f"linear fit over strain {x[0]:.4f}-{x[-1]:.4f}, R^2 = {r_squared:.4f}",
                notes="slope of the simulated stress-strain curve in its linear region",
            )
            report.gates.append(
                GateResult(
                    gate="mechanical:linear_region",
                    status=GateStatus.PASS,
                    message=f"linear region over strain {x[0]:.4f}-{x[-1]:.4f} with R^2 = {r_squared:.4f}",
                    value=r_squared,
                    threshold=0.98,
                    units="1",
                )
            )

        # -- peak and yield ---------------------------------------------
        peak_index = int(np.argmax(curve.stress_mpa))
        peak = Measurement(
            name="peak_stress",
            value=float(curve.stress_mpa[peak_index]),
            units="MPa",
            method="maximum of the simulated stress-strain curve",
            notes="a feature of the simulated curve, not an experimental tensile strength",
        )

        yield_index = self._yield_index(curve)
        if yield_index is None:
            yield_stress = Measurement.unknown(
                "yield_stress_proxy", units="MPa",
                reason="no yield point was reached within the simulated strain range",
                determination=Determination.INSUFFICIENT_DATA,
            )
            yield_strain = Measurement.unknown(
                "yield_strain", units="1",
                reason="no yield point was reached",
                determination=Determination.INSUFFICIENT_DATA,
            )
            report.gates.append(
                GateResult(
                    gate="mechanical:yield_reached",
                    status=GateStatus.INCONCLUSIVE,
                    message=(
                        f"the curve does not yield within the simulated strain of "
                        f"{curve.max_strain:.2%}; only the elastic response is characterised"
                    ),
                    value=curve.max_strain,
                    units="1",
                )
            )
        else:
            yield_stress = Measurement(
                name="yield_stress_proxy",
                value=float(curve.stress_mpa[yield_index]),
                units="MPa",
                method="first local stress maximum of the simulated curve",
                notes="a strain-rate-dependent simulation proxy, not an experimental yield stress",
            )
            yield_strain = Measurement(
                name="yield_strain",
                value=float(curve.strain[yield_index]),
                units="1",
                method="strain at the first local stress maximum",
            )
            report.gates.append(
                GateResult(
                    gate="mechanical:yield_reached",
                    status=GateStatus.PASS,
                    message=f"yield proxy at strain {curve.strain[yield_index]:.3f}",
                    value=float(curve.strain[yield_index]),
                    units="1",
                )
            )

        # -- strain hardening --------------------------------------------
        hardening = self._strain_hardening(curve, yield_index)

        report.gates.extend(
            self.sampling_gates(modulus, n_replicas=n_replicas, equilibration_shown=True)
        )
        report.gates.append(
            GateResult(
                gate="mechanical:experimental_comparability",
                status=GateStatus.WARN if not interpretation.comparable_to_experiment else GateStatus.PASS,
                message=interpretation.rationale,
                evidence=interpretation.as_dict(),
            )
        )

        return MechanicalAnalysis(
            elastic_modulus=modulus,
            peak_stress=peak,
            yield_stress_proxy=yield_stress,
            yield_strain=yield_strain,
            strain_hardening_slope=hardening,
            interpretation=interpretation,
            report=report,
            linear_region=linear_region,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _yield_index(curve: StressStrainCurve) -> int | None:
        """First local stress maximum, taken as the yield point of the simulated curve.

        A monotonically rising curve has not yielded; reporting its endpoint as a yield
        stress would be reporting where the simulation happened to stop.
        """
        stress = curve.stress_mpa
        if stress.size < 3:
            return None
        for i in range(1, stress.size - 1):
            if stress[i] >= stress[i - 1] and stress[i] > stress[i + 1]:
                return i
        return None

    @staticmethod
    def _strain_hardening(curve: StressStrainCurve, yield_index: int | None) -> Measurement:
        if yield_index is None or yield_index >= curve.strain.size - 3:
            return Measurement.unknown(
                "strain_hardening_slope", units="MPa",
                reason="no post-yield region was simulated",
                determination=Determination.INSUFFICIENT_DATA,
            )
        x = curve.strain[yield_index:]
        y = curve.stress_mpa[yield_index:]
        design = np.column_stack([np.ones_like(x), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        return Measurement(
            name="strain_hardening_slope",
            value=float(coefficients[1]),
            units="MPa",
            n_samples=int(x.size),
            effective_samples=float(x.size),
            method="linear fit to the simulated post-yield region",
            notes="a slope of the simulated curve; negative values indicate softening",
        )


class BulkModulus(PropertyCalculator):
    """Bulk modulus from equilibrium volume fluctuations.

    ``K = k_B T <V> / var(V)`` in the NPT ensemble.  This estimator converges slowly:
    it depends on the *variance* of the volume, so it needs far more sampling than the
    mean density does, and the calculator checks the effective sample size accordingly.
    """

    definition = PropertyDefinition(
        name="bulk_modulus",
        property_class=PropertyClass.MECHANICAL,
        units="MPa",
        observable="Equilibrium volume fluctuations in the NPT ensemble",
        estimator="K = kT <V> / var(V)",
        uncertainty_method=UncertaintyMethod.BLOCK_BOOTSTRAP,
        sampling=SamplingRequirement(
            min_effective_samples=200.0,
            min_replicas=3,
            min_simulation_ns=50.0,
            notes="A variance-based estimator converges much more slowly than a mean; "
                  "hundreds of effective samples are needed, not tens.",
        ),
        description="Isothermal bulk modulus from volume fluctuations.",
        caveats="Requires a correctly sampled NPT ensemble; a Berendsen barostat gives the "
                "wrong volume fluctuations and therefore the wrong modulus.",
    )

    def compute(
        self,
        volumes_nm3: Sequence[float] | np.ndarray,
        temperature_k: float,
        *,
        n_replicas: int = 1,
        simulation_ns: float | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> PropertyResult:
        from polymer_engine.analysis.statistics import effective_sample_size
        from polymer_engine.core.units import GAS_CONSTANT_KJ_PER_MOL_K

        volumes = np.asarray(volumes_nm3, dtype=float).ravel()
        if volumes.size < 10:
            return self.unknown("at least ten volume samples are required")
        if temperature_k <= 0:
            return self.unknown("temperature must be positive", determination=Determination.UNKNOWN)
        if not np.all(np.isfinite(volumes)):
            return self.unknown("volume series contains non-finite values",
                                determination=Determination.UNKNOWN)

        variance = float(volumes.var(ddof=1))
        if variance <= 0:
            return self.unknown(
                "volume does not fluctuate; the simulation was not run in an NPT ensemble"
            )

        mean_volume = float(volumes.mean())
        # k_B T in kJ/mol per particle-equivalent; volumes in nm^3.
        # K = kT<V>/var(V);  1 kJ/mol/nm^3 = 1.66054 MPa.
        kt = GAS_CONSTANT_KJ_PER_MOL_K * temperature_k
        modulus_kj_mol_nm3 = kt * mean_volume / variance
        modulus_mpa = modulus_kj_mol_nm3 * 1.6605390666

        ess = effective_sample_size(volumes)
        measurement = Measurement(
            name=self.definition.name,
            value=modulus_mpa,
            uncertainty=modulus_mpa * math.sqrt(2.0 / max(ess - 1.0, 1.0)),
            units="MPa",
            n_samples=int(volumes.size),
            effective_samples=ess,
            method="K = kT<V>/var(V) from NPT volume fluctuations",
            notes="uncertainty from the sampling error of a variance estimate",
        )
        report = GateReport(name=f"property:{self.definition.name}")
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
                "mean_volume_nm3": mean_volume,
                "volume_variance_nm6": variance,
                "temperature_k": temperature_k,
                **(provenance or {}),
            },
        )


def poisson_ratio(
    axial_strain: Sequence[float] | np.ndarray,
    transverse_strain: Sequence[float] | np.ndarray,
) -> Measurement:
    """Poisson ratio from the transverse response to axial strain.

    Physically bounded by -1 < nu < 0.5 for an isotropic material; a value outside that
    range means the fit is not describing linear elastic behaviour, and it is refused
    rather than reported.
    """
    axial = np.asarray(axial_strain, dtype=float).ravel()
    transverse = np.asarray(transverse_strain, dtype=float).ravel()
    if axial.size != transverse.size:
        raise ScientificError("Axial and transverse strain arrays must match in length")
    if axial.size < 3:
        return Measurement.unknown(
            "poisson_ratio", units="1", reason="at least three strain points are required",
            determination=Determination.INSUFFICIENT_DATA,
        )
    if np.allclose(axial, axial[0]):
        return Measurement.unknown(
            "poisson_ratio", units="1", reason="axial strain does not vary",
            determination=Determination.INSUFFICIENT_DATA,
        )

    design = np.column_stack([np.ones_like(axial), axial])
    coefficients, *_ = np.linalg.lstsq(design, transverse, rcond=None)
    nu = float(-coefficients[1])
    if not (-1.0 < nu < 0.5):
        return Measurement.unknown(
            "poisson_ratio", units="1",
            reason=(
                f"fitted value {nu:.3f} lies outside the thermodynamic bounds (-1, 0.5) for an "
                "isotropic material; the response is not linear elastic"
            ),
            determination=Determination.REQUIRES_VALIDATION,
        )
    return Measurement(
        name="poisson_ratio",
        value=nu,
        units="1",
        n_samples=int(axial.size),
        effective_samples=float(axial.size),
        method="negative slope of transverse against axial strain",
    )


def mechanical_calculators(defaults: AnalysisDefaults | None = None) -> list[PropertyCalculator]:
    return [TensileAnalysis(defaults), BulkModulus(defaults)]


__all__ = [
    "EXPERIMENTAL_STRAIN_RATE_CEILING",
    "BulkModulus",
    "MechanicalAnalysis",
    "MechanicalInterpretation",
    "MetricKind",
    "StressStrainCurve",
    "TensileAnalysis",
    "find_linear_region",
    "interpret_strain_rate",
    "mechanical_calculators",
    "poisson_ratio",
]
