"""Thermodynamic properties from equilibrium MD.

Everything here is a time average over an equilibrated production window, with the
error bar computed from the effective sample size rather than the frame count.

The thermal expansion coefficient is the interesting case: it is a *derivative*
obtained by finite difference across temperatures, so it needs several independent
state points and its uncertainty must be propagated from theirs.  With two closely
spaced, noisy densities the derivative is dominated by noise, and the calculator says
so rather than reporting a number.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from polymer_engine.core.config import AnalysisDefaults
from polymer_engine.core.errors import InsufficientDataError
from polymer_engine.core.models import Determination, GateReport, GateResult, GateStatus, Measurement
from polymer_engine.properties.base import (
    PropertyCalculator,
    PropertyClass,
    PropertyDefinition,
    PropertyResult,
    SamplingRequirement,
    UncertaintyMethod,
)


class TimeSeriesProperty(PropertyCalculator):
    """A property that is the equilibrium average of a per-frame observable."""

    def compute(
        self,
        values: Sequence[float] | np.ndarray,
        *,
        times_ps: Sequence[float] | np.ndarray | None = None,
        n_replicas: int = 1,
        simulation_ns: float | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> PropertyResult:
        from polymer_engine.analysis.convergence import analyse_series, convergence_gates

        array = np.asarray(values, dtype=float).ravel()
        if array.size == 0:
            return self.unknown("no samples were supplied")
        if not np.all(np.isfinite(array)):
            return self.unknown(
                "the time series contains non-finite values", determination=Determination.UNKNOWN
            )

        analysis = analyse_series(
            array,
            name=self.definition.name,
            units=self.definition.units,
            times_ps=times_ps,
            defaults=self.defaults,
        )
        report = GateReport(name=f"property:{self.definition.name}")
        report.gates.extend(convergence_gates(analysis, defaults=self.defaults))
        report.gates.extend(
            self.sampling_gates(
                analysis.production,
                n_replicas=n_replicas,
                equilibration_shown=analysis.equilibration.n_discarded >= 0,
                simulation_ns=simulation_ns,
            )
        )
        return PropertyResult(
            definition=self.definition,
            measurement=analysis.production,
            report=report,
            n_replicas=n_replicas,
            provenance={
                "equilibration": analysis.equilibration.as_dict(),
                "drift_fraction": analysis.drift_fraction,
                "drift_per_ns": analysis.drift_per_ns,
                **(provenance or {}),
            },
        )


class Density(TimeSeriesProperty):
    definition = PropertyDefinition(
        name="density",
        property_class=PropertyClass.THERMODYNAMIC,
        units="kg/m^3",
        observable="System mass divided by the instantaneous box volume, per frame",
        estimator="Equilibrium time average over the production window",
        uncertainty_method=UncertaintyMethod.CORRELATION_AWARE_SE,
        sampling=SamplingRequirement(
            min_effective_samples=20.0,
            min_replicas=3,
            min_simulation_ns=5.0,
            notes="Density equilibrates quickly but its fluctuations are correlated over "
                  "tens of picoseconds; a few nanoseconds of NPT are needed for a stable mean.",
        ),
        description="Bulk mass density at the simulated temperature and pressure.",
        caveats="Requires a converged NPT ensemble with a barostat that samples it correctly.",
    )


class PotentialEnergy(TimeSeriesProperty):
    definition = PropertyDefinition(
        name="potential_energy",
        property_class=PropertyClass.THERMODYNAMIC,
        units="kJ/mol",
        observable="Total potential energy of the system, per frame",
        estimator="Equilibrium time average over the production window",
        uncertainty_method=UncertaintyMethod.CORRELATION_AWARE_SE,
        sampling=SamplingRequirement(min_effective_samples=20.0, min_replicas=3),
        description="Total configurational energy.",
        caveats="An absolute potential energy is force-field specific and has no "
                "experimental counterpart; only differences between comparable systems mean anything.",
    )


class Temperature(TimeSeriesProperty):
    definition = PropertyDefinition(
        name="temperature",
        property_class=PropertyClass.THERMODYNAMIC,
        units="K",
        observable="Instantaneous kinetic temperature",
        estimator="Equilibrium time average over the production window",
        uncertainty_method=UncertaintyMethod.CORRELATION_AWARE_SE,
        sampling=SamplingRequirement(min_effective_samples=20.0, min_replicas=1),
        description="Kinetic temperature; a control check rather than a prediction.",
        caveats="Agreement with the thermostat set point is necessary, not sufficient, "
                "for a correct simulation.",
    )


class Pressure(TimeSeriesProperty):
    definition = PropertyDefinition(
        name="pressure",
        property_class=PropertyClass.THERMODYNAMIC,
        units="bar",
        observable="Instantaneous virial pressure",
        estimator="Equilibrium time average over the production window",
        uncertainty_method=UncertaintyMethod.CORRELATION_AWARE_SE,
        sampling=SamplingRequirement(
            min_effective_samples=50.0,
            min_replicas=3,
            notes="Instantaneous pressure fluctuates by hundreds of bar in a small system; "
                  "far more sampling is needed than for density.",
        ),
        description="Virial pressure.",
        caveats="Pressure is among the noisiest observables in MD; a large relative error "
                "on a small system is expected, not a defect.",
    )


class Volume(TimeSeriesProperty):
    definition = PropertyDefinition(
        name="volume",
        property_class=PropertyClass.THERMODYNAMIC,
        units="nm^3",
        observable="Instantaneous box volume in nm^3",
        estimator="Equilibrium time average over the production window",
        uncertainty_method=UncertaintyMethod.CORRELATION_AWARE_SE,
        sampling=SamplingRequirement(min_effective_samples=20.0, min_replicas=3),
        description="Box volume.",
    )


class Enthalpy(TimeSeriesProperty):
    definition = PropertyDefinition(
        name="enthalpy",
        property_class=PropertyClass.THERMODYNAMIC,
        units="kJ/mol",
        observable="Total energy plus pressure-volume work, per frame",
        estimator="Equilibrium time average over the production window",
        uncertainty_method=UncertaintyMethod.CORRELATION_AWARE_SE,
        sampling=SamplingRequirement(min_effective_samples=20.0, min_replicas=3),
        description="Enthalpy of the simulated system.",
        caveats="A heat of vaporisation requires a separate gas-phase reference simulation; "
                "this value alone is not one.",
    )


@dataclass(frozen=True, slots=True)
class StatePoint:
    """One equilibrated state point in a temperature series."""

    temperature_k: float
    density: Measurement

    def __post_init__(self) -> None:
        if self.temperature_k <= 0:
            raise InsufficientDataError("Temperature must be positive", temperature_k=self.temperature_k)


class ThermalExpansion(PropertyCalculator):
    """Volumetric thermal expansion coefficient from a density-temperature series.

    ``alpha = -(1/rho) (d rho / dT)`` at constant pressure, obtained by weighted linear
    regression of density against temperature.

    This is a derivative, which makes it far more demanding than the densities it comes
    from: the signal is the *slope*, and noise on individual densities propagates
    directly into it. The calculator refuses to report a value when the fitted slope is
    not distinguishable from zero, because a thermal expansion coefficient consistent
    with "no expansion" is not a measurement.
    """

    definition = PropertyDefinition(
        name="thermal_expansion_coefficient",
        property_class=PropertyClass.THERMODYNAMIC,
        units="1/K",
        observable="Relative density change with temperature at constant pressure",
        estimator="Weighted linear regression of density against temperature; alpha = -(1/rho) drho/dT",
        uncertainty_method=UncertaintyMethod.FIT_COVARIANCE,
        sampling=SamplingRequirement(
            min_effective_samples=20.0,
            min_replicas=3,
            notes="Needs at least three well-separated temperatures, each independently "
                  "equilibrated, because the quantity is a derivative.",
        ),
        description="Volumetric thermal expansion coefficient.",
        caveats="Crossing a glass transition invalidates a single linear fit; fit each "
                "regime separately.",
    )

    MIN_STATE_POINTS = 3
    #: A slope must exceed this many standard errors to count as resolved.
    MIN_SLOPE_SIGNIFICANCE = 2.0

    def compute(
        self,
        state_points: Sequence[StatePoint],
        *,
        provenance: dict[str, Any] | None = None,
    ) -> PropertyResult:
        usable = [
            p for p in state_points
            if p.density.determination is Determination.KNOWN and p.density.value is not None
        ]
        if len(usable) < self.MIN_STATE_POINTS:
            return self.unknown(
                f"{len(usable)} usable state point(s); a derivative needs at least "
                f"{self.MIN_STATE_POINTS} temperatures"
            )

        temperatures = np.array([p.temperature_k for p in usable], dtype=float)
        densities = np.array([p.density.value for p in usable], dtype=float)
        if len(set(temperatures.tolist())) < self.MIN_STATE_POINTS:
            return self.unknown("state points do not span enough distinct temperatures")

        uncertainties = np.array(
            [p.density.uncertainty if p.density.uncertainty else np.nan for p in usable], dtype=float
        )
        if np.all(np.isfinite(uncertainties)) and np.all(uncertainties > 0):
            weights = 1.0 / uncertainties**2
        else:
            weights = np.ones_like(densities)

        # Weighted least squares for rho = a + b*T.
        design = np.column_stack([np.ones_like(temperatures), temperatures])
        weighted = design * weights[:, None]
        normal = design.T @ weighted
        try:
            covariance = np.linalg.inv(normal)
        except np.linalg.LinAlgError:
            return self.unknown("the temperature design matrix is singular")
        coefficients = covariance @ (weighted.T @ densities)
        slope = float(coefficients[1])
        slope_error = float(math.sqrt(max(covariance[1, 1], 0.0)))

        mean_density = float(np.average(densities, weights=weights))
        if mean_density == 0:
            return self.unknown("mean density is zero")

        alpha = -slope / mean_density
        alpha_error = abs(slope_error / mean_density) if slope_error > 0 else None

        report = GateReport(name=f"property:{self.definition.name}")
        report.gates.append(
            GateResult(
                gate=f"{self.definition.name}:state_points",
                status=GateStatus.PASS,
                message=f"{len(usable)} state points spanning "
                        f"{temperatures.min():.0f}-{temperatures.max():.0f} K",
                value=float(len(usable)),
                threshold=float(self.MIN_STATE_POINTS),
                units="1",
            )
        )

        significant = slope_error > 0 and abs(slope) > self.MIN_SLOPE_SIGNIFICANCE * slope_error
        if not significant:
            report.gates.append(
                GateResult(
                    gate=f"{self.definition.name}:slope_resolved",
                    status=GateStatus.FAIL,
                    message=(
                        f"the density-temperature slope ({slope:.4g} +/- {slope_error:.4g}) is not "
                        f"distinguishable from zero; the expansion coefficient is unresolved"
                    ),
                    value=abs(slope) / slope_error if slope_error > 0 else 0.0,
                    threshold=self.MIN_SLOPE_SIGNIFICANCE,
                    units="1",
                )
            )
            return PropertyResult(
                definition=self.definition,
                measurement=Measurement.unknown(
                    self.definition.name, units=self.definition.units,
                    reason="density-temperature slope is not resolved above the noise",
                    determination=Determination.INSUFFICIENT_DATA,
                ),
                report=report,
                n_replicas=len(usable),
                per_replica=[p.density for p in usable],
                provenance={"slope": slope, "slope_error": slope_error, **(provenance or {})},
            )

        report.gates.append(
            GateResult(
                gate=f"{self.definition.name}:slope_resolved",
                status=GateStatus.PASS,
                message=f"slope resolved at {abs(slope) / slope_error:.1f} sigma",
                value=abs(slope) / slope_error,
                threshold=self.MIN_SLOPE_SIGNIFICANCE,
                units="1",
            )
        )
        return PropertyResult(
            definition=self.definition,
            measurement=Measurement(
                name=self.definition.name,
                value=alpha,
                uncertainty=alpha_error,
                units=self.definition.units,
                n_samples=len(usable),
                effective_samples=float(len(usable)),
                method="weighted linear regression of density against temperature",
                notes="K^-1; reported dimensionless because the engine's unit system has no "
                      "inverse-temperature dimension",
            ),
            report=report,
            n_replicas=len(usable),
            per_replica=[p.density for p in usable],
            provenance={
                "slope_kg_m3_per_K": slope,
                "slope_error": slope_error,
                "mean_density": mean_density,
                "temperatures": temperatures.tolist(),
                **(provenance or {}),
            },
        )


def thermodynamic_calculators(defaults: AnalysisDefaults | None = None) -> list[PropertyCalculator]:
    return [
        Density(defaults), PotentialEnergy(defaults), Temperature(defaults),
        Pressure(defaults), Volume(defaults), Enthalpy(defaults), ThermalExpansion(defaults),
    ]


__all__ = [
    "Density",
    "Enthalpy",
    "PotentialEnergy",
    "Pressure",
    "StatePoint",
    "Temperature",
    "ThermalExpansion",
    "TimeSeriesProperty",
    "Volume",
    "thermodynamic_calculators",
]
