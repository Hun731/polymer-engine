"""The property-calculation framework.

Every property in this engine declares five things, and none of them is optional:

``observable``
    What is being measured, and from what.
``units``
    Canonical engine units. A bare number is not a result.
``estimator``
    How the value is derived from the raw data.
``uncertainty method``
    How the error bar was obtained -- and it is correlation-aware, because MD frames
    are not independent samples.
``required sampling``
    What the property needs before it may be believed: minimum effective samples,
    minimum replicas, whether equilibration must be demonstrated.

:class:`PropertyResult` therefore carries a :class:`GateReport` alongside the number,
and :attr:`PropertyResult.usable` is false whenever the sampling requirement was not
met -- regardless of how precise the value looks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from polymer_engine.core.config import AnalysisDefaults
from polymer_engine.core.errors import ScientificError
from polymer_engine.core.models import (
    Determination,
    GateReport,
    GateResult,
    GateStatus,
    Measurement,
)
from polymer_engine.core.units import dimension_of


class PropertyClass(str, Enum):
    THERMODYNAMIC = "thermodynamic"
    STRUCTURAL = "structural"
    DYNAMICAL = "dynamical"
    INTERMOLECULAR = "intermolecular"
    MECHANICAL = "mechanical"
    TRANSPORT = "transport"


class UncertaintyMethod(str, Enum):
    #: Standard error corrected for autocorrelation (Chodera statistical inefficiency).
    CORRELATION_AWARE_SE = "correlation_aware_standard_error"
    #: Standard error of independent replica means.
    REPLICA_SE = "replica_standard_error"
    #: Percentile bootstrap, block-resampled for correlated data.
    BLOCK_BOOTSTRAP = "block_bootstrap"
    #: Propagated from a fit's covariance.
    FIT_COVARIANCE = "fit_covariance"
    #: No uncertainty could be estimated. Reported, never silently omitted.
    NONE = "none"


@dataclass(frozen=True, slots=True)
class SamplingRequirement:
    """What a property needs before its value may be believed."""

    min_effective_samples: float = 20.0
    min_replicas: int = 3
    requires_equilibration: bool = True
    min_simulation_ns: float | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_effective_samples": self.min_effective_samples,
            "min_replicas": self.min_replicas,
            "requires_equilibration": self.requires_equilibration,
            "min_simulation_ns": self.min_simulation_ns,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class PropertyDefinition:
    """The contract a property implementation must satisfy."""

    name: str
    property_class: PropertyClass
    units: str
    observable: str
    estimator: str
    uncertainty_method: UncertaintyMethod
    sampling: SamplingRequirement = field(default_factory=SamplingRequirement)
    description: str = ""
    caveats: str = ""

    def __post_init__(self) -> None:
        dimension_of(self.units)  # rejects an unknown unit at definition time

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class": self.property_class.value,
            "units": self.units,
            "observable": self.observable,
            "estimator": self.estimator,
            "uncertainty_method": self.uncertainty_method.value,
            "sampling": self.sampling.as_dict(),
            "description": self.description,
            "caveats": self.caveats,
        }


@dataclass
class PropertyResult:
    """A property value together with everything needed to judge it."""

    definition: PropertyDefinition
    measurement: Measurement
    report: GateReport
    n_replicas: int = 1
    per_replica: list[Measurement] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """True only when the value is known *and* the sampling requirement was met."""
        return (
            self.measurement.determination is Determination.KNOWN
            and self.report.promotable
        )

    @property
    def value(self) -> float | None:
        return self.measurement.value if self.usable else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition.as_dict(),
            "measurement": self.measurement.model_dump(mode="json"),
            "usable": self.usable,
            "n_replicas": self.n_replicas,
            "per_replica": [m.model_dump(mode="json") for m in self.per_replica],
            "gate_status": self.report.status.value,
            "gates": [g.model_dump(mode="json") for g in self.report.gates],
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
            "extra": self.extra,
        }


class PropertyCalculator(ABC):
    """Base class for every property calculation."""

    definition: PropertyDefinition

    def __init__(self, defaults: AnalysisDefaults | None = None) -> None:
        self.defaults = defaults or AnalysisDefaults()

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> PropertyResult:
        """Compute the property. Must not raise for an expected scientific shortfall."""

    # -- shared helpers ---------------------------------------------------
    def unknown(self, reason: str, *, determination: Determination = Determination.INSUFFICIENT_DATA) -> PropertyResult:
        """A property that could not be computed, with the reason attached."""
        report = GateReport(
            name=f"property:{self.definition.name}",
            gates=[
                GateResult(
                    gate=f"{self.definition.name}:computable",
                    status=GateStatus.FAIL,
                    message=reason,
                )
            ],
        )
        return PropertyResult(
            definition=self.definition,
            measurement=Measurement.unknown(
                self.definition.name, units=self.definition.units,
                reason=reason, determination=determination,
            ),
            report=report,
            diagnostics=[reason],
        )

    def sampling_gates(
        self,
        measurement: Measurement,
        *,
        n_replicas: int,
        equilibration_shown: bool | None = None,
        simulation_ns: float | None = None,
    ) -> list[GateResult]:
        """Check the value against this property's declared sampling requirement."""
        requirement = self.definition.sampling
        gates: list[GateResult] = []
        name = self.definition.name

        effective = measurement.effective_samples
        if effective is None:
            gates.append(
                GateResult(
                    gate=f"{name}:effective_samples",
                    status=GateStatus.INCONCLUSIVE,
                    message="effective sample size was not determined",
                )
            )
        else:
            ok = effective >= requirement.min_effective_samples
            gates.append(
                GateResult(
                    gate=f"{name}:effective_samples",
                    status=GateStatus.PASS if ok else GateStatus.FAIL,
                    message=(
                        f"{effective:.1f} effective samples from {measurement.n_samples} frames"
                        if ok
                        else f"only {effective:.1f} effective samples after correcting for "
                             f"autocorrelation; {requirement.min_effective_samples:.0f} are required"
                    ),
                    value=float(effective),
                    threshold=requirement.min_effective_samples,
                    units="1",
                )
            )

        if n_replicas < requirement.min_replicas:
            gates.append(
                GateResult(
                    gate=f"{name}:replicas",
                    status=GateStatus.INCONCLUSIVE,
                    message=(
                        f"{n_replicas} replica(s) but {requirement.min_replicas} are required; "
                        "reproducibility is not demonstrated"
                    ),
                    value=float(n_replicas),
                    threshold=float(requirement.min_replicas),
                    units="1",
                )
            )
        else:
            gates.append(
                GateResult(
                    gate=f"{name}:replicas",
                    status=GateStatus.PASS,
                    message=f"{n_replicas} independent replicas",
                    value=float(n_replicas),
                    threshold=float(requirement.min_replicas),
                    units="1",
                )
            )

        if requirement.requires_equilibration:
            if equilibration_shown is None:
                gates.append(
                    GateResult(
                        gate=f"{name}:equilibration",
                        status=GateStatus.INCONCLUSIVE,
                        message="equilibration was not assessed for this property",
                    )
                )
            else:
                gates.append(
                    GateResult(
                        gate=f"{name}:equilibration",
                        status=GateStatus.PASS if equilibration_shown else GateStatus.FAIL,
                        message=(
                            "the production window was selected after equilibration"
                            if equilibration_shown
                            else "equilibration was not demonstrated"
                        ),
                    )
                )

        if requirement.min_simulation_ns is not None:
            if simulation_ns is None:
                gates.append(
                    GateResult(
                        gate=f"{name}:duration",
                        status=GateStatus.INCONCLUSIVE,
                        message="simulation length is unknown, so it cannot be checked",
                    )
                )
            else:
                ok = simulation_ns >= requirement.min_simulation_ns
                gates.append(
                    GateResult(
                        gate=f"{name}:duration",
                        status=GateStatus.PASS if ok else GateStatus.FAIL,
                        message=(
                            f"{simulation_ns:.2f} ns of sampling"
                            if ok
                            else f"{simulation_ns:.2f} ns is below the {requirement.min_simulation_ns} ns "
                                 f"this property needs: {requirement.notes or 'see the property definition'}"
                        ),
                        value=simulation_ns,
                        threshold=requirement.min_simulation_ns,
                        units="1",
                    )
                )
        return gates


class PropertyRegistry:
    """Every property the engine can compute, discoverable by name or class."""

    def __init__(self) -> None:
        self._calculators: dict[str, PropertyCalculator] = {}

    def register(self, calculator: PropertyCalculator) -> PropertyCalculator:
        self._calculators[calculator.definition.name] = calculator
        return calculator

    def get(self, name: str) -> PropertyCalculator:
        try:
            return self._calculators[name]
        except KeyError:
            raise ScientificError(
                "Unknown property", name=name, known=sorted(self._calculators)
            ) from None

    def names(self) -> list[str]:
        return sorted(self._calculators)

    def by_class(self, property_class: PropertyClass) -> list[str]:
        return sorted(
            name for name, c in self._calculators.items()
            if c.definition.property_class is property_class
        )

    def definitions(self) -> dict[str, dict[str, Any]]:
        return {name: c.definition.as_dict() for name, c in sorted(self._calculators.items())}


def combine_replica_property(
    definition: PropertyDefinition,
    per_replica: Sequence[Measurement],
    *,
    equilibration_shown: bool | None = None,
    simulation_ns: float | None = None,
    defaults: AnalysisDefaults | None = None,
    provenance: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> PropertyResult:
    """Combine per-replica values into one property result with agreement gates.

    The combined uncertainty is the standard error of the **replica means**. Replicas
    are the independent experimental units here; using the frame-level error would be
    pseudoreplication.
    """
    from polymer_engine.analysis.convergence import combine_replicas, replica_agreement_gate

    agreement = combine_replicas(list(per_replica), name=definition.name, units=definition.units)
    report = GateReport(name=f"property:{definition.name}")

    calculator = _AdHocCalculator(definition, defaults)
    report.gates.extend(
        calculator.sampling_gates(
            agreement.combined,
            n_replicas=agreement.n_replicas,
            equilibration_shown=equilibration_shown,
            simulation_ns=simulation_ns,
        )
    )
    report.gates.append(
        replica_agreement_gate(agreement, required_replicas=definition.sampling.min_replicas)
    )

    return PropertyResult(
        definition=definition,
        measurement=agreement.combined,
        report=report,
        n_replicas=agreement.n_replicas,
        per_replica=list(agreement.per_replica),
        diagnostics=[agreement.note] if agreement.note else [],
        provenance=provenance or {},
        extra=extra or {},
    )


class _AdHocCalculator(PropertyCalculator):
    """Adapter so :func:`combine_replica_property` can reuse the gate helpers."""

    def __init__(self, definition: PropertyDefinition, defaults: AnalysisDefaults | None) -> None:
        super().__init__(defaults)
        self.definition = definition

    def compute(self, *args: Any, **kwargs: Any) -> PropertyResult:  # pragma: no cover
        raise NotImplementedError("this adapter only provides gate helpers")


__all__ = [
    "PropertyCalculator",
    "PropertyClass",
    "PropertyDefinition",
    "PropertyRegistry",
    "PropertyResult",
    "SamplingRequirement",
    "UncertaintyMethod",
    "combine_replica_property",
]
