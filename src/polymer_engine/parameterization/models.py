"""Typed contracts for parameterizing a polymer.

Three states are kept apart everywhere in this package, because conflating them is the
central failure mode of parameterization work:

``PARAMETERIZED``
    A topology exists.  Nothing has been checked.

``VALIDATED``
    The parameters were compared against an independent reference and agreed.

``QUALIFIED``
    Validated *for a stated property class*, with provenance, on a stated polymer family.

A topology file is evidence of the first and of nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from polymer_engine.core.models import Determination, GateReport
from polymer_engine.parameterization.capability import CapabilityState


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class PropertyClass(str, Enum):
    """What the parameters are going to be used *for*.

    Qualification is per property class on purpose. A torsional barrier that is 5 kJ/mol
    wrong barely moves a bulk density and ruins a conformational free energy, so one
    qualification standard for both would be either uselessly strict or dangerously lax.
    """

    BULK_DENSITY = "bulk_density"
    THERMODYNAMIC = "thermodynamic"
    CONFORMATIONAL_FREE_ENERGY = "conformational_free_energy"
    INTERFACIAL_FREE_ENERGY = "interfacial_free_energy"
    TRANSPORT = "transport"
    MECHANICAL = "mechanical"
    STRUCTURAL = "structural"


class QMPriority(str, Enum):
    """How badly a parameter set needs quantum-chemical scrutiny."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ParameterizationState(str, Enum):
    """The state machine a polymer/backend pair moves through."""

    DISCOVERED = "DISCOVERED"
    BACKEND_SELECTED = "BACKEND_SELECTED"
    PARAMETERIZED = "PARAMETERIZED"
    TOPOLOGY_VALIDATED = "TOPOLOGY_VALIDATED"
    PARAMETER_COMPLETENESS_VALIDATED = "PARAMETER_COMPLETENESS_VALIDATED"
    QM_VALIDATION_REQUIRED = "QM_VALIDATION_REQUIRED"
    QM_VALIDATED = "QM_VALIDATED"
    SYSTEM_VALIDATED = "SYSTEM_VALIDATED"
    QUALIFIED = "QUALIFIED"
    # Terminal, non-progress states.
    BLOCKED = "BLOCKED"
    INCONCLUSIVE = "INCONCLUSIVE"
    FAILED = "FAILED"
    REQUIRES_EXPERT_REVIEW = "REQUIRES_EXPERT_REVIEW"


#: Property classes that do not require torsional QM may go straight from completeness
#: to system validation. The alternative -- passing through QM_VALIDATED without running
#: any QM -- would record a claim that was never tested.
OPTIONAL_QM_SHORTCUT: tuple[ParameterizationState, ParameterizationState] = (
    ParameterizationState.PARAMETER_COMPLETENESS_VALIDATED,
    ParameterizationState.SYSTEM_VALIDATED,
)

#: The forward path.  Anything not here is a verdict, reachable from any state.
STATE_SEQUENCE: tuple[ParameterizationState, ...] = (
    ParameterizationState.DISCOVERED,
    ParameterizationState.BACKEND_SELECTED,
    ParameterizationState.PARAMETERIZED,
    ParameterizationState.TOPOLOGY_VALIDATED,
    ParameterizationState.PARAMETER_COMPLETENESS_VALIDATED,
    ParameterizationState.QM_VALIDATION_REQUIRED,
    ParameterizationState.QM_VALIDATED,
    ParameterizationState.SYSTEM_VALIDATED,
    ParameterizationState.QUALIFIED,
)

TERMINAL_STATES: frozenset[ParameterizationState] = frozenset({
    ParameterizationState.BLOCKED,
    ParameterizationState.INCONCLUSIVE,
    ParameterizationState.FAILED,
    ParameterizationState.REQUIRES_EXPERT_REVIEW,
    ParameterizationState.QUALIFIED,
})


@dataclass
class ForceFieldAssessment:
    """What one backend says it could do with one polymer, before doing anything."""

    backend: str
    polymer_id: str
    polymer_name: str
    state: CapabilityState
    force_field: str | None = None
    force_field_version: str | None = None
    reason: str = ""
    #: Chemistry the backend cannot type, as ``(element, environment)`` descriptions.
    unsupported: list[str] = field(default_factory=list)
    requires_credentials: bool = False
    requires_human_step: bool = False
    estimated_cost: str = "unknown"
    qm_priority: QMPriority | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    assessed_at: str = field(default_factory=utc_now)

    @property
    def usable(self) -> bool:
        """Can this backend actually produce parameters for this polymer?"""
        from polymer_engine.parameterization.capability import at_least

        return at_least(self.state, CapabilityState.PARAMETERIZATION_AVAILABLE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "polymer_id": self.polymer_id,
            "polymer_name": self.polymer_name, "state": self.state.value,
            "force_field": self.force_field,
            "force_field_version": self.force_field_version,
            "reason": self.reason, "unsupported": list(self.unsupported),
            "requires_credentials": self.requires_credentials,
            "requires_human_step": self.requires_human_step,
            "estimated_cost": self.estimated_cost,
            "qm_priority": self.qm_priority.value if self.qm_priority else None,
            "usable": self.usable, "evidence": dict(self.evidence),
            "assessed_at": self.assessed_at,
        }


@dataclass
class ParameterizationRequest:
    """What the caller wants parameterized, and under what assumptions.

    Nothing here has a scientific default.  A force field, a chain length and a target
    property each change the answer, so each is supplied explicitly or the request is
    incomplete.
    """

    polymer_id: str
    polymer_name: str
    repeat_unit_smiles: str
    property_class: PropertyClass
    degree_of_polymerization: int = 30
    n_chains: int = 20
    temperature_k: float = 300.0
    pressure_bar: float = 1.0
    target_density_kg_m3: float | None = None
    force_field: str | None = None
    #: Set by the operator once a human-in-the-loop backend has produced a job.
    external_job_id: str | None = None
    source_archive: str | None = None
    workdir: str | None = None
    notes: str = ""

    def problems(self) -> list[str]:
        issues: list[str] = []
        if not self.repeat_unit_smiles.strip():
            issues.append("a repeat unit is required")
        if self.degree_of_polymerization < 2:
            issues.append("degree of polymerisation must be at least 2")
        if self.n_chains < 1:
            issues.append("at least one chain is required")
        if self.temperature_k <= 0:
            issues.append("temperature must be positive")
        return issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id, "polymer_name": self.polymer_name,
            "repeat_unit_smiles": self.repeat_unit_smiles,
            "property_class": self.property_class.value,
            "degree_of_polymerization": self.degree_of_polymerization,
            "n_chains": self.n_chains, "temperature_k": self.temperature_k,
            "pressure_bar": self.pressure_bar,
            "target_density_kg_m3": self.target_density_kg_m3,
            "force_field": self.force_field, "external_job_id": self.external_job_id,
            "source_archive": self.source_archive, "workdir": self.workdir,
            "notes": self.notes,
        }


@dataclass
class ParameterizationResult:
    """What a backend produced.  Produced is not validated."""

    backend: str
    request: ParameterizationRequest
    state: ParameterizationState
    force_field: str | None = None
    force_field_version: str | None = None
    parameter_source: str = ""
    topology_path: str | None = None
    coordinate_path: str | None = None
    parameter_files: list[str] = field(default_factory=list)
    n_atoms: int | None = None
    net_charge: float | None = None
    diagnostics: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)
    #: Every file hashed, so a later result can be traced back to its inputs.
    artifacts: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    @property
    def parameterized(self) -> bool:
        """A topology exists.  This says nothing about whether it is right."""
        from polymer_engine.parameterization.models import STATE_SEQUENCE

        return (self.state in STATE_SEQUENCE
                and STATE_SEQUENCE.index(self.state)
                >= STATE_SEQUENCE.index(ParameterizationState.PARAMETERIZED))

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "state": self.state.value,
            "force_field": self.force_field,
            "force_field_version": self.force_field_version,
            "parameter_source": self.parameter_source,
            "topology_path": self.topology_path,
            "coordinate_path": self.coordinate_path,
            "parameter_files": list(self.parameter_files),
            "n_atoms": self.n_atoms, "net_charge": self.net_charge,
            "parameterized": self.parameterized,
            "diagnostics": list(self.diagnostics),
            "required_actions": list(self.required_actions),
            "artifacts": dict(self.artifacts), "provenance": dict(self.provenance),
            "created_at": self.created_at, "request": self.request.as_dict(),
        }


@dataclass
class ParameterizationValidation:
    """Independent evidence about a parameter set, and the verdict it supports."""

    backend: str
    polymer_id: str
    property_class: PropertyClass
    state: ParameterizationState
    determination: Determination = Determination.UNKNOWN
    completeness: GateReport | None = None
    charges: GateReport | None = None
    topology: GateReport | None = None
    penalties: GateReport | None = None
    qm: GateReport | None = None
    qm_priority: QMPriority | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    validated_at: str = field(default_factory=utc_now)

    def reports(self) -> list[tuple[str, GateReport]]:
        named = (("completeness", self.completeness), ("charges", self.charges),
                 ("topology", self.topology), ("penalties", self.penalties),
                 ("qm", self.qm))
        return [(name, report) for name, report in named if report is not None]

    @property
    def promotable(self) -> bool:
        """Every gate that ran must be promotable, and at least one must have run."""
        reports = self.reports()
        return bool(reports) and all(report.promotable for _name, report in reports)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "polymer_id": self.polymer_id,
            "property_class": self.property_class.value, "state": self.state.value,
            "determination": self.determination.value,
            "promotable": self.promotable,
            "qm_priority": self.qm_priority.value if self.qm_priority else None,
            "gates": {
                name: {"status": report.status.value,
                       "promotable": report.promotable,
                       "gates": [g.model_dump(mode="json") for g in report.gates]}
                for name, report in self.reports()
            },
            "metrics": dict(self.metrics), "diagnostics": list(self.diagnostics),
            "validated_at": self.validated_at,
        }


__all__ = [
    "STATE_SEQUENCE", "TERMINAL_STATES", "ForceFieldAssessment",
    "ParameterizationRequest", "ParameterizationResult", "ParameterizationState",
    "ParameterizationValidation", "PropertyClass", "QMPriority", "utc_now",
]
