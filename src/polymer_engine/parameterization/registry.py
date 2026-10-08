"""The parameterization registry: what was parameterized, how, and what it proved.

This is the scientific asset the subsystem exists to build.  Each record answers, for
one polymer/backend/force-field/property-class combination: which parameters were used,
whether they were validated, how, when, against what reference, and for what purpose.

Qualification is scoped on purpose.  A record qualifies a *polymer* for a *property
class* under a *named force-field version*; nothing here ever qualifies a force field
globally, and family-level claims require representative evidence rather than one
passing example.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.logging import get_logger
from polymer_engine.core.provenance import canonical_hash
from polymer_engine.parameterization.models import (
    ParameterizationState,
    PropertyClass,
    QMPriority,
    utc_now,
)

logger = get_logger("parameterization.registry")

#: Distinct representatives a family needs before a family-level claim is entertained.
MIN_FAMILY_REPRESENTATIVES = 2


@dataclass
class ParameterizationRecord:
    """One polymer, one backend, one force field, one property class."""

    polymer_id: str
    polymer_name: str
    family: str
    backend: str
    force_field: str
    force_field_version: str | None
    property_class: PropertyClass
    state: ParameterizationState
    parameter_source: str = ""
    topology_path: str | None = None
    parameter_files: list[str] = field(default_factory=list)
    net_charge: float | None = None
    max_penalty: float | None = None
    qm_priority: QMPriority | None = None
    qm_method: str | None = None
    qm_basis: str | None = None
    qm_reference: str | None = None
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    recorded_at: str = field(default_factory=utc_now)

    @property
    def key(self) -> str:
        """Identity of the combination, not of this particular run."""
        return canonical_hash({
            "polymer_id": self.polymer_id, "backend": self.backend,
            "force_field": self.force_field,
            "force_field_version": self.force_field_version,
            "property_class": self.property_class.value,
        })[:20]

    @property
    def qualified(self) -> bool:
        return self.state is ParameterizationState.QUALIFIED

    @property
    def parameterized(self) -> bool:
        """A topology exists.  Distinct from validated, and from qualified."""
        return self.state not in {
            ParameterizationState.DISCOVERED,
            ParameterizationState.BACKEND_SELECTED,
            ParameterizationState.BLOCKED,
            ParameterizationState.FAILED,
        }

    @property
    def qm_validated(self) -> bool:
        return self.state in {
            ParameterizationState.QM_VALIDATED,
            ParameterizationState.SYSTEM_VALIDATED,
            ParameterizationState.QUALIFIED,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "polymer_id": self.polymer_id,
            "polymer_name": self.polymer_name, "family": self.family,
            "backend": self.backend, "force_field": self.force_field,
            "force_field_version": self.force_field_version,
            "property_class": self.property_class.value, "state": self.state.value,
            "parameterized": self.parameterized, "qm_validated": self.qm_validated,
            "qualified": self.qualified,
            "parameter_source": self.parameter_source,
            "topology_path": self.topology_path,
            "parameter_files": list(self.parameter_files),
            "net_charge": self.net_charge, "max_penalty": self.max_penalty,
            "qm_priority": self.qm_priority.value if self.qm_priority else None,
            "qm_method": self.qm_method, "qm_basis": self.qm_basis,
            "qm_reference": self.qm_reference,
            "validation_metrics": dict(self.validation_metrics),
            "artifacts": dict(self.artifacts), "provenance": dict(self.provenance),
            "diagnostics": list(self.diagnostics), "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ParameterizationRecord:
        data = dict(payload)
        for derived in ("key", "parameterized", "qm_validated", "qualified"):
            data.pop(derived, None)
        data["property_class"] = PropertyClass(data["property_class"])
        data["state"] = ParameterizationState(data["state"])
        priority = data.get("qm_priority")
        data["qm_priority"] = QMPriority(priority) if priority else None
        return cls(**data)


@dataclass
class FamilyCoverage:
    """How much of a family a force field has actually been shown to handle."""

    family: str
    force_field: str
    property_class: PropertyClass
    representatives: list[str] = field(default_factory=list)
    qualified: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

    @property
    def qualified_for_family(self) -> bool:
        """Requires more than one passing example.

        One qualified polyester does not qualify polyesters: PLA and PET share a
        functional group and very little else, so a single representative is an
        anecdote rather than coverage.
        """
        return len(self.qualified) >= MIN_FAMILY_REPRESENTATIVES

    @property
    def confidence(self) -> str:
        if self.qualified_for_family:
            return "moderate"
        if self.qualified:
            return "single-representative"
        return "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family, "force_field": self.force_field,
            "property_class": self.property_class.value,
            "n_representatives": len(self.representatives),
            "n_qualified": len(self.qualified), "n_blocked": len(self.blocked),
            "representatives": list(self.representatives),
            "qualified": list(self.qualified), "blocked": list(self.blocked),
            "qualified_for_family": self.qualified_for_family,
            "confidence": self.confidence,
            "basis": (f"family-level qualification requires at least "
                      f"{MIN_FAMILY_REPRESENTATIVES} distinct qualified representatives"),
        }


class ParameterizationRegistry:
    """Durable store of parameterization records."""

    def __init__(self) -> None:
        self._records: dict[str, ParameterizationRecord] = {}

    def add(self, record: ParameterizationRecord) -> ParameterizationRecord:
        self._records[record.key] = record
        logger.info("Recorded %s/%s for %s (%s)", record.backend, record.force_field,
                    record.polymer_name, record.state.value)
        return record

    def get(self, key: str) -> ParameterizationRecord | None:
        return self._records.get(key)

    def all(self) -> list[ParameterizationRecord]:
        return sorted(self._records.values(), key=lambda r: (r.polymer_name, r.backend))

    def for_polymer(self, polymer_id: str) -> list[ParameterizationRecord]:
        return [r for r in self.all() if r.polymer_id == polymer_id]

    def qualified(self) -> list[ParameterizationRecord]:
        return [r for r in self.all() if r.qualified]

    def family_coverage(
        self, family: str, force_field: str, property_class: PropertyClass
    ) -> FamilyCoverage:
        coverage = FamilyCoverage(family=family, force_field=force_field,
                                  property_class=property_class)
        for record in self.all():
            if record.family != family or record.force_field != force_field:
                continue
            if record.property_class is not property_class:
                continue
            coverage.representatives.append(record.polymer_name)
            if record.qualified:
                coverage.qualified.append(record.polymer_name)
            elif record.state in {ParameterizationState.BLOCKED,
                                  ParameterizationState.FAILED}:
                coverage.blocked.append(record.polymer_name)
        return coverage

    def summary(self) -> dict[str, Any]:
        records = self.all()
        return {
            "n_records": len(records),
            "n_parameterized": sum(1 for r in records if r.parameterized),
            "n_qm_validated": sum(1 for r in records if r.qm_validated),
            "n_qualified": sum(1 for r in records if r.qualified),
            "by_backend": {
                backend: sum(1 for r in records if r.backend == backend)
                for backend in sorted({r.backend for r in records})
            },
            "by_state": {
                state: sum(1 for r in records if r.state.value == state)
                for state in sorted({r.state.value for r in records})
            },
        }

    # -- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"generated_at": utc_now(), "summary": self.summary(),
                   "records": [r.as_dict() for r in self.all()]}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(path)          # atomic: a killed write never truncates the registry
        return path

    @classmethod
    def load(cls, path: str | Path) -> ParameterizationRegistry:
        registry = cls()
        path = Path(path)
        if not path.is_file():
            return registry
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw in payload.get("records", []):
            record = ParameterizationRecord.from_dict(raw)
            registry._records[record.key] = record
        return registry


__all__ = [
    "MIN_FAMILY_REPRESENTATIVES", "FamilyCoverage", "ParameterizationRecord",
    "ParameterizationRegistry",
]
