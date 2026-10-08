"""Canonical domain models.

This module is the single source of truth for the engine's vocabulary.  Nothing
else in the package may define a competing ``Action``, ``GateResult`` or state enum.

Three ideas drive the design:

* **Four-valued gates.**  A scientific check that cannot be evaluated is
  ``INCONCLUSIVE``, never ``PASS`` and never ``FAIL``.  Collapsing the third and
  fourth states into a boolean is how a pipeline starts believing things.
* **Measurements carry uncertainty and units.**  A bare float is not a result.
* **Execution state is a validated machine.**  Illegal transitions raise.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from polymer_engine.core.errors import IllegalStateTransition
from polymer_engine.core.units import dimension_of


def utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Honest-unknown sentinels
# --------------------------------------------------------------------------
class Determination(str, Enum):
    """What the engine is allowed to say when it does not know something.

    Rule 29: never substitute a plausible-looking value for an unknown.
    """

    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    REQUIRES_VALIDATION = "REQUIRES_VALIDATION"


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
class GateStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"

    @property
    def blocks_promotion(self) -> bool:
        """Only an explicit PASS (or a WARN) permits promotion.

        ``INCONCLUSIVE`` blocks: absence of evidence is not evidence of adequacy.
        """
        return self in {GateStatus.FAIL, GateStatus.INCONCLUSIVE}


class GateResult(BaseModel):
    """The outcome of one deterministic scientific check."""

    model_config = {"extra": "forbid"}

    gate: str
    status: GateStatus
    message: str
    value: float | None = None
    uncertainty: float | None = None
    threshold: float | None = None
    units: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)

    @field_validator("units")
    @classmethod
    def _known_unit(cls, value: str | None) -> str | None:
        if value is not None:
            dimension_of(value)
        return value

    @property
    def passed(self) -> bool:
        return self.status is GateStatus.PASS


class GateReport(BaseModel):
    """An aggregate of gates with a conservative combination rule."""

    model_config = {"extra": "forbid"}

    name: str
    gates: list[GateResult] = Field(default_factory=list)

    @property
    def status(self) -> GateStatus:
        """Worst-case aggregation: FAIL > INCONCLUSIVE > WARN > PASS.

        An empty report is ``INCONCLUSIVE`` -- running no checks is not a pass.
        """
        if not self.gates:
            return GateStatus.INCONCLUSIVE
        statuses = {g.status for g in self.gates}
        for candidate in (GateStatus.FAIL, GateStatus.INCONCLUSIVE, GateStatus.WARN):
            if candidate in statuses:
                return candidate
        return GateStatus.PASS

    @property
    def promotable(self) -> bool:
        return not self.status.blocks_promotion

    def failures(self) -> list[GateResult]:
        return [g for g in self.gates if g.status is GateStatus.FAIL]

    def inconclusive(self) -> list[GateResult]:
        return [g for g in self.gates if g.status is GateStatus.INCONCLUSIVE]

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for gate in self.gates:
            counts[gate.status.value] = counts.get(gate.status.value, 0) + 1
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return f"{self.name}: {self.status.value} ({rendered or 'no gates'})"


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------
class Measurement(BaseModel):
    """A number with a unit, an uncertainty, and an honest determination flag.

    ``value`` is ``None`` whenever ``determination`` is not ``KNOWN``; that is
    enforced, so downstream code cannot read a placeholder number.
    """

    model_config = {"extra": "forbid"}

    name: str
    value: float | None = None
    uncertainty: float | None = Field(default=None, ge=0)
    units: str = "1"
    determination: Determination = Determination.KNOWN
    n_samples: int | None = Field(default=None, ge=0)
    effective_samples: float | None = Field(default=None, ge=0)
    method: str | None = None
    notes: str | None = None

    @field_validator("units")
    @classmethod
    def _known_unit(cls, value: str) -> str:
        dimension_of(value)
        return value

    @field_validator("value", "uncertainty")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Measurement values must be finite; use determination=UNKNOWN instead")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Measurement:
        if self.determination is Determination.KNOWN and self.value is None:
            raise ValueError(f"Measurement {self.name!r} is KNOWN but has no value")
        if self.determination is not Determination.KNOWN and self.value is not None:
            raise ValueError(
                f"Measurement {self.name!r} is {self.determination.value} and must not carry a value"
            )
        return self

    @classmethod
    def unknown(cls, name: str, units: str = "1", *, reason: str = "", determination: Determination = Determination.UNKNOWN) -> Measurement:
        return cls(name=name, units=units, determination=determination, notes=reason or None)

    @property
    def relative_uncertainty(self) -> float | None:
        if self.value is None or self.uncertainty is None or self.value == 0:
            return None
        return abs(self.uncertainty / self.value)

    def render(self) -> str:
        if self.determination is not Determination.KNOWN:
            return f"{self.name}={self.determination.value}"
        if self.uncertainty is None:
            return f"{self.name}={self.value:.6g} {self.units} (uncertainty unknown)"
        return f"{self.name}={self.value:.6g} +/- {self.uncertainty:.3g} {self.units}"


# --------------------------------------------------------------------------
# Execution lifecycle
# --------------------------------------------------------------------------
class ExecutionState(str, Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"

    @property
    def terminal(self) -> bool:
        return self in {ExecutionState.CANCELLED, ExecutionState.PROMOTED, ExecutionState.REJECTED}


#: Allowed transitions.  ``COMPLETED -> RUNNING`` is absent on purpose: rerunning a
#: finished job requires an explicit restart, which creates a *new* run record.
ALLOWED_TRANSITIONS: dict[ExecutionState, frozenset[ExecutionState]] = {
    ExecutionState.CREATED: frozenset({ExecutionState.VALIDATED, ExecutionState.FAILED, ExecutionState.CANCELLED}),
    ExecutionState.VALIDATED: frozenset({ExecutionState.QUEUED, ExecutionState.FAILED, ExecutionState.CANCELLED}),
    ExecutionState.QUEUED: frozenset({ExecutionState.RUNNING, ExecutionState.CANCELLED, ExecutionState.FAILED}),
    ExecutionState.RUNNING: frozenset(
        {ExecutionState.COMPLETED, ExecutionState.FAILED, ExecutionState.CANCELLED, ExecutionState.RETRYING}
    ),
    ExecutionState.RETRYING: frozenset({ExecutionState.QUEUED, ExecutionState.FAILED, ExecutionState.CANCELLED}),
    ExecutionState.COMPLETED: frozenset({ExecutionState.PROMOTED, ExecutionState.REJECTED}),
    ExecutionState.FAILED: frozenset({ExecutionState.RETRYING, ExecutionState.REJECTED, ExecutionState.CANCELLED}),
    ExecutionState.CANCELLED: frozenset(),
    ExecutionState.PROMOTED: frozenset(),
    ExecutionState.REJECTED: frozenset(),
}


def can_transition(source: ExecutionState, target: ExecutionState) -> bool:
    return target in ALLOWED_TRANSITIONS[source]


class StateTransition(BaseModel):
    """One audited move through the execution lifecycle."""

    model_config = {"extra": "forbid"}

    from_state: ExecutionState
    to_state: ExecutionState
    timestamp: datetime = Field(default_factory=utc_now)
    actor: str
    reason: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)


class ExecutionRecord(BaseModel):
    """Durable lifecycle for one unit of execution (a simulation, a download, ...).

    A restart does not reopen a ``COMPLETED`` record; call :meth:`restart` to obtain
    a fresh record that points back at this one.
    """

    model_config = {"extra": "forbid"}

    id: str = Field(default_factory=lambda: _new_id("run"))
    kind: str
    label: str = ""
    state: ExecutionState = ExecutionState.CREATED
    history: list[StateTransition] = Field(default_factory=list)
    restarted_from: str | None = None
    attempt: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def transition(
        self,
        target: ExecutionState,
        *,
        actor: str,
        reason: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
    ) -> StateTransition:
        if not can_transition(self.state, target):
            raise IllegalStateTransition(
                f"{self.state.value} -> {target.value} is not a permitted transition",
                record_id=self.id,
                kind=self.kind,
                allowed=sorted(s.value for s in ALLOWED_TRANSITIONS[self.state]),
            )
        entry = StateTransition(
            from_state=self.state,
            to_state=target,
            actor=actor,
            reason=reason,
            inputs=inputs or {},
            outputs=outputs or {},
        )
        self.history.append(entry)
        self.state = target
        self.updated_at = entry.timestamp
        if target is ExecutionState.RETRYING:
            self.attempt += 1
        return entry

    def restart(self, *, actor: str, reason: str) -> ExecutionRecord:
        """Create a successor record.  Explicit restart semantics, per the lifecycle rules."""
        successor = ExecutionRecord(kind=self.kind, label=self.label, restarted_from=self.id)
        successor.history.append(
            StateTransition(
                from_state=self.state,
                to_state=ExecutionState.CREATED,
                actor=actor,
                reason=f"restart of {self.id}: {reason}",
            )
        )
        return successor


# --------------------------------------------------------------------------
# Research objects
# --------------------------------------------------------------------------
class Objective(BaseModel):
    model_config = {"extra": "forbid"}

    id: str = Field(default_factory=lambda: _new_id("obj"))
    title: str
    description: str
    target_properties: dict[str, float | str] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    UNDER_TEST = "under_test"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    RETIRED = "retired"


class Hypothesis(BaseModel):
    model_config = {"extra": "forbid"}

    id: str = Field(default_factory=lambda: _new_id("hyp"))
    statement: str
    rationale: str
    objective_id: str | None = None
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    prior: float = Field(default=0.5, ge=0.0, le=1.0)
    posterior: float | None = Field(default=None, ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradictory_evidence: list[str] = Field(default_factory=list)
    discriminating_observables: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ActionStatus(str, Enum):
    PROPOSED = "proposed"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    RETIRED = "retired"


class CostEstimate(BaseModel):
    """What an action is expected to consume.

    ``determination`` records whether the numbers are real estimates or unknown --
    a missing estimate must not silently read as "free".
    """

    model_config = {"extra": "forbid"}

    cpu_hours: float | None = Field(default=None, ge=0)
    gpu_hours: float | None = Field(default=None, ge=0)
    wall_hours: float | None = Field(default=None, ge=0)
    determination: Determination = Determination.KNOWN
    basis: str = ""

    @property
    def scalar(self) -> float | None:
        """A single comparable cost in GPU-equivalent hours, or ``None`` if unknown."""
        if self.determination is not Determination.KNOWN:
            return None
        parts = [v for v in (self.cpu_hours, self.gpu_hours, self.wall_hours) if v is not None]
        if not parts:
            return None
        cpu = self.cpu_hours or 0.0
        gpu = self.gpu_hours or 0.0
        return gpu + cpu / 8.0 if (self.cpu_hours is not None or self.gpu_hours is not None) else float(self.wall_hours or 0.0)

    @classmethod
    def unknown(cls, basis: str = "no cost model available") -> CostEstimate:
        return cls(determination=Determination.UNKNOWN, basis=basis)


class Action(BaseModel):
    """A candidate experiment the planner may select."""

    model_config = {"extra": "forbid"}

    id: str = Field(default_factory=lambda: _new_id("act"))
    kind: str
    title: str
    question: str = ""
    hypothesis_id: str | None = None
    campaign_id: str | None = None
    strategy_id: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    cost: CostEstimate = Field(default_factory=CostEstimate.unknown)
    expected_information_gain: float | None = Field(default=None, ge=0.0)
    uncertainty_reduction: float | None = Field(default=None, ge=0.0)
    design_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    risk: float = Field(default=0.5, ge=0.0, le=1.0)
    status: ActionStatus = ActionStatus.PROPOSED
    depends_on: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class Observation(BaseModel):
    """A measured fact attributable to one action."""

    model_config = {"extra": "forbid"}

    id: str = Field(default_factory=lambda: _new_id("obs"))
    action_id: str
    campaign_id: str | None = None
    measurement: Measurement
    artifact_ids: list[str] = Field(default_factory=list)
    provenance_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def metric(self) -> str:
        return self.measurement.name


class ExperimentResult(BaseModel):
    """What an executor returns.  ``execution_mode`` never lies about dry runs."""

    model_config = {"extra": "forbid"}

    action_id: str
    status: ActionStatus
    execution_mode: Literal["real", "dry_run", "blocked"] = "real"
    artifacts: list[str] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    report: GateReport = Field(default_factory=lambda: GateReport(name="result"))
    summary: str = ""
    error: str | None = None
    provenance_id: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def scientifically_usable(self) -> bool:
        """True only for a real, succeeded, gate-passing result.

        A dry run is never scientifically usable, regardless of exit codes.
        """
        return (
            self.execution_mode == "real"
            and self.status is ActionStatus.SUCCEEDED
            and self.report.promotable
        )


__all__ = [
    "ALLOWED_TRANSITIONS",
    "Action",
    "ActionStatus",
    "CostEstimate",
    "Determination",
    "ExecutionRecord",
    "ExecutionState",
    "ExperimentResult",
    "GateReport",
    "GateResult",
    "GateStatus",
    "Hypothesis",
    "HypothesisStatus",
    "Measurement",
    "Objective",
    "Observation",
    "StateTransition",
    "can_transition",
    "utc_now",
]
