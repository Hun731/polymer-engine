"""Learning from failed computational approaches.

An engine that repeats the same failing approach on the same polymer family is not
learning.  This module records *why* things failed, in enough structure that the
strategy layer can notice a pattern -- "approach A fails for polyesters, approach B
does not" -- and prefer differently next time.

The boundary is strict and permanent: failure learning may change **what is tried and
in what order**.  It may never change a validation gate, relax a threshold, or promote
a result.  A strategy that keeps failing gets deprioritised; it does not get an easier
exam.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination, utc_now
from polymer_engine.polymer.taxonomy import PolymerFamily

logger = get_logger("orchestrator.failure_learning")


class FailureType(str, Enum):
    """What kind of failure this was.  The recovery advice differs sharply."""

    #: The system did not pass structural validation.
    SYSTEM_INVALID = "system_invalid"
    #: Parameters were rejected before anything ran.
    PARAMETER_INVALID = "parameter_invalid"
    #: The external tool crashed or errored.
    TOOL_FAILURE = "tool_failure"
    #: The tool ran but the physics diverged (LINCS warnings, exploding energies).
    SIMULATION_DIVERGED = "simulation_diverged"
    #: Ran cleanly but produced too little independent data.
    INSUFFICIENT_SAMPLING = "insufficient_sampling"
    #: Ran long enough but the observable never settled.
    NOT_CONVERGED = "not_converged"
    #: Replicas disagreed beyond their uncertainties.
    REPLICA_DISAGREEMENT = "replica_disagreement"
    #: Quantum calculation failed to converge.
    QM_NOT_CONVERGED = "qm_not_converged"
    #: Umbrella windows did not overlap.
    POOR_OVERLAP = "poor_overlap"
    #: The machine could not supply the resources.
    RESOURCE_EXHAUSTED = "resource_exhausted"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

    @property
    def recoverable(self) -> bool:
        """Whether more of the same, or a small change, would plausibly succeed.

        A diverged simulation or an invalid system will fail again identically; more
        sampling genuinely fixes an under-sampled run.
        """
        return self in {
            FailureType.INSUFFICIENT_SAMPLING,
            FailureType.NOT_CONVERGED,
            FailureType.POOR_OVERLAP,
            FailureType.RESOURCE_EXHAUSTED,
            FailureType.TIMEOUT,
        }

    @property
    def suggested_action(self) -> str:
        return {
            FailureType.SYSTEM_INVALID: "rebuild or repair the system; rerunning will fail identically",
            FailureType.PARAMETER_INVALID: "correct the simulation parameters",
            FailureType.TOOL_FAILURE: "check the tool installation and its inputs",
            FailureType.SIMULATION_DIVERGED: "re-equilibrate, shorten the timestep, or revisit the force field",
            FailureType.INSUFFICIENT_SAMPLING: "extend the run or add replicas",
            FailureType.NOT_CONVERGED: "extend the run; the observable has not settled",
            FailureType.REPLICA_DISAGREEMENT: "investigate the outlying replica before averaging",
            FailureType.QM_NOT_CONVERGED: "loosen the initial guess, change the SCF settings, or re-examine the geometry",
            FailureType.POOR_OVERLAP: "insert windows or raise the restraint force constant",
            FailureType.RESOURCE_EXHAUSTED: "reduce the resource request or wait for capacity",
            FailureType.TIMEOUT: "raise the timeout or shorten the job",
            FailureType.UNKNOWN: "inspect the logs",
        }[self]


@dataclass
class FailureRecord:
    """One failed attempt, in enough detail to learn from."""

    failure_type: FailureType
    cause: str
    simulation_kind: str
    polymer_family: PolymerFamily = PolymerFamily.UNCLASSIFIED
    polymer_id: str | None = None
    force_field: str | None = None
    strategy_id: str | None = None
    campaign_id: str | None = None
    resource_cost_gpu_hours: float = 0.0
    resource_cost_cpu_hours: float = 0.0
    recoverable: bool | None = None
    timestamp: str = field(default_factory=lambda: utc_now().isoformat())

    def __post_init__(self) -> None:
        if self.recoverable is None:
            self.recoverable = self.failure_type.recoverable

    @property
    def wasted_cost(self) -> float:
        """GPU-equivalent hours spent on something that produced no usable result."""
        return self.resource_cost_gpu_hours + self.resource_cost_cpu_hours / 8.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "cause": self.cause,
            "simulation_kind": self.simulation_kind,
            "polymer_family": self.polymer_family.value,
            "polymer_id": self.polymer_id,
            "force_field": self.force_field,
            "strategy_id": self.strategy_id,
            "campaign_id": self.campaign_id,
            "resource_cost_gpu_hours": self.resource_cost_gpu_hours,
            "resource_cost_cpu_hours": self.resource_cost_cpu_hours,
            "wasted_cost": self.wasted_cost,
            "recoverable": self.recoverable,
            "suggested_action": self.failure_type.suggested_action,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FailureRecord:
        return cls(
            failure_type=FailureType(payload["failure_type"]),
            cause=payload.get("cause", ""),
            simulation_kind=payload.get("simulation_kind", "unknown"),
            polymer_family=PolymerFamily(payload.get("polymer_family", "unclassified")),
            polymer_id=payload.get("polymer_id"),
            force_field=payload.get("force_field"),
            strategy_id=payload.get("strategy_id"),
            campaign_id=payload.get("campaign_id"),
            resource_cost_gpu_hours=float(payload.get("resource_cost_gpu_hours", 0.0)),
            resource_cost_cpu_hours=float(payload.get("resource_cost_cpu_hours", 0.0)),
            recoverable=payload.get("recoverable"),
            timestamp=payload.get("timestamp", utc_now().isoformat()),
        )


@dataclass
class FailurePattern:
    """A repeated failure the engine has enough evidence to act on."""

    scope: str
    key: str
    failure_type: FailureType
    occurrences: int
    total_wasted_cost: float
    determination: Determination
    recommendation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "key": self.key,
            "failure_type": self.failure_type.value,
            "occurrences": self.occurrences,
            "total_wasted_cost": self.total_wasted_cost,
            "determination": self.determination.value,
            "recommendation": self.recommendation,
        }


#: Repeats needed before a coincidence is treated as a pattern.
MIN_OCCURRENCES_FOR_PATTERN = 3


class FailureLedger:
    """Records failures and reports the patterns in them."""

    def __init__(self, records: Sequence[FailureRecord] = ()) -> None:
        self._records: list[FailureRecord] = list(records)

    def record(self, record: FailureRecord) -> FailureRecord:
        self._records.append(record)
        logger.info(
            "Recorded %s failure for %s (%s): %s",
            record.failure_type.value, record.polymer_family.value,
            record.simulation_kind, record.cause[:80],
        )
        return record

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> list[FailureRecord]:
        return list(self._records)

    def by_family(self, family: PolymerFamily) -> list[FailureRecord]:
        return [r for r in self._records if r.polymer_family is family]

    def by_strategy(self, strategy_id: str) -> list[FailureRecord]:
        return [r for r in self._records if r.strategy_id == strategy_id]

    def total_wasted_cost(self) -> float:
        return sum(r.wasted_cost for r in self._records)

    def patterns(self, *, min_occurrences: int = MIN_OCCURRENCES_FOR_PATTERN) -> list[FailurePattern]:
        """Repeated failures worth changing behaviour over.

        Below ``min_occurrences`` a repeat is a coincidence, and acting on it would make
        the engine superstitious about a single bad run.
        """
        found: list[FailurePattern] = []
        for scope, key_of in (
            ("strategy+family", lambda r: f"{r.strategy_id}|{r.polymer_family.value}"),
            ("force_field+family", lambda r: f"{r.force_field}|{r.polymer_family.value}"),
            ("simulation_kind+family", lambda r: f"{r.simulation_kind}|{r.polymer_family.value}"),
        ):
            buckets: dict[tuple[str, FailureType], list[FailureRecord]] = defaultdict(list)
            for record in self._records:
                key = key_of(record)
                if "None" in key:
                    continue
                buckets[(key, record.failure_type)].append(record)
            for (key, failure_type), records in buckets.items():
                if len(records) < min_occurrences:
                    continue
                found.append(
                    FailurePattern(
                        scope=scope,
                        key=key,
                        failure_type=failure_type,
                        occurrences=len(records),
                        total_wasted_cost=sum(r.wasted_cost for r in records),
                        determination=Determination.KNOWN,
                        recommendation=(
                            f"{key.replace('|', ' applied to ')} has failed {len(records)} times with "
                            f"{failure_type.value}; {failure_type.suggested_action}"
                        ),
                    )
                )
        return sorted(found, key=lambda p: (-p.occurrences, -p.total_wasted_cost))

    def penalty_for(
        self, *, strategy_id: str | None, family: PolymerFamily, min_occurrences: int = MIN_OCCURRENCES_FOR_PATTERN
    ) -> float:
        """A score penalty in [0, 1) for a strategy repeatedly failing on this family.

        Deprioritises, never forbids: a strategy that failed on three polyesters may
        still be right for the fourth, and the engine must not lock itself out of it.
        """
        if strategy_id is None:
            return 0.0
        matching = [
            r for r in self._records
            if r.strategy_id == strategy_id and r.polymer_family is family
        ]
        if len(matching) < min_occurrences:
            return 0.0
        unrecoverable = sum(1 for r in matching if not r.recoverable)
        # Saturating, so a long history cannot drive the score to zero.
        return min(0.6, 0.15 * len(matching) + 0.05 * unrecoverable)

    def report(self) -> dict[str, Any]:
        by_type: dict[str, int] = defaultdict(int)
        for record in self._records:
            by_type[record.failure_type.value] += 1
        return {
            "n_failures": len(self._records),
            "total_wasted_cost_gpu_hours": round(self.total_wasted_cost(), 3),
            "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
            "recoverable": sum(1 for r in self._records if r.recoverable),
            "unrecoverable": sum(1 for r in self._records if not r.recoverable),
            "patterns": [p.as_dict() for p in self.patterns()],
        }

    # -- persistence -----------------------------------------------------
    def save(self, store: Any) -> None:
        for record in self._records:
            store.log_event("failure_recorded", record.as_dict())

    @classmethod
    def load(cls, store: Any, *, limit: int = 1000) -> FailureLedger:
        events = store.list_events("failure_recorded", limit=limit)
        return cls([FailureRecord.from_dict(event["payload"]) for event in events])


def classify_failure(
    *,
    gate_messages: Sequence[str] = (),
    error: str | None = None,
    execution_mode: str = "real",
) -> FailureType:
    """Infer a failure type from gate messages and an error string.

    Pattern matching on messages is fragile by nature, so an unrecognised failure maps
    to ``UNKNOWN`` rather than to a plausible-looking guess.
    """
    haystack = " ".join([*gate_messages, error or ""]).lower()
    if execution_mode == "blocked":
        if "system" in haystack or "topology" in haystack or "coordinate" in haystack:
            return FailureType.SYSTEM_INVALID
        if "parameter" in haystack:
            return FailureType.PARAMETER_INVALID
    for needle, failure_type in (
        ("timed out", FailureType.TIMEOUT),
        ("timeout", FailureType.TIMEOUT),
        ("resource", FailureType.RESOURCE_EXHAUSTED),
        ("overlap", FailureType.POOR_OVERLAP),
        ("replicas disagree", FailureType.REPLICA_DISAGREEMENT),
        ("effective samples", FailureType.INSUFFICIENT_SAMPLING),
        # More specific patterns first: "SCF did not converge" must classify as a QM
        # failure, not as the generic non-convergence of an MD observable.
        ("scf", FailureType.QM_NOT_CONVERGED),
        ("wavefunction", FailureType.QM_NOT_CONVERGED),
        ("still drifting", FailureType.NOT_CONVERGED),
        ("did not converge", FailureType.NOT_CONVERGED),
        ("lincs", FailureType.SIMULATION_DIVERGED),
        ("nan", FailureType.SIMULATION_DIVERGED),
        ("grompp failed", FailureType.TOOL_FAILURE),
        ("mdrun failed", FailureType.TOOL_FAILURE),
        ("topology", FailureType.SYSTEM_INVALID),
        ("coordinates", FailureType.SYSTEM_INVALID),
    ):
        if needle in haystack:
            return failure_type
    return FailureType.UNKNOWN


__all__ = [
    "MIN_OCCURRENCES_FOR_PATTERN",
    "FailureLedger",
    "FailurePattern",
    "FailureRecord",
    "FailureType",
    "classify_failure",
]
