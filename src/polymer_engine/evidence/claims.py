"""Scientific claims and the evidence behind them.

A claim is a statement the engine might make about the world.  It is only ever as
strong as the evidence attached to it, and the promotion rules here are deliberately
hard to satisfy:

* ``SUPPORTED`` requires corroborating evidence that itself **passed** its validation
  gates, carries uncertainty, and is reproduced across independent replicas -- and no
  refuting evidence.
* Any refuting evidence at all prevents ``SUPPORTED``; the claim becomes
  ``PARTIALLY_SUPPORTED`` or ``CONTRADICTED`` depending on the balance.
* Everything else is ``INSUFFICIENT_EVIDENCE``, which is the default and the resting
  state.  A claim does not drift upward through accumulated near-misses.

:meth:`Claim.evaluate` is the only way the status changes, and it recomputes from the
evidence every time, so a status cannot be set by hand and then quietly kept.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from polymer_engine.core.errors import PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import GateStatus, Measurement, utc_now

logger = get_logger("evidence.claims")


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceKind(str, Enum):
    SIMULATION = "simulation"
    ANALYSIS = "analysis"
    LITERATURE = "literature"
    EXPERIMENT = "experiment"
    MODEL_PREDICTION = "model_prediction"


class Stance(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


#: Independent replicates required before a simulation claim may be ``SUPPORTED``.
MIN_INDEPENDENT_REPLICATES = 3


@dataclass
class Evidence:
    """One piece of evidence bearing on a claim."""

    evidence_id: str
    kind: EvidenceKind
    stance: Stance
    description: str
    measurement: Measurement | None = None
    artifact_ids: list[str] = field(default_factory=list)
    gate_status: GateStatus | None = None
    n_independent_replicates: int = 0
    source: str = ""
    campaign_id: str | None = None
    created_at: str = field(default_factory=lambda: utc_now().isoformat())

    @property
    def has_uncertainty(self) -> bool:
        return self.measurement is not None and self.measurement.uncertainty is not None

    @property
    def validated(self) -> bool:
        """Whether this evidence passed its own validation gates.

        A missing gate status is *not* validated: unrun checks are not passed checks.
        """
        return self.gate_status is GateStatus.PASS

    @property
    def usable(self) -> bool:
        """Strong enough to move a claim to ``SUPPORTED`` on its own merits."""
        if self.kind in (EvidenceKind.SIMULATION, EvidenceKind.ANALYSIS):
            return (
                self.validated
                and self.has_uncertainty
                and self.n_independent_replicates >= MIN_INDEPENDENT_REPLICATES
            )
        if self.kind is EvidenceKind.MODEL_PREDICTION:
            # A surrogate prediction is a reason to run something, never a result.
            return False
        if self.kind in (EvidenceKind.LITERATURE, EvidenceKind.EXPERIMENT):
            return bool(self.source)
        return False

    def weaknesses(self) -> list[str]:
        """Why this evidence is not usable, in words."""
        problems: list[str] = []
        if self.kind is EvidenceKind.MODEL_PREDICTION:
            problems.append("a surrogate model prediction is not evidence about the world")
            return problems
        if self.kind in (EvidenceKind.SIMULATION, EvidenceKind.ANALYSIS):
            if self.gate_status is None:
                problems.append("validation gates were never run")
            elif self.gate_status is not GateStatus.PASS:
                problems.append(f"validation gates returned {self.gate_status.value}")
            if not self.has_uncertainty:
                problems.append("no uncertainty was estimated")
            if self.n_independent_replicates < MIN_INDEPENDENT_REPLICATES:
                problems.append(
                    f"only {self.n_independent_replicates} independent replicate(s); "
                    f"{MIN_INDEPENDENT_REPLICATES} required"
                )
        if self.kind in (EvidenceKind.LITERATURE, EvidenceKind.EXPERIMENT) and not self.source:
            problems.append("no citable source recorded")
        return problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "stance": self.stance.value,
            "description": self.description,
            "measurement": self.measurement.model_dump(mode="json") if self.measurement else None,
            "artifact_ids": self.artifact_ids,
            "gate_status": self.gate_status.value if self.gate_status else None,
            "n_independent_replicates": self.n_independent_replicates,
            "source": self.source,
            "campaign_id": self.campaign_id,
            "created_at": self.created_at,
            "usable": self.usable,
            "weaknesses": self.weaknesses(),
        }


@dataclass
class Claim:
    """A statement plus the evidence that determines its standing."""

    claim_id: str
    statement: str
    hypothesis_id: str | None = None
    campaign_id: str | None = None
    evidence: list[Evidence] = field(default_factory=list)
    status: ClaimStatus = ClaimStatus.INSUFFICIENT_EVIDENCE
    rationale: str = "no evidence has been attached"
    literature_comparison: str = ""
    created_at: str = field(default_factory=lambda: utc_now().isoformat())
    updated_at: str = field(default_factory=lambda: utc_now().isoformat())

    def add_evidence(self, evidence: Evidence) -> Claim:
        if any(e.evidence_id == evidence.evidence_id for e in self.evidence):
            raise PolymerEngineError(
                "Evidence with this id is already attached", claim_id=self.claim_id,
                evidence_id=evidence.evidence_id,
            )
        self.evidence.append(evidence)
        self.evaluate()
        return self

    # -- the only path to a status -------------------------------------
    def evaluate(self) -> ClaimStatus:
        """Recompute the status from the attached evidence."""
        supporting = [e for e in self.evidence if e.stance is Stance.SUPPORTS]
        refuting = [e for e in self.evidence if e.stance is Stance.REFUTES]
        usable_support = [e for e in supporting if e.usable]
        usable_refutation = [e for e in refuting if e.usable]

        if not self.evidence:
            self.status = ClaimStatus.INSUFFICIENT_EVIDENCE
            self.rationale = "no evidence has been attached"
        elif usable_refutation and not usable_support:
            self.status = ClaimStatus.CONTRADICTED
            self.rationale = (
                f"{len(usable_refutation)} validated piece(s) of refuting evidence and no validated support"
            )
        elif usable_refutation and usable_support:
            self.status = ClaimStatus.PARTIALLY_SUPPORTED
            self.rationale = (
                f"{len(usable_support)} validated supporting and {len(usable_refutation)} validated "
                "refuting piece(s); the evidence conflicts"
            )
        elif usable_support and refuting:
            # Refuting evidence that is not itself validated still blocks a clean claim.
            self.status = ClaimStatus.PARTIALLY_SUPPORTED
            self.rationale = (
                f"{len(usable_support)} validated supporting piece(s), but {len(refuting)} "
                "unresolved refuting observation(s) remain"
            )
        elif usable_support:
            self.status = ClaimStatus.SUPPORTED
            self.rationale = (
                f"{len(usable_support)} validated, replicated supporting piece(s) with uncertainty; "
                "no refuting evidence"
            )
        else:
            self.status = ClaimStatus.INSUFFICIENT_EVIDENCE
            weaknesses = sorted({w for e in supporting for w in e.weaknesses()})
            self.rationale = (
                "evidence is attached but none of it is usable: " + "; ".join(weaknesses)
                if weaknesses
                else "no usable evidence"
            )
        self.updated_at = utc_now().isoformat()
        return self.status

    @property
    def promoted(self) -> bool:
        return self.status is ClaimStatus.SUPPORTED

    def supporting_artifacts(self) -> list[str]:
        return sorted({a for e in self.evidence for a in e.artifact_ids})

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "hypothesis_id": self.hypothesis_id,
            "campaign_id": self.campaign_id,
            "status": self.status.value,
            "rationale": self.rationale,
            "literature_comparison": self.literature_comparison,
            "evidence": [e.as_dict() for e in self.evidence],
            "supporting_artifacts": self.supporting_artifacts(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Claim:
        evidence = []
        for item in payload.get("evidence", []):
            measurement = item.get("measurement")
            evidence.append(
                Evidence(
                    evidence_id=item["evidence_id"],
                    kind=EvidenceKind(item["kind"]),
                    stance=Stance(item["stance"]),
                    description=item.get("description", ""),
                    measurement=Measurement.model_validate(measurement) if measurement else None,
                    artifact_ids=item.get("artifact_ids", []),
                    gate_status=GateStatus(item["gate_status"]) if item.get("gate_status") else None,
                    n_independent_replicates=int(item.get("n_independent_replicates", 0)),
                    source=item.get("source", ""),
                    campaign_id=item.get("campaign_id"),
                    created_at=item.get("created_at", utc_now().isoformat()),
                )
            )
        claim = cls(
            claim_id=payload["claim_id"],
            statement=payload["statement"],
            hypothesis_id=payload.get("hypothesis_id"),
            campaign_id=payload.get("campaign_id"),
            evidence=evidence,
            literature_comparison=payload.get("literature_comparison", ""),
            created_at=payload.get("created_at", utc_now().isoformat()),
        )
        # Recompute rather than trusting a persisted status.
        claim.evaluate()
        return claim


class ClaimLedger:
    """All claims in a project, with a report of what is actually established."""

    def __init__(self, claims: Iterable[Claim] = ()) -> None:
        self._claims: dict[str, Claim] = {c.claim_id: c for c in claims}

    def add(self, claim: Claim) -> Claim:
        claim.evaluate()
        self._claims[claim.claim_id] = claim
        return claim

    def get(self, claim_id: str) -> Claim:
        try:
            return self._claims[claim_id]
        except KeyError:
            raise PolymerEngineError("Unknown claim", claim_id=claim_id) from None

    def __len__(self) -> int:
        return len(self._claims)

    def __iter__(self):
        return iter(self._claims.values())

    def by_status(self, status: ClaimStatus) -> list[Claim]:
        return [c for c in self._claims.values() if c.status is status]

    def supported(self) -> list[Claim]:
        return self.by_status(ClaimStatus.SUPPORTED)

    def report(self) -> dict[str, Any]:
        counts = {status.value: len(self.by_status(status)) for status in ClaimStatus}
        return {
            "n_claims": len(self._claims),
            "counts": counts,
            "claims": [c.as_dict() for c in self._claims.values()],
        }

    def save(self, store: Any) -> None:
        for claim in self._claims.values():
            store.save_claim(claim.claim_id, claim.status.value, claim.as_dict())

    @classmethod
    def load(cls, store: Any) -> ClaimLedger:
        return cls(Claim.from_dict(p) for p in store.list_claims())


def evidence_from_gate_report(
    evidence_id: str,
    report: Any,
    *,
    stance: Stance,
    description: str,
    measurement: Measurement | None = None,
    n_independent_replicates: int = 0,
    artifact_ids: Sequence[str] = (),
    campaign_id: str | None = None,
) -> Evidence:
    """Build evidence from a :class:`GateReport`, carrying its verdict across."""
    return Evidence(
        evidence_id=evidence_id,
        kind=EvidenceKind.SIMULATION,
        stance=stance,
        description=description,
        measurement=measurement,
        artifact_ids=list(artifact_ids),
        gate_status=report.status,
        n_independent_replicates=n_independent_replicates,
        campaign_id=campaign_id,
    )


__all__ = [
    "MIN_INDEPENDENT_REPLICATES",
    "Claim",
    "ClaimLedger",
    "ClaimStatus",
    "Evidence",
    "EvidenceKind",
    "Stance",
    "evidence_from_gate_report",
]
