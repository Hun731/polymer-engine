"""What has actually been proven against the live service, and what has only been coded.

The distinction this module exists to hold is the one §26 and §27 are about. A
subsystem that runs cleanly against local fixtures has demonstrated that its *logic* is
right. It has demonstrated nothing about CHARMM-GUI. Until a real login happens, every
claim about the live service is untested -- and the registry says so in a machine-
readable way rather than leaving it to a paragraph in a report that drifts.

Two boundaries are enforced here rather than described:

* a capability moves to ``VERIFIED`` only when :meth:`verify` is handed evidence: a job
  id, an archive hash, a `grompp` exit. There is no way to assert one into existence;
* ``SYSTEM_GENERATION_VERIFIED`` is a separate, lower claim than
  ``FORCE_FIELD_QUALIFIED``, and this module cannot grant the latter at all. A build
  that produced a system says the *builder* works. Whether the parameters are right for
  a property is decided by evidence that no download can supply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from polymer_engine.core.logging import get_logger

logger = get_logger("browser.verification")

REGISTRY_PATH = Path("campaign/charmm_gui/capability_verification.json")


class VerificationState(str, Enum):
    """How much is actually known about one capability."""

    #: Code exists. Nothing has been run against anything.
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    #: Exercised against local fixtures with a real browser. Proves the logic.
    FIXTURE_VERIFIED = "FIXTURE_VERIFIED"
    #: Proven against the live service, with evidence recorded.
    LIVE_VERIFIED = "LIVE_VERIFIED"
    #: Tried against the live service and failed.
    LIVE_FAILED = "LIVE_FAILED"
    #: Cannot be attempted: a missing credential, tool or upstream capability.
    BLOCKED = "BLOCKED"
    #: A person must decide something before this can progress.
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"

    @property
    def proven_live(self) -> bool:
        return self is VerificationState.LIVE_VERIFIED


#: Every capability, in a reading order for reports.
#:
#: This is **presentation order only**. It was once used as the dependency relation --
#: "everything listed earlier is a prerequisite" -- which is wrong, and wrong in a way
#: that is easy to miss: a flat list forces a total order onto what is really a graph
#: with siblings, so discovering a form appeared to require discovering a catalogue,
#: and the failure message talked about downloads and job ids. Dependencies now live in
#: :data:`PREREQUISITES`, and this tuple orders nothing but the report.
CAPABILITY_ORDER: tuple[str, ...] = (
    "LIVE_LOGIN_VERIFIED",
    "POLYMER_BUILDER_REACHED",
    "FORM_SCHEMA_VERIFIED",
    "CATALOG_VERIFIED",
    "SPEC_MAPPING_VERIFIED",
    "SINGLE_CHAIN_BUILD_VERIFIED",
    "MELT_BUILD_VERIFIED",
    "JOB_ID_VERIFIED",
    "JOB_MONITORING_VERIFIED",
    "DOWNLOAD_VERIFIED",
    "SYSTEM_IMPORT_VERIFIED",
    "GROMACS_VALIDATION_VERIFIED",
    "SYSTEM_GENERATION_VERIFIED",
)

#: What each capability actually needs, as a directed acyclic graph.
#:
#: The shape that matters: ``FORM_SCHEMA_VERIFIED`` and ``CATALOG_VERIFIED`` are
#: **siblings**. Both are read off the same page and neither depends on the other, so a
#: page whose monomer control cannot be identified can still prove its form schema. And
#: nothing in the discovery half depends on anything in the build half -- a catalogue is
#: evidence about a page, not about a job.
PREREQUISITES: dict[str, tuple[str, ...]] = {
    "LIVE_LOGIN_VERIFIED": (),
    "POLYMER_BUILDER_REACHED": ("LIVE_LOGIN_VERIFIED",),
    "FORM_SCHEMA_VERIFIED": ("POLYMER_BUILDER_REACHED",),
    "CATALOG_VERIFIED": ("POLYMER_BUILDER_REACHED",),
    "SPEC_MAPPING_VERIFIED": ("FORM_SCHEMA_VERIFIED", "CATALOG_VERIFIED"),
    "SINGLE_CHAIN_BUILD_VERIFIED": ("SPEC_MAPPING_VERIFIED",),
    "MELT_BUILD_VERIFIED": ("SPEC_MAPPING_VERIFIED",),
    "JOB_ID_VERIFIED": (),
    "JOB_MONITORING_VERIFIED": ("JOB_ID_VERIFIED",),
    "DOWNLOAD_VERIFIED": ("JOB_MONITORING_VERIFIED",),
    "SYSTEM_IMPORT_VERIFIED": ("DOWNLOAD_VERIFIED",),
    "GROMACS_VALIDATION_VERIFIED": ("SYSTEM_IMPORT_VERIFIED",),
    "SYSTEM_GENERATION_VERIFIED": ("GROMACS_VALIDATION_VERIFIED",),
}

#: Capabilities satisfied by *any one* of a set, rather than all of it. A single chain
#: and a melt are alternative ways of having built something; requiring both would leave
#: a site that only offers melts unable to prove it can download its own jobs.
ANY_OF: dict[str, tuple[str, ...]] = {
    "JOB_ID_VERIFIED": ("SINGLE_CHAIN_BUILD_VERIFIED", "MELT_BUILD_VERIFIED"),
    "SYSTEM_GENERATION_VERIFIED": ("SINGLE_CHAIN_BUILD_VERIFIED",
                                   "MELT_BUILD_VERIFIED"),
}


def dependency_problems() -> list[str]:
    """Audit the graph for unknown names and cycles.

    Run as a test rather than trusted. A cycle here would make a capability
    unreachable and the reason unreadable, which is exactly the class of defect the
    flat-list version shipped with.
    """
    problems: list[str] = []
    known = set(CAPABILITY_ORDER)
    for name, deps in PREREQUISITES.items():
        if name not in known:
            problems.append(f"{name} has prerequisites but is not a capability")
        problems += [f"{name} requires unknown capability {d}" for d in deps
                     if d not in known]
    for name, group in ANY_OF.items():
        problems += [f"{name} any-of names unknown capability {d}" for d in group
                     if d not in known]
    problems += [f"{name} has no entry in PREREQUISITES" for name in CAPABILITY_ORDER
                 if name not in PREREQUISITES]

    # Depth-first cycle detection over both edge kinds.
    colour: dict[str, int] = {}

    def visit(node: str, path: list[str]) -> None:
        if colour.get(node) == 1:
            problems.append("cycle: " + " -> ".join([*path, node]))
            return
        if colour.get(node) == 2:
            return
        colour[node] = 1
        for nxt in (*PREREQUISITES.get(node, ()), *ANY_OF.get(node, ())):
            if nxt in known:
                visit(nxt, [*path, node])
        colour[node] = 2

    for name in CAPABILITY_ORDER:
        visit(name, [])
    return problems

#: What each capability means, and what counts as evidence for it. Written down because
#: "verified" with no stated criterion is how a checklist becomes decoration.
EVIDENCE_REQUIRED: dict[str, str] = {
    "LIVE_LOGIN_VERIFIED":
        "an authenticated session on charmm-gui.org: the login form is gone and no "
        "human-verification challenge is present",
    "POLYMER_BUILDER_REACHED":
        "the Polymer Builder page loaded in an authenticated session, identified by "
        "its own content rather than by its URL",
    "SPEC_MAPPING_VERIFIED":
        "every field a build needs mapped to exactly one discovered control, with no "
        "ambiguity left unresolved",
    "JOB_ID_VERIFIED":
        "a real job identifier read from the site after a submission",
    "CATALOG_VERIFIED":
        "a monomer catalogue captured from the live Polymer Builder page, with a "
        "fingerprint and at least one selectable monomer",
    "FORM_SCHEMA_VERIFIED":
        "a form schema discovered from the live page, with locators for every field a "
        "build requires",
    "SINGLE_CHAIN_BUILD_VERIFIED":
        "a real job id returned by the live site for a single-chain build whose form "
        "state was read back and agreed with the request",
    "MELT_BUILD_VERIFIED":
        "the same, for a melt system",
    "JOB_MONITORING_VERIFIED":
        "a documented status response for a real job id",
    "DOWNLOAD_VERIFIED":
        "an archive downloaded for a real job id, opening as a readable tar, with a "
        "recorded sha256",
    "SYSTEM_IMPORT_VERIFIED":
        "the archive extracted safely and its topology, coordinates and parameter "
        "files discovered and hashed",
    "GROMACS_VALIDATION_VERIFIED":
        "a real grompp accepting the imported system with no warnings suppressed, and "
        "a short mdrun completing",
    "SYSTEM_GENERATION_VERIFIED":
        "every capability above, for at least one polymer. This states that the "
        "acquisition route works -- NOT that the force field is qualified for anything",
}

#: Claims this registry is structurally unable to make. Qualification needs property
#: evidence from a campaign; no amount of successful downloading produces it.
NEVER_GRANTED_HERE: frozenset[str] = frozenset({
    "FORCE_FIELD_QUALIFIED", "QUALIFIED_FOR_DENSITY", "QUALIFIED_FOR_DYNAMICS",
    "QUALIFIED_FOR_FREE_ENERGY", "QUALIFIED_FOR_MECHANICS",
})


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class CapabilityRecord:
    """One capability, its state, and the evidence behind that state."""

    name: str
    state: VerificationState = VerificationState.NOT_IMPLEMENTED
    reason: str = ""
    #: Hashes, job ids, exit codes -- whatever actually demonstrates the claim.
    evidence: dict[str, Any] = field(default_factory=dict)
    verified_at: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    @property
    def requirement(self) -> str:
        return EVIDENCE_REQUIRED.get(self.name, "")

    def as_dict(self) -> dict[str, Any]:
        return {"capability": self.name, "state": self.state.value,
                "reason": self.reason, "evidence": dict(self.evidence),
                "verified_at": self.verified_at, "requirement": self.requirement,
                "history": list(self.history)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityRecord:
        return cls(name=data["capability"],
                   state=VerificationState(data.get("state", "NOT_IMPLEMENTED")),
                   reason=data.get("reason", ""),
                   evidence=dict(data.get("evidence") or {}),
                   verified_at=data.get("verified_at"),
                   history=list(data.get("history") or []))


class VerificationRegistry:
    """The record of what has been proven, and what evidence proved it."""

    def __init__(self, path: str | Path = REGISTRY_PATH) -> None:
        self.path = Path(path)
        self.records: dict[str, CapabilityRecord] = {
            name: CapabilityRecord(name) for name in CAPABILITY_ORDER
        }
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text())
        for entry in payload.get("capabilities", []):
            record = CapabilityRecord.from_dict(entry)
            self.records[record.name] = record

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "charmm_gui.capability_verification/1",
            "updated_at": _now(),
            "n_live_verified": len(self.live_verified()),
            "n_capabilities": len(self.records),
            "system_generation_verified": self.system_generation_verified,
            "force_field_qualified": (
                "not decidable here; qualification needs property evidence from a "
                "campaign, which no successful build can supply"
            ),
            "capabilities": [self.records[name].as_dict() for name in CAPABILITY_ORDER],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self.path)
        return self.path

    # -- state changes --------------------------------------------------
    def verify(
        self, name: str, *, evidence: dict[str, Any], reason: str = "",
    ) -> CapabilityRecord:
        """Mark a capability live-verified.  Refuses without evidence.

        The evidence check is not ceremony. A capability marked verified with an empty
        payload is indistinguishable from one marked verified by wishful thinking, and
        the whole point of this registry is that the difference survives.
        """
        if name in NEVER_GRANTED_HERE:
            raise ValueError(
                f"{name} cannot be granted by the acquisition registry: qualification "
                f"requires property evidence from a campaign, not a successful build"
            )
        record = self._record(name)
        if not evidence:
            raise ValueError(
                f"refusing to mark {name} verified with no evidence; it requires "
                f"{record.requirement or 'demonstration against the live service'}"
            )
        missing = self._unproven_prerequisites(name)
        if missing:
            # The reason names the actual missing edges. The previous version printed a
            # sentence about downloads and job ids no matter which capability it was.
            raise ValueError(
                f"{name} cannot be verified before {', '.join(missing)}, because it "
                f"requires {record.requirement or 'evidence from those stages'}"
            )
        return self._set(record, VerificationState.LIVE_VERIFIED,
                         reason or "demonstrated against the live service", evidence)

    def fixture_verified(self, name: str, reason: str) -> CapabilityRecord:
        """Record that the logic works against local fixtures.

        Deliberately a *lower* state than LIVE_VERIFIED, and it can never be raised to
        it without live evidence. Passing tests against a page we wrote ourselves proves
        our decisions, not the site's behaviour.
        """
        return self._set(self._record(name), VerificationState.FIXTURE_VERIFIED,
                         reason, {})

    def block(self, name: str, reason: str) -> CapabilityRecord:
        return self._set(self._record(name), VerificationState.BLOCKED, reason, {})

    def fail(self, name: str, reason: str, evidence: dict[str, Any] | None = None) -> CapabilityRecord:
        return self._set(self._record(name), VerificationState.LIVE_FAILED, reason,
                         evidence or {})

    def needs_review(self, name: str, reason: str) -> CapabilityRecord:
        return self._set(self._record(name), VerificationState.REQUIRES_HUMAN_REVIEW,
                         reason, {})

    def _record(self, name: str) -> CapabilityRecord:
        if name not in self.records:
            raise KeyError(f"unknown capability {name!r}; known: "
                           f"{', '.join(CAPABILITY_ORDER)}")
        return self.records[name]

    def _set(self, record: CapabilityRecord, state: VerificationState, reason: str,
             evidence: dict[str, Any]) -> CapabilityRecord:
        record.history.append({"from": record.state.value, "to": state.value,
                               "reason": reason, "at": _now()})
        record.state = state
        record.reason = reason
        if evidence:
            record.evidence.update(evidence)
        record.verified_at = _now() if state.proven_live else None
        logger.info("capability %s -> %s (%s)", record.name, state.value, reason[:80])
        return record

    def _unproven_prerequisites(self, name: str) -> list[str]:
        """Direct prerequisites of ``name`` that are not yet live-verified.

        Direct only. Transitive ones are covered because a prerequisite could not have
        been verified without *its* prerequisites, and reporting the whole ancestry
        would bury the one edge the caller can act on.
        """
        missing = [dep for dep in PREREQUISITES.get(name, ())
                   if not self.records[dep].state.proven_live]
        group = ANY_OF.get(name, ())
        if group and not any(self.records[c].state.proven_live for c in group):
            missing.append("a build of either kind (" + " or ".join(group) + ")")
        return missing

    # -- queries --------------------------------------------------------
    def live_verified(self) -> list[str]:
        return [n for n in CAPABILITY_ORDER if self.records[n].state.proven_live]

    @property
    def system_generation_verified(self) -> bool:
        return self.records["SYSTEM_GENERATION_VERIFIED"].state.proven_live

    def next_unproven(self) -> str | None:
        return next((n for n in CAPABILITY_ORDER
                     if not self.records[n].state.proven_live), None)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for record in self.records.values():
            counts[record.state.value] = counts.get(record.state.value, 0) + 1
        return {
            "counts": dict(sorted(counts.items())),
            "live_verified": self.live_verified(),
            "next_unproven": self.next_unproven(),
            "system_generation_verified": self.system_generation_verified,
        }

    def to_markdown(self) -> str:
        lines = ["# CHARMM-GUI capability verification", "",
                 f"{len(self.live_verified())} of {len(self.records)} capabilities "
                 f"proven against the live service.", "",
                 "| Capability | State | Evidence required | Reason |",
                 "|---|---|---|---|"]
        for name in CAPABILITY_ORDER:
            record = self.records[name]
            lines.append(f"| `{name}` | {record.state.value} | "
                         f"{record.requirement[:90]} | {record.reason[:80] or '—'} |")
        lines += ["", "## What this registry cannot say", "",
                  "`SYSTEM_GENERATION_VERIFIED` states that the acquisition route works. "
                  "It does **not** state that the force field is qualified for any "
                  "property. Qualification needs evidence a download cannot supply, and "
                  "this registry refuses to record it: "
                  + ", ".join(sorted(NEVER_GRANTED_HERE)) + ".", ""]
        return "\n".join(lines)


__all__ = [
    "ANY_OF", "CAPABILITY_ORDER", "EVIDENCE_REQUIRED", "NEVER_GRANTED_HERE",
    "PREREQUISITES", "REGISTRY_PATH", "CapabilityRecord", "VerificationRegistry",
    "VerificationState", "dependency_problems",
]
