"""The build queue, and the record that stops the same build being submitted twice.

CHARMM-GUI is a shared academic service. Submitting a build that already exists wastes
somebody else's compute as well as ours, and doing it by accident -- after a crash, a
timeout, or a restart -- is easy, because the moment a submission is in flight the
engine may not know whether it succeeded.

The rule this module enforces is asymmetric on purpose (§52):

* an existing job with the same request fingerprint is **reused**, never duplicated;
* a job whose state is in flight is **monitored**, never resubmitted;
* a submission whose outcome is unknown is **not** resubmitted. "We did not see a job
  id" is not evidence that no job was created, and resubmitting on that basis is how one
  request becomes three.

The only path that permits a fresh submission after a failure is one where the failure
proves nothing was created -- a refusal before the click, or a verification that blocked
it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polymer_engine.browser.states import BrowserState, JobState
from polymer_engine.core.logging import get_logger

logger = get_logger("browser.queue")

QUEUE_PATH = Path("campaign/charmm_gui/build_queue.json")

#: States that prove no job was created, so a fresh submission is safe.
NOTHING_SUBMITTED: frozenset[BrowserState] = frozenset({
    BrowserState.REFUSED, BrowserState.CREDENTIALS_MISSING,
    BrowserState.UI_SCHEMA_MISMATCH, BrowserState.STRUCTURE_MISMATCH,
    BrowserState.MONOMER_NOT_FOUND, BrowserState.NAVIGATION_FAILED,
    BrowserState.AUTHENTICATION_FAILED,
})


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class BuildEntry:
    """One requested build, from rationale to qualification."""

    polymer_id: str
    polymer_name: str
    request_fingerprint: str
    system_type: str
    #: Why this build is worth the compute. Free text, but required: a build queued
    #: without a stated reason is a build nobody can defend later.
    rationale: str
    priority: int = 50
    state: JobState = JobState.QUEUED
    job_id: str | None = None
    spec: dict[str, Any] = field(default_factory=dict)
    archive_path: str | None = None
    archive_sha256: str | None = None
    force_field: str | None = None
    parameterization_state: str | None = None
    validation_state: str | None = None
    diagnostics: list[str] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def advance(self, state: JobState, reason: str) -> None:
        self.history.append({"from": self.state.value, "to": state.value,
                             "reason": reason, "at": _now()})
        self.state = state
        self.updated_at = _now()
        logger.info("build %s (%s): %s -> %s (%s)", self.polymer_name,
                    self.request_fingerprint[:12], self.history[-1]["from"],
                    state.value, reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id, "polymer_name": self.polymer_name,
            "request_fingerprint": self.request_fingerprint,
            "system_type": self.system_type, "rationale": self.rationale,
            "priority": self.priority, "state": self.state.value,
            "job_id": self.job_id, "spec": dict(self.spec),
            "archive_path": self.archive_path, "archive_sha256": self.archive_sha256,
            "force_field": self.force_field,
            "parameterization_state": self.parameterization_state,
            "validation_state": self.validation_state,
            "diagnostics": list(self.diagnostics), "history": list(self.history),
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuildEntry:
        entry = cls(
            polymer_id=data["polymer_id"], polymer_name=data["polymer_name"],
            request_fingerprint=data["request_fingerprint"],
            system_type=data.get("system_type", "melt"),
            rationale=data.get("rationale", ""), priority=int(data.get("priority", 50)),
            state=JobState(data.get("state", "QUEUED")), job_id=data.get("job_id"),
            spec=dict(data.get("spec") or {}), archive_path=data.get("archive_path"),
            archive_sha256=data.get("archive_sha256"),
            force_field=data.get("force_field"),
            parameterization_state=data.get("parameterization_state"),
            validation_state=data.get("validation_state"),
            diagnostics=list(data.get("diagnostics") or []),
            history=list(data.get("history") or []),
        )
        entry.created_at = data.get("created_at", entry.created_at)
        entry.updated_at = data.get("updated_at", entry.updated_at)
        return entry


@dataclass
class DuplicateCheck:
    """Whether a build may be submitted, and what to do instead if not."""

    may_submit: bool
    reason: str
    existing: BuildEntry | None = None
    action: str = "submit"

    def as_dict(self) -> dict[str, Any]:
        return {"may_submit": self.may_submit, "reason": self.reason,
                "action": self.action,
                "existing_job_id": self.existing.job_id if self.existing else None,
                "existing_state": self.existing.state.value if self.existing else None}


class BuildQueue:
    """A persistent, fingerprint-indexed queue of CHARMM-GUI builds."""

    def __init__(self, path: str | Path = QUEUE_PATH) -> None:
        self.path = Path(path)
        self.entries: list[BuildEntry] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self.entries = [BuildEntry.from_dict(e) for e in data.get("entries", [])]

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "charmm_gui.build_queue/1", "updated_at": _now(),
            "n_entries": len(self.entries),
            "counts": self.counts(),
            "entries": [e.as_dict() for e in self.entries],
        }
        # Written atomically: a truncated queue read after a crash would look like a
        # queue with no in-flight jobs, which is precisely the state that permits a
        # duplicate submission.
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self.path)
        return self.path

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.state.value] = counts.get(entry.state.value, 0) + 1
        return dict(sorted(counts.items()))

    def find(self, fingerprint: str) -> BuildEntry | None:
        return next((e for e in self.entries
                     if e.request_fingerprint == fingerprint), None)

    def by_job(self, job_id: str) -> BuildEntry | None:
        return next((e for e in self.entries if e.job_id == job_id), None)

    def add(self, entry: BuildEntry) -> BuildEntry:
        """Queue a build, or return the existing entry for the same request."""
        existing = self.find(entry.request_fingerprint)
        if existing is not None:
            logger.info("build %s is already queued as %s; not duplicated",
                        entry.request_fingerprint[:12], existing.state.value)
            return existing
        if not entry.rationale.strip():
            raise ValueError(
                "a queued build needs a stated scientific rationale; compute spent "
                "without one cannot be defended afterwards"
            )
        self.entries.append(entry)
        return entry

    def may_submit(self, fingerprint: str) -> DuplicateCheck:
        """Decide whether this exact build may be sent to CHARMM-GUI (§51, §52)."""
        existing = self.find(fingerprint)
        if existing is None:
            return DuplicateCheck(True, "no build with this fingerprint exists")
        if existing.state is JobState.QUEUED:
            return DuplicateCheck(True, "queued but never submitted", existing)
        if existing.state.in_flight:
            return DuplicateCheck(
                False,
                f"job {existing.job_id or '(id unknown)'} for this exact request is "
                f"{existing.state.value}; monitor it rather than submitting again",
                existing, action="monitor",
            )
        if existing.state in {JobState.DOWNLOADED, JobState.VALIDATED}:
            return DuplicateCheck(
                False,
                f"job {existing.job_id} for this exact request is already "
                f"{existing.state.value}; reuse it",
                existing, action="reuse",
            )
        if existing.state is JobState.ERROR:
            return DuplicateCheck(
                False,
                f"job {existing.job_id} for this request failed on CHARMM-GUI's side. "
                f"Resubmitting an identical request would fail identically; the "
                f"specification has to change first",
                existing, action="requires_review",
            )
        if existing.state is JobState.FAILED_VALIDATION:
            return DuplicateCheck(
                False,
                f"job {existing.job_id} was built but failed local validation. The "
                f"problem is in the parameters, not the submission",
                existing, action="requires_review",
            )
        return DuplicateCheck(
            False, f"a build with this fingerprint is {existing.state.value}",
            existing, action="requires_review",
        )

    def record_submission_failure(
        self, entry: BuildEntry, state: BrowserState, reason: str
    ) -> None:
        """Record a failed submission, deciding whether a retry is even permissible.

        The distinction that matters: a failure *before* the click leaves the queue
        entry retryable; a failure after it does not, because the job may exist.
        """
        if state in NOTHING_SUBMITTED:
            entry.advance(JobState.QUEUED,
                          f"{state.value}: {reason}. Nothing was submitted, so this "
                          f"stays queued")
        else:
            entry.advance(
                JobState.BLOCKED,
                f"{state.value}: {reason}. A job may exist on CHARMM-GUI; a person must "
                f"check before anything is resubmitted",
            )

    def next_ready(self) -> BuildEntry | None:
        """The highest-priority queued build, or None.

        Returns one at a time on purpose (§54): one build in flight against a shared
        service is the polite and the debuggable choice.
        """
        if any(e.state.in_flight for e in self.entries):
            return None
        ready = [e for e in self.entries if e.state is JobState.QUEUED]
        ready.sort(key=lambda e: (-e.priority, e.created_at))
        return ready[0] if ready else None

    def to_markdown(self) -> str:
        lines = ["# CHARMM-GUI build queue", "",
                 f"{len(self.entries)} entries · " +
                 ", ".join(f"{k} {v}" for k, v in self.counts().items()) or "empty", "",
                 "| Polymer | System | State | Job | Priority | Rationale |",
                 "|---|---|---|---|---|---|"]
        for e in sorted(self.entries, key=lambda x: (-x.priority, x.polymer_name)):
            lines.append(f"| {e.polymer_name} | {e.system_type} | {e.state.value} | "
                         f"{e.job_id or '—'} | {e.priority} | {e.rationale[:70]} |")
        return "\n".join(lines) + "\n"


__all__ = ["NOTHING_SUBMITTED", "QUEUE_PATH", "BuildEntry", "BuildQueue",
           "DuplicateCheck"]
