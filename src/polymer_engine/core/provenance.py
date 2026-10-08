"""Artifact provenance and lineage.

Every file, dataset and derived number the engine produces gets an
:class:`Artifact` record carrying enough information to answer three questions:

1. Where did this come from?  (``parents``, ``source``, ``source_version``)
2. How was it made?  (``command``, ``parameters``, ``software``, ``random_seed``)
3. Is it still what it was?  (``input_hash``, ``output_hash``, :meth:`verify`)

Lineage is a DAG; :class:`ProvenanceGraph` walks it in both directions so a claim
can be traced back to a downloaded archive and an archive forward to every claim
that depends on it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import deque
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from polymer_engine.core.errors import ChecksumMismatch, PolymerEngineError
from polymer_engine.core.models import Determination, utc_now


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(payload: Any) -> str:
    """Stable digest of a JSON-serialisable structure.

    Key order is normalised so two logically identical parameter sets hash equal.
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return sha256_bytes(text.encode("utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path, datetime)):
        return str(value)
    if isinstance(value, set):
        return sorted(value, key=str)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


def environment_fingerprint() -> dict[str, str]:
    """Coarse description of the interpreter and OS.

    Deliberately excludes hostnames and usernames: provenance must be shareable.
    """
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "machine": platform.machine(),
    }


class Artifact(BaseModel):
    """A provenance record for one produced thing."""

    model_config = {"extra": "forbid"}

    artifact_id: str
    kind: str
    path: str | None = None
    parents: list[str] = Field(default_factory=list)
    source: str = "local"
    source_version: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    software: dict[str, str] = Field(default_factory=dict)
    command: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    input_hash: str | None = None
    output_hash: str | None = None
    size_bytes: int | None = None
    random_seed: int | None = None
    environment: dict[str, str] = Field(default_factory=environment_fingerprint)
    units: dict[str, str] = Field(default_factory=dict)
    validation_state: Determination = Determination.REQUIRES_VALIDATION
    notes: str = ""

    def verify(self) -> bool:
        """Re-hash ``path`` and compare with ``output_hash``.

        Returns ``True`` when they match.  Raises when the file is gone or the
        digest differs -- an artifact that changed under us is not a warning.
        """
        if self.path is None or self.output_hash is None:
            return False
        path = Path(self.path)
        if not path.exists():
            raise ChecksumMismatch("Artifact file is missing", artifact_id=self.artifact_id, path=self.path)
        actual = sha256_file(path)
        if actual != self.output_hash:
            raise ChecksumMismatch(
                "Artifact digest changed since registration",
                artifact_id=self.artifact_id,
                path=self.path,
                expected=self.output_hash,
                actual=actual,
            )
        return True


class ProvenanceGraph:
    """In-memory lineage DAG with persistence to JSON Lines."""

    def __init__(self, artifacts: Iterable[Artifact] = ()) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._children: dict[str, set[str]] = {}
        for artifact in artifacts:
            self.add(artifact)

    # -- mutation -------------------------------------------------------
    def add(self, artifact: Artifact) -> Artifact:
        existing = self._artifacts.get(artifact.artifact_id)
        if existing is not None and existing.output_hash != artifact.output_hash:
            raise PolymerEngineError(
                "Refusing to overwrite an artifact with a different digest",
                artifact_id=artifact.artifact_id,
            )
        self._artifacts[artifact.artifact_id] = artifact
        for parent in artifact.parents:
            self._children.setdefault(parent, set()).add(artifact.artifact_id)
        self._children.setdefault(artifact.artifact_id, set())
        return artifact

    def register_file(
        self,
        path: str | Path,
        *,
        kind: str,
        artifact_id: str | None = None,
        parents: Iterable[str] = (),
        source: str = "local",
        source_version: str | None = None,
        software: dict[str, str] | None = None,
        command: Iterable[str] = (),
        parameters: dict[str, Any] | None = None,
        random_seed: int | None = None,
        units: dict[str, str] | None = None,
        input_hash: str | None = None,
        validation_state: Determination = Determination.REQUIRES_VALIDATION,
        notes: str = "",
    ) -> Artifact:
        """Hash a file on disk and register it with its lineage."""
        path = Path(path)
        if not path.is_file():
            raise PolymerEngineError("Cannot register a non-file as an artifact", path=str(path))
        digest = sha256_file(path)
        artifact = Artifact(
            artifact_id=artifact_id or f"art_{digest[:16]}",
            kind=kind,
            path=str(path),
            parents=list(parents),
            source=source,
            source_version=source_version,
            software=software or {},
            command=list(command),
            parameters=parameters or {},
            input_hash=input_hash,
            output_hash=digest,
            size_bytes=path.stat().st_size,
            random_seed=random_seed,
            units=units or {},
            validation_state=validation_state,
            notes=notes,
        )
        return self.add(artifact)

    def register_derived(
        self,
        *,
        artifact_id: str,
        kind: str,
        parents: Iterable[str],
        parameters: dict[str, Any] | None = None,
        payload: Any = None,
        software: dict[str, str] | None = None,
        command: Iterable[str] = (),
        random_seed: int | None = None,
        units: dict[str, str] | None = None,
        validation_state: Determination = Determination.REQUIRES_VALIDATION,
        notes: str = "",
    ) -> Artifact:
        """Register a non-file artifact (an analysis result, a model, a decision)."""
        parents = list(parents)
        artifact = Artifact(
            artifact_id=artifact_id,
            kind=kind,
            parents=parents,
            parameters=parameters or {},
            software=software or {},
            command=list(command),
            input_hash=canonical_hash({"parents": sorted(parents), "parameters": parameters or {}}),
            output_hash=canonical_hash(payload) if payload is not None else None,
            random_seed=random_seed,
            units=units or {},
            validation_state=validation_state,
            notes=notes,
        )
        return self.add(artifact)

    # -- queries --------------------------------------------------------
    def get(self, artifact_id: str) -> Artifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError:
            raise PolymerEngineError("Unknown artifact", artifact_id=artifact_id) from None

    def __contains__(self, artifact_id: object) -> bool:
        return artifact_id in self._artifacts

    def __len__(self) -> int:
        return len(self._artifacts)

    def __iter__(self) -> Iterator[Artifact]:
        return iter(self._artifacts.values())

    def ancestors(self, artifact_id: str) -> list[str]:
        """All upstream artifact ids, nearest first, without duplicates."""
        self.get(artifact_id)
        seen: set[str] = set()
        order: list[str] = []
        queue = deque(self.get(artifact_id).parents)
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            order.append(current)
            if current in self._artifacts:
                queue.extend(self._artifacts[current].parents)
        return order

    def descendants(self, artifact_id: str) -> list[str]:
        self.get(artifact_id)
        seen: set[str] = set()
        order: list[str] = []
        queue = deque(self._children.get(artifact_id, ()))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            order.append(current)
            queue.extend(self._children.get(current, ()))
        return order

    def lineage(self, artifact_id: str) -> list[Artifact]:
        """The artifact plus every ancestor, resolved to records we actually hold."""
        ids = [artifact_id, *self.ancestors(artifact_id)]
        return [self._artifacts[i] for i in ids if i in self._artifacts]

    def roots(self, artifact_id: str) -> list[str]:
        """Ancestors that have no parents -- the original external inputs."""
        return [i for i in self.ancestors(artifact_id) if i in self._artifacts and not self._artifacts[i].parents]

    def dangling_parents(self) -> list[str]:
        """Referenced parents we hold no record for.  Should be empty in a healthy graph."""
        known = set(self._artifacts)
        missing: set[str] = set()
        for artifact in self._artifacts.values():
            missing.update(p for p in artifact.parents if p not in known)
        return sorted(missing)

    def verify_all(self) -> dict[str, str]:
        """Verify every file-backed artifact.  Returns ``{artifact_id: status}``."""
        report: dict[str, str] = {}
        for artifact in self._artifacts.values():
            if artifact.path is None or artifact.output_hash is None:
                report[artifact.artifact_id] = "not-file-backed"
                continue
            try:
                artifact.verify()
            except ChecksumMismatch as exc:
                report[artifact.artifact_id] = f"MISMATCH: {exc.message}"
            else:
                report[artifact.artifact_id] = "ok"
        return report

    # -- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for artifact in sorted(self._artifacts.values(), key=lambda a: (a.created_at, a.artifact_id)):
                fh.write(artifact.model_dump_json() + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ProvenanceGraph:
        graph = cls()
        path = Path(path)
        if not path.exists():
            return graph
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                graph.add(Artifact.model_validate_json(line))
            except Exception as exc:
                raise PolymerEngineError(
                    "Corrupt provenance record", path=str(path), line=line_no, error=str(exc)
                ) from exc
        return graph


__all__ = [
    "Artifact",
    "ProvenanceGraph",
    "canonical_hash",
    "environment_fingerprint",
    "sha256_bytes",
    "sha256_file",
]
