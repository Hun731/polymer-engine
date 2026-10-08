"""Common provider interface.

Every external data source implements :class:`Provider` so the orchestrator can add
sources without changing.  Two rules apply to all of them:

* A provider declares its :class:`Capability` set honestly.  Something it cannot do
  is absent from the set, and calling it raises :class:`UnsupportedCapability`.
* A provider never converts a transport failure into an empty scientific result.
  ``ProviderResult.ok is False`` always carries an ``error_type`` naming the class of
  failure so callers can tell "no records exist" from "the network is down".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from polymer_engine.core.errors import (
    CredentialsMissing,
    PolymerEngineError,
    ProviderError,
    UnsupportedCapability,
)
from polymer_engine.core.logging import get_logger
from polymer_engine.providers.http import HttpClient, RequestProvenance

logger = get_logger("providers")

T = TypeVar("T")


class Capability(str, Enum):
    """What a provider can actually do, verified against its implementation."""

    COMPOUND_LOOKUP = "compound_lookup"
    COMPOUND_PROPERTIES = "compound_properties"
    LITERATURE_SEARCH = "literature_search"
    LITERATURE_METADATA = "literature_metadata"
    CITATION_GRAPH = "citation_graph"
    FULLTEXT_DISCOVERY = "fulltext_discovery"
    STRUCTURE_SEARCH = "structure_search"
    STRUCTURE_DOWNLOAD = "structure_download"
    MATERIALS_REFERENCE = "materials_reference"
    JOB_LOGIN = "job_login"
    JOB_STATUS = "job_status"
    JOB_DOWNLOAD = "job_download"
    JOB_SUBMISSION = "job_submission"


@dataclass(slots=True)
class ProviderResult:
    """Outcome of one provider operation.

    ``records`` is the normalised payload; ``data`` keeps the raw response so a
    normalisation bug can be diagnosed without re-fetching.
    """

    provider: str
    operation: str
    ok: bool
    records: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_type: str | None = None
    total_available: int | None = None

    @property
    def empty(self) -> bool:
        """True when the call succeeded and the provider genuinely has no records.

        Distinct from ``not ok``, which means the call itself failed.
        """
        return self.ok and not self.records

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "operation": self.operation,
            "ok": self.ok,
            "records": self.records,
            "data": self.data,
            "artifacts": self.artifacts,
            "provenance": self.provenance,
            "error": self.error,
            "error_type": self.error_type,
            "total_available": self.total_available,
        }


class Provider(ABC):
    """Base class for external data sources."""

    name: str = "provider"
    capabilities: frozenset[Capability] = frozenset()

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    # -- contract -------------------------------------------------------
    @abstractmethod
    def health(self) -> ProviderResult:
        """Cheap reachability/configuration check.  Must not raise."""

    def configured(self) -> bool:
        """Whether the credentials this provider needs are present.  Default: none needed."""
        return True

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def require(self, capability: Capability) -> None:
        if not self.supports(capability):
            raise UnsupportedCapability(
                f"{self.name} does not support {capability.value}",
                provider=self.name,
                capability=capability.value,
                supported=sorted(c.value for c in self.capabilities),
            )

    # -- helpers --------------------------------------------------------
    def _provenance(self, **extra: Any) -> dict[str, Any]:
        record: dict[str, Any] = {"provider": self.name, **extra}
        prov: RequestProvenance | None = self.client.last_provenance
        if prov is not None:
            record["request"] = prov.as_dict()
        return record

    def _guard(
        self,
        operation: str,
        fn: Callable[[], ProviderResult],
        *,
        provenance: dict[str, Any] | None = None,
    ) -> ProviderResult:
        """Run ``fn``, converting *known* engine errors into a failed result.

        Only :class:`PolymerEngineError` subclasses are caught.  An unexpected
        exception (a bug in normalisation, say) propagates, because turning a bug
        into "the provider returned nothing" is exactly the failure mode that makes
        an autonomous pipeline draw wrong conclusions.
        """
        try:
            return fn()
        except (ProviderError, CredentialsMissing) as exc:
            logger.warning("%s.%s failed: %s", self.name, operation, exc)
            return ProviderResult(
                provider=self.name,
                operation=operation,
                ok=False,
                error=str(exc),
                error_type=type(exc).__name__,
                provenance={**(provenance or {}), **self._provenance()},
            )
        except PolymerEngineError as exc:
            logger.warning("%s.%s failed: %s", self.name, operation, exc)
            return ProviderResult(
                provider=self.name,
                operation=operation,
                ok=False,
                error=str(exc),
                error_type=type(exc).__name__,
                provenance={**(provenance or {}), **self._provenance()},
            )


def dedupe_records(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Drop duplicate records by ``key``, preserving first-seen order.

    Records missing the key are kept -- silently discarding them would lose data.
    """
    seen: set[Any] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        value = record.get(key)
        if value is None:
            out.append(record)
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(record)
    return out


__all__ = ["Capability", "Provider", "ProviderResult", "dedupe_records"]
