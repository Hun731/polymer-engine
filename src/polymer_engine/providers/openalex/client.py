"""OpenAlex works API.

Contract: https://docs.openalex.org

OpenAlex asks for a ``mailto`` to enter the polite pool.  Referenced-work ids are
surfaced so the citation graph can be walked without a second normalisation step.
"""

from __future__ import annotations

from typing import Any

from polymer_engine.core.errors import HttpStatusError, ResponseFormatError
from polymer_engine.providers.base import Capability, Provider, ProviderResult, dedupe_records
from polymer_engine.providers.http import HttpClient


class OpenAlexProvider(Provider):
    name = "openalex"
    capabilities = frozenset(
        {Capability.LITERATURE_SEARCH, Capability.LITERATURE_METADATA, Capability.CITATION_GRAPH}
    )
    BASE = "https://api.openalex.org"

    def __init__(self, client: HttpClient | None = None, *, mailto: str | None = None, base_url: str | None = None) -> None:
        super().__init__(client)
        self.mailto = mailto
        self.base = (base_url or self.BASE).rstrip("/")

    def health(self) -> ProviderResult:
        return self._guard("health", self._health)

    def _health(self) -> ProviderResult:
        url = f"{self.base}/works"
        self.client.get_json(url, params=self._params({"per-page": 1}))
        return ProviderResult(self.name, "health", True, provenance=self._provenance(endpoint=url))

    def _params(self, extra: dict[str, Any]) -> dict[str, Any]:
        params = dict(extra)
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    def search(self, query: str, per_page: int = 25) -> ProviderResult:
        self.require(Capability.LITERATURE_SEARCH)
        return self._guard("search", lambda: self._search(query, per_page))

    def _search(self, query: str, per_page: int) -> ProviderResult:
        if per_page < 1 or per_page > 200:
            return ProviderResult(
                self.name, "search", False, error="per_page must be in [1, 200]", error_type="ValueError"
            )
        url = f"{self.base}/works"
        payload = self.client.get_json(url, params=self._params({"search": query, "per-page": per_page}))
        results, meta = self._results(payload, url)
        records = dedupe_records([self._normalise(r) for r in results], "id")
        return ProviderResult(
            self.name,
            "search",
            True,
            records=records,
            data=payload if isinstance(payload, dict) else {},
            total_available=_as_int(meta.get("count")),
            provenance=self._provenance(endpoint=url, query=query, polite_pool=bool(self.mailto)),
        )

    def work(self, identifier: str) -> ProviderResult:
        """Fetch one work by OpenAlex id or DOI (``doi:10.xxxx/yyy``)."""
        self.require(Capability.LITERATURE_METADATA)
        return self._guard("work", lambda: self._work(identifier))

    def _work(self, identifier: str) -> ProviderResult:
        url = f"{self.base}/works/{identifier.strip()}"
        try:
            payload = self.client.get_json(url, params=self._params({}))
        except HttpStatusError as exc:
            if exc.status == 404:
                return ProviderResult(
                    self.name, "work", True, records=[], total_available=0,
                    provenance=self._provenance(endpoint=url, identifier=identifier, note="not found"),
                )
            raise
        if not isinstance(payload, dict):
            raise ResponseFormatError("OpenAlex work response is not a JSON object", url=url)
        return ProviderResult(
            self.name, "work", True, records=[self._normalise(payload)], data=payload,
            total_available=1, provenance=self._provenance(endpoint=url, identifier=identifier),
        )

    @staticmethod
    def _results(payload: Any, url: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not isinstance(payload, dict):
            raise ResponseFormatError("OpenAlex response is not a JSON object", url=url)
        results = payload.get("results")
        if results is None:
            results = []
        if not isinstance(results, list):
            raise ResponseFormatError("OpenAlex results is not a list", url=url)
        meta = payload.get("meta")
        return [r for r in results if isinstance(r, dict)], meta if isinstance(meta, dict) else {}

    @staticmethod
    def _normalise(item: dict[str, Any]) -> dict[str, Any]:
        primary = item.get("primary_location") or {}
        source = primary.get("source") if isinstance(primary, dict) else None
        authorships = item.get("authorships") or []
        return {
            "source": "openalex",
            "id": item.get("id"),
            "doi": (item.get("doi") or "").replace("https://doi.org/", "") or None,
            "title": item.get("title") or item.get("display_name"),
            "journal": source.get("display_name") if isinstance(source, dict) else None,
            "year": _as_int(item.get("publication_year")),
            "type": item.get("type"),
            "citation_count": _as_int(item.get("cited_by_count")),
            "is_open_access": bool((item.get("open_access") or {}).get("is_oa")),
            "authors": [
                (a.get("author") or {}).get("display_name")
                for a in authorships
                if isinstance(a, dict) and isinstance(a.get("author"), dict)
            ],
            "referenced_works": item.get("referenced_works") or [],
        }


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["OpenAlexProvider"]
