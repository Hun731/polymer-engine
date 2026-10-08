"""Crossref REST API.

Contract: https://api.crossref.org  (no ``/v1`` path segment exists)

Crossref asks clients to identify themselves with a ``mailto``; doing so moves the
request into the "polite" pool.  We pass it when configured and note in provenance
when we could not.
"""

from __future__ import annotations

from typing import Any

from polymer_engine.core.errors import ResponseFormatError
from polymer_engine.providers.base import Capability, Provider, ProviderResult, dedupe_records
from polymer_engine.providers.http import HttpClient


class CrossrefProvider(Provider):
    name = "crossref"
    capabilities = frozenset({Capability.LITERATURE_SEARCH, Capability.LITERATURE_METADATA})
    BASE = "https://api.crossref.org"

    def __init__(self, client: HttpClient | None = None, *, mailto: str | None = None, base_url: str | None = None) -> None:
        super().__init__(client)
        self.mailto = mailto
        self.base = (base_url or self.BASE).rstrip("/")

    def health(self) -> ProviderResult:
        return self._guard("health", self._health)

    def _health(self) -> ProviderResult:
        url = f"{self.base}/works"
        self.client.get_json(url, params=self._params({"rows": 0}))
        return ProviderResult(self.name, "health", True, provenance=self._provenance(endpoint=url))

    def _params(self, extra: dict[str, Any]) -> dict[str, Any]:
        params = dict(extra)
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    def search(self, query: str, rows: int = 20) -> ProviderResult:
        self.require(Capability.LITERATURE_SEARCH)
        return self._guard("search", lambda: self._search(query, rows))

    def _search(self, query: str, rows: int) -> ProviderResult:
        if rows < 0 or rows > 1000:
            return ProviderResult(self.name, "search", False, error="rows must be in [0, 1000]", error_type="ValueError")
        url = f"{self.base}/works"
        params = self._params({"query.bibliographic": query, "rows": rows})
        payload = self.client.get_json(url, params=params)
        message = self._message(payload, url)
        items = message.get("items")
        if items is None:
            items = []
        if not isinstance(items, list):
            raise ResponseFormatError("Crossref message.items is not a list", url=url)
        records = dedupe_records([self._normalise(item) for item in items if isinstance(item, dict)], "doi")
        return ProviderResult(
            self.name,
            "search",
            True,
            records=records,
            data=message,
            total_available=_as_int(message.get("total-results")),
            provenance=self._provenance(endpoint=url, query=query, polite_pool=bool(self.mailto)),
        )

    def work(self, doi: str) -> ProviderResult:
        """Fetch a single work by DOI."""
        self.require(Capability.LITERATURE_METADATA)
        return self._guard("work", lambda: self._work(doi))

    def _work(self, doi: str) -> ProviderResult:
        from polymer_engine.core.errors import HttpStatusError

        url = f"{self.base}/works/{doi.strip()}"
        try:
            payload = self.client.get_json(url, params=self._params({}))
        except HttpStatusError as exc:
            if exc.status == 404:
                return ProviderResult(
                    self.name, "work", True, records=[], total_available=0,
                    provenance=self._provenance(endpoint=url, doi=doi, note="not found"),
                )
            raise
        message = self._message(payload, url)
        return ProviderResult(
            self.name, "work", True, records=[self._normalise(message)], data=message,
            total_available=1, provenance=self._provenance(endpoint=url, doi=doi),
        )

    @staticmethod
    def _message(payload: Any, url: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ResponseFormatError("Crossref response is not a JSON object", url=url)
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ResponseFormatError("Crossref response has no message object", url=url, keys=sorted(payload))
        return message

    @staticmethod
    def _normalise(item: dict[str, Any]) -> dict[str, Any]:
        title = item.get("title") or []
        container = item.get("container-title") or []
        return {
            "source": "crossref",
            "doi": item.get("DOI"),
            "title": title[0] if isinstance(title, list) and title else None,
            "journal": container[0] if isinstance(container, list) and container else None,
            "year": _issued_year(item),
            "type": item.get("type"),
            "authors": [
                " ".join(filter(None, (a.get("given"), a.get("family"))))
                for a in item.get("author", [])
                if isinstance(a, dict)
            ],
            "url": item.get("URL"),
            "citation_count": item.get("is-referenced-by-count"),
        }


def _issued_year(item: dict[str, Any]) -> int | None:
    issued = item.get("issued")
    if not isinstance(issued, dict):
        return None
    parts = issued.get("date-parts")
    if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
        return _as_int(parts[0][0])
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["CrossrefProvider"]
