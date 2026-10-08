"""Europe PMC REST search.

Contract: https://www.ebi.ac.uk/europepmc/webservices/rest

Full-text availability is reported from the ``hasTextMinedTerms``/``isOpenAccess``
/``fullTextUrlList`` fields; the provider does not fetch full text itself.
"""

from __future__ import annotations

from typing import Any

from polymer_engine.core.errors import ResponseFormatError
from polymer_engine.providers.base import Capability, Provider, ProviderResult, dedupe_records
from polymer_engine.providers.http import HttpClient


class EuropePMCProvider(Provider):
    name = "europe_pmc"
    capabilities = frozenset({Capability.LITERATURE_SEARCH, Capability.FULLTEXT_DISCOVERY})
    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    def __init__(self, client: HttpClient | None = None, *, base_url: str | None = None) -> None:
        super().__init__(client)
        self.base = (base_url or self.BASE).rstrip("/")

    def health(self) -> ProviderResult:
        return self._guard("health", self._health)

    def _health(self) -> ProviderResult:
        url = f"{self.base}/search"
        self.client.get_json(url, params={"query": "polymer", "format": "json", "pageSize": 1})
        return ProviderResult(self.name, "health", True, provenance=self._provenance(endpoint=url))

    def search(self, query: str, rows: int = 25, *, result_type: str = "core") -> ProviderResult:
        self.require(Capability.LITERATURE_SEARCH)
        return self._guard("search", lambda: self._search(query, rows, result_type))

    def _search(self, query: str, rows: int, result_type: str) -> ProviderResult:
        if rows < 1 or rows > 1000:
            return ProviderResult(self.name, "search", False, error="rows must be in [1, 1000]", error_type="ValueError")
        url = f"{self.base}/search"
        params = {"query": query, "format": "json", "pageSize": rows, "resultType": result_type}
        payload = self.client.get_json(url, params=params)
        if not isinstance(payload, dict):
            raise ResponseFormatError("Europe PMC response is not a JSON object", url=url)
        result_list = payload.get("resultList")
        if result_list is None:
            # A well-formed zero-hit response may omit resultList entirely.
            results: list[Any] = []
        elif isinstance(result_list, dict):
            results = result_list.get("result") or []
        else:
            raise ResponseFormatError("Europe PMC resultList is not an object", url=url)
        if not isinstance(results, list):
            raise ResponseFormatError("Europe PMC resultList.result is not a list", url=url)
        records = dedupe_records([self._normalise(r) for r in results if isinstance(r, dict)], "id")
        return ProviderResult(
            self.name,
            "search",
            True,
            records=records,
            data=payload,
            total_available=_as_int(payload.get("hitCount")),
            provenance=self._provenance(endpoint=url, query=query),
        )

    @staticmethod
    def _normalise(item: dict[str, Any]) -> dict[str, Any]:
        full_text = item.get("fullTextUrlList") or {}
        urls = full_text.get("fullTextUrl", []) if isinstance(full_text, dict) else []
        return {
            "source": "europe_pmc",
            "id": item.get("id"),
            "source_db": item.get("source"),
            "pmid": item.get("pmid"),
            "pmcid": item.get("pmcid"),
            "doi": item.get("doi"),
            "title": item.get("title"),
            "journal": item.get("journalTitle"),
            "year": _as_int(item.get("pubYear")),
            "authors": item.get("authorString"),
            "citation_count": _as_int(item.get("citedByCount")),
            "is_open_access": str(item.get("isOpenAccess", "N")).upper() == "Y",
            "full_text_urls": [u.get("url") for u in urls if isinstance(u, dict) and u.get("url")],
        }


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["EuropePMCProvider"]
