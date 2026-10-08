"""RCSB PDB search and data APIs.

Contracts:
  search  https://search.rcsb.org/rcsbsearch/v2/query   (POST)
  data    https://data.rcsb.org/rest/v1/core/entry/{id} (GET)
  files   https://files.rcsb.org/download/{id}.{fmt}    (GET)

A search with zero hits returns **HTTP 204 with an empty body**, not an empty JSON
document.  Treating that as a parse failure is a common bug; it is handled
explicitly below.

Scope note: the PDB holds experimentally determined biomolecular structures.  For
synthetic polymers it is a source of *reference* conformations and comparison
targets, not of polymer entries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polymer_engine.core.errors import HttpStatusError, ResponseFormatError
from polymer_engine.providers.base import Capability, Provider, ProviderResult
from polymer_engine.providers.http import HttpClient

VALID_FORMATS = ("pdb", "cif", "pdb1", "bcif")


class RCSBProvider(Provider):
    name = "rcsb_pdb"
    capabilities = frozenset({Capability.STRUCTURE_SEARCH, Capability.STRUCTURE_DOWNLOAD})
    SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
    DATA = "https://data.rcsb.org/rest/v1/core"
    FILES = "https://files.rcsb.org/download"

    def __init__(
        self,
        client: HttpClient | None = None,
        *,
        search_url: str | None = None,
        data_url: str | None = None,
        files_url: str | None = None,
    ) -> None:
        super().__init__(client)
        self.search_url = search_url or self.SEARCH
        self.data_url = (data_url or self.DATA).rstrip("/")
        self.files_url = (files_url or self.FILES).rstrip("/")

    def health(self) -> ProviderResult:
        return self._guard("health", self._health)

    def _health(self) -> ProviderResult:
        self.search_text("water", rows=1)
        return ProviderResult(self.name, "health", True, provenance=self._provenance(endpoint=self.search_url))

    def search_text(self, query: str, rows: int = 25) -> ProviderResult:
        self.require(Capability.STRUCTURE_SEARCH)
        return self._guard("search_text", lambda: self._search_text(query, rows))

    def _search_text(self, query: str, rows: int) -> ProviderResult:
        if rows < 1 or rows > 10_000:
            return ProviderResult(
                self.name, "search_text", False, error="rows must be in [1, 10000]", error_type="ValueError"
            )
        payload = {
            "query": {"type": "terminal", "service": "full_text", "parameters": {"value": query}},
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": rows}},
        }
        response = self.client.request(self.search_url, method="POST", json_body=payload, use_cache=False)
        if response.status == 204 or not response.body.strip():
            # Documented "no hits" response.  An empty result, not an error.
            return ProviderResult(
                self.name, "search_text", True, records=[], total_available=0,
                provenance=self._provenance(endpoint=self.search_url, query=query, note="no hits (HTTP 204)"),
            )
        body = response.json()
        if not isinstance(body, dict):
            raise ResponseFormatError("RCSB search response is not a JSON object", url=self.search_url)
        result_set = body.get("result_set") or []
        if not isinstance(result_set, list):
            raise ResponseFormatError("RCSB result_set is not a list", url=self.search_url)
        records = [
            {"source": "rcsb_pdb", "entry_id": r.get("identifier"), "score": r.get("score")}
            for r in result_set
            if isinstance(r, dict)
        ]
        return ProviderResult(
            self.name, "search_text", True, records=records, data=body,
            total_available=_as_int(body.get("total_count")),
            provenance=self._provenance(endpoint=self.search_url, query=query),
        )

    def entry(self, entry_id: str) -> ProviderResult:
        self.require(Capability.STRUCTURE_SEARCH)
        return self._guard("entry", lambda: self._entry(entry_id))

    def _entry(self, entry_id: str) -> ProviderResult:
        url = f"{self.data_url}/entry/{entry_id.strip().upper()}"
        try:
            payload = self.client.get_json(url)
        except HttpStatusError as exc:
            if exc.status == 404:
                return ProviderResult(
                    self.name, "entry", True, records=[], total_available=0,
                    provenance=self._provenance(endpoint=url, entry_id=entry_id, note="not found"),
                )
            raise
        if not isinstance(payload, dict):
            raise ResponseFormatError("RCSB entry response is not a JSON object", url=url)
        info = payload.get("rcsb_entry_info") or {}
        citation = (payload.get("struct") or {}).get("title")
        record = {
            "source": "rcsb_pdb",
            "entry_id": (payload.get("rcsb_id") or entry_id).upper(),
            "title": citation,
            "experimental_method": (payload.get("exptl") or [{}])[0].get("method")
            if isinstance(payload.get("exptl"), list) and payload.get("exptl")
            else None,
            "resolution_angstrom": _first_float(info.get("resolution_combined")),
            "polymer_entity_count": _as_int(info.get("polymer_entity_count")),
            "deposited_atom_count": _as_int(info.get("deposited_atom_count")),
        }
        return ProviderResult(
            self.name, "entry", True, records=[record], data=payload, total_available=1,
            provenance=self._provenance(endpoint=url, entry_id=entry_id),
        )

    def download_structure(self, entry_id: str, destination: str | Path, *, fmt: str = "cif") -> ProviderResult:
        self.require(Capability.STRUCTURE_DOWNLOAD)
        return self._guard("download_structure", lambda: self._download(entry_id, destination, fmt))

    def _download(self, entry_id: str, destination: str | Path, fmt: str) -> ProviderResult:
        if fmt not in VALID_FORMATS:
            return ProviderResult(
                self.name, "download_structure", False,
                error=f"Unsupported format {fmt!r}; expected one of {VALID_FORMATS}", error_type="ValueError",
            )
        url = f"{self.files_url}/{entry_id.strip().upper()}.{fmt}"
        path = self.client.download(url, destination)
        return ProviderResult(
            self.name, "download_structure", True, artifacts=[str(path)],
            provenance=self._provenance(endpoint=url, entry_id=entry_id, format=fmt),
        )


def _first_float(value: Any) -> float | None:
    if isinstance(value, list) and value:
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["VALID_FORMATS", "RCSBProvider"]
