"""Materials Project (next-gen API).

Contract: https://api.materialsproject.org  (``X-API-KEY`` header)

Scope note, stated because it matters scientifically: Materials Project covers
**inorganic crystalline materials**.  It is not a source of polymer data.  The
engine uses it for reference/comparison values (e.g. inorganic fillers or
substrates in a composite), and the provider labels every record accordingly so a
downstream model cannot mistake an inorganic entry for a polymer.
"""

from __future__ import annotations

from typing import Any

from polymer_engine.core.config import Secret
from polymer_engine.core.errors import CredentialsMissing, ResponseFormatError
from polymer_engine.providers.base import Capability, Provider, ProviderResult
from polymer_engine.providers.http import HttpClient

DEFAULT_FIELDS = (
    "material_id",
    "formula_pretty",
    "structure",
    "symmetry",
    "density",
    "volume",
    "energy_above_hull",
    "band_gap",
    "is_stable",
)


class MaterialsProjectProvider(Provider):
    name = "materials_project"
    capabilities = frozenset({Capability.MATERIALS_REFERENCE})
    BASE = "https://api.materialsproject.org"

    #: Recorded in every result so a consumer cannot mistake the domain.
    DOMAIN = "inorganic-crystalline"

    def __init__(
        self,
        client: HttpClient | None = None,
        *,
        api_key: str | Secret | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(client)
        self._api_key = api_key if isinstance(api_key, Secret) else Secret(api_key)
        self.base = (base_url or self.BASE).rstrip("/")

    def configured(self) -> bool:
        return bool(self._api_key)

    def health(self) -> ProviderResult:
        return self._guard("health", self._health)

    def _health(self) -> ProviderResult:
        if not self.configured():
            return ProviderResult(
                self.name, "health", False,
                error="Materials Project API key is not configured",
                error_type="CredentialsMissing",
            )
        self._get("/materials/summary/", {"_limit": 1, "_fields": "material_id"})
        return ProviderResult(self.name, "health", True, provenance=self._provenance(endpoint=self.base))

    def _headers(self) -> dict[str, str]:
        key = self._api_key.reveal()
        if not key:
            raise CredentialsMissing(
                "Materials Project API key is not configured",
                hint="set MP_API_KEY or credentials.materials_project_api_key",
            )
        return {"X-API-KEY": key}

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        return self.client.get_json(f"{self.base}{path}", params=params, headers=self._headers())

    def search(
        self,
        *,
        formula: str | None = None,
        material_ids: list[str] | None = None,
        elements: list[str] | None = None,
        limit: int = 25,
        fields: tuple[str, ...] = DEFAULT_FIELDS,
    ) -> ProviderResult:
        self.require(Capability.MATERIALS_REFERENCE)
        return self._guard(
            "search", lambda: self._search(formula, material_ids, elements, limit, fields)
        )

    def _search(
        self,
        formula: str | None,
        material_ids: list[str] | None,
        elements: list[str] | None,
        limit: int,
        fields: tuple[str, ...],
    ) -> ProviderResult:
        if not any((formula, material_ids, elements)):
            return ProviderResult(
                self.name, "search", False,
                error="Provide at least one of formula, material_ids or elements",
                error_type="ValueError",
            )
        if limit < 1 or limit > 1000:
            return ProviderResult(self.name, "search", False, error="limit must be in [1, 1000]", error_type="ValueError")
        params: dict[str, Any] = {"_limit": limit, "_fields": ",".join(fields)}
        if formula:
            params["formula"] = formula
        if material_ids:
            params["material_ids"] = ",".join(material_ids)
        if elements:
            params["elements"] = ",".join(elements)
        path = "/materials/summary/"
        payload = self._get(path, params)
        if not isinstance(payload, dict):
            raise ResponseFormatError("Materials Project response is not a JSON object", url=self.base + path)
        data = payload.get("data")
        if data is None:
            data = []
        if not isinstance(data, list):
            raise ResponseFormatError("Materials Project data is not a list", url=self.base + path)
        records = [
            {"source": "materials_project", "domain": self.DOMAIN, **r} for r in data if isinstance(r, dict)
        ]
        raw_meta = payload.get("meta")
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        # Query params are recorded, but never the API key.
        return ProviderResult(
            self.name, "search", True, records=records, data=payload,
            total_available=_as_int(meta.get("total_doc")),
            provenance=self._provenance(endpoint=self.base + path, query=params, domain=self.DOMAIN),
        )


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["DEFAULT_FIELDS", "MaterialsProjectProvider"]
