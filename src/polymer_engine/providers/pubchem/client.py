"""PubChem PUG REST.

Contract: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest

Two behaviours worth knowing:

* A name with no match returns HTTP 404 with a ``Fault`` body.  That is an *empty
  result*, not a transport failure, and is reported as ``ok=True, records=[]``.
* PubChem renamed several SMILES property fields (``CanonicalSMILES`` ->
  ``SMILES``, ``IsomericSMILES`` -> ``ConnectivitySMILES``).  Requesting a retired
  name yields HTTP 400.  The client asks for the modern set first and falls back
  to the legacy set once, rather than pinning either and breaking.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from polymer_engine.core.errors import HttpStatusError, ResponseFormatError
from polymer_engine.providers.base import Capability, Provider, ProviderResult
from polymer_engine.providers.http import HttpClient

MODERN_PROPERTIES = (
    "Title",
    "MolecularFormula",
    "MolecularWeight",
    "InChI",
    "InChIKey",
    "SMILES",
    "ConnectivitySMILES",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "Charge",
)

LEGACY_PROPERTIES = (
    "Title",
    "MolecularFormula",
    "MolecularWeight",
    "InChI",
    "InChIKey",
    "CanonicalSMILES",
    "IsomericSMILES",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "Charge",
)

#: Map every historical spelling onto one normalised key.
_SMILES_ALIASES = {
    "SMILES": "smiles",
    "CanonicalSMILES": "smiles",
    "ConnectivitySMILES": "connectivity_smiles",
    "IsomericSMILES": "isomeric_smiles",
}

_FIELD_MAP = {
    "CID": "cid",
    "Title": "title",
    "MolecularFormula": "molecular_formula",
    "MolecularWeight": "molecular_weight",
    "InChI": "inchi",
    "InChIKey": "inchikey",
    "XLogP": "xlogp",
    "TPSA": "tpsa",
    "HBondDonorCount": "hbond_donor_count",
    "HBondAcceptorCount": "hbond_acceptor_count",
    "RotatableBondCount": "rotatable_bond_count",
    "HeavyAtomCount": "heavy_atom_count",
    "Charge": "formal_charge",
    **_SMILES_ALIASES,
}


class PubChemProvider(Provider):
    name = "pubchem"
    capabilities = frozenset({Capability.COMPOUND_LOOKUP, Capability.COMPOUND_PROPERTIES})
    BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def __init__(self, client: HttpClient | None = None, *, base_url: str | None = None) -> None:
        super().__init__(client)
        self.base = (base_url or self.BASE).rstrip("/")

    def health(self) -> ProviderResult:
        return self._guard("health", self._health)

    def _health(self) -> ProviderResult:
        url = f"{self.base}/compound/name/water/property/MolecularFormula/JSON"
        self.client.get_json(url)
        return ProviderResult(self.name, "health", True, provenance=self._provenance(endpoint=url))

    def resolve_compound(self, identifier: str, *, namespace: str = "name") -> ProviderResult:
        """Look up one compound by name, CID, SMILES or InChIKey."""
        self.require(Capability.COMPOUND_LOOKUP)
        return self._guard("resolve_compound", lambda: self._resolve(identifier, namespace))

    def _resolve(self, identifier: str, namespace: str) -> ProviderResult:
        if not identifier or not identifier.strip():
            return ProviderResult(
                self.name, "resolve_compound", False, error="Empty identifier", error_type="ValueError"
            )
        for properties in (MODERN_PROPERTIES, LEGACY_PROPERTIES):
            url = (
                f"{self.base}/compound/{namespace}/{quote(identifier.strip(), safe='')}"
                f"/property/{','.join(properties)}/JSON"
            )
            try:
                payload = self.client.get_json(url)
            except HttpStatusError as exc:
                if exc.status == 404:
                    return ProviderResult(
                        self.name,
                        "resolve_compound",
                        True,
                        records=[],
                        total_available=0,
                        provenance=self._provenance(endpoint=url, identifier=identifier, note="no match"),
                    )
                if exc.status == 400 and properties is MODERN_PROPERTIES:
                    continue  # retired property names; retry with the legacy set
                raise
            return ProviderResult(
                self.name,
                "resolve_compound",
                True,
                records=self._normalise(payload, url),
                data=payload if isinstance(payload, dict) else {"payload": payload},
                total_available=len(self._normalise(payload, url)),
                provenance=self._provenance(endpoint=url, identifier=identifier, namespace=namespace),
            )
        raise ResponseFormatError("PubChem rejected both the modern and legacy property sets", identifier=identifier)

    def _normalise(self, payload: Any, url: str) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            raise ResponseFormatError("PubChem response is not a JSON object", url=url)
        if "Fault" in payload:
            return []
        table = payload.get("PropertyTable")
        if not isinstance(table, dict):
            raise ResponseFormatError("PubChem response has no PropertyTable", url=url, keys=sorted(payload))
        rows = table.get("Properties")
        if not isinstance(rows, list):
            raise ResponseFormatError("PubChem PropertyTable.Properties is not a list", url=url)
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ResponseFormatError("PubChem property row is not an object", url=url)
            record: dict[str, Any] = {"source": "pubchem", "source_url": url}
            for key, value in row.items():
                record[str(_FIELD_MAP.get(key, key))] = value
            # A record without any SMILES still resolves; downstream decides if that is usable.
            fallback = record.get("connectivity_smiles") or record.get("isomeric_smiles")
            if record.get("smiles") is None and fallback is not None:
                record["smiles"] = fallback
            record.setdefault("smiles", None)
            out.append(record)
        return out


__all__ = ["LEGACY_PROPERTIES", "MODERN_PROPERTIES", "PubChemProvider"]
