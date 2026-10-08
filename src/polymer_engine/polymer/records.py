"""The canonical polymer record.

One model, used everywhere.  It bundles the resolved identity, the classified
family, the computed descriptors, normalised experimental properties, and the
provenance of each.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from polymer_engine.core.models import Determination, Measurement, utc_now
from polymer_engine.polymer.identity import PolymerIdentity, make_identity
from polymer_engine.polymer.taxonomy import Classification, PolymerFamily, classify


class PolymerRecord(BaseModel):
    """A curated polymer.

    ``curation_status`` is the honest summary: ``KNOWN`` only when the identity
    resolved cleanly and the family is confident.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    polymer_id: str
    name: str
    repeat_unit_smiles: str
    canonical_repeat_unit: str
    inchikey: str | None = None
    family: PolymerFamily = PolymerFamily.UNCLASSIFIED
    family_basis: str = ""
    family_confident: bool = False
    architecture: str = "linear"
    tacticity: str = "unknown"
    degree_of_polymerization: int | None = None
    descriptors: dict[str, Measurement] = Field(default_factory=dict)
    properties: dict[str, Measurement] = Field(default_factory=dict)
    curation_status: Determination = Determination.REQUIRES_VALIDATION
    issues: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    source: str = "unknown"
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def usable_for_modelling(self) -> bool:
        """Whether this record may enter a structure-property model.

        Requires a resolved identity and at least one computed descriptor; a record
        whose descriptors are all UNKNOWN would contribute only imputation noise.
        """
        if self.curation_status is not Determination.KNOWN:
            return False
        return any(m.determination is Determination.KNOWN for m in self.descriptors.values())

    def descriptor_values(self, names: Iterable[str]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for name in names:
            measurement = self.descriptors.get(name)
            out[name] = (
                measurement.value
                if measurement is not None and measurement.determination is Determination.KNOWN
                else None
            )
        return out

    def property_value(self, name: str) -> float | None:
        measurement = self.properties.get(name)
        if measurement is None or measurement.determination is not Determination.KNOWN:
            return None
        return measurement.value

    def summary(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id,
            "name": self.name,
            "canonical_repeat_unit": self.canonical_repeat_unit,
            "family": self.family.value,
            "family_confident": self.family_confident,
            "curation_status": self.curation_status.value,
            "n_descriptors_known": sum(
                1 for m in self.descriptors.values() if m.determination is Determination.KNOWN
            ),
            "n_properties_known": sum(
                1 for m in self.properties.values() if m.determination is Determination.KNOWN
            ),
            "issues": self.issues,
        }


def build_record(
    *,
    name: str,
    repeat_unit_smiles: str,
    properties: dict[str, Measurement] | None = None,
    source: str = "unknown",
    provenance: dict[str, Any] | None = None,
    architecture: str = "linear",
    tacticity: str = "unknown",
    degree_of_polymerization: int | None = None,
    compute_descriptors_now: bool = True,
) -> PolymerRecord:
    """Resolve identity, classify, and (optionally) compute descriptors."""
    from polymer_engine.polymer.descriptors import compute_descriptors

    identity: PolymerIdentity = make_identity(
        name=name,
        repeat_unit_smiles=repeat_unit_smiles,
        architecture=architecture,
        tacticity=tacticity,
        degree_of_polymerization=degree_of_polymerization,
    )
    classification: Classification = classify(identity.repeat_unit_smiles)

    descriptors: dict[str, Measurement] = {}
    issues = list(identity.issues)
    if compute_descriptors_now:
        descriptor_set = compute_descriptors(identity.repeat_unit_smiles, polymer_id=identity.polymer_id)
        descriptors = dict(descriptor_set.measurements)
        issues.extend(descriptor_set.issues)

    status = Determination.KNOWN
    if identity.issues or identity.canonicalisation is not Determination.KNOWN:
        status = Determination.REQUIRES_VALIDATION
    elif not classification.confident:
        status = Determination.REQUIRES_VALIDATION
        issues.append(f"family classification requires review: {classification.notes or classification.basis}")

    return PolymerRecord(
        polymer_id=identity.polymer_id,
        name=name,
        repeat_unit_smiles=identity.repeat_unit_smiles,
        canonical_repeat_unit=identity.canonical_repeat_unit,
        inchikey=identity.inchikey,
        family=classification.family,
        family_basis=classification.basis,
        family_confident=classification.confident,
        architecture=architecture,
        tacticity=tacticity,
        degree_of_polymerization=degree_of_polymerization,
        descriptors=descriptors,
        properties=properties or {},
        curation_status=status,
        issues=issues,
        provenance=provenance or {},
        source=source,
    )


__all__ = ["PolymerRecord", "build_record"]
