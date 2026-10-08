"""Polymer identity, curation, descriptors, and taxonomy."""

from polymer_engine.polymer.descriptors import DescriptorSet, compute_descriptors, descriptor_matrix
from polymer_engine.polymer.identity import (
    PolymerIdentity,
    build_oligomer,
    canonical_repeat_unit,
    capped_monomer_smiles,
    deduplicate,
    make_identity,
    validate_repeat_unit,
)
from polymer_engine.polymer.ingestion import IngestionReport, ingest_csv, ingest_jsonl, ingest_rows
from polymer_engine.polymer.normalization import normalize_property, normalize_property_name
from polymer_engine.polymer.records import PolymerRecord, build_record
from polymer_engine.polymer.taxonomy import Classification, PolymerFamily, classify, group_by_family

__all__ = [
    "Classification",
    "DescriptorSet",
    "IngestionReport",
    "PolymerFamily",
    "PolymerIdentity",
    "PolymerRecord",
    "build_oligomer",
    "build_record",
    "canonical_repeat_unit",
    "capped_monomer_smiles",
    "classify",
    "compute_descriptors",
    "deduplicate",
    "descriptor_matrix",
    "group_by_family",
    "ingest_csv",
    "ingest_jsonl",
    "ingest_rows",
    "make_identity",
    "normalize_property",
    "normalize_property_name",
    "validate_repeat_unit",
]
