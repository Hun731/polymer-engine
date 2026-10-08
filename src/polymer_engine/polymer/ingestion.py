"""Dataset ingestion.

Reads CSV or JSON Lines into :class:`PolymerRecord` objects, normalising property
names and units on the way in and recording, per row, exactly what happened.

Rows are never silently dropped.  A row that cannot be ingested appears in
:attr:`IngestionReport.rejected` with a reason, so the difference between "the
dataset has 400 polymers" and "the file had 500 rows" is always visible.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.errors import ChemistryError, PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination, Measurement
from polymer_engine.core.provenance import canonical_hash, sha256_file
from polymer_engine.polymer.identity import make_identity
from polymer_engine.polymer.normalization import normalize_property, normalize_property_name
from polymer_engine.polymer.records import PolymerRecord, build_record

logger = get_logger("polymer.ingestion")

#: Column names accepted for the repeat unit, in priority order.
SMILES_COLUMNS = ("repeat_unit_smiles", "repeat_unit", "smiles", "psmiles", "monomer_smiles")
NAME_COLUMNS = ("name", "polymer_name", "polymer", "label")

#: ``property_name [unit]`` or ``property_name (unit)``
_UNIT_IN_HEADER = re.compile(r"^(?P<name>.+?)\s*[\[(]\s*(?P<unit>[^\])]+)\s*[\])]\s*$")


@dataclass
class RejectedRow:
    index: int
    reason: str
    row: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionReport:
    """What ingestion actually did, in numbers that must reconcile."""

    records: list[PolymerRecord] = field(default_factory=list)
    rejected: list[RejectedRow] = field(default_factory=list)
    duplicates: dict[str, list[str]] = field(default_factory=dict)
    rows_read: int = 0
    source: str = ""
    source_sha256: str | None = None

    @property
    def n_accepted(self) -> int:
        return len(self.records)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    @property
    def n_duplicates(self) -> int:
        return sum(len(v) for v in self.duplicates.values())

    def reconciles(self) -> bool:
        """Every row read is accounted for as accepted, rejected, or duplicate."""
        return self.rows_read == self.n_accepted + self.n_rejected + self.n_duplicates

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_sha256": self.source_sha256,
            "rows_read": self.rows_read,
            "accepted": self.n_accepted,
            "rejected": self.n_rejected,
            "duplicates": self.n_duplicates,
            "reconciles": self.reconciles(),
            "curation": {
                status.value: sum(1 for r in self.records if r.curation_status is status)
                for status in Determination
                if any(r.curation_status is status for r in self.records)
            },
            "families": _family_counts(self.records),
            "rejection_reasons": _count(r.reason for r in self.rejected),
        }


def _family_counts(records: list[PolymerRecord]) -> dict[str, int]:
    return _count(r.family.value for r in records)


def _count(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def split_unit_from_header(header: str) -> tuple[str, str | None]:
    """``"Tg (degC)"`` -> ``("Tg", "degC")``."""
    match = _UNIT_IN_HEADER.match(header.strip())
    if match:
        return match.group("name").strip(), match.group("unit").strip()
    return header.strip(), None


def _pick(row: dict[str, Any], candidates: Iterable[str]) -> tuple[str | None, Any]:
    lowered = {str(k).strip().lower(): k for k in row}
    for candidate in candidates:
        key = lowered.get(candidate)
        if key is not None and str(row[key]).strip():
            return key, row[key]
    return None, None


def ingest_rows(
    rows: Iterable[dict[str, Any]],
    *,
    source: str = "in-memory",
    source_sha256: str | None = None,
    smiles_column: str | None = None,
    name_column: str | None = None,
    compute_descriptors: bool = True,
) -> IngestionReport:
    """Ingest an iterable of row dicts."""
    report = IngestionReport(source=source, source_sha256=source_sha256)
    seen: dict[str, str] = {}

    for index, row in enumerate(rows):
        report.rows_read += 1
        smiles_key: str | None
        smiles: Any
        if smiles_column:
            smiles = row.get(smiles_column)
            smiles_key = smiles_column
        else:
            smiles_key, smiles = _pick(row, SMILES_COLUMNS)
        if not smiles or not str(smiles).strip():
            report.rejected.append(
                RejectedRow(index, f"no repeat-unit SMILES column (looked for {list(SMILES_COLUMNS)})", dict(row))
            )
            continue

        raw_name: Any = row.get(name_column) if name_column else _pick(row, NAME_COLUMNS)[1]
        resolved_name: str = str(raw_name).strip() if raw_name else f"row-{index}"

        try:
            identity = make_identity(name=resolved_name, repeat_unit_smiles=str(smiles))
        except (ChemistryError, PolymerEngineError) as exc:
            report.rejected.append(RejectedRow(index, f"invalid structure: {exc}", dict(row)))
            continue

        previous = seen.get(identity.polymer_id)
        if previous is not None:
            report.duplicates.setdefault(identity.polymer_id, []).append(resolved_name)
            logger.debug("Row %d duplicates %s (%s)", index, previous, identity.polymer_id)
            continue
        seen[identity.polymer_id] = resolved_name

        skip_columns: set[str | None] = {smiles_key, *NAME_COLUMNS}
        properties = _extract_properties(row, skip=skip_columns)
        try:
            record = build_record(
                name=resolved_name,
                repeat_unit_smiles=str(smiles),
                properties=properties,
                source=source,
                provenance={
                    "source": source,
                    "source_sha256": source_sha256,
                    "row_index": index,
                    "row_hash": canonical_hash(row),
                },
                compute_descriptors_now=compute_descriptors,
            )
        except (ChemistryError, PolymerEngineError) as exc:
            report.rejected.append(RejectedRow(index, f"record construction failed: {exc}", dict(row)))
            continue
        report.records.append(record)

    return report


def _extract_properties(row: dict[str, Any], *, skip: set[str | None]) -> dict[str, Measurement]:
    """Normalise every column that looks like a known property."""
    properties: dict[str, Measurement] = {}
    skip_lower = {s.lower() for s in skip if s}
    for header, value in row.items():
        if header is None or str(header).strip().lower() in skip_lower:
            continue
        if value is None or str(value).strip() == "":
            continue
        base, unit = split_unit_from_header(str(header))
        canonical = normalize_property_name(base)
        if canonical is None:
            continue
        measurement = normalize_property(base, value, unit)
        # Keep the first usable reading; a later duplicate column does not overwrite.
        existing = properties.get(measurement.name)
        if existing is None or (
            existing.determination is not Determination.KNOWN
            and measurement.determination is Determination.KNOWN
        ):
            properties[measurement.name] = measurement
    return properties


def ingest_csv(path: str | Path, **kwargs: Any) -> IngestionReport:
    """Ingest a CSV file."""
    path = Path(path)
    if not path.exists():
        raise PolymerEngineError("Dataset not found", path=str(path))
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return ingest_rows(rows, source=str(path), source_sha256=sha256_file(path), **kwargs)


def ingest_jsonl(path: str | Path, **kwargs: Any) -> IngestionReport:
    """Ingest a JSON Lines file."""
    path = Path(path)
    if not path.exists():
        raise PolymerEngineError("Dataset not found", path=str(path))
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PolymerEngineError("Malformed JSONL", path=str(path), line=line_no, error=str(exc)) from exc
        if not isinstance(parsed, dict):
            raise PolymerEngineError("JSONL rows must be objects", path=str(path), line=line_no)
        rows.append(parsed)
    return ingest_rows(rows, source=str(path), source_sha256=sha256_file(path), **kwargs)


def save_records(records: list[PolymerRecord], path: str | Path) -> Path:
    """Write records as JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
    return path


def load_records(path: str | Path) -> list[PolymerRecord]:
    path = Path(path)
    if not path.exists():
        raise PolymerEngineError("Record file not found", path=str(path))
    return [
        PolymerRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = [
    "IngestionReport",
    "RejectedRow",
    "ingest_csv",
    "ingest_jsonl",
    "ingest_rows",
    "load_records",
    "save_records",
    "split_unit_from_header",
]
