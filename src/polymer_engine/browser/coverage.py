"""Scoring the candidate set against what CHARMM-GUI actually offers (§43, §58, §76).

The matrix answers one question per polymer -- *can CHARMM-GUI build this?* -- and it
answers it from a captured catalogue, never from a guess. With no catalogue on disk
every cell reads ``unknown``, which is the honest answer before discovery has run: not
"no", because we have not looked, and certainly not "yes".

The second matrix places CHARMM-GUI beside the local backends, so the routing decision
is visible in one place. Both keep *routed*, *parameterized* and *qualified* as separate
columns, because they are separate claims and collapsing them is the failure mode this
whole subsystem exists to avoid.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from polymer_engine.browser.catalog import Catalog
from polymer_engine.browser.matching import resolve_monomer
from polymer_engine.browser.states import BrowserState
from polymer_engine.core.logging import get_logger

logger = get_logger("browser.coverage")

COVERAGE_COLUMNS = (
    "polymer", "polymer_id", "family", "exact_monomer_match", "catalog_value",
    "variants", "tacticity_supported", "single_chain", "melt", "solution",
    "multi_monomer", "build_ready", "requires_credentials", "blocker",
)

MATRIX_COLUMNS = (
    "polymer", "polymer_id", "family", "charmm_gui", "openff", "gaff", "opls_aa",
    "selected_route", "requires_human_step", "confidence", "parameterized",
    "qm_validated", "system_validated", "qualified", "blocker",
)


def coverage_rows(records: list[Any], catalog: Catalog | None) -> list[dict[str, Any]]:
    """One row per candidate, describing what the live catalogue can and cannot do."""
    rows: list[dict[str, Any]] = []
    have_catalog = catalog is not None and bool(catalog.monomers)

    for record in sorted(records, key=lambda r: (r.family.value, r.name)):
        row: dict[str, Any] = {
            "polymer": record.name, "polymer_id": record.polymer_id,
            "family": record.family.value, "requires_credentials": True,
        }
        if not have_catalog:
            row.update({
                "exact_monomer_match": "unknown", "catalog_value": "",
                "variants": "", "tacticity_supported": "unknown",
                "single_chain": "unknown", "melt": "unknown", "solution": "unknown",
                "multi_monomer": "unknown", "build_ready": "unknown",
                "blocker": ("no catalogue has been captured; run "
                            "`polymer-engine browser discover` with credentials"),
            })
            rows.append(row)
            continue

        assert catalog is not None
        match = resolve_monomer(catalog, record.name)
        modes = {mode.lower() for mode in catalog.system_modes}
        row.update({
            "exact_monomer_match": "yes" if match.resolved else "no",
            "catalog_value": match.entry.value if match.entry else "",
            "variants": ";".join(match.entry.variants) if match.entry else "",
            "tacticity_supported": "yes" if catalog.tacticity_options else "unknown",
            # A mode the page did not list is *unknown*, not unsupported: the capture
            # may simply not have reached the control that offers it.
            "single_chain": _mode(modes, "single"),
            "melt": _mode(modes, "melt"),
            "solution": _mode(modes, "solution"),
            "multi_monomer": "unknown",
            "build_ready": "yes" if match.resolved else "no",
            "blocker": "" if match.resolved else _blocker(match),
        })
        rows.append(row)
    return rows


def _mode(modes: set[str], needle: str) -> str:
    if not modes:
        return "unknown"
    return "yes" if any(needle in mode for mode in modes) else "not offered in capture"


def _blocker(match: Any) -> str:
    if match.state is BrowserState.REQUIRES_HUMAN_REVIEW:
        return f"ambiguous: {match.reason[:120]}"
    names = [c["label"] for c in match.candidates][:3]
    return ("no exact catalogue entry"
            + (f"; nearest names for a person to check: {', '.join(names)}" if names
               else ""))


def matrix_rows(records: list[Any], catalog: Catalog | None) -> list[dict[str, Any]]:
    """One row per candidate across every backend, with the routing decision."""
    from polymer_engine.parameterization import ParameterizationEngine, PropertyClass

    engine = ParameterizationEngine()
    coverage = {row["polymer_id"]: row for row in coverage_rows(records, catalog)}
    rows: list[dict[str, Any]] = []

    for record in sorted(records, key=lambda r: (r.family.value, r.name)):
        assessments = {a.backend: a for a in engine.assess(record)}
        decision = engine.route(record, property_class=PropertyClass.BULK_DENSITY)
        rows.append({
            "polymer": record.name, "polymer_id": record.polymer_id,
            "family": record.family.value,
            "charmm_gui": assessments["charmm_gui"].state.value,
            "openff": assessments["openff"].state.value,
            "gaff": assessments["gaff"].state.value,
            "opls_aa": assessments["opls_aa"].state.value,
            "selected_route": decision.selected_backend or "NONE",
            "requires_human_step": decision.requires_human_step,
            "confidence": decision.confidence,
            # These three stay separate and start false. A route is not a parameter set,
            # a parameter set is not a validated system, and neither is a qualification.
            "parameterized": False, "qm_validated": False,
            "system_validated": False, "qualified": False,
            "blocker": ("" if decision.selected_backend
                        else decision.reason[:140]
                        or coverage.get(record.polymer_id, {}).get("blocker", "")),
        })
    return rows


def write_coverage(
    candidates: str | Path, output: str | Path,
    catalog_dir: str | Path = "data/charmm_gui",
) -> dict[str, Path]:
    """Write both matrices and a short readme explaining what they do not claim."""
    from polymer_engine.polymer.ingestion import ingest_csv

    records = ingest_csv(str(candidates)).records
    catalog = Catalog.read(Path(catalog_dir))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    coverage_path = out / "charmm_gui_coverage_matrix.csv"
    _write_csv(coverage_path, COVERAGE_COLUMNS, coverage_rows(records, catalog))
    matrix_path = out / "parameterization_matrix.csv"
    _write_csv(matrix_path, MATRIX_COLUMNS, matrix_rows(records, catalog))

    readme = out / "COVERAGE_README.md"
    readme.write_text(_readme(catalog, len(records)))
    logger.info("coverage written for %d candidates (catalogue %s)", len(records),
                catalog.version if catalog else "absent")
    return {"coverage": coverage_path, "matrix": matrix_path, "readme": readme}


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _readme(catalog: Catalog | None, n_records: int) -> str:
    captured = (f"catalogue `{catalog.version}` captured {catalog.captured_at}"
                if catalog and catalog.monomers else "**no catalogue captured yet**")
    return f"""# Coverage matrices

{n_records} candidates scored against {captured}.

## `charmm_gui_coverage_matrix.csv`

Whether CHARMM-GUI Polymer Builder can build each candidate. Every cell derives from a
captured catalogue; with no capture on disk each reads `unknown`, which is the honest
state before discovery has run — not `no`, because nobody has looked.

`build_ready = yes` means an exact monomer match exists. It does not mean a build has
been run, and it says nothing about whether the resulting parameters would be any good.

## `parameterization_matrix.csv`

Every backend's assessment side by side, plus the routing decision.

`parameterized`, `qm_validated`, `system_validated` and `qualified` are separate columns
because they are separate claims:

* **routed** — a backend exists that could try
* **parameterized** — a topology exists; nothing has been checked
* **system_validated** — the topology is complete, neutral and accepted by `grompp`
* **qualified** — validated *for a stated property class*, with evidence

A row can be `yes` in the first and `no` in every other, and that is the normal state.
"""


__all__ = ["COVERAGE_COLUMNS", "MATRIX_COLUMNS", "coverage_rows", "matrix_rows",
           "write_coverage"]
