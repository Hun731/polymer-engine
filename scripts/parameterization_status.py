#!/usr/bin/env python
"""Write the authoritative parameterization status (§75, §76).

Reports what is true right now: which tools exist, which routes each candidate has, and
which of the three separate claims -- parameterized, validated, qualified -- actually
hold. Nothing here is aspirational; a capability that cannot execute is reported as
absent.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymer_engine.browser.catalog import Catalog
from polymer_engine.browser.coverage import coverage_rows, matrix_rows
from polymer_engine.parameterization.capability import (
    discover_tools,
    summarise,
)
from polymer_engine.polymer.ingestion import ingest_csv

OUT = Path("campaign/parameterization")
CANDIDATES = Path("examples/benchmark_campaign/candidates.csv")


def main() -> int:
    records = ingest_csv(str(CANDIDATES)).records
    catalog = Catalog.read(Path("data/charmm_gui"))
    tools = summarise(discover_tools())
    matrix = matrix_rows(records, catalog)
    coverage = coverage_rows(records, catalog)

    routes: dict[str, int] = {}
    for row in matrix:
        routes[row["selected_route"]] = routes.get(row["selected_route"], 0) + 1

    status = {
        "schema": "parameterization_status/1",
        "generated_at": datetime.now(UTC).isoformat(),
        "n_candidates": len(records),
        "tools": {"available": tools["n_available"], "total": tools["n_tools"],
                  "by_backend": tools["backends"]},
        "routes": dict(sorted(routes.items())),
        "requires_human_step": sum(1 for r in matrix if r["requires_human_step"]),
        "no_route": sum(1 for r in matrix if r["selected_route"] == "NONE"),
        # The three claims, counted separately because they are separate claims.
        "parameterized": sum(1 for r in matrix if r["parameterized"]),
        "system_validated": sum(1 for r in matrix if r["system_validated"]),
        "qualified": sum(1 for r in matrix if r["qualified"]),
        "charmm_gui": {
            "catalog_captured": bool(catalog and catalog.monomers),
            "catalog_version": catalog.version if catalog else None,
            "completeness": catalog.completeness if catalog else "unknown",
            "build_ready": sum(1 for r in coverage if r["build_ready"] == "yes"),
            "unknown": sum(1 for r in coverage if r["build_ready"] == "unknown"),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parameterization_status.json").write_text(json.dumps(status, indent=2) + "\n")
    (OUT / "parameterization_status.md").write_text(_markdown(status, matrix))
    print(json.dumps(status, indent=2))
    return 0


def _markdown(status: dict, matrix: list[dict]) -> str:
    cg = status["charmm_gui"]
    lines = [
        "# Parameterization status", "",
        f"Generated {status['generated_at']} · {status['n_candidates']} candidates · "
        f"{status['tools']['available']} of {status['tools']['total']} tools available",
        "",
        "## Routing", "",
        "| Route | Candidates |", "|---|---|",
    ]
    lines += [f"| {route} | {n} |" for route, n in status["routes"].items()]
    lines += [
        "", f"{status['requires_human_step']} need a human step · "
        f"{status['no_route']} have no route at all.", "",
        "## The three claims, counted separately", "",
        "| Claim | Count | Meaning |", "|---|---|---|",
        f"| routed | {status['n_candidates'] - status['no_route']} | "
        "a backend exists that could try |",
        f"| parameterized | {status['parameterized']} | "
        "a topology exists; nothing has been checked |",
        f"| system validated | {status['system_validated']} | "
        "complete, neutral, accepted by real `grompp` |",
        f"| qualified | {status['qualified']} | "
        "validated for a stated property class, with evidence |",
        "",
        "A candidate can be routed and nothing else. That is the normal state, and "
        "collapsing these four numbers into one would be the central error this "
        "subsystem exists to avoid.", "",
        "## CHARMM-GUI", "",
    ]
    if cg["catalog_captured"]:
        lines += [f"Catalogue `{cg['catalog_version']}` captured · completeness "
                  f"{cg['completeness']} · {cg['build_ready']} candidates build-ready.", ""]
    else:
        lines += [
            "**No catalogue captured.** The browser subsystem is implemented and tested "
            "against a real browser on local fixtures, but has never loaded "
            "charmm-gui.org, because no credentials have been supplied. All "
            f"{cg['unknown']} candidates therefore read `unknown` -- not `no`, because "
            "nobody has looked.", "",
            "```bash", "export CHARMM_GUI_EMAIL='...'  # and CHARMM_GUI_PASSWORD",
            "polymer-engine browser discover", "```", "",
        ]
    lines += ["## Per candidate", "",
              "| Polymer | Family | Route | CHARMM-GUI | OpenFF | OPLS-AA | Qualified |",
              "|---|---|---|---|---|---|---|"]
    for row in matrix:
        lines.append(f"| {row['polymer']} | {row['family']} | {row['selected_route']} | "
                     f"{row['charmm_gui']} | {row['openff']} | {row['opls_aa']} | "
                     f"{'yes' if row['qualified'] else 'no'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
