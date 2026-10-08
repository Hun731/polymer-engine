#!/usr/bin/env python
"""Write CHARMM_GUI_DISCOVERY_REPORT.md from whatever discovery has actually produced.

Generated rather than hand-written, so it cannot drift from the artifacts. With no
capture on disk every count is zero and every field reads NOT_RUN -- which is the state
before a live login, and is reported as such rather than left ambiguous.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymer_engine.browser.catalog import Catalog
from polymer_engine.browser.selectors import FormSchema
from polymer_engine.browser.spec_mapping import MappingVerdict, default_probe_spec, map_spec_to_form
from polymer_engine.browser.verification import VerificationRegistry
from polymer_engine.simulation.charmm_gui_spec import SystemType

DATA = Path("data/charmm_gui")
REPORT = Path("CHARMM_GUI_DISCOVERY_REPORT.md")


def _load_schema() -> tuple[FormSchema, dict[str, list[str]]]:
    path = DATA / "builder_form_schema.json"
    if not path.exists():
        from polymer_engine.browser.selectors import builder_schema_placeholder

        return builder_schema_placeholder(), {}
    payload = json.loads(path.read_text())
    return FormSchema.from_dict(payload), payload.get("semantic_hints", {})


def summary() -> dict[str, object]:
    catalog = Catalog.read(DATA)
    schema, semantics = _load_schema()
    registry = VerificationRegistry()
    login = registry.records["LIVE_LOGIN_VERIFIED"]

    login_status = {
        "LIVE_VERIFIED": "PASS", "LIVE_FAILED": "FAIL",
        "REQUIRES_HUMAN_REVIEW": "HUMAN_INTERVENTION_REQUIRED",
    }.get(login.state.value, "NOT_RUN")

    counts: dict[str, int] = {}
    if schema.discovered and catalog and catalog.monomers:
        for system_type in (SystemType.SINGLE_CHAIN, SystemType.MELT):
            spec = default_probe_spec(catalog.monomers[0].label, system_type=system_type)
            for entry in map_spec_to_form(spec, schema, semantics).fields:
                if entry.verdict is not MappingVerdict.NOT_REQUESTED:
                    key = entry.verdict.value.lower()
                    counts[key] = counts.get(key, 0) + 1

    return {
        "login": login_status,
        "login_reason": login.reason,
        "polymer_builder": "FOUND" if schema.discovered else "NOT_FOUND",
        "catalog_entries": len(catalog.monomers) if catalog else 0,
        "catalog_version": catalog.version if catalog and catalog.monomers else None,
        "completeness": catalog.completeness if catalog else "unknown",
        "build_modes": (catalog.system_modes if catalog else []) or [],
        "form_fields": len(schema.fields),
        "spec_mapping": counts or {"mapped": 0, "ambiguous": 0, "unavailable": 0},
    }


def main() -> int:
    data = summary()
    modes = ", ".join(data["build_modes"]) or "none discovered"  # type: ignore[arg-type]
    mapping = data["spec_mapping"]
    REPORT.write_text(f"""# CHARMM-GUI live discovery report

Generated {datetime.now(UTC).isoformat()}

```
LOGIN            = {data['login']}
POLYMER_BUILDER  = {data['polymer_builder']}
CATALOG_ENTRIES  = {data['catalog_entries']}
BUILD_MODES      = {modes}
FORM_FIELDS      = {data['form_fields']}
SPEC_MAPPING     = mapped {mapping.get('mapped', 0)} / """  # type: ignore[union-attr]
f"""ambiguous {mapping.get('ambiguous', 0)} / """  # type: ignore[union-attr]
f"""unavailable {mapping.get('unavailable', 0)}
```

Catalogue version: `{data['catalog_version'] or 'none'}` · completeness
`{data['completeness']}`

{'' if data['login'] != 'NOT_RUN' else '''## Why every count is zero

No live login has been performed, so no page has been read. These are absences of
observation, not observations of absence: `CATALOG_ENTRIES = 0` means the catalogue was
never captured, and `BUILD_MODES` lists nothing because none has been seen.

The discovery phase is one command, run from a terminal so the password is typed into an
echo-off prompt rather than a shell:

```bash
.venv/bin/python scripts/charmm_gui_live.py discover
```

It prompts for anything not already in the environment, submits nothing, and writes
`monomer_catalog.{json,md}`, `builder_form_schema.json` and this report.'''}

## What the login reason records

{data['login_reason'] or 'not attempted'}
""")
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
