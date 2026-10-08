"""What Polymer Builder can actually build, read off the live page and versioned.

The CHARMM-GUI monomer list is not a constant. Monomers get added, labels get reworded,
and options appear and disappear between releases. Treating a catalogue captured on one
day as permanent truth is how an engine ends up confidently requesting a monomer that no
longer exists -- or, worse, silently matching a *different* one whose label happens to
have moved.

So every capture is fingerprinted and diffable, and the one thing this module refuses to
report is **completeness**. A page shows the options it shows; whether those are all the
options CHARMM-GUI supports is not observable from a page, and
:attr:`Catalog.completeness` says ``unknown`` for that reason rather than implying a
closed set we cannot see the boundary of.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from polymer_engine.core.logging import get_logger
from polymer_engine.core.provenance import canonical_hash

logger = get_logger("browser.catalog")

CATALOG_DIR = Path("data/charmm_gui")
CATALOG_JSON = "monomer_catalog.json"
CATALOG_MD = "monomer_catalog.md"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CatalogState(str, Enum):
    """Why a catalogue has the number of monomers it has.

    The distinction §6 is about, and it is not pedantic. "Zero monomers" from a page we
    could not read means something completely different from "zero monomers" on a page
    whose monomer control we found and which genuinely offers none -- and reporting the
    first as if it were the second would be a claim about CHARMM-GUI's chemistry
    coverage derived from a bug in our own selector.
    """

    #: A monomer control was identified and its options read.
    CAPTURED = "CATALOG_CAPTURED"
    #: A monomer control was identified and genuinely offers nothing.
    EMPTY = "CATALOG_EMPTY"
    #: No control on the page could be identified as the monomer selector.
    NOT_IDENTIFIED = "CATALOG_NOT_IDENTIFIED"
    #: Discovery has not run.
    NOT_ATTEMPTED = "CATALOG_NOT_ATTEMPTED"

    @property
    def usable(self) -> bool:
        return self is CatalogState.CAPTURED

    @property
    def needs_human(self) -> bool:
        """NOT_IDENTIFIED is a defect in our discovery, not a fact about the site."""
        return self is CatalogState.NOT_IDENTIFIED


@dataclass
class MonomerEntry:
    """One selectable monomer, exactly as the page presents it.

    ``value`` is the form value the site uses; ``label`` is what a person reads. Both are
    kept, because the two drift independently and a match on one is not a match on the
    other.
    """

    value: str
    label: str
    #: Which control it was found in, so two same-named options in different modes stay
    #: distinguishable.
    control: str = ""
    #: How the page represents this choice: select option, radio, checkbox, datalist.
    #: Recorded because selecting one differs by kind, and because a catalogue built
    #: from radios is evidence of a different page structure than one built from a
    #: select.
    kind: str = "select"
    system_modes: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.control}:{self.value}" if self.control else self.value

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "label": self.label, "control": self.control,
                "kind": self.kind, "system_modes": list(self.system_modes),
                "variants": list(self.variants), "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MonomerEntry:
        return cls(value=data["value"], label=data.get("label", data["value"]),
                   control=data.get("control", ""), kind=data.get("kind", "select"),
                   system_modes=tuple(data.get("system_modes", ())),
                   variants=tuple(data.get("variants", ())),
                   notes=data.get("notes", ""))


@dataclass
class Catalog:
    """A dated capture of what the Polymer Builder page offered."""

    monomers: list[MonomerEntry] = field(default_factory=list)
    system_modes: list[str] = field(default_factory=list)
    tacticity_options: list[str] = field(default_factory=list)
    other_options: dict[str, list[str]] = field(default_factory=dict)
    source_url: str = ""
    captured_at: str = field(default_factory=_now)
    #: Never "complete". A page cannot testify to what it does not show.
    completeness: str = "unknown"
    #: Why the catalogue is the size it is.  See :class:`CatalogState`.
    state: CatalogState = CatalogState.NOT_ATTEMPTED
    #: What discovery looked at, so a NOT_IDENTIFIED result is actionable.
    inspected: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def fingerprint(self) -> str:
        """Identifies the *content* of the catalogue, not when it was captured."""
        return canonical_hash({
            "monomers": sorted((m.control, m.value, m.label) for m in self.monomers),
            "system_modes": sorted(self.system_modes),
            "tacticity_options": sorted(self.tacticity_options),
            "other_options": {k: sorted(v) for k, v in sorted(self.other_options.items())},
        })

    @property
    def version(self) -> str:
        return self.fingerprint()[:12]

    def by_key(self) -> dict[str, MonomerEntry]:
        return {m.key: m for m in self.monomers}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "charmm_gui.monomer_catalog/1",
            "version": self.version, "fingerprint": self.fingerprint(),
            "captured_at": self.captured_at, "source_url": self.source_url,
            "completeness": self.completeness, "state": self.state.value,
            "inspected": dict(self.inspected),
            "n_monomers": len(self.monomers),
            "system_modes": list(self.system_modes),
            "tacticity_options": list(self.tacticity_options),
            "other_options": {k: list(v) for k, v in sorted(self.other_options.items())},
            "monomers": [m.as_dict() for m in self.monomers],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Catalog:
        return cls(
            monomers=[MonomerEntry.from_dict(m) for m in data.get("monomers", [])],
            system_modes=list(data.get("system_modes", [])),
            tacticity_options=list(data.get("tacticity_options", [])),
            other_options={k: list(v) for k, v in (data.get("other_options") or {}).items()},
            source_url=data.get("source_url", ""),
            captured_at=data.get("captured_at", _now()),
            completeness=data.get("completeness", "unknown"),
            state=CatalogState(data.get("state", "CATALOG_NOT_ATTEMPTED")),
            inspected=dict(data.get("inspected") or {}),
            notes=data.get("notes", ""),
        )

    def write(self, directory: str | Path = CATALOG_DIR) -> dict[str, Path]:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        json_path = root / CATALOG_JSON
        json_path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        md_path = root / CATALOG_MD
        md_path.write_text(self.to_markdown())
        logger.info("catalogue %s written: %d monomers", self.version, len(self.monomers))
        return {"json": json_path, "markdown": md_path}

    @classmethod
    def read(cls, directory: str | Path = CATALOG_DIR) -> Catalog | None:
        path = Path(directory) / CATALOG_JSON
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text()))

    def to_markdown(self) -> str:
        lines = [
            "# CHARMM-GUI Polymer Builder — monomer catalogue", "",
            f"Catalogue version `{self.version}` · state **{self.state.value}** · "
            f"captured {self.captured_at}",
            f"· source <{self.source_url or 'not recorded'}>", "",
            f"**Completeness: {self.completeness}.** A captured page shows the options it "
            "shows. Whether those are every monomer CHARMM-GUI supports is not observable "
            "from the page, so this catalogue is never described as complete.", "",
        ]
        if self.system_modes:
            lines += [f"System modes offered: {', '.join(self.system_modes)}", ""]
        if self.tacticity_options:
            lines += [f"Tacticity options: {', '.join(self.tacticity_options)}", ""]
        if self.state is CatalogState.NOT_IDENTIFIED:
            lines += [
                "**No control on the page could be identified as the monomer selector.**",
                "", "This is a limitation of our discovery, not a statement about "
                "CHARMM-GUI. Nothing here says the Polymer Builder offers no monomers; "
                "it says we did not find where it lists them.", "",
                f"What was inspected: `{self.inspected}`", "",
            ]
        elif not self.monomers:
            lines += ["No monomers have been discovered yet. Run catalogue discovery "
                      "against a live authenticated session.", ""]
        else:
            lines += ["| Label | Form value | Control | Variants |", "|---|---|---|---|"]
            for m in sorted(self.monomers, key=lambda x: (x.control, x.label.lower())):
                lines.append(f"| {m.label} | `{m.value}` | `{m.control}` | "
                             f"{', '.join(m.variants) or '—'} |")
            lines.append("")
        for name, options in sorted(self.other_options.items()):
            lines += [f"### `{name}`", "", ", ".join(f"`{o}`" for o in options), ""]
        if self.notes:
            lines += ["## Notes", "", self.notes, ""]
        return "\n".join(lines)


@dataclass
class CatalogDiff:
    """What changed between two captures (§19)."""

    previous_version: str
    current_version: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    relabelled: list[dict[str, str]] = field(default_factory=list)
    option_changes: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.relabelled or self.option_changes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "previous_version": self.previous_version,
            "current_version": self.current_version, "changed": self.changed,
            "added": list(self.added), "removed": list(self.removed),
            "relabelled": list(self.relabelled),
            "option_changes": dict(self.option_changes),
        }

    def summary(self) -> str:
        if not self.changed:
            return f"catalogue unchanged at version {self.current_version}"
        parts = []
        if self.added:
            parts.append(f"{len(self.added)} added")
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        if self.relabelled:
            parts.append(f"{len(self.relabelled)} relabelled")
        if self.option_changes:
            parts.append(f"{len(self.option_changes)} option set(s) changed")
        return (f"catalogue changed {self.previous_version} -> {self.current_version}: "
                + ", ".join(parts))


def diff(previous: Catalog, current: Catalog) -> CatalogDiff:
    """Compare two captures.

    A *relabelling* is called out separately from an add/remove pair because it is the
    dangerous case: the same form value with a different label means any matching done
    by label alone would now resolve differently, silently.
    """
    before, after = previous.by_key(), current.by_key()
    result = CatalogDiff(previous_version=previous.version, current_version=current.version)
    result.added = sorted(set(after) - set(before))
    result.removed = sorted(set(before) - set(after))
    for key in sorted(set(before) & set(after)):
        if before[key].label != after[key].label:
            result.relabelled.append(
                {"key": key, "before": before[key].label, "after": after[key].label}
            )
    for name in sorted(set(previous.other_options) | set(current.other_options)):
        old = set(previous.other_options.get(name, []))
        new = set(current.other_options.get(name, []))
        if old != new:
            result.option_changes[name] = {"added": sorted(new - old),
                                           "removed": sorted(old - new)}
    for name, old_list, new_list in (
        ("system_modes", previous.system_modes, current.system_modes),
        ("tacticity_options", previous.tacticity_options, current.tacticity_options),
    ):
        if set(old_list) != set(new_list):
            result.option_changes[name] = {
                "added": sorted(set(new_list) - set(old_list)),
                "removed": sorted(set(old_list) - set(new_list)),
            }
    return result


__all__ = [
    "CATALOG_DIR", "CATALOG_JSON", "CATALOG_MD", "Catalog", "CatalogDiff",
    "CatalogState", "MonomerEntry", "diff",
]
