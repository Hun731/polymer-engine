"""Is every interaction in this topology actually parameterized?

``grompp`` exiting 0 does not answer this.  It answers "I found a number for everything
I was asked to look up", and there are several ways to satisfy that while the physics is
wrong:

* a wildcard dihedral (``X CT CT X``) matched where a specific one should have;
* a placeholder written into the topology inline, so nothing was ever looked up;
* an interaction simply absent from the topology, so ``grompp`` had nothing to miss.

The last one is the quiet one.  A topology with no dihedral section at all preprocesses
perfectly and simulates a molecule with free internal rotation.  That is exactly what
``gmx x2top`` produced for n-butane -- three dihedrals where the molecule has 27 -- and
why this module counts interactions against the connectivity rather than trusting the
preprocessor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.errors import ChemistryError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import GateReport, GateResult, GateStatus

logger = get_logger("parameterization.completeness")

#: Sections a GROMACS ``[ moleculetype ]`` may carry, and whether their absence matters.
_SECTION = re.compile(r"^\s*\[\s*(\w+)\s*\]")
_WILDCARD_TOKENS = frozenset({"X", "x", "*"})
#: Values a topology writer uses as "fill this in later".  Any of them in a parameter
#: column means the number was never looked up.
_PLACEHOLDER_VALUES = frozenset({"0.0000", "0.000", "0.00", "0.0", "999.9", "9999.0"})


@dataclass
class CompletenessReport:
    """What the topology declares, against what the connectivity requires."""

    path: str = ""
    n_atoms: int = 0
    n_bonds: int = 0
    n_angles: int = 0
    n_dihedrals: int = 0
    n_impropers: int = 0
    n_pairs: int = 0
    expected_angles: int | None = None
    expected_dihedrals: int | None = None
    missing_sections: list[str] = field(default_factory=list)
    wildcards: list[str] = field(default_factory=list)
    inline_parameters: int = 0
    unresolved_includes: list[str] = field(default_factory=list)
    atom_types: set[str] = field(default_factory=set)
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "n_atoms": self.n_atoms, "n_bonds": self.n_bonds,
            "n_angles": self.n_angles, "n_dihedrals": self.n_dihedrals,
            "n_impropers": self.n_impropers, "n_pairs": self.n_pairs,
            "expected_angles": self.expected_angles,
            "expected_dihedrals": self.expected_dihedrals,
            "missing_sections": list(self.missing_sections),
            "wildcards": list(self.wildcards),
            "inline_parameters": self.inline_parameters,
            "unresolved_includes": list(self.unresolved_includes),
            "n_atom_types": len(self.atom_types),
            "problems": list(self.problems),
        }


def _rows(block: str) -> list[list[str]]:
    """Data rows of a topology section, comments and directives stripped."""
    rows = []
    for line in block.splitlines():
        text = line.split(";")[0].strip()
        if not text or text.startswith(("[", "#")):
            continue
        rows.append(text.split())
    return rows


def _blocks(text: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    current = ""
    for line in text.splitlines():
        match = _SECTION.match(line)
        if match:
            current = match.group(1).lower()
            blocks.setdefault(current, "")
            continue
        if current:
            blocks[current] += line + "\n"
    return blocks


def expected_counts(bonds: list[tuple[int, int]]) -> tuple[int, int]:
    """Angles and proper dihedrals implied by a bond list.

    Counting from connectivity is the point: it is independent of whatever the topology
    chose to declare, so a section that is short shows up as a shortfall rather than as
    a smaller-but-consistent-looking file.
    """
    neighbours: dict[int, set[int]] = {}
    for a, b in bonds:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)

    angles = sum(len(n) * (len(n) - 1) // 2 for n in neighbours.values())
    dihedrals = 0
    for j, k in bonds:
        left = neighbours.get(j, set()) - {k}
        right = neighbours.get(k, set()) - {j}
        dihedrals += sum(1 for i in left for l_atom in right if i != l_atom)
    return angles, dihedrals


def analyse_topology(path: str | Path) -> CompletenessReport:
    """Read a GROMACS topology and measure what it declares."""
    path = Path(path)
    if not path.is_file():
        raise ChemistryError("Topology not found", path=str(path))
    text = path.read_text(encoding="utf-8", errors="replace")
    report = CompletenessReport(path=str(path))
    blocks = _blocks(text)

    atoms = _rows(blocks.get("atoms", ""))
    report.n_atoms = len(atoms)
    report.atom_types = {row[1] for row in atoms if len(row) > 1}

    bond_rows = _rows(blocks.get("bonds", ""))
    report.n_bonds = len(bond_rows)
    bonds = [(int(r[0]), int(r[1])) for r in bond_rows
             if len(r) >= 2 and r[0].isdigit() and r[1].isdigit()]

    report.n_angles = len(_rows(blocks.get("angles", "")))
    report.n_dihedrals = len(_rows(blocks.get("dihedrals", "")))
    report.n_impropers = len(_rows(blocks.get("impropers", "")))
    report.n_pairs = len(_rows(blocks.get("pairs", "")))

    if bonds:
        report.expected_angles, report.expected_dihedrals = expected_counts(bonds)

    for section, count in (("bonds", report.n_bonds), ("angles", report.n_angles),
                           ("dihedrals", report.n_dihedrals)):
        if section not in blocks:
            report.missing_sections.append(section)
        elif count == 0 and report.n_atoms > 2:
            report.missing_sections.append(f"{section} (present but empty)")

    # Inline parameters: a bonded row carrying numbers past the function type was not
    # resolved from the force field, so the published value is not what will be used.
    for section, index in (("bonds", 3), ("angles", 4), ("dihedrals", 5)):
        for row in _rows(blocks.get(section, "")):
            if len(row) > index:
                report.inline_parameters += 1
                if any(token in _PLACEHOLDER_VALUES for token in row[index:]):
                    report.problems.append(
                        f"{section}: placeholder-looking inline parameters {row[index:]}"
                    )

    for section in ("bondtypes", "angletypes", "dihedraltypes"):
        for row in _rows(blocks.get(section, "")):
            if any(token in _WILDCARD_TOKENS for token in row[:4]):
                report.wildcards.append(f"{section}: {' '.join(row[:4])}")

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#include"):
            target = stripped.split(maxsplit=1)[-1].strip().strip('"')
            if target.startswith("<") or "/" in target:
                continue          # resolved against GMXLIB at run time
            if not (path.parent / target).is_file():
                report.unresolved_includes.append(target)
    return report


def completeness_gates(
    report: CompletenessReport, *, require_dihedrals: bool = True
) -> GateReport:
    """Judge a topology's completeness against its own connectivity."""
    gates = GateReport(name="parameter_completeness")

    gates.gates.append(GateResult(
        gate="completeness:atoms",
        status=GateStatus.PASS if report.n_atoms else GateStatus.FAIL,
        message=f"{report.n_atoms} atom(s) declared",
        value=float(report.n_atoms),
    ))

    if report.missing_sections:
        gates.gates.append(GateResult(
            gate="completeness:sections",
            status=GateStatus.FAIL,
            message=("missing or empty bonded section(s): "
                     + ", ".join(report.missing_sections)
                     + "; a topology without them simulates a different molecule"),
            evidence={"missing": report.missing_sections},
        ))
    else:
        gates.gates.append(GateResult(
            gate="completeness:sections", status=GateStatus.PASS,
            message="bonds, angles and dihedrals are all present and non-empty",
        ))

    for label, declared, expected in (
        ("angles", report.n_angles, report.expected_angles),
        ("dihedrals", report.n_dihedrals, report.expected_dihedrals),
    ):
        if expected is None:
            continue
        if label == "dihedrals" and not require_dihedrals:
            continue
        shortfall = expected - declared
        status = GateStatus.PASS if shortfall <= 0 else GateStatus.FAIL
        gates.gates.append(GateResult(
            gate=f"completeness:{label}_vs_connectivity", status=status,
            message=(f"{declared} {label} declared against {expected} implied by the "
                     f"bond list" + (f"; {shortfall} missing" if shortfall > 0 else "")),
            value=float(declared), threshold=float(expected),
            evidence={"declared": declared, "expected": expected},
        ))

    if report.wildcards:
        gates.gates.append(GateResult(
            gate="completeness:wildcards", status=GateStatus.WARN,
            message=(f"{len(report.wildcards)} wildcard parameter type(s); a wildcard "
                     f"match is weaker evidence than a specific one"),
            evidence={"wildcards": report.wildcards[:10]},
        ))

    if report.unresolved_includes:
        gates.gates.append(GateResult(
            gate="completeness:includes", status=GateStatus.FAIL,
            message="unresolved #include: " + ", ".join(report.unresolved_includes),
            evidence={"unresolved": report.unresolved_includes},
        ))

    if report.problems:
        gates.gates.append(GateResult(
            gate="completeness:placeholders", status=GateStatus.FAIL,
            message="; ".join(report.problems[:3]),
            evidence={"problems": report.problems},
        ))
    return gates


__all__ = [
    "CompletenessReport", "analyse_topology", "completeness_gates", "expected_counts",
]
