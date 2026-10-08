"""Charge consistency, and the repairs this module refuses to make.

A neutral polymer whose topology does not sum to zero is a defect somewhere upstream:
a mis-assigned atom type, a truncated charge group, a fragment charge carried over from
the reference molecule it was fitted to.  Every one of those is a real problem with a
real cause.

Renormalising the charges hides all of them.  Spreading a residual of +0.12 e over 182
atoms produces a topology that passes every subsequent check, simulates happily, and is
wrong in a way nothing downstream can detect.  So this module measures, reports, and
refuses -- it never rescales.  If a charge genuinely must be adjusted, that is a
recorded decision with a stated method, not a silent correction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.errors import ChemistryError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import GateReport, GateResult, GateStatus

logger = get_logger("parameterization.charges")

#: Charges in a topology are decimal literals, so a correct set sums to zero to within
#: floating-point noise.  Anything larger is a real mismatch, not round-off.
NEUTRALITY_TOLERANCE = 1.0e-6
#: Beyond this the cause is structural (a whole missing group), not accumulated error.
GROSS_MISMATCH = 0.5

_SECTION = re.compile(r"^\s*\[\s*(\w+)\s*\]")


@dataclass
class ChargeReport:
    """Charges as declared, per molecule and for the assembled system."""

    path: str = ""
    molecule_charges: dict[str, float] = field(default_factory=dict)
    molecule_counts: dict[str, int] = field(default_factory=dict)
    per_atom: dict[str, list[float]] = field(default_factory=dict)
    system_charge: float | None = None
    expected_neutral: bool = True

    @property
    def worst_molecule(self) -> tuple[str, float] | None:
        if not self.molecule_charges:
            return None
        name = max(self.molecule_charges, key=lambda k: abs(self.molecule_charges[k]))
        return name, self.molecule_charges[name]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "molecule_charges": {k: round(v, 9) for k, v in self.molecule_charges.items()},
            "molecule_counts": dict(self.molecule_counts),
            "system_charge": (round(self.system_charge, 9)
                              if self.system_charge is not None else None),
            "expected_neutral": self.expected_neutral,
            "worst_molecule": self.worst_molecule,
        }


def analyse_charges(path: str | Path) -> ChargeReport:
    """Sum the charge of every ``[ moleculetype ]`` and of the assembled system."""
    path = Path(path)
    if not path.is_file():
        raise ChemistryError("Topology not found", path=str(path))

    report = ChargeReport(path=str(path))
    section = ""
    molecule = ""
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";")[0].strip()
        match = _SECTION.match(raw)
        if match:
            section = match.group(1).lower()
            continue
        if not line or line.startswith("#"):
            continue

        if section == "moleculetype":
            molecule = line.split()[0]
            report.molecule_charges.setdefault(molecule, 0.0)
            report.per_atom.setdefault(molecule, [])
        elif section == "atoms" and molecule:
            fields = line.split()
            if len(fields) >= 7:
                try:
                    charge = float(fields[6])
                except ValueError:
                    continue
                report.molecule_charges[molecule] += charge
                report.per_atom[molecule].append(charge)
        elif section == "molecules":
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                report.molecule_counts[fields[0]] = int(fields[1])

    if report.molecule_counts:
        report.system_charge = sum(
            report.molecule_charges.get(name, 0.0) * count
            for name, count in report.molecule_counts.items()
        )
    return report


def charge_gates(
    report: ChargeReport, *, expected_system_charge: float = 0.0,
    tolerance: float = NEUTRALITY_TOLERANCE,
) -> GateReport:
    """Judge charge consistency.  Nothing here modifies a charge."""
    gates = GateReport(name="charge_consistency")

    if not report.molecule_charges:
        gates.gates.append(GateResult(
            gate="charge:present", status=GateStatus.INCONCLUSIVE,
            message="no [ atoms ] charges were found; charge cannot be checked",
        ))
        return gates

    for name, charge in sorted(report.molecule_charges.items()):
        if not report.per_atom.get(name):
            # An empty [ atoms ] section sums to 0.0 and would otherwise read as
            # "neutral". A molecule with no atoms has an unknown charge, not a zero one.
            gates.gates.append(GateResult(
                gate=f"charge:molecule:{name}", status=GateStatus.INCONCLUSIVE,
                message=(f"molecule {name} declares no atoms, so its charge is unknown "
                         f"rather than zero"),
            ))
            continue
        deviation = abs(charge)
        if deviation <= tolerance:
            status, note = GateStatus.PASS, "neutral"
        elif deviation >= GROSS_MISMATCH:
            status, note = GateStatus.FAIL, (
                "a residual this large is a structural error -- a missing charge group "
                "or a fragment charge carried over from a reference molecule"
            )
        else:
            # Small but real. A person decides; the engine does not rescale.
            status, note = GateStatus.INCONCLUSIVE, (
                "small but non-zero; this is not rounding, and it is not repaired here"
            )
        gates.gates.append(GateResult(
            gate=f"charge:molecule:{name}", status=status,
            message=f"molecule {name} sums to {charge:+.6f} e ({note})",
            value=charge, threshold=tolerance, units="e",
            evidence={"n_atoms": len(report.per_atom.get(name, []))},
        ))

    if report.system_charge is not None:
        deviation = abs(report.system_charge - expected_system_charge)
        gates.gates.append(GateResult(
            gate="charge:system",
            status=GateStatus.PASS if deviation <= tolerance else GateStatus.FAIL,
            message=(f"system charge {report.system_charge:+.6f} e against an expected "
                     f"{expected_system_charge:+.6f} e"),
            value=report.system_charge, threshold=expected_system_charge, units="e",
            evidence={"molecule_counts": report.molecule_counts},
        ))
    return gates


@dataclass(frozen=True)
class ChargeAdjustment:
    """A deliberate, recorded change to a charge set.

    This type exists so that adjusting charges is *possible* but never *quiet*: it
    cannot be applied without naming a method and a reason, and the record travels with
    the parameter set.
    """

    molecule: str
    original_charge: float
    new_charge: float
    method: str
    reason: str
    software: str
    author: str

    def __post_init__(self) -> None:
        for name in ("method", "reason", "software", "author"):
            if not str(getattr(self, name)).strip():
                raise ChemistryError(
                    "A charge adjustment must state its provenance", missing=name
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "molecule": self.molecule, "original_charge": self.original_charge,
            "new_charge": self.new_charge, "delta": self.new_charge - self.original_charge,
            "method": self.method, "reason": self.reason,
            "software": self.software, "author": self.author,
        }


__all__ = [
    "GROSS_MISMATCH", "NEUTRALITY_TOLERANCE", "ChargeAdjustment", "ChargeReport",
    "analyse_charges", "charge_gates",
]
