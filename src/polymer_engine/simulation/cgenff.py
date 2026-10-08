"""Reading CGenFF penalty scores out of a CHARMM stream file.

CGenFF assigns parameters **by analogy**: when a bond, angle or dihedral in your molecule
is not in the published force field, the program finds the closest thing it does have and
reports how far it had to reach.  That distance is the *penalty*, and it is the single
most important number in a CGenFF stream file, because a topology with high penalties
runs perfectly well and quietly produces wrong energetics.

The penalties live in comments, which is why they are so easy to lose::

    RESI MOL  0.000 ! param penalty=  12.500 ; charge penalty=  25.673
    ATOM C1   CG321  -0.180 !    0.000
    CG321 CG321   222.50   1.5300 ! MOL , from CG321 CG321, penalty= 0.6
    CG321 OG302 CG2O2  75.70  108.00 ! MOL , from CG321 OG302 CG2O2, penalty= 32.5

Every parser that treats a `!` as "rest of line is a comment" throws this away. This
module does the opposite: it reads the comments *for* the penalties and refuses to
report a parameter set without them.

Interpretation
--------------
The tiers in :data:`PENALTY_TIERS` are the CGenFF program's own published guidance
(Vanommeslaeghe and MacKerell, *J. Chem. Inf. Model.* 2012), not thresholds invented
here:

* below 10 — parameters are considered good quality;
* 10 to 50 — moderate quality, some validation recommended;
* above 50 — poor quality, extensive validation or optimisation required.

They are a **convention**, and this module treats them as one: it reports which tier a
parameter set falls in and never decides on your behalf that a high-penalty parameter is
acceptable for your application.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.errors import ChemistryError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination, GateReport, GateResult, GateStatus

logger = get_logger("simulation.cgenff")

#: Published CGenFF guidance, as ``(upper_bound_exclusive, label, description)``.
#: A convention from the CGenFF documentation, not a derivation.
PENALTY_TIERS: tuple[tuple[float, str, str], ...] = (
    (10.0, "good", "parameters are of good quality"),
    (50.0, "moderate", "moderate quality; some validation is recommended"),
    (float("inf"), "poor", "poor quality; extensive validation or optimisation required"),
)

#: Above this the parameter set should not enter production without QM validation.
HIGH_PENALTY = 50.0
#: Above this at least some validation is advised.
MODERATE_PENALTY = 10.0

#: ``... ! MOL , from CG321 CG321, penalty= 32.500``
_PARAM_PENALTY = re.compile(r"penalty\s*=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
#: ``RESI MOL  0.000 ! param penalty=  12.500 ; charge penalty=  25.673``
_RESI_PENALTIES = re.compile(
    r"param\s+penalty\s*=\s*(-?\d+(?:\.\d+)?)\s*;\s*charge\s+penalty\s*=\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_RESI = re.compile(r"^\s*RESI\s+(\S+)", re.IGNORECASE)
#: ``ATOM C1  CG321  -0.180 !   12.345``
_ATOM = re.compile(
    r"^\s*ATOM\s+(\S+)\s+(\S+)\s+(-?\d+(?:\.\d+)?)\s*(?:!\s*(-?\d+(?:\.\d+)?))?", re.IGNORECASE
)
#: CGenFF version banner.  The real header reads
#: ``* CHARMM General Force Field (CGenFF) program version 2.5``, so the closing
#: parenthesis sits between the name and "program" and must be allowed for.
_VERSION = re.compile(r"CGenFF[)\s]+program\s+version\s+([0-9][0-9.]*)", re.IGNORECASE)
#: Section headers inside the parameter block.
_SECTIONS = ("BONDS", "ANGLES", "DIHEDRALS", "IMPROPERS", "IMPROPER", "NONBONDED", "CMAP")


@dataclass(frozen=True)
class PenalisedParameter:
    """One analogy-assigned parameter and how far CGenFF had to reach for it."""

    section: str
    atoms: str
    penalty: float
    source: str
    line_number: int

    @property
    def tier(self) -> str:
        return tier_for(self.penalty)

    def as_dict(self) -> dict[str, Any]:
        return {
            "section": self.section, "atoms": self.atoms, "penalty": self.penalty,
            "tier": self.tier, "source": self.source, "line_number": self.line_number,
        }


@dataclass
class PenaltyReport:
    """Every penalty in one stream file, with nothing filtered out."""

    path: str = ""
    cgenff_version: str | None = None
    residues: list[str] = field(default_factory=list)
    parameters: list[PenalisedParameter] = field(default_factory=list)
    atom_charge_penalties: dict[str, float] = field(default_factory=dict)
    residue_param_penalty: float | None = None
    residue_charge_penalty: float | None = None
    parsed_lines: int = 0

    @property
    def max_penalty(self) -> float:
        """The worst parameter penalty, which is what governs usability."""
        values = [p.penalty for p in self.parameters]
        if self.residue_param_penalty is not None:
            values.append(self.residue_param_penalty)
        return max(values) if values else 0.0

    @property
    def max_charge_penalty(self) -> float:
        values = list(self.atom_charge_penalties.values())
        if self.residue_charge_penalty is not None:
            values.append(self.residue_charge_penalty)
        return max(values) if values else 0.0

    @property
    def mean_penalty(self) -> float:
        return (sum(p.penalty for p in self.parameters) / len(self.parameters)
                if self.parameters else 0.0)

    @property
    def tier(self) -> str:
        return tier_for(max(self.max_penalty, self.max_charge_penalty))

    def worst(self, limit: int = 10) -> list[PenalisedParameter]:
        return sorted(self.parameters, key=lambda p: -p.penalty)[:limit]

    def above(self, threshold: float) -> list[PenalisedParameter]:
        return [p for p in self.parameters if p.penalty > threshold]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "cgenff_version": self.cgenff_version,
            "residues": list(self.residues),
            "n_parameters": len(self.parameters),
            "max_penalty": self.max_penalty,
            "max_charge_penalty": self.max_charge_penalty,
            "mean_penalty": round(self.mean_penalty, 4),
            "tier": self.tier,
            "residue_param_penalty": self.residue_param_penalty,
            "residue_charge_penalty": self.residue_charge_penalty,
            "n_above_moderate": len(self.above(MODERATE_PENALTY)),
            "n_above_high": len(self.above(HIGH_PENALTY)),
            "worst": [p.as_dict() for p in self.worst()],
            "atom_charge_penalties": dict(self.atom_charge_penalties),
        }


def tier_for(penalty: float) -> str:
    for bound, label, _description in PENALTY_TIERS:
        if penalty < bound:
            return label
    return PENALTY_TIERS[-1][1]


def describe_tier(label: str) -> str:
    for _bound, name, description in PENALTY_TIERS:
        if name == label:
            return description
    return "unknown tier"


def parse_stream_file(path: str | Path) -> PenaltyReport:
    """Read a CHARMM stream file and keep every penalty it declares.

    Raises rather than returning an empty report when the file is not a CGenFF stream
    file: silently reporting "no penalties" for an unparsed file would read exactly like
    a clean parameter set.
    """
    path = Path(path)
    if not path.is_file():
        raise ChemistryError("CGenFF stream file not found", path=str(path))
    text = path.read_text(encoding="utf-8", errors="replace")
    report = parse_stream_text(text)
    report.path = str(path)
    if not report.residues and not report.parameters:
        raise ChemistryError(
            "File contains no RESI block and no penalised parameters; it does not look "
            "like a CGenFF stream file",
            path=str(path),
        )
    logger.info(
        "%s: %d penalised parameters, max penalty %.3f (%s)",
        path.name, len(report.parameters), report.max_penalty, report.tier,
    )
    return report


def parse_stream_text(text: str) -> PenaltyReport:
    """Parse stream-file *contents*.  Separated from IO so tests can use literals."""
    report = PenaltyReport()
    section = ""
    for number, raw in enumerate(text.splitlines(), start=1):
        report.parsed_lines = number
        line = raw.rstrip()
        if not line.strip():
            continue

        if report.cgenff_version is None:
            version = _VERSION.search(line)
            if version:
                report.cgenff_version = version.group(1)

        upper = line.strip().upper()
        head = upper.split()[0] if upper.split() else ""
        if head in _SECTIONS:
            section = head
            continue
        if upper.startswith(("READ ", "END", "RETURN")):
            if upper.startswith("READ "):
                section = ""
            continue

        residue = _RESI.match(line)
        if residue:
            report.residues.append(residue.group(1))
            penalties = _RESI_PENALTIES.search(line)
            if penalties:
                report.residue_param_penalty = float(penalties.group(1))
                report.residue_charge_penalty = float(penalties.group(2))
            continue

        atom = _ATOM.match(line)
        if atom:
            if atom.group(4) is not None:
                report.atom_charge_penalties[atom.group(1)] = float(atom.group(4))
            continue

        # A penalised parameter line: the numbers are before the `!`, the penalty after.
        if "!" not in line:
            continue
        body, _, comment = line.partition("!")
        match = _PARAM_PENALTY.search(comment)
        if not match or not body.strip():
            continue
        atoms = " ".join(body.split()[: _atom_count(section)]) or body.split()[0]
        report.parameters.append(
            PenalisedParameter(
                section=section or "UNKNOWN",
                atoms=atoms,
                penalty=float(match.group(1)),
                source=comment.strip(),
                line_number=number,
            )
        )
    return report


def _atom_count(section: str) -> int:
    return {"BONDS": 2, "ANGLES": 3, "DIHEDRALS": 4, "IMPROPERS": 4, "IMPROPER": 4}.get(
        section, 2
    )


def find_stream_files(directory: str | Path) -> list[Path]:
    """Locate CHARMM stream files in an extracted CHARMM-GUI job."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ChemistryError("Directory not found", path=str(directory))
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in {".str", ".prm", ".rtf"}
    )


def penalty_gates(report: PenaltyReport, *, max_penalty: float | None = None) -> GateReport:
    """Judge a parameter set by its penalties.

    ``max_penalty`` is the tolerance *you* accept.  Without it the gate reports the
    measured penalties and returns ``INCONCLUSIVE`` for anything above the published
    "good" tier, because whether a moderate-penalty parameter is acceptable depends on
    what is being measured -- a torsion penalty matters enormously for a conformational
    free energy and much less for a bulk density.
    """
    gates = GateReport(name="cgenff_penalty")

    if not report.parameters and report.residue_param_penalty is None:
        gates.gates.append(
            GateResult(
                gate="cgenff:penalties_present",
                status=GateStatus.INCONCLUSIVE,
                message=("no penalty annotations were found; either this is not a "
                         "CGenFF-generated parameter set or the penalties were stripped"),
            )
        )
        return gates

    gates.gates.append(
        GateResult(
            gate="cgenff:penalties_present",
            status=GateStatus.PASS,
            message=f"{len(report.parameters)} penalised parameter(s) read",
            value=float(len(report.parameters)),
            evidence={"cgenff_version": report.cgenff_version,
                      "residues": report.residues},
        )
    )

    worst = report.max_penalty
    tier = tier_for(worst)
    if max_penalty is not None:
        status = GateStatus.PASS if worst <= max_penalty else GateStatus.FAIL
        message = (f"maximum parameter penalty {worst:.3f} against the configured "
                   f"tolerance {max_penalty:.3f}")
    elif tier == "good":
        status, message = GateStatus.PASS, (
            f"maximum parameter penalty {worst:.3f}; {describe_tier(tier)}"
        )
    else:
        # Not a FAIL: it is a decision the operator has to make, not one to guess.
        status, message = GateStatus.INCONCLUSIVE, (
            f"maximum parameter penalty {worst:.3f} ({tier}); {describe_tier(tier)}. "
            f"Set an explicit tolerance or validate against QM before using these "
            f"parameters"
        )
    gates.gates.append(
        GateResult(
            gate="cgenff:max_parameter_penalty",
            status=status, message=message, value=worst,
            threshold=max_penalty if max_penalty is not None else MODERATE_PENALTY,
            evidence={"tier": tier,
                      "n_above_moderate": len(report.above(MODERATE_PENALTY)),
                      "n_above_high": len(report.above(HIGH_PENALTY)),
                      "worst": [p.as_dict() for p in report.worst(5)]},
        )
    )

    charge = report.max_charge_penalty
    charge_tier = tier_for(charge)
    gates.gates.append(
        GateResult(
            gate="cgenff:max_charge_penalty",
            status=GateStatus.PASS if charge_tier == "good" else GateStatus.WARN,
            message=f"maximum charge penalty {charge:.3f} ({charge_tier})",
            value=charge, threshold=MODERATE_PENALTY,
        )
    )
    return gates


def determination_for(report: PenaltyReport, *, max_penalty: float | None = None) -> Determination:
    """What may be claimed about a parameter set on the strength of its penalties."""
    if not report.parameters and report.residue_param_penalty is None:
        return Determination.UNKNOWN
    worst = report.max_penalty
    if max_penalty is not None:
        return Determination.KNOWN if worst <= max_penalty else Determination.REQUIRES_VALIDATION
    return Determination.KNOWN if tier_for(worst) == "good" else Determination.REQUIRES_VALIDATION


__all__ = [
    "HIGH_PENALTY",
    "MODERATE_PENALTY",
    "PENALTY_TIERS",
    "PenalisedParameter",
    "PenaltyReport",
    "describe_tier",
    "determination_for",
    "find_stream_files",
    "parse_stream_file",
    "parse_stream_text",
    "penalty_gates",
    "tier_for",
]
