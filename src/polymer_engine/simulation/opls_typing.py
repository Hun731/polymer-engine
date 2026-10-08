"""OPLS-AA atom typing for polymer repeat units.

This module *assigns* published OPLS-AA parameters; it never invents them.  Every
atom type it can emit is read from the installed ``oplsaa.ff`` directory, and the
charge attached to each type is the force field's own tabulated value.  When a repeat
unit contains an atom this module cannot type from that table, the answer is
``UNSUPPORTED`` -- not a guess.

Why the scope is what it is
---------------------------
OPLS-AA charges are defined per *reference molecule*, and a per-type default charge is
only correct in the environment it was fitted for.  For saturated hydrocarbons that
distinction collapses, because every alkane group is neutral on its own::

    CH3  -0.180 + 3(+0.060) = 0
    CH2  -0.120 + 2(+0.060) = 0
    CH   -0.060 + 1(+0.060) = 0
    C     0.000             = 0

So any alkane assembled from these groups is neutral by construction, with no charge
derivation and no fragment template.  That is why hydrocarbon polymers can be typed
from the table alone.

For polar repeat units it does not collapse.  Poly(ethylene oxide) is the clearest
case: two ``C(H2OR)`` groups (``opls_182``, +0.140, each with two H at +0.060) and one
dialkyl ether oxygen (``opls_180``, -0.400) sum to **+0.120 per repeat unit**, not
zero.  The missing charge lives in the terminal groups of diethyl ether, the molecule
those numbers were fitted to.  Producing a neutral polymer would mean redistributing
charge, which is parameterisation rather than assignment.  :func:`type_repeat_unit`
therefore reports such a unit as ``REQUIRES_CALIBRATION``, with the residual it
measured, instead of quietly scaling the charges to zero.

Polystyrene is a third case worth naming.  OPLS-AA tabulates the benzylic series
``opls_148`` (toluene CH3, -0.065) and ``opls_149`` (ethylbenzene CH2, -0.005), each
chosen so the benzylic group carries +0.115 against the ipso carbon's -0.115.  The
polystyrene backbone needs the next member -- a benzylic **CH** -- and that member is
not in the table.  The convention makes its value obvious, which is exactly why it must
not be written here as though it had been looked up.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from polymer_engine.core.errors import ChemistryError
from polymer_engine.core.logging import get_logger

logger = get_logger("simulation.opls_typing")

#: A repeat unit whose net charge exceeds this is not a neutral polymer.  Tight, because
#: the tabulated OPLS charges are exact decimals: a correct assignment sums to zero to
#: floating-point noise, and anything larger is a real mismatch rather than round-off.
NEUTRALITY_TOLERANCE = 1.0e-6


class TypingStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    #: Every atom typed, but the repeat unit does not come out neutral.
    REQUIRES_CALIBRATION = "REQUIRES_CALIBRATION"
    #: At least one atom has no tabulated type in this assignment table.
    UNSUPPORTED = "UNSUPPORTED"
    #: RDKit missing, or the repeat unit could not be parsed.
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class OplsType:
    """One entry from the installed force field."""

    name: str
    bonded_type: str
    charge: float
    sigma_nm: float
    epsilon_kj: float
    comment: str = ""


@dataclass
class TypingResult:
    status: TypingStatus
    force_field: str = "OPLS-AA"
    force_field_source: str = ""
    assignments: list[tuple[int, str, str, float]] = field(default_factory=list)
    net_charge: float | None = None
    untyped: list[tuple[int, str, str]] = field(default_factory=list)
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.status is TypingStatus.SUPPORTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "force_field": self.force_field,
            "force_field_source": self.force_field_source,
            "n_atoms_typed": len(self.assignments),
            "net_charge": self.net_charge,
            "untyped": [{"index": i, "element": e, "environment": d} for i, e, d in self.untyped],
            "reason": self.reason,
        }


#: Alkane rules as ``(element, carbon_neighbours, hydrogens, opls_type)``.  Every one is
#: a tabulated OPLS-AA alkane type whose group charge is exactly zero.
_ALKANE_RULES: tuple[tuple[str, int, int, str], ...] = (
    ("C", 1, 3, "opls_135"),   # alkane CH3
    ("C", 2, 2, "opls_136"),   # alkane CH2
    ("C", 3, 1, "opls_137"),   # alkane CH
    ("C", 4, 0, "opls_139"),   # alkane C (quaternary)
)
_ALKANE_HYDROGEN = "opls_140"

_TYPE_LINE = re.compile(
    r"^\s*(opls_\S+)\s+(\S+)\s+\d+\s+[\d.]+\s+(-?[\d.]+)\s+\S+\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
)
_ATP_LINE = re.compile(r"^\s*(opls_\S+)\s+[\d.]+\s*;\s*(.*)$")


def load_opls_types(force_field_dir: str | Path) -> dict[str, OplsType]:
    """Read ``ffnonbonded.itp`` (and the ``.atp`` comments) from an installed OPLS-AA."""
    directory = Path(force_field_dir)
    nonbonded = directory / "ffnonbonded.itp"
    if not nonbonded.is_file():
        raise ChemistryError(
            "OPLS-AA force field not found; ffnonbonded.itp is missing",
            directory=str(directory),
        )
    comments: dict[str, str] = {}
    atp = directory / "atomtypes.atp"
    if atp.is_file():
        for line in atp.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _ATP_LINE.match(line)
            if match:
                comments[match.group(1)] = match.group(2).strip()

    types: dict[str, OplsType] = {}
    for line in nonbonded.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith(("[", ";", "#")):
            continue
        match = _TYPE_LINE.match(line)
        if not match:
            continue
        name = match.group(1)
        types[name] = OplsType(
            name=name,
            bonded_type=match.group(2),
            charge=float(match.group(3)),
            sigma_nm=float(match.group(4)),
            epsilon_kj=float(match.group(5)),
            comment=comments.get(name, ""),
        )
    if not types:
        raise ChemistryError(
            "OPLS-AA ffnonbonded.itp contained no atom types", path=str(nonbonded)
        )
    logger.info("Loaded %d OPLS-AA atom types from %s", len(types), directory)
    return types


def find_opls_directory(search: str | Path | None = None) -> Path | None:
    """Locate an installed ``oplsaa.ff``.

    Checks an explicit path, then ``GMXDATA``/``GMXLIB``, then beside the ``gmx``
    executable.  Returns ``None`` rather than raising, so the caller can report the
    force field as unavailable instead of crashing.
    """
    if search:
        candidate = Path(search)
        return candidate if (candidate / "ffnonbonded.itp").is_file() else None

    roots: list[Path] = []
    for variable in ("GMXDATA", "GMXLIB"):
        value = os.environ.get(variable)
        if value:
            roots.extend(Path(part) for part in value.split(os.pathsep) if part)
    gmx = shutil.which("gmx")
    if gmx:
        prefix = Path(gmx).resolve().parent.parent
        roots.append(prefix / "share" / "gromacs" / "top")
        roots.append(prefix / "share" / "top")
    for root in roots:
        for candidate in (root / "oplsaa.ff", root):
            if (candidate / "ffnonbonded.itp").is_file():
                return candidate
    return None


def _describe(atom: Any) -> str:
    neighbours = sorted(n.GetSymbol() for n in atom.GetNeighbors())
    aromatic = " aromatic" if atom.GetIsAromatic() else ""
    return f"{atom.GetSymbol()}{aromatic} bonded to {'+'.join(neighbours) or 'nothing'}"


def type_repeat_unit(
    repeat_unit_smiles: str,
    *,
    types: dict[str, OplsType],
    force_field_source: str = "",
    n_units: int = 3,
) -> TypingResult:
    """Assign OPLS-AA types to a polymer repeat unit, via an explicit oligomer.

    The oligomer matters: hydrogen-capping a repeat unit destroys the in-chain linkage,
    so typing would see a different molecule from the one that will be simulated.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return TypingResult(
            status=TypingStatus.UNCERTAIN,
            reason="RDKit is not installed; repeat units cannot be typed",
        )
    from polymer_engine.polymer.identity import build_oligomer

    built = build_oligomer(repeat_unit_smiles, n_units=n_units)
    if built is None:
        return TypingResult(
            status=TypingStatus.UNCERTAIN,
            reason="repeat unit does not have exactly two attachment points",
        )
    molecule = Chem.AddHs(built[0])

    result = TypingResult(status=TypingStatus.SUPPORTED, force_field_source=force_field_source)
    total = 0.0
    for atom in molecule.GetAtoms():
        symbol, index = atom.GetSymbol(), atom.GetIdx()
        assigned: str | None = None
        if symbol == "H":
            neighbours = list(atom.GetNeighbors())
            if (
                len(neighbours) == 1
                and neighbours[0].GetSymbol() == "C"
                and not neighbours[0].GetIsAromatic()
            ):
                assigned = _ALKANE_HYDROGEN
        elif symbol == "C" and not atom.GetIsAromatic():
            carbons = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "C")
            hydrogens = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "H")
            others = [n for n in atom.GetNeighbors() if n.GetSymbol() not in ("C", "H")]
            if not others and carbons + hydrogens == atom.GetDegree():
                for _element, n_carbons, n_hydrogens, name in _ALKANE_RULES:
                    if carbons == n_carbons and hydrogens == n_hydrogens:
                        assigned = name
                        break
        if assigned is None or assigned not in types:
            result.untyped.append((index, symbol, _describe(atom)))
            continue
        entry = types[assigned]
        result.assignments.append((index, assigned, entry.bonded_type, entry.charge))
        total += entry.charge

    result.net_charge = round(total, 9)
    if result.untyped:
        elements = sorted({e for _, e, _ in result.untyped})
        result.status = TypingStatus.UNSUPPORTED
        result.reason = (
            f"{len(result.untyped)} atom(s) have no tabulated OPLS-AA type in this "
            f"assignment table (elements: {', '.join(elements)}); "
            f"example: {result.untyped[0][2]}"
        )
    elif abs(result.net_charge) > NEUTRALITY_TOLERANCE:
        result.status = TypingStatus.REQUIRES_CALIBRATION
        result.reason = (
            f"every atom was typed, but the tabulated charges sum to "
            f"{result.net_charge:+.4f} e rather than zero; a neutral polymer would "
            f"require redistributing charge, which is parameterisation not assignment"
        )
    else:
        result.reason = (
            f"all {len(result.assignments)} atoms typed from tabulated OPLS-AA "
            f"parameters; net charge {result.net_charge:+.1e} e"
        )
    return result


__all__ = [
    "NEUTRALITY_TOLERANCE",
    "OPLS_DEFAULTS",
    "OplsType",
    "TypingResult",
    "TypingStatus",
    "find_opls_directory",
    "load_opls_types",
    "type_molecule",
    "type_repeat_unit",
    "write_opls_topology",
]


# ==========================================================================
# Topology generation
# ==========================================================================
#: Interactions are emitted *without* inline parameters so GROMACS resolves them from
#: ``ffbonded.itp`` by bonded type.  Writing numbers here would mean copying published
#: parameters into a second place where they could drift out of step with the force
#: field -- and it is how ``gmx x2top`` produces a topology that grompp accepts and
#: mdrun runs while being physically wrong: for n-butane it emits three dihedrals
#: carrying placeholder Ryckaert-Bellemans coefficients (60, 5, 3, 60, 5, 3) in place of
#: the ~27 real ones, giving a torsional energy of several hundred kJ/mol.
OPLS_DEFAULTS = """\
; Generated by polymer-engine from the installed OPLS-AA force field.
; Bonded interactions carry no inline parameters: GROMACS looks each one up in
; ffbonded.itp by bonded type, so the published values are used and never copied.
#include "oplsaa.ff/forcefield.itp"
"""


def _connectivity(molecule: Any) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]],
                                          list[tuple[int, int, int, int]]]:
    """Bonds, angles and proper dihedrals, one-based, in GROMACS order."""
    bonds = sorted(
        tuple(sorted((b.GetBeginAtomIdx() + 1, b.GetEndAtomIdx() + 1)))
        for b in molecule.GetBonds()
    )
    neighbours: dict[int, list[int]] = {}
    for atom in molecule.GetAtoms():
        neighbours[atom.GetIdx() + 1] = sorted(n.GetIdx() + 1 for n in atom.GetNeighbors())

    angles: list[tuple[int, int, int]] = []
    for centre, attached in neighbours.items():
        for i, left in enumerate(attached):
            for right in attached[i + 1:]:
                angles.append((left, centre, right))

    dihedrals: list[tuple[int, int, int, int]] = []
    for j, k in bonds:
        for i in neighbours[j]:
            if i == k:
                continue
            for l_atom in neighbours[k]:
                if l_atom in (j, i):
                    continue
                dihedrals.append((i, j, k, l_atom))
    return bonds, sorted(angles), sorted(dihedrals)


def write_opls_topology(
    result: TypingResult,
    molecule: Any,
    path: str | Path,
    *,
    molecule_name: str = "POL",
    include_line: str | None = None,
    atom_names: list[str] | None = None,
    residue_name: str | None = None,
) -> Path:
    """Write a GROMACS topology for an already-typed molecule.

    Refuses unless ``result`` is :attr:`TypingStatus.SUPPORTED`.  A topology built from
    a partial typing would be exactly the silent-wrongness this module exists to
    prevent.

    ``atom_names`` and ``residue_name`` should be taken from the structure file this
    topology will be paired with.  GROMACS matches topology and coordinates by *order*,
    not by name, so a name mismatch is only a warning -- but it is a warning that fires
    once per atom and would drown out a genuine ordering error in the same output.
    Inventing a second naming scheme here also means ``grompp`` can only run with
    ``-maxwarn``, which is exactly the flag that hides broken systems.
    """
    if not result.usable:
        raise ChemistryError(
            "Refusing to write a topology from a typing that is not SUPPORTED",
            status=result.status.value, reason=result.reason,
        )
    if len(result.assignments) != molecule.GetNumAtoms():
        raise ChemistryError(
            "Typing does not cover every atom in the molecule",
            typed=len(result.assignments), atoms=molecule.GetNumAtoms(),
        )

    bonds, angles, dihedrals = _connectivity(molecule)
    # 1-4 pairs come from the proper dihedrals, deduplicated: GROMACS needs them to
    # apply the OPLS 0.5 scaling to intramolecular LJ and Coulomb.
    pairs = sorted({tuple(sorted((d[0], d[3]))) for d in dihedrals})

    residue = residue_name or molecule_name
    if atom_names is not None and len(atom_names) != len(result.assignments):
        raise ChemistryError(
            "atom_names does not match the number of typed atoms",
            names=len(atom_names), atoms=len(result.assignments),
        )
    lines: list[str] = [include_line or OPLS_DEFAULTS, "", "[ moleculetype ]", "; name  nrexcl",
                        f"{molecule_name}   3", "", "[ atoms ]",
                        ";  nr   type  resnr residue  atom  cgnr    charge      mass"]
    for order, (index, type_name, _bonded, charge) in enumerate(result.assignments, start=1):
        if atom_names is not None:
            label = atom_names[order - 1]
        else:
            label = molecule.GetAtomWithIdx(index).GetSymbol() + str(order)
        lines.append(
            f"{order:5d} {type_name:>10s}      1  {residue:<4s} "
            f"{label:>5s} {order:5d} {charge:10.4f}"
        )
    lines += ["", "[ bonds ]", ";  ai    aj funct"]
    lines += [f"{a:5d} {b:5d}     1" for a, b in bonds]
    lines += ["", "[ pairs ]", ";  ai    aj funct"]
    lines += [f"{a:5d} {b:5d}     1" for a, b in pairs]
    lines += ["", "[ angles ]", ";  ai    aj    ak funct"]
    lines += [f"{a:5d} {b:5d} {c:5d}     1" for a, b, c in angles]
    lines += ["", "[ dihedrals ]", ";  ai    aj    ak    al funct"]
    lines += [f"{a:5d} {b:5d} {c:5d} {d:5d}     3" for a, b, c, d in dihedrals]
    lines += ["", "[ system ]", f"{molecule_name} in vacuo", "", "[ molecules ]",
              "; molecule  count", f"{molecule_name}          1", ""]

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "Wrote %s: %d atoms, %d bonds, %d angles, %d dihedrals, %d pairs",
        destination.name, len(result.assignments), len(bonds), len(angles),
        len(dihedrals), len(pairs),
    )
    return destination


def type_molecule(molecule: Any, *, types: dict[str, OplsType],
                  force_field_source: str = "") -> TypingResult:
    """Type an explicit molecule (already H-complete), rather than a repeat unit."""
    result = TypingResult(status=TypingStatus.SUPPORTED, force_field_source=force_field_source)
    total = 0.0
    for atom in molecule.GetAtoms():
        symbol, index = atom.GetSymbol(), atom.GetIdx()
        assigned: str | None = None
        if symbol == "H":
            neighbours = list(atom.GetNeighbors())
            if (len(neighbours) == 1 and neighbours[0].GetSymbol() == "C"
                    and not neighbours[0].GetIsAromatic()):
                assigned = _ALKANE_HYDROGEN
        elif symbol == "C" and not atom.GetIsAromatic():
            carbons = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "C")
            hydrogens = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "H")
            others = [n for n in atom.GetNeighbors() if n.GetSymbol() not in ("C", "H")]
            if not others and carbons + hydrogens == atom.GetDegree():
                for _element, n_carbons, n_hydrogens, name in _ALKANE_RULES:
                    if carbons == n_carbons and hydrogens == n_hydrogens:
                        assigned = name
                        break
        if assigned is None or assigned not in types:
            result.untyped.append((index, symbol, _describe(atom)))
            continue
        entry = types[assigned]
        result.assignments.append((index, assigned, entry.bonded_type, entry.charge))
        total += entry.charge
    result.net_charge = round(total, 9)
    if result.untyped:
        result.status = TypingStatus.UNSUPPORTED
        result.reason = f"{len(result.untyped)} atom(s) have no tabulated OPLS-AA type"
    elif abs(result.net_charge) > NEUTRALITY_TOLERANCE:
        result.status = TypingStatus.REQUIRES_CALIBRATION
        result.reason = f"charges sum to {result.net_charge:+.4f} e rather than zero"
    else:
        result.reason = f"all {len(result.assignments)} atoms typed; net charge zero"
    return result
