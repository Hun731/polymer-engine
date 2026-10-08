"""Quantum-chemistry job specifications.

Every scientifically consequential choice -- method, basis set, solvation model,
convergence criteria -- is an explicit field with no silently-applied default that
could change an answer.  Where a default exists it is a *technical* one (memory,
scratch handling) rather than a scientific one.

The engine deliberately does **not** pick a method or basis for you.  ``method`` and
``basis`` are required.  A wrong functional produces a plausible number, and a
plausible wrong number is worse than a refusal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from polymer_engine.core.errors import ParameterValidationError

#: ORCA convergence keyword tiers, loosest to tightest.
SCF_CONVERGENCE_LEVELS = ("SloppySCF", "LooseSCF", "NormalSCF", "TightSCF", "VeryTightSCF")
GEOM_CONVERGENCE_LEVELS = ("LooseOpt", "NormalOpt", "TightOpt", "VeryTightOpt")

#: Implicit-solvation models ORCA exposes through simple keywords.
SOLVATION_MODELS = ("CPCM", "SMD")


class JobKind(str, Enum):
    SINGLE_POINT = "single_point"
    OPTIMIZATION = "optimization"
    FREQUENCY = "frequency"
    OPT_FREQ = "opt_freq"
    TORSION_SCAN = "torsion_scan"
    CONFORMER = "conformer"


@dataclass(frozen=True, slots=True)
class Atom:
    """One atom in a Cartesian structure.  Coordinates are angstrom, as ORCA expects."""

    symbol: str
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol[0].isalpha():
            raise ParameterValidationError("Atom symbol must start with a letter", symbol=self.symbol)
        for name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if not math.isfinite(value):
                raise ParameterValidationError(f"Atom coordinate {name} is not finite", symbol=self.symbol)

    def as_line(self) -> str:
        return f"{self.symbol:<3s} {self.x:>14.8f} {self.y:>14.8f} {self.z:>14.8f}"

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(slots=True)
class Structure:
    """A molecular structure with its charge and spin multiplicity.

    Charge and multiplicity are not guessable from coordinates, so both are required
    and their consistency with the electron count is checked.
    """

    atoms: list[Atom]
    charge: int = 0
    multiplicity: int = 1
    name: str = "structure"

    def __post_init__(self) -> None:
        if not self.atoms:
            raise ParameterValidationError("A structure needs at least one atom")
        if self.multiplicity < 1:
            raise ParameterValidationError("Multiplicity must be >= 1", multiplicity=self.multiplicity)
        problems = self.electron_consistency()
        if problems:
            raise ParameterValidationError(problems, charge=self.charge, multiplicity=self.multiplicity)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def electron_count(self) -> int | None:
        """Total electrons, or ``None`` if any symbol is not a known element."""
        total = 0
        for atom in self.atoms:
            number = ATOMIC_NUMBERS.get(atom.symbol.capitalize())
            if number is None:
                return None
            total += number
        return total - self.charge

    def electron_consistency(self) -> str:
        """Empty string when charge and multiplicity can coexist, else the reason.

        An even electron count cannot support an even multiplicity (and vice versa);
        ORCA would reject the job, but catching it here saves the round trip and gives
        a clearer message.
        """
        electrons = self.electron_count()
        if electrons is None:
            return ""
        if electrons < 0:
            return f"charge {self.charge} implies a negative electron count"
        unpaired = self.multiplicity - 1
        if (electrons - unpaired) % 2 != 0:
            return (
                f"{electrons} electrons cannot have multiplicity {self.multiplicity}; "
                f"an {'even' if electrons % 2 == 0 else 'odd'} electron count requires an "
                f"{'odd' if electrons % 2 == 0 else 'even'} multiplicity"
            )
        return ""

    def as_xyz(self) -> str:
        header = f"{self.n_atoms}\n{self.name}\n"
        return header + "\n".join(a.as_line() for a in self.atoms) + "\n"

    @classmethod
    def from_xyz(cls, text: str, *, charge: int = 0, multiplicity: int = 1) -> Structure:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ParameterValidationError("XYZ text is empty")
        try:
            declared = int(lines[0].split()[0])
        except (ValueError, IndexError):
            raise ParameterValidationError("XYZ first line must be an atom count") from None
        body = lines[2 : 2 + declared] if len(lines) > 2 else []
        if len(body) != declared:
            raise ParameterValidationError(
                "XYZ atom count does not match the number of coordinate lines",
                declared=declared,
                found=len(body),
            )
        atoms = []
        for line in body:
            parts = line.split()
            if len(parts) < 4:
                raise ParameterValidationError("XYZ coordinate line is malformed", line=line[:60])
            atoms.append(Atom(parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
        name = lines[1].strip() if len(lines) > 1 else "structure"
        return cls(atoms=atoms, charge=charge, multiplicity=multiplicity, name=name or "structure")


@dataclass(frozen=True, slots=True)
class TorsionSpec:
    """A dihedral to scan, by zero-based atom index (ORCA's convention)."""

    atoms: tuple[int, int, int, int]
    start_deg: float
    stop_deg: float
    n_points: int

    def __post_init__(self) -> None:
        if len(set(self.atoms)) != 4:
            raise ParameterValidationError("A torsion needs four distinct atoms", atoms=list(self.atoms))
        if any(i < 0 for i in self.atoms):
            raise ParameterValidationError("Atom indices must be non-negative", atoms=list(self.atoms))
        if self.n_points < 2:
            raise ParameterValidationError("A scan needs at least 2 points", n_points=self.n_points)
        if self.start_deg == self.stop_deg:
            raise ParameterValidationError("Scan start and stop are identical", start=self.start_deg)

    @property
    def step_deg(self) -> float:
        return (self.stop_deg - self.start_deg) / (self.n_points - 1)

    def angles(self) -> list[float]:
        return [self.start_deg + i * self.step_deg for i in range(self.n_points)]


@dataclass(slots=True)
class QMJobSpec:
    """A complete, reproducible description of one quantum-chemistry calculation."""

    kind: JobKind
    structure: Structure
    method: str
    basis: str
    label: str = "qm"
    dispersion: str | None = None
    solvation_model: Literal["CPCM", "SMD"] | None = None
    solvent: str | None = None
    scf_convergence: str = "TightSCF"
    geom_convergence: str = "NormalOpt"
    max_scf_iterations: int = 125
    max_geom_iterations: int = 60
    torsion: TorsionSpec | None = None
    n_procs: int = 1
    memory_mb_per_core: int = 2000
    timeout_s: float | None = 3600.0
    extra_keywords: tuple[str, ...] = ()
    extra_blocks: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        problems = self.validate()
        if problems:
            raise ParameterValidationError("QM job specification is not usable", problems=problems)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.method.strip():
            problems.append("method is required; the engine will not choose one for you")
        if not self.basis.strip():
            problems.append("basis set is required; the engine will not choose one for you")
        if self.scf_convergence not in SCF_CONVERGENCE_LEVELS:
            problems.append(f"scf_convergence must be one of {SCF_CONVERGENCE_LEVELS}")
        if self.geom_convergence not in GEOM_CONVERGENCE_LEVELS:
            problems.append(f"geom_convergence must be one of {GEOM_CONVERGENCE_LEVELS}")
        if self.solvation_model is not None and self.solvation_model not in SOLVATION_MODELS:
            problems.append(f"solvation_model must be one of {SOLVATION_MODELS}")
        if self.solvation_model is not None and not self.solvent:
            problems.append("a solvation model requires a solvent name")
        if self.solvent and self.solvation_model is None:
            problems.append("a solvent was given without a solvation model")
        if self.n_procs < 1:
            problems.append("n_procs must be at least 1")
        if self.memory_mb_per_core < 100:
            problems.append("memory_mb_per_core below 100 MB will not run a useful job")
        if self.max_scf_iterations < 1:
            problems.append("max_scf_iterations must be positive")
        if self.kind is JobKind.TORSION_SCAN and self.torsion is None:
            problems.append("a torsion scan requires a TorsionSpec")
        if self.torsion is not None:
            n = self.structure.n_atoms
            out_of_range = [i for i in self.torsion.atoms if i >= n]
            if out_of_range:
                problems.append(
                    f"torsion atom indices {out_of_range} are outside the structure "
                    f"({n} atoms, zero-based)"
                )
        if self.timeout_s is not None and self.timeout_s <= 0:
            problems.append("timeout_s must be positive when set")
        return problems

    @property
    def requires_geometry_convergence(self) -> bool:
        return self.kind in {JobKind.OPTIMIZATION, JobKind.OPT_FREQ, JobKind.TORSION_SCAN}

    @property
    def requires_frequencies(self) -> bool:
        return self.kind in {JobKind.FREQUENCY, JobKind.OPT_FREQ}

    def fingerprint(self) -> str:
        """Digest of everything that determines the result.

        Excludes ``n_procs``, memory and timeout: those change how long the job takes,
        not what it computes.
        """
        from polymer_engine.core.provenance import canonical_hash

        return canonical_hash(
            {
                "kind": self.kind.value,
                "method": self.method,
                "basis": self.basis,
                "dispersion": self.dispersion,
                "solvation_model": self.solvation_model,
                "solvent": self.solvent,
                "scf_convergence": self.scf_convergence,
                "geom_convergence": self.geom_convergence,
                "charge": self.structure.charge,
                "multiplicity": self.structure.multiplicity,
                "atoms": [(a.symbol, round(a.x, 8), round(a.y, 8), round(a.z, 8)) for a in self.structure.atoms],
                "torsion": None
                if self.torsion is None
                else {
                    "atoms": list(self.torsion.atoms),
                    "start": self.torsion.start_deg,
                    "stop": self.torsion.stop_deg,
                    "n_points": self.torsion.n_points,
                },
                "extra_keywords": list(self.extra_keywords),
                "extra_blocks": list(self.extra_blocks),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "label": self.label,
            "method": self.method,
            "basis": self.basis,
            "dispersion": self.dispersion,
            "solvation_model": self.solvation_model,
            "solvent": self.solvent,
            "scf_convergence": self.scf_convergence,
            "geom_convergence": self.geom_convergence,
            "max_scf_iterations": self.max_scf_iterations,
            "max_geom_iterations": self.max_geom_iterations,
            "charge": self.structure.charge,
            "multiplicity": self.structure.multiplicity,
            "n_atoms": self.structure.n_atoms,
            "n_procs": self.n_procs,
            "memory_mb_per_core": self.memory_mb_per_core,
            "timeout_s": self.timeout_s,
            "torsion": None if self.torsion is None else list(self.torsion.atoms),
            "extra_keywords": list(self.extra_keywords),
            "fingerprint": self.fingerprint(),
            "notes": self.notes,
        }


#: Atomic numbers for the elements that appear in organic/polymer chemistry.
ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Ti": 22, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
    "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "Sn": 50, "Sb": 51, "Te": 52,
    "I": 53, "Xe": 54, "Pt": 78, "Au": 79, "Hg": 80, "Pb": 82,
}


__all__ = [
    "ATOMIC_NUMBERS",
    "GEOM_CONVERGENCE_LEVELS",
    "SCF_CONVERGENCE_LEVELS",
    "SOLVATION_MODELS",
    "Atom",
    "JobKind",
    "QMJobSpec",
    "Structure",
    "TorsionSpec",
]
