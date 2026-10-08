"""Structured parsing of ORCA output.

The organising rule: **ORCA exiting 0 does not mean the calculation succeeded.**  A
relaxed surface scan that hits its optimisation-cycle limit prints
``ORCA finished by error termination``, never prints ``ORCA TERMINATED NORMALLY``, and
still exits with status 0.  A geometry optimisation can terminate normally without
converging.  Both are scientific failures and both are detected here from the *output*,
not from the exit status.

Every quantity is extracted with an anchored pattern rather than a loose search, and
anything the parser cannot find is reported as absent rather than defaulted.
"""

from __future__ import annotations

import gzip
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from polymer_engine.core.errors import ResponseFormatError

#: Hartree -> kJ/mol, the engine's canonical molar energy unit.
HARTREE_TO_KJ_MOL = 2625.4996394799


class QMStatus(str, Enum):
    """What actually happened, as distinct from the process exit status."""

    COMPLETED = "COMPLETED"
    #: Ran to a normal end but the science did not converge.
    FAILED_SCIENTIFICALLY = "FAILED_SCIENTIFICALLY"
    #: ORCA reported an internal error termination.
    FAILED_TERMINATION = "FAILED_TERMINATION"
    #: Output ended mid-calculation (killed, out of time, out of disk).
    INCOMPLETE = "INCOMPLETE"
    #: Output could not be interpreted at all.
    UNPARSEABLE = "UNPARSEABLE"


# --------------------------------------------------------------------------
# Anchored patterns
# --------------------------------------------------------------------------
_NORMAL_TERMINATION = re.compile(r"\*+ORCA TERMINATED NORMALLY\*+")
_ERROR_TERMINATION = re.compile(r"ORCA finished by error termination", re.IGNORECASE)
_ABORTING = re.compile(r"^\s*(?:ORCA\s+)?(?:ABORTING THE RUN|aborting the run)", re.MULTILINE)
_FINAL_ENERGY = re.compile(r"^FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", re.MULTILINE)
_SCF_CONVERGED = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
_SCF_NOT_CONVERGED = re.compile(
    r"SCF NOT CONVERGED AFTER\s+(\d+)\s+CYCLES|"
    r"This wavefunction IS NOT FULLY CONVERGED",
    re.IGNORECASE,
)
_GEOM_CONVERGED = re.compile(r"THE OPTIMIZATION HAS CONVERGED")
_GEOM_NOT_CONVERGED = re.compile(
    r"The optimization did not converge but reached the maximum number of\s*\n\s*optimization cycles",
    re.IGNORECASE,
)
_OPT_CYCLES = re.compile(r"\(AFTER\s+(\d+)\s+CYCLES\)")
_VERSION = re.compile(r"Program Version\s+([0-9][0-9.]*)")
_DIPOLE_DEBYE = re.compile(r"^Magnitude \(Debye\)\s*:\s*(-?\d+\.\d+)", re.MULTILINE)
_RUN_TIME = re.compile(
    r"TOTAL RUN TIME:\s*(\d+)\s*days\s*(\d+)\s*hours\s*(\d+)\s*minutes\s*(\d+)\s*seconds\s*(\d+)\s*msec"
)
_INPUT_KEYWORDS = re.compile(r"^\|\s*\d+>\s*!\s*(.+)$", re.MULTILINE)

_COORD_HEADER = re.compile(
    r"^-{3,}\s*\nCARTESIAN COORDINATES \(ANGSTROEM\)\s*\n-{3,}\s*\n", re.MULTILINE
)
_COORD_LINE = re.compile(r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")

_FREQ_LINE = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s*cm\*\*-1", re.MULTILINE)
_ZPE = re.compile(r"^Zero point energy\s*\.\.\.\s*(-?\d+\.\d+)\s*Eh", re.MULTILINE)
_GIBBS = re.compile(r"^Final Gibbs free energy\s*\.\.\.\s*(-?\d+\.\d+)\s*Eh", re.MULTILINE)
_ENTHALPY = re.compile(r"^Total Enthalpy\s*\.\.\.\s*(-?\d+\.\d+)\s*Eh", re.MULTILINE)

_CHARGE_BLOCK = re.compile(
    r"^(MULLIKEN|LOEWDIN) ATOMIC CHARGES\s*\n-+\s*\n(.*?)(?:\n\s*\n|\nSum of atomic charges)",
    re.MULTILINE | re.DOTALL,
)
_CHARGE_LINE = re.compile(r"^\s*(\d+)\s+([A-Z][a-z]?)\s*:\s*(-?\d+\.\d+)", re.MULTILINE)

_SCAN_SURFACE = re.compile(
    r"The Calculated Surface using the '?(?P<which>[^'\n]+?)'? ?(?:energy)?\s*\n(?P<body>(?:\s*-?\d+\.\d+\s+-?\d+\.\d+\s*\n)+)",
)


@dataclass(slots=True)
class GeometryConvergence:
    """The five criteria from ORCA's geometry-convergence table."""

    energy_change: float | None = None
    rms_gradient: float | None = None
    max_gradient: float | None = None
    rms_step: float | None = None
    max_step: float | None = None
    all_converged: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "energy_change": self.energy_change,
            "rms_gradient": self.rms_gradient,
            "max_gradient": self.max_gradient,
            "rms_step": self.rms_step,
            "max_step": self.max_step,
            "all_converged": self.all_converged,
        }


@dataclass(slots=True)
class ScanPoint:
    coordinate: float
    energy_hartree: float

    @property
    def energy_kj_mol(self) -> float:
        return self.energy_hartree * HARTREE_TO_KJ_MOL


@dataclass
class OrcaOutput:
    """Everything the parser could establish about one ORCA run."""

    status: QMStatus
    diagnostics: list[str] = field(default_factory=list)
    orca_version: str | None = None
    keywords: list[str] = field(default_factory=list)

    normal_termination: bool = False
    error_termination: bool = False

    final_energy_hartree: float | None = None
    all_energies_hartree: list[float] = field(default_factory=list)
    scf_converged: bool | None = None
    scf_cycles: int | None = None

    geometry_converged: bool | None = None
    geometry_convergence: GeometryConvergence = field(default_factory=GeometryConvergence)
    optimization_cycles: int | None = None
    final_geometry: list[tuple[str, float, float, float]] = field(default_factory=list)
    initial_geometry: list[tuple[str, float, float, float]] = field(default_factory=list)

    frequencies_cm: list[float] = field(default_factory=list)
    n_imaginary: int | None = None
    zero_point_energy_hartree: float | None = None
    gibbs_free_energy_hartree: float | None = None
    enthalpy_hartree: float | None = None

    dipole_debye: float | None = None
    mulliken_charges: list[float] = field(default_factory=list)
    loewdin_charges: list[float] = field(default_factory=list)

    scan_points: list[ScanPoint] = field(default_factory=list)
    run_time_s: float | None = None

    @property
    def succeeded(self) -> bool:
        """True only for a scientifically complete calculation."""
        return self.status is QMStatus.COMPLETED

    @property
    def final_energy_kj_mol(self) -> float | None:
        if self.final_energy_hartree is None:
            return None
        return self.final_energy_hartree * HARTREE_TO_KJ_MOL

    def scan_profile_kj_mol(self, *, relative: bool = True) -> list[tuple[float, float]]:
        """The scan as ``(angle, energy)`` pairs in kJ/mol, relative to the minimum."""
        if not self.scan_points:
            return []
        energies = [p.energy_kj_mol for p in self.scan_points]
        offset = min(energies) if relative else 0.0
        return [(p.coordinate, e - offset) for p, e in zip(self.scan_points, energies, strict=True)]

    def torsional_barrier_kj_mol(self) -> float | None:
        """Highest minus lowest point of a scan, or ``None`` without a scan."""
        if len(self.scan_points) < 2:
            return None
        energies = [p.energy_kj_mol for p in self.scan_points]
        return max(energies) - min(energies)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "succeeded": self.succeeded,
            "diagnostics": self.diagnostics,
            "orca_version": self.orca_version,
            "keywords": self.keywords,
            "normal_termination": self.normal_termination,
            "error_termination": self.error_termination,
            "final_energy_hartree": self.final_energy_hartree,
            "final_energy_kj_mol": self.final_energy_kj_mol,
            "scf_converged": self.scf_converged,
            "scf_cycles": self.scf_cycles,
            "geometry_converged": self.geometry_converged,
            "geometry_convergence": self.geometry_convergence.as_dict(),
            "optimization_cycles": self.optimization_cycles,
            "n_atoms": len(self.final_geometry),
            "frequencies_cm": self.frequencies_cm,
            "n_imaginary": self.n_imaginary,
            "zero_point_energy_hartree": self.zero_point_energy_hartree,
            "gibbs_free_energy_hartree": self.gibbs_free_energy_hartree,
            "enthalpy_hartree": self.enthalpy_hartree,
            "dipole_debye": self.dipole_debye,
            "mulliken_charges": self.mulliken_charges,
            "loewdin_charges": self.loewdin_charges,
            "n_scan_points": len(self.scan_points),
            "torsional_barrier_kj_mol": self.torsional_barrier_kj_mol(),
            "run_time_s": self.run_time_s,
        }


def read_output_text(path: str | Path) -> str:
    """Read an ORCA log, transparently handling gzip."""
    path = Path(path)
    if not path.exists():
        raise ResponseFormatError("ORCA output file does not exist", path=str(path))
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def parse_orca_output(text: str, *, expect_geometry: bool = False, expect_frequencies: bool = False) -> OrcaOutput:
    """Parse an ORCA log into a structured result and classify what happened.

    ``expect_geometry`` / ``expect_frequencies`` let the caller say what the job was
    supposed to produce, so a missing result is a *failure* rather than merely absent.
    """
    if not text or not text.strip():
        return OrcaOutput(status=QMStatus.UNPARSEABLE, diagnostics=["output is empty"])

    result = OrcaOutput(status=QMStatus.UNPARSEABLE)

    version = _VERSION.search(text)
    if version:
        result.orca_version = version.group(1)
    elif "O   R   C   A" not in text:
        result.diagnostics.append("output does not look like an ORCA log")
        return result

    keyword_lines = _INPUT_KEYWORDS.findall(text)
    result.keywords = [k.strip() for k in keyword_lines]

    result.normal_termination = bool(_NORMAL_TERMINATION.search(text))
    result.error_termination = bool(_ERROR_TERMINATION.search(text)) or bool(_ABORTING.search(text))

    _parse_energies(text, result)
    _parse_scf(text, result)
    _parse_geometry(text, result)
    _parse_frequencies(text, result)
    _parse_properties(text, result)
    _parse_scan(text, result)

    match = _RUN_TIME.search(text)
    if match:
        days, hours, minutes, seconds, msec = (int(g) for g in match.groups())
        result.run_time_s = days * 86400 + hours * 3600 + minutes * 60 + seconds + msec / 1000.0

    result.status = _classify(result, expect_geometry=expect_geometry, expect_frequencies=expect_frequencies)
    return result


def _parse_energies(text: str, result: OrcaOutput) -> None:
    energies = [float(m) for m in _FINAL_ENERGY.findall(text)]
    result.all_energies_hartree = energies
    if energies:
        # The last one is the energy at the final geometry.
        result.final_energy_hartree = energies[-1]


def _parse_scf(text: str, result: OrcaOutput) -> None:
    if _SCF_NOT_CONVERGED.search(text):
        result.scf_converged = False
        result.diagnostics.append("SCF did not converge")
        return
    matches = _SCF_CONVERGED.findall(text)
    if matches:
        result.scf_converged = True
        result.scf_cycles = int(matches[-1])


def _parse_geometry(text: str, result: OrcaOutput) -> None:
    blocks = _coordinate_blocks(text)
    if blocks:
        result.initial_geometry = blocks[0]
        result.final_geometry = blocks[-1]

    if _GEOM_NOT_CONVERGED.search(text):
        result.geometry_converged = False
        result.diagnostics.append(
            "geometry optimisation reached the maximum number of cycles without converging"
        )
    elif _GEOM_CONVERGED.search(text):
        result.geometry_converged = True

    cycles = _OPT_CYCLES.findall(text)
    if cycles:
        result.optimization_cycles = int(cycles[-1])

    result.geometry_convergence = _parse_convergence_table(text)


def _coordinate_blocks(text: str) -> list[list[tuple[str, float, float, float]]]:
    """Every ``CARTESIAN COORDINATES (ANGSTROEM)`` block, in file order."""
    blocks: list[list[tuple[str, float, float, float]]] = []
    for match in _COORD_HEADER.finditer(text):
        atoms: list[tuple[str, float, float, float]] = []
        for line in text[match.end() :].splitlines():
            if not line.strip():
                break
            parsed = _COORD_LINE.match(line)
            if parsed is None:
                break
            atoms.append(
                (parsed.group(1), float(parsed.group(2)), float(parsed.group(3)), float(parsed.group(4)))
            )
        if atoms:
            blocks.append(atoms)
    return blocks


_CONVERGENCE_ROW = re.compile(
    r"^\s*(Energy change|RMS gradient|MAX gradient|RMS step|MAX step)\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(YES|NO)\s*$",
    re.MULTILINE,
)


def _parse_convergence_table(text: str) -> GeometryConvergence:
    """Read the last geometry-convergence table in the log."""
    start = text.rfind("|Geometry convergence|")
    if start < 0:
        return GeometryConvergence()
    window = text[start : start + 2000]
    convergence = GeometryConvergence()
    flags: list[bool] = []
    for row in _CONVERGENCE_ROW.finditer(window):
        item, value, _tolerance, converged = row.groups()
        numeric = float(value)
        flags.append(converged == "YES")
        if item == "Energy change":
            convergence.energy_change = numeric
        elif item == "RMS gradient":
            convergence.rms_gradient = numeric
        elif item == "MAX gradient":
            convergence.max_gradient = numeric
        elif item == "RMS step":
            convergence.rms_step = numeric
        elif item == "MAX step":
            convergence.max_step = numeric
    if flags:
        convergence.all_converged = all(flags)
    return convergence


def _parse_frequencies(text: str, result: OrcaOutput) -> None:
    start = text.rfind("VIBRATIONAL FREQUENCIES")
    if start < 0:
        return
    end = text.find("NORMAL MODES", start)
    window = text[start : end if end > 0 else start + 20000]
    frequencies = [float(value) for _, value in _FREQ_LINE.findall(window)]
    if not frequencies:
        return
    result.frequencies_cm = frequencies
    # ORCA prints the six (or five) zero modes explicitly; a genuinely imaginary mode
    # is printed as a negative number.
    result.n_imaginary = sum(1 for f in frequencies if f < 0.0)

    for pattern, attribute in (
        (_ZPE, "zero_point_energy_hartree"),
        (_GIBBS, "gibbs_free_energy_hartree"),
        (_ENTHALPY, "enthalpy_hartree"),
    ):
        match = pattern.search(text)
        if match:
            setattr(result, attribute, float(match.group(1)))


def _parse_properties(text: str, result: OrcaOutput) -> None:
    dipole = _DIPOLE_DEBYE.findall(text)
    if dipole:
        result.dipole_debye = float(dipole[-1])

    for match in _CHARGE_BLOCK.finditer(text):
        kind, body = match.group(1), match.group(2)
        charges = [float(value) for _, _, value in _CHARGE_LINE.findall(body)]
        if not charges:
            continue
        if kind == "MULLIKEN":
            result.mulliken_charges = charges
        else:
            result.loewdin_charges = charges


def _parse_scan(text: str, result: OrcaOutput) -> None:
    """Read the relaxed-surface-scan energy table.

    ORCA prints several surfaces ('Actual Energy', 'SCF energy', ...).  The first is
    the one that corresponds to the requested level of theory.
    """
    match = _SCAN_SURFACE.search(text)
    if match is None:
        return
    points: list[ScanPoint] = []
    for line in match.group("body").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            points.append(ScanPoint(coordinate=float(parts[0]), energy_hartree=float(parts[1])))
        except ValueError:
            continue
    result.scan_points = points


def _classify(result: OrcaOutput, *, expect_geometry: bool, expect_frequencies: bool) -> QMStatus:
    """Decide what actually happened.

    Order matters: an explicit error termination outranks everything, and a missing
    normal termination means the log stopped mid-calculation.
    """
    if result.error_termination:
        result.diagnostics.append("ORCA reported an error termination")
        return QMStatus.FAILED_TERMINATION

    if not result.normal_termination:
        result.diagnostics.append(
            "output has no 'ORCA TERMINATED NORMALLY' banner; the run did not finish"
        )
        return QMStatus.INCOMPLETE

    if result.scf_converged is False:
        return QMStatus.FAILED_SCIENTIFICALLY

    if expect_geometry:
        if result.geometry_converged is False:
            return QMStatus.FAILED_SCIENTIFICALLY
        if result.geometry_converged is None:
            result.diagnostics.append(
                "a geometry optimisation was expected but no convergence statement was found"
            )
            return QMStatus.FAILED_SCIENTIFICALLY

    if expect_frequencies and not result.frequencies_cm:
        result.diagnostics.append("frequencies were expected but none were found in the output")
        return QMStatus.FAILED_SCIENTIFICALLY

    if result.final_energy_hartree is None:
        result.diagnostics.append("no FINAL SINGLE POINT ENERGY was found")
        return QMStatus.FAILED_SCIENTIFICALLY

    if not math.isfinite(result.final_energy_hartree):
        result.diagnostics.append("final energy is not finite")
        return QMStatus.FAILED_SCIENTIFICALLY

    return QMStatus.COMPLETED


def parse_orca_file(
    path: str | Path, *, expect_geometry: bool = False, expect_frequencies: bool = False
) -> OrcaOutput:
    return parse_orca_output(
        read_output_text(path), expect_geometry=expect_geometry, expect_frequencies=expect_frequencies
    )


__all__ = [
    "HARTREE_TO_KJ_MOL",
    "GeometryConvergence",
    "OrcaOutput",
    "QMStatus",
    "ScanPoint",
    "parse_orca_file",
    "parse_orca_output",
    "read_output_text",
]
