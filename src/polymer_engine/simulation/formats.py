"""Parsers for the GROMACS file formats the engine must validate.

These are deliberately strict readers, not tolerant ones.  Their job is to catch a
malformed system *before* it becomes a simulation, so a line that does not conform
raises rather than being skipped.

Supported:

* ``.gro`` -- fixed-column coordinate format, including the box vectors
* ``.top`` / ``.itp`` -- enough of the topology grammar to resolve ``#include``
  directives, count atoms per ``[ moleculetype ]``, and expand ``[ molecules ]``
* ``.xvg`` -- the analysis output format, with legend metadata
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from polymer_engine.core.errors import SystemValidationError


# --------------------------------------------------------------------------
# GRO coordinates
# --------------------------------------------------------------------------
@dataclass(slots=True)
class GroFile:
    """A parsed ``.gro`` file.

    Positions are in nm (the GROMACS convention); ``box`` holds the three box
    vector lengths, also nm.
    """

    title: str
    n_atoms: int
    residue_numbers: list[int]
    residue_names: list[str]
    atom_names: list[str]
    atom_numbers: list[int]
    positions: np.ndarray
    velocities: np.ndarray | None
    box: tuple[float, ...]
    path: str | None = None

    @property
    def has_box(self) -> bool:
        return len(self.box) >= 3 and all(math.isfinite(v) for v in self.box[:3])

    @property
    def box_volume_nm3(self) -> float | None:
        if not self.has_box:
            return None
        if len(self.box) == 3:
            return float(self.box[0] * self.box[1] * self.box[2])
        # Triclinic: v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
        return float(self.box[0] * self.box[1] * self.box[2])

    @property
    def positions_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.positions)))

    def extent_nm(self) -> tuple[float, float, float]:
        if self.n_atoms == 0:
            return (0.0, 0.0, 0.0)
        span = self.positions.max(axis=0) - self.positions.min(axis=0)
        return (float(span[0]), float(span[1]), float(span[2]))


def _parse_float(text: str, *, context: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise SystemValidationError(f"Non-numeric value {text.strip()!r} in {context}") from None


def read_gro(path: str | Path) -> GroFile:
    """Parse a ``.gro`` file using its fixed-column layout.

    Whitespace splitting is not safe here: atom names and residue names routinely run
    together with adjacent columns, and coordinates can lose their separator for
    large boxes.
    """
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise SystemValidationError(f"Could not read coordinate file: {exc}", path=str(path)) from exc

    if len(lines) < 3:
        raise SystemValidationError("GRO file has fewer than 3 lines", path=str(path), lines=len(lines))

    title = lines[0].strip()
    try:
        n_atoms = int(lines[1].strip())
    except ValueError:
        raise SystemValidationError(
            "GRO atom-count line is not an integer", path=str(path), line=lines[1][:80]
        ) from None
    if n_atoms < 0:
        raise SystemValidationError("GRO atom count is negative", path=str(path), n_atoms=n_atoms)

    # Count only the coordinate lines: the trailing box line is validated separately
    # so that "no box" reports as a box problem rather than as a truncated file.
    available = len(lines) - 2
    if available < n_atoms:
        raise SystemValidationError(
            "GRO file is truncated: fewer coordinate lines than the declared atom count",
            path=str(path),
            declared=n_atoms,
            found=available,
        )

    residue_numbers: list[int] = []
    residue_names: list[str] = []
    atom_names: list[str] = []
    atom_numbers: list[int] = []
    positions = np.empty((n_atoms, 3), dtype=float)
    velocities: list[list[float]] = []
    has_velocities = True

    for index in range(n_atoms):
        line = lines[2 + index]
        if len(line) < 44:
            raise SystemValidationError(
                "GRO coordinate line is too short for the fixed-column format",
                path=str(path),
                line_number=3 + index,
                length=len(line),
            )
        context = f"{path} line {3 + index}"
        try:
            residue_numbers.append(int(line[0:5]))
            atom_numbers.append(int(line[15:20]))
        except ValueError:
            raise SystemValidationError(
                "GRO residue or atom index is not an integer", path=str(path), line_number=3 + index
            ) from None
        residue_names.append(line[5:10].strip())
        atom_names.append(line[10:15].strip())
        positions[index, 0] = _parse_float(line[20:28], context=context)
        positions[index, 1] = _parse_float(line[28:36], context=context)
        positions[index, 2] = _parse_float(line[36:44], context=context)
        if has_velocities and len(line) >= 68:
            velocities.append(
                [
                    _parse_float(line[44:52], context=context),
                    _parse_float(line[52:60], context=context),
                    _parse_float(line[60:68], context=context),
                ]
            )
        else:
            has_velocities = False

    box_line = lines[2 + n_atoms] if len(lines) > 2 + n_atoms else ""
    box_values = tuple(_parse_float(v, context=f"{path} box line") for v in box_line.split())
    if box_values and len(box_values) not in (3, 9):
        raise SystemValidationError(
            "GRO box line must hold 3 or 9 values", path=str(path), found=len(box_values)
        )

    return GroFile(
        title=title,
        n_atoms=n_atoms,
        residue_numbers=residue_numbers,
        residue_names=residue_names,
        atom_names=atom_names,
        atom_numbers=atom_numbers,
        positions=positions,
        velocities=np.asarray(velocities, dtype=float) if has_velocities and velocities else None,
        box=box_values,
        path=str(path),
    )


def count_gro_atoms(path: str | Path) -> int:
    """Cheap atom count that does not parse every coordinate line."""
    path = Path(path)
    with path.open(encoding="utf-8", errors="replace") as fh:
        fh.readline()
        second = fh.readline()
    try:
        return int(second.strip())
    except ValueError:
        raise SystemValidationError("GRO atom-count line is not an integer", path=str(path)) from None


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------
_INCLUDE = re.compile(r'^\s*#include\s+([<"])([^>"]+)[>"]')
_SECTION = re.compile(r"^\s*\[\s*([A-Za-z0-9_ ]+?)\s*\]")
_DIRECTIVE = re.compile(r"^\s*#(ifdef|ifndef|else|endif|define|undef)\b(.*)$")


@dataclass(slots=True)
class MoleculeType:
    name: str
    n_atoms: int
    source: str


@dataclass(slots=True)
class Topology:
    """A resolved GROMACS topology."""

    path: str
    molecule_types: dict[str, MoleculeType] = field(default_factory=dict)
    molecules: list[tuple[str, int]] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)
    resolved_includes: list[str] = field(default_factory=list)
    missing_includes: list[str] = field(default_factory=list)
    system_name: str | None = None
    has_molecules_section: bool = False
    has_defaults_section: bool = False
    defines: set[str] = field(default_factory=set)

    @property
    def total_atoms(self) -> int | None:
        """Atoms implied by ``[ molecules ]``, or ``None`` if a type is unresolved.

        ``None`` matters: it means we cannot check the topology against the
        coordinates, which is a different situation from a mismatch.
        """
        if not self.molecules:
            return None
        total = 0
        for name, count in self.molecules:
            moltype = self.molecule_types.get(name)
            if moltype is None:
                return None
            total += moltype.n_atoms * count
        return total

    @property
    def unresolved_molecule_types(self) -> list[str]:
        return [name for name, _ in self.molecules if name not in self.molecule_types]


def read_topology(
    path: str | Path,
    *,
    include_dirs: list[Path] | None = None,
    _seen: set[Path] | None = None,
    _root: Topology | None = None,
    defines: set[str] | None = None,
) -> Topology:
    """Parse a ``.top``/``.itp``, following ``#include`` directives.

    ``#include <name>`` refers to the GROMACS library (``GMXLIB``/``GMXDATA``), which
    we may not have locally.  Those are recorded but *not* treated as missing files,
    because a force-field include resolving at run time is normal and flagging it as
    an error would block every valid CHARMM-GUI system.
    """
    path = Path(path)
    root_topology = _root or Topology(path=str(path))
    seen = _seen if _seen is not None else set()
    resolved = path.resolve()
    if resolved in seen:
        return root_topology
    seen.add(resolved)

    if not path.is_file():
        raise SystemValidationError("Topology file not found", path=str(path))

    include_dirs = include_dirs or []
    active_defines = defines if defines is not None else set()
    section: str | None = None
    pending_moleculetype: str | None = None
    atom_count = 0
    # Simple #ifdef handling: a stack of "are we currently emitting" flags.
    emit_stack: list[bool] = []

    def emitting() -> bool:
        return all(emit_stack)

    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].rstrip()
        if not line.strip():
            continue

        directive = _DIRECTIVE.match(line)
        if directive:
            kind, rest = directive.group(1), directive.group(2).strip()
            if kind == "ifdef":
                emit_stack.append(rest.split()[0] in active_defines if rest else False)
            elif kind == "ifndef":
                emit_stack.append(rest.split()[0] not in active_defines if rest else True)
            elif kind == "else":
                if emit_stack:
                    emit_stack[-1] = not emit_stack[-1]
            elif kind == "endif":
                if emit_stack:
                    emit_stack.pop()
            elif kind == "define" and emitting() and rest:
                token = rest.split()[0]
                active_defines.add(token)
                root_topology.defines.add(token)
            elif kind == "undef" and emitting() and rest:
                active_defines.discard(rest.split()[0])
            continue

        if not emitting():
            continue

        include = _INCLUDE.match(line)
        if include:
            bracket, target = include.group(1), include.group(2)
            root_topology.includes.append(target)
            if bracket == "<":
                # GROMACS library include; resolved by gmx at run time.
                continue
            candidates = [path.parent / target, *(d / target for d in include_dirs)]
            found = next((c for c in candidates if c.is_file()), None)
            if found is None:
                root_topology.missing_includes.append(target)
                continue
            root_topology.resolved_includes.append(str(found))
            # Flush any moleculetype in progress before descending.
            if pending_moleculetype is not None:
                _record_moleculetype(root_topology, pending_moleculetype, atom_count, str(path))
                pending_moleculetype, atom_count = None, 0
            read_topology(
                found,
                include_dirs=include_dirs,
                _seen=seen,
                _root=root_topology,
                defines=active_defines,
            )
            section = None
            continue

        header = _SECTION.match(line)
        if header:
            # A [ moleculetype ] block spans the sections that follow it ([ atoms ],
            # [ bonds ], ...), so the pending type is flushed when the *next*
            # moleculetype starts or at end of file -- not on every section header.
            section = header.group(1).strip().lower().replace(" ", "")
            if section == "molecules":
                root_topology.has_molecules_section = True
            elif section == "defaults":
                root_topology.has_defaults_section = True
            continue

        fields = line.split()
        if section == "moleculetype":
            if pending_moleculetype is not None:
                _record_moleculetype(root_topology, pending_moleculetype, atom_count, str(path))
            pending_moleculetype = fields[0]
            atom_count = 0
        elif section == "atoms" and pending_moleculetype is not None:
            atom_count += 1
        elif section == "molecules":
            if len(fields) >= 2:
                try:
                    root_topology.molecules.append((fields[0], int(fields[1])))
                except ValueError:
                    raise SystemValidationError(
                        "Molecule count in [ molecules ] is not an integer",
                        path=str(path),
                        line=line.strip(),
                    ) from None
        elif section == "system":
            root_topology.system_name = line.strip()

    if pending_moleculetype is not None:
        _record_moleculetype(root_topology, pending_moleculetype, atom_count, str(path))
    return root_topology


def _record_moleculetype(topology: Topology, name: str, n_atoms: int, source: str) -> None:
    if n_atoms <= 0:
        return
    existing = topology.molecule_types.get(name)
    if existing is None or existing.n_atoms == 0:
        topology.molecule_types[name] = MoleculeType(name=name, n_atoms=n_atoms, source=source)


# --------------------------------------------------------------------------
# XVG
# --------------------------------------------------------------------------
@dataclass(slots=True)
class XvgData:
    x: np.ndarray
    y: np.ndarray
    columns: np.ndarray
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    legends: list[str] = field(default_factory=list)
    path: str | None = None

    @property
    def n_points(self) -> int:
        return int(self.x.size)

    def column(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.columns.shape[1]:
            raise SystemValidationError(
                "XVG column out of range", requested=index, available=self.columns.shape[1]
            )
        return self.columns[:, index]


_XVG_LABEL = re.compile(r'@\s+(\S+)\s+label\s+"([^"]*)"')
_XVG_TITLE = re.compile(r'@\s+title\s+"([^"]*)"')
_XVG_LEGEND = re.compile(r'@\s+s(\d+)\s+legend\s+"([^"]*)"')


def read_xvg(path: str | Path) -> XvgData:
    """Parse a GROMACS ``.xvg``, keeping its legend metadata.

    The legends carry the unit strings GROMACS used, which is how downstream analysis
    avoids assuming what "Density" is measured in.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SystemValidationError(f"Could not read XVG file: {exc}", path=str(path)) from exc

    title = x_label = y_label = ""
    legends: dict[int, str] = {}
    rows: list[list[float]] = []
    width: int | None = None

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("@"):
            if match := _XVG_TITLE.match(line):
                title = match.group(1)
            elif match := _XVG_LABEL.match(line):
                if match.group(1) == "xaxis":
                    x_label = match.group(2)
                elif match.group(1) == "yaxis":
                    y_label = match.group(2)
            elif match := _XVG_LEGEND.match(line):
                legends[int(match.group(1))] = match.group(2)
            continue
        if line.startswith("&"):
            continue
        parts = line.split()
        try:
            values = [float(p) for p in parts]
        except ValueError:
            raise SystemValidationError(
                "Non-numeric data row in XVG file", path=str(path), line_number=line_no, line=line[:100]
            ) from None
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise SystemValidationError(
                "Ragged XVG data: rows have differing column counts",
                path=str(path),
                line_number=line_no,
                expected=width,
                found=len(values),
            )
        rows.append(values)

    if not rows:
        raise SystemValidationError("XVG file contains no numeric data", path=str(path))

    data = np.asarray(rows, dtype=float)
    return XvgData(
        x=data[:, 0],
        y=data[:, 1] if data.shape[1] > 1 else data[:, 0],
        columns=data[:, 1:] if data.shape[1] > 1 else data[:, :1],
        title=title,
        x_label=x_label,
        y_label=y_label,
        legends=[legends[k] for k in sorted(legends)],
        path=str(path),
    )


__all__ = [
    "GroFile",
    "MoleculeType",
    "Topology",
    "XvgData",
    "count_gro_atoms",
    "read_gro",
    "read_topology",
    "read_xvg",
]
