"""Umbrella sampling: reaction coordinates, window planning, PLUMED inputs.

Window spacing is not a free parameter.  A harmonic restraint of stiffness ``k`` at
temperature ``T`` produces a sampled distribution of width

    sigma = sqrt(kT / k)

so adjacent windows separated by much more than ~2 sigma will not overlap, and a PMF
built from non-overlapping windows is not merely imprecise -- it is unconstrained
between the gaps.  :func:`recommend_spacing` derives the spacing from the physics, and
:func:`plan_windows` refuses a configuration that cannot overlap instead of quietly
producing windows that will waste a week of compute.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np

from polymer_engine.core.config import UmbrellaDefaults
from polymer_engine.core.errors import ParameterValidationError
from polymer_engine.core.units import kT

CVKind = Literal["distance", "com_distance", "radius_of_gyration", "torsion", "coordination"]

#: Windows this far apart in units of sigma are not expected to overlap usefully.
MAX_SPACING_SIGMA = 2.0


@dataclass(frozen=True, slots=True)
class ReactionCoordinate:
    """A collective variable with the atom selections it needs.

    ``justification`` is required text, not decoration: a PMF along a physically
    unjustified coordinate is a number without a meaning, and the campaign manifest
    carries this string so a reviewer can judge it.
    """

    name: str
    kind: CVKind
    units: str
    justification: str
    group_a: str | None = None
    group_b: str | None = None
    atoms: str | None = None

    def plumed_definition(self) -> list[str]:
        """Render the CV as PLUMED input lines."""
        if self.kind in ("distance", "com_distance"):
            if not self.group_a or not self.group_b:
                raise ParameterValidationError(
                    "A distance coordinate needs two atom groups", coordinate=self.name
                )
            if self.kind == "com_distance":
                return [
                    f"ga: COM ATOMS={self.group_a}",
                    f"gb: COM ATOMS={self.group_b}",
                    f"{self.name}: DISTANCE ATOMS=ga,gb",
                ]
            return [f"{self.name}: DISTANCE ATOMS={self.group_a},{self.group_b}"]
        if self.kind == "radius_of_gyration":
            if not self.atoms:
                raise ParameterValidationError("Rg needs an atom selection", coordinate=self.name)
            return [f"{self.name}: GYRATION TYPE=RADIUS ATOMS={self.atoms}"]
        if self.kind == "torsion":
            if not self.atoms:
                raise ParameterValidationError("A torsion needs four atoms", coordinate=self.name)
            if len(self.atoms.split(",")) != 4:
                raise ParameterValidationError(
                    "A torsion needs exactly four atoms", coordinate=self.name, atoms=self.atoms
                )
            return [f"{self.name}: TORSION ATOMS={self.atoms}"]
        if self.kind == "coordination":
            if not self.group_a or not self.group_b:
                raise ParameterValidationError(
                    "A coordination number needs two groups", coordinate=self.name
                )
            return [
                f"{self.name}: COORDINATION GROUPA={self.group_a} GROUPB={self.group_b} R_0=0.5"
            ]
        raise ParameterValidationError("Unsupported coordinate kind", kind=self.kind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "units": self.units,
            "justification": self.justification,
            "group_a": self.group_a,
            "group_b": self.group_b,
            "atoms": self.atoms,
        }


@dataclass(frozen=True, slots=True)
class Window:
    index: int
    center: float
    force_constant: float
    units: str

    @property
    def sigma(self) -> float | None:
        return None  # set by the plan, which knows the temperature

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "center": self.center,
            "force_constant": self.force_constant,
            "units": self.units,
        }


@dataclass
class WindowPlan:
    coordinate: ReactionCoordinate
    windows: list[Window]
    temperature_k: float
    spacing: float
    sigma: float
    recommended_spacing: float
    window_ns: float
    equilibration_fraction: float
    warnings: list[str] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def spacing_in_sigma(self) -> float:
        return self.spacing / self.sigma if self.sigma > 0 else float("inf")

    @property
    def expected_to_overlap(self) -> bool:
        return self.spacing_in_sigma <= MAX_SPACING_SIGMA

    @property
    def centers(self) -> list[float]:
        return [w.center for w in self.windows]

    def as_dict(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate.as_dict(),
            "temperature_k": self.temperature_k,
            "n_windows": self.n_windows,
            "spacing": self.spacing,
            "sigma": self.sigma,
            "spacing_in_sigma": self.spacing_in_sigma,
            "recommended_spacing": self.recommended_spacing,
            "expected_to_overlap": self.expected_to_overlap,
            "window_ns": self.window_ns,
            "equilibration_fraction": self.equilibration_fraction,
            "warnings": self.warnings,
            "windows": [w.as_dict() for w in self.windows],
        }


def restraint_sigma(force_constant: float, temperature_k: float) -> float:
    """Width of the sampled distribution under a harmonic restraint: ``sqrt(kT/k)``."""
    if force_constant <= 0:
        raise ParameterValidationError("Force constant must be positive", force_constant=force_constant)
    return math.sqrt(kT(temperature_k) / force_constant)


def recommend_spacing(force_constant: float, temperature_k: float, *, sigma_factor: float = 1.0) -> float:
    """Spacing that gives good adjacent-window overlap for this stiffness."""
    return sigma_factor * restraint_sigma(force_constant, temperature_k)


def recommend_force_constant(spacing: float, temperature_k: float, *, sigma_factor: float = 1.0) -> float:
    """Inverse of :func:`recommend_spacing`: stiffness that suits a fixed spacing."""
    if spacing <= 0:
        raise ParameterValidationError("Spacing must be positive", spacing=spacing)
    sigma = spacing / sigma_factor
    return kT(temperature_k) / (sigma * sigma)


def plan_windows(
    coordinate: ReactionCoordinate,
    *,
    minimum: float,
    maximum: float,
    temperature_k: float,
    defaults: UmbrellaDefaults | None = None,
    spacing: float | None = None,
    force_constant: float | None = None,
    strict: bool = False,
) -> WindowPlan:
    """Lay out umbrella windows across ``[minimum, maximum]``.

    With ``strict=True`` a spacing that cannot overlap raises.  Otherwise it is
    recorded as a warning on the plan and surfaced by the overlap gate later.
    """
    defaults = defaults or UmbrellaDefaults()
    if maximum <= minimum:
        raise ParameterValidationError(
            "Umbrella range must be increasing", minimum=minimum, maximum=maximum
        )
    spacing = spacing if spacing is not None else defaults.spacing_nm
    force_constant = (
        force_constant if force_constant is not None else defaults.force_constant_kj_mol_nm2
    )
    if spacing <= 0:
        raise ParameterValidationError("Window spacing must be positive", spacing=spacing)

    sigma = restraint_sigma(force_constant, temperature_k)
    recommended = recommend_spacing(force_constant, temperature_k)

    n_intervals = math.floor((maximum - minimum) / spacing + 1e-9)
    centers = [round(minimum + i * spacing, 9) for i in range(n_intervals + 1)]
    if centers[-1] < maximum - 1e-9:
        centers.append(round(maximum, 9))  # always cover the endpoint

    warnings: list[str] = []
    if len(centers) < defaults.min_windows:
        warnings.append(
            f"{len(centers)} windows is below the configured minimum of {defaults.min_windows}"
        )
    if spacing > MAX_SPACING_SIGMA * sigma * (1.0 + 1e-9):
        message = (
            f"Spacing {spacing:.4g} {coordinate.units} is {spacing / sigma:.1f} sigma for "
            f"k={force_constant:g}; windows are not expected to overlap. "
            f"Use spacing <= {MAX_SPACING_SIGMA * sigma:.4g} or k >= "
            f"{recommend_force_constant(spacing, temperature_k, sigma_factor=MAX_SPACING_SIGMA):.4g}."
        )
        warnings.append(message)
        if strict:
            raise ParameterValidationError(message, spacing=spacing, sigma=sigma)

    windows = [
        Window(index=i, center=c, force_constant=force_constant, units=coordinate.units)
        for i, c in enumerate(centers)
    ]
    return WindowPlan(
        coordinate=coordinate,
        windows=windows,
        temperature_k=temperature_k,
        spacing=spacing,
        sigma=sigma,
        recommended_spacing=recommended,
        window_ns=defaults.window_ns,
        equilibration_fraction=defaults.equilibration_fraction,
        warnings=warnings,
    )


def insert_windows(plan: WindowPlan, gaps: Sequence[tuple[float, float]]) -> WindowPlan:
    """Add a window at the midpoint of each under-overlapped gap.

    This is the adaptive refinement step: the overlap diagnostic reports which
    adjacent pairs failed, and each gets a new window between them.
    """
    centers = set(plan.centers)
    for lower, upper in gaps:
        midpoint = round((lower + upper) / 2.0, 9)
        centers.add(midpoint)
    ordered = sorted(centers)
    windows = [
        Window(index=i, center=c, force_constant=plan.windows[0].force_constant, units=plan.coordinate.units)
        for i, c in enumerate(ordered)
    ]
    spacing = min(b - a for a, b in pairwise(ordered)) if len(ordered) > 1 else plan.spacing
    return WindowPlan(
        coordinate=plan.coordinate,
        windows=windows,
        temperature_k=plan.temperature_k,
        spacing=spacing,
        sigma=plan.sigma,
        recommended_spacing=plan.recommended_spacing,
        window_ns=plan.window_ns,
        equilibration_fraction=plan.equilibration_fraction,
        warnings=[*plan.warnings, f"inserted {len(ordered) - plan.n_windows} adaptive window(s)"],
    )


def plumed_input(
    coordinate: ReactionCoordinate,
    window: Window,
    *,
    stride: int = 500,
    colvar_file: str = "COLVAR",
) -> str:
    """Render the PLUMED input for one window."""
    lines = [
        "# Umbrella window generated by polymer-engine",
        f"# coordinate: {coordinate.name} ({coordinate.kind}), window {window.index}",
        "UNITS LENGTH=nm ENERGY=kj/mol TIME=ps",
        "",
        *coordinate.plumed_definition(),
        "",
        (
            f"restraint: RESTRAINT ARG={coordinate.name} "
            f"AT={window.center:.6f} KAPPA={window.force_constant:.6f}"
        ),
        "",
        f"PRINT ARG={coordinate.name},restraint.bias FILE={colvar_file} STRIDE={stride}",
        "",
    ]
    return "\n".join(lines)


def write_windows(
    plan: WindowPlan,
    root: str | Path,
    *,
    stride: int = 500,
) -> list[Path]:
    """Write one directory per window, each with its PLUMED input and metadata."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for window in plan.windows:
        directory = root / f"window_{window.index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "plumed.dat"
        path.write_text(plumed_input(plan.coordinate, window, stride=stride), encoding="utf-8")
        (directory / "window.json").write_text(
            json.dumps(
                {
                    **window.as_dict(),
                    "sigma": plan.sigma,
                    "temperature_k": plan.temperature_k,
                    "window_ns": plan.window_ns,
                    "equilibration_fraction": plan.equilibration_fraction,
                    "coordinate": plan.coordinate.as_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written.append(path)
    (root / "umbrella_plan.json").write_text(json.dumps(plan.as_dict(), indent=2), encoding="utf-8")
    return written


def assign_starting_structures(
    plan: WindowPlan, available: dict[float, str]
) -> dict[int, str | None]:
    """Pick the closest available structure for each window centre.

    Returns ``None`` for a window with no structure within one sigma -- starting a
    window far from its restraint centre wastes the equilibration period and can
    trap the system in the wrong basin.
    """
    assignments: dict[int, str | None] = {}
    if not available:
        return {w.index: None for w in plan.windows}
    positions = np.array(sorted(available), dtype=float)
    for window in plan.windows:
        nearest = float(positions[np.argmin(np.abs(positions - window.center))])
        assignments[window.index] = available[nearest] if abs(nearest - window.center) <= plan.sigma else None
    return assignments


__all__ = [
    "MAX_SPACING_SIGMA",
    "ReactionCoordinate",
    "Window",
    "WindowPlan",
    "assign_starting_structures",
    "insert_windows",
    "plan_windows",
    "plumed_input",
    "recommend_force_constant",
    "recommend_spacing",
    "restraint_sigma",
    "write_windows",
]
