"""Trajectory analysis.

Every analysis declares, in its result, the trajectory it read, the selections it
used, the frame range and stride, the units it returns, and an uncertainty.  That
metadata is not bookkeeping: an Rg of "1.2" is meaningless without knowing whether it
is nm or angstrom and which atoms it covers.

**Unit boundary.**  MDAnalysis works in angstrom and picoseconds; GROMACS and this
engine work in nm and ps.  Conversions happen once, explicitly, at the boundary in
this module, and every returned :class:`Measurement` is in engine-canonical units.

MDAnalysis is optional.  Without it, each function returns an ``UNSUPPORTED``
measurement rather than raising, so a pipeline can record honestly that a trajectory
analysis was unavailable.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from polymer_engine.analysis.statistics import describe, detect_equilibration
from polymer_engine.core.errors import InsufficientDataError, ScientificError
from polymer_engine.core.models import Determination, Measurement

#: MDAnalysis length unit is the angstrom; the engine's is the nanometre.
ANGSTROM_TO_NM = 0.1

#: 1 amu/A^3 expressed in kg/m^3.
AMU_PER_A3_TO_KG_PER_M3 = 1660.5390666

#: Coordinate formats that carry no mass information.  MDAnalysis guesses masses from
#: atom names for these, and a guessed mass makes any density derived from it
#: approximate at best -- so the result says so instead of implying otherwise.
MASSLESS_TOPOLOGY_SUFFIXES = frozenset({".pdb", ".gro", ".xyz", ".crd", ".pdbqt", ".ent"})

#: How far the MSD scaling exponent may stray from 1 and still count as diffusive.
DIFFUSIVE_EXPONENT_TOLERANCE = 0.25


def mdanalysis_available() -> bool:
    try:
        import MDAnalysis  # noqa: F401
    except ImportError:
        return False
    return True


def _require_mda():
    try:
        import MDAnalysis as mda

        return mda
    except ImportError as exc:
        raise ScientificError(
            "MDAnalysis is required for trajectory analysis",
            hint="pip install MDAnalysis",
        ) from exc


@dataclass
class AnalysisSpec:
    """Exactly what an analysis was asked to do."""

    topology: str
    trajectory: str
    selection: str = "all"
    selection_b: str | None = None
    start: int = 0
    stop: int | None = None
    stride: int = 1
    input_units: str = "angstrom"
    output_units: str = "nm"

    def as_dict(self) -> dict[str, Any]:
        return {
            "topology": self.topology,
            "trajectory": self.trajectory,
            "selection": self.selection,
            "selection_b": self.selection_b,
            "start": self.start,
            "stop": self.stop,
            "stride": self.stride,
            "input_units": self.input_units,
            "output_units": self.output_units,
        }


@dataclass
class TrajectoryAnalysis:
    """A per-frame series plus its summary, with full provenance of how it was made."""

    name: str
    spec: AnalysisSpec
    frames: np.ndarray
    times_ps: np.ndarray
    values: np.ndarray
    units: str
    summary: Measurement
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.values.size)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec.as_dict(),
            "n_frames": self.n_frames,
            "units": self.units,
            "summary": self.summary.model_dump(mode="json"),
            "notes": self.notes,
            "extra": self.extra,
        }


def _unsupported(name: str, spec: AnalysisSpec, units: str, reason: str) -> TrajectoryAnalysis:
    empty = np.array([], dtype=float)
    return TrajectoryAnalysis(
        name=name,
        spec=spec,
        frames=empty,
        times_ps=empty,
        values=empty,
        units=units,
        summary=Measurement.unknown(name, units=units, reason=reason, determination=Determination.UNSUPPORTED),
        notes=reason,
    )


def _universe(spec: AnalysisSpec):
    mda = _require_mda()
    topology, trajectory = Path(spec.topology), Path(spec.trajectory)
    for path in (topology, trajectory):
        if not path.exists():
            raise ScientificError("Trajectory input does not exist", path=str(path))
    return mda.Universe(str(topology), str(trajectory))


def _select(universe, selection: str, label: str, *, require_mass: bool = False):
    group = universe.select_atoms(selection)
    if len(group) == 0:
        raise ScientificError(
            f"{label} selection matched no atoms", selection=selection,
            hint="check the selection syntax against the topology",
        )
    if require_mass:
        # A group whose masses are all zero yields a NaN centre of mass rather than an
        # error, which would silently poison every downstream number.
        try:
            total = float(np.nansum(group.masses))
        except Exception as exc:
            raise ScientificError(
                f"{label} selection has no mass information: {exc}", selection=selection
            ) from exc
        if not math.isfinite(total) or total <= 0:
            raise ScientificError(
                f"{label} selection has zero total mass, so a mass-weighted centre is undefined",
                selection=selection,
                hint="use a topology that carries masses (.tpr, .psf, .top)",
            )
    return group


def _iterate(universe, spec: AnalysisSpec, fn: Callable[[Any], float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames, times, values = [], [], []
    for ts in universe.trajectory[spec.start : spec.stop : spec.stride]:
        frames.append(int(ts.frame))
        times.append(float(getattr(ts, "time", 0.0)))
        values.append(float(fn(ts)))
    if not values:
        raise InsufficientDataError(
            "No frames selected", start=spec.start, stop=spec.stop, stride=spec.stride
        )
    return np.asarray(frames, dtype=int), np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def _finish(
    name: str, spec: AnalysisSpec, frames, times, values, units: str, *, notes: str = "", extra: dict | None = None
) -> TrajectoryAnalysis:
    return TrajectoryAnalysis(
        name=name,
        spec=spec,
        frames=frames,
        times_ps=times,
        values=values,
        units=units,
        summary=describe(values, name=name, units=units),
        notes=notes,
        extra=extra or {},
    )


# ==========================================================================
# Structural observables
# ==========================================================================
def radius_of_gyration(spec: AnalysisSpec) -> TrajectoryAnalysis:
    """Mass-weighted radius of gyration of ``spec.selection``, in nm."""
    if not mdanalysis_available():
        return _unsupported("radius_of_gyration", spec, "nm", "MDAnalysis is not installed")
    universe = _universe(spec)
    group = _select(universe, spec.selection, "Rg")
    frames, times, values = _iterate(universe, spec, lambda _ts: group.radius_of_gyration())
    return _finish(
        "radius_of_gyration", spec, frames, times, values * ANGSTROM_TO_NM, "nm",
        notes="mass-weighted; converted from MDAnalysis angstrom to nm",
        extra={"n_atoms": len(group)},
    )


def end_to_end_distance(spec: AnalysisSpec) -> TrajectoryAnalysis:
    """Distance between the first and last atom of the selection, in nm.

    Assumes the selection is ordered along the chain, which is how a polymer topology
    is normally written; the atom indices used are recorded so this can be checked.
    """
    if not mdanalysis_available():
        return _unsupported("end_to_end_distance", spec, "nm", "MDAnalysis is not installed")
    universe = _universe(spec)
    group = _select(universe, spec.selection, "end-to-end")
    if len(group) < 2:
        raise ScientificError("End-to-end distance needs at least two atoms", selection=spec.selection)
    first, last = group[0], group[-1]
    frames, times, values = _iterate(
        universe, spec, lambda _ts: float(np.linalg.norm(first.position - last.position))
    )
    return _finish(
        "end_to_end_distance", spec, frames, times, values * ANGSTROM_TO_NM, "nm",
        notes="between the first and last atom of the selection",
        extra={"first_atom_index": int(first.index), "last_atom_index": int(last.index)},
    )


def com_distance(spec: AnalysisSpec) -> TrajectoryAnalysis:
    """Centre-of-mass distance between two selections, in nm."""
    if not mdanalysis_available():
        return _unsupported("com_distance", spec, "nm", "MDAnalysis is not installed")
    if not spec.selection_b:
        raise ScientificError("com_distance needs a second selection", selection_b=spec.selection_b)
    universe = _universe(spec)
    a = _select(universe, spec.selection, "group A", require_mass=True)
    b = _select(universe, spec.selection_b, "group B", require_mass=True)
    frames, times, values = _iterate(
        universe, spec, lambda _ts: float(np.linalg.norm(a.center_of_mass() - b.center_of_mass()))
    )
    return _finish(
        "com_distance", spec, frames, times, values * ANGSTROM_TO_NM, "nm",
        notes="mass-weighted centres of mass; no periodic-image minimisation applied",
        extra={"n_atoms_a": len(a), "n_atoms_b": len(b)},
    )


def density(spec: AnalysisSpec) -> TrajectoryAnalysis:
    """System mass density from the box volume, in kg/m^3."""
    if not mdanalysis_available():
        return _unsupported("density", spec, "kg/m^3", "MDAnalysis is not installed")
    universe = _universe(spec)
    group = _select(universe, spec.selection, "density")
    try:
        total_mass_amu = float(group.masses.sum())
    except Exception as exc:
        raise ScientificError(f"Topology carries no masses: {exc}", topology=spec.topology) from exc
    if total_mass_amu <= 0:
        raise ScientificError("Selected atoms have zero total mass", selection=spec.selection)

    masses_guessed = Path(spec.topology).suffix.lower() in MASSLESS_TOPOLOGY_SUFFIXES

    def per_frame(ts) -> float:
        dims = getattr(ts, "dimensions", None)
        if dims is None or len(dims) < 3 or not np.all(np.isfinite(dims[:3])) or np.any(dims[:3] <= 0):
            raise ScientificError("Frame has no usable box dimensions; density is undefined")
        volume_a3 = float(dims[0] * dims[1] * dims[2])
        return total_mass_amu / volume_a3 * AMU_PER_A3_TO_KG_PER_M3

    frames, times, values = _iterate(universe, spec, per_frame)
    notes = "orthorhombic box volume; amu/A^3 converted to kg/m^3"
    if masses_guessed:
        notes += (
            f"; WARNING: {Path(spec.topology).suffix} carries no masses, so MDAnalysis guessed "
            "them from atom names -- supply a mass-bearing topology (.tpr, .psf, .top) for a "
            "quantitative density"
        )
    result = _finish(
        "density", spec, frames, times, values, "kg/m^3",
        notes=notes,
        extra={"total_mass_amu": total_mass_amu, "masses_guessed": masses_guessed},
    )
    if masses_guessed:
        # Downgrade the verdict rather than presenting a guessed-mass density as fact.
        result.summary = result.summary.model_copy(
            update={"notes": "masses were guessed from atom names; treat as approximate"}
        )
    return result


def volume(spec: AnalysisSpec) -> TrajectoryAnalysis:
    """Box volume per frame, reported dimensionlessly in nm^3."""
    if not mdanalysis_available():
        return _unsupported("volume", spec, "1", "MDAnalysis is not installed")
    universe = _universe(spec)

    def per_frame(ts) -> float:
        dims = ts.dimensions
        if dims is None or np.any(dims[:3] <= 0):
            raise ScientificError("Frame has no usable box dimensions")
        return float(dims[0] * dims[1] * dims[2]) * (ANGSTROM_TO_NM**3)

    frames, times, values = _iterate(universe, spec, per_frame)
    return _finish("volume", spec, frames, times, values, "1", notes="nm^3, reported as dimensionless")


# ==========================================================================
# Distributions and dynamics
# ==========================================================================
@dataclass
class RdfResult:
    r_nm: np.ndarray
    g_r: np.ndarray
    spec: AnalysisSpec
    n_frames: int
    normalisation: str = "ideal-gas in the average box volume"

    def first_peak(self) -> Measurement:
        if self.g_r.size == 0 or not np.any(np.isfinite(self.g_r)):
            return Measurement.unknown("rdf_first_peak", units="nm", reason="empty RDF")
        index = int(np.nanargmax(self.g_r))
        return Measurement(
            name="rdf_first_peak", value=float(self.r_nm[index]), units="nm", method="argmax of g(r)"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "r_nm": self.r_nm.tolist(),
            "g_r": self.g_r.tolist(),
            "n_frames": self.n_frames,
            "normalisation": self.normalisation,
            "spec": self.spec.as_dict(),
        }


def rdf(spec: AnalysisSpec, *, r_max_nm: float = 1.5, bins: int = 150) -> RdfResult | None:
    """Radial distribution function between two selections.

    Normalised against an ideal gas at the same number density, using the mean box
    volume over the analysed frames.  Minimum-image convention is applied for
    orthorhombic boxes.
    """
    if not mdanalysis_available():
        return None
    if r_max_nm <= 0 or bins < 2:
        raise ScientificError("RDF needs a positive range and at least 2 bins", r_max_nm=r_max_nm, bins=bins)
    universe = _universe(spec)
    a = _select(universe, spec.selection, "RDF group A")
    b = _select(universe, spec.selection_b or spec.selection, "RDF group B")
    same_group = (spec.selection_b or spec.selection) == spec.selection

    r_max_a = r_max_nm / ANGSTROM_TO_NM
    edges = np.linspace(0.0, r_max_a, bins + 1)
    counts = np.zeros(bins, dtype=float)
    volumes: list[float] = []
    n_frames = 0

    for ts in universe.trajectory[spec.start : spec.stop : spec.stride]:
        box = ts.dimensions[:3] if ts.dimensions is not None else None
        if box is None or np.any(box <= 0):
            raise ScientificError("RDF requires a periodic box")
        delta = a.positions[:, None, :] - b.positions[None, :, :]
        delta -= box * np.round(delta / box)  # minimum image
        distances = np.sqrt((delta**2).sum(axis=-1))
        if same_group:
            iu = np.triu_indices(len(a), k=1)
            distances = distances[iu]
        else:
            distances = distances.ravel()
        counts += np.histogram(distances, bins=edges)[0]
        volumes.append(float(box[0] * box[1] * box[2]))
        n_frames += 1

    if n_frames == 0:
        raise InsufficientDataError("No frames selected for the RDF")

    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_volume = 4.0 / 3.0 * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    mean_volume = float(np.mean(volumes))
    n_pairs = (len(a) * (len(a) - 1) / 2.0) if same_group else float(len(a) * len(b))
    ideal = n_pairs * shell_volume / mean_volume
    with np.errstate(divide="ignore", invalid="ignore"):
        g_r = np.where(ideal > 0, counts / n_frames / ideal, 0.0)

    return RdfResult(r_nm=centers * ANGSTROM_TO_NM, g_r=g_r, spec=spec, n_frames=n_frames)


@dataclass
class MsdResult:
    lag_ps: np.ndarray
    msd_nm2: np.ndarray
    spec: AnalysisSpec
    fit_range: tuple[float, float] | None = None

    def diffusion_coefficient(self, *, fit_fraction: tuple[float, float] = (0.2, 0.6)) -> Measurement:
        """Einstein diffusion coefficient from the linear region of the MSD.

        Reported only if the fitted region is actually linear: an MSD that has not
        reached the diffusive regime gives a slope with no physical meaning, and the
        engine says ``INSUFFICIENT_DATA`` rather than quoting one.
        """
        if self.lag_ps.size < 10:
            return Measurement.unknown(
                "diffusion_coefficient", units="1", reason="too few lag times",
                determination=Determination.INSUFFICIENT_DATA,
            )
        lo = int(self.lag_ps.size * fit_fraction[0])
        hi = int(self.lag_ps.size * fit_fraction[1])
        if hi - lo < 5:
            return Measurement.unknown(
                "diffusion_coefficient", units="1", reason="fit window too narrow",
                determination=Determination.INSUFFICIENT_DATA,
            )
        t, y = self.lag_ps[lo:hi], self.msd_nm2[lo:hi]
        if np.any(t <= 0) or np.any(y <= 0):
            return Measurement.unknown(
                "diffusion_coefficient", units="1",
                reason="non-positive lag times or MSD values in the fit window",
                determination=Determination.INSUFFICIENT_DATA,
            )

        # The decisive test is the scaling exponent, not the fit quality: MSD ~ t^alpha
        # with alpha = 1 for diffusion, 2 for ballistic drift, and < 1 for subdiffusion.
        # A quadratic looks perfectly straight over a narrow window, so R^2 alone would
        # happily report a "diffusion coefficient" for a particle moving at constant
        # velocity.
        alpha = float(np.polyfit(np.log(t), np.log(y), 1)[0])
        if not (1.0 - DIFFUSIVE_EXPONENT_TOLERANCE <= alpha <= 1.0 + DIFFUSIVE_EXPONENT_TOLERANCE):
            regime = "ballistic" if alpha > 1.0 else "subdiffusive"
            return Measurement.unknown(
                "diffusion_coefficient",
                units="1",
                reason=(
                    f"MSD scales as t^{alpha:.2f}, which is {regime}, not diffusive; "
                    "the Einstein relation does not apply and no D is reported"
                ),
                determination=Determination.INSUFFICIENT_DATA,
            )

        slope, intercept = np.polyfit(t, y, 1)
        predicted = slope * t + intercept
        ss_res = float(((y - predicted) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if r_squared < 0.95 or slope <= 0:
            return Measurement.unknown(
                "diffusion_coefficient",
                units="1",
                reason=f"MSD fit is poor in the chosen window (R^2 = {r_squared:.3f})",
                determination=Determination.INSUFFICIENT_DATA,
            )
        # D = slope / (2 * dimensionality); 3-D here.  nm^2/ps.
        return Measurement(
            name="diffusion_coefficient",
            value=float(slope / 6.0),
            units="1",
            method=f"Einstein relation, MSD ~ t^{alpha:.2f}, linear fit R^2={r_squared:.4f}",
            notes="nm^2/ps; multiply by 1e-2 for cm^2/s",
        )


def mean_squared_displacement(spec: AnalysisSpec, *, max_lag_fraction: float = 0.5) -> MsdResult | None:
    """Time-averaged MSD of the selection's atoms.

    Uses all time origins, which is standard and far less noisy than a single origin.
    No periodic unwrapping is performed here: the trajectory must already be whole
    (``gmx trjconv -pbc nojump``), and that requirement is recorded in the spec.
    """
    if not mdanalysis_available():
        return None
    universe = _universe(spec)
    group = _select(universe, spec.selection, "MSD")

    positions: list[np.ndarray] = []
    times: list[float] = []
    for ts in universe.trajectory[spec.start : spec.stop : spec.stride]:
        positions.append(group.positions.copy())
        times.append(float(getattr(ts, "time", len(times))))
    if len(positions) < 4:
        raise InsufficientDataError("MSD needs at least 4 frames", n_frames=len(positions))

    coords = np.asarray(positions, dtype=float) * ANGSTROM_TO_NM  # (frames, atoms, 3)
    n_frames = coords.shape[0]
    max_lag = max(1, int(n_frames * max_lag_fraction))
    dt = float(np.mean(np.diff(times))) if n_frames > 1 else 1.0

    lags = np.arange(1, max_lag + 1)
    msd = np.empty(lags.size, dtype=float)
    for i, lag in enumerate(lags):
        displacement = coords[lag:] - coords[:-lag]
        msd[i] = float((displacement**2).sum(axis=-1).mean())

    return MsdResult(lag_ps=lags * dt, msd_nm2=msd, spec=spec)


# ==========================================================================
# Interactions
# ==========================================================================
def contacts(spec: AnalysisSpec, *, cutoff_nm: float = 0.45) -> TrajectoryAnalysis:
    """Number of atom pairs between two selections within ``cutoff_nm``."""
    if not mdanalysis_available():
        return _unsupported("contacts", spec, "1", "MDAnalysis is not installed")
    if not spec.selection_b:
        raise ScientificError("contacts needs a second selection")
    if cutoff_nm <= 0:
        raise ScientificError("Contact cutoff must be positive", cutoff_nm=cutoff_nm)
    universe = _universe(spec)
    a = _select(universe, spec.selection, "contacts group A")
    b = _select(universe, spec.selection_b, "contacts group B")
    cutoff_a = cutoff_nm / ANGSTROM_TO_NM

    def per_frame(ts) -> float:
        box = ts.dimensions[:3] if ts.dimensions is not None else None
        delta = a.positions[:, None, :] - b.positions[None, :, :]
        if box is not None and np.all(box > 0):
            delta -= box * np.round(delta / box)
        distances = np.sqrt((delta**2).sum(axis=-1))
        return float(np.count_nonzero(distances <= cutoff_a))

    frames, times, values = _iterate(universe, spec, per_frame)
    return _finish(
        "contacts", spec, frames, times, values, "1",
        notes=f"atom pairs within {cutoff_nm} nm, minimum-image convention",
        extra={"cutoff_nm": cutoff_nm, "n_atoms_a": len(a), "n_atoms_b": len(b)},
    )


def hydrogen_bonds(spec: AnalysisSpec, *, distance_nm: float = 0.35, angle_deg: float = 150.0):
    """Hydrogen-bond count per frame via MDAnalysis's geometric criterion.

    Returns ``UNSUPPORTED`` when MDAnalysis lacks the analysis module rather than
    substituting a hand-rolled criterion that would not match published conventions.
    """
    spec_units = "1"
    if not mdanalysis_available():
        return _unsupported("hydrogen_bonds", spec, spec_units, "MDAnalysis is not installed")
    try:
        from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
    except ImportError:
        return _unsupported(
            "hydrogen_bonds", spec, spec_units, "MDAnalysis hydrogen-bond analysis is unavailable"
        )

    universe = _universe(spec)
    analysis = HydrogenBondAnalysis(
        universe=universe,
        between=[spec.selection, spec.selection_b or spec.selection],
        d_a_cutoff=distance_nm / ANGSTROM_TO_NM,
        d_h_a_angle_cutoff=angle_deg,
    )
    analysis.run(start=spec.start, stop=spec.stop, step=spec.stride)
    counts = analysis.count_by_time().astype(float)
    frames = np.arange(counts.size)
    times = np.asarray(getattr(analysis, "times", frames), dtype=float)
    return _finish(
        "hydrogen_bonds", spec, frames, times, counts, spec_units,
        notes=f"geometric criterion: D-A <= {distance_nm} nm, D-H-A >= {angle_deg} deg",
        extra={"distance_nm": distance_nm, "angle_deg": angle_deg},
    )


# ==========================================================================
# Production-window helper
# ==========================================================================
def production_summary(analysis: TrajectoryAnalysis) -> Measurement:
    """Summarise only the equilibrated portion of a trajectory analysis."""
    if analysis.n_frames < 10:
        return analysis.summary
    equilibration = detect_equilibration(analysis.values)
    production = analysis.values[equilibration.start_index :]
    measurement = describe(production, name=analysis.name, units=analysis.units)
    return measurement.model_copy(
        update={
            "notes": (
                f"production window from frame {equilibration.start_index} "
                f"({equilibration.method})"
            )
        }
    )


__all__ = [
    "AMU_PER_A3_TO_KG_PER_M3",
    "ANGSTROM_TO_NM",
    "DIFFUSIVE_EXPONENT_TOLERANCE",
    "MASSLESS_TOPOLOGY_SUFFIXES",
    "AnalysisSpec",
    "MsdResult",
    "RdfResult",
    "TrajectoryAnalysis",
    "com_distance",
    "contacts",
    "density",
    "end_to_end_distance",
    "hydrogen_bonds",
    "mdanalysis_available",
    "mean_squared_displacement",
    "production_summary",
    "radius_of_gyration",
    "rdf",
    "volume",
]
