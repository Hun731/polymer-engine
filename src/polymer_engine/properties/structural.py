"""Structural and interchain properties.

Two of these deserve a note before use.

**Persistence length.** The standard estimator fits an exponential decay to the
bond-vector correlation along the chain, ``<u_0 . u_n> = exp(-n l_b / l_p)``.  That fit
is only meaningful while the correlation is still measurable: once it has decayed into
the noise, the tail contributes nothing but scatter, and fitting it produces a
confident-looking number driven by noise.  The estimator therefore truncates at the
first non-positive correlation and refuses to report a value from too few usable points.

**Free volume.** What is computed here is a *geometric* free-volume fraction from a
probe-sphere insertion test.  It is not positron-annihilation free volume and it is not
directly comparable to one; the probe radius changes the answer and is recorded with it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from polymer_engine.core.config import AnalysisDefaults
from polymer_engine.core.models import Determination, GateReport, GateResult, GateStatus, Measurement
from polymer_engine.properties.base import (
    PropertyCalculator,
    PropertyClass,
    PropertyDefinition,
    PropertyResult,
    SamplingRequirement,
    UncertaintyMethod,
)
from polymer_engine.properties.thermodynamic import TimeSeriesProperty


class RadiusOfGyration(TimeSeriesProperty):
    definition = PropertyDefinition(
        name="radius_of_gyration",
        property_class=PropertyClass.STRUCTURAL,
        units="nm",
        observable="Mass-weighted radius of gyration of the selected chain, per frame",
        estimator="Equilibrium time average over the production window",
        uncertainty_method=UncertaintyMethod.CORRELATION_AWARE_SE,
        sampling=SamplingRequirement(
            min_effective_samples=20.0,
            min_replicas=3,
            min_simulation_ns=20.0,
            notes="Chain conformations decorrelate far more slowly than local observables; "
                  "the Rouse time of even a short chain is nanoseconds.",
        ),
        description="Chain size measure.",
        caveats="A single chain gives one conformational sample at a time; several chains "
                "or several replicas are needed for an ensemble average.",
    )


class EndToEndDistance(TimeSeriesProperty):
    definition = PropertyDefinition(
        name="end_to_end_distance",
        property_class=PropertyClass.STRUCTURAL,
        units="nm",
        observable="Distance between the first and last backbone atom, per frame",
        estimator="Equilibrium time average over the production window",
        uncertainty_method=UncertaintyMethod.CORRELATION_AWARE_SE,
        sampling=SamplingRequirement(
            min_effective_samples=20.0, min_replicas=3, min_simulation_ns=20.0,
            notes="Decorrelates on the chain relaxation time, not the frame interval.",
        ),
        description="End-to-end vector magnitude.",
        caveats="Requires an unwrapped trajectory; a chain crossing a periodic boundary "
                "otherwise produces a spuriously large value.",
    )


class ContactNumber(TimeSeriesProperty):
    definition = PropertyDefinition(
        name="contact_number",
        property_class=PropertyClass.INTERMOLECULAR,
        units="1",
        observable="Number of atom pairs between two selections within a cutoff, per frame",
        estimator="Equilibrium time average over the production window",
        uncertainty_method=UncertaintyMethod.CORRELATION_AWARE_SE,
        sampling=SamplingRequirement(min_effective_samples=20.0, min_replicas=3),
        description="Interchain contact count, a proxy for cohesion.",
        caveats="The value depends on the cutoff, which must be reported with it. "
                "A contact count is not an interaction energy.",
    )


class HydrogenBondCount(TimeSeriesProperty):
    definition = PropertyDefinition(
        name="hydrogen_bond_count",
        property_class=PropertyClass.INTERMOLECULAR,
        units="1",
        observable="Hydrogen bonds satisfying a geometric criterion, per frame",
        estimator="Equilibrium time average over the production window",
        uncertainty_method=UncertaintyMethod.CORRELATION_AWARE_SE,
        sampling=SamplingRequirement(min_effective_samples=20.0, min_replicas=3),
        description="Hydrogen-bond population.",
        caveats="Geometric criteria are conventions; the distance and angle cutoffs must "
                "be reported, and counts from different criteria are not comparable.",
    )


class PersistenceLength(PropertyCalculator):
    """Persistence length from bond-vector correlation decay along the chain."""

    definition = PropertyDefinition(
        name="persistence_length",
        property_class=PropertyClass.STRUCTURAL,
        units="nm",
        observable="Decay of <u_0 . u_n> with separation along the backbone",
        estimator="Exponential fit to the correlation decay, truncated where it enters the noise",
        uncertainty_method=UncertaintyMethod.FIT_COVARIANCE,
        sampling=SamplingRequirement(
            min_effective_samples=20.0, min_replicas=3, min_simulation_ns=20.0,
            notes="Needs conformational sampling, not just frames.",
        ),
        description="Backbone stiffness length scale.",
        caveats="The exponential model assumes a worm-like chain; it does not describe "
                "a chain with helical or otherwise oscillating correlations.",
    )

    #: Points needed after truncation for the fit to mean anything.
    MIN_FIT_POINTS = 4

    def compute(
        self,
        correlations: Sequence[float] | np.ndarray,
        bond_length_nm: float,
        *,
        n_replicas: int = 1,
        provenance: dict[str, Any] | None = None,
    ) -> PropertyResult:
        rho = np.asarray(correlations, dtype=float).ravel()
        if rho.size < self.MIN_FIT_POINTS + 1:
            return self.unknown(
                f"only {rho.size} correlation points; the fit needs at least {self.MIN_FIT_POINTS + 1}"
            )
        if bond_length_nm <= 0:
            return self.unknown("bond length must be positive", determination=Determination.UNKNOWN)
        if not np.all(np.isfinite(rho)):
            return self.unknown("correlation series contains non-finite values",
                                determination=Determination.UNKNOWN)

        # Truncate at the first non-positive correlation: beyond it the signal has
        # decayed into the noise and log(rho) is undefined or meaningless.
        positive = np.flatnonzero(rho <= 0.0)
        cutoff = int(positive[0]) if positive.size else rho.size
        usable = rho[:cutoff]
        if usable.size < self.MIN_FIT_POINTS:
            return self.unknown(
                f"correlation decays into the noise after {usable.size} points; "
                f"{self.MIN_FIT_POINTS} are needed for a fit"
            )

        separations = np.arange(usable.size, dtype=float)
        log_rho = np.log(usable)
        # rho(n) = exp(-n * l_b / l_p)  =>  log rho = -(l_b / l_p) * n
        coefficients, *_ = np.linalg.lstsq(separations.reshape(-1, 1), log_rho, rcond=None)
        decay = float(-coefficients[0])
        report = GateReport(name=f"property:{self.definition.name}")

        if decay <= 0:
            report.gates.append(
                GateResult(
                    gate=f"{self.definition.name}:decay_positive",
                    status=GateStatus.FAIL,
                    message="the fitted correlation does not decay; the worm-like-chain model does not apply",
                    value=decay,
                )
            )
            return PropertyResult(
                definition=self.definition,
                measurement=Measurement.unknown(
                    self.definition.name, units="nm",
                    reason="correlation does not decay exponentially",
                    determination=Determination.INSUFFICIENT_DATA,
                ),
                report=report,
                n_replicas=n_replicas,
            )

        persistence = bond_length_nm / decay
        predicted = -decay * separations
        ss_res = float(((log_rho - predicted) ** 2).sum())
        ss_tot = float(((log_rho - log_rho.mean()) ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        dof = max(1, usable.size - 1)
        variance = ss_res / dof / max(float((separations**2).sum()), 1e-12)
        decay_error = math.sqrt(max(variance, 0.0))
        persistence_error = (
            bond_length_nm * decay_error / decay**2 if decay > 0 and decay_error > 0 else None
        )

        report.gates.append(
            GateResult(
                gate=f"{self.definition.name}:fit_quality",
                status=GateStatus.PASS if r_squared >= 0.9 else GateStatus.WARN,
                message=f"exponential fit R^2 = {r_squared:.3f} over {usable.size} points",
                value=r_squared,
                threshold=0.9,
                units="1",
            )
        )
        report.gates.append(
            GateResult(
                gate=f"{self.definition.name}:usable_points",
                status=GateStatus.PASS if usable.size >= self.MIN_FIT_POINTS else GateStatus.FAIL,
                message=f"{usable.size} correlation points above the noise floor",
                value=float(usable.size),
                threshold=float(self.MIN_FIT_POINTS),
                units="1",
            )
        )
        report.gates.extend(
            self.sampling_gates(
                Measurement(
                    name=self.definition.name, value=persistence, units="nm",
                    n_samples=usable.size, effective_samples=float(usable.size),
                ),
                n_replicas=n_replicas,
                equilibration_shown=True,
            )
        )

        return PropertyResult(
            definition=self.definition,
            measurement=Measurement(
                name=self.definition.name,
                value=persistence,
                uncertainty=persistence_error,
                units="nm",
                n_samples=int(usable.size),
                effective_samples=float(usable.size),
                method="exponential fit to bond-vector correlation decay",
            ),
            report=report,
            n_replicas=n_replicas,
            provenance={
                "bond_length_nm": bond_length_nm,
                "decay_per_bond": decay,
                "r_squared": r_squared,
                "points_used": int(usable.size),
                "points_supplied": int(rho.size),
                **(provenance or {}),
            },
        )


class FreeVolume(PropertyCalculator):
    """Geometric free-volume fraction by probe-sphere insertion.

    A grid of points is tested against the van der Waals spheres of every atom; the
    fraction of points that could accommodate a probe of the given radius is the free
    volume.  The probe radius is part of the answer, not a detail: a larger probe finds
    less free volume, and two values computed with different probes are not comparable.
    """

    definition = PropertyDefinition(
        name="free_volume_fraction",
        property_class=PropertyClass.STRUCTURAL,
        units="1",
        observable="Fraction of the box that admits a probe sphere of a stated radius",
        estimator="Grid-based probe insertion against atomic van der Waals radii",
        uncertainty_method=UncertaintyMethod.BLOCK_BOOTSTRAP,
        sampling=SamplingRequirement(min_effective_samples=20.0, min_replicas=3),
        description="Geometric free-volume fraction.",
        caveats="This is a geometric construct, not positron-annihilation free volume. "
                "It depends on the probe radius and on the van der Waals radii used, both "
                "of which are recorded with the result.",
    )

    def compute(
        self,
        positions_nm: np.ndarray,
        radii_nm: np.ndarray,
        box_nm: tuple[float, float, float],
        *,
        probe_radius_nm: float = 0.11,
        grid_spacing_nm: float = 0.05,
        n_replicas: int = 1,
        provenance: dict[str, Any] | None = None,
    ) -> PropertyResult:
        positions = np.asarray(positions_nm, dtype=float)
        radii = np.asarray(radii_nm, dtype=float).ravel()
        if positions.ndim != 2 or positions.shape[1] != 3:
            return self.unknown("positions must be an (N, 3) array", determination=Determination.UNKNOWN)
        if radii.size != positions.shape[0]:
            return self.unknown("one radius per atom is required", determination=Determination.UNKNOWN)
        if any(v <= 0 for v in box_nm):
            return self.unknown("box dimensions must be positive", determination=Determination.UNKNOWN)
        if probe_radius_nm < 0 or grid_spacing_nm <= 0:
            return self.unknown("probe radius and grid spacing must be positive",
                                determination=Determination.UNKNOWN)

        axes = [np.arange(0.0, length, grid_spacing_nm) for length in box_nm]
        n_points = int(np.prod([len(a) for a in axes]))
        if n_points == 0:
            return self.unknown("grid spacing is larger than the box")
        if n_points > 20_000_000:
            return self.unknown(
                f"grid of {n_points} points is too large; increase grid_spacing_nm",
                determination=Determination.UNKNOWN,
            )

        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        box = np.asarray(box_nm, dtype=float)
        occupied = np.zeros(grid.shape[0], dtype=bool)
        effective_radii = radii + probe_radius_nm

        # Chunked to bound peak memory on a large system.
        chunk = max(1, int(5_000_000 / max(grid.shape[0], 1)))
        for start in range(0, positions.shape[0], chunk):
            block = positions[start : start + chunk]
            block_radii = effective_radii[start : start + chunk]
            delta = grid[:, None, :] - block[None, :, :]
            delta -= box * np.round(delta / box)  # minimum image
            distance_sq = (delta**2).sum(axis=-1)
            occupied |= np.any(distance_sq < block_radii[None, :] ** 2, axis=1)

        free_fraction = float(1.0 - occupied.mean())
        # Binomial standard error on the grid sampling itself.
        grid_error = math.sqrt(max(free_fraction * (1 - free_fraction) / grid.shape[0], 0.0))

        report = GateReport(name=f"property:{self.definition.name}")
        report.gates.append(
            GateResult(
                gate=f"{self.definition.name}:grid_resolution",
                status=GateStatus.PASS if grid_spacing_nm <= 0.06 else GateStatus.WARN,
                message=f"grid spacing {grid_spacing_nm} nm over {grid.shape[0]} points",
                value=grid_spacing_nm,
                threshold=0.06,
                units="nm",
            )
        )
        measurement = Measurement(
            name=self.definition.name,
            value=free_fraction,
            uncertainty=grid_error,
            units="1",
            n_samples=int(grid.shape[0]),
            effective_samples=float(grid.shape[0]),
            method=f"probe insertion, probe radius {probe_radius_nm} nm",
            notes="geometric free volume; not comparable to positron-annihilation values",
        )
        report.gates.extend(
            self.sampling_gates(measurement, n_replicas=n_replicas, equilibration_shown=True)
        )
        return PropertyResult(
            definition=self.definition,
            measurement=measurement,
            report=report,
            n_replicas=n_replicas,
            provenance={
                "probe_radius_nm": probe_radius_nm,
                "grid_spacing_nm": grid_spacing_nm,
                "n_grid_points": int(grid.shape[0]),
                "n_atoms": int(positions.shape[0]),
                "box_nm": list(box_nm),
                **(provenance or {}),
            },
        )


def structural_calculators(defaults: AnalysisDefaults | None = None) -> list[PropertyCalculator]:
    return [
        RadiusOfGyration(defaults), EndToEndDistance(defaults), ContactNumber(defaults),
        HydrogenBondCount(defaults), PersistenceLength(defaults), FreeVolume(defaults),
    ]


__all__ = [
    "ContactNumber",
    "EndToEndDistance",
    "FreeVolume",
    "HydrogenBondCount",
    "PersistenceLength",
    "RadiusOfGyration",
    "structural_calculators",
]
