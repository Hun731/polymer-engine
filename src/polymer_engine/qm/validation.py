"""QM validation gates and force-field-versus-QM comparison.

Two distinct things live here:

1. :func:`validate_qm_run` -- did the quantum calculation itself succeed, at the
   level of rigour the caller asked for?
2. :func:`compare_geometries` / :func:`compare_torsion_profiles` -- does a candidate
   force field reproduce the QM reference?

For (2) the engine deliberately has **no universal acceptance threshold**.  What counts
as "close enough" depends on the property, the level of theory, and what the model will
be used for; a heat-of-vaporisation study and a conformational-free-energy study do not
share a tolerance.  Thresholds are therefore supplied as an explicit
:class:`AcceptanceCriteria` object, and calling a comparison without one yields
``REQUIRES_EXPERT_DECISION`` rather than a made-up pass mark.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polymer_engine.core.errors import ScientificError
from polymer_engine.core.models import Determination, GateReport, GateResult, GateStatus, Measurement
from polymer_engine.qm.orca_parser import HARTREE_TO_KJ_MOL, OrcaOutput, QMStatus
from polymer_engine.qm.spec import QMJobSpec, Structure


@dataclass(frozen=True, slots=True)
class AcceptanceCriteria:
    """Scientific tolerances for accepting a force field against a QM reference.

    Every field is required to be set deliberately.  The class-level helpers below
    provide *named, cited* starting points rather than a silent default, so that a
    manifest records which convention was chosen.
    """

    name: str
    justification: str
    max_rmsd_angstrom: float | None = None
    max_bond_deviation_angstrom: float | None = None
    max_angle_deviation_deg: float | None = None
    max_torsion_rmse_kj_mol: float | None = None
    max_barrier_error_kj_mol: float | None = None
    max_barrier_relative_error: float | None = None
    require_no_imaginary_modes: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "justification": self.justification,
            "max_rmsd_angstrom": self.max_rmsd_angstrom,
            "max_bond_deviation_angstrom": self.max_bond_deviation_angstrom,
            "max_angle_deviation_deg": self.max_angle_deviation_deg,
            "max_torsion_rmse_kj_mol": self.max_torsion_rmse_kj_mol,
            "max_barrier_error_kj_mol": self.max_barrier_error_kj_mol,
            "max_barrier_relative_error": self.max_barrier_relative_error,
            "require_no_imaginary_modes": self.require_no_imaginary_modes,
        }

    @classmethod
    def general_organic_forcefield(cls) -> AcceptanceCriteria:
        """A commonly used tolerance set for general-purpose organic force fields.

        These are the tolerances typical of GAFF/OPLS-style parameterisation papers,
        not a derivation.  Adopt them only if they suit your application, and record
        the choice: a manifest that names this preset is auditable, one that inherits
        an invisible default is not.
        """
        return cls(
            name="general_organic_forcefield",
            justification=(
                "Tolerances in the range routinely quoted for general-purpose organic "
                "force fields (GAFF/OPLS-style parameterisation). A convention, not a "
                "derivation; review per application."
            ),
            max_rmsd_angstrom=0.2,
            max_bond_deviation_angstrom=0.02,
            max_angle_deviation_deg=3.0,
            max_torsion_rmse_kj_mol=2.0,
            max_barrier_error_kj_mol=4.0,
            max_barrier_relative_error=0.25,
        )

    @classmethod
    def conformational_free_energy(cls) -> AcceptanceCriteria:
        """Tighter torsional tolerances for work whose answer is a conformer population.

        Populations depend exponentially on relative energies, so a 4 kJ/mol torsional
        error is roughly a factor of five in a population ratio at 300 K.
        """
        return cls(
            name="conformational_free_energy",
            justification=(
                "Conformer populations depend exponentially on relative energies: at "
                "300 K, 4 kJ/mol is about a factor of five in a population ratio, so "
                "torsional agreement must be tighter than for bulk-property work."
            ),
            max_rmsd_angstrom=0.15,
            max_torsion_rmse_kj_mol=1.0,
            max_barrier_error_kj_mol=2.0,
            max_barrier_relative_error=0.15,
        )


@dataclass
class QMValidationResult:
    """The formal outcome of a QM validation step."""

    passed: bool
    status: QMStatus
    method: str
    basis: str
    convergence: dict[str, Any] = field(default_factory=dict)
    energy: Measurement | None = None
    geometry: list[tuple[str, float, float, float]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    report: GateReport = field(default_factory=lambda: GateReport(name="qm_validation"))

    @property
    def determination(self) -> Determination:
        if self.passed:
            return Determination.KNOWN
        if self.status is QMStatus.UNPARSEABLE:
            return Determination.UNKNOWN
        return Determination.REQUIRES_VALIDATION

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status.value,
            "determination": self.determination.value,
            "method": self.method,
            "basis": self.basis,
            "convergence": self.convergence,
            "energy": self.energy.model_dump(mode="json") if self.energy else None,
            "n_atoms": len(self.geometry),
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
            "gate_status": self.report.status.value,
            "gates": [g.model_dump(mode="json") for g in self.report.gates],
        }


def validate_qm_run(
    spec: QMJobSpec,
    output: OrcaOutput | None,
    *,
    require_no_imaginary: bool | None = None,
    provenance: dict[str, Any] | None = None,
) -> QMValidationResult:
    """Turn a parsed ORCA result into a formal validation outcome.

    ``require_no_imaginary`` defaults to True for jobs that computed frequencies: an
    imaginary mode means the structure is a saddle point, not a minimum, and using it
    as a reference geometry would be wrong.
    """
    report = GateReport(name=f"qm:{spec.label}")

    if output is None:
        report.gates.append(
            GateResult(
                gate="qm:output_available",
                status=GateStatus.FAIL,
                message="no ORCA output was produced",
            )
        )
        return QMValidationResult(
            passed=False,
            status=QMStatus.INCOMPLETE,
            method=spec.method,
            basis=spec.basis,
            diagnostics=["no ORCA output was produced"],
            provenance=provenance or {},
            report=report,
        )

    # -- did it finish? --------------------------------------------------
    report.gates.append(
        GateResult(
            gate="qm:termination",
            status=GateStatus.PASS if output.normal_termination and not output.error_termination
            else GateStatus.FAIL,
            message=(
                "ORCA terminated normally"
                if output.normal_termination and not output.error_termination
                else "ORCA did not terminate normally"
            ),
            evidence={
                "normal_termination": output.normal_termination,
                "error_termination": output.error_termination,
            },
        )
    )

    # -- SCF -------------------------------------------------------------
    if output.scf_converged is None:
        report.gates.append(
            GateResult(
                gate="qm:scf_converged",
                status=GateStatus.INCONCLUSIVE,
                message="no SCF convergence statement was found in the output",
            )
        )
    else:
        report.gates.append(
            GateResult(
                gate="qm:scf_converged",
                status=GateStatus.PASS if output.scf_converged else GateStatus.FAIL,
                message=(
                    f"SCF converged in {output.scf_cycles} cycles"
                    if output.scf_converged
                    else "SCF did not converge"
                ),
                value=float(output.scf_cycles) if output.scf_cycles else None,
                units="1",
            )
        )

    # -- geometry --------------------------------------------------------
    if spec.requires_geometry_convergence:
        if output.geometry_converged is None:
            report.gates.append(
                GateResult(
                    gate="qm:geometry_converged",
                    status=GateStatus.INCONCLUSIVE,
                    message="a geometry optimisation was requested but no convergence statement was found",
                )
            )
        else:
            report.gates.append(
                GateResult(
                    gate="qm:geometry_converged",
                    status=GateStatus.PASS if output.geometry_converged else GateStatus.FAIL,
                    message=(
                        f"geometry converged after {output.optimization_cycles} cycles"
                        if output.geometry_converged
                        else "geometry optimisation did not converge"
                    ),
                    evidence=output.geometry_convergence.as_dict(),
                )
            )

    # -- frequencies ------------------------------------------------------
    check_imaginary = (
        require_no_imaginary if require_no_imaginary is not None else spec.requires_frequencies
    )
    if spec.requires_frequencies:
        if not output.frequencies_cm:
            report.gates.append(
                GateResult(
                    gate="qm:frequencies_present",
                    status=GateStatus.FAIL,
                    message="frequencies were requested but none were found",
                )
            )
        elif check_imaginary:
            n_imaginary = output.n_imaginary or 0
            report.gates.append(
                GateResult(
                    gate="qm:no_imaginary_modes",
                    status=GateStatus.PASS if n_imaginary == 0 else GateStatus.FAIL,
                    message=(
                        "no imaginary modes; the structure is a local minimum"
                        if n_imaginary == 0
                        else f"{n_imaginary} imaginary mode(s): this is a saddle point, not a minimum"
                    ),
                    value=float(n_imaginary),
                    threshold=0.0,
                    units="1",
                )
            )

    # -- energy -----------------------------------------------------------
    energy: Measurement | None = None
    if output.final_energy_hartree is None:
        report.gates.append(
            GateResult(
                gate="qm:energy_present",
                status=GateStatus.FAIL,
                message="no final energy was reported",
            )
        )
    else:
        energy = Measurement(
            name="qm_total_energy",
            value=output.final_energy_hartree * HARTREE_TO_KJ_MOL,
            units="kJ/mol",
            method=f"{spec.method}/{spec.basis}",
            notes="absolute electronic energy; only differences are physically meaningful",
        )
        report.gates.append(
            GateResult(
                gate="qm:energy_present",
                status=GateStatus.PASS,
                message=f"final energy {output.final_energy_hartree:.8f} Eh",
                value=energy.value,
                units="kJ/mol",
            )
        )

    passed = output.status is QMStatus.COMPLETED and report.promotable
    return QMValidationResult(
        passed=passed,
        status=output.status,
        method=spec.method,
        basis=spec.basis,
        convergence={
            "scf_converged": output.scf_converged,
            "scf_cycles": output.scf_cycles,
            "geometry_converged": output.geometry_converged,
            "optimization_cycles": output.optimization_cycles,
            **output.geometry_convergence.as_dict(),
        },
        energy=energy,
        geometry=output.final_geometry,
        diagnostics=list(output.diagnostics),
        provenance={
            "method": spec.method,
            "basis": spec.basis,
            "orca_version": output.orca_version,
            "fingerprint": spec.fingerprint(),
            **(provenance or {}),
        },
        report=report,
    )


# --------------------------------------------------------------------------
# Force field versus QM
# --------------------------------------------------------------------------
def kabsch_rmsd(a: Sequence[Sequence[float]] | np.ndarray, b: Sequence[Sequence[float]] | np.ndarray) -> float:
    """Minimum RMSD between two structures after optimal translation and rotation.

    Without superposition, RMSD measures how differently the two structures happen to
    be oriented, which says nothing about whether they are the same shape.
    """
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if first.shape != second.shape:
        raise ScientificError(
            "Structures must have the same number of atoms to compare",
            first=first.shape,
            second=second.shape,
        )
    if first.ndim != 2 or first.shape[1] != 3:
        raise ScientificError("Coordinates must be an (N, 3) array", shape=first.shape)
    if first.shape[0] == 0:
        raise ScientificError("Cannot compare empty structures")

    first = first - first.mean(axis=0)
    second = second - second.mean(axis=0)
    covariance = first.T @ second
    u, _, vt = np.linalg.svd(covariance)
    # Guard against an improper rotation (a reflection), which would report a mirror
    # image as a perfect match.
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign])
    rotation = vt.T @ correction @ u.T
    rotated = first @ rotation.T
    return float(np.sqrt(((rotated - second) ** 2).sum() / first.shape[0]))


@dataclass
class ComparisonResult:
    """Force-field versus QM agreement, judged against explicit criteria."""

    name: str
    passed: bool | None
    determination: Determination
    metrics: dict[str, Measurement] = field(default_factory=dict)
    criteria: AcceptanceCriteria | None = None
    report: GateReport = field(default_factory=lambda: GateReport(name="ff_vs_qm"))
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "determination": self.determination.value,
            "metrics": {k: v.model_dump(mode="json") for k, v in self.metrics.items()},
            "criteria": self.criteria.as_dict() if self.criteria else None,
            "gate_status": self.report.status.value,
            "gates": [g.model_dump(mode="json") for g in self.report.gates],
            "notes": self.notes,
        }


def compare_geometries(
    candidate: Structure | Sequence[Sequence[float]],
    reference: Structure | Sequence[Sequence[float]],
    *,
    criteria: AcceptanceCriteria | None = None,
    label: str = "geometry",
) -> ComparisonResult:
    """Compare a candidate geometry against a QM reference.

    Without ``criteria`` the result is ``REQUIRES_EXPERT_DECISION``: the RMSD is
    computed and reported, but the engine will not decide whether it is acceptable.
    """
    candidate_xyz = _coordinates(candidate)
    reference_xyz = _coordinates(reference)
    rmsd = kabsch_rmsd(candidate_xyz, reference_xyz)

    measurement = Measurement(
        name="geometry_rmsd",
        value=rmsd,
        units="A",
        method="Kabsch superposition (translation and rotation removed)",
    )
    report = GateReport(name=f"ff_vs_qm:{label}")

    if criteria is None or criteria.max_rmsd_angstrom is None:
        report.gates.append(
            GateResult(
                gate=f"ff_vs_qm:{label}:rmsd",
                status=GateStatus.INCONCLUSIVE,
                message=(
                    f"RMSD is {rmsd:.4f} A, but no acceptance criterion was supplied; "
                    "the engine will not invent a tolerance"
                ),
                value=rmsd,
                units="A",
            )
        )
        return ComparisonResult(
            name=label,
            passed=None,
            determination=Determination.REQUIRES_VALIDATION,
            metrics={"geometry_rmsd": measurement},
            criteria=criteria,
            report=report,
            notes="REQUIRES_EXPERT_DECISION: no RMSD tolerance was configured",
        )

    ok = rmsd <= criteria.max_rmsd_angstrom
    report.gates.append(
        GateResult(
            gate=f"ff_vs_qm:{label}:rmsd",
            status=GateStatus.PASS if ok else GateStatus.FAIL,
            message=f"geometry RMSD {rmsd:.4f} A against a tolerance of {criteria.max_rmsd_angstrom} A",
            value=rmsd,
            threshold=criteria.max_rmsd_angstrom,
            units="A",
        )
    )
    return ComparisonResult(
        name=label,
        passed=ok,
        determination=Determination.KNOWN,
        metrics={"geometry_rmsd": measurement},
        criteria=criteria,
        report=report,
    )


def compare_torsion_profiles(
    candidate: Sequence[tuple[float, float]],
    reference: Sequence[tuple[float, float]],
    *,
    criteria: AcceptanceCriteria | None = None,
    label: str = "torsion",
) -> ComparisonResult:
    """Compare a force-field torsional profile against a QM one.

    Both profiles are shifted to their own minimum before comparison.  A torsional
    profile is only meaningful up to an additive constant, so comparing raw energies
    would measure the arbitrary zero rather than the shape.
    """
    candidate_angles, candidate_energies = _profile(candidate, "candidate")
    reference_angles, reference_energies = _profile(reference, "reference")

    if not np.allclose(candidate_angles, reference_angles, atol=1e-6):
        raise ScientificError(
            "Torsion profiles must be sampled at the same angles to compare pointwise",
            candidate_angles=candidate_angles.tolist()[:8],
            reference_angles=reference_angles.tolist()[:8],
        )

    candidate_shifted = candidate_energies - candidate_energies.min()
    reference_shifted = reference_energies - reference_energies.min()
    residual = candidate_shifted - reference_shifted
    rmse = float(np.sqrt((residual**2).mean()))
    max_deviation = float(np.abs(residual).max())

    candidate_barrier = float(candidate_shifted.max())
    reference_barrier = float(reference_shifted.max())
    barrier_error = candidate_barrier - reference_barrier
    relative_error = (
        abs(barrier_error) / reference_barrier if reference_barrier > 1e-9 else None
    )

    metrics = {
        "torsion_rmse": Measurement(
            name="torsion_rmse", value=rmse, units="kJ/mol",
            method="pointwise RMSE after shifting each profile to its own minimum",
        ),
        "torsion_max_deviation": Measurement(
            name="torsion_max_deviation", value=max_deviation, units="kJ/mol",
        ),
        "candidate_barrier": Measurement(
            name="candidate_barrier", value=candidate_barrier, units="kJ/mol",
        ),
        "reference_barrier": Measurement(
            name="reference_barrier", value=reference_barrier, units="kJ/mol",
        ),
        "barrier_error": Measurement(
            name="barrier_error", value=barrier_error, units="kJ/mol",
        ),
    }
    report = GateReport(name=f"ff_vs_qm:{label}")

    if criteria is None or (
        criteria.max_torsion_rmse_kj_mol is None and criteria.max_barrier_error_kj_mol is None
    ):
        report.gates.append(
            GateResult(
                gate=f"ff_vs_qm:{label}:rmse",
                status=GateStatus.INCONCLUSIVE,
                message=(
                    f"torsional RMSE is {rmse:.3f} kJ/mol and the barrier differs by "
                    f"{barrier_error:+.3f} kJ/mol, but no acceptance criterion was supplied"
                ),
                value=rmse,
                units="kJ/mol",
            )
        )
        return ComparisonResult(
            name=label,
            passed=None,
            determination=Determination.REQUIRES_VALIDATION,
            metrics=metrics,
            criteria=criteria,
            report=report,
            notes="REQUIRES_EXPERT_DECISION: no torsional tolerance was configured",
        )

    if criteria.max_torsion_rmse_kj_mol is not None:
        ok = rmse <= criteria.max_torsion_rmse_kj_mol
        report.gates.append(
            GateResult(
                gate=f"ff_vs_qm:{label}:rmse",
                status=GateStatus.PASS if ok else GateStatus.FAIL,
                message=f"torsional RMSE {rmse:.3f} kJ/mol against {criteria.max_torsion_rmse_kj_mol} kJ/mol",
                value=rmse,
                threshold=criteria.max_torsion_rmse_kj_mol,
                units="kJ/mol",
            )
        )
    if criteria.max_barrier_error_kj_mol is not None:
        ok = abs(barrier_error) <= criteria.max_barrier_error_kj_mol
        report.gates.append(
            GateResult(
                gate=f"ff_vs_qm:{label}:barrier",
                status=GateStatus.PASS if ok else GateStatus.FAIL,
                message=(
                    f"barrier differs by {barrier_error:+.3f} kJ/mol against a tolerance of "
                    f"{criteria.max_barrier_error_kj_mol} kJ/mol"
                ),
                value=abs(barrier_error),
                threshold=criteria.max_barrier_error_kj_mol,
                units="kJ/mol",
            )
        )
    if criteria.max_barrier_relative_error is not None and relative_error is not None:
        ok = relative_error <= criteria.max_barrier_relative_error
        report.gates.append(
            GateResult(
                gate=f"ff_vs_qm:{label}:barrier_relative",
                status=GateStatus.PASS if ok else GateStatus.FAIL,
                message=f"relative barrier error {relative_error:.1%}",
                value=relative_error,
                threshold=criteria.max_barrier_relative_error,
                units="1",
            )
        )

    return ComparisonResult(
        name=label,
        passed=report.status is GateStatus.PASS,
        determination=Determination.KNOWN,
        metrics=metrics,
        criteria=criteria,
        report=report,
    )


def _coordinates(value: Structure | Sequence[Sequence[float]]) -> np.ndarray:
    if isinstance(value, Structure):
        return np.asarray([atom.as_tuple() for atom in value.atoms], dtype=float)
    return np.asarray(value, dtype=float)


def _profile(points: Sequence[tuple[float, float]], label: str) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        raise ScientificError(f"{label} torsion profile needs at least two points", n=len(points))
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ScientificError(f"{label} profile must be (angle, energy) pairs", shape=array.shape)
    if not np.all(np.isfinite(array)):
        raise ScientificError(f"{label} profile contains non-finite values")
    order = np.argsort(array[:, 0])
    return array[order, 0], array[order, 1]


__all__ = [
    "AcceptanceCriteria",
    "ComparisonResult",
    "QMValidationResult",
    "compare_geometries",
    "compare_torsion_profiles",
    "kabsch_rmsd",
    "validate_qm_run",
]
