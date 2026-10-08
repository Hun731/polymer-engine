"""Quantum-chemistry workflows: specification, execution, parsing, validation."""

from polymer_engine.qm.orca_input import build_input, write_input
from polymer_engine.qm.orca_parser import (
    HARTREE_TO_KJ_MOL,
    OrcaOutput,
    QMStatus,
    parse_orca_file,
    parse_orca_output,
)
from polymer_engine.qm.orca_runner import ORCARunner, QMRun, build_orca_runner
from polymer_engine.qm.spec import Atom, JobKind, QMJobSpec, Structure, TorsionSpec
from polymer_engine.qm.validation import (
    AcceptanceCriteria,
    ComparisonResult,
    QMValidationResult,
    compare_geometries,
    compare_torsion_profiles,
    kabsch_rmsd,
    validate_qm_run,
)

__all__ = [
    "HARTREE_TO_KJ_MOL",
    "AcceptanceCriteria",
    "Atom",
    "ComparisonResult",
    "JobKind",
    "ORCARunner",
    "OrcaOutput",
    "QMJobSpec",
    "QMRun",
    "QMStatus",
    "QMValidationResult",
    "Structure",
    "TorsionSpec",
    "build_input",
    "build_orca_runner",
    "compare_geometries",
    "compare_torsion_profiles",
    "kabsch_rmsd",
    "parse_orca_file",
    "parse_orca_output",
    "validate_qm_run",
    "write_input",
]
