"""QM job specification, input generation, validation gates, and FF-vs-QM comparison."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from polymer_engine.core.errors import ParameterValidationError, ScientificError
from polymer_engine.core.models import Determination, GateStatus
from polymer_engine.qm.orca_input import build_input
from polymer_engine.qm.orca_parser import QMStatus, parse_orca_file
from polymer_engine.qm.spec import Atom, JobKind, QMJobSpec, Structure, TorsionSpec
from polymer_engine.qm.validation import (
    AcceptanceCriteria,
    compare_geometries,
    compare_torsion_profiles,
    kabsch_rmsd,
    validate_qm_run,
)
from tests.markers import requires_orca

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "orca"

WATER = [Atom("O", 0.0, 0.0, 0.0), Atom("H", 0.0, 0.0, 0.99), Atom("H", 0.96, 0.0, -0.25)]
ETHANE = [
    Atom("C", -0.765, 0.0, 0.0), Atom("C", 0.765, 0.0, 0.0),
    Atom("H", -1.14, 1.018, 0.0), Atom("H", -1.14, -0.509, 0.8815),
    Atom("H", -1.14, -0.509, -0.8815), Atom("H", 1.14, -1.018, 0.0),
    Atom("H", 1.14, 0.509, 0.8815), Atom("H", 1.14, 0.509, -0.8815),
]


def water(**kwargs) -> Structure:
    return Structure(atoms=list(WATER), name="water", **kwargs)


# ==========================================================================
# Structures
# ==========================================================================
class TestStructure:
    def test_valid_structure(self) -> None:
        assert water().n_atoms == 3
        assert water().electron_count() == 10

    def test_empty_structure_is_rejected(self) -> None:
        with pytest.raises(ParameterValidationError):
            Structure(atoms=[])

    def test_non_finite_coordinate_is_rejected(self) -> None:
        with pytest.raises(ParameterValidationError):
            Atom("C", float("nan"), 0.0, 0.0)

    def test_charge_changes_the_electron_count(self) -> None:
        assert Structure(atoms=list(WATER), charge=1, multiplicity=2).electron_count() == 9

    def test_impossible_charge_multiplicity_pair_is_rejected(self) -> None:
        """Ten electrons cannot be a doublet."""
        with pytest.raises(ParameterValidationError, match="multiplicity"):
            Structure(atoms=list(WATER), charge=0, multiplicity=2)

    def test_radical_cation_is_accepted(self) -> None:
        assert Structure(atoms=list(WATER), charge=1, multiplicity=2).multiplicity == 2

    def test_zero_multiplicity_is_rejected(self) -> None:
        with pytest.raises(ParameterValidationError):
            Structure(atoms=list(WATER), multiplicity=0)

    def test_xyz_round_trip(self) -> None:
        original = water()
        restored = Structure.from_xyz(original.as_xyz())
        assert restored.n_atoms == 3
        assert restored.atoms[0].symbol == "O"
        assert restored.atoms[1].z == pytest.approx(0.99)

    def test_malformed_xyz_is_rejected(self) -> None:
        with pytest.raises(ParameterValidationError):
            Structure.from_xyz("3\nname\nO 0 0 0\n")


# ==========================================================================
# Job specification
# ==========================================================================
class TestJobSpec:
    def test_method_is_required(self) -> None:
        """The engine will not choose a functional for you."""
        with pytest.raises(ParameterValidationError, match="method"):
            QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="  ", basis="def2-SVP")

    def test_basis_is_required(self) -> None:
        with pytest.raises(ParameterValidationError, match="basis"):
            QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="B3LYP", basis="")

    def test_unknown_convergence_tier_is_rejected(self) -> None:
        with pytest.raises(ParameterValidationError):
            QMJobSpec(
                kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G",
                scf_convergence="ExtremelySloppy",
            )

    def test_solvent_without_a_model_is_rejected(self) -> None:
        with pytest.raises(ParameterValidationError, match="solvation model"):
            QMJobSpec(
                kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G",
                solvent="water",
            )

    def test_model_without_a_solvent_is_rejected(self) -> None:
        with pytest.raises(ParameterValidationError, match="solvent"):
            QMJobSpec(
                kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G",
                solvation_model="CPCM",
            )

    def test_torsion_scan_requires_a_torsion(self) -> None:
        with pytest.raises(ParameterValidationError, match="TorsionSpec"):
            QMJobSpec(kind=JobKind.TORSION_SCAN, structure=water(), method="HF", basis="STO-3G")

    def test_torsion_indices_must_be_inside_the_structure(self) -> None:
        with pytest.raises(ParameterValidationError, match="outside the structure"):
            QMJobSpec(
                kind=JobKind.TORSION_SCAN, structure=water(), method="HF", basis="STO-3G",
                torsion=TorsionSpec(atoms=(0, 1, 2, 99), start_deg=0, stop_deg=180, n_points=5),
            )

    def test_torsion_needs_four_distinct_atoms(self) -> None:
        with pytest.raises(ParameterValidationError, match="distinct"):
            TorsionSpec(atoms=(0, 1, 1, 2), start_deg=0, stop_deg=180, n_points=5)

    def test_fingerprint_is_deterministic(self) -> None:
        a = QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G")
        b = QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G")
        assert a.fingerprint() == b.fingerprint()

    def test_changing_the_method_changes_the_fingerprint(self) -> None:
        a = QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G")
        b = QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="B3LYP", basis="STO-3G")
        assert a.fingerprint() != b.fingerprint()

    def test_resources_do_not_change_the_fingerprint(self) -> None:
        """n_procs changes how long a job takes, not what it computes."""
        a = QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G", n_procs=1)
        b = QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G", n_procs=8)
        assert a.fingerprint() == b.fingerprint()

    def test_torsion_angles(self) -> None:
        spec = TorsionSpec(atoms=(0, 1, 2, 3), start_deg=0.0, stop_deg=120.0, n_points=5)
        assert spec.angles() == pytest.approx([0.0, 30.0, 60.0, 90.0, 120.0])


# ==========================================================================
# Input generation
# ==========================================================================
class TestInputGeneration:
    def test_single_point_has_no_run_type_keyword(self) -> None:
        text = build_input(QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G"))
        keyword_line = next(line for line in text.splitlines() if line.startswith("!"))
        assert " Opt" not in keyword_line
        assert " Freq" not in keyword_line

    def test_optimization_requests_opt(self) -> None:
        text = build_input(QMJobSpec(kind=JobKind.OPTIMIZATION, structure=water(), method="HF", basis="STO-3G"))
        assert "Opt" in next(line for line in text.splitlines() if line.startswith("!"))

    def test_opt_freq_requests_both(self) -> None:
        line = next(
            line for line in build_input(
                QMJobSpec(kind=JobKind.OPT_FREQ, structure=water(), method="HF", basis="STO-3G")
            ).splitlines() if line.startswith("!")
        )
        assert "Opt" in line and "Freq" in line

    def test_charge_and_multiplicity_appear_on_the_coordinate_block(self) -> None:
        spec = QMJobSpec(
            kind=JobKind.SINGLE_POINT,
            structure=Structure(atoms=list(WATER), charge=1, multiplicity=2),
            method="HF", basis="STO-3G",
        )
        assert "* xyz 1 2" in build_input(spec)

    def test_solvation_is_rendered(self) -> None:
        spec = QMJobSpec(
            kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G",
            solvation_model="SMD", solvent="chloroform",
        )
        assert "SMD(chloroform)" in build_input(spec)

    def test_parallelism_is_rendered_only_when_requested(self) -> None:
        serial = build_input(QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G"))
        parallel = build_input(
            QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G", n_procs=4)
        )
        assert "%pal" not in serial
        assert "%pal nprocs 4 end" in parallel

    def test_scan_block_is_rendered(self) -> None:
        spec = QMJobSpec(
            kind=JobKind.TORSION_SCAN,
            structure=Structure(atoms=list(ETHANE), name="ethane"),
            method="HF", basis="STO-3G",
            torsion=TorsionSpec(atoms=(2, 0, 1, 6), start_deg=0.0, stop_deg=120.0, n_points=5),
        )
        text = build_input(spec)
        assert "D 2 0 1 6 = 0.0000, 120.0000, 5" in text
        assert "Scan" in text

    def test_fingerprint_is_embedded_for_traceability(self) -> None:
        spec = QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G")
        assert spec.fingerprint() in build_input(spec)

    def test_no_scientific_keyword_appears_that_was_not_requested(self) -> None:
        """The keyword line contains exactly what the spec asked for."""
        spec = QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G")
        line = next(line for line in build_input(spec).splitlines() if line.startswith("!"))
        assert set(line[1:].split()) == {"HF", "STO-3G", "TightSCF"}


# ==========================================================================
# Validation gates
# ==========================================================================
class TestValidationGates:
    def test_a_good_frequency_job_passes(self) -> None:
        spec = QMJobSpec(kind=JobKind.OPT_FREQ, structure=water(), method="HF", basis="STO-3G")
        output = parse_orca_file(FIXTURES / "freq.out.gz", expect_geometry=True, expect_frequencies=True)
        result = validate_qm_run(spec, output)
        assert result.passed is True
        assert result.report.status is GateStatus.PASS
        assert result.determination is Determination.KNOWN

    def test_a_failed_job_does_not_pass(self) -> None:
        spec = QMJobSpec(kind=JobKind.OPTIMIZATION, structure=water(), method="HF", basis="STO-3G")
        output = parse_orca_file(FIXTURES / "scan_nonconverged.out", expect_geometry=True)
        result = validate_qm_run(spec, output)
        assert result.passed is False
        assert result.status is QMStatus.FAILED_TERMINATION
        assert result.determination is Determination.REQUIRES_VALIDATION

    def test_missing_output_fails_rather_than_raising(self) -> None:
        spec = QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G")
        result = validate_qm_run(spec, None)
        assert result.passed is False
        assert result.report.status is GateStatus.FAIL

    def test_energy_is_reported_with_units(self) -> None:
        spec = QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G")
        output = parse_orca_file(FIXTURES / "h2.out.gz")
        result = validate_qm_run(spec, output)
        assert result.energy is not None
        assert result.energy.units == "kJ/mol"
        assert "only differences" in (result.energy.notes or "")

    def test_provenance_records_the_level_of_theory(self) -> None:
        spec = QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G")
        result = validate_qm_run(spec, parse_orca_file(FIXTURES / "h2.out.gz"))
        assert result.provenance["method"] == "HF"
        assert result.provenance["basis"] == "STO-3G"
        assert result.provenance["orca_version"].startswith("6.")


# ==========================================================================
# RMSD
# ==========================================================================
class TestKabschRmsd:
    def test_identical_structures_have_zero_rmsd(self) -> None:
        coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        assert kabsch_rmsd(coords, coords) == pytest.approx(0.0, abs=1e-12)

    def test_translation_is_removed(self) -> None:
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert kabsch_rmsd(a, a + 100.0) == pytest.approx(0.0, abs=1e-10)

    def test_rotation_is_removed(self) -> None:
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        theta = 0.7
        rotation = np.array(
            [[math.cos(theta), -math.sin(theta), 0.0],
             [math.sin(theta), math.cos(theta), 0.0],
             [0.0, 0.0, 1.0]]
        )
        assert kabsch_rmsd(a, a @ rotation.T) == pytest.approx(0.0, abs=1e-10)

    def test_a_mirror_image_is_not_treated_as_identical(self) -> None:
        """A reflection is not a rotation; a chiral centre must not superpose onto its enantiomer."""
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mirrored = a * np.array([1.0, 1.0, -1.0])
        assert kabsch_rmsd(a, mirrored) > 0.1

    def test_a_known_displacement(self) -> None:
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
        assert kabsch_rmsd(a, b) == pytest.approx(0.1, abs=1e-9)

    def test_mismatched_atom_counts_raise(self) -> None:
        with pytest.raises(ScientificError, match="same number of atoms"):
            kabsch_rmsd([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


# ==========================================================================
# Force field vs QM
# ==========================================================================
class TestForceFieldComparison:
    def test_without_criteria_the_engine_refuses_to_judge(self) -> None:
        """No universal tolerance exists, so none is invented."""
        a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        b = [[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [0.0, 1.0, 0.0]]
        result = compare_geometries(a, b)
        assert result.passed is None
        assert result.determination is Determination.REQUIRES_VALIDATION
        assert "REQUIRES_EXPERT_DECISION" in result.notes
        assert result.report.status is GateStatus.INCONCLUSIVE
        # The number is still reported, so a human can judge it.
        assert result.metrics["geometry_rmsd"].value > 0

    def test_with_criteria_a_close_geometry_passes(self) -> None:
        a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        b = [[0.0, 0.0, 0.0], [1.02, 0.0, 0.0], [0.0, 1.0, 0.0]]
        result = compare_geometries(a, b, criteria=AcceptanceCriteria.general_organic_forcefield())
        assert result.passed is True

    def test_with_criteria_a_distant_geometry_fails(self) -> None:
        a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        b = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        result = compare_geometries(a, b, criteria=AcceptanceCriteria.general_organic_forcefield())
        assert result.passed is False
        assert result.report.status is GateStatus.FAIL

    def test_criteria_carry_their_justification(self) -> None:
        criteria = AcceptanceCriteria.conformational_free_energy()
        assert "exponentially" in criteria.justification
        assert criteria.max_torsion_rmse_kj_mol < (
            AcceptanceCriteria.general_organic_forcefield().max_torsion_rmse_kj_mol
        )

    def test_torsion_profiles_are_compared_after_gauge_removal(self) -> None:
        """A profile is meaningful only up to an additive constant."""
        reference = [(0.0, 12.0), (30.0, 6.0), (60.0, 0.0), (90.0, 6.0), (120.0, 12.0)]
        shifted = [(angle, energy + 500.0) for angle, energy in reference]
        result = compare_torsion_profiles(
            shifted, reference, criteria=AcceptanceCriteria.general_organic_forcefield()
        )
        assert result.passed is True
        assert result.metrics["torsion_rmse"].value == pytest.approx(0.0, abs=1e-9)

    def test_a_wrong_barrier_fails(self) -> None:
        reference = [(0.0, 12.0), (30.0, 6.0), (60.0, 0.0), (90.0, 6.0), (120.0, 12.0)]
        candidate = [(0.0, 30.0), (30.0, 15.0), (60.0, 0.0), (90.0, 15.0), (120.0, 30.0)]
        result = compare_torsion_profiles(
            candidate, reference, criteria=AcceptanceCriteria.general_organic_forcefield()
        )
        assert result.passed is False
        assert result.metrics["barrier_error"].value == pytest.approx(18.0)

    def test_mismatched_angles_raise(self) -> None:
        with pytest.raises(ScientificError, match="same angles"):
            compare_torsion_profiles([(0.0, 0.0), (30.0, 1.0)], [(0.0, 0.0), (60.0, 1.0)])

    def test_real_qm_profile_against_a_perfect_mimic(self) -> None:
        """Use the real ethane scan as the reference; a copy of it must agree exactly."""
        scan = parse_orca_file(FIXTURES / "scan_ethane.out", expect_geometry=True)
        reference = scan.scan_profile_kj_mol()
        result = compare_torsion_profiles(
            list(reference), reference, criteria=AcceptanceCriteria.conformational_free_energy()
        )
        assert result.passed is True
        assert result.metrics["reference_barrier"].value == pytest.approx(12.01, abs=0.2)


# ==========================================================================
# Real ORCA execution
# ==========================================================================
@requires_orca
class TestRealOrcaExecution:
    @pytest.mark.slow
    def test_real_single_point_runs_and_validates(self, tmp_path: Path) -> None:
        from polymer_engine.core.config import load_config
        from polymer_engine.qm.orca_runner import build_orca_runner

        config = load_config(
            discover=False, use_env=False, overrides={"safety": {"execution_enabled": True}}
        )
        runner = build_orca_runner(config)
        spec = QMJobSpec(
            kind=JobKind.SINGLE_POINT,
            structure=Structure(
                atoms=[Atom("H", 0.0, 0.0, 0.0), Atom("H", 0.0, 0.0, 0.74)], name="h2"
            ),
            method="HF", basis="STO-3G", label="h2_sp", timeout_s=300,
        )
        run = runner.run_job(spec, tmp_path)
        assert run.succeeded, run.error
        # Independently reproduced value for H2 at 0.74 A, HF/STO-3G.
        assert run.final_energy_hartree == pytest.approx(-1.1167, abs=1e-3)
        assert validate_qm_run(spec, run.output).passed is True

    @pytest.mark.slow
    def test_real_optimization_produces_a_usable_structure(self, tmp_path: Path) -> None:
        from polymer_engine.core.config import load_config
        from polymer_engine.qm.orca_runner import build_orca_runner

        config = load_config(
            discover=False, use_env=False, overrides={"safety": {"execution_enabled": True}}
        )
        runner = build_orca_runner(config)
        spec = QMJobSpec(
            kind=JobKind.OPTIMIZATION, structure=water(), method="HF", basis="STO-3G",
            label="water_opt", timeout_s=600,
        )
        run = runner.run_job(spec, tmp_path)
        assert run.succeeded, run.error
        optimized = run.optimized_structure()
        assert optimized is not None
        assert optimized.n_atoms == 3
        bond = math.dist(optimized.atoms[0].as_tuple(), optimized.atoms[1].as_tuple())
        assert bond == pytest.approx(0.99, abs=0.05)

    def test_execution_disabled_produces_no_success(self, tmp_path: Path) -> None:
        from polymer_engine.core.config import load_config
        from polymer_engine.qm.orca_runner import build_orca_runner

        config = load_config(discover=False, use_env=False)  # execution off by default
        runner = build_orca_runner(config)
        spec = QMJobSpec(kind=JobKind.SINGLE_POINT, structure=water(), method="HF", basis="STO-3G")
        run = runner.run_job(spec, tmp_path)
        assert run.executed is False
        assert run.succeeded is False
        assert "disabled" in (run.error or "")
        # The input is still written, so the job can be inspected or run by hand.
        assert Path(run.input_path).exists()
