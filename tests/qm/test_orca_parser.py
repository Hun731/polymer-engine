"""ORCA parsing, against real ORCA 6.1.1 output captured on this machine.

Every fixture in ``tests/fixtures/orca`` is genuine ORCA output, not a hand-written
approximation.  ``scan_nonconverged.out`` is the important one: ORCA exited with
status 0, printed no normal-termination banner, and reported that the optimisation hit
its cycle limit.  A parser that trusted the exit status would call it a success.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from polymer_engine.core.errors import ResponseFormatError
from polymer_engine.qm.orca_parser import (
    HARTREE_TO_KJ_MOL,
    QMStatus,
    parse_orca_file,
    parse_orca_output,
    read_output_text,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "orca"


@pytest.fixture(scope="module")
def single_point():
    return parse_orca_file(FIXTURES / "h2.out.gz")


@pytest.fixture(scope="module")
def optimization():
    return parse_orca_file(FIXTURES / "opt.out.gz", expect_geometry=True)


@pytest.fixture(scope="module")
def frequency():
    return parse_orca_file(FIXTURES / "freq.out.gz", expect_geometry=True, expect_frequencies=True)


@pytest.fixture(scope="module")
def scan():
    return parse_orca_file(FIXTURES / "scan_ethane.out", expect_geometry=True)


@pytest.fixture(scope="module")
def failed_scan():
    return parse_orca_file(FIXTURES / "scan_nonconverged.out", expect_geometry=True)


# ==========================================================================
# Single point
# ==========================================================================
class TestSinglePoint:
    def test_status_and_energy(self, single_point) -> None:
        assert single_point.status is QMStatus.COMPLETED
        assert single_point.succeeded is True
        # H2 at 0.74 A, HF/STO-3G. Verified by running ORCA 6.1.1 locally.
        assert single_point.final_energy_hartree == pytest.approx(-1.116759307204, abs=1e-9)

    def test_energy_converts_to_kj_per_mol(self, single_point) -> None:
        expected = -1.116759307204 * HARTREE_TO_KJ_MOL
        assert single_point.final_energy_kj_mol == pytest.approx(expected, rel=1e-12)

    def test_scf_convergence_is_reported(self, single_point) -> None:
        assert single_point.scf_converged is True
        assert single_point.scf_cycles is not None and single_point.scf_cycles >= 1

    def test_version_is_extracted(self, single_point) -> None:
        assert single_point.orca_version is not None
        assert single_point.orca_version.startswith("6.")

    def test_keywords_are_recovered_from_the_echoed_input(self, single_point) -> None:
        joined = " ".join(single_point.keywords).upper()
        assert "HF" in joined and "STO-3G" in joined

    def test_no_geometry_convergence_claimed_for_a_single_point(self, single_point) -> None:
        assert single_point.geometry_converged is None


# ==========================================================================
# Optimization
# ==========================================================================
class TestOptimization:
    def test_converged(self, optimization) -> None:
        assert optimization.status is QMStatus.COMPLETED
        assert optimization.geometry_converged is True

    def test_final_geometry_is_extracted(self, optimization) -> None:
        assert len(optimization.final_geometry) == 3
        symbols = [atom[0] for atom in optimization.final_geometry]
        assert symbols == ["O", "H", "H"]

    def test_optimized_water_geometry_is_physically_sensible(self, optimization) -> None:
        """HF/STO-3G water: O-H near 0.99 A, H-O-H near 100 degrees."""
        atoms = optimization.final_geometry
        o = atoms[0][1:]
        h1 = atoms[1][1:]
        h2 = atoms[2][1:]
        d1 = math.dist(o, h1)
        d2 = math.dist(o, h2)
        assert d1 == pytest.approx(0.99, abs=0.03)
        assert d2 == pytest.approx(0.99, abs=0.03)

        v1 = [h1[i] - o[i] for i in range(3)]
        v2 = [h2[i] - o[i] for i in range(3)]
        cos = sum(a * b for a, b in zip(v1, v2, strict=True)) / (d1 * d2)
        angle = math.degrees(math.acos(cos))
        assert angle == pytest.approx(100.0, abs=3.0)

    def test_initial_and_final_geometries_differ(self, optimization) -> None:
        assert optimization.initial_geometry != optimization.final_geometry

    def test_convergence_table_is_parsed(self, optimization) -> None:
        convergence = optimization.geometry_convergence
        assert convergence.all_converged is True
        assert convergence.rms_gradient is not None
        assert convergence.max_gradient is not None
        assert abs(convergence.rms_gradient) < 1e-3

    def test_energy_decreases_during_optimization(self, optimization) -> None:
        energies = optimization.all_energies_hartree
        assert len(energies) > 1
        assert energies[-1] <= energies[0] + 1e-9

    def test_dipole_is_extracted(self, optimization) -> None:
        assert optimization.dipole_debye is not None
        assert optimization.dipole_debye > 1.0  # water is polar

    def test_cycles_reported(self, optimization) -> None:
        assert optimization.optimization_cycles is not None
        assert optimization.optimization_cycles > 0


# ==========================================================================
# Frequencies
# ==========================================================================
class TestFrequencies:
    def test_frequencies_parsed(self, frequency) -> None:
        assert frequency.status is QMStatus.COMPLETED
        # Water: 3N = 9 printed modes, 6 of which are the zero translations/rotations.
        assert len(frequency.frequencies_cm) == 9

    def test_three_real_vibrational_modes(self, frequency) -> None:
        real = [f for f in frequency.frequencies_cm if f > 1.0]
        assert len(real) == 3

    def test_no_imaginary_modes_at_a_minimum(self, frequency) -> None:
        assert frequency.n_imaginary == 0

    def test_frequencies_are_in_the_expected_range(self, frequency) -> None:
        """HF/STO-3G overestimates; bend near 2170, stretches near 4100-4400 cm-1."""
        real = sorted(f for f in frequency.frequencies_cm if f > 1.0)
        assert real[0] == pytest.approx(2169.83, abs=1.0)
        assert real[1] == pytest.approx(4139.63, abs=1.0)
        assert real[2] == pytest.approx(4390.68, abs=1.0)

    def test_thermochemistry_is_extracted(self, frequency) -> None:
        assert frequency.zero_point_energy_hartree is not None
        assert frequency.zero_point_energy_hartree > 0
        assert frequency.gibbs_free_energy_hartree is not None
        assert frequency.gibbs_free_energy_hartree < frequency.final_energy_hartree + 0.1

    def test_charges_are_extracted(self, frequency) -> None:
        assert len(frequency.mulliken_charges) == 3
        # Oxygen carries the negative charge in water.
        assert frequency.mulliken_charges[0] < 0
        assert sum(frequency.mulliken_charges) == pytest.approx(0.0, abs=1e-4)


# ==========================================================================
# Torsional scan
# ==========================================================================
class TestTorsionScan:
    def test_scan_points_parsed(self, scan) -> None:
        assert scan.status is QMStatus.COMPLETED
        assert len(scan.scan_points) == 5
        angles = [p.coordinate for p in scan.scan_points]
        assert angles == pytest.approx([0.0, 30.0, 60.0, 90.0, 120.0])

    def test_ethane_barrier_matches_experiment(self, scan) -> None:
        """Real HF/STO-3G ethane rotational barrier vs the experimental ~12.1 kJ/mol.

        This is a physics check on the whole chain: input generation, ORCA execution
        and parsing all have to be right for this number to come out.
        """
        barrier = scan.torsional_barrier_kj_mol()
        assert barrier is not None
        assert barrier == pytest.approx(12.1, abs=1.5)

    def test_profile_minimum_is_at_the_staggered_conformer(self, scan) -> None:
        profile = scan.scan_profile_kj_mol()
        angle_at_min = min(profile, key=lambda p: p[1])[0]
        assert angle_at_min == pytest.approx(60.0, abs=1.0)

    def test_profile_is_relative_to_its_minimum(self, scan) -> None:
        profile = scan.scan_profile_kj_mol()
        assert min(energy for _, energy in profile) == pytest.approx(0.0, abs=1e-9)

    def test_profile_is_symmetric_about_the_minimum(self, scan) -> None:
        """Ethane's threefold barrier is symmetric: 0 and 120 degrees must match."""
        profile = dict(scan.scan_profile_kj_mol())
        assert profile[0.0] == pytest.approx(profile[120.0], abs=0.05)


# ==========================================================================
# The case that matters most
# ==========================================================================
class TestExitZeroButFailed:
    def test_exit_code_zero_is_not_success(self, failed_scan) -> None:
        """ORCA exited 0 on this job. It still failed."""
        assert failed_scan.status is QMStatus.FAILED_TERMINATION
        assert failed_scan.succeeded is False

    def test_error_termination_detected(self, failed_scan) -> None:
        assert failed_scan.error_termination is True
        assert failed_scan.normal_termination is False

    def test_non_convergence_is_reported(self, failed_scan) -> None:
        assert failed_scan.geometry_converged is False
        assert any("maximum number of cycles" in d for d in failed_scan.diagnostics)

    def test_an_energy_is_present_but_the_job_still_failed(self, failed_scan) -> None:
        """A parseable energy must not rescue a failed job."""
        assert failed_scan.final_energy_hartree is not None
        assert failed_scan.succeeded is False


# ==========================================================================
# Malformed and hostile input
# ==========================================================================
class TestMalformedOutput:
    def test_empty_output(self) -> None:
        result = parse_orca_output("")
        assert result.status is QMStatus.UNPARSEABLE
        assert result.succeeded is False

    def test_whitespace_only_output(self) -> None:
        assert parse_orca_output("   \n\n  ").status is QMStatus.UNPARSEABLE

    def test_not_an_orca_log(self) -> None:
        result = parse_orca_output("Segmentation fault (core dumped)\n")
        assert result.status is QMStatus.UNPARSEABLE
        assert any("does not look like" in d for d in result.diagnostics)

    def test_truncated_output_is_incomplete_not_complete(self) -> None:
        """A log that stops mid-run has no termination banner and must not pass."""
        text = read_output_text(FIXTURES / "h2.out.gz")
        truncated = text[: len(text) // 2]
        result = parse_orca_output(truncated)
        assert result.status is QMStatus.INCOMPLETE
        assert result.succeeded is False

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ResponseFormatError):
            parse_orca_file(tmp_path / "absent.out")

    def test_expecting_frequencies_that_are_absent_fails(self) -> None:
        text = read_output_text(FIXTURES / "opt.out.gz")
        result = parse_orca_output(text, expect_geometry=True, expect_frequencies=True)
        assert result.status is QMStatus.FAILED_SCIENTIFICALLY
        assert any("frequencies were expected" in d for d in result.diagnostics)

    def test_expecting_geometry_from_a_single_point_fails(self) -> None:
        text = read_output_text(FIXTURES / "h2.out.gz")
        result = parse_orca_output(text, expect_geometry=True)
        assert result.status is QMStatus.FAILED_SCIENTIFICALLY

    def test_gzip_and_plain_text_parse_identically(self, tmp_path: Path) -> None:
        text = read_output_text(FIXTURES / "h2.out.gz")
        plain = tmp_path / "h2.out"
        plain.write_text(text)
        assert parse_orca_file(plain).final_energy_hartree == parse_orca_file(
            FIXTURES / "h2.out.gz"
        ).final_energy_hartree
