"""Trajectory analysis verified against a fixture with analytically known answers.

The synthetic system is a rigid 5-atom chain at x = 0..4 angstrom (unit spacing) plus
one tracer atom that moves exactly 1 angstrom per frame.  Every expected value below
is derived by hand, so these tests check numbers rather than merely that code ran.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from polymer_engine.analysis.md import (
    AMU_PER_A3_TO_KG_PER_M3,
    AnalysisSpec,
    com_distance,
    contacts,
    density,
    end_to_end_distance,
    mdanalysis_available,
    mean_squared_displacement,
    production_summary,
    radius_of_gyration,
    rdf,
    volume,
)
from polymer_engine.core.errors import InsufficientDataError, ScientificError
from polymer_engine.core.models import Determination
from tests.markers import requires_mdanalysis

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "trajectories"
BOX_A = 50.0
BOX_VOLUME_A3 = BOX_A**3


@pytest.fixture(scope="module")
def trajectory_files() -> dict[str, Path]:
    if not mdanalysis_available():
        pytest.skip("MDAnalysis is not installed; trajectory fixtures cannot be built")
    topology, trajectory = FIXTURES / "chain.pdb", FIXTURES / "chain.xtc"
    if not (topology.exists() and trajectory.exists()):
        import sys

        sys.path.insert(0, str(FIXTURES.parent))
        from make_trajectory import build

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build(FIXTURES)
    return {"topology": topology, "trajectory": trajectory}


@pytest.fixture
def chain(trajectory_files) -> AnalysisSpec:
    return AnalysisSpec(
        topology=str(trajectory_files["topology"]),
        trajectory=str(trajectory_files["trajectory"]),
        selection="resname POL",
    )


@pytest.fixture
def tracer(trajectory_files) -> AnalysisSpec:
    return AnalysisSpec(
        topology=str(trajectory_files["topology"]),
        trajectory=str(trajectory_files["trajectory"]),
        selection="resname TRC",
    )


@requires_mdanalysis
class TestStructuralObservables:
    def test_radius_of_gyration_matches_the_analytic_value(self, chain: AnalysisSpec) -> None:
        """Five unit masses at x = 0..4 give Rg^2 = 2 A^2, so Rg = sqrt(2) A = 0.1414 nm."""
        result = radius_of_gyration(chain)
        expected_nm = math.sqrt(2.0) * 0.1
        assert result.summary.value == pytest.approx(expected_nm, rel=1e-6)
        assert result.units == "nm"

    def test_radius_of_gyration_is_returned_in_nm_not_angstrom(self, chain: AnalysisSpec) -> None:
        """A silent unit slip here would be a factor of ten."""
        assert radius_of_gyration(chain).summary.value < 1.0

    def test_end_to_end_distance_matches_the_analytic_value(self, chain: AnalysisSpec) -> None:
        result = end_to_end_distance(chain)
        assert result.summary.value == pytest.approx(0.4, rel=1e-6)
        assert result.units == "nm"
        assert result.extra["first_atom_index"] == 0
        assert result.extra["last_atom_index"] == 4

    def test_rigid_chain_has_zero_variance(self, chain: AnalysisSpec) -> None:
        assert radius_of_gyration(chain).summary.uncertainty == pytest.approx(0.0, abs=1e-12)

    def test_volume_matches_the_box(self, chain: AnalysisSpec) -> None:
        result = volume(chain)
        assert result.summary.value == pytest.approx(BOX_VOLUME_A3 * 1e-3, rel=1e-6)

    def test_com_distance_between_chain_and_tracer(self, trajectory_files) -> None:
        """Chain COM is at x = 2; the tracer starts at x = 20 and advances 1 A/frame."""
        spec = AnalysisSpec(
            topology=str(trajectory_files["topology"]),
            trajectory=str(trajectory_files["trajectory"]),
            selection="resname POL",
            selection_b="resname TRC",
            stop=1,
        )
        result = com_distance(spec)
        separation = math.sqrt((20.0 - 2.0) ** 2 + 10.0**2 + 10.0**2) * 0.1
        assert result.summary.value == pytest.approx(separation, rel=1e-5)


@requires_mdanalysis
class TestDensity:
    def test_unit_conversion_arithmetic_is_exact(self, chain: AnalysisSpec) -> None:
        spec = AnalysisSpec(topology=chain.topology, trajectory=chain.trajectory, selection="all")
        result = density(spec)
        mass = result.extra["total_mass_amu"]
        assert result.summary.value == pytest.approx(mass / BOX_VOLUME_A3 * AMU_PER_A3_TO_KG_PER_M3)

    def test_guessed_masses_are_flagged_not_hidden(self, chain: AnalysisSpec) -> None:
        """A .pdb carries no masses, so any density from it is approximate."""
        spec = AnalysisSpec(topology=chain.topology, trajectory=chain.trajectory, selection="all")
        result = density(spec)
        assert result.extra["masses_guessed"] is True
        assert "guessed" in (result.summary.notes or "")


@requires_mdanalysis
class TestDynamics:
    def test_msd_grows_quadratically_for_ballistic_motion(self, tracer: AnalysisSpec) -> None:
        """The tracer moves 1 A = 0.1 nm per frame, so MSD(lag) = (0.1 * lag)^2."""
        result = mean_squared_displacement(tracer)
        assert result.msd_nm2[0] == pytest.approx(0.01, rel=1e-6)
        assert result.msd_nm2[9] == pytest.approx(1.00, rel=1e-6)
        assert result.msd_nm2[4] == pytest.approx(0.25, rel=1e-6)

    def test_lag_times_use_the_frame_spacing(self, tracer: AnalysisSpec) -> None:
        result = mean_squared_displacement(tracer)
        assert result.lag_ps[0] == pytest.approx(1.0)
        assert result.lag_ps[9] == pytest.approx(10.0)

    def test_ballistic_motion_is_refused_a_diffusion_coefficient(self, tracer: AnalysisSpec) -> None:
        """Constant-velocity drift is not diffusion; quoting D here would be wrong."""
        result = mean_squared_displacement(tracer)
        coefficient = result.diffusion_coefficient()
        assert coefficient.value is None
        assert coefficient.determination is Determination.INSUFFICIENT_DATA
        assert "ballistic" in (coefficient.notes or "")
        assert "t^2.0" in (coefficient.notes or ""), "the scaling exponent must be reported"

    def test_true_random_walk_recovers_the_diffusion_coefficient(self, tmp_path: Path) -> None:
        """A 3-D random walk with known step variance has D = var_step / (6 * dt)."""
        from polymer_engine.analysis.md import MsdResult

        rng = np.random.default_rng(5)
        n_frames, n_atoms, step_sd_nm = 4000, 40, 0.05
        steps = rng.normal(0.0, step_sd_nm, size=(n_frames, n_atoms, 3))
        coords_nm = np.cumsum(steps, axis=0)

        # Build the MSD directly from the known walk, bypassing file IO.
        lags = np.arange(1, 400)
        msd = np.array(
            [float(((coords_nm[lag:] - coords_nm[:-lag]) ** 2).sum(axis=-1).mean()) for lag in lags]
        )
        result = MsdResult(lag_ps=lags.astype(float), msd_nm2=msd, spec=AnalysisSpec("t", "x"))

        coefficient = result.diffusion_coefficient()
        assert coefficient.value is not None, coefficient.notes
        # Each step adds 3 * sd^2 to the MSD per unit time, so D = 3*sd^2 / 6.
        expected = 3 * step_sd_nm**2 / 6.0
        assert coefficient.value == pytest.approx(expected, rel=0.1)

    def test_msd_needs_enough_frames(self, tracer: AnalysisSpec) -> None:
        spec = AnalysisSpec(
            topology=tracer.topology, trajectory=tracer.trajectory, selection=tracer.selection, stop=2
        )
        with pytest.raises(InsufficientDataError):
            mean_squared_displacement(spec)


@requires_mdanalysis
class TestPairwise:
    def test_rdf_returns_a_normalised_function(self, chain: AnalysisSpec) -> None:
        result = rdf(chain, r_max_nm=1.0, bins=50)
        assert result is not None
        assert result.r_nm.size == 50
        assert np.all(result.g_r >= 0)
        assert result.n_frames == 40

    def test_rdf_first_peak_is_at_the_bond_length(self, chain: AnalysisSpec) -> None:
        """Neighbouring chain atoms sit 1 A = 0.1 nm apart."""
        result = rdf(chain, r_max_nm=0.6, bins=60)
        assert result.first_peak().value == pytest.approx(0.1, abs=0.015)

    def test_rdf_rejects_a_degenerate_range(self, chain: AnalysisSpec) -> None:
        with pytest.raises(ScientificError):
            rdf(chain, r_max_nm=0.0)

    def test_contacts_counts_pairs_within_the_cutoff(self, trajectory_files) -> None:
        """Every chain atom is within 0.5 nm of at least one neighbour; the tracer is far away."""
        spec = AnalysisSpec(
            topology=str(trajectory_files["topology"]),
            trajectory=str(trajectory_files["trajectory"]),
            selection="resname POL",
            selection_b="resname TRC",
            stop=1,
        )
        assert contacts(spec, cutoff_nm=0.5).summary.value == pytest.approx(0.0)

    def test_contacts_requires_a_second_selection(self, chain: AnalysisSpec) -> None:
        with pytest.raises(ScientificError, match="second selection"):
            contacts(chain)

    def test_contacts_rejects_a_non_positive_cutoff(self, trajectory_files) -> None:
        spec = AnalysisSpec(
            topology=str(trajectory_files["topology"]),
            trajectory=str(trajectory_files["trajectory"]),
            selection="resname POL",
            selection_b="resname TRC",
        )
        with pytest.raises(ScientificError):
            contacts(spec, cutoff_nm=-1.0)


@requires_mdanalysis
class TestSpecHandling:
    def test_empty_selection_is_an_explicit_error(self, chain: AnalysisSpec) -> None:
        spec = AnalysisSpec(topology=chain.topology, trajectory=chain.trajectory, selection="resname NOPE")
        with pytest.raises(ScientificError, match="matched no atoms"):
            radius_of_gyration(spec)

    def test_missing_file_is_an_explicit_error(self, chain: AnalysisSpec) -> None:
        spec = AnalysisSpec(topology=chain.topology, trajectory="/nonexistent/traj.xtc")
        with pytest.raises(ScientificError, match="does not exist"):
            radius_of_gyration(spec)

    def test_stride_reduces_the_frame_count(self, chain: AnalysisSpec) -> None:
        strided = AnalysisSpec(
            topology=chain.topology, trajectory=chain.trajectory, selection=chain.selection, stride=4
        )
        assert radius_of_gyration(strided).n_frames == 10

    def test_frame_range_is_honoured(self, chain: AnalysisSpec) -> None:
        windowed = AnalysisSpec(
            topology=chain.topology, trajectory=chain.trajectory, selection=chain.selection,
            start=10, stop=20,
        )
        assert radius_of_gyration(windowed).n_frames == 10

    def test_spec_is_recorded_in_the_result(self, chain: AnalysisSpec) -> None:
        payload = radius_of_gyration(chain).as_dict()
        assert payload["spec"]["selection"] == "resname POL"
        assert payload["units"] == "nm"

    def test_production_summary_trims_the_transient(self, chain: AnalysisSpec) -> None:
        result = radius_of_gyration(chain)
        assert production_summary(result).value == pytest.approx(result.summary.value, rel=1e-9)


def test_analyses_report_unsupported_without_mdanalysis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without MDAnalysis the engine says so rather than fabricating a value."""
    monkeypatch.setattr("polymer_engine.analysis.md.mdanalysis_available", lambda: False)
    spec = AnalysisSpec(topology="x.pdb", trajectory="x.xtc")
    result = radius_of_gyration(spec)
    assert result.summary.determination is Determination.UNSUPPORTED
    assert result.summary.value is None
