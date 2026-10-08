"""The OpenFF backend and its isolated environment.

OpenFF lives in `.paramenv`, not the engine's `.venv`, so these tests exercise a
subprocess bridge. They skip explicitly when that environment is absent rather than
mocking it, because a mocked parameterization proves nothing about whether OpenFF can
actually type a polymer.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from polymer_engine.core.models import GateStatus
from polymer_engine.parameterization.backends.openff import (
    PARAMENV,
    WORKER,
    OpenFFBackend,
)
from polymer_engine.parameterization.capability import CapabilityState
from polymer_engine.parameterization.models import (
    ParameterizationRequest,
    ParameterizationState,
    PropertyClass,
)

requires_openff = pytest.mark.skipif(
    not PARAMENV.is_file(),
    reason=(f"isolated OpenFF environment not found at {PARAMENV}; create it with "
            f"mamba create -p .paramenv -c conda-forge openff-toolkit openff-interchange"),
)
requires_gromacs = pytest.mark.skipif(
    shutil.which("gmx") is None,
    reason="GROMACS 'gmx' is not on PATH; the topology cannot be preprocessed",
)


def polymer(name: str, smiles: str):
    from polymer_engine.polymer.records import build_record

    return build_record(name=name, repeat_unit_smiles=smiles, properties={},
                        source="test")


class TestEnvironmentIsolation:
    def test_the_worker_script_exists(self) -> None:
        assert WORKER.is_file()

    def test_openff_is_not_in_the_engine_environment(self) -> None:
        """The campaign's venv must not acquire this dependency tree."""
        import importlib.util

        assert importlib.util.find_spec("openff") is None, (
            "OpenFF leaked into the engine environment; it belongs in .paramenv"
        )

    def test_an_absent_environment_reports_unavailable_not_blocked(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """'Not installed' and 'cannot type this chemistry' are different answers."""
        monkeypatch.setattr(
            "polymer_engine.parameterization.backends.openff.PARAMENV",
            tmp_path / "absent" / "python",
        )
        backend = OpenFFBackend()
        assert backend.capabilities()["available"] is False
        assessment = backend.assess(polymer("polyethylene", "*CC*"))
        assert assessment.state is CapabilityState.UNAVAILABLE


@requires_openff
class TestWorkerBridge:
    def test_capabilities_are_measured_from_the_real_environment(self) -> None:
        capabilities = OpenFFBackend().capabilities()
        assert capabilities["available"] is True
        assert capabilities["version"], "toolkit version should be reported"
        assert capabilities["charge_methods"], "a charge model is required"

    def test_the_charge_model_is_named_not_implicit(self) -> None:
        """The charge model is part of the force field and must be recorded."""
        capabilities = OpenFFBackend().capabilities()
        assert capabilities["charge_model"], "the NAGL model must be named explicitly"

    def test_an_unknown_worker_action_fails_cleanly(self) -> None:
        proc = subprocess.run(
            [str(PARAMENV), str(WORKER), '{"action": "nonsense"}'],
            capture_output=True, text=True, timeout=120, check=False,
        )
        assert proc.returncode != 0
        assert "unknown action" in proc.stdout

    def test_a_non_am1bcc_charge_model_is_refused(self, tmp_path: Path) -> None:
        """Pairing Sage with Gasteiger would silently change the force field."""
        import json

        proc = subprocess.run(
            [str(PARAMENV), str(WORKER), json.dumps({
                "action": "parameterize", "smiles": "CCO",
                "charge_method": "gasteiger", "prefix": str(tmp_path / "x"),
            })],
            capture_output=True, text=True, timeout=300, check=False,
        )
        assert "silently change the force field" in proc.stdout


@requires_openff
class TestChemistryCoverage:
    @pytest.mark.parametrize(
        ("name", "smiles"),
        [
            ("poly(lactic acid)", "*OC(C)C(=O)*"),
            ("nylon-6", "*NCCCCCC(=O)*"),
            ("poly(ethylene oxide)", "*CCO*"),
            ("poly(vinyl chloride)", "*CC(Cl)*"),
            ("polystyrene", "*CC(*)c1ccccc1"),
            ("polyacrylonitrile", "*CC(*)C#N"),
        ],
    )
    def test_previously_blocked_chemistry_is_now_typable(self, name, smiles) -> None:
        """Each of these is BLOCKED under the local OPLS table."""
        assessment = OpenFFBackend().assess(polymer(name, smiles))
        assert assessment.state is CapabilityState.PARAMETERIZATION_AVAILABLE, (
            f"{name}: {assessment.reason}"
        )
        assert assessment.usable is True

    def test_typing_is_not_reported_as_system_build(self) -> None:
        """Coverage of an oligomer is not validation for a melt."""
        assessment = OpenFFBackend().assess(polymer("poly(lactic acid)", "*OC(C)C(=O)*"))
        assert assessment.state is not CapabilityState.SYSTEM_BUILD_AVAILABLE
        assert "not melt validation" in assessment.reason


@requires_openff
class TestRealParameterization:
    @pytest.fixture
    def result(self, tmp_path: Path):
        backend = OpenFFBackend()
        request = ParameterizationRequest(
            polymer_id="pol_test_pla", polymer_name="poly(lactic acid)",
            repeat_unit_smiles="*OC(C)C(=O)*",
            property_class=PropertyClass.BULK_DENSITY,
            degree_of_polymerization=3, n_chains=1, workdir=str(tmp_path),
        )
        return backend, backend.parameterize(request)

    def test_a_real_topology_is_produced(self, result) -> None:
        _backend, produced = result
        assert produced.state is ParameterizationState.PARAMETERIZED
        assert Path(produced.topology_path).is_file()
        assert Path(produced.coordinate_path).is_file()

    def test_the_charges_are_neutral(self, result) -> None:
        _backend, produced = result
        assert produced.net_charge == pytest.approx(0.0, abs=1e-6)

    def test_provenance_names_the_force_field_and_charge_model(self, result) -> None:
        """Both are part of the force field; neither may be implicit."""
        _backend, produced = result
        assert "openff-2.2.0" in produced.provenance["offxml"]
        assert "am1bcc" in produced.provenance["charge_method"]
        assert produced.provenance["oligomer_smiles"]

    def test_every_artifact_is_hashed(self, result) -> None:
        _backend, produced = result
        assert produced.artifacts
        assert all(len(digest) == 64 for digest in produced.artifacts.values())

    def test_validation_passes_completeness_and_charges(self, result) -> None:
        backend, produced = result
        validation = backend.validate(produced)
        assert validation.completeness.status is GateStatus.PASS
        assert validation.charges.status is GateStatus.PASS
        assert validation.promotable is True

    def test_the_topology_declares_every_implied_interaction(self, result) -> None:
        from polymer_engine.parameterization.completeness import analyse_topology

        _backend, produced = result
        report = analyse_topology(produced.topology_path)
        assert report.n_angles >= (report.expected_angles or 0)
        assert report.n_dihedrals >= (report.expected_dihedrals or 0)
        assert report.missing_sections == []

    @requires_gromacs
    @pytest.mark.slow
    def test_real_grompp_accepts_it_without_maxwarn(self, result, tmp_path: Path) -> None:
        """The end of the chain: a previously blocked polymer preprocesses for real."""
        _backend, produced = result
        workdir = Path(produced.topology_path).parent
        (workdir / "em.mdp").write_text(
            "integrator = steep\nnsteps = 0\ncutoff-scheme = Verlet\n"
            "coulombtype = PME\nrvdw = 1.0\nrcoulomb = 1.0\nrlist = 1.0\npbc = xyz\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            ["gmx", "grompp", "-f", "em.mdp", "-c", Path(produced.coordinate_path).name,
             "-p", Path(produced.topology_path).name, "-o", "em.tpr"],
            cwd=workdir, capture_output=True, text=True, timeout=300, check=False,
        )
        assert proc.returncode == 0, proc.stderr[-800:]
        assert "WARNING" not in proc.stderr.upper() or "0 WARNING" in proc.stderr.upper()
