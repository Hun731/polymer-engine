"""System construction abstraction and force-field decision recording."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import pytest

from polymer_engine.core.errors import ScientificError
from polymer_engine.core.models import Determination
from polymer_engine.core.provenance import ProvenanceGraph
from polymer_engine.polymer.taxonomy import PolymerFamily
from polymer_engine.simulation.builder import (
    BuildCapability,
    BuildStatus,
    CharmmGuiImportBackend,
    LocalDirectoryBackend,
    SystemBuilder,
    SystemBuildRequest,
    default_builder,
)
from polymer_engine.simulation.forcefield import (
    Confidence,
    ForceFieldAdvisor,
    ForceFieldEvidence,
    require_ready_force_field,
)

VALID_SYSTEM = Path(__file__).resolve().parents[1] / "fixtures" / "systems" / "valid_system"


def request(**kwargs) -> SystemBuildRequest:
    base = {"polymer_id": "pol_test", "force_field": "CHARMM36"}
    base.update(kwargs)
    return SystemBuildRequest(**base)


# ==========================================================================
# Request validation
# ==========================================================================
class TestBuildRequest:
    def test_force_field_is_required(self) -> None:
        assert any("force_field" in p for p in request(force_field="").validate())

    def test_unspecified_force_field_is_rejected(self) -> None:
        """'UNSPECIFIED' is honest as a config placeholder but cannot build a system."""
        assert any("force_field" in p for p in request(force_field="UNSPECIFIED").validate())

    def test_solvation_requires_a_water_model(self) -> None:
        assert any("water model" in p for p in request(solvate=True).validate())

    def test_negative_box_is_rejected(self) -> None:
        assert any("box" in p for p in request(box_nm=(-1.0, 2.0, 2.0)).validate())

    def test_valid_request_has_no_problems(self) -> None:
        assert request(source_directory=str(VALID_SYSTEM)).validate() == []

    def test_fingerprint_is_deterministic(self) -> None:
        assert request().fingerprint() == request().fingerprint()

    def test_force_field_changes_the_fingerprint(self) -> None:
        assert request(force_field="OPLS-AA").fingerprint() != request(force_field="CHARMM36").fingerprint()


# ==========================================================================
# Local directory backend
# ==========================================================================
class TestLocalDirectoryBackend:
    def test_imports_a_valid_system(self, tmp_path: Path) -> None:
        graph = ProvenanceGraph()
        result = LocalDirectoryBackend().build(
            request(source_directory=str(VALID_SYSTEM)), tmp_path, graph=graph
        )
        assert result.status is BuildStatus.IMPORTED
        assert result.usable is True
        assert result.determination is Determination.KNOWN
        assert result.force_field == "CHARMM36"
        assert result.manifest is not None
        assert result.artifact_id in graph

    def test_a_broken_system_is_not_usable(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken"
        shutil.copytree(VALID_SYSTEM, broken)
        (broken / "topol.top").unlink()
        result = LocalDirectoryBackend().build(request(source_directory=str(broken)), tmp_path / "out")
        assert result.status is BuildStatus.FAILED
        assert result.usable is False
        assert result.diagnostics

    def test_missing_directory_fails_cleanly(self, tmp_path: Path) -> None:
        result = LocalDirectoryBackend().build(
            request(source_directory=str(tmp_path / "absent")), tmp_path / "out"
        )
        assert result.status is BuildStatus.FAILED
        assert "not found" in result.diagnostics[0]

    def test_imports_an_archive(self, tmp_path: Path) -> None:
        archive = tmp_path / "system.tgz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(VALID_SYSTEM, arcname="job")
        result = LocalDirectoryBackend().build(
            request(source_archive=str(archive)), tmp_path / "out"
        )
        assert result.usable is True

    def test_an_unsafe_archive_fails_rather_than_extracting(self, tmp_path: Path) -> None:
        import io

        archive = tmp_path / "evil.tgz"
        with tarfile.open(archive, "w:gz") as tf:
            info = tarfile.TarInfo("../escaped.txt")
            info.size = 3
            tf.addfile(info, io.BytesIO(b"bad"))
        result = LocalDirectoryBackend().build(request(source_archive=str(archive)), tmp_path / "out")
        assert result.status is BuildStatus.FAILED
        assert not (tmp_path / "escaped.txt").exists()

    def test_an_invalid_request_is_refused_before_any_work(self, tmp_path: Path) -> None:
        result = LocalDirectoryBackend().build(
            request(force_field="UNSPECIFIED", source_directory=str(VALID_SYSTEM)), tmp_path
        )
        assert result.status is BuildStatus.REQUIRES_EXPERT_DECISION


# ==========================================================================
# CHARMM-GUI backend
# ==========================================================================
class TestCharmmGuiBackend:
    def test_submission_is_declared_unsupported(self, tmp_path: Path) -> None:
        """CHARMM-GUI publishes no submission endpoint; the engine does not guess one."""
        result = CharmmGuiImportBackend().build(request(), tmp_path)
        assert result.status is BuildStatus.UNSUPPORTED
        assert result.determination is Determination.UNSUPPORTED
        assert "no job-submission endpoint" in result.diagnostics[0]
        assert any("web interface" in action for action in result.required_actions)

    def test_submission_is_not_in_the_capability_set(self) -> None:
        assert BuildCapability.SUBMIT_REMOTE_JOB not in CharmmGuiImportBackend().capabilities

    def test_a_job_id_without_a_provider_asks_for_credentials(self, tmp_path: Path) -> None:
        result = CharmmGuiImportBackend().build(request(external_job_id="1234"), tmp_path)
        assert result.status is BuildStatus.REQUIRES_INPUT
        assert any("CHARMM_GUI" in action for action in result.required_actions)

    def test_a_downloaded_archive_is_imported(self, tmp_path: Path) -> None:
        archive = tmp_path / "job.tgz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(VALID_SYSTEM, arcname="charmm-gui-1234")
        result = CharmmGuiImportBackend().build(
            request(source_archive=str(archive)), tmp_path / "out"
        )
        assert result.usable is True
        assert "CHARMM-GUI" in (result.parameter_source or "")

    def test_a_failed_download_is_reported(self, tmp_path: Path) -> None:
        class FailingProvider:
            def download_job(self, job_id, destination):
                from polymer_engine.providers.base import ProviderResult

                return ProviderResult("charmm_gui", "job_download", False, error="HTTP 500")

        result = CharmmGuiImportBackend(FailingProvider()).build(
            request(external_job_id="1234"), tmp_path
        )
        assert result.status is BuildStatus.FAILED
        assert "HTTP 500" in result.diagnostics[0]


# ==========================================================================
# Builder dispatch
# ==========================================================================
class TestSystemBuilder:
    def test_default_builder_has_both_import_backends(self) -> None:
        assert set(default_builder().backends()) == {"local_directory", "charmm_gui_import"}

    def test_building_from_scratch_is_refused_with_guidance(self, tmp_path: Path) -> None:
        """The engine validates systems; it does not construct them."""
        builder = SystemBuilder([LocalDirectoryBackend()])
        result = builder.build(request(repeat_unit_smiles="*CC*"), tmp_path)
        assert result.status is BuildStatus.REQUIRES_INPUT
        assert any("external tool" in action for action in result.required_actions)

    def test_a_backend_can_be_named_explicitly(self, tmp_path: Path) -> None:
        result = default_builder().build(
            request(source_directory=str(VALID_SYSTEM)), tmp_path, backend="local_directory"
        )
        assert result.backend == "local_directory"

    def test_an_unknown_backend_raises(self, tmp_path: Path) -> None:
        from polymer_engine.core.errors import PolymerEngineError

        with pytest.raises(PolymerEngineError):
            default_builder().build(request(), tmp_path, backend="nonexistent")

    def test_a_new_backend_can_be_registered(self, tmp_path: Path) -> None:
        from polymer_engine.simulation.builder import SystemBuilderBackend

        class InHouse(SystemBuilderBackend):
            name = "in_house"
            capabilities = frozenset({BuildCapability.BUILD_FROM_SMILES})

            def build(self, request, destination, *, graph=None):
                return self._reject(request, BuildStatus.UNSUPPORTED, "demo backend")

        builder = default_builder()
        builder.register(InHouse())
        assert "in_house" in builder.backends()
        assert builder.capabilities()["in_house"] == ["build_from_smiles"]

    def test_capabilities_are_reported_per_backend(self) -> None:
        capabilities = default_builder().capabilities()
        assert "import_directory" in capabilities["local_directory"]
        assert "submit_remote_job" not in capabilities["charmm_gui_import"]


# ==========================================================================
# Force-field strategy
# ==========================================================================
class TestForceFieldAdvisor:
    def test_many_candidates_means_no_automatic_choice(self) -> None:
        """Five force fields cover polyolefins and they are not interchangeable."""
        strategy = ForceFieldAdvisor().propose("p1", PolymerFamily.POLYOLEFIN)
        assert strategy.confidence is Confidence.REQUIRES_EXPERT_DECISION
        assert strategy.selected_force_field is None
        assert strategy.ready_to_simulate is False

    def test_a_requested_field_within_coverage_is_supported(self) -> None:
        strategy = ForceFieldAdvisor().propose(
            "p1", PolymerFamily.POLYOLEFIN, requested="CHARMM36"
        )
        assert strategy.confidence is Confidence.SUPPORTED
        assert strategy.selected_force_field == "CHARMM36"
        assert "published coverage" in strategy.justification

    def test_supported_is_not_enough_to_simulate_unattended(self) -> None:
        """Coverage is not validation."""
        strategy = ForceFieldAdvisor().propose("p1", PolymerFamily.POLYOLEFIN, requested="CHARMM36")
        assert strategy.ready_to_simulate is False

    def test_a_field_outside_its_coverage_is_uncertain(self) -> None:
        strategy = ForceFieldAdvisor().propose(
            "p1", PolymerFamily.POLYSILOXANE, requested="CHARMM36"
        )
        assert strategy.confidence is Confidence.UNCERTAIN
        assert any("does not claim coverage" in w for w in strategy.warnings)

    def test_an_unknown_field_is_recorded_but_uncertain(self) -> None:
        strategy = ForceFieldAdvisor().propose(
            "p1", PolymerFamily.POLYOLEFIN, requested="MyCustomFF"
        )
        assert strategy.confidence is Confidence.UNCERTAIN
        assert strategy.selected_force_field == "MyCustomFF"

    def test_unclassified_family_always_requires_a_human(self) -> None:
        strategy = ForceFieldAdvisor().propose(
            "p1", PolymerFamily.UNCLASSIFIED, requested="OPLS-AA"
        )
        assert strategy.confidence is Confidence.REQUIRES_EXPERT_DECISION

    def test_a_sole_covering_field_is_selected_but_only_supported(self) -> None:
        strategy = ForceFieldAdvisor().propose("p1", PolymerFamily.POLYSILOXANE)
        assert strategy.selected_force_field == "PCFF"
        assert strategy.confidence is Confidence.SUPPORTED

    def test_passing_qm_validation_promotes_to_known(self) -> None:
        strategy = ForceFieldAdvisor().propose(
            "p1", PolymerFamily.POLYOLEFIN, requested="CHARMM36",
            qm_validation={"passed": True},
        )
        assert strategy.confidence is Confidence.KNOWN
        assert strategy.ready_to_simulate is True

    def test_failing_qm_validation_blocks_promotion(self) -> None:
        strategy = ForceFieldAdvisor().propose(
            "p1", PolymerFamily.POLYOLEFIN, requested="CHARMM36",
            qm_validation={"passed": False},
        )
        assert strategy.confidence is Confidence.UNCERTAIN
        assert strategy.ready_to_simulate is False

    def test_contrary_evidence_blocks_promotion(self) -> None:
        strategy = ForceFieldAdvisor().propose(
            "p1", PolymerFamily.POLYOLEFIN, requested="CHARMM36",
            qm_validation={"passed": True},
            evidence=[
                ForceFieldEvidence("torsion", "barrier off by 8 kJ/mol", supports=False)
            ],
        )
        assert strategy.confidence is Confidence.UNCERTAIN
        assert strategy.ready_to_simulate is False

    def test_parameter_generation_requirement_is_warned_about(self) -> None:
        strategy = ForceFieldAdvisor().propose(
            "p1", PolymerFamily.POLYOLEFIN, requested="GAFF2"
        )
        assert any("parameter generation" in w for w in strategy.warnings)

    def test_require_ready_raises_for_an_unvalidated_choice(self) -> None:
        strategy = ForceFieldAdvisor().propose("p1", PolymerFamily.POLYOLEFIN, requested="CHARMM36")
        with pytest.raises(ScientificError, match="not validated"):
            require_ready_force_field(strategy)

    def test_require_ready_passes_for_a_validated_choice(self) -> None:
        strategy = ForceFieldAdvisor().propose(
            "p1", PolymerFamily.POLYOLEFIN, requested="CHARMM36", qm_validation={"passed": True}
        )
        require_ready_force_field(strategy)  # must not raise

    def test_strategy_serialises_its_full_reasoning(self) -> None:
        payload = ForceFieldAdvisor().propose(
            "p1", PolymerFamily.POLYESTER, requested="OPLS-AA"
        ).as_dict()
        assert payload["justification"]
        assert payload["compatible_force_fields"]
        assert payload["confidence"] == "SUPPORTED"
        assert payload["ready_to_simulate"] is False
