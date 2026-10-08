"""Integration: provider -> store, system -> campaign, campaign -> analysis, provenance lineage."""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest

from polymer_engine.core.config import load_config
from polymer_engine.core.models import (
    Action,
    ActionStatus,
    ExecutionState,
    GateStatus,
    Measurement,
    Observation,
)
from polymer_engine.core.provenance import ProvenanceGraph
from polymer_engine.db.store import Store
from polymer_engine.executors.registry import ExecutorRegistry
from polymer_engine.orchestrator.campaign import (
    CampaignBuilder,
    CampaignSpec,
    CampaignStatus,
    campaign_fingerprint,
    load_campaign,
    save_campaign,
)
from polymer_engine.providers.http import HttpClient
from polymer_engine.providers.pubchem import PubChemProvider
from polymer_engine.providers.testing import FixtureTransport
from polymer_engine.simulation.system import import_archive, import_directory

VALID_SYSTEM = Path(__file__).resolve().parents[1] / "fixtures" / "systems" / "valid_system"
PROVIDER_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "providers"


@pytest.fixture
def config(tmp_path: Path):
    cfg = load_config(
        discover=False,
        use_env=False,
        overrides={
            "paths": {"root": str(tmp_path)},
            "http": {"cache_enabled": False},
            "simulation": {"replicas": 3, "production_ns": 5.0, "force_field": "CHARMM36"},
        },
    )
    cfg.paths.ensure()
    return cfg


@pytest.fixture
def store(config):
    with Store(config.paths.resolved("database")) as s:
        yield s


# ==========================================================================
# Provider -> database
# ==========================================================================
class TestProviderToStore:
    def test_provider_records_become_observations_with_provenance(self, store) -> None:
        transport = FixtureTransport()
        transport.add_json("pubchem", json.loads((PROVIDER_FIXTURES / "pubchem_ethanol.json").read_text()))
        provider = PubChemProvider(HttpClient(transport=transport, sleeper=lambda s: None))
        result = provider.resolve_compound("ethanol")

        action = Action(kind="pubchem_resolve", title="Resolve ethanol")
        store.save_action(action)
        record = result.records[0]
        store.save_observation(
            Observation(
                action_id=action.id,
                measurement=Measurement(
                    name="molecular_weight", value=float(record["molecular_weight"]), units="g/mol"
                ),
            )
        )
        observations = store.list_observations(metric="molecular_weight")
        assert len(observations) == 1
        assert observations[0].measurement.value == pytest.approx(46.07)
        assert result.provenance["request"]["response_sha256"]

    def test_a_provider_failure_does_not_write_observations(self, store) -> None:
        from polymer_engine.providers.testing import StubResponse

        transport = FixtureTransport()
        transport.add("pubchem", StubResponse.json({}, status=500))
        provider = PubChemProvider(HttpClient(transport=transport, max_retries=0, sleeper=lambda s: None))
        result = provider.resolve_compound("ethanol")
        assert result.ok is False
        assert result.records == []
        assert store.list_observations() == []


# ==========================================================================
# System -> campaign
# ==========================================================================
class TestSystemToCampaign:
    def test_valid_system_yields_a_planned_campaign(self, config, store) -> None:
        graph = ProvenanceGraph()
        builder = CampaignBuilder(config, graph=graph, software={"gromacs": "2026.3"})
        spec = CampaignSpec.from_config(config, campaign_id="c1", polymer_id="pol_x")
        campaign = builder.create(spec)
        builder.attach_system(campaign, import_directory(VALID_SYSTEM, graph=graph))
        assert campaign.status is CampaignStatus.SYSTEM_VALIDATED
        builder.plan(campaign, gromacs_version="2026.3")
        assert campaign.status is CampaignStatus.PLANNED
        assert campaign.replicas.n_replicas == 3
        assert campaign.replicas.seeds_are_distinct()
        assert len(campaign.actions) == 4

    def test_invalid_system_blocks_the_campaign_and_refuses_planning(self, config, tmp_path) -> None:
        from polymer_engine.core.errors import SystemValidationError

        broken = tmp_path / "broken"
        shutil.copytree(VALID_SYSTEM, broken)
        (broken / "topol.top").unlink()

        builder = CampaignBuilder(config)
        spec = CampaignSpec.from_config(config, campaign_id="c2", polymer_id="pol_x")
        campaign = builder.create(spec)
        builder.attach_system(campaign, import_directory(broken))
        assert campaign.status is CampaignStatus.BLOCKED
        assert "topology" in campaign.blocked_reason.lower()
        with pytest.raises(SystemValidationError):
            builder.plan(campaign)

    def test_archive_import_feeds_a_campaign(self, config, tmp_path) -> None:
        archive = tmp_path / "job.tgz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(VALID_SYSTEM, arcname="charmm-gui-9999")
        graph = ProvenanceGraph()
        imported = import_archive(archive, tmp_path / "unpacked", graph=graph, source={"provider": "charmm_gui"})
        assert imported.usable

        builder = CampaignBuilder(config, graph=graph)
        campaign = builder.create(CampaignSpec.from_config(config, campaign_id="c3", polymer_id="pol_x"))
        builder.attach_system(campaign, imported)
        builder.plan(campaign)
        assert campaign.status is CampaignStatus.PLANNED

    def test_replica_inputs_are_grompp_shaped(self, config) -> None:
        from polymer_engine.simulation.mdp import parse_mdp

        builder = CampaignBuilder(config)
        campaign = builder.create(CampaignSpec.from_config(config, campaign_id="c4", polymer_id="pol_x"))
        builder.attach_system(campaign, import_directory(VALID_SYSTEM))
        builder.plan(campaign, gromacs_version="2026.3")

        for replica in campaign.replicas.replicas:
            directory = Path(replica.directory)
            for stage in ("em", "nvt", "npt", "prod"):
                assert (directory / f"{stage}.mdp").exists()
            assert (directory / "system.gro").exists()
            assert (directory / "topol.top").exists()
            nvt = parse_mdp((directory / "nvt.mdp").read_text())
            assert nvt["ref-t"] == str(config.simulation.temperature_k)
            assert nvt["gen-seed"] == str(replica.seeds["nvt"])


# ==========================================================================
# Campaign -> analysis
# ==========================================================================
def write_xvg(path: Path, values: np.ndarray, *, dt: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['@    title "Test"', '@    xaxis  label "Time (ps)"']
    lines += [f"{i * dt:.3f} {v:.6f}" for i, v in enumerate(values)]
    path.write_text("\n".join(lines) + "\n")


class TestCampaignToAnalysis:
    def test_converged_agreeing_replicas_pass(self, config) -> None:
        rng = np.random.default_rng(1)
        workdir = config.paths.resolved("workspace_dir") / "analysis_ok"
        dirs = []
        for i in range(3):
            directory = workdir / f"replica_{i + 1:02d}"
            write_xvg(directory / "density.xvg", 1000.0 + rng.normal(0, 1.5, 3000))
            write_xvg(directory / "temperature.xvg", 300.0 + rng.normal(0, 1.0, 3000))
            dirs.append(str(directory))

        action = Action(
            kind="analyze_replicas", title="analyse",
            inputs={"workdir": str(workdir), "replica_dirs": dirs, "metrics": ["density", "temperature"]},
        )
        result = ExecutorRegistry(config).get("analyze_replicas").run(action)
        assert result.report.status is GateStatus.PASS
        assert result.status is ActionStatus.SUCCEEDED
        combined = [o for o in result.observations if o.measurement.method.startswith("mean of independent")]
        assert combined
        assert combined[0].measurement.value == pytest.approx(1000.0, abs=1.0)

    def test_disagreeing_replicas_fail(self, config) -> None:
        rng = np.random.default_rng(2)
        workdir = config.paths.resolved("workspace_dir") / "analysis_bad"
        dirs = []
        for i, centre in enumerate((980.0, 1000.0, 1080.0)):
            directory = workdir / f"replica_{i + 1:02d}"
            write_xvg(directory / "density.xvg", centre + rng.normal(0, 1.0, 3000))
            dirs.append(str(directory))
        action = Action(
            kind="analyze_replicas", title="analyse",
            inputs={"workdir": str(workdir), "replica_dirs": dirs, "metrics": ["density"]},
        )
        result = ExecutorRegistry(config).get("analyze_replicas").run(action)
        assert result.report.status is GateStatus.FAIL
        assert any("disagree" in g.message for g in result.report.gates)

    def test_a_replica_with_no_data_is_not_silently_skipped(self, config) -> None:
        rng = np.random.default_rng(3)
        workdir = config.paths.resolved("workspace_dir") / "analysis_missing"
        dirs = []
        for i in range(3):
            directory = workdir / f"replica_{i + 1:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            if i < 2:
                write_xvg(directory / "density.xvg", 1000.0 + rng.normal(0, 1.0, 2000))
            dirs.append(str(directory))
        action = Action(
            kind="analyze_replicas", title="analyse",
            inputs={"workdir": str(workdir), "replica_dirs": dirs, "metrics": ["density"]},
        )
        result = ExecutorRegistry(config).get("analyze_replicas").run(action)
        assert result.report.status is GateStatus.FAIL
        assert any("produced no density data" in g.message for g in result.report.gates)

    def test_two_replicas_are_inconclusive_not_passing(self, config) -> None:
        rng = np.random.default_rng(4)
        workdir = config.paths.resolved("workspace_dir") / "analysis_two"
        dirs = []
        for i in range(2):
            directory = workdir / f"replica_{i + 1:02d}"
            write_xvg(directory / "density.xvg", 1000.0 + rng.normal(0, 1.0, 3000))
            dirs.append(str(directory))
        action = Action(
            kind="analyze_replicas", title="analyse",
            inputs={"workdir": str(workdir), "replica_dirs": dirs, "metrics": ["density"]},
        )
        result = ExecutorRegistry(config).get("analyze_replicas").run(action)
        assert result.report.status is GateStatus.INCONCLUSIVE
        assert result.report.promotable is False


# ==========================================================================
# Persistence and resume
# ==========================================================================
class TestResume:
    def test_a_campaign_survives_a_reload(self, config, store) -> None:
        builder = CampaignBuilder(config, software={"gromacs": "2026.3"})
        campaign = builder.create(CampaignSpec.from_config(config, campaign_id="resume1", polymer_id="pol_x"))
        builder.attach_system(campaign, import_directory(VALID_SYSTEM))
        builder.plan(campaign)
        save_campaign(store, campaign)

        restored = load_campaign(store, "resume1")
        assert restored is not None
        assert restored.status is CampaignStatus.PLANNED
        assert len(restored.actions) == len(campaign.actions)
        assert restored.replicas is not None
        assert restored.replicas.n_replicas == campaign.replicas.n_replicas

    def test_reloading_preserves_the_random_seeds(self, config, store) -> None:
        """A lost seed record makes a campaign irreproducible."""
        builder = CampaignBuilder(config)
        campaign = builder.create(CampaignSpec.from_config(config, campaign_id="resume2", polymer_id="pol_x"))
        builder.attach_system(campaign, import_directory(VALID_SYSTEM))
        builder.plan(campaign)
        save_campaign(store, campaign)
        original = campaign.manifest()["random_seeds"]

        restored = load_campaign(store, "resume2")
        save_campaign(store, restored)
        assert store.get_campaign("resume2")["random_seeds"] == original

    def test_execution_records_persist_their_history(self, config, store) -> None:
        from polymer_engine.core.models import ExecutionRecord

        record = ExecutionRecord(kind="md", label="act_1")
        record.transition(ExecutionState.VALIDATED, actor="t", reason="ok")
        record.transition(ExecutionState.QUEUED, actor="t", reason="ok")
        store.save_execution_record(record, campaign_id="c", action_id="act_1")

        restored = store.get_execution_record(record.id)
        assert restored.state is ExecutionState.QUEUED
        assert len(restored.history) == 2
        assert restored.history[0].actor == "t"


# ==========================================================================
# Provenance lineage
# ==========================================================================
class TestLineage:
    def test_lineage_survives_download_extraction_build_and_analysis(self, config, tmp_path, store) -> None:
        archive = tmp_path / "job.tgz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(VALID_SYSTEM, arcname="job")

        graph = ProvenanceGraph()
        imported = import_archive(archive, tmp_path / "unpacked", graph=graph, source={"provider": "charmm_gui"})
        builder = CampaignBuilder(config, graph=graph)
        campaign = builder.create(CampaignSpec.from_config(config, campaign_id="lineage", polymer_id="pol_x"))
        builder.attach_system(campaign, imported)
        builder.plan(campaign)

        analysis = graph.register_derived(
            artifact_id="analysis_1",
            kind="replica_analysis",
            parents=[f"{campaign.campaign_id}:replica_01"],
            parameters={"metric": "density"},
            payload={"density": 1050.0},
        )

        assert graph.dangling_parents() == [], "every referenced parent must exist in the graph"
        ancestors = graph.ancestors(analysis.artifact_id)
        assert imported.artifact_id in ancestors, "analysis must trace back to the imported system"
        roots = graph.roots(analysis.artifact_id)
        assert roots, "lineage must reach an original external input"
        root_artifact = graph.get(roots[0])
        assert root_artifact.kind == "provider_archive"
        assert graph.dangling_parents() == []

    def test_lineage_round_trips_through_the_store(self, config, store, tmp_path) -> None:
        graph = ProvenanceGraph()
        imported = import_directory(VALID_SYSTEM, graph=graph)
        store.save_provenance_graph(graph)

        restored = store.load_provenance_graph()
        assert imported.artifact_id in restored
        assert len(restored) == len(graph)

    def test_artifact_digests_are_verifiable(self, tmp_path) -> None:
        from polymer_engine.core.errors import ChecksumMismatch

        graph = ProvenanceGraph()
        path = tmp_path / "data.gro"
        path.write_text("original")
        artifact = graph.register_file(path, kind="coordinates")
        assert artifact.verify() is True

        path.write_text("tampered")
        with pytest.raises(ChecksumMismatch):
            artifact.verify()
        assert "MISMATCH" in graph.verify_all()[artifact.artifact_id]


# ==========================================================================
# Reproducibility
# ==========================================================================
class TestReproducibility:
    def test_building_the_same_campaign_twice_is_deterministic(self, config) -> None:
        fingerprints, seeds, digests = [], [], []
        for index in range(2):
            builder = CampaignBuilder(config)
            spec = CampaignSpec.from_config(
                config, campaign_id=f"repro{index}", polymer_id="pol_x"
            )
            campaign = builder.create(spec)
            builder.attach_system(campaign, import_directory(VALID_SYSTEM))
            builder.plan(campaign, gromacs_version="2026.3")
            manifest = campaign.manifest()
            fingerprints.append(manifest["fingerprint"])
            seeds.append(manifest["random_seeds"])
            digests.append(
                [
                    (Path(r.directory) / "nvt.mdp").read_text()
                    for r in campaign.replicas.replicas
                ]
            )
        assert fingerprints[0] == fingerprints[1]
        assert seeds[0] == seeds[1]
        assert digests[0] == digests[1], "identical specifications must produce identical mdp files"

    def test_changing_a_parameter_changes_the_fingerprint(self, config) -> None:
        base = CampaignSpec.from_config(config, campaign_id="a", polymer_id="pol_x")
        changed = CampaignSpec.from_config(config, campaign_id="b", polymer_id="pol_x")
        changed.simulation.temperature_k = 350.0
        assert campaign_fingerprint(base) != campaign_fingerprint(changed)

    def test_the_manifest_records_every_scientific_assumption(self, config) -> None:
        builder = CampaignBuilder(config, software={"gromacs": "2026.3"})
        campaign = builder.create(CampaignSpec.from_config(config, campaign_id="m", polymer_id="pol_x"))
        builder.attach_system(campaign, import_directory(VALID_SYSTEM))
        builder.plan(campaign)
        manifest = campaign.manifest()

        assert manifest["force_field"] == "CHARMM36"
        assert manifest["software"]["gromacs"] == "2026.3"
        assert manifest["specification"]["simulation"]["temperature_k"] == 300.0
        assert manifest["specification"]["simulation"]["barostat"]
        assert manifest["analysis_settings"]["confidence_level"]
        assert manifest["random_seeds"]
        assert manifest["system"]["digest"]
        assert manifest["replicas"]["seeds_distinct"] is True
