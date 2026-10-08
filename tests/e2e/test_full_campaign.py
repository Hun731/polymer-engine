"""End-to-end campaign flows.

``TestDryRunCampaign`` runs the complete pipeline without any external software, so
it works everywhere.  ``TestRealExecution`` runs actual GROMACS and is skipped with an
explicit reason when ``gmx`` is not installed.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import pytest

from polymer_engine.core.config import load_config
from polymer_engine.core.models import ExecutionState, Objective
from polymer_engine.core.provenance import ProvenanceGraph
from polymer_engine.db.store import Store
from polymer_engine.evidence.claims import (
    Claim,
    ClaimLedger,
    ClaimStatus,
    Stance,
    evidence_from_gate_report,
)
from polymer_engine.orchestrator.campaign import (
    CampaignBuilder,
    CampaignSpec,
    CampaignStatus,
    save_campaign,
)
from polymer_engine.orchestrator.runner import CampaignRunner
from polymer_engine.orchestrator.strategy import StrategyOutcome, StrategyRegistry
from polymer_engine.simulation.system import import_archive
from tests.markers import requires_gromacs

VALID_SYSTEM = Path(__file__).resolve().parents[1] / "fixtures" / "systems" / "valid_system"


def make_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "charmm_gui_job.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(VALID_SYSTEM, arcname="charmm-gui-1234")
    return archive


def make_config(tmp_path: Path, **overrides):
    base = {
        "paths": {"root": str(tmp_path)},
        "http": {"cache_enabled": False, "offline": True},
        "simulation": {
            "replicas": 3,
            "production_ns": 5.0,
            "force_field": "CHARMM36",
            "water_model": "TIP3P",
        },
    }
    base.update(overrides)
    config = load_config(discover=False, use_env=False, overrides=base)
    config.paths.ensure()
    return config


class TestDryRunCampaign:
    """The whole pipeline, with no external software involved."""

    def test_full_pipeline_without_executing_anything(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        archive = make_archive(tmp_path)

        with Store(config.paths.resolved("database")) as store:
            objective = Objective(
                title="Cohesion study",
                description="Relate interchain cohesion to mechanical response.",
            )
            store.save_objective(objective)
            StrategyRegistry().save(store)

            graph = ProvenanceGraph()
            imported = import_archive(
                archive, tmp_path / "systems" / "job1234", graph=graph,
                source={"provider": "charmm_gui", "job_id": "1234"},
            )
            assert imported.usable, imported.report.summary()

            builder = CampaignBuilder(config, graph=graph, software={"gromacs": "2026.3"})
            spec = CampaignSpec.from_config(
                config,
                campaign_id="e2e001",
                polymer_id="pol_test",
                polymer_name="Test polymer",
                objective_id=objective.id,
                question="What is the equilibrium density?",
            )
            campaign = builder.create(spec)
            builder.attach_system(campaign, imported)
            assert campaign.status is CampaignStatus.SYSTEM_VALIDATED

            builder.plan(campaign, gromacs_version="2026.3")
            assert campaign.status is CampaignStatus.PLANNED
            store.save_provenance_graph(graph)
            save_campaign(store, campaign)

            summary = CampaignRunner(config, store).run("e2e001")

        # Execution was disabled, so nothing may claim to have succeeded.
        assert summary["n_executed"] >= 1
        for entry in summary["executed"]:
            assert entry["execution_mode"] == "dry_run"
            assert entry["status"] == "skipped"
            assert entry["scientifically_usable"] is False
            assert entry["state"] == ExecutionState.CANCELLED.value
        assert summary["gate_status"] == "inconclusive"

    def test_dry_run_does_not_satisfy_a_dependency(self, tmp_path: Path) -> None:
        """The analysis step must not run on the back of replicas that never executed."""
        config = make_config(tmp_path)
        with Store(config.paths.resolved("database")) as store:
            builder = CampaignBuilder(config)
            campaign = builder.create(
                CampaignSpec.from_config(config, campaign_id="dep", polymer_id="pol_x")
            )
            builder.attach_system(campaign, _import_dir(tmp_path))
            builder.plan(campaign)
            save_campaign(store, campaign)
            summary = CampaignRunner(config, store).run("dep")

        kinds = [e["kind"] for e in summary["executed"]]
        assert "analyze_replicas" not in kinds

    def test_every_decision_is_recorded_with_its_reasoning(self, tmp_path: Path) -> None:
        """'Why did the engine run this?' must always be answerable."""
        config = make_config(tmp_path)
        with Store(config.paths.resolved("database")) as store:
            builder = CampaignBuilder(config)
            campaign = builder.create(
                CampaignSpec.from_config(config, campaign_id="why", polymer_id="pol_x")
            )
            builder.attach_system(campaign, _import_dir(tmp_path))
            builder.plan(campaign)
            save_campaign(store, campaign)
            CampaignRunner(config, store).run("why")

            decisions = store.list_decisions(campaign_id="why")
            assert decisions
            chosen = [d for d in decisions if d["selected_action"]]
            assert chosen
            for decision in chosen:
                assert decision["reason"]
                assert decision["candidate_actions"]
                assert decision["estimated_cost"] is not None
                assert decision["estimated_information_gain"] is not None

    def test_manifest_is_written_and_reconstructable(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        with Store(config.paths.resolved("database")) as store:
            builder = CampaignBuilder(config, software={"gromacs": "2026.3"})
            campaign = builder.create(
                CampaignSpec.from_config(config, campaign_id="man", polymer_id="pol_x")
            )
            builder.attach_system(campaign, _import_dir(tmp_path))
            builder.plan(campaign)
            save_campaign(store, campaign)

        manifest_path = campaign.workdir / "campaign_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())

        rebuilt = CampaignSpec.from_dict(manifest["specification"])
        assert rebuilt.polymer_id == campaign.spec.polymer_id
        assert rebuilt.simulation.temperature_k == campaign.spec.simulation.temperature_k
        assert rebuilt.simulation.force_field == "CHARMM36"

        from polymer_engine.orchestrator.campaign import campaign_fingerprint

        assert campaign_fingerprint(rebuilt) == campaign_fingerprint(campaign.spec)

    def test_claims_cannot_be_promoted_from_a_dry_run(self, tmp_path: Path) -> None:
        """The pipeline may run to completion without licensing any claim."""
        config = make_config(tmp_path)
        with Store(config.paths.resolved("database")) as store:
            builder = CampaignBuilder(config)
            campaign = builder.create(
                CampaignSpec.from_config(config, campaign_id="claims", polymer_id="pol_x")
            )
            builder.attach_system(campaign, _import_dir(tmp_path))
            builder.plan(campaign)
            save_campaign(store, campaign)
            summary = CampaignRunner(config, store).run("claims")

            ledger = ClaimLedger()
            claim = ledger.add(
                Claim(claim_id="density_claim", statement="The polymer's density is 1050 kg/m^3")
            )
            for entry in summary["executed"]:
                report = campaign.gate_reports.get(entry["action_id"])
                if report is None:
                    continue
                claim.add_evidence(
                    evidence_from_gate_report(
                        entry["action_id"], report, stance=Stance.SUPPORTS,
                        description=entry["summary"], n_independent_replicates=0,
                    )
                )
            assert claim.status is ClaimStatus.INSUFFICIENT_EVIDENCE
            ledger.save(store)
            assert ClaimLedger.load(store).supported() == []

    def test_strategy_outcomes_persist_across_the_run(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        with Store(config.paths.resolved("database")) as store:
            registry = StrategyRegistry()
            registry.save(store)
            for _ in range(6):
                registry.record_outcome(
                    StrategyOutcome("three_replica_baseline", "e2e", succeeded=True, cost=1.0, information_gain=0.7)
                )
            registry.save(store)

            reloaded = StrategyRegistry.load(store)
            assert reloaded.get("three_replica_baseline").applications == 6
            assert reloaded.best().strategy_id == "three_replica_baseline"

    def test_provenance_reaches_from_analysis_back_to_the_archive(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        archive = make_archive(tmp_path)
        graph = ProvenanceGraph()
        imported = import_archive(archive, tmp_path / "sys", graph=graph, source={"provider": "charmm_gui"})
        builder = CampaignBuilder(config, graph=graph)
        campaign = builder.create(CampaignSpec.from_config(config, campaign_id="prov", polymer_id="pol_x"))
        builder.attach_system(campaign, imported)
        builder.plan(campaign)

        replica_artifact = "prov:replica_01"
        assert replica_artifact in graph
        assert imported.artifact_id in graph.ancestors(replica_artifact)
        assert graph.dangling_parents() == []
        roots = graph.roots(replica_artifact)
        assert graph.get(roots[0]).kind == "provider_archive"


def _import_dir(tmp_path: Path):
    from polymer_engine.simulation.system import import_directory

    return import_directory(VALID_SYSTEM)


# ==========================================================================
# Real execution -- skipped with a reason when GROMACS is absent
# ==========================================================================
@requires_gromacs
class TestRealExecution:
    """Runs actual GROMACS.  Deliberately tiny; still a real MD run."""

    @pytest.mark.slow
    def test_real_campaign_runs_and_gates_refuse_an_undersampled_result(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            safety={"execution_enabled": True},
            resources={"gpu_available": False},
            simulation={
                "replicas": 2,
                "minimization_steps": 100,
                "nvt_ns": 0.002,
                "npt_ns": 0.002,
                "production_ns": 0.01,
                "force_field": "test-UA",
                "trajectory_output_ps": 0.02,
                "energy_output_ps": 0.02,
                "log_output_ps": 1.0,
            },
        )
        with Store(config.paths.resolved("database")) as store:
            builder = CampaignBuilder(config, software={"gromacs": "real"})
            campaign = builder.create(
                CampaignSpec.from_config(config, campaign_id="real", polymer_id="pol_x")
            )
            builder.attach_system(campaign, _import_dir(tmp_path))
            builder.plan(campaign)
            save_campaign(store, campaign)
            summary = CampaignRunner(config, store).run("real")

        equilibrations = [e for e in summary["executed"] if e["kind"] == "gromacs_equilibrate"]
        assert equilibrations, "no replica was executed"
        for entry in equilibrations:
            assert entry["execution_mode"] == "real"
            assert entry["status"] == "succeeded", entry["error"]

        analyses = [e for e in summary["executed"] if e["kind"] == "analyze_replicas"]
        assert analyses, "analysis did not run after the replicas completed"
        # GROMACS exited 0 everywhere, yet 10 ps across 2 replicas is not a result.
        assert analyses[0]["gate_status"] in {"fail", "inconclusive"}
        assert analyses[0]["scientifically_usable"] is False

    @pytest.mark.slow
    def test_generated_mdp_files_are_accepted_by_grompp(self, tmp_path: Path) -> None:
        """The strongest check on the mdp generator: real grompp, zero warnings."""
        import subprocess

        from polymer_engine.core.config import SimulationDefaults
        from polymer_engine.simulation.mdp import generate_stages

        workdir = tmp_path / "grompp"
        shutil.copytree(VALID_SYSTEM, workdir)
        defaults = SimulationDefaults(
            minimization_steps=100, nvt_ns=0.002, npt_ns=0.002, production_ns=0.002
        )
        for stage in generate_stages(defaults, replica_index=0, gromacs_version="2026"):
            stage.write(workdir)
            result = subprocess.run(
                ["gmx", "grompp", "-f", f"{stage.name}.mdp", "-c", "system.gro",
                 "-p", "topol.top", "-o", f"{stage.name}.tpr"],
                cwd=workdir, capture_output=True, text=True, check=False,
            )
            output = result.stdout + result.stderr
            assert result.returncode == 0, f"grompp rejected {stage.name}.mdp:\n{output[-2000:]}"
            assert "WARNING" not in output, f"{stage.name}.mdp produced warnings:\n{output[-2000:]}"
