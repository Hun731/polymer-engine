"""End-to-end scientific pipelines.

Two benchmarks, as small as they can be while remaining real:

1. ``TestFullPipeline`` -- polymer candidate through QM, MD, convergence, property
   extraction, model feature and candidate ranking. Uses real ORCA and real GROMACS
   when they are installed, and is skipped with an explicit reason when they are not.

2. ``TestUmbrellaPipeline`` -- an equilibrated interface through windows, PLUMED
   inputs, PMF, uncertainty and the promotion gate. The window sampling is analytic, so
   the recovered free-energy surface has a right answer; PLUMED itself is not required.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from polymer_engine.core.config import load_config
from polymer_engine.core.models import Determination, GateStatus
from polymer_engine.core.provenance import ProvenanceGraph
from polymer_engine.core.units import kT
from polymer_engine.db.store import Store
from polymer_engine.discovery.qspr import Dataset
from polymer_engine.orchestrator.campaign import CampaignBuilder, CampaignSpec, save_campaign
from polymer_engine.orchestrator.runner import CampaignRunner
from polymer_engine.orchestrator.scheduler import (
    Job,
    JobState,
    LocalExecutor,
    ResourcePool,
    ResourceRequest,
    Scheduler,
)
from polymer_engine.polymer.records import build_record
from polymer_engine.polymer.taxonomy import PolymerFamily
from polymer_engine.properties.thermodynamic import Density
from polymer_engine.qm.spec import Atom, JobKind, QMJobSpec, Structure, TorsionSpec
from polymer_engine.qm.validation import AcceptanceCriteria, compare_torsion_profiles, validate_qm_run
from polymer_engine.simulation.builder import SystemBuildRequest, default_builder
from polymer_engine.simulation.forcefield import ForceFieldAdvisor
from polymer_engine.simulation.umbrella import ReactionCoordinate
from polymer_engine.simulation.umbrella_execution import (
    UmbrellaCampaign,
    UmbrellaJustification,
    UmbrellaStatus,
    WindowExecution,
)
from tests.markers import (
    requires_gromacs,
    requires_orca,
    requires_rdkit,
    requires_sklearn,
)

VALID_SYSTEM = Path(__file__).resolve().parents[1] / "fixtures" / "systems" / "valid_system"

ETHANE = [
    Atom("C", -0.765, 0.0, 0.0), Atom("C", 0.765, 0.0, 0.0),
    Atom("H", -1.14, 1.018, 0.0), Atom("H", -1.14, -0.509, 0.8815),
    Atom("H", -1.14, -0.509, -0.8815), Atom("H", 1.14, -1.018, 0.0),
    Atom("H", 1.14, 0.509, 0.8815), Atom("H", 1.14, 0.509, -0.8815),
]


def pipeline_config(tmp_path: Path, **overrides):
    base = {
        "paths": {"root": str(tmp_path)},
        "http": {"cache_enabled": False, "offline": True},
        "simulation": {
            "replicas": 2,
            "minimization_steps": 100,
            "nvt_ns": 0.002,
            "npt_ns": 0.002,
            "production_ns": 0.01,
            "force_field": "test-UA",
            "water_model": "SPC-like",
            "trajectory_output_ps": 0.02,
            "energy_output_ps": 0.02,
            "log_output_ps": 1.0,
        },
        "resources": {"gpu_available": False},
    }
    base.update(overrides)
    config = load_config(discover=False, use_env=False, overrides=base)
    config.paths.ensure()
    return config


class TestFullPipeline:
    """Candidate -> QM -> system -> MD -> convergence -> property -> model -> ranking."""

    @requires_rdkit
    def test_candidate_is_curated_and_classified(self) -> None:
        record = build_record(name="polyethylene", repeat_unit_smiles="*CC*")
        assert record.family is PolymerFamily.POLYOLEFIN
        assert record.curation_status is Determination.KNOWN
        assert record.usable_for_modelling is True

    @requires_rdkit
    def test_force_field_choice_is_recorded_not_assumed(self) -> None:
        strategy = ForceFieldAdvisor().propose(
            "pol_pe", PolymerFamily.POLYOLEFIN, requested="CHARMM36"
        )
        assert strategy.selected_force_field == "CHARMM36"
        # Coverage alone is not validation, so this may not run unattended.
        assert strategy.ready_to_simulate is False

    @requires_orca
    @pytest.mark.slow
    def test_qm_torsional_profile_is_computed_and_validated(self, tmp_path: Path) -> None:
        """Real ORCA relaxed scan of the ethane rotational barrier."""
        from polymer_engine.qm.orca_runner import build_orca_runner

        config = pipeline_config(tmp_path, safety={"execution_enabled": True})
        runner = build_orca_runner(config)
        spec = QMJobSpec(
            kind=JobKind.TORSION_SCAN,
            structure=Structure(atoms=list(ETHANE), name="ethane"),
            method="HF", basis="STO-3G", label="ethane_scan",
            torsion=TorsionSpec(atoms=(2, 0, 1, 6), start_deg=0.0, stop_deg=120.0, n_points=5),
            timeout_s=900,
        )
        run = runner.run_torsion_scan(spec, tmp_path / "qm")
        assert run.succeeded, run.error
        assert validate_qm_run(spec, run.output).passed is True

        barrier = run.output.torsional_barrier_kj_mol()
        # The experimental ethane rotational barrier is about 12.1 kJ/mol.
        assert barrier == pytest.approx(12.1, abs=2.0)

        # A force field reproducing this profile exactly must pass its comparison.
        profile = run.output.scan_profile_kj_mol()
        comparison = compare_torsion_profiles(
            list(profile), profile, criteria=AcceptanceCriteria.general_organic_forcefield()
        )
        assert comparison.passed is True

    def test_system_is_built_and_validated(self, tmp_path: Path) -> None:
        graph = ProvenanceGraph()
        result = default_builder().build(
            SystemBuildRequest(
                polymer_id="pol_pe", force_field="test-UA", source_directory=str(VALID_SYSTEM)
            ),
            tmp_path / "system",
            graph=graph,
        )
        assert result.usable is True
        assert result.validation.status is GateStatus.PASS

    @requires_gromacs
    @pytest.mark.slow
    def test_real_md_runs_and_the_gates_judge_it(self, tmp_path: Path) -> None:
        """Real GROMACS EM/NVT/NPT/production, then convergence and property gates."""
        config = pipeline_config(tmp_path, safety={"execution_enabled": True})
        with Store(config.paths.resolved("database")) as store:
            builder = CampaignBuilder(config, software={"gromacs": "real"})
            campaign = builder.create(
                CampaignSpec.from_config(
                    config, campaign_id="pipeline", polymer_id="pol_pe",
                    question="What is the equilibrium density?",
                )
            )
            from polymer_engine.simulation.system import import_directory

            builder.attach_system(campaign, import_directory(VALID_SYSTEM))
            builder.plan(campaign)
            save_campaign(store, campaign)
            summary = CampaignRunner(config, store).run("pipeline")

        equilibrations = [e for e in summary["executed"] if e["kind"] == "gromacs_equilibrate"]
        assert equilibrations, "no replica ran"
        for entry in equilibrations:
            assert entry["execution_mode"] == "real"
            assert entry["status"] == "succeeded", entry["error"]

        analyses = [e for e in summary["executed"] if e["kind"] == "analyze_replicas"]
        assert analyses, "analysis did not run"
        # 10 ps across 2 replicas cannot support a density; the gates must say so.
        assert analyses[0]["scientifically_usable"] is False
        assert analyses[0]["gate_status"] in {"fail", "inconclusive"}

    def test_property_extraction_feeds_a_model_feature(self, tmp_path: Path) -> None:
        """A property with enough sampling becomes a usable model feature."""
        rng = np.random.default_rng(1)
        result = Density().compute(
            1050.0 + rng.normal(0, 2.0, 5000), n_replicas=3, simulation_ns=50.0
        )
        assert result.usable is True
        assert result.measurement.units == "kg/m^3"
        assert result.measurement.uncertainty is not None

    @requires_sklearn
    def test_candidate_ranking_closes_the_loop(self, tmp_path: Path) -> None:
        """Descriptors -> model -> uncertainty -> ranked next experiments."""
        from polymer_engine.discovery.active_learning import Candidate, select_next_experiments
        from polymer_engine.discovery.qspr import QsprModel

        rng = np.random.default_rng(2)
        n = 60
        X = rng.normal(size=(n, 4))
        y = 300.0 + 40.0 * X[:, 0] - 20.0 * X[:, 1] + rng.normal(0, 3.0, n)
        dataset = Dataset(
            polymer_ids=[f"known{i}" for i in range(n)],
            feature_names=["a", "b", "c", "d"], X=X, y=y,
            target_name="glass_transition_temperature", target_units="K",
        )
        model = QsprModel(n_estimators=60).fit(dataset)
        candidates = [
            Candidate(polymer_id=f"cand{i}", name=f"candidate {i}", features=rng.normal(size=4))
            for i in range(12)
        ]
        report = select_next_experiments(candidates, model, dataset.X, batch_size=3)
        assert len(report.selected) == 3
        assert all(c.polymer_id.startswith("cand") for c in report.selected)
        assert all(c.uncertainty is not None for c in report.selected)

    def test_the_scheduler_runs_replicas_without_oversubscription(self, tmp_path: Path) -> None:
        """Four GPU replicas on one GPU serialise rather than all starting."""
        import threading

        active = {"n": 0, "max": 0}
        lock = threading.Lock()

        def handler(job: Job) -> str:
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            with lock:
                active["n"] -= 1
            return "ok"

        pool = ResourcePool(total_cpus=8, total_gpus=1, total_memory_mb=16000)
        scheduler = Scheduler(pool, LocalExecutor(handler))
        scheduler.submit_all(
            [
                Job(job_id=f"replica{i}", kind="md", resources=ResourceRequest(cpus=2, gpus=1))
                for i in range(4)
            ]
        )
        report = scheduler.run()
        assert active["max"] == 1
        assert len(report.by_state(JobState.SUCCEEDED)) == 4


class TestUmbrellaPipeline:
    """Equilibrated interface -> windows -> PLUMED -> PMF -> uncertainty -> promotion."""

    TEMPERATURE = 300.0
    TRUE_K = 200.0
    TRUE_MINIMUM = 1.0

    def _runner(self, seed: int = 0, n: int = 8000):
        rng = np.random.default_rng(seed)
        beta = 1.0 / kT(self.TEMPERATURE)

        def run(window, directory: Path) -> WindowExecution:
            k = window.force_constant
            mean = (self.TRUE_K * self.TRUE_MINIMUM + k * window.center) / (self.TRUE_K + k)
            sd = math.sqrt(1.0 / (beta * (self.TRUE_K + k)))
            values = rng.normal(mean, sd, n)
            path = Path(directory) / "COLVAR"
            path.write_text(
                "#! FIELDS time d\n"
                + "\n".join(f"{i * 0.5:.2f} {v:.6f}" for i, v in enumerate(values)),
                encoding="utf-8",
            )
            return WindowExecution(succeeded=True, colvar_path=path)

        return run

    def _justification(self) -> UmbrellaJustification:
        return UmbrellaJustification(
            question="What is the free energy of separating two chains at the interface?",
            reaction_coordinate="centre-of-mass distance between the two chains",
            physical_interpretation="reversible work to pull the chains apart",
            expected_observable="depth of the PMF well relative to the separated plateau",
            reason_for_method="the barrier region is not sampled in equilibrium MD",
            starting_state="equilibrated contact pair at 0.6 nm",
            endpoint_definition="the plateau beyond 1.4 nm",
            author="benchmark",
        )

    def test_pipeline_produces_a_trustworthy_pmf(self, tmp_path: Path) -> None:
        graph = ProvenanceGraph()
        coordinate = ReactionCoordinate(
            name="d", kind="com_distance", units="nm",
            justification="Interchain separation at the interface.",
            group_a="1-100", group_b="101-200",
        )
        campaign = UmbrellaCampaign(coordinate, temperature_k=self.TEMPERATURE, graph=graph)
        result = campaign.run(
            minimum=0.6, maximum=1.4, root=tmp_path / "umbrella",
            justification=self._justification(), runner=self._runner(1),
        )
        assert result.status is UmbrellaStatus.COMPLETED
        assert result.trustworthy is True
        assert result.determination is Determination.KNOWN

        # The recovered surface must match the analytic PMF it was sampled from.
        mask = np.isfinite(result.pmf.pmf) & result.pmf.well_sampled
        x = result.pmf.coordinate[mask]
        recovered = result.pmf.pmf[mask] - result.pmf.pmf[mask].min()
        analytic = 0.5 * self.TRUE_K * (x - self.TRUE_MINIMUM) ** 2
        analytic -= analytic.min()
        assert np.abs(recovered - analytic).max() < 1.5

    def test_plumed_inputs_are_written_for_every_window(self, tmp_path: Path) -> None:
        coordinate = ReactionCoordinate(
            name="d", kind="com_distance", units="nm", justification="x",
            group_a="1-100", group_b="101-200",
        )
        root = tmp_path / "umbrella"
        UmbrellaCampaign(coordinate, temperature_k=self.TEMPERATURE).run(
            minimum=0.6, maximum=1.0, root=root,
            justification=self._justification(), runner=self._runner(2),
        )
        window_dirs = sorted(root.glob("window_*"))
        assert window_dirs
        for directory in window_dirs:
            text = (directory / "plumed.dat").read_text()
            assert "RESTRAINT" in text
            assert "UNITS LENGTH=nm ENERGY=kj/mol" in text

    def test_uncertainty_is_estimated_and_the_gate_reports_it(self, tmp_path: Path) -> None:
        coordinate = ReactionCoordinate(
            name="d", kind="com_distance", units="nm", justification="x",
            group_a="1-100", group_b="101-200",
        )
        result = UmbrellaCampaign(coordinate, temperature_k=self.TEMPERATURE).run(
            minimum=0.6, maximum=1.4, root=tmp_path / "umbrella",
            justification=self._justification(), runner=self._runner(3),
        )
        assert result.pmf.uncertainty is not None
        uncertainty_gate = next(
            g for g in result.report.gates if g.gate == "umbrella:uncertainty_estimated"
        )
        assert uncertainty_gate.value is not None

    def test_the_promotion_gate_refuses_an_unjustified_campaign(self, tmp_path: Path) -> None:
        coordinate = ReactionCoordinate(
            name="d", kind="com_distance", units="nm", justification="x",
            group_a="1-100", group_b="101-200",
        )
        result = UmbrellaCampaign(coordinate, temperature_k=self.TEMPERATURE).run(
            minimum=0.6, maximum=1.4, root=tmp_path / "umbrella",
            justification=None, runner=self._runner(4),
        )
        assert result.trustworthy is False
        assert result.runs == []
