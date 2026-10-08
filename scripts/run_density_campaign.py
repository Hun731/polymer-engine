#!/usr/bin/env python
"""Drive the 96-hour autonomous polyolefin density campaign.

Run it, disconnect, run it again: state lives in ``campaign_state.json`` and the
campaign resumes from the last checkpoint.  Nothing here needs the conversation that
started it.

    python scripts/run_density_campaign.py --config configs/campaign_96h.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymer_engine.analysis.ordering import OrderingVerdict
from polymer_engine.core.config import ToolConfig
from polymer_engine.core.logging import get_logger
from polymer_engine.core.provenance import ProvenanceGraph
from polymer_engine.local.discovery import discover_tool
from polymer_engine.local.runner import GROMACSRunner
from polymer_engine.orchestrator.density_campaign import (
    STATUS_JSON,
    STATUS_MD,
    CampaignState,
    CandidateSpec,
    ExperimentResult,
    ReplicaResult,
    analyse_replica_density,
    combine_replica_densities,
    read_density_series,
    register_campaign_artifacts,
    requeue_unfinished,
    utc_now_iso,
)
from polymer_engine.orchestrator.stop_policy import CampaignStatus, StopPolicy
from polymer_engine.simulation.melt_builder import build_melt
from polymer_engine.simulation.opls_typing import (
    find_opls_directory,
    load_opls_types,
    type_repeat_unit,
)

logger = get_logger("campaign")

#: Stage A: dataset chemistry with known experimental densities, for calibration.
STAGE_A: list[tuple[str, str, float]] = [
    ("polyethylene", "*CC*", 940.0),
    ("polypropylene", "*CC(C)*", 900.0),
    ("polyisobutylene", "*CC(C)(C)*", 920.0),
]

#: Stage B: designed expansion.  Every entry is a saturated hydrocarbon, so the whole
#: series stays inside the force field that was actually qualified.
STAGE_B: list[tuple[str, str, str, str]] = [
    ("poly(1-butene)", "*CC(CC)*", "polypropylene",
     "lengthen the side chain by one carbon; isolates side-chain length at fixed backbone"),
    ("poly(1-hexene)", "*CC(CCCC)*", "poly(1-butene)",
     "extend the linear side chain further along the same axis"),
    ("poly(1-octene)", "*CC(CCCCCC)*", "poly(1-hexene)",
     "longest linear side chain in the series; tests whether the trend saturates"),
    ("poly(3-methyl-1-butene)", "*CC(C(C)C)*", "poly(1-butene)",
     "branch the side chain at the alpha carbon at constant heavy-atom count"),
    ("poly(4-methyl-1-pentene)", "*CC(CC(C)C)*", "poly(1-hexene)",
     "branch the side chain further out; the classic low-density polyolefin"),
    ("poly(2-methyl-1-butene)", "*CC(C)(CC)*", "polyisobutylene",
     "asymmetric backbone disubstitution, between PIB and poly(1-butene)"),
    ("poly(2-methyl-1-pentene)", "*CC(C)(CCC)*", "poly(2-methyl-1-butene)",
     "lengthen one branch of a disubstituted backbone"),
    ("poly(neopentylethylene)", "*CC(CC(C)(C)C)*", "poly(4-methyl-1-pentene)",
     "quaternary carbon in the side chain; maximal side-chain bulk"),
    ("poly(vinylcyclohexane)", "*CC(C1CCCCC1)*", "polypropylene",
     "cyclic saturated side chain; rigidity at high heavy-atom count"),
    ("poly(1-pentene)", "*CC(CCC)*", "poly(1-butene)",
     "fills the odd-carbon gap in the linear side-chain series"),
]


def load_campaign_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def gmx_env() -> dict[str, str]:
    """GROMACS needs GMXLIB to resolve the force-field include from a working directory."""
    gmx = shutil.which("gmx")
    if not gmx:
        return {}
    top = Path(gmx).resolve().parent.parent / "share" / "gromacs" / "top"
    return {"GMXLIB": str(top)} if top.is_dir() else {}


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def gpu_utilisation() -> dict[str, Any]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout.strip().splitlines()
        if not out:
            return {}
        fields = [f.strip() for f in out[0].split(",")]
        return {"utilisation_percent": float(fields[0]),
                "memory_used_mb": float(fields[1]), "memory_total_mb": float(fields[2])}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


#: Per-stage wall-clock cap. Six hours was too tight: at 100 ns production a replica
#: takes about 4.1 hours, leaving 45% headroom, and one of three exceeded it and was
#: killed at exactly 360 minutes with a valid checkpoint on disk. A stage that is killed
#: loses its replica, and a candidate that loses a replica cannot demonstrate
#: reproducibility -- so a marginal timeout does not degrade a result, it discards one.
#:
#: The cap still exists to catch a genuinely hung run rather than to bound a healthy one.
DEFAULT_STAGE_TIMEOUT_S = 43200.0   # 12 hours


def _stage_timeout_s(cfg: dict[str, Any]) -> float:
    return float(cfg.get("resources", {}).get("stage_timeout_s", DEFAULT_STAGE_TIMEOUT_S))


def diagnose_ordering(runner: GROMACSRunner, directory: Path) -> Any:
    """Classify a finished replica's trajectory: stationary, relaxing, or ordering.

    Returns ``None`` when the potential energy cannot be extracted -- without it there
    is no way to tell the two apart, and guessing would defeat the purpose.
    """
    from polymer_engine.analysis.ordering import analyse

    density_file = directory / "density.xvg"
    energy_file = directory / "potential.xvg"
    if not density_file.exists():
        return None
    if not energy_file.exists():
        extracted = runner.run(
            ["energy", "-f", "prod.edr", "-o", "potential.xvg"],
            cwd=directory, stdin="Potential\n", timeout_s=300.0,
        )
        if not extracted.succeeded or not energy_file.exists():
            logger.info("no potential-energy series in %s; ordering not diagnosed",
                        directory.name)
            return None
    times, density = read_density_series(density_file)
    _t, energy = read_density_series(energy_file)
    n = min(len(density), len(energy))
    if n < 8:
        return None
    # The whole production window: a system that ordered early looks stationary if the
    # early part is discarded, and that is the run most needing the diagnosis.
    return analyse(times[:n], density[:n], energy[:n])


def write_mdps(directory: Path, cfg: dict[str, Any], seed: int,
               replica_index: int = 1) -> list[tuple[str, str]]:
    """Write every stage's mdp and return the (stage, previous) chain to run.

    The chain is derived from what was generated rather than written out here. It used
    to be a hard-coded tuple in two places, which is how an added stage gets its mdp
    written and then never executed.
    """
    from polymer_engine.core.config import load_config as load_engine_config
    from polymer_engine.simulation.mdp import generate_stages

    sim = cfg["simulation"]
    engine = load_engine_config(discover=False, use_env=False, overrides={"simulation": {
        "minimization_steps": sim["minimization_steps"],
        "nvt_ns": sim["nvt_ns"], "npt_ns": sim["npt_ns"],
        "production_ns": sim["production_ns"],
        "temperature_k": sim["temperature_k"], "pressure_bar": sim["pressure_bar"],
        "force_field": cfg["force_field"]["name"], "water_model": "none",
        "trajectory_output_ps": sim["trajectory_output_ps"],
        "energy_output_ps": sim["energy_output_ps"], "log_output_ps": 100.0,
        "anneal_ns": sim.get("anneal_ns", 0.0),
        "anneal_temperature_k": sim.get("anneal_temperature_k", 500.0),
        # The caller's seed is derived from the candidate name, so two different
        # polymers never share a random stream. generate_stages then derives a distinct
        # seed per (replica, stage) from it. Passing base_seed here keeps both
        # properties; letting it fall back to the engine default would give
        # polyethylene replica 1 and polypropylene replica 1 the same seed.
        "base_seed": seed,
    }})
    stages = generate_stages(engine.simulation, replica_index=replica_index)
    chain: list[tuple[str, str]] = []
    previous = "packed"
    for stage in stages:
        (directory / f"{stage.name}.mdp").write_text(stage.text, encoding="utf-8")
        chain.append((stage.name, previous))
        previous = stage.output_prefix
    return chain


def run_stage(runner: GROMACSRunner, directory: Path, stage: str, previous: str,
              *, ntomp: int, timeout_s: float) -> tuple[bool, str]:
    """One EM/NVT/NPT/production stage.  Returns ``(succeeded, message)``."""
    grompp = runner.grompp(
        mdp=f"{stage}.mdp", structure=f"{previous}.gro", topology="topol.top",
        output=f"{stage}.tpr", cwd=directory,
    )
    if not grompp.succeeded:
        return False, f"{stage} grompp: {(grompp.stderr or grompp.error or '')[-400:]}"
    # PME on the GPU needs a dynamical integrator; minimisation uses `steep`.
    md = runner.mdrun(deffnm=stage, cwd=directory, use_gpu=True, gpu_pme=(stage != "em"),
                      ntomp=ntomp, timeout_s=timeout_s)
    if not md.succeeded:
        return False, f"{stage} mdrun: {(md.stderr or md.error or '')[-400:]}"
    if not (directory / f"{stage}.gro").is_file():
        return False, f"{stage} exited 0 but wrote no structure"
    return True, ""


def extract_density(runner: GROMACSRunner, directory: Path) -> Path | None:
    result = runner.energy(edr="prod.edr", terms=["Density"], output="density.xvg",
                           cwd=directory)
    path = directory / "density.xvg"
    return path if result.succeeded and path.is_file() else None


def descriptors_for(smiles: str) -> dict[str, float]:
    from polymer_engine.polymer.records import build_record

    record = build_record(name="x", repeat_unit_smiles=smiles, properties={}, source="campaign")
    out: dict[str, float] = {}
    for key, measurement in record.descriptors.items():
        if measurement.determination.value == "KNOWN" and measurement.value is not None:
            out[key] = float(measurement.value)
    return out


def run_candidate(candidate: CandidateSpec, cfg: dict[str, Any], root: Path,
                  runner: GROMACSRunner, types: dict[str, Any], ff_source: str,
                  graph: ProvenanceGraph, deadline: float,
                  checkpoint: Any = None) -> ExperimentResult:
    """Build, simulate and judge one polymer.  Never returns a fabricated result."""
    sim, val = cfg["simulation"], cfg["validation"]
    started = time.time()
    slug = candidate.name.replace("(", "").replace(")", "").replace(" ", "_")
    work = root / "experiments" / slug
    work.mkdir(parents=True, exist_ok=True)
    result = ExperimentResult(candidate=candidate.name, started_at=utc_now_iso())

    for replica in range(1, int(sim["replicas"]) + 1):
        if time.time() > deadline:
            result.diagnostics.append(
                f"campaign deadline reached before replica {replica} started")
            break
        # Distinct, reproducible per-replica seed: derived from the name, not random.
        seed = 100003 + (sum(ord(c) for c in candidate.name) * 31) % 90000 + replica * 7919
        directory = work / f"replica_{replica:02d}"
        record = ReplicaResult(replica=replica, seed=seed, directory=str(directory))
        replica_started = time.time()

        if not (directory / "prod.gro").is_file():
            try:
                system = build_melt(
                    name=f"{slug}_r{replica}",
                    repeat_unit_smiles=candidate.repeat_unit_smiles,
                    degree_of_polymerization=int(sim["degree_of_polymerization"]),
                    n_chains=int(sim["chains_per_system"]),
                    target_density_kg_m3=candidate.experimental_density_kg_m3 or 900.0,
                    directory=directory, types=types, runner=runner,
                    force_field_source=ff_source, seed=seed,
                    box_scale=float(sim["packing_box_scale"]),
                )
            except Exception as exc:  # noqa: BLE001 - a build failure is data, not a crash
                record.error, record.stage_failed = f"system build failed: {exc}", "build"
                result.replicas.append(record)
                logger.warning("%s replica %d: %s", candidate.name, replica, record.error)
                continue
            result.n_chains = system.n_chains_packed
            result.atoms_per_chain = system.atoms_per_chain
            result.total_atoms = system.total_atoms
            result.diagnostics.extend(system.warnings)

            chain = write_mdps(directory, cfg, seed, replica_index=replica)
            stage_timeout_s = _stage_timeout_s(cfg)
            ok = True
            for stage, previous in chain:
                remaining = deadline - time.time()
                if remaining <= 60:
                    record.error, record.stage_failed = "campaign deadline reached mid-run", stage
                    ok = False
                    break
                ok, message = run_stage(runner, directory, stage, previous,
                                        ntomp=int(cfg["resources"]["cpus_per_job"]),
                                        timeout_s=min(remaining, stage_timeout_s))
                if not ok:
                    record.error, record.stage_failed = message, stage
                    logger.warning("%s replica %d: %s", candidate.name, replica, message)
                    break
            if not ok:
                record.wall_seconds = time.time() - replica_started
                result.replicas.append(record)
                continue
        else:
            logger.info("%s replica %d already finished; reusing", candidate.name, replica)

        xvg = extract_density(runner, directory)
        if xvg is None:
            record.error, record.stage_failed = "gmx energy produced no density series", "analysis"
            result.replicas.append(record)
            continue
        _times, densities = read_density_series(xvg)
        stats = analyse_replica_density(
            densities, min_effective_samples=float(val["min_effective_samples"]),
            max_drift_fraction=float(val["max_drift_fraction"]),
        )
        record.n_frames = int(stats.get("n_frames", 0))
        record.effective_samples = stats.get("effective_samples")
        record.statistical_inefficiency = stats.get("statistical_inefficiency")
        record.equilibration_index = stats.get("equilibration_index")
        record.density_kg_m3 = stats.get("mean")
        record.density_stderr = stats.get("stderr")
        record.succeeded = bool(stats.get("usable"))
        if not record.succeeded:
            record.error = stats.get("reason") or "replica did not pass its sampling gates"
            result.diagnostics.append(f"replica {replica}: {record.error}")
        # Why the density will not settle, when it will not. The sampling gates above
        # report "not enough independent samples" for a melt still relaxing and for one
        # that is crystallising, and those need opposite responses -- more time, or a
        # different question. Ordering releases potential energy; relaxation does not.
        ordering = diagnose_ordering(runner, directory)
        if ordering is not None:
            record.ordering_verdict = ordering.verdict.value
            record.ordering_reason = ordering.reason
            if ordering.verdict is OrderingVerdict.ORDERING:
                result.diagnostics.append(
                    f"replica {replica}: {ordering.reason}")
                logger.warning("%s replica %d is ordering, not under-sampled: %s",
                               candidate.name, replica, ordering.reason[:120])

        record.wall_seconds = time.time() - replica_started
        result.replicas.append(record)
        register_campaign_artifacts(graph, directory, parents=[],
                                    kind_prefix=f"{slug}_r{replica}")
        if checkpoint is not None:
            checkpoint(result)

    combined = combine_replica_densities(
        result.replicas, chi_square_max=float(val["replica_chi_square_max"]),
        required=int(sim["replicas"]),
    )
    result.gate_status = combined["status"]
    result.scientifically_usable = bool(combined["usable"])
    result.density_kg_m3 = combined.get("mean")
    result.density_uncertainty = combined.get("stderr")
    result.determination = "KNOWN" if result.scientifically_usable else "INSUFFICIENT_DATA"
    if combined.get("reason"):
        result.diagnostics.append(combined["reason"])
    result.finished_at = utc_now_iso()
    result.wall_seconds = time.time() - started
    return result


def train_model(state: CampaignState, root: Path) -> dict[str, Any]:
    """Relate descriptors to density using validated points only."""
    validated = [r for r in state.results.values()
                 if r.scientifically_usable and r.density_kg_m3 is not None and r.descriptors]
    if len(validated) < 3:
        return {"status": "insufficient validated data",
                "n_validated": len(validated), "n_required": 3}
    import numpy as np

    keys = sorted(set.intersection(*(set(r.descriptors) for r in validated)))
    if not keys:
        return {"status": "no shared descriptors"}
    matrix = np.array([[r.descriptors[k] for k in keys] for r in validated], dtype=float)
    target = np.array([r.density_kg_m3 for r in validated], dtype=float)

    correlations: dict[str, float] = {}
    for index, key in enumerate(keys):
        column = matrix[:, index]
        if float(np.std(column)) == 0.0:
            continue
        value = float(np.corrcoef(column, target)[0, 1])
        if math.isfinite(value):
            correlations[key] = round(value, 4)

    n_tests = len(correlations)
    # Bonferroni: screening many descriptors against one target manufactures hits.
    threshold = 0.05 / max(n_tests, 1)
    report = {
        "status": "trained", "n_validated": len(validated), "n_tests": n_tests,
        "alpha_uncorrected": 0.05, "alpha_bonferroni": round(threshold, 6),
        "correlations_pearson_r": dict(sorted(correlations.items(),
                                              key=lambda kv: -abs(kv[1]))),
        "note": ("Pearson r over validated points only. Association, not causation. "
                 "With this few points no correlation should be read as established; "
                 "n_tests is reported so the multiplicity burden is visible."),
        "points": [{"candidate": r.candidate, "density_kg_m3": r.density_kg_m3,
                    "uncertainty": r.density_uncertainty} for r in validated],
    }
    (root / "model_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def write_status(state: CampaignState, cfg: dict[str, Any], root: Path,
                 next_action: str, model: dict[str, Any] | None,
                 budget: float | None) -> None:
    """``budget`` is the budget actually in force; ``None`` means open-ended.

    Reading it back out of the config here would make a rehearsal report a budget it is
    not going to run for.
    """
    configured = cfg["campaign"].get("duration_hours")
    validated = [r for r in state.results.values() if r.scientifically_usable]
    ranking = sorted(state.results.values(),
                     key=lambda x: (x.density_kg_m3 is None, x.density_kg_m3 or 0.0))
    payload = {
        "campaign_id": cfg["campaign"]["id"],
        "question": cfg["campaign"]["question"],
        "start_time": state.started_at,
        "current_time": utc_now_iso(),
        "elapsed_hours": round(state.elapsed_hours(), 3),
        "remaining_hours": (round(state.remaining_hours(budget), 3)
                            if budget is not None else None),
        "open_ended": budget is None,
        "iteration": state.iteration,
        "decisions_recorded": state.decisions,
        "queued": [c.name for c in state.queue],
        "running_jobs": 1 if next_action.startswith("running") else 0,
        "completed_jobs": len(state.completed),
        "failed_jobs": len(state.failed),
        "validated_jobs": len(validated),
        "candidates_evaluated": len(state.results),
        "candidate_ranking": [
            {"candidate": r.candidate, "density_kg_m3": r.density_kg_m3,
             "uncertainty": r.density_uncertainty, "gate": r.gate_status,
             "usable": r.scientifically_usable,
             "replicas_ok": sum(1 for x in r.replicas if x.succeeded)}
            for r in ranking
        ],
        "model": model or {"status": "not yet trained"},
        "next_action": next_action,
        "force_field": f"{cfg['force_field']['name']} (QM-qualified)",
        "umbrella_enabled": cfg["umbrella"]["enabled"],
        "mechanics_enabled": cfg["mechanics"]["enabled"],
        "gpu": gpu_utilisation(),
        "disk_free_gb": round(disk_free_gb(root), 1),
        "budget_hours": budget,
        "configured_budget_hours": configured,
        "stopped_reason": state.stopped_reason,
    }
    (root / STATUS_JSON).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    lines = [
        f"# {payload['campaign_id']}", "",
        f"**Question.** {payload['question']}", "", "| | |", "|---|---|",
        f"| Started | {payload['start_time']} |",
        f"| Now | {payload['current_time']} |",
        (f"| Elapsed | {payload['elapsed_hours']:.2f} h (open-ended) |"
         if budget is None
         else f"| Elapsed | {payload['elapsed_hours']:.2f} h of {budget:g} |"),
        ("| Remaining | — campaign stops on science, not a clock |"
         if budget is None
         else f"| Remaining | {payload['remaining_hours']:.2f} h |"),
        f"| Iteration | {payload['iteration']} |",
        f"| Candidates evaluated | {payload['candidates_evaluated']} |",
        f"| Validated | {payload['validated_jobs']} |",
        f"| Failed | {payload['failed_jobs']} |",
        f"| Queued | {len(payload['queued'])} |",
        f"| Force field | {payload['force_field']} |",
        f"| Disk free | {payload['disk_free_gb']} GB |", "",
        "## Candidates", "",
        "| Polymer | Density (kg/m^3) | Uncertainty | Replicas OK | Gate | Usable |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["candidate_ranking"]:
        density = f"{row['density_kg_m3']:.1f}" if row["density_kg_m3"] is not None else "--"
        uncertainty = f"{row['uncertainty']:.1f}" if row["uncertainty"] is not None else "--"
        lines.append(
            f"| {row['candidate']} | {density} | {uncertainty} | {row['replicas_ok']} | "
            f"{row['gate']} | {'yes' if row['usable'] else 'no'} |"
        )
    lines += ["", f"**Next action.** {payload['next_action']}", ""]
    if state.stopped_reason:
        lines += [f"**Stopped.** {state.stopped_reason}", ""]
    (root / STATUS_MD).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/campaign_96h.yaml")
    parser.add_argument("--root", default="campaign/run")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="override the configured budget (for a short rehearsal)")
    args = parser.parse_args()

    cfg = load_campaign_config(Path(args.config))
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # Open-ended by default. `duration_hours: null` (or absent) means the campaign runs
    # until science, resources or a person stops it -- not until a clock runs out.
    configured = cfg["campaign"].get("duration_hours")
    budget = args.max_hours if args.max_hours is not None else (
        float(configured) if configured is not None else None
    )
    policy = StopPolicy(
        duration_hours=budget,
        disk_reserve_gb=float(cfg["resources"]["disk_reserve_gb"]),
        target_validated=cfg["campaign"].get("target_validated"),
    )
    consecutive_failures = 0
    logger.info("Stop policy: %s", policy.as_dict())
    state = CampaignState.load(root) or CampaignState(root)
    if not state.queue and not state.results:
        for name, smiles, density in STAGE_A:
            state.queue.append(CandidateSpec(
                name=name, repeat_unit_smiles=smiles, origin="dataset",
                design_reason="Stage A calibration: dataset chemistry with a known density",
                experimental_density_kg_m3=density))
        logger.info("Seeded Stage A with %d calibration candidates", len(state.queue))
    # Checkpoint before the first candidate starts. The first candidate takes tens of
    # minutes, and without this the campaign has nothing to resume from for that whole
    # window -- a failure that only ever shows up when something crashes.
    state.save()

    # Anything a crash left unfinished goes back on the queue before new work starts.
    all_candidates = {
        name: CandidateSpec(name=name, repeat_unit_smiles=smiles, origin="dataset",
                            design_reason="Stage A calibration",
                            experimental_density_kg_m3=density)
        for name, smiles, density in STAGE_A
    }
    all_candidates.update({
        name: CandidateSpec(name=name, repeat_unit_smiles=smiles, origin="generated",
                            parent=parent, mutation="side-chain / backbone substitution",
                            design_reason=reason)
        for name, smiles, parent, reason in STAGE_B
    })
    restored = requeue_unfinished(
        state, all_candidates,
        required_replicas=int(cfg["simulation"]["replicas"]),
        max_attempts=int(cfg["retries"]["max_attempts"]),
    )
    if restored:
        logger.info("Re-queued after an interrupted run: %s", ", ".join(restored))
        state.save()

    ff_dir = find_opls_directory()
    if ff_dir is None:
        logger.error("OPLS-AA not found; the campaign cannot start")
        return 2
    types = load_opls_types(ff_dir)
    runner = GROMACSRunner(discover_tool("gromacs", ToolConfig(executable="gmx")),
                           enabled=True, extra_env=gmx_env(),
                           default_timeout_s=_stage_timeout_s(cfg))
    graph = ProvenanceGraph()
    # No deadline in an open-ended campaign; stages get a generous per-run cap instead.
    deadline = (state.started_monotonic + budget * 3600.0
                if budget is not None else float("inf"))
    minimum_points = int(cfg["active_learning"]["min_validated_points_before_exploration"])
    model: dict[str, Any] = {"status": "not yet trained"}

    write_status(state, cfg, root, "starting", model, budget)

    while True:
        state.iteration += 1
        validated = [r for r in state.results.values() if r.scientifically_usable]
        decision = policy.evaluate(
            elapsed_hours=state.elapsed_hours(),
            disk_free_gb=disk_free_gb(root),
            queue_depth=len(state.queue) or 1,   # refilled below if Stage B can extend
            n_validated=len(validated),
            consecutive_failures=consecutive_failures,
        )
        if not decision.should_continue:
            state.stopped_reason = f"{decision.status.value}: {decision.reason}"
            logger.info("Stopping: %s", state.stopped_reason)
            break
        if not state.queue:
            queued_or_done = set(state.results) | {c.name for c in state.queue}
            if len(validated) >= minimum_points:
                known = {r.candidate for r in validated} | set(state.results)
                for name, smiles, parent, reason in STAGE_B:
                    if name in queued_or_done or parent not in known:
                        continue
                    state.queue.append(CandidateSpec(
                        name=name, repeat_unit_smiles=smiles, origin="generated",
                        parent=parent, mutation="side-chain / backbone substitution",
                        design_reason=reason))
                    break
            if not state.queue:
                status = (CampaignStatus.WAITING_FOR_INPUT
                          if len(validated) < minimum_points
                          else CampaignStatus.COMPLETED)
                state.stopped_reason = (
                    f"{status.value}: only {len(validated)} validated point(s); Stage B "
                    f"requires {minimum_points} before proposing new chemistry"
                    if len(validated) < minimum_points
                    else f"{status.value}: candidate space exhausted"
                )
                break

        candidate = state.queue.pop(0)
        state.attempts[candidate.name] = state.attempts.get(candidate.name, 0) + 1
        typing = type_repeat_unit(candidate.repeat_unit_smiles, types=types)
        if not typing.usable:
            state.failed.append(candidate.name)
            state.record_decision({
                "iteration": state.iteration, "candidate": candidate.name,
                "action": "rejected_before_simulation",
                "why_this_candidate": candidate.design_reason,
                "reason": typing.reason, "status": typing.status.value})
            state.save()
            continue

        remaining = state.remaining_hours(budget) if budget is not None else float("inf")
        state.record_decision({
            "iteration": state.iteration, "candidate": candidate.name, "action": "simulate",
            "why_this_candidate": candidate.design_reason,
            "why_this_simulation": ("melt density at 300 K / 1 bar is the campaign's primary "
                                    "observable and is measurable within the window"),
            "why_now": (f"{len(validated)} validated point(s) so far; "
                        + ("open-ended campaign" if budget is None
                           else f"{remaining:.1f} h remain")),
            "uncertainty_to_reduce": ("how side-chain length and branching shift amorphous "
                                      "polyolefin packing density"),
            "design_decision_at_stake": "which structural motif to pursue for a target density",
            "origin": candidate.origin, "parent": candidate.parent,
            "force_field": cfg["force_field"]["name"],
            "n_replicas": cfg["simulation"]["replicas"]})
        write_status(state, cfg, root, f"running {candidate.name}", model, budget)
        logger.info("=== iteration %d: %s ===", state.iteration, candidate.name)

        def checkpoint_partial(partial: ExperimentResult, name: str = candidate.name) -> None:
            """Persist after every replica, so a crash loses one replica, not a candidate."""
            state.results[name] = partial
            state.save()

        result = run_candidate(candidate, cfg, root, runner, types, str(ff_dir), graph,
                               deadline, checkpoint=checkpoint_partial)
        result.descriptors = descriptors_for(candidate.repeat_unit_smiles)
        state.results[candidate.name] = result
        (state.completed if result.scientifically_usable else state.failed).append(candidate.name)
        consecutive_failures = 0 if result.scientifically_usable else consecutive_failures + 1
        state.record_decision({
            "iteration": state.iteration, "candidate": candidate.name, "action": "judged",
            "gate_status": result.gate_status,
            "scientifically_usable": result.scientifically_usable,
            "density_kg_m3": result.density_kg_m3, "uncertainty": result.density_uncertainty,
            "diagnostics": result.diagnostics,
            "wall_seconds": round(result.wall_seconds, 1)})
        state.save()

        model = train_model(state, root)
        write_status(state, cfg, root, "selecting next candidate", model, budget)

    state.save()
    write_status(state, cfg, root, "stopped", model, budget)
    logger.info("Campaign stopped: %s", state.stopped_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
