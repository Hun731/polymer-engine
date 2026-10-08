"""GROMACS execution and replica analysis.

The equilibration executor refuses to run against a system that has not passed
validation, runs each stage in order, and stops at the first failure with the actual
GROMACS error attached.  It never reports success for a stage it did not run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from polymer_engine.analysis.convergence import (
    analyse_series,
    combine_replicas,
    convergence_gates,
    replica_agreement_gate,
)
from polymer_engine.core.config import AnalysisDefaults
from polymer_engine.core.errors import SystemValidationError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import (
    Action,
    ActionStatus,
    Determination,
    ExperimentResult,
    GateReport,
    GateResult,
    GateStatus,
    Measurement,
    Observation,
)
from polymer_engine.executors.base import Executor
from polymer_engine.local.runner import GROMACSRunner
from polymer_engine.simulation.formats import read_xvg
from polymer_engine.simulation.mdp import STAGE_ORDER
from polymer_engine.simulation.system import validate_system

logger = get_logger("executors.gromacs")

#: Stage -> (input structure, mdp file, output prefix)
#: Stages that must be present, and the structure each takes when it is the only
#: predecessor. The chain is resolved at run time by :func:`stage_chain`, because an
#: optional stage changes what the stage after it reads.
REQUIRED_STAGES: tuple[str, ...] = ("em", "nvt", "npt", "prod")

#: Stages that run only when their mdp was generated. ``anneal`` is optional: a
#: configuration with ``anneal_ns = 0`` produces no anneal.mdp and the protocol is the
#: original four.
OPTIONAL_STAGES: frozenset[str] = frozenset({"anneal"})

STAGE_INPUTS: dict[str, tuple[str, str, str]] = {
    "em": ("system.gro", "em.mdp", "em"),
    "nvt": ("em.gro", "nvt.mdp", "nvt"),
    "anneal": ("nvt.gro", "anneal.mdp", "anneal"),
    "npt": ("nvt.gro", "npt.mdp", "npt"),
    "prod": ("npt.gro", "prod.mdp", "prod"),
}


def stage_chain(workdir: Path) -> list[tuple[str, str, str, str]]:
    """Resolve which stages run and what each one reads.

    Derived from the mdp files actually present rather than from a fixed table. A second
    hard-coded stage list is how an optional stage ends up in the ordering but not in
    the input mapping -- which is exactly what adding the anneal exposed here.

    Returns ``(stage, input_structure, mdp, output_prefix)``.
    """
    chain: list[tuple[str, str, str, str]] = []
    previous_output: str | None = None
    for stage in STAGE_ORDER:
        _default_input, mdp, prefix = STAGE_INPUTS[stage]
        if stage in OPTIONAL_STAGES and not (workdir / mdp).exists():
            continue
        structure = ("system.gro" if previous_output is None
                     else f"{previous_output}.gro")
        chain.append((stage, structure, mdp, prefix))
        previous_output = prefix
    return chain

#: Energy terms extracted after the run, with the units GROMACS reports them in.
ENERGY_TERMS: dict[str, str] = {
    "Temperature": "K",
    "Pressure": "bar",
    "Density": "kg/m^3",
    "Potential": "kJ/mol",
    "Volume": "1",
}


class GromacsEquilibrateExecutor(Executor):
    """Runs EM -> NVT -> NPT -> production for one replica."""

    kind = "gromacs_equilibrate"

    def __init__(self, runner: GROMACSRunner, *, use_gpu: bool = False, stage_timeout_s: float | None = None) -> None:
        self.runner = runner
        self.use_gpu = use_gpu
        self.stage_timeout_s = stage_timeout_s

    def run(self, action: Action) -> ExperimentResult:
        workdir = Path(action.inputs.get("workdir", ""))
        if not workdir.is_dir():
            return self.blocked(action, f"Replica directory does not exist: {workdir}")

        # -- preconditions ---------------------------------------------
        report = validate_system(workdir)
        if not report.promotable:
            failures = [g.message for g in report.gates if g.status.blocks_promotion]
            result = self.blocked(
                action, "Replica did not pass system validation", failures=failures
            )
            result.report = report
            return result

        missing = [STAGE_INPUTS[stage][1] for stage in REQUIRED_STAGES
                   if not (workdir / STAGE_INPUTS[stage][1]).exists()]
        if missing:
            return self.blocked(action, "Missing GROMACS input files", missing=missing)

        if not self.runner.enabled:
            return self.dry_run(
                action,
                f"Execution disabled; validated inputs for {workdir.name} but ran nothing",
                artifacts=[str(workdir / mdp) for _s, _i, mdp, _p in stage_chain(workdir)],
            )
        if not self.runner.available():
            return self.blocked(
                action, "GROMACS is not usable", issues=self.runner.status.issues
            )

        # -- run ---------------------------------------------------------
        artifacts: list[str] = []
        observations: list[Observation] = []
        gates: list[GateResult] = []

        for stage, structure, mdp, prefix in stage_chain(workdir):
            if not (workdir / structure).exists() and stage != "em":
                return self._stage_failure(
                    action, stage, f"input structure {structure} was not produced by the previous stage",
                    artifacts, observations, gates,
                )
            tpr = workdir / f"{prefix}.tpr"
            grompp = self.runner.grompp(
                mdp=mdp, structure=structure, topology="topol.top", output=f"{prefix}.tpr", cwd=workdir
            )
            observations.append(self._command_observation(action, f"{stage}.grompp", grompp))
            if not grompp.succeeded:
                return self._stage_failure(
                    action, stage, f"grompp failed: {grompp.stderr[-600:] or grompp.error}",
                    artifacts, observations, gates,
                )
            artifacts.append(str(tpr))

            mdrun = self.runner.mdrun(
                deffnm=prefix, cwd=workdir, use_gpu=self.use_gpu, timeout_s=self.stage_timeout_s
            )
            observations.append(self._command_observation(action, f"{stage}.mdrun", mdrun))
            if not mdrun.succeeded:
                return self._stage_failure(
                    action, stage, f"mdrun failed: {mdrun.stderr[-600:] or mdrun.error}",
                    artifacts, observations, gates,
                )
            for suffix in (".gro", ".edr", ".log", ".xtc", ".cpt"):
                produced = workdir / f"{prefix}{suffix}"
                if produced.exists():
                    artifacts.append(str(produced))
            gates.append(
                GateResult(
                    gate=f"{stage}:completed",
                    status=GateStatus.PASS,
                    message=f"{stage} completed in {mdrun.duration_s:.1f}s",
                    value=mdrun.duration_s,
                    units="1",
                )
            )

        # -- extract observables ----------------------------------------
        artifacts.extend(self._extract_energies(workdir))
        return ExperimentResult(
            action_id=action.id,
            status=ActionStatus.SUCCEEDED,
            execution_mode="real",
            artifacts=artifacts,
            observations=observations,
            report=GateReport(name=f"{self.kind}:stages", gates=gates),
            summary=f"Completed all {len(STAGE_ORDER)} stages for {workdir.name}",
        )

    def _extract_energies(self, workdir: Path) -> list[str]:
        """Pull the standard observables out of the production energy file."""
        edr = workdir / "prod.edr"
        if not edr.exists():
            return []
        produced: list[str] = []
        for term in ENERGY_TERMS:
            output = f"{term.lower()}.xvg"
            result = self.runner.energy(edr="prod.edr", terms=[term], output=output, cwd=workdir)
            if result.succeeded and (workdir / output).exists():
                produced.append(str(workdir / output))
            else:
                logger.warning("Could not extract %s from %s", term, edr)
        return produced

    @staticmethod
    def _command_observation(action: Action, metric: str, result: Any) -> Observation:
        return Observation(
            action_id=action.id,
            campaign_id=action.campaign_id,
            measurement=Measurement(
                name=f"{metric}.returncode",
                value=float(result.returncode) if result.returncode is not None else None,
                units="1",
                determination=Determination.KNOWN if result.returncode is not None else Determination.UNKNOWN,
                method="subprocess exit status",
            ),
            provenance_id=None,
        )

    def _stage_failure(
        self,
        action: Action,
        stage: str,
        reason: str,
        artifacts: list[str],
        observations: list[Observation],
        gates: list[GateResult],
    ) -> ExperimentResult:
        gates.append(
            GateResult(gate=f"{stage}:completed", status=GateStatus.FAIL, message=reason)
        )
        logger.error("Stage %s failed: %s", stage, reason)
        return ExperimentResult(
            action_id=action.id,
            status=ActionStatus.FAILED,
            execution_mode="real",
            artifacts=artifacts,
            observations=observations,
            report=GateReport(name=f"{self.kind}:stages", gates=gates),
            summary=f"Failed at stage {stage}",
            error=reason,
        )


class ReplicaAnalysisExecutor(Executor):
    """Analyses replica outputs and applies the convergence/agreement gates."""

    kind = "analyze_replicas"

    def __init__(self, defaults: AnalysisDefaults | None = None, *, required_replicas: int = 3) -> None:
        self.defaults = defaults or AnalysisDefaults()
        self.required_replicas = required_replicas

    def run(self, action: Action) -> ExperimentResult:
        replica_dirs = [Path(p) for p in action.inputs.get("replica_dirs", [])]
        metrics: Sequence[str] = action.inputs.get("metrics", ["density", "temperature", "pressure"])
        workdir = Path(action.inputs.get("workdir", "."))

        if not replica_dirs:
            return self.blocked(action, "No replica directories were supplied")

        per_metric: dict[str, list[Measurement]] = {m: [] for m in metrics}
        gates: list[GateResult] = []
        observations: list[Observation] = []
        missing: dict[str, list[str]] = {}

        for directory in replica_dirs:
            for metric in metrics:
                path = self._find_xvg(directory, metric)
                if path is None:
                    missing.setdefault(metric, []).append(directory.name)
                    continue
                try:
                    data = read_xvg(path)
                except SystemValidationError as exc:
                    missing.setdefault(metric, []).append(f"{directory.name} (unreadable: {exc.message})")
                    continue
                analysis = analyse_series(
                    data.y,
                    name=metric,
                    units=self._units_for(metric),
                    times_ps=data.x,
                    defaults=self.defaults,
                )
                gates.extend(convergence_gates(analysis, defaults=self.defaults))
                per_metric[metric].append(analysis.production)
                observations.append(
                    Observation(
                        action_id=action.id,
                        campaign_id=action.campaign_id,
                        measurement=analysis.production,
                        artifact_ids=[str(path)],
                    )
                )

        # A replica that produced no data must not be silently excluded from the gate.
        for metric, absent in missing.items():
            gates.append(
                GateResult(
                    gate=f"{metric}:data_present",
                    status=GateStatus.FAIL,
                    message=f"{len(absent)} replica(s) produced no {metric} data: {absent}",
                    evidence={"replicas_missing_data": absent},
                )
            )

        for metric, measurements in per_metric.items():
            agreement = combine_replicas(measurements, name=metric, units=self._units_for(metric))
            gates.append(replica_agreement_gate(agreement, required_replicas=self.required_replicas))
            if agreement.combined.determination is Determination.KNOWN:
                observations.append(
                    Observation(
                        action_id=action.id,
                        campaign_id=action.campaign_id,
                        measurement=agreement.combined,
                    )
                )

        report = GateReport(name="replica_analysis", gates=gates)
        artifact = self._write_report(workdir, report, per_metric)
        status = ActionStatus.SUCCEEDED if report.promotable else ActionStatus.FAILED
        return ExperimentResult(
            action_id=action.id,
            status=status,
            execution_mode="real",
            artifacts=[str(artifact)],
            observations=observations,
            report=report,
            summary=report.summary(),
        )

    @staticmethod
    def _units_for(metric: str) -> str:
        return {
            "density": "kg/m^3",
            "temperature": "K",
            "pressure": "bar",
            "potential": "kJ/mol",
            "volume": "1",
        }.get(metric, "1")

    @staticmethod
    def _find_xvg(directory: Path, metric: str) -> Path | None:
        for name in (f"{metric}.xvg", f"{metric[:4]}.xvg"):
            candidates = sorted(directory.rglob(name))
            if candidates:
                return candidates[-1]
        return None

    @staticmethod
    def _write_report(workdir: Path, report: GateReport, per_metric: dict[str, list[Measurement]]) -> Path:
        target = workdir / "analysis" / "replica_analysis.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "status": report.status.value,
                    "promotable": report.promotable,
                    "summary": report.summary(),
                    "gates": [g.model_dump(mode="json") for g in report.gates],
                    "per_replica": {
                        metric: [m.model_dump(mode="json") for m in values]
                        for metric, values in per_metric.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return target


__all__ = ["ENERGY_TERMS", "STAGE_INPUTS", "GromacsEquilibrateExecutor", "ReplicaAnalysisExecutor"]
