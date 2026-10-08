"""Umbrella sampling execution and adaptive window refinement.

Planning windows is not the hard part; deciding whether to run them, and what to do
when they come back badly overlapped, is.

**Justification is mandatory.**  The planner can always propose umbrella sampling; that
is not a reason to spend a week of GPU time on it.  A PMF along a coordinate nobody can
interpret is a number without a meaning, so :class:`UmbrellaJustification` must be
supplied and complete before any window runs.  Without it the campaign reports
``REQUIRES_EXPERT_DECISION`` and stops.

**Adaptive refinement.**  After a round, overlap is diagnosed per adjacent *pair*.  A
gap between windows 4 and 5 leaves the free-energy difference across it undetermined by
the data, so a window is inserted at the midpoint and only the new windows are run.
Refinement is bounded by ``max_adaptive_rounds``: an unbounded loop on a badly chosen
coordinate would insert windows forever.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from polymer_engine.analysis.free_energy import (
    OverlapDiagnostic,
    PmfResult,
    WindowSamples,
    build_pmf,
    diagnose_overlap,
    pmf_gates,
    trim_equilibration,
)
from polymer_engine.core.config import UmbrellaDefaults
from polymer_engine.core.errors import InsufficientDataError, PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination, GateReport, GateResult, GateStatus
from polymer_engine.core.provenance import ProvenanceGraph, canonical_hash
from polymer_engine.simulation.umbrella import (
    ReactionCoordinate,
    Window,
    WindowPlan,
    assign_starting_structures,
    insert_windows,
    plan_windows,
    plumed_input,
    write_windows,
)

logger = get_logger("simulation.umbrella_execution")


class UmbrellaStatus(str, Enum):
    COMPLETED = "COMPLETED"
    #: Windows ran but the PMF is not trustworthy.
    INSUFFICIENT_SAMPLING = "INSUFFICIENT_SAMPLING"
    #: Refinement hit its limit with gaps remaining.
    REFINEMENT_EXHAUSTED = "REFINEMENT_EXHAUSTED"
    #: No justification was supplied, so nothing was run.
    REQUIRES_EXPERT_DECISION = "REQUIRES_EXPERT_DECISION"
    FAILED = "FAILED"
    #: Execution is disabled; inputs were written but nothing ran.
    NOT_EXECUTED = "NOT_EXECUTED"


@dataclass
class UmbrellaJustification:
    """Why umbrella sampling is the right method for this question.

    Every field is required. Umbrella sampling is expensive and its output is easy to
    over-interpret, so the engine will not start one on the planner's say-so alone.
    """

    question: str
    reaction_coordinate: str
    physical_interpretation: str
    expected_observable: str
    reason_for_method: str
    starting_state: str
    endpoint_definition: str
    author: str = ""

    def missing_fields(self) -> list[str]:
        required = (
            "question", "reaction_coordinate", "physical_interpretation",
            "expected_observable", "reason_for_method", "starting_state",
            "endpoint_definition",
        )
        return [name for name in required if not str(getattr(self, name)).strip()]

    @property
    def complete(self) -> bool:
        return not self.missing_fields()

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "reaction_coordinate": self.reaction_coordinate,
            "physical_interpretation": self.physical_interpretation,
            "expected_observable": self.expected_observable,
            "reason_for_method": self.reason_for_method,
            "starting_state": self.starting_state,
            "endpoint_definition": self.endpoint_definition,
            "author": self.author,
            "complete": self.complete,
            "missing_fields": self.missing_fields(),
        }


@dataclass
class WindowRun:
    """The outcome of running one umbrella window."""

    index: int
    center: float
    force_constant: float
    directory: str
    executed: bool = False
    succeeded: bool = False
    colvar_path: str | None = None
    n_samples: int = 0
    n_discarded: int = 0
    mean_cv: float | None = None
    error: str | None = None
    round_added: int = 0
    starting_structure: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "center": self.center,
            "force_constant": self.force_constant,
            "directory": self.directory,
            "executed": self.executed,
            "succeeded": self.succeeded,
            "colvar_path": self.colvar_path,
            "n_samples": self.n_samples,
            "n_discarded": self.n_discarded,
            "mean_cv": self.mean_cv,
            "error": self.error,
            "round_added": self.round_added,
            "starting_structure": self.starting_structure,
        }


@dataclass
class UmbrellaCampaignResult:
    """Everything an umbrella campaign produced."""

    status: UmbrellaStatus
    justification: UmbrellaJustification | None
    plan: WindowPlan | None = None
    runs: list[WindowRun] = field(default_factory=list)
    pmf: PmfResult | None = None
    overlap: OverlapDiagnostic | None = None
    rounds: int = 0
    report: GateReport = field(default_factory=lambda: GateReport(name="umbrella"))
    diagnostics: list[str] = field(default_factory=list)
    inserted_centers: list[float] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def trustworthy(self) -> bool:
        """A PMF may be used only when it exists, is trustworthy, and gates passed."""
        return (
            self.status is UmbrellaStatus.COMPLETED
            and self.pmf is not None
            and self.pmf.trustworthy
            and self.report.promotable
        )

    @property
    def determination(self) -> Determination:
        if self.trustworthy:
            return Determination.KNOWN
        if self.status is UmbrellaStatus.REQUIRES_EXPERT_DECISION:
            return Determination.REQUIRES_VALIDATION
        return Determination.INSUFFICIENT_DATA

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "trustworthy": self.trustworthy,
            "determination": self.determination.value,
            "justification": self.justification.as_dict() if self.justification else None,
            "plan": self.plan.as_dict() if self.plan else None,
            "n_windows": len(self.runs),
            "rounds": self.rounds,
            "inserted_centers": self.inserted_centers,
            "runs": [r.as_dict() for r in self.runs],
            "pmf": self.pmf.as_dict() if self.pmf else None,
            "overlap": self.overlap.as_dict() if self.overlap else None,
            "gate_status": self.report.status.value,
            "gates": [g.model_dump(mode="json") for g in self.report.gates],
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
        }


#: Signature of a window runner. Returns the COLVAR path, or None on failure.
WindowRunner = Callable[[Window, Path], "WindowExecution"]


@dataclass
class WindowExecution:
    """What a window runner reports back."""

    succeeded: bool
    colvar_path: Path | None = None
    error: str | None = None
    executed: bool = True


def read_colvar(path: str | Path, *, column: int = 1) -> np.ndarray:
    """Read the collective-variable column from a PLUMED COLVAR file.

    Blank lines and ``#`` headers are skipped -- PLUMED legitimately re-emits its
    ``#! FIELDS`` header when a run appends to an existing file.

    A **data** row that cannot be read is an error, not something to step over. Silently
    dropping malformed rows shortens the series without saying so, and the loss is not
    random: a COLVAR truncated by a full disk or a killed job loses its *tail*, so the
    surviving samples are exactly the early, least-equilibrated ones. The result still
    passes every finiteness check and simply reports fewer effective samples, which is
    indistinguishable from an honestly short run.
    """
    path = Path(path)
    if not path.exists():
        raise InsufficientDataError("COLVAR file does not exist", path=str(path))
    values: list[float] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) <= column:
            raise InsufficientDataError(
                "COLVAR row has too few columns; the file is truncated or the wrong "
                "column was requested",
                path=str(path), line=number, columns=len(parts), requested_column=column,
            )
        try:
            values.append(float(parts[column]))
        except ValueError as exc:
            raise InsufficientDataError(
                "COLVAR row does not contain a numeric collective variable",
                path=str(path), line=number, value=parts[column],
            ) from exc
    if not values:
        raise InsufficientDataError("COLVAR file contains no usable data", path=str(path))
    return np.asarray(values, dtype=float)


class UmbrellaCampaign:
    """Runs umbrella windows, diagnoses overlap, and refines adaptively."""

    def __init__(
        self,
        coordinate: ReactionCoordinate,
        defaults: UmbrellaDefaults | None = None,
        *,
        temperature_k: float = 300.0,
        graph: ProvenanceGraph | None = None,
    ) -> None:
        self.coordinate = coordinate
        self.defaults = defaults or UmbrellaDefaults()
        self.temperature_k = temperature_k
        self.graph = graph

    def run(
        self,
        *,
        minimum: float,
        maximum: float,
        root: str | Path,
        justification: UmbrellaJustification | None,
        runner: WindowRunner | None = None,
        starting_structures: dict[float, str] | None = None,
        max_rounds: int | None = None,
    ) -> UmbrellaCampaignResult:
        """Plan, run, diagnose and refine until the PMF is trustworthy or the budget runs out."""
        root = Path(root)

        # -- justification gate -----------------------------------------
        if justification is None or not justification.complete:
            missing = justification.missing_fields() if justification else ["the entire justification"]
            report = GateReport(
                name="umbrella",
                gates=[
                    GateResult(
                        gate="umbrella:justification",
                        status=GateStatus.FAIL,
                        message=(
                            "umbrella sampling requires an explicit scientific justification; "
                            f"missing: {', '.join(missing)}"
                        ),
                        evidence={"missing_fields": missing},
                    )
                ],
            )
            logger.info("Umbrella campaign refused: justification incomplete (%s)", missing)
            return UmbrellaCampaignResult(
                status=UmbrellaStatus.REQUIRES_EXPERT_DECISION,
                justification=justification,
                report=report,
                diagnostics=[
                    "A PMF along an unjustified reaction coordinate is a number without a "
                    "meaning. Supply an UmbrellaJustification before running windows."
                ],
            )

        plan = plan_windows(
            self.coordinate,
            minimum=minimum,
            maximum=maximum,
            temperature_k=self.temperature_k,
            defaults=self.defaults,
        )
        result = UmbrellaCampaignResult(
            status=UmbrellaStatus.NOT_EXECUTED,
            justification=justification,
            plan=plan,
            provenance={
                "coordinate": self.coordinate.as_dict(),
                "temperature_k": self.temperature_k,
                "justification_hash": canonical_hash(justification.as_dict()),
            },
        )
        result.diagnostics.extend(plan.warnings)

        write_windows(plan, root)
        (root / "justification.json").write_text(
            json.dumps(justification.as_dict(), indent=2), encoding="utf-8"
        )

        if runner is None:
            result.status = UmbrellaStatus.NOT_EXECUTED
            result.report.gates.append(
                GateResult(
                    gate="umbrella:executed",
                    status=GateStatus.INCONCLUSIVE,
                    message="no window runner was supplied; inputs were written but nothing ran",
                )
            )
            return result

        budget = max_rounds if max_rounds is not None else self.defaults.max_adaptive_rounds
        samples: dict[int, WindowSamples] = {}
        current_plan = plan
        available = dict(starting_structures or {})

        if available:
            assignments = assign_starting_structures(plan, available)
            unassigned = [i for i, path in assignments.items() if path is None]
            if unassigned:
                result.diagnostics.append(
                    f"{len(unassigned)} window(s) have no starting structure within one sigma of "
                    "their restraint centre; they will need a longer equilibration and may be "
                    "trapped in the wrong basin"
                )
                result.report.gates.append(
                    GateResult(
                        gate="umbrella:starting_structures",
                        status=GateStatus.WARN,
                        message=(
                            f"{len(unassigned)} of {plan.n_windows} windows start further than one "
                            "sigma from their restraint centre"
                        ),
                        value=float(len(unassigned)),
                        units="1",
                    )
                )
            else:
                result.report.gates.append(
                    GateResult(
                        gate="umbrella:starting_structures",
                        status=GateStatus.PASS,
                        message="every window starts within one sigma of its restraint centre",
                    )
                )
        else:
            # A warning rather than a blocker: starting each window from a
            # pre-equilibrated structure saves time and avoids basin trapping, but the
            # evidence that actually decides whether a PMF can be trusted is window
            # overlap and half-split convergence, both of which are checked below.
            result.report.gates.append(
                GateResult(
                    gate="umbrella:starting_structures",
                    status=GateStatus.WARN,
                    message=(
                        "no equilibrated starting structures were supplied; each window relaxes "
                        "into its restraint from whatever configuration the runner uses, so its "
                        "equilibration period will be longer"
                    ),
                )
            )

        for round_index in range(budget + 1):
            result.rounds = round_index + 1
            pending = [w for w in current_plan.windows if w.index not in samples]
            assignments = assign_starting_structures(current_plan, available) if available else {}
            for window in pending:
                run = self._run_window(
                    window, root, runner, round_index,
                    starting_structure=assignments.get(window.index),
                )
                result.runs.append(run)
                if run.succeeded and run.colvar_path:
                    try:
                        raw = read_colvar(run.colvar_path)
                    except InsufficientDataError as exc:
                        run.succeeded = False
                        run.error = str(exc)
                        continue
                    trimmed, discarded = trim_equilibration(
                        raw, self.defaults.equilibration_fraction
                    )
                    run.n_samples = int(trimmed.size)
                    run.n_discarded = discarded
                    run.mean_cv = float(trimmed.mean()) if trimmed.size else None
                    samples[window.index] = WindowSamples(
                        index=window.index,
                        center=window.center,
                        force_constant=window.force_constant,
                        values=trimmed,
                        units=self.coordinate.units,
                        discarded=discarded,
                    )

            usable = [s for s in samples.values() if s.n > 0]
            if len(usable) < 2:
                result.status = UmbrellaStatus.FAILED
                result.report.gates.append(
                    GateResult(
                        gate="umbrella:windows_ran",
                        status=GateStatus.FAIL,
                        message=f"only {len(usable)} window(s) produced usable samples",
                    )
                )
                return result

            result.overlap = diagnose_overlap(usable, min_overlap=self.defaults.min_pair_overlap)
            if result.overlap.sufficient or round_index == budget:
                break

            gaps = result.overlap.gaps
            logger.info(
                "Umbrella round %d: %d under-overlapped pair(s); inserting windows",
                round_index + 1, len(gaps),
            )
            refined = insert_windows(current_plan, gaps)
            new_centers = [
                w.center for w in refined.windows
                if not any(math.isclose(w.center, s.center, abs_tol=1e-9) for s in samples.values())
            ]
            result.inserted_centers.extend(new_centers)
            # Re-index so inserted windows get fresh indices and existing samples survive.
            current_plan = self._reindex(refined, samples)
            write_windows(current_plan, root)

        # -- final PMF ---------------------------------------------------
        usable = [s for s in samples.values() if s.n > 0]
        try:
            result.pmf = build_pmf(
                usable,
                temperature_k=self.temperature_k,
                defaults=self.defaults,
                bootstrap_samples=self.defaults.bootstrap_samples,
            )
        except InsufficientDataError as exc:
            result.status = UmbrellaStatus.FAILED
            result.diagnostics.append(str(exc))
            return result

        # Keep the gates recorded during execution (starting structures, and any
        # early diagnostics); replacing the report here would silently discard them.
        pmf_report = pmf_gates(result.pmf, defaults=self.defaults)
        result.report = GateReport(
            name="umbrella",
            gates=[
                GateResult(
                    gate="umbrella:justification",
                    status=GateStatus.PASS,
                    message="a complete scientific justification was supplied",
                    evidence={"question": justification.question},
                ),
                *result.report.gates,
                *pmf_report.gates,
            ],
        )
        result.report.gates.append(
            GateResult(
                gate="umbrella:refinement",
                status=GateStatus.PASS
                if result.overlap and result.overlap.sufficient
                else GateStatus.FAIL,
                message=(
                    f"overlap satisfied after {result.rounds} round(s), "
                    f"{len(result.inserted_centers)} window(s) inserted"
                    if result.overlap and result.overlap.sufficient
                    else f"gaps remain after {result.rounds} round(s); the refinement budget is exhausted"
                ),
                value=float(len(result.inserted_centers)),
                units="1",
            )
        )

        if result.pmf.trustworthy and result.report.promotable:
            result.status = UmbrellaStatus.COMPLETED
        elif result.overlap and not result.overlap.sufficient:
            result.status = UmbrellaStatus.REFINEMENT_EXHAUSTED
        else:
            result.status = UmbrellaStatus.INSUFFICIENT_SAMPLING
        result.diagnostics.extend(result.pmf.problems)

        if self.graph is not None:
            # Record the evidence *before* the conclusion drawn from it. Registering only
            # the PMF leaves a graph in which nothing file-backed can be verified, so a
            # later `verify_all()` cannot tell whether the COLVARs behind a published
            # free energy were altered after the fact.
            parents = self._record_window_provenance(usable, root)
            self.graph.register_derived(
                artifact_id=f"pmf_{canonical_hash(justification.as_dict())[:16]}",
                kind="pmf",
                parents=parents,
                parameters={
                    "coordinate": self.coordinate.as_dict(),
                    "temperature_k": self.temperature_k,
                    "n_windows": len(usable),
                    "rounds": result.rounds,
                },
                payload=[float(v) for v in result.pmf.pmf if math.isfinite(v)],
                units={"energy": "kJ/mol", "length": self.coordinate.units},
                validation_state=(
                    Determination.KNOWN if result.trustworthy else Determination.REQUIRES_VALIDATION
                ),
                notes=result.status.value,
            )
        return result

    def _record_window_provenance(
        self, windows: Sequence[Any], root: Path
    ) -> list[str]:
        """Hash every per-window input and output that the PMF depends on.

        Returns the artifact ids to use as the PMF's parents. Files that a window never
        produced are skipped rather than registered as empty: a provenance record must
        describe what exists, not what was intended.
        """
        if self.graph is None:
            return []
        parents: list[str] = []
        for window in windows:
            index = getattr(window, "index", None)
            if index is None:
                continue
            directory = Path(root) / f"window_{index:03d}"
            for filename, kind in (("COLVAR", "colvar"), ("plumed.dat", "plumed_input")):
                path = directory / filename
                if not path.is_file():
                    continue
                try:
                    artifact = self.graph.register_file(
                        path,
                        kind=kind,
                        artifact_id=f"{kind}_window_{index:03d}_{canonical_hash(str(path))[:12]}",
                        parameters={
                            "window_index": index,
                            "center": getattr(window, "center", None),
                            "force_constant": getattr(window, "force_constant", None),
                            "coordinate": self.coordinate.name,
                            "units": self.coordinate.units,
                        },
                    )
                except PolymerEngineError as exc:  # unreadable/vanished between run and record
                    logger.warning("Could not record provenance for %s: %s", path, exc)
                    continue
                parents.append(artifact.artifact_id)
        return parents

    def _run_window(
        self,
        window: Window,
        root: Path,
        runner: WindowRunner,
        round_index: int,
        *,
        starting_structure: str | None = None,
    ) -> WindowRun:
        directory = root / f"window_{window.index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "plumed.dat").write_text(
            plumed_input(self.coordinate, window), encoding="utf-8"
        )
        run = WindowRun(
            index=window.index,
            center=window.center,
            force_constant=window.force_constant,
            directory=str(directory),
            round_added=round_index,
            starting_structure=starting_structure,
        )
        if starting_structure:
            source = Path(starting_structure)
            if source.is_file():
                shutil.copy2(source, directory / "start.gro")
            else:
                run.error = f"starting structure not found: {source}"
                return run
        try:
            execution = runner(window, directory)
        except Exception as exc:
            run.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Umbrella window %d failed", window.index)
            return run
        run.executed = execution.executed
        run.succeeded = execution.succeeded
        run.colvar_path = str(execution.colvar_path) if execution.colvar_path else None
        run.error = execution.error
        return run

    @staticmethod
    def _reindex(plan: WindowPlan, samples: dict[int, WindowSamples]) -> WindowPlan:
        """Renumber a refined plan so existing samples keep their window indices."""
        by_center = {round(s.center, 9): index for index, s in samples.items()}
        next_index = max(samples) + 1 if samples else 0
        windows: list[Window] = []
        for window in sorted(plan.windows, key=lambda w: w.center):
            key = round(window.center, 9)
            if key in by_center:
                index = by_center[key]
            else:
                index = next_index
                next_index += 1
            windows.append(
                Window(
                    index=index,
                    center=window.center,
                    force_constant=window.force_constant,
                    units=window.units,
                )
            )
        return WindowPlan(
            coordinate=plan.coordinate,
            windows=windows,
            temperature_k=plan.temperature_k,
            spacing=plan.spacing,
            sigma=plan.sigma,
            recommended_spacing=plan.recommended_spacing,
            window_ns=plan.window_ns,
            equilibration_fraction=plan.equilibration_fraction,
            warnings=plan.warnings,
        )


def make_gromacs_plumed_runner(
    runner: Any,
    *,
    topology: str = "topol.top",
    structure_for: Callable[[Window], str] | None = None,
    mdp: str = "umbrella.mdp",
    timeout_s: float | None = None,
) -> WindowRunner:
    """A window runner that drives ``gmx grompp`` + ``gmx mdrun -plumed``.

    Requires GROMACS built with PLUMED support. When execution is disabled the runner
    reports ``executed=False`` rather than a success, in line with the rest of the engine.
    """

    def run_window(window: Window, directory: Path) -> WindowExecution:
        structure = structure_for(window) if structure_for else "start.gro"
        prefix = f"window_{window.index:03d}"
        grompp = runner.grompp(
            mdp=mdp, structure=structure, topology=topology, output=f"{prefix}.tpr", cwd=directory
        )
        if not grompp.executed:
            return WindowExecution(succeeded=False, executed=False, error=grompp.error)
        if not grompp.succeeded:
            return WindowExecution(
                succeeded=False, error=f"grompp failed: {grompp.stderr[-400:] or grompp.error}"
            )
        mdrun = runner.mdrun(
            deffnm=prefix, cwd=directory, plumed="plumed.dat", timeout_s=timeout_s
        )
        if not mdrun.succeeded:
            return WindowExecution(
                succeeded=False, error=f"mdrun failed: {mdrun.stderr[-400:] or mdrun.error}"
            )
        colvar = directory / "COLVAR"
        if not colvar.exists():
            return WindowExecution(
                succeeded=False, error="mdrun completed but PLUMED wrote no COLVAR file"
            )
        return WindowExecution(succeeded=True, colvar_path=colvar)

    return run_window


__all__ = [
    "UmbrellaCampaign",
    "UmbrellaCampaignResult",
    "UmbrellaJustification",
    "UmbrellaStatus",
    "WindowExecution",
    "WindowRun",
    "WindowRunner",
    "make_gromacs_plumed_runner",
    "read_colvar",
]
