"""The 96-hour autonomous polyolefin density campaign.

State lives on disk, not in this process.  Every finished stage is checkpointed, so the
campaign survives a terminal disconnect, an agent restart, or a killed job: rerunning
the driver resumes from the last valid checkpoint rather than starting over.

The loop is deliberately boring in structure -- select, build, run, judge, learn -- and
all of the interesting behaviour is in the refusals.  A simulation that finishes is not
a result; a result that fails a convergence gate does not update the model; a candidate
the force field cannot type never reaches the queue.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polymer_engine.analysis.convergence import analyse_series
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination
from polymer_engine.core.provenance import ProvenanceGraph, sha256_file
from polymer_engine.simulation.formats import read_xvg

logger = get_logger("orchestrator.density_campaign")

#: Written after every meaningful action, and read on startup to resume.
CHECKPOINT = "campaign_state.json"
STATUS_JSON = "campaign_status.json"
STATUS_MD = "campaign_status.md"
DECISIONS = "decision_log.jsonl"


@dataclass
class CandidateSpec:
    """One polymer the campaign may investigate."""

    name: str
    repeat_unit_smiles: str
    origin: str                       # "dataset" or "generated"
    parent: str | None = None
    mutation: str | None = None
    design_reason: str = ""
    experimental_density_kg_m3: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplicaResult:
    replica: int
    seed: int
    directory: str
    succeeded: bool = False
    stage_failed: str | None = None
    error: str | None = None
    density_kg_m3: float | None = None
    density_stderr: float | None = None
    n_frames: int = 0
    effective_samples: float | None = None
    statistical_inefficiency: float | None = None
    equilibration_index: int | None = None
    #: STATIONARY, EQUILIBRATING or ORDERING -- why this density did or did not settle.
    #: Per replica, because replicas of one candidate genuinely differ: a badly packed
    #: one is still relaxing while its siblings are already stationary.
    ordering_verdict: str | None = None
    ordering_reason: str | None = None
    wall_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Everything the campaign learned, or failed to learn, about one candidate."""

    candidate: str
    started_at: str
    finished_at: str | None = None
    n_chains: int = 0
    atoms_per_chain: int = 0
    total_atoms: int = 0
    replicas: list[ReplicaResult] = field(default_factory=list)
    density_kg_m3: float | None = None
    density_uncertainty: float | None = None
    gate_status: str = "INCONCLUSIVE"
    scientifically_usable: bool = False
    determination: str = Determination.UNKNOWN.value
    diagnostics: list[str] = field(default_factory=list)
    descriptors: dict[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["replicas"] = [r.as_dict() for r in self.replicas]
        return out


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def read_density_series(xvg: Path) -> tuple[list[float], list[float]]:
    """Time and density columns from a ``gmx energy`` export."""
    data = read_xvg(xvg)
    return [float(v) for v in data.x], [float(v) for v in data.column(0)]


def analyse_replica_density(
    values: list[float], *, min_effective_samples: float, max_drift_fraction: float = 0.02
) -> dict[str, Any]:
    """Autocorrelation-aware statistics for one replica's density series.

    The series is passed whole: :func:`analyse_series` detects equilibration itself and
    discards the transient it finds. Pre-trimming by a fixed fraction would either throw
    away good data or keep part of the NPT compression, and in the second case the
    reported number would be the packing density rather than the equilibrium one.
    """
    if len(values) < 10:
        return {"usable": False, "reason": f"only {len(values)} density frames"}
    analysis = analyse_series(values, name="density", units="kg/m^3")
    production = analysis.production
    equilibration = analysis.equilibration
    effective = production.effective_samples or 0.0
    drift = analysis.drift_fraction
    problems: list[str] = []
    if effective < min_effective_samples:
        problems.append(
            f"{effective:.1f} effective samples after autocorrelation "
            f"({analysis.n_raw} raw frames); need {min_effective_samples:g}"
        )
    if drift is not None and abs(drift) > max_drift_fraction:
        problems.append(
            f"observable is still drifting: {abs(drift) * 100:.3f}% exceeds the "
            f"{max_drift_fraction * 100:.3f}% threshold"
        )
    if production.determination is not Determination.KNOWN:
        problems.append(f"density determination is {production.determination.value}")
    return {
        "usable": not problems,
        "mean": production.value,
        "stderr": production.uncertainty,
        "n_frames": analysis.n_raw,
        "effective_samples": effective,
        "statistical_inefficiency": equilibration.statistical_inefficiency,
        "equilibration_index": equilibration.start_index,
        "drift_fraction": drift,
        "reason": "; ".join(problems),
    }


def combine_replica_densities(
    replicas: list[ReplicaResult], *, chi_square_max: float, required: int
) -> dict[str, Any]:
    """Combine replica means, treating replicas as the independent units.

    The uncertainty is the standard error of the **replica means**, not of the frames.
    Frame-level error would be pseudoreplication: 10,000 correlated frames from three
    boxes are not 10,000 measurements of three boxes.
    """
    usable = [r for r in replicas if r.succeeded and r.density_kg_m3 is not None]
    if len(usable) < required:
        return {
            "usable": False, "status": "INCONCLUSIVE",
            "reason": (f"{len(usable)} replica(s) produced a density but {required} are "
                       f"required; reproducibility is not demonstrated"),
        }
    means = [r.density_kg_m3 for r in usable if r.density_kg_m3 is not None]
    n = len(means)
    grand = sum(means) / n
    variance = sum((m - grand) ** 2 for m in means) / (n - 1)
    stderr = math.sqrt(variance / n)

    # Reduced chi-square against each replica's own statistical error.
    terms = []
    for r in usable:
        if r.density_kg_m3 is None or not r.density_stderr:
            continue
        terms.append(((r.density_kg_m3 - grand) / r.density_stderr) ** 2)
    chi2 = sum(terms) / max(1, len(terms) - 1) if len(terms) > 1 else None

    if chi2 is not None and chi2 > chi_square_max:
        return {
            "usable": False, "status": "FAIL", "mean": grand, "stderr": stderr,
            "chi_square": chi2, "n_replicas": n,
            "reason": (f"replicas disagree beyond their own uncertainties "
                       f"(reduced chi-square {chi2:.2f} > {chi_square_max})"),
        }
    return {
        "usable": True, "status": "PASS", "mean": grand, "stderr": stderr,
        "chi_square": chi2, "n_replicas": n, "reason": "",
        "spread_kg_m3": max(means) - min(means),
    }


class CampaignState:
    """Durable campaign state.  The filesystem is the source of truth."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.started_at: str = utc_now_iso()
        self.started_monotonic: float = time.time()
        self.results: dict[str, ExperimentResult] = {}
        self.queue: list[CandidateSpec] = []
        self.completed: list[str] = []
        self.failed: list[str] = []
        self.decisions: int = 0
        self.iteration: int = 0
        self.stopped_reason: str | None = None
        #: How many times each candidate has been started, so a retry is bounded.
        self.attempts: dict[str, int] = {}

    # -- persistence ----------------------------------------------------
    def save(self) -> Path:
        path = self.root / CHECKPOINT
        payload = {
            "started_at": self.started_at,
            "started_epoch": self.started_monotonic,
            "iteration": self.iteration,
            "decisions": self.decisions,
            "completed": self.completed,
            "failed": self.failed,
            "stopped_reason": self.stopped_reason,
            "attempts": self.attempts,
            "queue": [c.as_dict() for c in self.queue],
            "results": {k: v.as_dict() for k, v in self.results.items()},
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(path)          # atomic: a killed write never truncates the checkpoint
        return path

    @classmethod
    def load(cls, root: Path) -> CampaignState | None:
        path = Path(root) / CHECKPOINT
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = cls(Path(root))
        state.started_at = payload["started_at"]
        state.started_monotonic = payload["started_epoch"]
        state.iteration = payload.get("iteration", 0)
        state.decisions = payload.get("decisions", 0)
        state.completed = payload.get("completed", [])
        state.failed = payload.get("failed", [])
        state.stopped_reason = payload.get("stopped_reason")
        state.attempts = dict(payload.get("attempts", {}))
        state.queue = [CandidateSpec(**c) for c in payload.get("queue", [])]
        for name, raw in payload.get("results", {}).items():
            replicas = [ReplicaResult(**r) for r in raw.pop("replicas", [])]
            state.results[name] = ExperimentResult(**raw, replicas=replicas)
        logger.info("Resumed campaign from %s at iteration %d", path, state.iteration)
        return state

    # -- time -----------------------------------------------------------
    def elapsed_hours(self) -> float:
        return (time.time() - self.started_monotonic) / 3600.0

    def remaining_hours(self, budget_hours: float) -> float:
        return max(0.0, budget_hours - self.elapsed_hours())

    def record_decision(self, decision: dict[str, Any]) -> None:
        self.decisions += 1
        with (self.root / DECISIONS).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"recorded_at": utc_now_iso(), **decision}) + "\n")


def requeue_unfinished(
    state: CampaignState, candidates: dict[str, CandidateSpec], *,
    required_replicas: int, max_attempts: int,
) -> list[str]:
    """Put back any candidate a crash left unfinished.

    A candidate is popped from the queue before it runs, so an interrupted one is
    recorded partial and then skipped forever. That is safe -- it reports INCONCLUSIVE
    rather than a density from one replica -- but over 96 hours any transient silently
    costs a whole candidate.

    Bounded by ``max_attempts`` so a candidate that genuinely cannot converge is retried
    a fixed number of times and then left failed, rather than looping.
    """
    queued = {c.name for c in state.queue}
    restored: list[str] = []
    for name, result in state.results.items():
        if result.scientifically_usable or name in queued or name not in candidates:
            continue
        usable = sum(1 for r in result.replicas if r.succeeded)
        if usable >= required_replicas:
            continue
        if state.attempts.get(name, 0) >= max_attempts:
            continue
        state.queue.insert(0, candidates[name])
        restored.append(name)
        logger.info(
            "Re-queued %s: %d of %d replicas usable, attempt %d of %d",
            name, usable, required_replicas, state.attempts.get(name, 0) + 1, max_attempts,
        )
    return restored


def register_campaign_artifacts(graph: ProvenanceGraph, directory: Path, *,
                                parents: list[str], kind_prefix: str) -> list[str]:
    """Hash the files a stage produced so a later verify can detect tampering."""
    recorded: list[str] = []
    for pattern in ("*.gro", "*.top", "*.mdp", "*.edr", "*.xvg"):
        for path in sorted(directory.glob(pattern)):
            if path.stat().st_size == 0:
                continue
            try:
                artifact = graph.register_file(
                    path, kind=f"{kind_prefix}:{path.suffix.lstrip('.')}",
                    artifact_id=f"{kind_prefix}_{path.stem}_{sha256_file(path)[:10]}",
                    parents=parents,
                )
            except Exception as exc:      # noqa: BLE001 - provenance must never stop science
                logger.warning("Could not record %s: %s", path, exc)
                continue
            recorded.append(artifact.artifact_id)
    return recorded


__all__ = [
    "CHECKPOINT", "DECISIONS", "STATUS_JSON", "STATUS_MD",
    "CampaignState", "CandidateSpec", "ExperimentResult", "ReplicaResult",
    "analyse_replica_density", "combine_replica_densities",
    "read_density_series", "register_campaign_artifacts", "utc_now_iso",
]
