"""Candidate selection: acquisition, Pareto ranking, and diversity.

Selection is where a discovery loop most easily fools itself, so the rules are
enforced rather than assumed:

* A candidate already in the training set is **rejected**, not merely down-weighted.
  "Discovering" something you already measured is the purest form of a leaked result.
* Duplicate candidates are collapsed by canonical polymer id, so one molecule cannot
  occupy several slots in a batch.
* Scores mix quantities with different units, so every component is normalised to
  [0, 1] *within the batch* and the normalisation is reported.  Adding a raw kelvin
  uncertainty to a [0,1] novelty would let the units decide the science.
* Batch selection enforces diversity, because the top-k by score are usually near
  neighbours and would spend the whole budget in one region.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from polymer_engine.core.errors import ScientificError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination
from polymer_engine.discovery.qspr import Prediction, QsprModel

logger = get_logger("discovery.active_learning")

Objective = Literal["maximise", "minimise"]


@dataclass
class Candidate:
    """One thing we might run next."""

    polymer_id: str
    name: str = ""
    features: np.ndarray | None = None
    cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredCandidate:
    candidate: Candidate
    predicted: float | None
    uncertainty: float | None
    novelty: float
    cost: float
    score: float
    in_domain: bool
    components: dict[str, float] = field(default_factory=dict)
    determination: Determination = Determination.KNOWN
    note: str = ""

    @property
    def polymer_id(self) -> str:
        return self.candidate.polymer_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id,
            "name": self.candidate.name,
            "predicted": self.predicted,
            "uncertainty": self.uncertainty,
            "novelty": self.novelty,
            "cost": self.cost,
            "score": self.score,
            "in_domain": self.in_domain,
            "components": self.components,
            "determination": self.determination.value,
            "note": self.note,
        }


@dataclass
class SelectionReport:
    """What was selected and, just as importantly, what was thrown out and why."""

    selected: list[ScoredCandidate] = field(default_factory=list)
    scored: list[ScoredCandidate] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)
    normalisation: dict[str, Any] = field(default_factory=dict)
    strategy: str = ""

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "selected": [c.as_dict() for c in self.selected],
            "n_scored": len(self.scored),
            "rejected": self.rejected,
            "normalisation": self.normalisation,
        }


def _normalise(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Min-max normalise, reporting the range used.

    A constant vector maps to all-zeros: with no variation there is nothing to prefer,
    and mapping it to all-ones would let a uniform component dominate the score.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values), {"min": 0.0, "max": 0.0, "constant": True}
    low, high = float(finite.min()), float(finite.max())
    if high - low <= 1e-12:
        return np.zeros_like(values), {"min": low, "max": high, "constant": True}
    out = (values - low) / (high - low)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0), {"min": low, "max": high, "constant": False}


def compute_novelty(candidate_features: np.ndarray, training_features: np.ndarray) -> np.ndarray:
    """Distance from each candidate to its nearest training point.

    Raw distances, not normalised here: normalisation happens once, in the scorer, so
    the same convention applies to every component.
    """
    if training_features.size == 0:
        return np.full(candidate_features.shape[0], np.inf)
    difference = candidate_features[:, None, :] - training_features[None, :, :]
    return np.sqrt((difference**2).sum(axis=-1)).min(axis=1)


def screen_candidates(
    candidates: Sequence[Candidate], *, training_ids: set[str]
) -> tuple[list[Candidate], dict[str, str]]:
    """Drop candidates that must not be scored, with a reason for each."""
    kept: list[Candidate] = []
    rejected: dict[str, str] = {}
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.polymer_id:
            rejected[candidate.name or "<unnamed>"] = "candidate has no polymer id"
            continue
        if candidate.polymer_id in training_ids:
            rejected[candidate.polymer_id] = "already present in the training set"
            continue
        if candidate.polymer_id in seen:
            rejected[candidate.polymer_id] = "duplicate candidate in this batch"
            continue
        if candidate.features is None:
            rejected[candidate.polymer_id] = "candidate has no feature vector"
            continue
        features = np.asarray(candidate.features, dtype=float)
        if not np.all(np.isfinite(features)):
            rejected[candidate.polymer_id] = "candidate features contain non-finite values"
            continue
        seen.add(candidate.polymer_id)
        kept.append(candidate)
    return kept, rejected


def score_candidates(
    candidates: Sequence[Candidate],
    model: QsprModel,
    training_features: np.ndarray,
    *,
    objective: Objective = "maximise",
    weight_uncertainty: float = 1.0,
    weight_novelty: float = 0.3,
    weight_value: float = 0.5,
    weight_cost: float = 0.2,
    default_cost: float = 1.0,
) -> SelectionReport:
    """Score candidates by exploration value, exploitation value, novelty and cost."""
    report = SelectionReport(strategy="uncertainty+novelty+value-cost")
    kept, rejected = screen_candidates(candidates, training_ids=model.training_ids)
    report.rejected = rejected
    if not kept:
        report.normalisation = {"note": "no candidate survived screening"}
        return report

    features = np.vstack([np.asarray(c.features, dtype=float) for c in kept])
    ids = [c.polymer_id for c in kept]
    predictions: list[Prediction] = model.predict(features, ids)

    predicted = np.array([p.value if p.value is not None else np.nan for p in predictions])
    uncertainty = np.array([p.uncertainty if p.uncertainty is not None else np.nan for p in predictions])
    novelty_raw = compute_novelty(features, np.asarray(training_features, dtype=float))
    costs = np.array([c.cost if c.cost is not None else default_cost for c in kept], dtype=float)

    value_signal = predicted if objective == "maximise" else -predicted
    norm_value, value_stats = _normalise(value_signal)
    norm_uncertainty, uncertainty_stats = _normalise(uncertainty)
    norm_novelty, novelty_stats = _normalise(np.where(np.isfinite(novelty_raw), novelty_raw, np.nan))
    norm_cost, cost_stats = _normalise(costs)

    report.normalisation = {
        "value": value_stats,
        "uncertainty": uncertainty_stats,
        "novelty": novelty_stats,
        "cost": cost_stats,
        "weights": {
            "uncertainty": weight_uncertainty,
            "novelty": weight_novelty,
            "value": weight_value,
            "cost": weight_cost,
        },
    }

    scores = (
        weight_uncertainty * norm_uncertainty
        + weight_novelty * norm_novelty
        + weight_value * norm_value
        - weight_cost * norm_cost
    )

    for index, (candidate, prediction) in enumerate(zip(kept, predictions, strict=True)):
        report.scored.append(
            ScoredCandidate(
                candidate=candidate,
                predicted=prediction.value,
                uncertainty=prediction.uncertainty,
                novelty=float(novelty_raw[index]) if math.isfinite(novelty_raw[index]) else float("inf"),
                cost=float(costs[index]),
                score=float(scores[index]) if math.isfinite(scores[index]) else float("-inf"),
                in_domain=prediction.in_domain,
                components={
                    "value": float(norm_value[index]),
                    "uncertainty": float(norm_uncertainty[index]),
                    "novelty": float(norm_novelty[index]),
                    "cost": float(norm_cost[index]),
                },
                determination=prediction.determination,
                note=prediction.note,
            )
        )
    report.scored.sort(key=lambda c: -c.score)
    return report


def pareto_front(
    scored: Sequence[ScoredCandidate], *, objectives: Sequence[tuple[str, Objective]]
) -> list[ScoredCandidate]:
    """Non-dominated candidates under the given objectives.

    ``objectives`` names attributes of :class:`ScoredCandidate` and the direction each
    should go.  A candidate is on the front when nothing else is at least as good on
    every objective and strictly better on one.
    """
    if not scored:
        return []
    matrix = np.empty((len(scored), len(objectives)), dtype=float)
    for j, (attribute, direction) in enumerate(objectives):
        column = np.array(
            [_objective_value(c, attribute) for c in scored], dtype=float
        )
        matrix[:, j] = column if direction == "maximise" else -column
    # Unknown components must not dominate; treat them as worst-case.
    matrix = np.where(np.isfinite(matrix), matrix, -np.inf)

    front: list[ScoredCandidate] = []
    for i in range(len(scored)):
        dominated = np.all(matrix >= matrix[i], axis=1) & np.any(matrix > matrix[i], axis=1)
        if not dominated.any():
            front.append(scored[i])
    return front


def _objective_value(candidate: ScoredCandidate, attribute: str) -> float:
    value = getattr(candidate, attribute, None)
    if value is None:
        return float("nan")
    return float(value)


def select_diverse_batch(
    scored: Sequence[ScoredCandidate], *, batch_size: int, min_separation: float = 0.0
) -> list[ScoredCandidate]:
    """Greedy max-min diversity selection over the scored candidates.

    Pure top-k tends to pick a cluster of near-identical structures; this takes the
    best candidate, then repeatedly the highest-scoring one that is farthest from
    everything already chosen.
    """
    if batch_size <= 0:
        raise ScientificError("Batch size must be positive", batch_size=batch_size)
    usable = [c for c in scored if c.candidate.features is not None and math.isfinite(c.score)]
    if not usable:
        return []
    ordered = sorted(usable, key=lambda c: -c.score)
    features = {c.polymer_id: np.asarray(c.candidate.features, dtype=float) for c in ordered}

    chosen: list[ScoredCandidate] = [ordered[0]]
    while len(chosen) < batch_size:
        best: ScoredCandidate | None = None
        best_key = (-math.inf, -math.inf)
        for candidate in ordered:
            if any(c.polymer_id == candidate.polymer_id for c in chosen):
                continue
            distance = min(
                float(np.linalg.norm(features[candidate.polymer_id] - features[c.polymer_id]))
                for c in chosen
            )
            if distance < min_separation:
                continue
            key = (distance, candidate.score)
            if key > best_key:
                best_key, best = key, candidate
        if best is None:
            break
        chosen.append(best)
    return chosen


def select_next_experiments(
    candidates: Sequence[Candidate],
    model: QsprModel,
    training_features: np.ndarray,
    *,
    batch_size: int = 5,
    objective: Objective = "maximise",
    enforce_diversity: bool = True,
    allow_extrapolation: bool = True,
    **weights: float,
) -> SelectionReport:
    """End-to-end selection: screen, score, then pick a diverse batch."""
    report = score_candidates(
        candidates, model, training_features, objective=objective, **weights
    )
    pool = report.scored
    if not allow_extrapolation:
        excluded = [c for c in pool if not c.in_domain]
        for candidate in excluded:
            report.rejected[candidate.polymer_id] = "outside the model's applicability domain"
        pool = [c for c in pool if c.in_domain]
    if not pool:
        return report
    report.selected = (
        select_diverse_batch(pool, batch_size=batch_size)
        if enforce_diversity
        else list(pool[:batch_size])
    )
    logger.info("Selected %d of %d candidates", len(report.selected), len(report.scored))
    return report


__all__ = [
    "Candidate",
    "ScoredCandidate",
    "SelectionReport",
    "compute_novelty",
    "pareto_front",
    "score_candidates",
    "screen_candidates",
    "select_diverse_batch",
    "select_next_experiments",
]
