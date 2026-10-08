"""Structure-property relationship analysis.

The purpose of this module is to find and quantify relationships between descriptors,
simulated observables, free energies, mechanical metrics and experimental properties --
while never letting the *language* run ahead of the evidence.

:class:`RelationshipEvidence` therefore carries an explicit :class:`EvidenceStrength`,
and :attr:`Relationship.language` returns the strongest phrasing the evidence supports:

``"is associated with"``
    A correlation was observed.
``"predicts"``
    The association survives out-of-sample testing.
``"is associated with, controlling for X"``
    The association survives partial correlation against a confounder.

Nothing here ever returns causal language.  Establishing causation needs an
intervention -- changing one factor while holding others fixed -- and an observational
correlation across a polymer dataset is not one.  :func:`analyse_relationship` will
report a partial correlation controlling for named confounders, which is the strongest
claim this kind of evidence supports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from polymer_engine.analysis.statistics import (
    CorrelationResult,
    bootstrap_ci,
    partial_correlation,
    pearson,
    permutation_test,
    spearman,
)
from polymer_engine.core.errors import InsufficientDataError, ScientificError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination

logger = get_logger("science.correlation")

#: Below this many paired observations, a correlation is not worth reporting.
MIN_OBSERVATIONS = 8


class EvidenceStrength(str, Enum):
    NONE = "none"
    ASSOCIATION = "association"
    ROBUST_ASSOCIATION = "robust_association"
    PREDICTIVE = "predictive"
    INSUFFICIENT = "insufficient"

    @property
    def language(self) -> str:
        return {
            EvidenceStrength.NONE: "shows no detectable relationship with",
            EvidenceStrength.ASSOCIATION: "is associated with",
            EvidenceStrength.ROBUST_ASSOCIATION: "is robustly associated with",
            EvidenceStrength.PREDICTIVE: "predicts",
            EvidenceStrength.INSUFFICIENT: "cannot be related to (insufficient data)",
        }[self]


@dataclass
class Relationship:
    """One quantified descriptor-observable relationship."""

    predictor: str
    response: str
    n_observations: int
    pearson: CorrelationResult | None = None
    spearman: CorrelationResult | None = None
    partial: CorrelationResult | None = None
    permutation: CorrelationResult | None = None
    confidence_interval: tuple[float, float] | None = None
    controlled_for: list[str] = field(default_factory=list)
    strength: EvidenceStrength = EvidenceStrength.INSUFFICIENT
    determination: Determination = Determination.INSUFFICIENT_DATA
    notes: list[str] = field(default_factory=list)

    @property
    def statement(self) -> str:
        """A sentence whose verb matches the evidence."""
        controlled = (
            f", controlling for {', '.join(self.controlled_for)}" if self.controlled_for else ""
        )
        coefficient = self.spearman.statistic if self.spearman and self.spearman.statistic is not None else None
        magnitude = f" (rho = {coefficient:+.2f}, n = {self.n_observations})" if coefficient is not None else ""
        return f"{self.predictor} {self.strength.language} {self.response}{controlled}{magnitude}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "predictor": self.predictor,
            "response": self.response,
            "n_observations": self.n_observations,
            "pearson": self.pearson.as_dict() if self.pearson else None,
            "spearman": self.spearman.as_dict() if self.spearman else None,
            "partial": self.partial.as_dict() if self.partial else None,
            "permutation": self.permutation.as_dict() if self.permutation else None,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "controlled_for": self.controlled_for,
            "strength": self.strength.value,
            "determination": self.determination.value,
            "statement": self.statement,
            "notes": self.notes,
        }


def analyse_relationship(
    predictor_values: Sequence[float],
    response_values: Sequence[float],
    *,
    predictor: str,
    response: str,
    confounders: dict[str, Sequence[float]] | None = None,
    n_permutations: int = 2000,
    confidence_level: float = 0.95,
    significance: float = 0.05,
    seed: int = 20240101,
) -> Relationship:
    """Quantify one relationship with every check the data supports.

    Runs Pearson, Spearman, a permutation test, a bootstrap interval, and -- when
    confounders are supplied -- a partial correlation.  The evidence strength is set by
    what survives, not by the headline coefficient.
    """
    x = np.asarray(predictor_values, dtype=float).ravel()
    y = np.asarray(response_values, dtype=float).ravel()
    if x.size != y.size:
        raise ScientificError(
            "Predictor and response must have the same length", n_x=int(x.size), n_y=int(y.size)
        )

    mask = np.isfinite(x) & np.isfinite(y)
    covariate_names = list(confounders or {})
    covariate_matrix: np.ndarray | None = None
    if confounders:
        stacked = []
        for name, values in confounders.items():
            column = np.asarray(values, dtype=float).ravel()
            if column.size != x.size:
                raise ScientificError(
                    "Confounder length does not match the data", confounder=name, n=int(column.size)
                )
            mask &= np.isfinite(column)
            stacked.append(column)
        covariate_matrix = np.column_stack(stacked)

    x, y = x[mask], y[mask]
    if covariate_matrix is not None:
        covariate_matrix = covariate_matrix[mask]

    relationship = Relationship(
        predictor=predictor, response=response, n_observations=int(x.size),
        controlled_for=covariate_names,
    )

    if x.size < MIN_OBSERVATIONS:
        relationship.notes.append(
            f"only {x.size} complete observation(s); at least {MIN_OBSERVATIONS} are needed"
        )
        return relationship
    if np.std(x) == 0 or np.std(y) == 0:
        relationship.notes.append("a constant input has no defined relationship")
        return relationship

    relationship.pearson = pearson(x, y)
    relationship.spearman = spearman(x, y)
    relationship.permutation = permutation_test(x, y, n_permutations=n_permutations, seed=seed)
    relationship.determination = Determination.KNOWN

    try:
        _, low, high = bootstrap_ci(
            _spearman_samples(x, y), n_resamples=1000,
            confidence_level=confidence_level, block_size=1, seed=seed,
        )
        relationship.confidence_interval = (low, high)
    except InsufficientDataError:
        relationship.notes.append("bootstrap interval could not be computed")

    if covariate_matrix is not None and covariate_matrix.size:
        relationship.partial = partial_correlation(x, y, covariate_matrix)

    relationship.strength = _grade(relationship, significance=significance)
    return relationship


def _spearman_samples(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-observation contributions used to bootstrap the rank correlation."""
    from scipy.stats import rankdata  # type: ignore[import-untyped]

    try:
        rx, ry = rankdata(x), rankdata(y)
    except Exception:  # noqa: BLE001 - SciPy is optional
        rx = np.argsort(np.argsort(x)).astype(float) + 1
        ry = np.argsort(np.argsort(y)).astype(float) + 1
    rx = (rx - rx.mean()) / (rx.std() or 1.0)
    ry = (ry - ry.mean()) / (ry.std() or 1.0)
    return rx * ry


def _grade(relationship: Relationship, *, significance: float) -> EvidenceStrength:
    """Decide how strongly the evidence may be phrased."""
    spearman_result = relationship.spearman
    if spearman_result is None or spearman_result.statistic is None:
        return EvidenceStrength.INSUFFICIENT

    permutation_p = relationship.permutation.p_value if relationship.permutation else None
    if permutation_p is None or permutation_p > significance:
        relationship.notes.append(
            f"the association is not significant under a permutation test (p = {permutation_p})"
        )
        return EvidenceStrength.NONE

    interval = relationship.confidence_interval
    if interval is not None and interval[0] <= 0.0 <= interval[1]:
        relationship.notes.append(
            "the bootstrap confidence interval for the rank correlation includes zero"
        )
        return EvidenceStrength.ASSOCIATION

    if relationship.partial is not None and relationship.partial.statistic is not None:
        raw = abs(spearman_result.statistic)
        controlled = abs(relationship.partial.statistic)
        if controlled < 0.5 * raw:
            relationship.notes.append(
                f"the association weakens from {raw:.2f} to {controlled:.2f} when controlling for "
                f"{', '.join(relationship.controlled_for)}; much of it is explained by the confounder"
            )
            return EvidenceStrength.ASSOCIATION
        relationship.notes.append(
            f"the association survives controlling for {', '.join(relationship.controlled_for)}"
        )
        return EvidenceStrength.ROBUST_ASSOCIATION

    return EvidenceStrength.ASSOCIATION


@dataclass
class CorrelationMatrix:
    """A screen of many predictors against many responses."""

    predictors: list[str]
    responses: list[str]
    relationships: dict[tuple[str, str], Relationship] = field(default_factory=dict)
    n_tests: int = 0

    def get(self, predictor: str, response: str) -> Relationship | None:
        return self.relationships.get((predictor, response))

    def significant(self, *, alpha: float = 0.05, correct_multiplicity: bool = True) -> list[Relationship]:
        """Relationships that survive, with a multiple-comparison correction by default.

        Screening 15 descriptors against 5 properties is 75 tests; at alpha = 0.05 about
        four will look significant by chance alone. The Bonferroni correction is
        conservative but it is honest, and it is applied unless explicitly disabled.
        """
        threshold = alpha / max(self.n_tests, 1) if correct_multiplicity else alpha
        out = []
        for relationship in self.relationships.values():
            p = relationship.permutation.p_value if relationship.permutation else None
            if p is not None and p <= threshold:
                out.append(relationship)
        return sorted(
            out,
            key=lambda r: -abs(r.spearman.statistic if r.spearman and r.spearman.statistic else 0.0),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "predictors": self.predictors,
            "responses": self.responses,
            "n_tests": self.n_tests,
            "relationships": [r.as_dict() for r in self.relationships.values()],
        }


def screen_relationships(
    predictors: dict[str, Sequence[float]],
    responses: dict[str, Sequence[float]],
    *,
    confounders: dict[str, Sequence[float]] | None = None,
    n_permutations: int = 1000,
    seed: int = 20240101,
) -> CorrelationMatrix:
    """Test every predictor against every response.

    The multiple-comparison burden is recorded in ``n_tests`` so
    :meth:`CorrelationMatrix.significant` can correct for it. Reporting the best of
    seventy-five correlations at p < 0.05 without that correction is how spurious
    structure-property "laws" get published.
    """
    matrix = CorrelationMatrix(
        predictors=sorted(predictors), responses=sorted(responses),
        n_tests=len(predictors) * len(responses),
    )
    for predictor_name in matrix.predictors:
        for response_name in matrix.responses:
            controls = None
            if confounders:
                controls = {k: v for k, v in confounders.items() if k != predictor_name}
            try:
                relationship = analyse_relationship(
                    predictors[predictor_name], responses[response_name],
                    predictor=predictor_name, response=response_name,
                    confounders=controls, n_permutations=n_permutations, seed=seed,
                )
            except ScientificError as exc:
                logger.warning("Skipping %s vs %s: %s", predictor_name, response_name, exc)
                continue
            matrix.relationships[(predictor_name, response_name)] = relationship
    logger.info(
        "Screened %d predictor-response pairs; %d significant after correction",
        matrix.n_tests, len(matrix.significant()),
    )
    return matrix


__all__ = [
    "MIN_OBSERVATIONS",
    "CorrelationMatrix",
    "EvidenceStrength",
    "Relationship",
    "analyse_relationship",
    "screen_relationships",
]
