"""Predictive models with fingerprints and structural leakage protection.

Beyond the leakage routes already closed in :mod:`polymer_engine.discovery.qspr`
(duplicate identities across a split, preprocessing fitted on all the data, scoring a
training member as a discovery), polymer datasets carry two more that are easy to miss:

**Near-identical structures.** Two polymers differing by one methylene are not
independent test cases. Splitting them apart inflates every score. :func:`cluster_by_similarity`
groups structurally similar polymers so a whole cluster lands on one side of a split.

**Derived measurements from the same trajectory.** Density and volume from one
simulation are not two observations. Grouping by ``source_id`` keeps them together.

Every model records a ``dataset_fingerprint`` and a ``model_fingerprint`` so a
prediction can be traced to exactly the data and hyperparameters that produced it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from polymer_engine.core.errors import InsufficientDataError, ScientificError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination
from polymer_engine.core.provenance import canonical_hash
from polymer_engine.discovery.qspr import (
    ApplicabilityDomain,
    Dataset,
    Preprocessor,
    grouped_kfold_indices,
    sklearn_available,
)

logger = get_logger("science.models")


class ModelKind(str, Enum):
    LINEAR = "linear"
    RIDGE = "ridge"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    GAUSSIAN_PROCESS = "gaussian_process"


#: Which model kinds provide a usable per-prediction uncertainty, and how.
UNCERTAINTY_SOURCE: dict[ModelKind, str] = {
    ModelKind.LINEAR: "none",
    ModelKind.RIDGE: "none",
    ModelKind.RANDOM_FOREST: "ensemble spread across trees (relative, not calibrated)",
    ModelKind.GRADIENT_BOOSTING: "none",
    ModelKind.GAUSSIAN_PROCESS: "posterior standard deviation (calibrated under its own prior)",
}


@dataclass
class ModelEvaluation:
    """Cross-validated performance, with the split policy that produced it."""

    model_kind: ModelKind
    r2: float | None
    rmse: float | None
    mae: float | None
    spearman: float | None
    n_folds: int
    n_samples: int
    n_groups: int
    grouping: str
    determination: Determination = Determination.KNOWN
    note: str = ""

    @property
    def better_than_the_mean(self) -> bool:
        """R^2 > 0. A model worse than predicting the mean has learned nothing."""
        return self.r2 is not None and self.r2 > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_kind": self.model_kind.value,
            "r2": self.r2,
            "rmse": self.rmse,
            "mae": self.mae,
            "spearman": self.spearman,
            "n_folds": self.n_folds,
            "n_samples": self.n_samples,
            "n_groups": self.n_groups,
            "grouping": self.grouping,
            "better_than_the_mean": self.better_than_the_mean,
            "determination": self.determination.value,
            "note": self.note,
        }


@dataclass
class TrainedModel:
    """A fitted model with everything needed to trace a prediction back to its data."""

    model_kind: ModelKind
    estimator: Any
    preprocessor: Preprocessor
    domain: ApplicabilityDomain
    feature_names: list[str]
    target_name: str
    target_units: str
    training_ids: set[str]
    dataset_fingerprint: str
    model_fingerprint: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    evaluation: ModelEvaluation | None = None

    @property
    def provides_uncertainty(self) -> bool:
        return UNCERTAINTY_SOURCE[self.model_kind] != "none"

    def predict(
        self, features: np.ndarray, polymer_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Predict, flagging training members and extrapolation."""
        X = np.asarray(features, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ScientificError(
                "Feature matrix does not match the model",
                expected=len(self.feature_names),
                got=None if X.ndim != 2 else X.shape[1],
            )
        if len(polymer_ids) != X.shape[0]:
            raise ScientificError("One id per row is required")

        scaled = self.preprocessor.transform(X)
        mean = np.asarray(self.estimator.predict(scaled), dtype=float)
        spread = self._uncertainty(scaled)
        distances = self.domain.distance(scaled)
        inside = distances <= self.domain.threshold

        out: list[dict[str, Any]] = []
        for i, identifier in enumerate(polymer_ids):
            determination = Determination.KNOWN
            notes: list[str] = []
            if not math.isfinite(mean[i]):
                determination = Determination.UNKNOWN
                notes.append("model produced a non-finite prediction")
            if identifier in self.training_ids:
                determination = Determination.REQUIRES_VALIDATION
                notes.append("this polymer is in the training set; the prediction is not out-of-sample")
            elif not inside[i]:
                determination = Determination.REQUIRES_VALIDATION
                notes.append(
                    f"outside the applicability domain (distance {distances[i]:.2f} > "
                    f"{self.domain.threshold:.2f}); this is an extrapolation"
                )
            out.append(
                {
                    "polymer_id": identifier,
                    "value": float(mean[i]) if math.isfinite(mean[i]) else None,
                    "uncertainty": float(spread[i]) if spread is not None else None,
                    "in_domain": bool(inside[i]),
                    "domain_distance": float(distances[i]),
                    "determination": determination.value,
                    "notes": "; ".join(notes),
                    "model_fingerprint": self.model_fingerprint,
                    "dataset_fingerprint": self.dataset_fingerprint,
                }
            )
        return out

    def _uncertainty(self, scaled: np.ndarray) -> np.ndarray | None:
        if self.model_kind is ModelKind.RANDOM_FOREST:
            members = np.vstack([tree.predict(scaled) for tree in self.estimator.estimators_])
            return members.std(axis=0)
        if self.model_kind is ModelKind.GAUSSIAN_PROCESS:
            _, sigma = self.estimator.predict(scaled, return_std=True)
            return np.asarray(sigma, dtype=float)
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_kind": self.model_kind.value,
            "target": self.target_name,
            "target_units": self.target_units,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "n_training_polymers": len(self.training_ids),
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "hyperparameters": self.hyperparameters,
            "provides_uncertainty": self.provides_uncertainty,
            "uncertainty_source": UNCERTAINTY_SOURCE[self.model_kind],
            "evaluation": self.evaluation.as_dict() if self.evaluation else None,
        }


def dataset_fingerprint(dataset: Dataset) -> str:
    """Digest of the exact data a model was fitted to."""
    return canonical_hash(
        {
            "polymer_ids": sorted(dataset.polymer_ids),
            "feature_names": dataset.feature_names,
            "target_name": dataset.target_name,
            "target_units": dataset.target_units,
            "X": np.round(dataset.X, 8).tolist(),
            "y": np.round(dataset.y, 8).tolist(),
        }
    )


def cluster_by_similarity(
    features: np.ndarray, *, threshold: float = 0.05
) -> list[int]:
    """Group rows whose standardised features are nearly identical.

    Two polymers differing by one methylene are not independent test cases. Assigning
    them to the same group keeps them on the same side of a cross-validation split, so
    the score reflects generalisation rather than memorisation.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ScientificError("Features must be a 2-D array", shape=X.shape)
    if X.shape[0] == 0:
        return []

    scale = X.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardised = (X - X.mean(axis=0)) / scale

    labels = [-1] * X.shape[0]
    next_label = 0
    for i in range(X.shape[0]):
        if labels[i] >= 0:
            continue
        labels[i] = next_label
        distances = np.linalg.norm(standardised - standardised[i], axis=1)
        for j in np.flatnonzero(distances <= threshold):
            if labels[j] < 0:
                labels[j] = next_label
        next_label += 1
    return labels


def build_groups(
    dataset: Dataset,
    *,
    similarity_threshold: float | None = 0.05,
    source_ids: Sequence[str] | None = None,
) -> tuple[list[str], str]:
    """Assign each row to a leakage-safe group, and describe the policy used."""
    groups = list(dataset.polymer_ids)
    policy = ["polymer identity"]

    if source_ids is not None:
        if len(source_ids) != dataset.n_samples:
            raise ScientificError("One source id per row is required")
        # Measurements from one trajectory are not independent observations.
        merged: dict[str, str] = {}
        for group, source in zip(groups, source_ids, strict=True):
            merged.setdefault(source, group)
        groups = [merged[source] for source in source_ids]
        policy.append("simulation source")

    if similarity_threshold is not None:
        labels = cluster_by_similarity(dataset.X, threshold=similarity_threshold)
        representative: dict[int, str] = {}
        for label, group in zip(labels, groups, strict=True):
            representative.setdefault(label, group)
        groups = [representative[label] for label in labels]
        policy.append(f"structural similarity (threshold {similarity_threshold})")

    return groups, " + ".join(policy)


def _make_estimator(kind: ModelKind, seed: int, hyperparameters: dict[str, Any]) -> Any:
    if not sklearn_available():
        raise ScientificError(
            "scikit-learn is required to fit models", hint="pip install scikit-learn"
        )
    if kind is ModelKind.LINEAR:
        from sklearn.linear_model import LinearRegression

        return LinearRegression(**hyperparameters)
    if kind is ModelKind.RIDGE:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=hyperparameters.get("alpha", 1.0), random_state=seed)
    if kind is ModelKind.RANDOM_FOREST:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=hyperparameters.get("n_estimators", 300),
            max_depth=hyperparameters.get("max_depth"),
            random_state=seed,
            n_jobs=1,
        )
    if kind is ModelKind.GRADIENT_BOOSTING:
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=hyperparameters.get("n_estimators", 200),
            learning_rate=hyperparameters.get("learning_rate", 0.05),
            max_depth=hyperparameters.get("max_depth", 3),
            random_state=seed,
        )
    if kind is ModelKind.GAUSSIAN_PROCESS:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        return GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=seed,
            n_restarts_optimizer=hyperparameters.get("n_restarts_optimizer", 2),
        )
    raise ScientificError("Unsupported model kind", kind=kind.value)


def evaluate_model(
    dataset: Dataset,
    kind: ModelKind,
    *,
    n_folds: int = 5,
    seed: int = 20240101,
    hyperparameters: dict[str, Any] | None = None,
    similarity_threshold: float | None = 0.05,
    source_ids: Sequence[str] | None = None,
) -> ModelEvaluation:
    """Grouped cross-validation with per-fold preprocessing."""
    hyperparameters = hyperparameters or {}
    if not sklearn_available():
        return ModelEvaluation(
            model_kind=kind, r2=None, rmse=None, mae=None, spearman=None,
            n_folds=0, n_samples=dataset.n_samples, n_groups=0, grouping="none",
            determination=Determination.UNSUPPORTED, note="scikit-learn is not installed",
        )

    groups, policy = build_groups(
        dataset, similarity_threshold=similarity_threshold, source_ids=source_ids
    )
    n_groups = len(set(groups))
    if n_groups < n_folds:
        return ModelEvaluation(
            model_kind=kind, r2=None, rmse=None, mae=None, spearman=None,
            n_folds=0, n_samples=dataset.n_samples, n_groups=n_groups, grouping=policy,
            determination=Determination.INSUFFICIENT_DATA,
            note=(
                f"{n_groups} independent group(s) after grouping by {policy}; "
                f"{n_folds}-fold cross-validation needs at least {n_folds}"
            ),
        )

    folds = grouped_kfold_indices(groups, n_folds, seed=seed)
    predictions = np.full(dataset.n_samples, np.nan)

    for fold, test_index in enumerate(folds):
        train_index = np.setdiff1d(np.arange(dataset.n_samples), test_index)
        if train_index.size < 2 or test_index.size == 0:
            continue
        overlap = {groups[i] for i in train_index} & {groups[i] for i in test_index}
        if overlap:  # pragma: no cover - guards grouped_kfold_indices
            raise ScientificError("Cross-validation split leaks groups", overlapping=sorted(overlap))
        preprocessor = Preprocessor().fit(dataset.X[train_index])
        estimator = _make_estimator(kind, seed + fold, hyperparameters)
        estimator.fit(preprocessor.transform(dataset.X[train_index]), dataset.y[train_index])
        predictions[test_index] = estimator.predict(preprocessor.transform(dataset.X[test_index]))

    mask = np.isfinite(predictions)
    if mask.sum() < 3:
        return ModelEvaluation(
            model_kind=kind, r2=None, rmse=None, mae=None, spearman=None,
            n_folds=n_folds, n_samples=dataset.n_samples, n_groups=n_groups, grouping=policy,
            determination=Determination.INSUFFICIENT_DATA, note="too few held-out predictions",
        )

    truth, predicted = dataset.y[mask], predictions[mask]
    residual = truth - predicted
    ss_res = float((residual**2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())

    from polymer_engine.analysis.statistics import spearman as spearman_fn

    return ModelEvaluation(
        model_kind=kind,
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        rmse=float(math.sqrt((residual**2).mean())),
        mae=float(np.abs(residual).mean()),
        spearman=spearman_fn(truth, predicted).statistic,
        n_folds=n_folds,
        n_samples=dataset.n_samples,
        n_groups=n_groups,
        grouping=policy,
    )


def train_model(
    dataset: Dataset,
    kind: ModelKind = ModelKind.RANDOM_FOREST,
    *,
    seed: int = 20240101,
    hyperparameters: dict[str, Any] | None = None,
    evaluate: bool = True,
    n_folds: int = 5,
    similarity_threshold: float | None = 0.05,
    source_ids: Sequence[str] | None = None,
) -> TrainedModel:
    """Fit a model on the whole dataset, after evaluating it honestly."""
    hyperparameters = hyperparameters or {}
    if dataset.n_samples < 10:
        raise InsufficientDataError(
            "Too few samples to fit a usable model", n_samples=dataset.n_samples, minimum=10
        )
    if not np.all(np.isfinite(dataset.y)):
        raise ScientificError("Target vector contains non-finite values")
    if float(np.std(dataset.y)) <= 1e-12:
        raise ScientificError("Target is constant; a model fitted to it predicts one value forever")

    evaluation = (
        evaluate_model(
            dataset, kind, n_folds=n_folds, seed=seed, hyperparameters=hyperparameters,
            similarity_threshold=similarity_threshold, source_ids=source_ids,
        )
        if evaluate
        else None
    )

    preprocessor = Preprocessor().fit(dataset.X)
    scaled = preprocessor.transform(dataset.X)
    estimator = _make_estimator(kind, seed, hyperparameters)
    estimator.fit(scaled, dataset.y)

    data_hash = dataset_fingerprint(dataset)
    model_hash = canonical_hash(
        {
            "dataset": data_hash,
            "kind": kind.value,
            "seed": seed,
            "hyperparameters": hyperparameters,
        }
    )
    logger.info(
        "Trained %s for %s on %d polymers (dataset %s)",
        kind.value, dataset.target_name, dataset.n_samples, data_hash[:12],
    )
    return TrainedModel(
        model_kind=kind,
        estimator=estimator,
        preprocessor=preprocessor,
        domain=ApplicabilityDomain.fit(scaled),
        feature_names=list(dataset.feature_names),
        target_name=dataset.target_name,
        target_units=dataset.target_units,
        training_ids=set(dataset.polymer_ids),
        dataset_fingerprint=data_hash,
        model_fingerprint=model_hash,
        hyperparameters=hyperparameters,
        evaluation=evaluation,
    )


def compare_models(
    dataset: Dataset,
    kinds: Sequence[ModelKind] | None = None,
    *,
    n_folds: int = 5,
    seed: int = 20240101,
    similarity_threshold: float | None = 0.05,
) -> list[ModelEvaluation]:
    """Evaluate several model families on the same grouped splits."""
    kinds = kinds or [
        ModelKind.LINEAR, ModelKind.RIDGE, ModelKind.RANDOM_FOREST, ModelKind.GRADIENT_BOOSTING
    ]
    results = [
        evaluate_model(
            dataset, kind, n_folds=n_folds, seed=seed, similarity_threshold=similarity_threshold
        )
        for kind in kinds
    ]
    return sorted(results, key=lambda e: (e.r2 is None, -(e.r2 or -math.inf)))


__all__ = [
    "UNCERTAINTY_SOURCE",
    "ModelEvaluation",
    "ModelKind",
    "TrainedModel",
    "build_groups",
    "cluster_by_similarity",
    "compare_models",
    "dataset_fingerprint",
    "evaluate_model",
    "train_model",
]
