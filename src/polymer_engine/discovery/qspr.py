"""Structure-property models with leakage protection built in.

Data leakage is the default outcome of careless QSPR, not an exotic failure.  The
three routes that matter here are all closed structurally rather than by convention:

1. **Duplicate polymers across the split.**  Splits are made on ``polymer_id``, which
   is derived from the canonical repeat unit, so the same material described two ways
   cannot land on both sides.
2. **Preprocessing fitted on all the data.**  Imputation and scaling are fitted
   *inside* each fold, on the training part only.
3. **Feature selection using the test set.**  Selection, when used, happens inside the
   fold for the same reason.

A model also reports its applicability domain, so a prediction far outside the
training distribution is labelled an extrapolation rather than presented as a number
with the same standing as an interpolation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from polymer_engine.core.errors import InsufficientDataError, ScientificError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination, Measurement

logger = get_logger("discovery.qspr")


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class Dataset:
    """Feature matrix with the identities needed to split without leaking."""

    polymer_ids: list[str]
    feature_names: list[str]
    X: np.ndarray
    y: np.ndarray
    target_name: str = "target"
    target_units: str = "1"
    groups: list[str] | None = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=float).ravel()
        if self.X.ndim != 2:
            raise ScientificError("Feature matrix must be 2-D", shape=self.X.shape)
        if self.X.shape[0] != self.y.size:
            raise ScientificError(
                "Feature and target counts differ", n_rows=self.X.shape[0], n_targets=int(self.y.size)
            )
        if len(self.polymer_ids) != self.X.shape[0]:
            raise ScientificError(
                "One polymer id per row is required",
                n_ids=len(self.polymer_ids),
                n_rows=self.X.shape[0],
            )
        if self.X.shape[1] != len(self.feature_names):
            raise ScientificError(
                "One feature name per column is required",
                n_names=len(self.feature_names),
                n_columns=self.X.shape[1],
            )

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def duplicate_ids(self) -> list[str]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for identifier in self.polymer_ids:
            if identifier in seen:
                duplicates.add(identifier)
            seen.add(identifier)
        return sorted(duplicates)

    def split_groups(self) -> list[str]:
        """Grouping used for cross-validation; defaults to polymer identity."""
        return list(self.groups) if self.groups is not None else list(self.polymer_ids)


@dataclass
class Preprocessor:
    """Median imputation plus standardisation, fitted on training data only."""

    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> Preprocessor:
        with np.errstate(all="ignore"):
            self.medians = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        filled = self._impute(X)
        self.means = filled.mean(axis=0)
        scales = filled.std(axis=0)
        # A constant feature carries no information; leave it at scale 1 rather than
        # dividing by zero and producing infinities.
        self.scales = np.where(scales > 1e-12, scales, 1.0)
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        assert self.medians is not None
        out = np.array(X, dtype=float, copy=True)
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = np.take(self.medians, np.nonzero(bad)[1])
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.medians is None or self.means is None or self.scales is None:
            raise ScientificError("Preprocessor must be fitted before use")
        return (self._impute(X) - self.means) / self.scales


@dataclass
class ApplicabilityDomain:
    """Where the model has actually seen data.

    Distance to the training set, in standardised feature space, with a threshold set
    from the training distribution itself.
    """

    center: np.ndarray
    reference_distances: np.ndarray
    threshold: float

    @classmethod
    def fit(cls, X_scaled: np.ndarray, *, quantile: float = 0.95) -> ApplicabilityDomain:
        center = X_scaled.mean(axis=0)
        distances = np.linalg.norm(X_scaled - center, axis=1)
        return cls(center=center, reference_distances=distances, threshold=float(np.quantile(distances, quantile)))

    def distance(self, X_scaled: np.ndarray) -> np.ndarray:
        return np.linalg.norm(X_scaled - self.center, axis=1)

    def inside(self, X_scaled: np.ndarray) -> np.ndarray:
        return self.distance(X_scaled) <= self.threshold


@dataclass
class CrossValidationResult:
    r2: float | None
    rmse: float | None
    mae: float | None
    spearman: float | None
    n_folds: int
    n_samples: int
    predictions: np.ndarray
    truth: np.ndarray
    determination: Determination = Determination.KNOWN
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "r2": self.r2,
            "rmse": self.rmse,
            "mae": self.mae,
            "spearman": self.spearman,
            "n_folds": self.n_folds,
            "n_samples": self.n_samples,
            "determination": self.determination.value,
            "note": self.note,
        }


@dataclass
class Prediction:
    polymer_id: str
    value: float | None
    uncertainty: float | None
    in_domain: bool
    domain_distance: float
    determination: Determination = Determination.KNOWN
    note: str = ""

    def as_measurement(self, name: str, units: str) -> Measurement:
        if self.determination is not Determination.KNOWN or self.value is None:
            return Measurement.unknown(name, units=units, reason=self.note, determination=self.determination)
        return Measurement(
            name=name,
            value=self.value,
            uncertainty=self.uncertainty,
            units=units,
            method="QSPR surrogate",
            notes=self.note or None,
        )


class QsprModel:
    """A random-forest surrogate with ensemble-spread uncertainty.

    The uncertainty is the spread of the ensemble's member predictions.  That is a
    useful *relative* signal for choosing what to run next, and it is deliberately not
    presented as a calibrated confidence interval -- calling it one would be an
    overclaim.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        random_state: int = 20240101,
        max_depth: int | None = None,
        min_training_samples: int = 10,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_training_samples = min_training_samples
        self.preprocessor: Preprocessor | None = None
        self.domain: ApplicabilityDomain | None = None
        self.model: Any = None
        self.feature_names: list[str] = []
        self.training_ids: set[str] = set()
        self.target_name = "target"
        self.target_units = "1"

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def fit(self, dataset: Dataset) -> QsprModel:
        if not sklearn_available():
            raise ScientificError(
                "scikit-learn is required to fit a QSPR model", hint="pip install scikit-learn"
            )
        if dataset.n_samples < self.min_training_samples:
            raise InsufficientDataError(
                "Too few training samples for a usable surrogate",
                n_samples=dataset.n_samples,
                minimum=self.min_training_samples,
            )
        if not np.all(np.isfinite(dataset.y)):
            raise ScientificError("Target vector contains non-finite values")
        if float(np.std(dataset.y)) <= 1e-12:
            raise ScientificError(
                "Target is constant; a model fitted to it would predict one value forever"
            )

        from sklearn.ensemble import RandomForestRegressor

        self.preprocessor = Preprocessor().fit(dataset.X)
        X_scaled = self.preprocessor.transform(dataset.X)
        self.domain = ApplicabilityDomain.fit(X_scaled)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth,
            n_jobs=1,
        )
        self.model.fit(X_scaled, dataset.y)
        self.feature_names = list(dataset.feature_names)
        self.training_ids = set(dataset.polymer_ids)
        self.target_name = dataset.target_name
        self.target_units = dataset.target_units
        logger.info(
            "Fitted QSPR for %s on %d polymers, %d features",
            dataset.target_name,
            dataset.n_samples,
            dataset.n_features,
        )
        return self

    def predict(self, X: np.ndarray, polymer_ids: Sequence[str]) -> list[Prediction]:
        """Predict, flagging extrapolation and refusing to score training members."""
        if not self.fitted or self.preprocessor is None or self.domain is None:
            raise ScientificError("Model must be fitted before predicting")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ScientificError(
                "Candidate features do not match the model",
                expected_features=len(self.feature_names),
                got=None if X.ndim != 2 else X.shape[1],
            )
        if len(polymer_ids) != X.shape[0]:
            raise ScientificError("One id per candidate row is required")

        X_scaled = self.preprocessor.transform(X)
        members = np.vstack([tree.predict(X_scaled) for tree in self.model.estimators_])
        mean = members.mean(axis=0)
        spread = members.std(axis=0)
        distances = self.domain.distance(X_scaled)
        inside = distances <= self.domain.threshold

        predictions: list[Prediction] = []
        for i, identifier in enumerate(polymer_ids):
            if not math.isfinite(mean[i]):
                predictions.append(
                    Prediction(
                        polymer_id=identifier, value=None, uncertainty=None, in_domain=False,
                        domain_distance=float(distances[i]),
                        determination=Determination.UNKNOWN, note="model produced a non-finite prediction",
                    )
                )
                continue
            note = ""
            determination = Determination.KNOWN
            if identifier in self.training_ids:
                note = "candidate is already in the training set; its prediction is not out-of-sample"
                determination = Determination.REQUIRES_VALIDATION
            elif not inside[i]:
                note = (
                    f"outside the applicability domain (distance {distances[i]:.2f} > "
                    f"{self.domain.threshold:.2f}); this is an extrapolation"
                )
                determination = Determination.REQUIRES_VALIDATION
            predictions.append(
                Prediction(
                    polymer_id=identifier,
                    value=float(mean[i]),
                    uncertainty=float(spread[i]),
                    in_domain=bool(inside[i]),
                    domain_distance=float(distances[i]),
                    determination=determination,
                    note=note,
                )
            )
        return predictions

    def feature_importance(self) -> dict[str, float]:
        if not self.fitted:
            raise ScientificError("Model must be fitted before reporting importances")
        return dict(
            sorted(
                zip(self.feature_names, (float(v) for v in self.model.feature_importances_), strict=True),
                key=lambda kv: -kv[1],
            )
        )


def grouped_kfold_indices(groups: Sequence[str], n_folds: int, *, seed: int = 20240101) -> list[np.ndarray]:
    """Assign whole groups to folds so no group spans the split."""
    unique = sorted(set(groups))
    if len(unique) < n_folds:
        raise InsufficientDataError(
            "Fewer distinct groups than folds; the split would leak",
            n_groups=len(unique),
            n_folds=n_folds,
        )
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(np.array(unique, dtype=object)))
    assignment = {group: index % n_folds for index, group in enumerate(shuffled)}
    array = np.array([assignment[g] for g in groups])
    return [np.flatnonzero(array == fold) for fold in range(n_folds)]


def cross_validate(
    dataset: Dataset,
    *,
    n_folds: int = 5,
    seed: int = 20240101,
) -> CrossValidationResult:
    """Grouped k-fold cross-validation with per-fold preprocessing.

    Preprocessing is fitted inside each fold.  Fitting the imputer or scaler on the
    whole dataset first is the most common silent leak in QSPR work and would inflate
    every score reported here.
    """
    if not sklearn_available():
        return CrossValidationResult(
            r2=None, rmse=None, mae=None, spearman=None, n_folds=0, n_samples=dataset.n_samples,
            predictions=np.array([]), truth=np.array([]),
            determination=Determination.UNSUPPORTED, note="scikit-learn is not installed",
        )
    if dataset.n_samples < n_folds * 2:
        return CrossValidationResult(
            r2=None, rmse=None, mae=None, spearman=None, n_folds=0, n_samples=dataset.n_samples,
            predictions=np.array([]), truth=np.array([]),
            determination=Determination.INSUFFICIENT_DATA,
            note=f"{dataset.n_samples} samples is too few for {n_folds}-fold cross-validation",
        )

    from sklearn.ensemble import RandomForestRegressor

    groups = dataset.split_groups()
    folds = grouped_kfold_indices(groups, n_folds, seed=seed)
    predictions = np.full(dataset.n_samples, np.nan)

    for fold, test_index in enumerate(folds):
        train_index = np.setdiff1d(np.arange(dataset.n_samples), test_index)
        if train_index.size < 2 or test_index.size == 0:
            continue
        train_groups = {groups[i] for i in train_index}
        test_groups = {groups[i] for i in test_index}
        overlap = train_groups & test_groups
        if overlap:  # pragma: no cover - guards grouped_kfold_indices
            raise ScientificError("Cross-validation split leaks groups", overlapping=sorted(overlap))

        preprocessor = Preprocessor().fit(dataset.X[train_index])
        model = RandomForestRegressor(n_estimators=200, random_state=seed + fold, n_jobs=1)
        model.fit(preprocessor.transform(dataset.X[train_index]), dataset.y[train_index])
        predictions[test_index] = model.predict(preprocessor.transform(dataset.X[test_index]))

    mask = np.isfinite(predictions)
    if mask.sum() < 3:
        return CrossValidationResult(
            r2=None, rmse=None, mae=None, spearman=None, n_folds=n_folds, n_samples=dataset.n_samples,
            predictions=predictions, truth=dataset.y,
            determination=Determination.INSUFFICIENT_DATA, note="too few held-out predictions",
        )

    truth = dataset.y[mask]
    predicted = predictions[mask]
    residual = truth - predicted
    ss_res = float((residual**2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    from polymer_engine.analysis.statistics import spearman as spearman_fn

    rank = spearman_fn(truth, predicted)
    return CrossValidationResult(
        r2=r2,
        rmse=float(math.sqrt((residual**2).mean())),
        mae=float(np.abs(residual).mean()),
        spearman=rank.statistic,
        n_folds=n_folds,
        n_samples=dataset.n_samples,
        predictions=predictions,
        truth=dataset.y,
    )


def audit_leakage(train_ids: Sequence[str], test_ids: Sequence[str]) -> list[str]:
    """Report identities that appear on both sides of a split."""
    return sorted(set(train_ids) & set(test_ids))


__all__ = [
    "ApplicabilityDomain",
    "CrossValidationResult",
    "Dataset",
    "Prediction",
    "Preprocessor",
    "QsprModel",
    "audit_leakage",
    "cross_validate",
    "grouped_kfold_indices",
    "sklearn_available",
]
