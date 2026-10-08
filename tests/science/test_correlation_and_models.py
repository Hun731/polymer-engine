"""Structure-property correlation and predictive modelling."""

from __future__ import annotations

import numpy as np
import pytest

from polymer_engine.core.errors import InsufficientDataError, ScientificError
from polymer_engine.core.models import Determination
from polymer_engine.discovery.qspr import Dataset
from polymer_engine.science.correlation import (
    MIN_OBSERVATIONS,
    EvidenceStrength,
    analyse_relationship,
    screen_relationships,
)
from polymer_engine.science.models import (
    UNCERTAINTY_SOURCE,
    ModelKind,
    build_groups,
    cluster_by_similarity,
    compare_models,
    dataset_fingerprint,
    evaluate_model,
    train_model,
)
from tests.markers import requires_sklearn


def linear_dataset(n: int = 80, seed: int = 0, noise: float = 0.2) -> Dataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, noise, n)
    return Dataset(
        polymer_ids=[f"p{i:04d}" for i in range(n)],
        feature_names=["a", "b", "c", "d"],
        X=X, y=y, target_name="tg", target_units="K",
    )


# ==========================================================================
# Correlation
# ==========================================================================
class TestRelationships:
    def test_a_real_relationship_is_detected(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.uniform(30, 200, 60)
        y = 250 + 0.8 * x + rng.normal(0, 8, 60)
        relationship = analyse_relationship(x, y, predictor="mass", response="tg")
        assert relationship.strength in {
            EvidenceStrength.ASSOCIATION, EvidenceStrength.ROBUST_ASSOCIATION,
        }
        assert relationship.spearman.statistic > 0.8

    def test_noise_shows_no_relationship(self) -> None:
        rng = np.random.default_rng(2)
        relationship = analyse_relationship(
            rng.normal(size=60), rng.normal(size=60), predictor="noise", response="tg"
        )
        assert relationship.strength is EvidenceStrength.NONE

    def test_a_confounded_association_is_exposed(self) -> None:
        """Two variables both tracking a third correlate until it is controlled for."""
        rng = np.random.default_rng(3)
        mass = rng.uniform(30, 200, 60)
        tg = 250 + 0.8 * mass + rng.normal(0, 8, 60)
        rings = mass / 50 + rng.normal(0, 0.2, 60)

        relationship = analyse_relationship(
            rings, tg, predictor="ring_count", response="tg",
            confounders={"repeat_unit_mass": mass},
        )
        assert relationship.partial is not None
        assert abs(relationship.partial.statistic) < 0.3
        assert any("explained by the confounder" in note for note in relationship.notes)

    def test_an_association_surviving_control_is_graded_higher(self) -> None:
        rng = np.random.default_rng(4)
        independent = rng.normal(size=80)
        unrelated = rng.normal(size=80)
        y = 2.0 * independent + rng.normal(0, 0.3, 80)
        relationship = analyse_relationship(
            independent, y, predictor="x", response="y", confounders={"unrelated": unrelated}
        )
        assert relationship.strength is EvidenceStrength.ROBUST_ASSOCIATION

    def test_no_statement_ever_claims_causation(self) -> None:
        """Observational correlation across a dataset is not an intervention."""
        rng = np.random.default_rng(5)
        x = rng.normal(size=60)
        y = 2 * x + rng.normal(0, 0.1, 60)
        statement = analyse_relationship(x, y, predictor="x", response="y").statement.lower()
        for forbidden in ("causes", "caused", "because of", "due to", "leads to", "drives"):
            assert forbidden not in statement

    def test_too_few_observations_is_insufficient(self) -> None:
        relationship = analyse_relationship(
            list(range(MIN_OBSERVATIONS - 1)), list(range(MIN_OBSERVATIONS - 1)),
            predictor="x", response="y",
        )
        assert relationship.strength is EvidenceStrength.INSUFFICIENT
        assert relationship.determination is Determination.INSUFFICIENT_DATA

    def test_a_constant_predictor_has_no_relationship(self) -> None:
        relationship = analyse_relationship(
            [1.0] * 30, np.random.default_rng(6).normal(size=30), predictor="x", response="y"
        )
        assert relationship.strength is EvidenceStrength.INSUFFICIENT

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ScientificError):
            analyse_relationship([1, 2, 3], [1, 2], predictor="x", response="y")

    def test_missing_values_are_dropped_pairwise(self) -> None:
        x = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        assert analyse_relationship(x, y, predictor="x", response="y").n_observations == 9


class TestScreening:
    def test_multiplicity_is_recorded_and_corrected(self) -> None:
        """Screening 3 x 2 pairs at alpha = 0.05 will show a false positive by chance."""
        rng = np.random.default_rng(7)
        n = 60
        mass = rng.uniform(30, 200, n)
        tg = 250 + 0.8 * mass + rng.normal(0, 8, n)
        matrix = screen_relationships(
            {"mass": mass, "noise1": rng.normal(size=n), "noise2": rng.normal(size=n)},
            {"tg": tg, "random": rng.normal(size=n)},
        )
        assert matrix.n_tests == 6
        corrected = matrix.significant(correct_multiplicity=True)
        uncorrected = matrix.significant(correct_multiplicity=False)
        assert len(corrected) <= len(uncorrected)
        assert any(r.predictor == "mass" and r.response == "tg" for r in corrected)

    def test_pure_noise_screens_clean_after_correction(self) -> None:
        rng = np.random.default_rng(8)
        predictors = {f"x{i}": rng.normal(size=50) for i in range(4)}
        responses = {f"y{i}": rng.normal(size=50) for i in range(3)}
        matrix = screen_relationships(predictors, responses, n_permutations=500)
        assert matrix.significant(correct_multiplicity=True) == []


# ==========================================================================
# Models
# ==========================================================================
class TestLeakageProtection:
    def test_near_duplicates_are_clustered(self) -> None:
        """Two polymers differing by a methylene are not independent test cases."""
        rng = np.random.default_rng(9)
        base = rng.normal(size=(20, 4))
        with_duplicates = np.vstack([base, base[:5] + 1e-7])
        labels = cluster_by_similarity(with_duplicates, threshold=0.05)
        for i in range(5):
            assert labels[i] == labels[20 + i]

    def test_distinct_structures_are_not_clustered(self) -> None:
        rng = np.random.default_rng(10)
        labels = cluster_by_similarity(rng.normal(size=(20, 4)), threshold=0.01)
        assert len(set(labels)) == 20

    def test_grouping_keeps_near_duplicates_together(self) -> None:
        dataset = linear_dataset(40, seed=11)
        X = np.vstack([dataset.X, dataset.X[:5] + 1e-7])
        y = np.concatenate([dataset.y, dataset.y[:5]])
        expanded = Dataset(
            polymer_ids=[*dataset.polymer_ids, *[f"dup{i}" for i in range(5)]],
            feature_names=dataset.feature_names, X=X, y=y,
        )
        groups, policy = build_groups(expanded)
        assert groups[0] == groups[40]
        assert "structural similarity" in policy

    def test_measurements_from_one_trajectory_are_grouped(self) -> None:
        """Density and volume from one run are not two independent observations."""
        dataset = linear_dataset(20, seed=12)
        sources = [f"traj{i // 2}" for i in range(20)]
        groups, policy = build_groups(dataset, similarity_threshold=None, source_ids=sources)
        assert groups[0] == groups[1]
        assert groups[0] != groups[2]
        assert "simulation source" in policy

    def test_mismatched_source_ids_raise(self) -> None:
        with pytest.raises(ScientificError):
            build_groups(linear_dataset(20), source_ids=["a", "b"])


@requires_sklearn
class TestModelTraining:
    def test_a_learnable_relationship_is_learned(self) -> None:
        model = train_model(linear_dataset(80, seed=13), ModelKind.LINEAR)
        assert model.evaluation.r2 > 0.9
        assert model.evaluation.better_than_the_mean is True

    def test_pure_noise_is_not_learned(self) -> None:
        rng = np.random.default_rng(14)
        dataset = Dataset(
            polymer_ids=[f"p{i}" for i in range(80)], feature_names=["a", "b", "c", "d"],
            X=rng.normal(size=(80, 4)), y=rng.normal(size=80),
        )
        evaluation = evaluate_model(dataset, ModelKind.RANDOM_FOREST)
        assert evaluation.r2 < 0.3

    def test_a_constant_target_is_refused(self) -> None:
        dataset = Dataset(
            polymer_ids=[f"p{i}" for i in range(20)], feature_names=["a"],
            X=np.random.default_rng(15).normal(size=(20, 1)), y=np.full(20, 7.0),
        )
        with pytest.raises(ScientificError, match="constant"):
            train_model(dataset)

    def test_too_few_samples_are_refused(self) -> None:
        with pytest.raises(InsufficientDataError):
            train_model(linear_dataset(5, seed=16))

    def test_fingerprints_are_deterministic(self) -> None:
        dataset = linear_dataset(40, seed=17)
        a = train_model(dataset, ModelKind.RIDGE, evaluate=False)
        b = train_model(dataset, ModelKind.RIDGE, evaluate=False)
        assert a.dataset_fingerprint == b.dataset_fingerprint
        assert a.model_fingerprint == b.model_fingerprint

    def test_changing_the_data_changes_the_dataset_fingerprint(self) -> None:
        assert dataset_fingerprint(linear_dataset(40, seed=18)) != dataset_fingerprint(
            linear_dataset(40, seed=19)
        )

    def test_changing_hyperparameters_changes_the_model_fingerprint(self) -> None:
        dataset = linear_dataset(40, seed=20)
        a = train_model(dataset, ModelKind.RIDGE, hyperparameters={"alpha": 1.0}, evaluate=False)
        b = train_model(dataset, ModelKind.RIDGE, hyperparameters={"alpha": 10.0}, evaluate=False)
        assert a.model_fingerprint != b.model_fingerprint
        assert a.dataset_fingerprint == b.dataset_fingerprint

    def test_predictions_carry_their_provenance(self) -> None:
        model = train_model(linear_dataset(40, seed=21), ModelKind.RANDOM_FOREST, evaluate=False)
        prediction = model.predict(np.zeros((1, 4)), ["new"])[0]
        assert prediction["model_fingerprint"] == model.model_fingerprint
        assert prediction["dataset_fingerprint"] == model.dataset_fingerprint

    def test_a_training_member_is_flagged_not_scored_as_a_discovery(self) -> None:
        dataset = linear_dataset(40, seed=22)
        model = train_model(dataset, ModelKind.RANDOM_FOREST, evaluate=False)
        prediction = model.predict(dataset.X[:1], [dataset.polymer_ids[0]])[0]
        assert prediction["determination"] == Determination.REQUIRES_VALIDATION.value
        assert "training set" in prediction["notes"]

    def test_extrapolation_is_flagged(self) -> None:
        model = train_model(linear_dataset(60, seed=23), ModelKind.RANDOM_FOREST, evaluate=False)
        prediction = model.predict(np.full((1, 4), 50.0), ["far"])[0]
        assert prediction["in_domain"] is False
        assert "extrapolation" in prediction["notes"]

    def test_uncertainty_availability_is_declared_per_model(self) -> None:
        """Linear models offer no per-prediction uncertainty and must not pretend to."""
        linear = train_model(linear_dataset(40, seed=24), ModelKind.LINEAR, evaluate=False)
        forest = train_model(linear_dataset(40, seed=24), ModelKind.RANDOM_FOREST, evaluate=False)
        assert linear.provides_uncertainty is False
        assert forest.provides_uncertainty is True
        assert linear.predict(np.zeros((1, 4)), ["x"])[0]["uncertainty"] is None
        assert forest.predict(np.zeros((1, 4)), ["x"])[0]["uncertainty"] is not None

    def test_forest_uncertainty_is_labelled_uncalibrated(self) -> None:
        assert "not calibrated" in UNCERTAINTY_SOURCE[ModelKind.RANDOM_FOREST]

    def test_gaussian_process_provides_a_posterior_sd(self) -> None:
        model = train_model(linear_dataset(40, seed=25), ModelKind.GAUSSIAN_PROCESS, evaluate=False)
        prediction = model.predict(np.zeros((1, 4)), ["x"])[0]
        assert prediction["uncertainty"] is not None
        assert prediction["uncertainty"] > 0

    def test_comparing_models_ranks_them(self) -> None:
        results = compare_models(linear_dataset(80, seed=26))
        assert len(results) == 4
        assert results[0].r2 >= results[-1].r2

    def test_too_few_groups_for_the_fold_count_is_reported(self) -> None:
        dataset = linear_dataset(12, seed=27)
        # Force every row into one group so cross-validation cannot be built.
        evaluation = evaluate_model(dataset, ModelKind.RIDGE, n_folds=5, similarity_threshold=1e9)
        assert evaluation.determination is Determination.INSUFFICIENT_DATA
        assert evaluation.r2 is None
        assert "independent group" in evaluation.note

    def test_the_grouping_policy_is_recorded_with_the_score(self) -> None:
        evaluation = evaluate_model(linear_dataset(60, seed=28), ModelKind.RIDGE)
        assert "polymer identity" in evaluation.grouping
        assert evaluation.n_groups > 0
