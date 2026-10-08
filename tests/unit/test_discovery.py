"""QSPR leakage protection, candidate selection, and rational generation."""

from __future__ import annotations

import numpy as np
import pytest

from polymer_engine.core.errors import ChemistryError, InsufficientDataError, ScientificError
from polymer_engine.core.models import Determination
from polymer_engine.discovery.active_learning import (
    Candidate,
    compute_novelty,
    pareto_front,
    score_candidates,
    screen_candidates,
    select_diverse_batch,
    select_next_experiments,
)
from polymer_engine.discovery.candidates import (
    CandidateStatus,
    Constraints,
    MutationKind,
    generate_candidates,
    insert_backbone_spacer,
    replace_backbone_heteroatom,
    substitute_side_chain,
    validate_candidate,
)
from polymer_engine.discovery.qspr import (
    Dataset,
    Preprocessor,
    QsprModel,
    audit_leakage,
    cross_validate,
    grouped_kfold_indices,
)
from tests.markers import requires_rdkit, requires_sklearn


def make_dataset(n: int = 60, seed: int = 0, *, duplicate_every: int = 0) -> Dataset:
    """A learnable synthetic dataset: y depends on the first two features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.1, size=n)
    ids = [f"pol_{i:04d}" for i in range(n)]
    if duplicate_every:
        for i in range(0, n, duplicate_every):
            ids[i] = "pol_0000"
    return Dataset(
        polymer_ids=ids,
        feature_names=["a", "b", "c", "d"],
        X=X,
        y=y,
        target_name="tg",
        target_units="K",
    )


# ==========================================================================
# Dataset integrity
# ==========================================================================
class TestDataset:
    def test_mismatched_shapes_are_rejected(self) -> None:
        with pytest.raises(ScientificError, match="Feature and target counts"):
            Dataset(polymer_ids=["a", "b"], feature_names=["x"], X=np.zeros((2, 1)), y=np.zeros(3))

    def test_missing_feature_names_are_rejected(self) -> None:
        with pytest.raises(ScientificError, match="feature name per column"):
            Dataset(polymer_ids=["a"], feature_names=["x"], X=np.zeros((1, 3)), y=np.zeros(1))

    def test_missing_ids_are_rejected(self) -> None:
        with pytest.raises(ScientificError, match="polymer id per row"):
            Dataset(polymer_ids=["a"], feature_names=["x"], X=np.zeros((2, 1)), y=np.zeros(2))

    def test_duplicates_are_detectable(self) -> None:
        assert make_dataset(20, duplicate_every=5).duplicate_ids() == ["pol_0000"]


class TestPreprocessor:
    def test_imputes_with_the_training_median(self) -> None:
        X = np.array([[1.0, 10.0], [3.0, 20.0], [np.nan, 30.0]])
        transformed = Preprocessor().fit(X).transform(X)
        assert np.all(np.isfinite(transformed))

    def test_constant_feature_does_not_divide_by_zero(self) -> None:
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        transformed = Preprocessor().fit(X).transform(X)
        assert np.all(np.isfinite(transformed))
        assert np.allclose(transformed[:, 1], 0.0)

    def test_must_be_fitted_first(self) -> None:
        with pytest.raises(ScientificError, match="fitted"):
            Preprocessor().transform(np.zeros((2, 2)))


# ==========================================================================
# Leakage protection
# ==========================================================================
class TestLeakage:
    def test_grouped_folds_never_split_a_group(self) -> None:
        groups = [f"g{i % 10}" for i in range(50)]
        folds = grouped_kfold_indices(groups, 5, seed=1)
        for fold in folds:
            held_out = {groups[i] for i in fold}
            others = {groups[i] for i in range(50) if i not in set(fold)}
            assert not (held_out & others), "a group appeared on both sides of the split"

    def test_all_samples_appear_exactly_once(self) -> None:
        groups = [f"g{i}" for i in range(20)]
        folds = grouped_kfold_indices(groups, 4, seed=1)
        assert sorted(np.concatenate(folds).tolist()) == list(range(20))

    def test_too_few_groups_for_the_fold_count_is_rejected(self) -> None:
        with pytest.raises(InsufficientDataError, match="leak"):
            grouped_kfold_indices(["a", "b"], 5)

    def test_audit_reports_overlapping_identities(self) -> None:
        assert audit_leakage(["a", "b", "c"], ["c", "d"]) == ["c"]

    def test_audit_is_clean_for_disjoint_splits(self) -> None:
        assert audit_leakage(["a", "b"], ["c", "d"]) == []


@requires_sklearn
class TestCrossValidation:
    def test_recovers_a_learnable_relationship(self) -> None:
        result = cross_validate(make_dataset(80, seed=1), n_folds=5)
        assert result.r2 is not None
        assert result.r2 > 0.7

    def test_pure_noise_does_not_score_well(self) -> None:
        """The strongest leakage signal is a model that 'predicts' noise."""
        rng = np.random.default_rng(2)
        dataset = Dataset(
            polymer_ids=[f"p{i}" for i in range(80)],
            feature_names=["a", "b", "c", "d"],
            X=rng.normal(size=(80, 4)),
            y=rng.normal(size=80),
        )
        result = cross_validate(dataset, n_folds=5)
        assert result.r2 < 0.3

    def test_too_few_samples_is_reported_not_crashed(self) -> None:
        result = cross_validate(make_dataset(6), n_folds=5)
        assert result.determination is Determination.INSUFFICIENT_DATA
        assert result.r2 is None

    def test_duplicate_polymers_are_kept_together(self) -> None:
        dataset = make_dataset(50, seed=3, duplicate_every=5)
        result = cross_validate(dataset, n_folds=5)
        assert result.r2 is not None


@requires_sklearn
class TestQsprModel:
    def test_fits_and_predicts(self) -> None:
        dataset = make_dataset(60, seed=4)
        model = QsprModel(n_estimators=50).fit(dataset)
        rng = np.random.default_rng(99)
        X = rng.normal(size=(3, 4))
        predictions = model.predict(X, ["new_a", "new_b", "new_c"])
        assert len(predictions) == 3
        assert all(p.value is not None for p in predictions)
        assert all(p.uncertainty is not None and p.uncertainty >= 0 for p in predictions)

    def test_constant_target_is_refused(self) -> None:
        dataset = Dataset(
            polymer_ids=[f"p{i}" for i in range(20)],
            feature_names=["a"],
            X=np.random.default_rng(5).normal(size=(20, 1)),
            y=np.full(20, 7.0),
        )
        with pytest.raises(ScientificError, match="constant"):
            QsprModel().fit(dataset)

    def test_non_finite_target_is_refused(self) -> None:
        dataset = make_dataset(20, seed=6)
        dataset.y[3] = np.nan
        with pytest.raises(ScientificError, match="non-finite"):
            QsprModel().fit(dataset)

    def test_too_few_samples_is_refused(self) -> None:
        with pytest.raises(InsufficientDataError):
            QsprModel(min_training_samples=10).fit(make_dataset(5, seed=7))

    def test_predicting_a_training_member_is_flagged(self) -> None:
        dataset = make_dataset(40, seed=8)
        model = QsprModel(n_estimators=30).fit(dataset)
        prediction = model.predict(dataset.X[:1], [dataset.polymer_ids[0]])[0]
        assert prediction.determination is Determination.REQUIRES_VALIDATION
        assert "training set" in prediction.note

    def test_extrapolation_is_flagged(self) -> None:
        dataset = make_dataset(60, seed=9)
        model = QsprModel(n_estimators=30).fit(dataset)
        far = np.full((1, 4), 50.0)
        prediction = model.predict(far, ["way_out_there"])[0]
        assert prediction.in_domain is False
        assert prediction.determination is Determination.REQUIRES_VALIDATION
        assert "extrapolation" in prediction.note

    def test_feature_count_mismatch_is_refused(self) -> None:
        model = QsprModel(n_estimators=20).fit(make_dataset(30, seed=10))
        with pytest.raises(ScientificError, match="do not match"):
            model.predict(np.zeros((1, 2)), ["x"])

    def test_predicting_before_fitting_is_refused(self) -> None:
        with pytest.raises(ScientificError, match="fitted"):
            QsprModel().predict(np.zeros((1, 4)), ["x"])

    def test_feature_importance_is_ranked(self) -> None:
        model = QsprModel(n_estimators=60).fit(make_dataset(80, seed=11))
        importance = model.feature_importance()
        assert next(iter(importance)) in {"a", "b"}, "the informative features should rank first"


# ==========================================================================
# Candidate selection
# ==========================================================================
def candidate(identifier: str, features: list[float], cost: float | None = 1.0) -> Candidate:
    return Candidate(polymer_id=identifier, name=identifier, features=np.array(features), cost=cost)


class TestScreening:
    def test_training_members_are_rejected(self) -> None:
        kept, rejected = screen_candidates([candidate("known", [0, 0, 0, 0])], training_ids={"known"})
        assert kept == []
        assert "training set" in rejected["known"]

    def test_duplicates_within_a_batch_are_rejected(self) -> None:
        kept, rejected = screen_candidates(
            [candidate("a", [0, 0, 0, 0]), candidate("a", [1, 1, 1, 1])], training_ids=set()
        )
        assert len(kept) == 1
        assert "duplicate" in rejected["a"]

    def test_candidates_without_features_are_rejected(self) -> None:
        kept, rejected = screen_candidates([Candidate(polymer_id="x")], training_ids=set())
        assert kept == []
        assert "feature vector" in rejected["x"]

    def test_non_finite_features_are_rejected(self) -> None:
        kept, rejected = screen_candidates(
            [candidate("bad", [0.0, np.nan, 0.0, 0.0])], training_ids=set()
        )
        assert kept == []
        assert "non-finite" in rejected["bad"]

    def test_candidates_without_an_id_are_rejected(self) -> None:
        kept, rejected = screen_candidates([Candidate(polymer_id="", name="n")], training_ids=set())
        assert kept == []
        assert rejected


class TestNovelty:
    def test_distance_to_the_nearest_training_point(self) -> None:
        training = np.array([[0.0, 0.0], [10.0, 10.0]])
        novelty = compute_novelty(np.array([[0.0, 3.0], [10.0, 10.0]]), training)
        assert novelty[0] == pytest.approx(3.0)
        assert novelty[1] == pytest.approx(0.0)

    def test_empty_training_set_gives_infinite_novelty(self) -> None:
        assert np.isinf(compute_novelty(np.array([[1.0, 2.0]]), np.empty((0, 2)))).all()


@requires_sklearn
class TestSelection:
    def test_empty_candidate_set_returns_nothing(self) -> None:
        model = QsprModel(n_estimators=20).fit(make_dataset(30, seed=12))
        report = select_next_experiments([], model, make_dataset(30, seed=12).X)
        assert report.selected == []
        assert report.scored == []

    def test_all_candidates_in_training_yields_nothing(self) -> None:
        dataset = make_dataset(30, seed=13)
        model = QsprModel(n_estimators=20).fit(dataset)
        candidates = [candidate(i, list(row)) for i, row in zip(dataset.polymer_ids[:5], dataset.X[:5], strict=True)]
        report = select_next_experiments(candidates, model, dataset.X)
        assert report.selected == []
        assert report.n_rejected == 5

    def test_selection_prefers_uncertain_candidates(self) -> None:
        dataset = make_dataset(60, seed=14)
        model = QsprModel(n_estimators=40).fit(dataset)
        rng = np.random.default_rng(15)
        candidates = [candidate(f"c{i}", list(rng.normal(size=4))) for i in range(20)]
        report = score_candidates(
            candidates, model, dataset.X,
            weight_uncertainty=1.0, weight_novelty=0.0, weight_value=0.0, weight_cost=0.0,
        )
        uncertainties = [c.uncertainty for c in report.scored]
        assert uncertainties == sorted(uncertainties, reverse=True)

    def test_normalisation_is_recorded(self) -> None:
        dataset = make_dataset(40, seed=16)
        model = QsprModel(n_estimators=20).fit(dataset)
        rng = np.random.default_rng(17)
        report = score_candidates(
            [candidate(f"c{i}", list(rng.normal(size=4))) for i in range(8)], model, dataset.X
        )
        assert set(report.normalisation) >= {"value", "uncertainty", "novelty", "cost", "weights"}

    def test_extrapolating_candidates_can_be_excluded(self) -> None:
        dataset = make_dataset(50, seed=18)
        model = QsprModel(n_estimators=20).fit(dataset)
        candidates = [candidate("far", [40.0, 40.0, 40.0, 40.0]), candidate("near", [0.1, 0.1, 0.1, 0.1])]
        report = select_next_experiments(
            candidates, model, dataset.X, allow_extrapolation=False, batch_size=5
        )
        assert [c.polymer_id for c in report.selected] == ["near"]
        assert "applicability domain" in report.rejected["far"]

    def test_diverse_batch_avoids_near_duplicates(self) -> None:
        dataset = make_dataset(40, seed=19)
        model = QsprModel(n_estimators=20).fit(dataset)
        cluster = [candidate(f"cluster{i}", [0.5 + 0.001 * i, 0.5, 0.5, 0.5]) for i in range(5)]
        spread = [candidate(f"spread{i}", [float(i) - 2.0, -1.0, 1.0, 0.0]) for i in range(3)]
        report = select_next_experiments(cluster + spread, model, dataset.X, batch_size=3)
        chosen = [c.polymer_id for c in report.selected]
        assert sum(1 for c in chosen if c.startswith("cluster")) <= 2

    def test_batch_size_must_be_positive(self) -> None:
        with pytest.raises(ScientificError):
            select_diverse_batch([], batch_size=0)


class TestParetoFront:
    def test_dominated_candidates_are_excluded(self) -> None:
        from polymer_engine.discovery.active_learning import ScoredCandidate

        def scored(identifier: str, predicted: float, uncertainty: float, cost: float):
            return ScoredCandidate(
                candidate=candidate(identifier, [0.0]),
                predicted=predicted, uncertainty=uncertainty, novelty=0.0,
                cost=cost, score=predicted, in_domain=True,
            )

        items = [
            scored("best", 10.0, 1.0, 1.0),
            scored("dominated", 5.0, 0.5, 2.0),
            scored("cheap", 6.0, 0.2, 0.5),
        ]
        front = pareto_front(items, objectives=[("predicted", "maximise"), ("cost", "minimise")])
        names = {c.polymer_id for c in front}
        assert "best" in names and "cheap" in names
        assert "dominated" not in names

    def test_empty_input(self) -> None:
        assert pareto_front([], objectives=[("predicted", "maximise")]) == []


# ==========================================================================
# Candidate generation
# ==========================================================================
@requires_rdkit
class TestGeneration:
    def test_polyethylene_yields_polypropylene(self) -> None:
        """Replacing a backbone hydrogen with a methyl is the canonical mutation."""
        generated = generate_candidates("*CC*", max_candidates=40)
        canonical = {c.canonical_repeat_unit for c in generated}
        assert "CC(*)C*" in canonical or "*CC(C)*" in canonical

    def test_backbone_heteroatom_substitution_produces_a_polyether(self) -> None:
        result = replace_backbone_heteroatom("*CCC*", "O")
        assert result is not None
        assert "O" in result

    def test_spacer_insertion_lengthens_the_backbone(self) -> None:
        from polymer_engine.polymer.descriptors import compute_descriptors

        longer = insert_backbone_spacer("*CC*", "CC")
        assert longer is not None
        assert compute_descriptors(longer).value("backbone_atom_count") == 4

    def test_every_generated_candidate_is_chemically_valid(self) -> None:
        from rdkit import Chem

        from polymer_engine.polymer.identity import capped_monomer_smiles

        for generated in generate_candidates("*CC(*)c1ccccc1", max_candidates=30):
            assert Chem.MolFromSmiles(capped_monomer_smiles(generated.repeat_unit_smiles)) is not None

    def test_every_candidate_keeps_two_attachment_points(self) -> None:
        from polymer_engine.polymer.identity import count_attachment_points

        for generated in generate_candidates("*CC*", max_candidates=30):
            assert count_attachment_points(generated.repeat_unit_smiles) == 2

    def test_no_duplicate_candidates_are_returned(self) -> None:
        generated = generate_candidates("*CC*", max_candidates=40)
        ids = [c.polymer_id for c in generated]
        assert len(ids) == len(set(ids))

    def test_the_parent_is_never_returned_as_a_candidate(self) -> None:
        generated = generate_candidates("*CC*", max_candidates=40)
        assert all(c.canonical_repeat_unit != "*CC*" for c in generated)

    def test_known_polymers_are_excluded(self) -> None:
        first = generate_candidates("*CC*", max_candidates=40)
        known = {c.polymer_id for c in first}
        second = generate_candidates("*CC*", max_candidates=40, known_ids=known)
        assert second == []

    def test_uncertain_chemistry_is_marked_for_review_not_accepted(self) -> None:
        """Silicon force-field coverage is patchy; the engine must say so."""
        generated = generate_candidates("*[Si](C)(C)O*", max_candidates=10)
        assert generated
        assert all(c.status is CandidateStatus.REQUIRES_REVIEW for c in generated)
        assert all(any("Si" in r for r in c.review_reasons) for c in generated)

    def test_mass_constraint_is_enforced(self) -> None:
        tight = Constraints(max_repeat_unit_mass=40.0)
        generated = generate_candidates("*CC*", constraints=tight, max_candidates=40)
        for candidate_ in generated:
            if candidate_.status is CandidateStatus.INVALID:
                continue
            assert candidate_.descriptors.get("repeat_unit_mass", 0) <= 40.0

    def test_element_whitelist_is_enforced(self) -> None:
        carbon_only = Constraints(allowed_elements=frozenset({"C", "H"}))
        generated = generate_candidates("*CC*", constraints=carbon_only, max_candidates=40)
        invalid = [c for c in generated if c.status is CandidateStatus.INVALID]
        assert any("disallowed element" in " ".join(c.issues) for c in invalid)

    def test_mutation_kinds_can_be_restricted(self) -> None:
        generated = generate_candidates(
            "*CC*", mutations=[MutationKind.BACKBONE_MODIFICATION], max_candidates=20
        )
        assert generated
        assert all(c.mutation is MutationKind.BACKBONE_MODIFICATION for c in generated)

    def test_invalid_parent_is_rejected(self) -> None:
        with pytest.raises(ChemistryError):
            generate_candidates("*C(((C*")

    def test_a_discrete_molecule_cannot_be_a_parent(self) -> None:
        """Mutating ethanol would yield "candidates" that are not polymers at all."""
        with pytest.raises(ChemistryError, match="not a usable repeat unit"):
            generate_candidates("CCO")

    def test_string_mutation_garbage_is_never_valid(self) -> None:
        """A malformed SMILES must not be accepted as a polymer candidate."""
        status, issues, _, _ = validate_candidate("*CC((*)", parent_smiles="*CC*")
        assert status is CandidateStatus.INVALID
        assert issues

    def test_identical_to_parent_is_rejected(self) -> None:
        status, issues, _, _ = validate_candidate("*CC*", parent_smiles="*CC*")
        assert status is CandidateStatus.INVALID
        assert any("identical to its parent" in i for i in issues)

    def test_known_duplicate_is_reported_as_duplicate(self) -> None:
        from polymer_engine.polymer.identity import canonical_repeat_unit, derive_polymer_id

        canonical, _ = canonical_repeat_unit("*CC(C)*")
        status, _, _, _ = validate_candidate(
            "*CC(C)*", parent_smiles="*CC*", known_ids=[derive_polymer_id(canonical)]
        )
        assert status is CandidateStatus.DUPLICATE

    def test_single_attachment_point_is_rejected(self) -> None:
        status, issues, _, _ = validate_candidate("*CCO", parent_smiles="*CC*")
        assert status is CandidateStatus.INVALID
        assert any("attachment point" in i for i in issues)

    def test_substitution_on_an_invalid_fragment_is_rejected(self) -> None:
        with pytest.raises(ChemistryError):
            substitute_side_chain("*CC*", "not-a-fragment")
