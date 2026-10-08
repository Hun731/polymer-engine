"""Surrogate modelling, active learning, and rational candidate design."""

from polymer_engine.discovery.active_learning import (
    Candidate,
    ScoredCandidate,
    SelectionReport,
    pareto_front,
    score_candidates,
    select_diverse_batch,
    select_next_experiments,
)
from polymer_engine.discovery.candidates import (
    CandidateStatus,
    Constraints,
    GeneratedCandidate,
    MutationKind,
    generate_candidates,
    validate_candidate,
)
from polymer_engine.discovery.qspr import (
    Dataset,
    Prediction,
    QsprModel,
    audit_leakage,
    cross_validate,
    grouped_kfold_indices,
)

__all__ = [
    "Candidate",
    "CandidateStatus",
    "Constraints",
    "Dataset",
    "GeneratedCandidate",
    "MutationKind",
    "Prediction",
    "QsprModel",
    "ScoredCandidate",
    "SelectionReport",
    "audit_leakage",
    "cross_validate",
    "generate_candidates",
    "grouped_kfold_indices",
    "pareto_front",
    "score_candidates",
    "select_diverse_batch",
    "select_next_experiments",
    "validate_candidate",
]
