"""The scientific decision engine.

The planner does **not** simply take the largest utility score.  A score is a summary,
and summarising away the reasons is how an autonomous loop ends up spending a week of
GPU time on something nobody can justify afterwards.  Instead each candidate action is
evaluated as a structured :class:`Assessment` carrying:

* the scientific question it answers
* the hypothesis it bears on
* expected information gain and how that was estimated
* computational cost, or an explicit "unknown"
* uncertainty reduction, design relevance, and risk
* hard feasibility gates that can veto an action outright

Selection then proceeds in stages: reject the infeasible, reject the unaffordable,
prefer actions that discriminate between live hypotheses, and only then rank.  Every
decision is emitted as a structured record so "why did the engine run this?" is always
answerable.

Missing information is never optimistic.  An action with no cost estimate is not
treated as free; it is treated as unknown and ranked below an equivalent action whose
cost is known, because committing to an unbounded cost is itself a risk.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from polymer_engine.core.config import PlanningConfig
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import (
    Action,
    ActionStatus,
    CostEstimate,
    Determination,
    Hypothesis,
    HypothesisStatus,
    utc_now,
)

logger = get_logger("orchestrator.planner")


class Feasibility(str, Enum):
    FEASIBLE = "feasible"
    BLOCKED_BY_DEPENDENCY = "blocked_by_dependency"
    MISSING_INPUTS = "missing_inputs"
    TOO_EXPENSIVE = "too_expensive"
    TOOL_UNAVAILABLE = "tool_unavailable"
    POLICY_FORBIDDEN = "policy_forbidden"


@dataclass
class Assessment:
    """A structured argument for or against running one action."""

    action: Action
    feasibility: Feasibility = Feasibility.FEASIBLE
    veto_reason: str = ""
    expected_information_gain: float | None = None
    information_basis: str = ""
    cost: float | None = None
    cost_determination: Determination = Determination.KNOWN
    uncertainty_reduction: float | None = None
    design_relevance: float = 0.5
    risk: float = 0.5
    discriminates_hypotheses: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.feasibility is Feasibility.FEASIBLE

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action.id,
            "kind": self.action.kind,
            "title": self.action.title,
            "question": self.action.question,
            "feasibility": self.feasibility.value,
            "veto_reason": self.veto_reason,
            "expected_information_gain": self.expected_information_gain,
            "information_basis": self.information_basis,
            "cost": self.cost,
            "cost_determination": self.cost_determination.value,
            "uncertainty_reduction": self.uncertainty_reduction,
            "design_relevance": self.design_relevance,
            "risk": self.risk,
            "discriminates_hypotheses": self.discriminates_hypotheses,
            "contradictions": self.contradictions,
            "notes": self.notes,
        }


@dataclass
class Decision:
    """The outcome of one planning round, in reconstructable form."""

    selected: Action | None
    assessments: list[Assessment]
    scores: dict[str, float]
    reason: str
    timestamp: str = field(default_factory=lambda: utc_now().isoformat())
    campaign_id: str | None = None
    tie_break: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": "select_action" if self.selected else "no_action",
            "timestamp": self.timestamp,
            "campaign_id": self.campaign_id,
            "selected_action": self.selected.id if self.selected else None,
            "selected_kind": self.selected.kind if self.selected else None,
            "selected_title": self.selected.title if self.selected else None,
            "reason": self.reason,
            "tie_break": self.tie_break,
            "estimated_cost": next(
                (a.cost for a in self.assessments if self.selected and a.action.id == self.selected.id), None
            ),
            "estimated_information_gain": next(
                (
                    a.expected_information_gain
                    for a in self.assessments
                    if self.selected and a.action.id == self.selected.id
                ),
                None,
            ),
            "candidate_actions": [a.as_dict() for a in self.assessments],
            "scores": self.scores,
        }


#: Cost assumed for an action whose cost is unknown, expressed as a fraction of the
#: per-action budget.  Deliberately pessimistic: an unestimated job is a commitment of
#: unknown size, and treating it as free would systematically favour the actions we
#: understand least.
UNKNOWN_COST_PENALTY = 0.75


class Planner:
    """Evaluates and selects the next action."""

    def __init__(self, config: PlanningConfig | None = None) -> None:
        self.config = config or PlanningConfig()

    # -- assessment ----------------------------------------------------
    def assess(
        self,
        action: Action,
        *,
        hypotheses: Sequence[Hypothesis] = (),
        completed_action_ids: set[str] | None = None,
        available_tools: set[str] | None = None,
        forbidden_kinds: set[str] | None = None,
    ) -> Assessment:
        """Build the structured argument for one action."""
        completed = completed_action_ids or set()
        assessment = Assessment(
            action=action,
            design_relevance=action.design_relevance,
            risk=action.risk,
            uncertainty_reduction=action.uncertainty_reduction,
        )

        # -- hard feasibility gates ------------------------------------
        if forbidden_kinds and action.kind in forbidden_kinds:
            assessment.feasibility = Feasibility.POLICY_FORBIDDEN
            assessment.veto_reason = f"action kind {action.kind!r} is forbidden by policy"
        else:
            missing = [d for d in action.depends_on if d not in completed]
            if missing:
                assessment.feasibility = Feasibility.BLOCKED_BY_DEPENDENCY
                assessment.veto_reason = f"waiting on {len(missing)} incomplete dependency/dependencies"
                assessment.notes.append(f"missing dependencies: {missing}")
            elif available_tools is not None:
                required = action.inputs.get("requires_tool")
                if isinstance(required, str) and required not in available_tools:
                    assessment.feasibility = Feasibility.TOOL_UNAVAILABLE
                    assessment.veto_reason = f"requires {required}, which is not available"

        # -- cost -------------------------------------------------------
        assessment.cost, assessment.cost_determination = self._cost(action.cost)
        if (
            assessment.cost is not None
            and assessment.cost > self.config.max_cost_per_action
            and assessment.feasible
        ):
            assessment.feasibility = Feasibility.TOO_EXPENSIVE
            assessment.veto_reason = (
                f"estimated cost {assessment.cost:.2f} exceeds the per-action budget "
                f"{self.config.max_cost_per_action:.2f}"
            )
        if assessment.cost_determination is not Determination.KNOWN:
            assessment.notes.append(
                "cost estimate unavailable; treated pessimistically rather than as free"
            )

        # -- information -----------------------------------------------
        if action.expected_information_gain is not None:
            assessment.expected_information_gain = action.expected_information_gain
            assessment.information_basis = "declared on the action"
        else:
            assessment.expected_information_gain = None
            assessment.information_basis = "not estimated"
            assessment.notes.append("expected information gain was not estimated")

        # -- hypothesis linkage -----------------------------------------
        live = [h for h in hypotheses if h.status in {HypothesisStatus.PROPOSED, HypothesisStatus.UNDER_TEST}]
        for hypothesis in live:
            if action.hypothesis_id == hypothesis.id or action.kind in hypothesis.discriminating_observables:
                assessment.discriminates_hypotheses.append(hypothesis.id)
        for hypothesis in hypotheses:
            if hypothesis.supporting_evidence and hypothesis.contradictory_evidence:
                assessment.contradictions.append(hypothesis.id)
        return assessment

    @staticmethod
    def _cost(estimate: CostEstimate) -> tuple[float | None, Determination]:
        scalar = estimate.scalar
        if estimate.determination is not Determination.KNOWN or scalar is None:
            return None, Determination.UNKNOWN
        return scalar, Determination.KNOWN

    # -- scoring -------------------------------------------------------
    def score(self, assessment: Assessment) -> float:
        """Combine the assessment into one comparable number.

        Used only for ranking *after* the feasibility gates have run; it never
        overrides a veto.
        """
        config = self.config
        information = assessment.expected_information_gain
        if information is None:
            # An unestimated gain contributes nothing rather than a made-up value.
            information = 0.0
        uncertainty = assessment.uncertainty_reduction or 0.0

        if assessment.cost is not None:
            cost = assessment.cost / max(config.max_cost_per_action, 1e-9)
        else:
            cost = UNKNOWN_COST_PENALTY

        score = (
            config.information_weight * information
            + config.value_weight * assessment.design_relevance
            + 0.5 * config.information_weight * uncertainty
            - config.cost_weight * cost
            - config.risk_weight * assessment.risk
        )
        # An action that separates live hypotheses is worth more than one that does not.
        if assessment.discriminates_hypotheses:
            score += 0.25 * config.value_weight * len(assessment.discriminates_hypotheses)
        return float(score)

    # -- selection -----------------------------------------------------
    def decide(
        self,
        actions: Sequence[Action],
        *,
        hypotheses: Sequence[Hypothesis] = (),
        completed_action_ids: set[str] | None = None,
        available_tools: set[str] | None = None,
        forbidden_kinds: set[str] | None = None,
        campaign_id: str | None = None,
    ) -> Decision:
        """Choose the next action, or explain why none can be run."""
        pending = [a for a in actions if a.status in {ActionStatus.PROPOSED, ActionStatus.QUEUED}]
        if not pending:
            return Decision(
                selected=None, assessments=[], scores={},
                reason="no proposed or queued actions are available",
                campaign_id=campaign_id,
            )

        assessments = [
            self.assess(
                action,
                hypotheses=hypotheses,
                completed_action_ids=completed_action_ids,
                available_tools=available_tools,
                forbidden_kinds=forbidden_kinds,
            )
            for action in pending
        ]
        feasible = [a for a in assessments if a.feasible]
        scores = {a.action.id: self.score(a) for a in assessments}

        if not feasible:
            blocked = {a.feasibility.value for a in assessments}
            if blocked == {Feasibility.TOO_EXPENSIVE.value}:
                reason = "every candidate action exceeds the per-action cost budget"
            else:
                reason = (
                    "no candidate action is feasible: "
                    + ", ".join(sorted(f"{a.action.kind}={a.feasibility.value}" for a in assessments))
                )
            return Decision(
                selected=None, assessments=assessments, scores=scores, reason=reason, campaign_id=campaign_id
            )

        best_score = max(scores[a.action.id] for a in feasible)
        tied = [a for a in feasible if math.isclose(scores[a.action.id], best_score, rel_tol=1e-9, abs_tol=1e-9)]

        tie_break: str | None = None
        if len(tied) == 1:
            chosen = tied[0]
        else:
            chosen, tie_break = self._break_tie(tied)

        reason = self._explain(chosen, scores[chosen.action.id], len(feasible), len(assessments))
        logger.info("Planner selected %s (%s): %s", chosen.action.id, chosen.action.kind, reason)
        return Decision(
            selected=chosen.action,
            assessments=assessments,
            scores=scores,
            reason=reason,
            campaign_id=campaign_id,
            tie_break=tie_break,
        )

    def _break_tie(self, tied: list[Assessment]) -> tuple[Assessment, str]:
        """Deterministic tie-breaking, so a replayed campaign makes the same choice."""
        strategy = self.config.tie_break
        if strategy == "cheapest":
            # Unknown cost sorts last: it is not cheaper, it is unmeasured.
            ordered = sorted(
                tied, key=lambda a: (a.cost is None, a.cost if a.cost is not None else 0.0, a.action.id)
            )
            return ordered[0], "cheapest feasible action (unknown cost ranked last)"
        if strategy == "lowest-risk":
            ordered = sorted(tied, key=lambda a: (a.risk, a.action.id))
            return ordered[0], "lowest risk"
        ordered = sorted(tied, key=lambda a: a.action.id)
        return ordered[0], "lowest action id (deterministic fallback)"

    @staticmethod
    def _explain(assessment: Assessment, score: float, n_feasible: int, n_total: int) -> str:
        parts = [f"highest-scoring of {n_feasible} feasible action(s) out of {n_total} considered"]
        if assessment.expected_information_gain is not None:
            parts.append(f"expected information gain {assessment.expected_information_gain:.2f}")
        else:
            parts.append("information gain not estimated")
        if assessment.cost is not None:
            parts.append(f"estimated cost {assessment.cost:.2f}")
        else:
            parts.append("cost unknown (penalised)")
        if assessment.discriminates_hypotheses:
            parts.append(f"discriminates {len(assessment.discriminates_hypotheses)} live hypothesis/es")
        parts.append(f"score {score:.3f}")
        return "; ".join(parts)


__all__ = ["UNKNOWN_COST_PENALTY", "Assessment", "Decision", "Feasibility", "Planner"]
