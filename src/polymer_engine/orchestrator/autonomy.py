"""The autonomous research loop.

One iteration is:

    objective -> current knowledge -> open hypotheses -> available actions
    -> utility and information gain -> resource check -> execution -> validation
    -> knowledge update -> model update -> strategy update -> next action

Three properties make this an engine rather than a script:

**It persists.**  Every iteration writes its state, so an interrupted loop resumes
where it stopped rather than restarting.

**Every decision is auditable.**  :class:`ResearchDecision` records what was
considered, what was chosen, why, what it was expected to cost, and what uncertainty it
was expected to reduce.  "Why did the engine run this?" is answerable for every job.

**Validation is not in the loop's gift.**  The loop chooses what to try and in what
order.  Whether a result may be believed is decided by the gates, and nothing here can
override them.  A strategy that keeps failing is deprioritised; it is never given an
easier test.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from polymer_engine.core.config import EngineConfig
from polymer_engine.core.logging import get_logger, log_event
from polymer_engine.core.models import (
    Action,
    ActionStatus,
    Hypothesis,
    Objective,
    utc_now,
)
from polymer_engine.orchestrator.failure_learning import (
    FailureLedger,
    FailureRecord,
    classify_failure,
)
from polymer_engine.orchestrator.planner import Decision, Planner
from polymer_engine.orchestrator.scheduler import ResourcePool
from polymer_engine.orchestrator.strategy import StrategyOutcome, StrategyRegistry
from polymer_engine.polymer.taxonomy import PolymerFamily

logger = get_logger("orchestrator.autonomy")


class LoopStatus(str, Enum):
    RUNNING = "running"
    #: No feasible action remains.
    EXHAUSTED = "exhausted"
    #: The objective's stopping criterion was met.
    OBJECTIVE_MET = "objective_met"
    #: The compute budget ran out.
    BUDGET_EXHAUSTED = "budget_exhausted"
    #: Something needs a human.
    BLOCKED = "blocked"
    STOPPED = "stopped"


@dataclass
class KnowledgeState:
    """What the engine currently believes, and how firmly."""

    objective_id: str
    n_polymers_characterised: int = 0
    n_properties_measured: int = 0
    n_claims_supported: int = 0
    model_r2: float | None = None
    model_fingerprint: str | None = None
    mean_prediction_uncertainty: float | None = None
    open_hypotheses: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: utc_now().isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "n_polymers_characterised": self.n_polymers_characterised,
            "n_properties_measured": self.n_properties_measured,
            "n_claims_supported": self.n_claims_supported,
            "model_r2": self.model_r2,
            "model_fingerprint": self.model_fingerprint,
            "mean_prediction_uncertainty": self.mean_prediction_uncertainty,
            "open_hypotheses": self.open_hypotheses,
            "updated_at": self.updated_at,
        }


@dataclass
class ResearchDecision:
    """A fully auditable record of one autonomous choice.

    The five questions this must always answer are named explicitly, so the record is
    readable by someone who was not present when it was made.
    """

    iteration: int
    why_this_candidate: str
    why_this_simulation: str
    why_now: str
    uncertainty_to_reduce: str
    design_decision_at_stake: str
    selected_action_id: str | None
    selected_action_kind: str | None
    estimated_cost: float | None
    estimated_information_gain: float | None
    planner_reason: str
    n_candidates_considered: int
    resource_snapshot: dict[str, Any] = field(default_factory=dict)
    strategy_id: str | None = None
    strategy_score: float | None = None
    failure_penalty: float = 0.0
    timestamp: str = field(default_factory=lambda: utc_now().isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": "autonomous_iteration",
            "iteration": self.iteration,
            "why_this_candidate": self.why_this_candidate,
            "why_this_simulation": self.why_this_simulation,
            "why_now": self.why_now,
            "uncertainty_to_reduce": self.uncertainty_to_reduce,
            "design_decision_at_stake": self.design_decision_at_stake,
            "selected_action": self.selected_action_id,
            "selected_action_kind": self.selected_action_kind,
            "estimated_cost": self.estimated_cost,
            "estimated_information_gain": self.estimated_information_gain,
            "reason": self.planner_reason,
            "n_candidates_considered": self.n_candidates_considered,
            "resources": self.resource_snapshot,
            "strategy_id": self.strategy_id,
            "strategy_score": self.strategy_score,
            "failure_penalty": self.failure_penalty,
            "timestamp": self.timestamp,
        }


@dataclass
class IterationResult:
    """What one turn of the loop did."""

    iteration: int
    decision: ResearchDecision
    action: Action | None
    executed: bool
    succeeded: bool
    gate_status: str | None = None
    scientifically_usable: bool = False
    summary: str = ""
    error: str | None = None
    knowledge_before: dict[str, Any] = field(default_factory=dict)
    knowledge_after: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "decision": self.decision.as_dict(),
            "action_id": self.action.id if self.action else None,
            "action_kind": self.action.kind if self.action else None,
            "executed": self.executed,
            "succeeded": self.succeeded,
            "gate_status": self.gate_status,
            "scientifically_usable": self.scientifically_usable,
            "summary": self.summary,
            "error": self.error,
            "knowledge_before": self.knowledge_before,
            "knowledge_after": self.knowledge_after,
        }


@dataclass
class LoopState:
    """Everything needed to resume the loop after an interruption."""

    objective_id: str
    iteration: int = 0
    status: LoopStatus = LoopStatus.RUNNING
    knowledge: dict[str, Any] = field(default_factory=dict)
    completed_action_ids: list[str] = field(default_factory=list)
    failed_action_ids: list[str] = field(default_factory=list)
    spent_cost: float = 0.0
    started_at: str = field(default_factory=lambda: utc_now().isoformat())
    updated_at: str = field(default_factory=lambda: utc_now().isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "iteration": self.iteration,
            "status": self.status.value,
            "knowledge": self.knowledge,
            "completed_action_ids": self.completed_action_ids,
            "failed_action_ids": self.failed_action_ids,
            "spent_cost": self.spent_cost,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LoopState:
        return cls(
            objective_id=payload["objective_id"],
            iteration=int(payload.get("iteration", 0)),
            status=LoopStatus(payload.get("status", "running")),
            knowledge=payload.get("knowledge", {}),
            completed_action_ids=list(payload.get("completed_action_ids", [])),
            failed_action_ids=list(payload.get("failed_action_ids", [])),
            spent_cost=float(payload.get("spent_cost", 0.0)),
            started_at=payload.get("started_at", utc_now().isoformat()),
            updated_at=payload.get("updated_at", utc_now().isoformat()),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> LoopState | None:
        target = Path(path)
        if not target.exists():
            return None
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))


#: Executes one action and reports the outcome.
ActionExecutor = Callable[[Action], "ExecutionOutcome"]


@dataclass
class ExecutionOutcome:
    """What an executor reports back to the loop."""

    succeeded: bool
    executed: bool = True
    gate_status: str | None = None
    scientifically_usable: bool = False
    summary: str = ""
    error: str | None = None
    gate_messages: list[str] = field(default_factory=list)
    execution_mode: str = "real"
    cost: float = 0.0
    observations: dict[str, Any] = field(default_factory=dict)


class ResearchLoop:
    """Drives the autonomous cycle, persisting state and auditing every decision."""

    def __init__(
        self,
        config: EngineConfig,
        objective: Objective,
        *,
        planner: Planner | None = None,
        strategies: StrategyRegistry | None = None,
        failures: FailureLedger | None = None,
        pool: ResourcePool | None = None,
        store: Any = None,
        state_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.objective = objective
        self.planner = planner or Planner(config.planning)
        self.strategies = strategies or StrategyRegistry()
        self.failures = failures or FailureLedger()
        self.pool = pool
        self.store = store
        self.state_path = Path(state_path) if state_path else None
        self.knowledge = KnowledgeState(objective_id=objective.id)

    # -- the loop ---------------------------------------------------------
    def run(
        self,
        actions: Sequence[Action],
        executor: ActionExecutor,
        *,
        hypotheses: Sequence[Hypothesis] = (),
        max_iterations: int = 20,
        cost_budget: float | None = None,
        state: LoopState | None = None,
        stopping_criterion: Callable[[KnowledgeState], bool] | None = None,
    ) -> tuple[LoopState, list[IterationResult]]:
        """Run until nothing feasible remains, the budget runs out, or the objective is met."""
        state = state or self._load_or_create()
        pending = {a.id: a for a in actions}
        for action_id in state.completed_action_ids:
            if action_id in pending:
                pending[action_id].status = ActionStatus.SUCCEEDED
        for action_id in state.failed_action_ids:
            if action_id in pending:
                pending[action_id].status = ActionStatus.FAILED

        results: list[IterationResult] = []
        state.status = LoopStatus.RUNNING

        while state.iteration < max_iterations:
            if cost_budget is not None and state.spent_cost >= cost_budget:
                state.status = LoopStatus.BUDGET_EXHAUSTED
                break
            if stopping_criterion and stopping_criterion(self.knowledge):
                state.status = LoopStatus.OBJECTIVE_MET
                break

            decision = self._decide(list(pending.values()), hypotheses, state)
            if decision.selected_action_id is None:
                state.status = LoopStatus.EXHAUSTED
                self._persist_decision(decision)
                break

            action = pending[decision.selected_action_id]
            knowledge_before = self.knowledge.as_dict()

            state.iteration += 1
            decision.iteration = state.iteration
            self._persist_decision(decision)

            outcome = self._execute(action, executor)
            state.spent_cost += outcome.cost

            if outcome.succeeded:
                action.status = ActionStatus.SUCCEEDED
                state.completed_action_ids.append(action.id)
            else:
                action.status = ActionStatus.FAILED
                state.failed_action_ids.append(action.id)

            self._update_knowledge(outcome)
            self._update_strategies(action, outcome)
            self._record_failure(action, outcome)

            results.append(
                IterationResult(
                    iteration=state.iteration,
                    decision=decision,
                    action=action,
                    executed=outcome.executed,
                    succeeded=outcome.succeeded,
                    gate_status=outcome.gate_status,
                    scientifically_usable=outcome.scientifically_usable,
                    summary=outcome.summary,
                    error=outcome.error,
                    knowledge_before=knowledge_before,
                    knowledge_after=self.knowledge.as_dict(),
                )
            )

            state.knowledge = self.knowledge.as_dict()
            state.updated_at = utc_now().isoformat()
            self._checkpoint(state)

        if state.status is LoopStatus.RUNNING:
            state.status = LoopStatus.STOPPED
        self._checkpoint(state)
        logger.info(
            "Research loop finished after %d iteration(s): %s", state.iteration, state.status.value
        )
        return state, results

    # -- steps -------------------------------------------------------------
    def _decide(
        self, actions: Sequence[Action], hypotheses: Sequence[Hypothesis], state: LoopState
    ) -> ResearchDecision:
        completed = set(state.completed_action_ids)
        planner_decision: Decision = self.planner.decide(
            actions,
            hypotheses=hypotheses,
            completed_action_ids=completed,
            campaign_id=self.objective.id,
        )
        selected = planner_decision.selected
        assessment = next(
            (a for a in planner_decision.assessments if selected and a.action.id == selected.id), None
        )

        strategy_id = selected.strategy_id if selected else None
        strategy_score = None
        penalty = 0.0
        if strategy_id and strategy_id in self.strategies:
            strategy = self.strategies.get(strategy_id)
            strategy_score = self.strategies.score(strategy)
            family = self._family_of(selected)
            penalty = self.failures.penalty_for(strategy_id=strategy_id, family=family)

        return ResearchDecision(
            iteration=state.iteration + 1,
            why_this_candidate=self._why_candidate(selected),
            why_this_simulation=self._why_simulation(selected),
            why_now=self._why_now(selected, planner_decision),
            uncertainty_to_reduce=self._uncertainty_target(selected, assessment),
            design_decision_at_stake=self._design_stake(selected),
            selected_action_id=selected.id if selected else None,
            selected_action_kind=selected.kind if selected else None,
            estimated_cost=assessment.cost if assessment else None,
            estimated_information_gain=(
                assessment.expected_information_gain if assessment else None
            ),
            planner_reason=planner_decision.reason,
            n_candidates_considered=len(planner_decision.assessments),
            resource_snapshot=self.pool.snapshot() if self.pool else {},
            strategy_id=strategy_id,
            strategy_score=strategy_score,
            failure_penalty=penalty,
        )

    @staticmethod
    def _family_of(action: Action | None) -> PolymerFamily:
        if action is None:
            return PolymerFamily.UNCLASSIFIED
        raw = action.inputs.get("polymer_family")
        try:
            return PolymerFamily(raw) if raw else PolymerFamily.UNCLASSIFIED
        except ValueError:
            return PolymerFamily.UNCLASSIFIED

    @staticmethod
    def _why_candidate(action: Action | None) -> str:
        if action is None:
            return "no action was selected"
        polymer = action.inputs.get("polymer_id", "unspecified polymer")
        reason = action.inputs.get("selection_reason")
        if reason:
            return f"{polymer}: {reason}"
        return f"{polymer} was the subject of the highest-utility available action"

    @staticmethod
    def _why_simulation(action: Action | None) -> str:
        if action is None:
            return "no action was selected"
        return action.question or f"{action.kind} was the action type available for this question"

    @staticmethod
    def _why_now(action: Action | None, decision: Decision) -> str:
        if action is None:
            return decision.reason
        return (
            f"its dependencies are satisfied and it scored highest among "
            f"{len([a for a in decision.assessments if a.feasible])} feasible action(s)"
        )

    @staticmethod
    def _uncertainty_target(action: Action | None, assessment: Any) -> str:
        if action is None:
            return "none"
        if assessment is not None and assessment.uncertainty_reduction is not None:
            return (
                f"expected to reduce uncertainty in {action.inputs.get('target_observable', 'the target observable')} "
                f"by a relative factor of {assessment.uncertainty_reduction:.2f}"
            )
        return (
            f"uncertainty in {action.inputs.get('target_observable', 'the target observable')}; "
            "the reduction was not quantified"
        )

    @staticmethod
    def _design_stake(action: Action | None) -> str:
        if action is None:
            return "none"
        stake = action.inputs.get("design_decision")
        if stake:
            return str(stake)
        return (
            "whether this candidate advances to synthesis-scale consideration, and whether "
            "the surrogate model's ranking of its neighbours changes"
        )

    def _execute(self, action: Action, executor: ActionExecutor) -> ExecutionOutcome:
        try:
            return executor(action)
        except Exception as exc:
            logger.exception("Action %s raised", action.id)
            return ExecutionOutcome(
                succeeded=False, executed=True, error=f"{type(exc).__name__}: {exc}",
                summary="executor raised an exception",
            )

    def _update_knowledge(self, outcome: ExecutionOutcome) -> None:
        """Only a scientifically usable result changes what the engine believes."""
        if not outcome.scientifically_usable:
            return
        observations = outcome.observations or {}
        self.knowledge.n_properties_measured += int(observations.get("n_properties", 0))
        if observations.get("polymer_characterised"):
            self.knowledge.n_polymers_characterised += 1
        if observations.get("claim_supported"):
            self.knowledge.n_claims_supported += 1
        if observations.get("model_r2") is not None:
            self.knowledge.model_r2 = float(observations["model_r2"])
        if observations.get("model_fingerprint"):
            self.knowledge.model_fingerprint = str(observations["model_fingerprint"])
        if observations.get("mean_prediction_uncertainty") is not None:
            self.knowledge.mean_prediction_uncertainty = float(
                observations["mean_prediction_uncertainty"]
            )
        self.knowledge.updated_at = utc_now().isoformat()

    def _update_strategies(self, action: Action, outcome: ExecutionOutcome) -> None:
        if not action.strategy_id or action.strategy_id not in self.strategies:
            return
        self.strategies.record_outcome(
            StrategyOutcome(
                strategy_id=action.strategy_id,
                campaign_id=action.campaign_id,
                succeeded=outcome.scientifically_usable,
                cost=outcome.cost,
                information_gain=action.expected_information_gain,
            )
        )

    def _record_failure(self, action: Action, outcome: ExecutionOutcome) -> None:
        if outcome.scientifically_usable:
            return
        failure_type = classify_failure(
            gate_messages=outcome.gate_messages,
            error=outcome.error,
            execution_mode=outcome.execution_mode,
        )
        self.failures.record(
            FailureRecord(
                failure_type=failure_type,
                cause=outcome.error or outcome.summary or "no usable result",
                simulation_kind=action.kind,
                polymer_family=self._family_of(action),
                polymer_id=action.inputs.get("polymer_id"),
                force_field=action.inputs.get("force_field"),
                strategy_id=action.strategy_id,
                campaign_id=action.campaign_id,
                resource_cost_gpu_hours=outcome.cost,
            )
        )

    # -- persistence -------------------------------------------------------
    def _load_or_create(self) -> LoopState:
        if self.state_path:
            existing = LoopState.load(self.state_path)
            if existing is not None and existing.objective_id == self.objective.id:
                logger.info(
                    "Resuming research loop at iteration %d", existing.iteration
                )
                if existing.knowledge:
                    self.knowledge = KnowledgeState(
                        objective_id=self.objective.id, **{
                            k: v for k, v in existing.knowledge.items()
                            if k not in {"objective_id", "updated_at"}
                        }
                    )
                return existing
        return LoopState(objective_id=self.objective.id)

    def _checkpoint(self, state: LoopState) -> None:
        if self.state_path:
            state.save(self.state_path)
        if self.store is not None:
            self.store.log_event("research_loop_state", state.as_dict())

    def _persist_decision(self, decision: ResearchDecision) -> None:
        payload = decision.as_dict()
        log_event(logger, "research_decision", payload)
        if self.store is not None:
            self.store.record_decision(payload, campaign_id=self.objective.id)


__all__ = [
    "ActionExecutor",
    "ExecutionOutcome",
    "IterationResult",
    "KnowledgeState",
    "LoopState",
    "LoopStatus",
    "ResearchDecision",
    "ResearchLoop",
]
