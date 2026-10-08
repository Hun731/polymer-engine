"""Decision engine and strategy learning."""

from __future__ import annotations

import pytest

from polymer_engine.core.config import PlanningConfig
from polymer_engine.core.models import (
    Action,
    ActionStatus,
    CostEstimate,
    Determination,
    Hypothesis,
    HypothesisStatus,
)
from polymer_engine.orchestrator.planner import Feasibility, Planner
from polymer_engine.orchestrator.strategy import (
    ParameterSpec,
    Strategy,
    StrategyOutcome,
    StrategyRegistry,
)


def action(
    kind: str = "md",
    *,
    eig: float | None = 0.5,
    cost: float | None = 1.0,
    risk: float = 0.2,
    relevance: float = 0.5,
    depends_on: list[str] | None = None,
    hypothesis_id: str | None = None,
    inputs: dict | None = None,
) -> Action:
    estimate = (
        CostEstimate(gpu_hours=cost, determination=Determination.KNOWN, basis="test")
        if cost is not None
        else CostEstimate.unknown()
    )
    return Action(
        kind=kind,
        title=f"{kind} action",
        question="test question",
        cost=estimate,
        expected_information_gain=eig,
        design_relevance=relevance,
        risk=risk,
        depends_on=depends_on or [],
        hypothesis_id=hypothesis_id,
        inputs=inputs or {},
    )


class TestUnambiguousChoices:
    def test_higher_information_gain_wins_when_all_else_is_equal(self) -> None:
        low, high = action(eig=0.1), action(eig=0.9)
        decision = Planner().decide([low, high])
        assert decision.selected is high

    def test_lower_risk_wins_when_all_else_is_equal(self) -> None:
        risky, safe = action(risk=0.9), action(risk=0.1)
        assert Planner().decide([risky, safe]).selected is safe

    def test_cheaper_wins_when_all_else_is_equal(self) -> None:
        expensive, cheap = action(cost=100.0), action(cost=1.0)
        planner = Planner(PlanningConfig(max_cost_per_action=1000.0))
        assert planner.decide([expensive, cheap]).selected is cheap

    def test_higher_design_relevance_wins(self) -> None:
        low, high = action(relevance=0.1), action(relevance=0.9)
        assert Planner().decide([low, high]).selected is high


class TestFeasibility:
    def test_unmet_dependencies_block_an_action(self) -> None:
        blocked = action(depends_on=["act_missing"])
        decision = Planner().decide([blocked])
        assert decision.selected is None
        assessment = decision.assessments[0]
        assert assessment.feasibility is Feasibility.BLOCKED_BY_DEPENDENCY

    def test_satisfied_dependencies_unblock_an_action(self) -> None:
        ready = action(depends_on=["act_done"])
        decision = Planner().decide([ready], completed_action_ids={"act_done"})
        assert decision.selected is ready

    def test_missing_tools_block_an_action(self) -> None:
        needs_plumed = action(inputs={"requires_tool": "plumed"})
        decision = Planner().decide([needs_plumed], available_tools={"gromacs"})
        assert decision.selected is None
        assert decision.assessments[0].feasibility is Feasibility.TOOL_UNAVAILABLE

    def test_forbidden_kinds_are_vetoed(self) -> None:
        decision = Planner().decide([action("unregistered")], forbidden_kinds={"unregistered"})
        assert decision.selected is None
        assert decision.assessments[0].feasibility is Feasibility.POLICY_FORBIDDEN

    def test_all_actions_too_expensive_is_reported_specifically(self) -> None:
        planner = Planner(PlanningConfig(max_cost_per_action=1.0))
        decision = planner.decide([action(cost=500.0), action(cost=900.0)])
        assert decision.selected is None
        assert "cost budget" in decision.reason

    def test_no_actions_at_all(self) -> None:
        decision = Planner().decide([])
        assert decision.selected is None
        assert "no proposed or queued actions" in decision.reason

    def test_only_pending_actions_are_considered(self) -> None:
        done = action()
        done.status = ActionStatus.SUCCEEDED
        assert Planner().decide([done]).selected is None


class TestMissingInformation:
    def test_unknown_cost_is_penalised_not_treated_as_free(self) -> None:
        """An unestimated job is a commitment of unknown size, not a free one."""
        known = action(cost=0.1)
        unknown = action(cost=None)
        planner = Planner(PlanningConfig(max_cost_per_action=10.0))
        decision = planner.decide([known, unknown])
        assert decision.selected is known
        assessment = next(a for a in decision.assessments if a.action is unknown)
        assert assessment.cost_determination is Determination.UNKNOWN
        assert any("pessimistically" in n for n in assessment.notes)

    def test_missing_information_gain_contributes_nothing(self) -> None:
        estimated, unestimated = action(eig=0.5), action(eig=None)
        decision = Planner().decide([estimated, unestimated])
        assert decision.selected is estimated
        assessment = next(a for a in decision.assessments if a.action is unestimated)
        assert assessment.expected_information_gain is None
        assert "not estimated" in assessment.information_basis


class TestHypotheses:
    def test_discriminating_actions_are_preferred(self) -> None:
        hypothesis = Hypothesis(statement="s", rationale="r", status=HypothesisStatus.UNDER_TEST)
        linked = action(hypothesis_id=hypothesis.id)
        unlinked = action()
        decision = Planner().decide([unlinked, linked], hypotheses=[hypothesis])
        assert decision.selected is linked

    def test_contradictory_evidence_is_surfaced(self) -> None:
        hypothesis = Hypothesis(
            statement="s", rationale="r",
            supporting_evidence=["e1"], contradictory_evidence=["e2"],
        )
        decision = Planner().decide([action()], hypotheses=[hypothesis])
        assert hypothesis.id in decision.assessments[0].contradictions

    def test_retired_hypotheses_do_not_attract_actions(self) -> None:
        hypothesis = Hypothesis(statement="s", rationale="r", status=HypothesisStatus.RETIRED)
        linked = action(hypothesis_id=hypothesis.id)
        decision = Planner().decide([linked], hypotheses=[hypothesis])
        assert decision.assessments[0].discriminates_hypotheses == []


class TestTieBreaking:
    def test_ties_break_deterministically(self) -> None:
        a, b = action(cost=1.0), action(cost=1.0)
        first = Planner().decide([a, b]).selected
        second = Planner().decide([a, b]).selected
        assert first is not None and first.id == second.id

    def test_cheapest_tie_break_prefers_the_cheaper_action(self) -> None:
        cheap = action(cost=0.5, eig=0.5)
        pricey = action(cost=0.5, eig=0.5)
        pricey.cost = CostEstimate(gpu_hours=0.6, determination=Determination.KNOWN, basis="t")
        planner = Planner(PlanningConfig(cost_weight=0.0, tie_break="cheapest"))
        decision = planner.decide([pricey, cheap])
        assert decision.selected is cheap
        assert decision.tie_break is not None

    def test_unknown_cost_never_wins_the_cheapest_tie_break(self) -> None:
        known = action(cost=5.0)
        unknown = action(cost=None)
        planner = Planner(PlanningConfig(cost_weight=0.0, information_weight=0.0, risk_weight=0.0, value_weight=0.0))
        decision = planner.decide([unknown, known])
        assert decision.selected is known

    def test_lowest_risk_tie_break(self) -> None:
        risky = action(risk=0.4, eig=0.5)
        safe = action(risk=0.1, eig=0.5)
        planner = Planner(PlanningConfig(risk_weight=0.0, tie_break="lowest-risk"))
        assert planner.decide([risky, safe]).selected is safe


class TestDecisionRecord:
    def test_every_candidate_is_recorded_with_its_reasoning(self) -> None:
        decision = Planner().decide([action("a"), action("b"), action("c")])
        payload = decision.as_dict()
        assert len(payload["candidate_actions"]) == 3
        assert payload["selected_action"]
        assert payload["reason"]
        assert payload["estimated_cost"] is not None
        assert payload["estimated_information_gain"] is not None

    def test_a_no_action_decision_explains_itself(self) -> None:
        payload = Planner().decide([action(depends_on=["missing"])]).as_dict()
        assert payload["decision"] == "no_action"
        assert payload["selected_action"] is None
        assert payload["reason"]


# ==========================================================================
# Strategy learning
# ==========================================================================
class TestStrategyRegistry:
    def test_defaults_are_registered(self) -> None:
        registry = StrategyRegistry()
        assert len(registry) >= 5
        assert "three_replica_baseline" in registry

    def test_untried_strategies_are_unknown_not_failing(self) -> None:
        strategy = StrategyRegistry().get("three_replica_baseline")
        assert strategy.success_rate is None
        assert strategy.determination is Determination.UNKNOWN

    def test_a_thin_record_is_insufficient_data(self) -> None:
        registry = StrategyRegistry()
        registry.record_outcome(StrategyOutcome("three_replica_baseline", "c", succeeded=True))
        assert registry.get("three_replica_baseline").determination is Determination.INSUFFICIENT_DATA

    def test_successful_strategies_outrank_failing_ones(self) -> None:
        registry = StrategyRegistry()
        for _ in range(10):
            registry.record_outcome(
                StrategyOutcome("three_replica_baseline", "c", succeeded=True, cost=1.0, information_gain=0.8)
            )
            registry.record_outcome(
                StrategyOutcome("short_screen_before_production", "c", succeeded=False, cost=1.0)
            )
        ranked = [s.strategy_id for s, _ in registry.ranked()]
        assert ranked.index("three_replica_baseline") < ranked.index("short_screen_before_production")

    def test_one_lucky_run_does_not_beat_a_long_good_record(self) -> None:
        """Shrinkage toward the prior keeps small samples from dominating."""
        registry = StrategyRegistry()
        registry.record_outcome(StrategyOutcome("short_screen_before_production", "c", succeeded=True))
        for _ in range(20):
            registry.record_outcome(
                StrategyOutcome("three_replica_baseline", "c", succeeded=True, cost=1.0, information_gain=0.9)
            )
        assert registry.score(registry.get("three_replica_baseline")) > registry.score(
            registry.get("short_screen_before_production")
        )

    def test_family_filtering(self) -> None:
        registry = StrategyRegistry([
            Strategy("polyester_only", "d", applicable_families=("polyester",)),
            Strategy("universal", "d"),
        ])
        ids = {s.strategy_id for s in registry.applicable(family="polyolefin")}
        assert ids == {"universal"}
        ids = {s.strategy_id for s in registry.applicable(family="polyester")}
        assert ids == {"universal", "polyester_only"}

    def test_experiment_kind_filtering(self) -> None:
        registry = StrategyRegistry()
        ids = {s.strategy_id for s in registry.applicable(experiment_kind="umbrella_plan")}
        assert "umbrella_after_contact_evidence" in ids
        assert "descriptor_baseline_first" not in ids

    def test_unknown_strategy_raises(self) -> None:
        from polymer_engine.core.errors import PolymerEngineError

        with pytest.raises(PolymerEngineError):
            StrategyRegistry().get("nope")


class TestParameterAdaptation:
    def test_in_bounds_adaptation_is_applied(self) -> None:
        registry = StrategyRegistry()
        result = registry.adapt_parameters("three_replica_baseline", {"replicas": 5.0})
        assert result["applied"] == {"replicas": 5.0}
        assert registry.get("three_replica_baseline").parameters["replicas"].value == 5.0

    def test_out_of_bounds_adaptation_is_rejected_not_clamped(self) -> None:
        """Self-modification stops at the boundaries the author declared."""
        registry = StrategyRegistry()
        result = registry.adapt_parameters("three_replica_baseline", {"replicas": 1000.0})
        assert result["applied"] == {}
        assert "outside the declared bounds" in result["rejected"]["replicas"]
        assert registry.get("three_replica_baseline").parameters["replicas"].value == 3.0

    def test_unknown_parameters_are_rejected(self) -> None:
        result = StrategyRegistry().adapt_parameters("three_replica_baseline", {"invented": 1.0})
        assert "not a declared parameter" in result["rejected"]["invented"]

    def test_non_finite_proposals_are_rejected(self) -> None:
        result = StrategyRegistry().adapt_parameters("three_replica_baseline", {"replicas": float("nan")})
        assert result["applied"] == {}

    def test_parameter_bounds_are_inclusive(self) -> None:
        spec = ParameterSpec("x", 1.0, 0.0, 10.0)
        assert spec.within_bounds(0.0) and spec.within_bounds(10.0)
        assert not spec.within_bounds(10.001)


class TestStrategyPersistence:
    def test_registry_round_trips_through_the_store(self, tmp_path) -> None:
        from polymer_engine.db.store import Store

        store = Store(tmp_path / "e.sqlite")
        registry = StrategyRegistry()
        for _ in range(6):
            registry.record_outcome(
                StrategyOutcome("three_replica_baseline", "c", succeeded=True, cost=1.0, information_gain=0.5)
            )
        registry.save(store)

        restored = StrategyRegistry.load(store)
        strategy = restored.get("three_replica_baseline")
        assert strategy.applications == 6
        assert strategy.success_rate == 1.0
        assert strategy.determination is Determination.KNOWN
        store.close()

    def test_loading_an_empty_store_seeds_the_defaults(self, tmp_path) -> None:
        from polymer_engine.db.store import Store

        store = Store(tmp_path / "e.sqlite")
        assert len(StrategyRegistry.load(store)) >= 5
        assert len(store.list_strategies()) >= 5
        store.close()
