"""The autonomous research loop and failure learning.

The loop must be auditable, resumable, and unable to promote a result on its own.
"""

from __future__ import annotations

import json

import pytest

from polymer_engine.core.config import load_config
from polymer_engine.core.models import (
    Action,
    CostEstimate,
    Determination,
    Objective,
)
from polymer_engine.orchestrator.autonomy import (
    ExecutionOutcome,
    LoopState,
    LoopStatus,
    ResearchLoop,
)
from polymer_engine.orchestrator.failure_learning import (
    MIN_OCCURRENCES_FOR_PATTERN,
    FailureLedger,
    FailureRecord,
    FailureType,
    classify_failure,
)
from polymer_engine.polymer.taxonomy import PolymerFamily


@pytest.fixture
def config():
    return load_config(discover=False, use_env=False)


@pytest.fixture
def objective():
    return Objective(title="Find high-Tg polymers", description="maximise glass transition")


def make_action(
    kind: str = "gromacs_equilibrate",
    *,
    gpu_hours: float = 2.0,
    eig: float = 0.8,
    depends_on: list[str] | None = None,
    family: str = "polyolefin",
    strategy_id: str | None = "three_replica_baseline",
    **inputs,
) -> Action:
    payload = {
        "polymer_id": "pol_A",
        "polymer_family": family,
        "target_observable": "density",
        "selection_reason": "highest surrogate uncertainty",
    }
    payload.update(inputs)
    return Action(
        kind=kind,
        title=f"{kind} action",
        question="What is the equilibrium density?",
        strategy_id=strategy_id,
        cost=CostEstimate(gpu_hours=gpu_hours, determination=Determination.KNOWN, basis="test"),
        expected_information_gain=eig,
        uncertainty_reduction=0.5,
        depends_on=depends_on or [],
        inputs=payload,
    )


def succeeding(action: Action) -> ExecutionOutcome:
    return ExecutionOutcome(
        succeeded=True, scientifically_usable=True, gate_status="pass",
        summary="completed", cost=2.0,
        observations={"n_properties": 1, "polymer_characterised": True},
    )


def failing(action: Action) -> ExecutionOutcome:
    return ExecutionOutcome(
        succeeded=False, scientifically_usable=False, gate_status="fail",
        summary="gate failed", error="density:effective_samples: Only 3 effective samples",
        gate_messages=["Only 3 effective samples"], cost=2.0,
    )


# ==========================================================================
# Auditability
# ==========================================================================
class TestAuditability:
    def test_every_iteration_answers_the_five_questions(self, config, objective, tmp_path) -> None:
        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        _, results = loop.run([make_action()], succeeding, max_iterations=3)
        assert results
        decision = results[0].decision
        assert decision.why_this_candidate
        assert decision.why_this_simulation
        assert decision.why_now
        assert decision.uncertainty_to_reduce
        assert decision.design_decision_at_stake

    def test_the_selection_reason_reaches_the_record(self, config, objective, tmp_path) -> None:
        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        _, results = loop.run(
            [make_action(selection_reason="largest predicted Tg gain")], succeeding, max_iterations=2
        )
        assert "largest predicted Tg gain" in results[0].decision.why_this_candidate

    def test_cost_and_information_gain_are_recorded(self, config, objective, tmp_path) -> None:
        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        _, results = loop.run([make_action(gpu_hours=4.0, eig=0.9)], succeeding, max_iterations=2)
        decision = results[0].decision
        assert decision.estimated_cost == pytest.approx(4.0)
        assert decision.estimated_information_gain == pytest.approx(0.9)

    def test_decisions_reach_the_store(self, config, objective, tmp_path) -> None:
        from polymer_engine.db.store import Store

        with Store(tmp_path / "e.sqlite") as store:
            loop = ResearchLoop(config, objective, store=store, state_path=tmp_path / "loop.json")
            loop.run([make_action()], succeeding, max_iterations=2)
            decisions = store.list_decisions(campaign_id=objective.id)
            assert decisions
            assert decisions[0]["why_this_candidate"]
            assert decisions[0]["decision"] == "autonomous_iteration"

    def test_the_decision_record_is_serialisable(self, config, objective, tmp_path) -> None:
        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        _, results = loop.run([make_action()], succeeding, max_iterations=2)
        assert json.dumps(results[0].as_dict())


# ==========================================================================
# Knowledge and validation
# ==========================================================================
class TestKnowledgeUpdates:
    def test_a_usable_result_updates_knowledge(self, config, objective, tmp_path) -> None:
        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        loop.run([make_action()], succeeding, max_iterations=2)
        assert loop.knowledge.n_polymers_characterised == 1
        assert loop.knowledge.n_properties_measured == 1

    def test_an_unusable_result_does_not_update_knowledge(self, config, objective, tmp_path) -> None:
        """The loop cannot promote a result the gates refused."""
        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        loop.run([make_action()], failing, max_iterations=2)
        assert loop.knowledge.n_polymers_characterised == 0
        assert loop.knowledge.n_properties_measured == 0

    def test_a_succeeded_but_gate_failing_result_is_not_knowledge(
        self, config, objective, tmp_path
    ) -> None:
        """Exit status is not the criterion; scientific usability is."""

        def exit_zero_but_gates_failed(action: Action) -> ExecutionOutcome:
            return ExecutionOutcome(
                succeeded=True, scientifically_usable=False, gate_status="fail",
                summary="ran cleanly but did not converge", cost=1.0,
                gate_messages=["density:drift: still drifting"],
                observations={"n_properties": 5, "polymer_characterised": True},
            )

        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        loop.run([make_action()], exit_zero_but_gates_failed, max_iterations=2)
        assert loop.knowledge.n_properties_measured == 0

    def test_the_loop_stops_when_the_objective_is_met(self, config, objective, tmp_path) -> None:
        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        actions = [make_action() for _ in range(5)]
        state, _ = loop.run(
            actions, succeeding, max_iterations=10,
            stopping_criterion=lambda k: k.n_polymers_characterised >= 2,
        )
        assert state.status is LoopStatus.OBJECTIVE_MET
        assert state.iteration == 2

    def test_the_loop_stops_on_budget(self, config, objective, tmp_path) -> None:
        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        actions = [make_action() for _ in range(10)]
        state, _ = loop.run(actions, succeeding, max_iterations=10, cost_budget=5.0)
        assert state.status is LoopStatus.BUDGET_EXHAUSTED
        assert state.spent_cost >= 5.0

    def test_the_loop_stops_when_nothing_is_feasible(self, config, objective, tmp_path) -> None:
        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        blocked = make_action(depends_on=["never_completed"])
        state, results = loop.run([blocked], succeeding, max_iterations=5)
        assert state.status is LoopStatus.EXHAUSTED
        assert results == []


# ==========================================================================
# Resumption
# ==========================================================================
class TestResumption:
    def test_state_is_persisted_each_iteration(self, config, objective, tmp_path) -> None:
        path = tmp_path / "loop.json"
        loop = ResearchLoop(config, objective, state_path=path)
        loop.run([make_action(), make_action()], succeeding, max_iterations=2)
        assert path.exists()
        assert LoopState.load(path).iteration == 2

    def test_an_interrupted_loop_resumes_where_it_stopped(self, config, objective, tmp_path) -> None:
        path = tmp_path / "loop.json"
        actions = [make_action() for _ in range(4)]

        first = ResearchLoop(config, objective, state_path=path)
        state_a, _ = first.run(actions, succeeding, max_iterations=2)
        assert state_a.iteration == 2

        # A fresh loop object, as after a process restart.
        second = ResearchLoop(config, objective, state_path=path)
        fresh_actions = [make_action() for _ in range(4)]
        for action, original in zip(fresh_actions, actions, strict=True):
            action.id = original.id
        state_b, results = second.run(fresh_actions, succeeding, max_iterations=4)
        assert state_b.iteration == 4
        assert len(results) == 2, "only the remaining iterations should run"

    def test_knowledge_survives_a_restart(self, config, objective, tmp_path) -> None:
        path = tmp_path / "loop.json"
        first = ResearchLoop(config, objective, state_path=path)
        first.run([make_action(), make_action()], succeeding, max_iterations=2)
        assert first.knowledge.n_polymers_characterised == 2

        second = ResearchLoop(config, objective, state_path=path)
        second.run([], succeeding, max_iterations=1)
        assert second.knowledge.n_polymers_characterised == 2

    def test_a_different_objective_does_not_resume_the_wrong_state(
        self, config, objective, tmp_path
    ) -> None:
        path = tmp_path / "loop.json"
        ResearchLoop(config, objective, state_path=path).run(
            [make_action()], succeeding, max_iterations=1
        )
        other = Objective(title="different", description="different question")
        loop = ResearchLoop(config, other, state_path=path)
        state, _ = loop.run([make_action()], succeeding, max_iterations=1)
        assert state.objective_id == other.id
        assert state.iteration == 1


# ==========================================================================
# Robustness
# ==========================================================================
class TestRobustness:
    def test_an_exploding_executor_does_not_kill_the_loop(self, config, objective, tmp_path) -> None:
        calls = {"n": 0}

        def flaky(action: Action) -> ExecutionOutcome:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("segfault")
            return succeeding(action)

        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        _, results = loop.run([make_action(), make_action()], flaky, max_iterations=3)
        assert len(results) == 2
        assert results[0].succeeded is False
        assert "segfault" in results[0].error
        assert results[1].succeeded is True

    def test_strategy_outcomes_are_recorded(self, config, objective, tmp_path) -> None:
        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        loop.run([make_action() for _ in range(3)], succeeding, max_iterations=3)
        strategy = loop.strategies.get("three_replica_baseline")
        assert strategy.applications == 3
        assert strategy.successes == 3

    def test_failures_are_recorded_with_a_classification(self, config, objective, tmp_path) -> None:
        loop = ResearchLoop(config, objective, state_path=tmp_path / "loop.json")
        loop.run([make_action()], failing, max_iterations=2)
        assert len(loop.failures) == 1
        assert loop.failures.records()[0].failure_type is FailureType.INSUFFICIENT_SAMPLING


# ==========================================================================
# Failure learning
# ==========================================================================
class TestFailureLearning:
    def test_a_repeated_failure_becomes_a_pattern(self) -> None:
        ledger = FailureLedger()
        for _ in range(MIN_OCCURRENCES_FOR_PATTERN):
            ledger.record(
                FailureRecord(
                    FailureType.SIMULATION_DIVERGED, "LINCS warnings then NaN",
                    "gromacs_equilibrate", PolymerFamily.POLYESTER,
                    strategy_id="fast_screen", force_field="GAFF2",
                    resource_cost_gpu_hours=2.0,
                )
            )
        patterns = ledger.patterns()
        assert patterns
        assert any("fast_screen" in p.key for p in patterns)
        assert patterns[0].occurrences >= MIN_OCCURRENCES_FOR_PATTERN

    def test_a_single_failure_is_not_a_pattern(self) -> None:
        """One bad run must not make the engine superstitious."""
        ledger = FailureLedger()
        ledger.record(
            FailureRecord(
                FailureType.SIMULATION_DIVERGED, "diverged", "md",
                PolymerFamily.POLYESTER, strategy_id="fast_screen",
            )
        )
        assert ledger.patterns() == []

    def test_the_penalty_deprioritises_without_forbidding(self) -> None:
        """A strategy that failed three times may still be right for the fourth."""
        ledger = FailureLedger()
        for _ in range(6):
            ledger.record(
                FailureRecord(
                    FailureType.SIMULATION_DIVERGED, "diverged", "md",
                    PolymerFamily.POLYESTER, strategy_id="fast_screen",
                )
            )
        penalty = ledger.penalty_for(strategy_id="fast_screen", family=PolymerFamily.POLYESTER)
        assert 0.0 < penalty < 1.0

    def test_a_penalty_is_family_specific(self) -> None:
        ledger = FailureLedger()
        for _ in range(5):
            ledger.record(
                FailureRecord(
                    FailureType.SIMULATION_DIVERGED, "diverged", "md",
                    PolymerFamily.POLYESTER, strategy_id="fast_screen",
                )
            )
        assert ledger.penalty_for(strategy_id="fast_screen", family=PolymerFamily.POLYESTER) > 0
        assert ledger.penalty_for(strategy_id="fast_screen", family=PolymerFamily.POLYOLEFIN) == 0.0

    def test_recoverability_differs_by_failure_type(self) -> None:
        assert FailureType.INSUFFICIENT_SAMPLING.recoverable is True
        assert FailureType.SYSTEM_INVALID.recoverable is False
        assert "rerunning will fail identically" in FailureType.SYSTEM_INVALID.suggested_action

    def test_wasted_cost_is_tracked(self) -> None:
        ledger = FailureLedger()
        ledger.record(
            FailureRecord(
                FailureType.TIMEOUT, "too slow", "md", PolymerFamily.POLYOLEFIN,
                resource_cost_gpu_hours=4.0, resource_cost_cpu_hours=8.0,
            )
        )
        assert ledger.total_wasted_cost() == pytest.approx(5.0)

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("Only 2.9 effective samples", FailureType.INSUFFICIENT_SAMPLING),
            ("Observable is still drifting", FailureType.NOT_CONVERGED),
            ("Replicas disagree: reduced chi-square 40", FailureType.REPLICA_DISAGREEMENT),
            ("insufficient overlap between windows", FailureType.POOR_OVERLAP),
            ("SCF did not converge", FailureType.QM_NOT_CONVERGED),
            ("LINCS warnings", FailureType.SIMULATION_DIVERGED),
            ("grompp failed", FailureType.TOOL_FAILURE),
        ],
    )
    def test_classification_from_gate_messages(self, message: str, expected: FailureType) -> None:
        assert classify_failure(gate_messages=[message]) is expected

    def test_an_unrecognised_failure_is_unknown_not_guessed(self) -> None:
        assert classify_failure(error="the flux capacitor destabilised") is FailureType.UNKNOWN

    def test_a_blocked_system_is_classified_from_context(self) -> None:
        assert classify_failure(
            gate_messages=["No topology (.top) file found"], execution_mode="blocked"
        ) is FailureType.SYSTEM_INVALID

    def test_the_ledger_round_trips_through_the_store(self, tmp_path) -> None:
        from polymer_engine.db.store import Store

        with Store(tmp_path / "e.sqlite") as store:
            ledger = FailureLedger()
            ledger.record(
                FailureRecord(
                    FailureType.POOR_OVERLAP, "gaps at 1.2 nm", "umbrella",
                    PolymerFamily.POLYAMIDE, strategy_id="umbrella_after_contact_evidence",
                    resource_cost_gpu_hours=12.0,
                )
            )
            ledger.save(store)
            restored = FailureLedger.load(store)
            assert len(restored) == 1
            assert restored.records()[0].failure_type is FailureType.POOR_OVERLAP
            assert restored.total_wasted_cost() == pytest.approx(12.0)

    def test_the_report_summarises_everything(self) -> None:
        ledger = FailureLedger()
        for kind in (FailureType.TIMEOUT, FailureType.TIMEOUT, FailureType.SYSTEM_INVALID):
            ledger.record(FailureRecord(kind, "x", "md", PolymerFamily.POLYOLEFIN))
        report = ledger.report()
        assert report["n_failures"] == 3
        assert report["by_type"]["timeout"] == 2
        assert report["recoverable"] == 2
        assert report["unrecoverable"] == 1
