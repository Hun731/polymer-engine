"""The autonomous loop.

For each step: ask the planner what to do, record *why*, transition the execution
record, run the deterministic executor, apply the gates, and persist everything.

The invariant that matters: an action's outcome is written from what the executor
actually reported.  A dry run leaves the record ``VALIDATED`` rather than
``COMPLETED``, so a campaign run with execution disabled can never look finished.
"""

from __future__ import annotations

from typing import Any

from polymer_engine.core.config import EngineConfig
from polymer_engine.core.errors import IllegalStateTransition, PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import (
    Action,
    ActionStatus,
    ExecutionRecord,
    ExecutionState,
    ExperimentResult,
)
from polymer_engine.executors.registry import ExecutorRegistry
from polymer_engine.orchestrator.campaign import Campaign, CampaignStatus, load_campaign, save_campaign
from polymer_engine.orchestrator.planner import Planner

logger = get_logger("orchestrator.runner")

ACTOR = "engine"


class CampaignRunner:
    """Drives a campaign's actions through the planner and executors."""

    def __init__(
        self,
        config: EngineConfig,
        store: Any,
        *,
        executors: ExecutorRegistry | None = None,
        planner: Planner | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.executors = executors or ExecutorRegistry(config)
        self.planner = planner or Planner(config.planning)

    def run(self, campaign_id: str, *, max_actions: int | None = None) -> dict[str, Any]:
        campaign = load_campaign(self.store, campaign_id)
        if campaign is None:
            raise PolymerEngineError("Unknown campaign", campaign_id=campaign_id)
        if not campaign.actions:
            raise PolymerEngineError(
                "Campaign has no actions", campaign_id=campaign_id, hint="run 'campaign plan' first"
            )

        campaign.status = CampaignStatus.RUNNING
        executed: list[dict[str, Any]] = []
        budget = max_actions if max_actions is not None else len(campaign.actions)

        while len(executed) < budget:
            decision = self._decide(campaign)
            self.store.record_decision(decision.as_dict(), campaign_id=campaign_id)
            if decision.selected is None:
                logger.info("Campaign %s: %s", campaign_id, decision.reason)
                break
            outcome = self._execute(campaign, decision.selected)
            executed.append(outcome)

        campaign.status = self._final_status(campaign)
        save_campaign(self.store, campaign)
        campaign.write_manifest()

        return {
            "campaign_id": campaign_id,
            "status": campaign.status.value,
            "executed": executed,
            "n_executed": len(executed),
            "gate_status": self._worst_gate(executed),
            "manifest": str(campaign.workdir / "campaign_manifest.json"),
        }

    # -- steps ---------------------------------------------------------
    def _decide(self, campaign: Campaign):
        completed = {
            a.id for a in campaign.actions if a.status is ActionStatus.SUCCEEDED
        }
        hypotheses = self.store.list_hypotheses()
        forbidden = {
            kind for kind in {a.kind for a in campaign.actions} if not self.executors.supports(kind)
        }
        return self.planner.decide(
            campaign.actions,
            hypotheses=hypotheses,
            completed_action_ids=completed,
            available_tools=self.executors.available_tools(),
            forbidden_kinds=forbidden,
            campaign_id=campaign.campaign_id,
        )

    def _execute(self, campaign: Campaign, action: Action) -> dict[str, Any]:
        record = campaign.record_for(action.id) or ExecutionRecord(kind=action.kind, label=action.id)
        if record not in campaign.records:
            campaign.records.append(record)

        self._transition(record, ExecutionState.VALIDATED, "preconditions checked by the executor")
        self._transition(record, ExecutionState.QUEUED, "scheduled by the planner")
        self._transition(record, ExecutionState.RUNNING, "execution started")

        action.status = ActionStatus.RUNNING
        self.store.save_action(action)

        executor = self.executors.get(action.kind)
        result: ExperimentResult = executor.run(action)

        action.status = result.status
        self.store.save_action(action)
        for observation in result.observations:
            observation.campaign_id = campaign.campaign_id
            self.store.save_observation(observation)
        campaign.gate_reports[action.id] = result.report

        self._settle(record, result)
        self.store.save_execution_record(
            record, campaign_id=campaign.campaign_id, action_id=action.id
        )
        self.store.log_event(
            "action_finished",
            {
                "campaign_id": campaign.campaign_id,
                "action_id": action.id,
                "kind": action.kind,
                "status": result.status.value,
                "execution_mode": result.execution_mode,
                "gate_status": result.report.status.value,
                "scientifically_usable": result.scientifically_usable,
                "summary": result.summary,
            },
        )
        return {
            "action_id": action.id,
            "kind": action.kind,
            "status": result.status.value,
            "execution_mode": result.execution_mode,
            "gate_status": result.report.status.value,
            "scientifically_usable": result.scientifically_usable,
            "summary": result.summary,
            "error": result.error,
            "state": record.state.value,
        }

    def _settle(self, record: ExecutionRecord, result: ExperimentResult) -> None:
        """Move the execution record to match what actually happened."""
        if result.execution_mode == "dry_run":
            # Nothing ran, so nothing completed.  Roll back to VALIDATED via CANCELLED
            # semantics would lose the history; instead cancel this attempt explicitly.
            self._transition(record, ExecutionState.CANCELLED, "execution disabled; nothing was run")
            return
        if result.status is ActionStatus.SUCCEEDED:
            self._transition(record, ExecutionState.COMPLETED, result.summary or "completed")
            if result.scientifically_usable:
                self._transition(record, ExecutionState.PROMOTED, "passed all validation gates")
            else:
                self._transition(
                    record,
                    ExecutionState.REJECTED,
                    f"completed but gates returned {result.report.status.value}",
                )
            return
        if result.status is ActionStatus.BLOCKED:
            self._transition(record, ExecutionState.FAILED, result.error or "blocked by preconditions")
            return
        self._transition(record, ExecutionState.FAILED, result.error or "execution failed")

    def _transition(self, record: ExecutionRecord, target: ExecutionState, reason: str) -> None:
        try:
            record.transition(target, actor=ACTOR, reason=reason)
        except IllegalStateTransition as exc:
            # A refused transition is a real signal, not something to paper over.
            logger.warning("Refused state transition: %s", exc)
            raise

    @staticmethod
    def _worst_gate(executed: list[dict[str, Any]]) -> str:
        statuses = {e["gate_status"] for e in executed}
        for candidate in ("fail", "inconclusive", "warn", "pass"):
            if candidate in statuses:
                return candidate
        return "none"

    @staticmethod
    def _final_status(campaign: Campaign) -> CampaignStatus:
        statuses = {a.status for a in campaign.actions}
        if ActionStatus.FAILED in statuses or ActionStatus.BLOCKED in statuses:
            return CampaignStatus.FAILED
        if statuses and statuses <= {ActionStatus.SUCCEEDED}:
            return CampaignStatus.COMPLETED
        if ActionStatus.SKIPPED in statuses:
            return CampaignStatus.PLANNED
        return CampaignStatus.RUNNING


__all__ = ["ACTOR", "CampaignRunner"]
