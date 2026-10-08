"""Executor interface.

An executor turns an :class:`Action` into an :class:`ExperimentResult`.  The contract
has one rule that overrides everything else: **the result must describe what actually
happened.**  A dry run reports ``execution_mode="dry_run"`` and never
``ActionStatus.SUCCEEDED``; a blocked action reports why it was blocked.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from polymer_engine.core.models import (
    Action,
    ActionStatus,
    ExperimentResult,
    GateReport,
    GateResult,
    GateStatus,
)


class Executor(ABC):
    """Base class for every deterministic action executor."""

    kind: str = "abstract"

    @abstractmethod
    def run(self, action: Action) -> ExperimentResult:
        """Execute the action.  Must not raise for expected scientific failures."""

    # -- helpers for consistent results --------------------------------
    @staticmethod
    def blocked(action: Action, reason: str, **evidence: Any) -> ExperimentResult:
        return ExperimentResult(
            action_id=action.id,
            status=ActionStatus.BLOCKED,
            execution_mode="blocked",
            summary=reason,
            error=reason,
            report=GateReport(
                name=f"{action.kind}:preconditions",
                gates=[
                    GateResult(
                        gate=f"{action.kind}:preconditions",
                        status=GateStatus.FAIL,
                        message=reason,
                        evidence=evidence,
                    )
                ],
            ),
        )

    @staticmethod
    def dry_run(action: Action, summary: str, artifacts: list[str] | None = None) -> ExperimentResult:
        """A dry run: inputs were checked, nothing was executed.

        Status is ``SKIPPED``, never ``SUCCEEDED``.  Reporting a dry run as a success
        is how a pipeline ends up believing it has results it never computed.
        """
        return ExperimentResult(
            action_id=action.id,
            status=ActionStatus.SKIPPED,
            execution_mode="dry_run",
            artifacts=artifacts or [],
            summary=summary,
            report=GateReport(
                name=f"{action.kind}:dry_run",
                gates=[
                    GateResult(
                        gate=f"{action.kind}:executed",
                        status=GateStatus.INCONCLUSIVE,
                        message="Execution is disabled; nothing was run and nothing was measured",
                    )
                ],
            ),
        )

    @staticmethod
    def failed(action: Action, reason: str, **evidence: Any) -> ExperimentResult:
        return ExperimentResult(
            action_id=action.id,
            status=ActionStatus.FAILED,
            execution_mode="real",
            summary=reason,
            error=reason,
            report=GateReport(
                name=f"{action.kind}:execution",
                gates=[
                    GateResult(
                        gate=f"{action.kind}:execution",
                        status=GateStatus.FAIL,
                        message=reason,
                        evidence=evidence,
                    )
                ],
            ),
        )


__all__ = ["Executor"]
