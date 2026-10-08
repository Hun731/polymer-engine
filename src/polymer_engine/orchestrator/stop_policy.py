"""When an open-ended campaign should stop, and what kind of stop it is.

A campaign is no longer a 96-hour box. It runs until *science* says stop, and the
distinction this module draws is between a stop that is a conclusion and a stop that is
an interruption:

* **COMPLETED** -- the candidate space is exhausted, or the objective has saturated.
  There is nothing more to learn by continuing.
* **SCIENTIFICALLY_BLOCKED** -- something needs a person: a force field to choose, a
  reaction coordinate to justify, a contradiction to resolve.
* **RESOURCE_BLOCKED** -- disk, memory or a missing tool. Nothing is wrong with the
  science; the machine cannot continue.
* **PAUSED** -- a transient problem. The right response to almost every failure, because
  a paused campaign resumes and a terminated one has to be re-reasoned about.
* **FAILED** -- repeated systemic failure with no path forward.

The default for anything ambiguous is PAUSED. A campaign that pauses when it should have
stopped costs an idle machine; a campaign that terminates when it should have paused
costs the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from polymer_engine.core.logging import get_logger

logger = get_logger("orchestrator.stop_policy")

#: Free space below which no new job may start. Not a hard failure: running work
#: finishes, and the campaign parks rather than corrupting a trajectory mid-write.
DEFAULT_DISK_RESERVE_GB = 100.0

#: Consecutive candidates that may fail before the campaign is treated as systemically
#: broken rather than merely unlucky. Three is enough to distinguish a bad candidate
#: from a bad machine.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3


class CampaignStatus(str, Enum):
    """What a campaign is doing, and why it is not doing anything else."""

    RUNNING = "RUNNING"
    #: Transient; resumes on its own or on the next start.
    PAUSED = "PAUSED"
    #: A person must supply something before progress is possible.
    WAITING_FOR_INPUT = "WAITING_FOR_INPUT"
    #: Disk, memory, GPU or a missing tool.
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
    #: The science cannot proceed without a decision no engine should make.
    SCIENTIFICALLY_BLOCKED = "SCIENTIFICALLY_BLOCKED"
    #: Nothing further to learn: candidates exhausted or objective saturated.
    COMPLETED = "COMPLETED"
    #: Repeated systemic failure with no path forward.
    FAILED = "FAILED"

    @property
    def resumable(self) -> bool:
        """Whether restarting the driver could sensibly continue this campaign."""
        return self in {
            CampaignStatus.RUNNING, CampaignStatus.PAUSED,
            CampaignStatus.RESOURCE_BLOCKED, CampaignStatus.WAITING_FOR_INPUT,
        }

    @property
    def terminal(self) -> bool:
        return self in {CampaignStatus.COMPLETED, CampaignStatus.FAILED}


@dataclass
class StopDecision:
    """Whether to keep going, and the reason either way."""

    status: CampaignStatus
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def should_continue(self) -> bool:
        return self.status is CampaignStatus.RUNNING

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value, "reason": self.reason,
            "should_continue": self.should_continue,
            "resumable": self.status.resumable, "terminal": self.status.terminal,
            "detail": dict(self.detail),
        }


@dataclass
class StopPolicy:
    """Configuration for when an open-ended campaign should stop.

    ``duration_hours = None`` means open-ended, which is now the default. A duration may
    still be set for a deliberately time-boxed run -- a rehearsal, or a shared machine
    with a booking -- but it is no longer an assumption baked into the engine.
    """

    duration_hours: float | None = None
    disk_reserve_gb: float = DEFAULT_DISK_RESERVE_GB
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    #: Stop once this many candidates have been validated, if set.
    target_validated: int | None = None
    #: Stop when the model stops improving by more than this over `saturation_window`
    #: successive updates. None disables the check.
    saturation_tolerance: float | None = None
    saturation_window: int = 5

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration_hours": self.duration_hours,
            "open_ended": self.duration_hours is None,
            "disk_reserve_gb": self.disk_reserve_gb,
            "max_consecutive_failures": self.max_consecutive_failures,
            "target_validated": self.target_validated,
            "saturation_tolerance": self.saturation_tolerance,
            "saturation_window": self.saturation_window,
        }

    def evaluate(
        self,
        *,
        elapsed_hours: float,
        disk_free_gb: float,
        queue_depth: int,
        n_validated: int,
        consecutive_failures: int,
        recent_scores: list[float] | None = None,
        awaiting_human: bool = False,
    ) -> StopDecision:
        """Decide whether the campaign continues.

        Ordered by severity: safety first, then things needing a person, then scientific
        conclusions, then merely running out of work.
        """
        # -- resource safety, before anything else ----------------------
        if disk_free_gb < self.disk_reserve_gb:
            return StopDecision(
                CampaignStatus.RESOURCE_BLOCKED,
                (f"free disk {disk_free_gb:.1f} GB is below the "
                 f"{self.disk_reserve_gb:.0f} GB reserve; no new job may start"),
                {"disk_free_gb": disk_free_gb},
            )

        # -- systemic failure -------------------------------------------
        if consecutive_failures >= self.max_consecutive_failures:
            return StopDecision(
                CampaignStatus.FAILED,
                (f"{consecutive_failures} consecutive candidates failed; this is a "
                 f"machine or configuration problem rather than bad luck"),
                {"consecutive_failures": consecutive_failures},
            )

        # -- a person is needed -----------------------------------------
        if awaiting_human:
            return StopDecision(
                CampaignStatus.WAITING_FOR_INPUT,
                "the next action needs a decision or an artifact only a person can supply",
            )

        # -- an explicit time box, when one was deliberately set ---------
        if self.duration_hours is not None and elapsed_hours >= self.duration_hours:
            return StopDecision(
                CampaignStatus.COMPLETED,
                (f"the configured {self.duration_hours:g} h budget is spent "
                 f"({elapsed_hours:.2f} h elapsed)"),
                {"elapsed_hours": elapsed_hours},
            )

        # -- scientific conclusions -------------------------------------
        if self.target_validated is not None and n_validated >= self.target_validated:
            return StopDecision(
                CampaignStatus.COMPLETED,
                f"{n_validated} validated candidates reached the target of "
                f"{self.target_validated}",
                {"n_validated": n_validated},
            )

        if self.saturation_tolerance is not None and recent_scores:
            window = recent_scores[-self.saturation_window:]
            if len(window) >= self.saturation_window:
                spread = max(window) - min(window)
                if spread < self.saturation_tolerance:
                    return StopDecision(
                        CampaignStatus.COMPLETED,
                        (f"the objective has saturated: model score varied by "
                         f"{spread:.4g} over the last {len(window)} updates, below the "
                         f"{self.saturation_tolerance:g} tolerance"),
                        {"window": window, "spread": spread},
                    )

        if queue_depth <= 0:
            # Out of work is not a failure, and it is not necessarily the end either --
            # new candidates may become available once a blocked family is unlocked.
            return StopDecision(
                CampaignStatus.COMPLETED,
                "no candidates remain that this campaign can act on",
                {"queue_depth": queue_depth},
            )

        return StopDecision(CampaignStatus.RUNNING, "")


__all__ = [
    "DEFAULT_DISK_RESERVE_GB", "DEFAULT_MAX_CONSECUTIVE_FAILURES",
    "CampaignStatus", "StopDecision", "StopPolicy",
]
