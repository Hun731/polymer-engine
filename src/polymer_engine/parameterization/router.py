"""Choosing a parameterization route, and refusing to choose when it is not obvious.

The router is deterministic and auditable: given the same assessments it returns the
same decision, and the decision carries every alternative it weighed and why it ranked
them as it did.

The rule that matters most is the one about ties. When two backends are both capable and
neither has better evidence, the router does **not** pick one. Two routes that disagree
about a force field are a scientific question, and answering it by sort order would
manufacture a decision nobody made. It returns ``REQUIRES_EXPERT_REVIEW`` with both
routes attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from polymer_engine.core.logging import get_logger
from polymer_engine.parameterization.capability import CapabilityState, at_least, rank
from polymer_engine.parameterization.models import ForceFieldAssessment, PropertyClass

logger = get_logger("parameterization.router")

#: Cost ordering, cheapest first. Only used to break ties between routes whose
#: *evidence* is equal -- never to prefer a weaker route because it is cheaper.
COST_ORDER: tuple[str, ...] = ("seconds", "seconds to minutes", "minutes",
                               "minutes (AM1-BCC charges)",
                               "minutes of human time, then a download", "unknown")


def cost_rank(cost: str) -> int:
    return COST_ORDER.index(cost) if cost in COST_ORDER else len(COST_ORDER)


@dataclass
class RouteDecision:
    """Which backend to use, what else was considered, and how confident this is."""

    polymer_id: str
    polymer_name: str
    property_class: PropertyClass
    selected_backend: str | None
    reason: str
    confidence: str = "low"
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    requires_human_step: bool = False
    ambiguous: bool = False
    considered: list[dict[str, Any]] = field(default_factory=list)

    @property
    def decided(self) -> bool:
        return self.selected_backend is not None and not self.ambiguous

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id, "polymer_name": self.polymer_name,
            "property_class": self.property_class.value,
            "selected_backend": self.selected_backend, "reason": self.reason,
            "confidence": self.confidence, "decided": self.decided,
            "ambiguous": self.ambiguous,
            "requires_human_step": self.requires_human_step,
            "alternatives": list(self.alternatives),
            "considered": list(self.considered),
        }


def _score(assessment: ForceFieldAssessment) -> tuple[int, int]:
    """Rank by capability first, then by cost.  Higher is better on the first element."""
    return (rank(assessment.state), -cost_rank(assessment.estimated_cost))


class SystemBuildRouter:
    """Pick a parameterization route from a set of assessments."""

    def route(
        self,
        assessments: list[ForceFieldAssessment],
        *,
        property_class: PropertyClass,
        prefer: str | None = None,
    ) -> RouteDecision:
        considered = [a.as_dict() for a in assessments]
        polymer_id = assessments[0].polymer_id if assessments else "unknown"
        polymer_name = assessments[0].polymer_name if assessments else "unknown"

        def decision(**kwargs: Any) -> RouteDecision:
            return RouteDecision(
                polymer_id=polymer_id, polymer_name=polymer_name,
                property_class=property_class, considered=considered, **kwargs,
            )

        # Routes that can actually produce parameters without a human step.
        automatic = [a for a in assessments
                     if at_least(a.state, CapabilityState.PARAMETERIZATION_AVAILABLE)
                     and not a.requires_human_step]
        # Routes that can get there but need a person partway.
        manual = [a for a in assessments
                  if a.requires_human_step
                  and a.state is not CapabilityState.UNAVAILABLE
                  and a.state is not CapabilityState.BLOCKED]

        if prefer:
            chosen = next((a for a in assessments if a.backend == prefer), None)
            if chosen is None:
                return decision(selected_backend=None,
                                reason=f"preferred backend {prefer!r} was not assessed")
            if not at_least(chosen.state, CapabilityState.PARAMETERIZATION_AVAILABLE):
                return decision(
                    selected_backend=None,
                    reason=(f"preferred backend {prefer!r} is {chosen.state.value}: "
                            f"{chosen.reason}"),
                )
            return decision(
                selected_backend=chosen.backend,
                reason=f"explicitly requested; {chosen.reason}",
                confidence="high", requires_human_step=chosen.requires_human_step,
            )

        if automatic:
            ordered = sorted(automatic, key=_score, reverse=True)
            best = ordered[0]
            tied = [a for a in ordered[1:] if _score(a) == _score(best)]
            if tied:
                # Two equally-capable routes with equal evidence. Choosing between them
                # is a scientific judgement, not a sort.
                return decision(
                    selected_backend=None, ambiguous=True,
                    reason=("two routes are equally supported and the evidence does not "
                            "separate them: "
                            + ", ".join(a.backend for a in [best, *tied])),
                    alternatives=[a.as_dict() for a in [best, *tied]],
                )
            return decision(
                selected_backend=best.backend,
                reason=(f"highest capability ({best.state.value}) among automatic "
                        f"routes; {best.reason}"),
                confidence="high" if at_least(
                    best.state, CapabilityState.SYSTEM_BUILD_AVAILABLE) else "medium",
                alternatives=[a.as_dict() for a in ordered[1:]],
            )

        if manual:
            best = sorted(manual, key=_score, reverse=True)[0]
            return decision(
                selected_backend=best.backend,
                reason=(f"no automatic route covers this chemistry; {best.backend} can "
                        f"reach it but needs a human step: {best.reason}"),
                confidence="medium", requires_human_step=True,
                alternatives=[a.as_dict() for a in manual if a is not best],
            )

        blocked = [a for a in assessments if a.state is CapabilityState.BLOCKED]
        unavailable = [a for a in assessments if a.state is CapabilityState.UNAVAILABLE]
        return decision(
            selected_backend=None,
            reason=(
                f"no route: {len(blocked)} backend(s) cannot type this chemistry and "
                f"{len(unavailable)} are not installed"
                + (f". Blocked because: {blocked[0].reason}" if blocked else "")
            ),
            alternatives=[a.as_dict() for a in unavailable],
        )


@dataclass
class ForceFieldComparison:
    """Every route that could parameterize one polymer, kept side by side.

    Deliberately preserves all candidates rather than collapsing to a winner: when two
    force fields both cover a chemistry, the disagreement between them is information,
    and discarding it would hide the main reason to run more than one.
    """

    polymer_id: str
    polymer_name: str
    property_class: PropertyClass
    routes: list[dict[str, Any]] = field(default_factory=list)
    decision: RouteDecision | None = None

    def add(self, assessment: ForceFieldAssessment, evidence: dict[str, Any] | None = None) -> None:
        entry = assessment.as_dict()
        entry["evidence"] = {**entry.get("evidence", {}), **(evidence or {})}
        self.routes.append(entry)

    @property
    def viable(self) -> list[dict[str, Any]]:
        return [r for r in self.routes if r.get("usable")]

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id, "polymer_name": self.polymer_name,
            "property_class": self.property_class.value,
            "n_routes": len(self.routes), "n_viable": len(self.viable),
            "routes": list(self.routes),
            "decision": self.decision.as_dict() if self.decision else None,
        }


def compare(
    assessments: list[ForceFieldAssessment], *, property_class: PropertyClass
) -> ForceFieldComparison:
    comparison = ForceFieldComparison(
        polymer_id=assessments[0].polymer_id if assessments else "unknown",
        polymer_name=assessments[0].polymer_name if assessments else "unknown",
        property_class=property_class,
    )
    for assessment in assessments:
        comparison.add(assessment)
    comparison.decision = SystemBuildRouter().route(
        assessments, property_class=property_class
    )
    return comparison


__all__ = [
    "COST_ORDER", "ForceFieldComparison", "RouteDecision", "SystemBuildRouter",
    "compare", "cost_rank",
]
