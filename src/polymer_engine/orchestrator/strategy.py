"""Strategy registry and performance learning.

The engine adapts by learning **which strategies work**, not by rewriting its own
source.  A strategy is a named, parameterised way of pursuing a question (for example
"screen with a short NPT run before committing to production", or "use umbrella
sampling only once cohesion contacts exceed a threshold").  Outcomes are recorded, and
a strategy's future priority follows from its record.

The autonomy boundary is explicit: adaptation happens through strategy ranking,
parameter adjustment within declared bounds, candidate selection, and scheduling.  It
never happens by modifying Python source, and it never relaxes a validation gate --
:meth:`StrategyRegistry.adapt_parameters` refuses to move a parameter outside the
bounds its author declared.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from polymer_engine.core.errors import PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination, utc_now

logger = get_logger("orchestrator.strategy")


@dataclass
class ParameterSpec:
    """A tunable parameter with hard bounds its author declared."""

    name: str
    value: float
    minimum: float
    maximum: float
    description: str = ""

    def clamped(self, proposed: float) -> float:
        return min(self.maximum, max(self.minimum, proposed))

    def within_bounds(self, proposed: float) -> bool:
        return self.minimum <= proposed <= self.maximum

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "description": self.description,
        }


@dataclass
class StrategyOutcome:
    """One recorded application of a strategy."""

    strategy_id: str
    campaign_id: str | None
    succeeded: bool
    cost: float | None = None
    information_gain: float | None = None
    prediction_improvement: float | None = None
    notes: str = ""
    timestamp: str = field(default_factory=lambda: utc_now().isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "campaign_id": self.campaign_id,
            "succeeded": self.succeeded,
            "cost": self.cost,
            "information_gain": self.information_gain,
            "prediction_improvement": self.prediction_improvement,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }


@dataclass
class Strategy:
    """A named approach the engine can choose between."""

    strategy_id: str
    description: str
    applicable_families: tuple[str, ...] = ()
    experiment_kinds: tuple[str, ...] = ()
    parameters: dict[str, ParameterSpec] = field(default_factory=dict)
    enabled: bool = True

    # accumulated record
    applications: int = 0
    successes: int = 0
    failures: int = 0
    total_cost: float = 0.0
    total_information_gain: float = 0.0
    total_prediction_improvement: float = 0.0

    def applies_to(self, family: str | None) -> bool:
        if not self.applicable_families:
            return True
        return family is not None and family in self.applicable_families

    @property
    def success_rate(self) -> float | None:
        """Fraction of applications that succeeded, or ``None`` with no record.

        ``None`` rather than 0.0: an untried strategy has not failed.
        """
        if self.applications == 0:
            return None
        return self.successes / self.applications

    @property
    def failure_rate(self) -> float | None:
        if self.applications == 0:
            return None
        return self.failures / self.applications

    @property
    def mean_cost(self) -> float | None:
        return self.total_cost / self.applications if self.applications else None

    @property
    def mean_information_gain(self) -> float | None:
        return self.total_information_gain / self.applications if self.applications else None

    @property
    def mean_prediction_improvement(self) -> float | None:
        return self.total_prediction_improvement / self.applications if self.applications else None

    @property
    def determination(self) -> Determination:
        """How much the record can be trusted.

        Below :data:`MIN_APPLICATIONS_FOR_CONFIDENCE` the record is real but too thin
        to rank on, and the registry says so rather than treating one lucky run as
        evidence of a good strategy.
        """
        if self.applications == 0:
            return Determination.UNKNOWN
        if self.applications < MIN_APPLICATIONS_FOR_CONFIDENCE:
            return Determination.INSUFFICIENT_DATA
        return Determination.KNOWN

    def information_per_cost(self) -> float | None:
        gain, cost = self.mean_information_gain, self.mean_cost
        if gain is None or cost is None or cost <= 0:
            return None
        return gain / cost

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "description": self.description,
            "applicable_families": list(self.applicable_families),
            "experiment_kinds": list(self.experiment_kinds),
            "parameters": {k: v.as_dict() for k, v in self.parameters.items()},
            "enabled": self.enabled,
            "applications": self.applications,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "mean_cost": self.mean_cost,
            "mean_information_gain": self.mean_information_gain,
            "mean_prediction_improvement": self.mean_prediction_improvement,
            "information_per_cost": self.information_per_cost(),
            "determination": self.determination.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Strategy:
        parameters = {
            name: ParameterSpec(**spec) for name, spec in (payload.get("parameters") or {}).items()
        }
        return cls(
            strategy_id=payload["strategy_id"],
            description=payload.get("description", ""),
            applicable_families=tuple(payload.get("applicable_families") or ()),
            experiment_kinds=tuple(payload.get("experiment_kinds") or ()),
            parameters=parameters,
            enabled=bool(payload.get("enabled", True)),
            applications=int(payload.get("applications", 0)),
            successes=int(payload.get("successes", 0)),
            failures=int(payload.get("failures", 0)),
            total_cost=float(payload.get("total_cost", 0.0)),
            total_information_gain=float(payload.get("total_information_gain", 0.0)),
            total_prediction_improvement=float(payload.get("total_prediction_improvement", 0.0)),
        )


#: Applications needed before a success rate is treated as informative.
MIN_APPLICATIONS_FOR_CONFIDENCE = 5

#: Optimistic prior for an untried strategy, so exploration is not starved by a
#: registry that only ever reinforces whatever happened to work first.
UNTRIED_PRIOR_SCORE = 0.6


def default_strategies() -> list[Strategy]:
    """The strategies the engine ships with."""
    return [
        Strategy(
            strategy_id="short_screen_before_production",
            description=(
                "Run a short NPT equilibration and check density stability before committing "
                "to a long production run."
            ),
            experiment_kinds=("gromacs_equilibrate",),
            parameters={
                "screen_ns": ParameterSpec("screen_ns", 2.0, 0.5, 20.0, "Length of the screening NPT run"),
                "density_tolerance": ParameterSpec(
                    "density_tolerance", 0.02, 0.005, 0.10, "Relative density drift accepted before promotion"
                ),
            },
        ),
        Strategy(
            strategy_id="three_replica_baseline",
            description="Run three independent replicas before quoting any ensemble average.",
            experiment_kinds=("gromacs_equilibrate", "analyze_replicas"),
            parameters={
                "replicas": ParameterSpec("replicas", 3.0, 2.0, 16.0, "Number of independent replicas"),
            },
        ),
        Strategy(
            strategy_id="umbrella_after_contact_evidence",
            description=(
                "Only commit to umbrella sampling once equilibrium MD shows sustained interchain "
                "contact, so the reaction coordinate is known to be physically relevant."
            ),
            experiment_kinds=("umbrella_plan", "umbrella_run"),
            parameters={
                "min_contact_fraction": ParameterSpec(
                    "min_contact_fraction", 0.3, 0.05, 0.9, "Fraction of frames in contact before umbrella work"
                ),
            },
        ),
        Strategy(
            strategy_id="descriptor_baseline_first",
            description="Establish an interpretable descriptor baseline before spending compute on MD.",
            experiment_kinds=("baseline_qspr",),
            parameters={
                "min_training_polymers": ParameterSpec(
                    "min_training_polymers", 20.0, 10.0, 500.0, "Records required before fitting"
                ),
            },
        ),
        Strategy(
            strategy_id="uncertainty_first_active_learning",
            description="Select the next candidates by ensemble uncertainty, with a diversity constraint.",
            experiment_kinds=("select_candidates",),
            parameters={
                "batch_size": ParameterSpec("batch_size", 5.0, 1.0, 50.0, "Candidates per round"),
                "novelty_weight": ParameterSpec("novelty_weight", 0.3, 0.0, 2.0, "Weight on novelty"),
            },
        ),
    ]


class StrategyRegistry:
    """Persistent registry of strategies and their track records."""

    def __init__(self, strategies: Iterable[Strategy] | None = None) -> None:
        self._strategies: dict[str, Strategy] = {}
        for strategy in strategies if strategies is not None else default_strategies():
            self.register(strategy)

    def register(self, strategy: Strategy) -> Strategy:
        self._strategies[strategy.strategy_id] = strategy
        return strategy

    def get(self, strategy_id: str) -> Strategy:
        try:
            return self._strategies[strategy_id]
        except KeyError:
            raise PolymerEngineError("Unknown strategy", strategy_id=strategy_id) from None

    def __contains__(self, strategy_id: object) -> bool:
        return strategy_id in self._strategies

    def __len__(self) -> int:
        return len(self._strategies)

    def all(self) -> list[Strategy]:
        return list(self._strategies.values())

    def applicable(self, *, family: str | None = None, experiment_kind: str | None = None) -> list[Strategy]:
        out = []
        for strategy in self._strategies.values():
            if not strategy.enabled:
                continue
            if not strategy.applies_to(family):
                continue
            if experiment_kind and strategy.experiment_kinds and experiment_kind not in strategy.experiment_kinds:
                continue
            out.append(strategy)
        return out

    # -- learning ------------------------------------------------------
    def record_outcome(self, outcome: StrategyOutcome) -> Strategy:
        strategy = self.get(outcome.strategy_id)
        strategy.applications += 1
        if outcome.succeeded:
            strategy.successes += 1
        else:
            strategy.failures += 1
        if outcome.cost is not None:
            strategy.total_cost += outcome.cost
        if outcome.information_gain is not None:
            strategy.total_information_gain += outcome.information_gain
        if outcome.prediction_improvement is not None:
            strategy.total_prediction_improvement += outcome.prediction_improvement
        logger.info(
            "Recorded %s outcome for %s (%d applications, success rate %s)",
            "successful" if outcome.succeeded else "failed",
            strategy.strategy_id,
            strategy.applications,
            "unknown" if strategy.success_rate is None else f"{strategy.success_rate:.2f}",
        )
        return strategy

    def score(self, strategy: Strategy) -> float:
        """Rank a strategy from its record, shrunk toward the prior when thin.

        A strategy with two successes is not obviously better than one with fifty at
        90%; the shrinkage makes that explicit instead of letting small samples win.
        """
        rate = strategy.success_rate
        if rate is None:
            return UNTRIED_PRIOR_SCORE
        weight = strategy.applications / (strategy.applications + MIN_APPLICATIONS_FOR_CONFIDENCE)
        shrunk = weight * rate + (1.0 - weight) * UNTRIED_PRIOR_SCORE
        efficiency = strategy.information_per_cost()
        bonus = 0.0 if efficiency is None else 0.1 * math.tanh(efficiency)
        return float(shrunk + bonus)

    def ranked(self, *, family: str | None = None, experiment_kind: str | None = None) -> list[tuple[Strategy, float]]:
        candidates = self.applicable(family=family, experiment_kind=experiment_kind)
        return sorted(
            ((s, self.score(s)) for s in candidates), key=lambda pair: (-pair[1], pair[0].strategy_id)
        )

    def best(self, *, family: str | None = None, experiment_kind: str | None = None) -> Strategy | None:
        ranked = self.ranked(family=family, experiment_kind=experiment_kind)
        return ranked[0][0] if ranked else None

    def adapt_parameters(
        self, strategy_id: str, proposals: dict[str, float], *, actor: str = "engine"
    ) -> dict[str, Any]:
        """Adjust parameters within their declared bounds.

        A proposal outside the bounds is **rejected**, not clamped silently: the caller
        asked for something its author ruled out, and that should be visible.
        """
        strategy = self.get(strategy_id)
        applied: dict[str, float] = {}
        rejected: dict[str, str] = {}
        for name, proposed in proposals.items():
            spec = strategy.parameters.get(name)
            if spec is None:
                rejected[name] = "not a declared parameter of this strategy"
                continue
            if not math.isfinite(proposed):
                rejected[name] = "proposed value is not finite"
                continue
            if not spec.within_bounds(proposed):
                rejected[name] = (
                    f"proposed {proposed} is outside the declared bounds "
                    f"[{spec.minimum}, {spec.maximum}]"
                )
                continue
            previous = spec.value
            spec.value = proposed
            applied[name] = proposed
            logger.info("Adapted %s.%s: %s -> %s (%s)", strategy_id, name, previous, proposed, actor)
        return {"strategy_id": strategy_id, "actor": actor, "applied": applied, "rejected": rejected}

    # -- persistence ---------------------------------------------------
    def save(self, store: Any) -> None:
        for strategy in self._strategies.values():
            store.save_strategy(strategy.strategy_id, strategy.as_dict())

    @classmethod
    def load(cls, store: Any, *, seed_defaults: bool = True) -> StrategyRegistry:
        payloads = store.list_strategies()
        if not payloads:
            registry = cls() if seed_defaults else cls([])
            registry.save(store)
            return registry
        return cls(Strategy.from_dict(p) for p in payloads)


__all__ = [
    "MIN_APPLICATIONS_FOR_CONFIDENCE",
    "UNTRIED_PRIOR_SCORE",
    "ParameterSpec",
    "Strategy",
    "StrategyOutcome",
    "StrategyRegistry",
    "default_strategies",
]
