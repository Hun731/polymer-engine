"""Choosing which polymers to spend validation effort on.

Validating the easiest member of a family and declaring the family covered is the
failure this module exists to prevent. Polyethylene is trivial for any force field;
learning that it works tells you almost nothing about poly(ethylene terephthalate).

Selection therefore rewards the *awkward* members: unusual chemistry relative to the
rest of the family, high analogy penalties where they are known, and descriptor distance
from everything already validated. Cost is a tie-breaker, never a driver -- picking the
cheap case first is exactly the bias being corrected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from polymer_engine.core.logging import get_logger

logger = get_logger("parameterization.representatives")

#: Descriptors used to measure how different two polymers are.  Chosen because each is
#: cheap, defined for every repeat unit, and captures a different axis of variation.
DESCRIPTOR_KEYS: tuple[str, ...] = (
    "heavy_atom_count", "heteroatom_count", "rotatable_bond_count",
    "aromatic_ring_count", "hbond_donor_count", "hbond_acceptor_count",
    "fraction_csp3", "tpsa",
)


@dataclass
class Candidate:
    """One polymer as seen by the selector."""

    polymer_id: str
    name: str
    family: str
    descriptors: dict[str, float] = field(default_factory=dict)
    functional_groups: list[str] = field(default_factory=list)
    max_penalty: float | None = None
    already_validated: bool = False

    def vector(self) -> list[float]:
        return [float(self.descriptors.get(key, 0.0)) for key in DESCRIPTOR_KEYS]


@dataclass
class Selection:
    """A chosen representative and the reason it was chosen."""

    polymer_id: str
    name: str
    family: str
    score: float
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"polymer_id": self.polymer_id, "name": self.name,
                "family": self.family, "score": round(self.score, 4),
                "reasons": list(self.reasons)}


def _standardise(candidates: list[Candidate]) -> dict[str, tuple[float, float]]:
    """Mean and spread per descriptor, so axes with big units do not dominate."""
    stats: dict[str, tuple[float, float]] = {}
    for index, key in enumerate(DESCRIPTOR_KEYS):
        values = [c.vector()[index] for c in candidates]
        mean = sum(values) / len(values) if values else 0.0
        variance = (sum((v - mean) ** 2 for v in values) / len(values)) if values else 0.0
        stats[key] = (mean, math.sqrt(variance) or 1.0)
    return stats


def _distance(a: Candidate, b: Candidate, stats: dict[str, tuple[float, float]]) -> float:
    total = 0.0
    for index, key in enumerate(DESCRIPTOR_KEYS):
        _mean, spread = stats[key]
        total += ((a.vector()[index] - b.vector()[index]) / spread) ** 2
    return math.sqrt(total)


def select_representatives(
    candidates: list[Candidate], *, per_family: int = 2
) -> list[Selection]:
    """Pick representatives that between them stress the parameterization.

    Within each family the first pick is the member furthest from the family centroid --
    the most chemically unusual one -- and subsequent picks maximise distance from what
    has already been chosen, so a second representative is not a near-duplicate of the
    first.
    """
    if not candidates:
        return []
    stats = _standardise(candidates)
    by_family: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_family.setdefault(candidate.family, []).append(candidate)

    selections: list[Selection] = []
    for family, members in sorted(by_family.items()):
        pool = [m for m in members if not m.already_validated]
        chosen: list[Candidate] = [m for m in members if m.already_validated]
        for _ in range(min(per_family, len(pool))):
            best, best_score, best_reasons = None, -1.0, []
            for candidate in pool:
                score, reasons = 0.0, []

                if chosen:
                    gap = min(_distance(candidate, other, stats) for other in chosen)
                    score += gap
                    reasons.append(f"descriptor distance {gap:.2f} from those already chosen")
                else:
                    centroid = sum(
                        _distance(candidate, other, stats) for other in members
                    ) / max(1, len(members))
                    score += centroid
                    reasons.append(
                        f"most chemically unusual in {family} (mean distance {centroid:.2f})"
                    )

                if candidate.max_penalty:
                    # A weak analogy is precisely what needs checking.
                    contribution = min(candidate.max_penalty / 25.0, 3.0)
                    score += contribution
                    reasons.append(f"analogy penalty {candidate.max_penalty:g}")

                exotic = [g for g in candidate.functional_groups
                          if g in {"ester", "amide", "nitrile", "C-F", "C-Cl"}]
                if exotic:
                    score += 0.5 * len(exotic)
                    reasons.append("difficult groups: " + ", ".join(exotic))

                if score > best_score:
                    best, best_score, best_reasons = candidate, score, reasons
            if best is None:
                break
            selections.append(Selection(polymer_id=best.polymer_id, name=best.name,
                                        family=family, score=best_score,
                                        reasons=best_reasons))
            chosen.append(best)
            pool.remove(best)
    logger.info("Selected %d representative(s) across %d families",
                len(selections), len(by_family))
    return selections


__all__ = ["DESCRIPTOR_KEYS", "Candidate", "Selection", "select_representatives"]
