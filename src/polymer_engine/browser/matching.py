"""Resolving a requested polymer to a catalogue entry, exactly or not at all.

This module is short on purpose, and the shortness is the point. The only thing it will
do is match a request to a monomer the catalogue actually lists, by form value or by
normalised label. What it will *not* do:

* pick the closest name;
* pick a chemically similar monomer;
* pick the first of several matches;
* fall back to a related family.

Every one of those would be a silent substitution, and a silent substitution means the
engine simulates one polymer and reports it as another. When the answer is not exactly
one entry, the result is ``MONOMER_NOT_FOUND`` or ``REQUIRES_HUMAN_REVIEW`` and the
workflow stops.

Similarity *is* computed -- but only to put candidate names in front of a person in the
diagnostic, never to choose. A suggestion a human confirms is a decision; a suggestion
the engine acts on is a substitution.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from polymer_engine.browser.catalog import Catalog, MonomerEntry
from polymer_engine.browser.states import BrowserState
from polymer_engine.core.logging import get_logger

logger = get_logger("browser.matching")

#: Only for ordering the suggestions shown to a person. Never a threshold above which
#: something is accepted -- there is no such threshold anywhere in this module.
SUGGESTION_CUTOFF = 0.55
MAX_SUGGESTIONS = 5


def normalise(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace.

    This is the *only* latitude allowed. "Poly(lactic acid)" and "poly lactic acid" are
    the same string written two ways; "poly(lactic acid)" and "poly(glycolic acid)" are
    two different molecules, and nothing here will ever conflate them.
    """
    lowered = text.strip().lower()
    without_punctuation = re.sub(r"[(),\[\]{}._/\\-]+", " ", lowered)
    return re.sub(r"\s+", " ", without_punctuation).strip()


@dataclass
class MonomerMatch:
    """The outcome of resolving one requested monomer."""

    state: BrowserState
    requested: str
    entry: MonomerEntry | None = None
    matched_on: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.state.ok and self.entry is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value, "requested": self.requested,
            "resolved": self.resolved, "matched_on": self.matched_on,
            "entry": self.entry.as_dict() if self.entry else None,
            "candidates": list(self.candidates), "reason": self.reason,
        }


def resolve_monomer(
    catalog: Catalog, requested: str, *, aliases: dict[str, str] | None = None,
    control: str | None = None,
) -> MonomerMatch:
    """Find the one catalogue entry that *is* the requested monomer.

    ``aliases`` maps a requested name to a catalogue form value. An alias is a human's
    recorded decision that two names denote the same chemistry, which is exactly the
    judgement this module refuses to make on its own; supplying one is how that decision
    enters the system reviewably.
    """
    if not catalog.monomers:
        return MonomerMatch(
            BrowserState.MONOMER_NOT_FOUND, requested,
            reason="the catalogue is empty; run discovery against a live session first",
        )
    entries = [m for m in catalog.monomers if control is None or m.control == control]
    if not entries:
        return MonomerMatch(
            BrowserState.MONOMER_NOT_FOUND, requested,
            reason=f"the catalogue has no monomers in control {control!r}",
        )

    alias_target = (aliases or {}).get(requested) or (aliases or {}).get(normalise(requested))
    if alias_target:
        exact = [m for m in entries if m.value == alias_target]
        if len(exact) == 1:
            return MonomerMatch(BrowserState.OK, requested, exact[0],
                                matched_on="alias",
                                reason=f"resolved by a recorded alias to form value "
                                       f"{alias_target!r}")
        return MonomerMatch(
            BrowserState.REQUIRES_HUMAN_REVIEW, requested,
            candidates=[m.as_dict() for m in exact],
            reason=(f"the recorded alias {requested!r} -> {alias_target!r} matches "
                    f"{len(exact)} catalogue entries; the alias is stale or ambiguous"),
        )

    by_value = [m for m in entries if m.value == requested]
    if len(by_value) == 1:
        return MonomerMatch(BrowserState.OK, requested, by_value[0], matched_on="value",
                            reason="exact match on the catalogue form value")

    target = normalise(requested)
    by_label = [m for m in entries if normalise(m.label) == target]
    if len(by_label) == 1:
        return MonomerMatch(BrowserState.OK, requested, by_label[0], matched_on="label",
                            reason="exact match on the normalised catalogue label")
    if len(by_label) > 1:
        return MonomerMatch(
            BrowserState.REQUIRES_HUMAN_REVIEW, requested,
            candidates=[m.as_dict() for m in by_label],
            reason=(f"{len(by_label)} catalogue entries share the label {requested!r}; "
                    f"choosing between them is a scientific decision, not a sort order"),
        )
    if len(by_value) > 1:
        return MonomerMatch(
            BrowserState.REQUIRES_HUMAN_REVIEW, requested,
            candidates=[m.as_dict() for m in by_value],
            reason=f"{len(by_value)} catalogue entries share the form value {requested!r}",
        )

    return MonomerMatch(
        BrowserState.MONOMER_NOT_FOUND, requested,
        candidates=suggest(catalog, requested, control=control),
        reason=(
            f"no catalogue entry is exactly {requested!r}. Near names are listed as "
            f"candidates for a person to check; none is selected, because a similar "
            f"name is a different molecule until someone confirms otherwise"
        ),
    )


def suggest(catalog: Catalog, requested: str, *, control: str | None = None) -> list[dict[str, Any]]:
    """Nearest catalogue names, ordered, purely for a human to read."""
    entries = [m for m in catalog.monomers if control is None or m.control == control]
    target = normalise(requested)
    scored = [
        {**entry.as_dict(),
         "similarity": round(difflib.SequenceMatcher(
             None, target, normalise(entry.label)).ratio(), 3)}
        for entry in entries
    ]
    close = [s for s in scored if s["similarity"] >= SUGGESTION_CUTOFF]
    close.sort(key=lambda s: -s["similarity"])
    return close[:MAX_SUGGESTIONS]


def resolve_option(
    available: list[str], requested: str | None, *, what: str
) -> tuple[str | None, str]:
    """Match a requested option (tacticity, system type) against what the page offers.

    Returns ``(value, reason)``; a ``None`` value with a reason means the workflow must
    stop. ``requested is None`` returns ``None`` with an empty reason: not asking for an
    option is different from asking for one that does not exist.
    """
    if requested is None:
        return None, ""
    if not available:
        return None, (f"the page offers no {what} options, so {requested!r} cannot be "
                      f"selected; the form may have changed")
    exact = [option for option in available if option == requested]
    if len(exact) == 1:
        return exact[0], ""
    normalised = [option for option in available if normalise(option) == normalise(requested)]
    if len(normalised) == 1:
        return normalised[0], ""
    if len(normalised) > 1:
        return None, (f"{len(normalised)} {what} options match {requested!r}: "
                      f"{', '.join(normalised)}")
    return None, (f"{what} {requested!r} is not offered; the page lists: "
                  f"{', '.join(available)}")


__all__ = [
    "MAX_SUGGESTIONS", "SUGGESTION_CUTOFF", "MonomerMatch", "normalise",
    "resolve_monomer", "resolve_option", "suggest",
]
