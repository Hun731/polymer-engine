"""The parameterization state machine, and its persisted history.

Forward progress is strictly sequential: a polymer cannot be QM-validated before it has
been parameterized, and the machine refuses transitions that skip a step rather than
letting a caller assert a state it has not earned. Verdicts (BLOCKED, FAILED,
INCONCLUSIVE, REQUIRES_EXPERT_REVIEW) are reachable from anywhere, because any step can
discover that the answer is "no".

Every transition records why it happened, so a QUALIFIED record can be read backwards to
the evidence that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.errors import PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.parameterization.models import (
    OPTIONAL_QM_SHORTCUT,
    STATE_SEQUENCE,
    TERMINAL_STATES,
    ParameterizationState,
    utc_now,
)

logger = get_logger("parameterization.state")

#: Verdicts, reachable from any forward state.
VERDICTS: frozenset[ParameterizationState] = frozenset({
    ParameterizationState.BLOCKED,
    ParameterizationState.FAILED,
    ParameterizationState.INCONCLUSIVE,
    ParameterizationState.REQUIRES_EXPERT_REVIEW,
})


def allowed_transitions(state: ParameterizationState) -> set[ParameterizationState]:
    """Where a state may go next: one step forward, or any verdict."""
    if state in TERMINAL_STATES and state not in VERDICTS:
        return set()                       # QUALIFIED is an end state
    moves: set[ParameterizationState] = set(VERDICTS)
    if state in STATE_SEQUENCE:
        index = STATE_SEQUENCE.index(state)
        if index + 1 < len(STATE_SEQUENCE):
            moves.add(STATE_SEQUENCE[index + 1])
        # Skipping QM is allowed only where the property class does not require it.
        # Recording QM_VALIDATED without running QM would be the false claim.
        if state is OPTIONAL_QM_SHORTCUT[0]:
            moves.add(OPTIONAL_QM_SHORTCUT[1])
    else:
        # A verdict may be revisited once the blocking condition is addressed.
        moves.add(ParameterizationState.DISCOVERED)
    return moves


@dataclass
class Transition:
    from_state: ParameterizationState
    to_state: ParameterizationState
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=utc_now)

    def as_dict(self) -> dict[str, Any]:
        return {"from": self.from_state.value, "to": self.to_state.value,
                "reason": self.reason, "evidence": dict(self.evidence), "at": self.at}


@dataclass
class ParameterizationTrack:
    """One polymer/backend pair's journey, with every step retained."""

    polymer_id: str
    backend: str
    state: ParameterizationState = ParameterizationState.DISCOVERED
    history: list[Transition] = field(default_factory=list)

    def advance(
        self, to_state: ParameterizationState, reason: str,
        evidence: dict[str, Any] | None = None,
    ) -> Transition:
        """Move to ``to_state``, or raise if that would skip a step."""
        if to_state not in allowed_transitions(self.state):
            raise PolymerEngineError(
                "Illegal parameterization transition",
                polymer_id=self.polymer_id, backend=self.backend,
                from_state=self.state.value, to_state=to_state.value,
                allowed=sorted(s.value for s in allowed_transitions(self.state)),
            )
        transition = Transition(from_state=self.state, to_state=to_state,
                                reason=reason, evidence=evidence or {})
        self.history.append(transition)
        self.state = to_state
        logger.debug("%s/%s: %s -> %s (%s)", self.polymer_id, self.backend,
                     transition.from_state.value, to_state.value, reason)
        return transition

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id, "backend": self.backend,
            "state": self.state.value, "terminal": self.terminal,
            "history": [t.as_dict() for t in self.history],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ParameterizationTrack:
        track = cls(polymer_id=payload["polymer_id"], backend=payload["backend"],
                    state=ParameterizationState(payload["state"]))
        for raw in payload.get("history", []):
            track.history.append(Transition(
                from_state=ParameterizationState(raw["from"]),
                to_state=ParameterizationState(raw["to"]),
                reason=raw["reason"], evidence=raw.get("evidence", {}),
                at=raw.get("at", utc_now()),
            ))
        return track


class TrackStore:
    """Durable set of tracks, keyed by polymer and backend."""

    def __init__(self) -> None:
        self._tracks: dict[tuple[str, str], ParameterizationTrack] = {}

    def track(self, polymer_id: str, backend: str) -> ParameterizationTrack:
        key = (polymer_id, backend)
        if key not in self._tracks:
            self._tracks[key] = ParameterizationTrack(polymer_id=polymer_id,
                                                      backend=backend)
        return self._tracks[key]

    def all(self) -> list[ParameterizationTrack]:
        return [self._tracks[k] for k in sorted(self._tracks)]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"generated_at": utc_now(),
                        "tracks": [t.as_dict() for t in self.all()]}, indent=1),
            encoding="utf-8",
        )
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> TrackStore:
        store = cls()
        path = Path(path)
        if not path.is_file():
            return store
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw in payload.get("tracks", []):
            track = ParameterizationTrack.from_dict(raw)
            store._tracks[(track.polymer_id, track.backend)] = track
        return store


__all__ = [
    "VERDICTS", "ParameterizationTrack", "TrackStore", "Transition",
    "allowed_transitions",
]
