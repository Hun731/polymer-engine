"""Offline transports for tests and reproducible runs.

The core test suite must never touch the network (charter rule 12).  These
transports make that structural rather than aspirational: a test supplies canned
responses, and an unmatched request is a loud failure rather than a live call.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.errors import TransportError
from polymer_engine.providers.http import HttpRequest, HttpResponse


@dataclass
class StubResponse:
    """A canned reply, or an exception to raise instead."""

    status: int = 200
    body: bytes = b"{}"
    headers: dict[str, str] = field(default_factory=dict)
    raises: Exception | None = None

    @classmethod
    def json(cls, payload: Any, status: int = 200, **headers: str) -> StubResponse:
        return cls(status=status, body=json.dumps(payload).encode("utf-8"), headers=dict(headers))

    @classmethod
    def text(cls, body: str, status: int = 200, **headers: str) -> StubResponse:
        return cls(status=status, body=body.encode("utf-8"), headers=dict(headers))

    @classmethod
    def error(cls, exc: Exception) -> StubResponse:
        return cls(raises=exc)


class FixtureTransport:
    """Serve responses matched by substring against the full request URL.

    Rules are checked in insertion order.  Each rule may hold several responses,
    which are returned in sequence (then the last one repeats) -- that is how a
    "fails twice then succeeds" retry test is written.
    """

    def __init__(self, rules: Iterable[tuple[str, StubResponse | list[StubResponse]]] = ()) -> None:
        self._rules: list[tuple[str, list[StubResponse]]] = []
        self._cursor: dict[int, int] = {}
        self.requests: list[HttpRequest] = []
        for pattern, responses in rules:
            self.add(pattern, responses)

    def add(self, pattern: str, responses: StubResponse | list[StubResponse]) -> FixtureTransport:
        items = responses if isinstance(responses, list) else [responses]
        if not items:
            raise ValueError("A rule needs at least one response")
        self._rules.append((pattern, items))
        return self

    def add_json(self, pattern: str, payload: Any, status: int = 200) -> FixtureTransport:
        return self.add(pattern, StubResponse.json(payload, status))

    def send(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        url = request.full_url()
        for index, (pattern, responses) in enumerate(self._rules):
            if pattern in url:
                position = self._cursor.get(index, 0)
                stub = responses[min(position, len(responses) - 1)]
                self._cursor[index] = position + 1
                if stub.raises is not None:
                    raise stub.raises
                return HttpResponse(
                    status=stub.status,
                    body=stub.body,
                    headers=dict(stub.headers),
                    url=url,
                    elapsed_s=0.0,
                    fetched_at="1970-01-01T00:00:00+00:00",
                )
        raise TransportError(
            "No fixture matches this request -- the test would have hit the network",
            url=url,
            method=request.method,
            known_patterns=[p for p, _ in self._rules],
        )

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def urls(self) -> list[str]:
        return [r.full_url() for r in self.requests]


class OfflineTransport:
    """Refuses every request.  Used to prove a code path never needs the network."""

    def send(self, request: HttpRequest) -> HttpResponse:  # pragma: no cover - trivial
        raise TransportError("Network access is disabled in this context", url=request.full_url())


def load_fixture(name: str, *, root: str | Path | None = None) -> Any:
    """Read a recorded JSON fixture from ``tests/fixtures/providers``."""
    base = Path(root) if root else Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "providers"
    path = Path(base) / name
    if not path.exists():
        raise FileNotFoundError(f"Provider fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fake_clock(start: float = 0.0, step: float = 0.0) -> Callable[[], float]:
    """A monotonic clock that advances by ``step`` on each call."""
    state = {"t": start}

    def clock() -> float:
        value = state["t"]
        state["t"] += step
        return value

    return clock


class RecordingSleeper:
    """Captures sleep durations instead of sleeping, so retry tests run instantly."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)

    @property
    def total(self) -> float:
        return sum(self.calls)


__all__ = [
    "FixtureTransport",
    "OfflineTransport",
    "RecordingSleeper",
    "StubResponse",
    "fake_clock",
    "load_fixture",
]
