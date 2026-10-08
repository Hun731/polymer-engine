"""Shared test fixtures.

Every fixture here is offline.  ``no_network`` is autouse: if any test constructs a
real transport and tries to open a socket, it fails loudly instead of silently
depending on the internet.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from polymer_engine.core.config import EngineConfig, load_config
from polymer_engine.providers.http import HttpClient, ResponseCache
from polymer_engine.providers.testing import FixtureTransport, RecordingSleeper

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make real socket connections impossible for the duration of a test."""

    def _blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "This test attempted a real network connection. "
            "Use FixtureTransport or a recorded fixture instead."
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture
def sleeper() -> RecordingSleeper:
    """A sleep replacement that records durations instead of waiting."""
    return RecordingSleeper()


@pytest.fixture
def transport() -> FixtureTransport:
    return FixtureTransport()


@pytest.fixture
def client(transport: FixtureTransport, sleeper: RecordingSleeper) -> HttpClient:
    """An HttpClient with no delays, no cache, and deterministic backoff."""
    return HttpClient(
        transport=transport,
        max_retries=2,
        backoff_base_s=1.0,
        sleeper=sleeper,
        jitter=lambda: 1.0,
    )


@pytest.fixture
def caching_client(tmp_path: Path, transport: FixtureTransport, sleeper: RecordingSleeper) -> HttpClient:
    return HttpClient(
        transport=transport,
        cache=ResponseCache(tmp_path / "cache", ttl_s=3600, clock=lambda: 1000.0),
        sleeper=sleeper,
        jitter=lambda: 1.0,
    )


@pytest.fixture
def config(tmp_path: Path) -> EngineConfig:
    """A fully isolated configuration rooted in tmp_path."""
    cfg = load_config(
        discover=False,
        use_env=False,
        overrides={"paths": {"root": str(tmp_path)}, "http": {"cache_enabled": False}},
    )
    cfg.paths.ensure()
    return cfg


@pytest.fixture
def fixture_root() -> Path:
    return FIXTURE_ROOT


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_gromacs: needs a real gmx executable")
    config.addinivalue_line("markers", "requires_orca: needs a real orca executable")
    config.addinivalue_line("markers", "requires_plumed: needs a real plumed executable")
    config.addinivalue_line("markers", "requires_rdkit: needs RDKit installed")
    config.addinivalue_line("markers", "requires_mdanalysis: needs MDAnalysis installed")
    config.addinivalue_line("markers", "requires_sklearn: needs scikit-learn installed")
    config.addinivalue_line("markers", "slow: takes more than a second")
