"""The handler-driven build flow, end to end against the wizard fixture.

The live Polymer Builder has no monomer form control: units are picked from a hover
menu wired to ``set_monomer``, the repeat count lives in ``subtext[1]``, and the wizard
advances -- creating a real job -- on one button. These tests drive that whole flow in a
real browser against a fixture with the same structure, including the hidden skeleton
clone that makes every menu entry appear twice.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import importlib.util
import json
import socket
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from polymer_engine.browser.driver import WorkerDriver
from polymer_engine.browser.queue import BuildQueue

pytestmark = pytest.mark.skipif(not WorkerDriver.available(), reason="no .browserenv")

FIXTURES = Path(__file__).parent / "fixtures"


def _live():
    if "charmm_gui_live" in sys.modules:
        return sys.modules["charmm_gui_live"]
    spec = importlib.util.spec_from_file_location(
        "charmm_gui_live",
        Path(__file__).resolve().parents[2] / "scripts" / "charmm_gui_live.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["charmm_gui_live"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def site() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(FIXTURES))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()


class _Record:
    def __init__(self) -> None:
        self.state = _State()


class _State:
    proven_live = False


class _Records(dict):
    def __missing__(self, key):
        self[key] = _Record()
        return self[key]


class _Registry:
    """Faithful enough for the build flow: it exposes .records and marks proven_live,
    so the dependency-ordered verification helper runs to completion instead of being
    swallowed by the run's best-effort guard."""

    def __init__(self) -> None:
        self.verified: list[str] = []
        self.records = _Records()

    def _record(self, name):
        return self.records.setdefault(name, _Record())

    def verify(self, name, **kwargs):
        self.verified.append(name)
        self._record(name).state.proven_live = True

    def save(self) -> None:
        pass


@pytest.fixture
def run(site, monkeypatch):
    """Run the build subcommand against the fixture, capturing queue and registry."""
    monkeypatch.setenv("CHARMM_GUI_EMAIL", "a@b.c")
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", "correct-horse")
    live = _live()
    base = site
    original_session = live.Session

    class Local(original_session):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["base_url"] = base
            super().__init__(*args, **kwargs)

        def login(self, *args, **kwargs):
            return super().login(url=f"{base}/login.html", settle_s=0.3)

    queue_dir = Path(tempfile.mkdtemp())
    registry = _Registry()
    original_preflight = live._preflight
    monkeypatch.setattr(live, "Session", Local)
    monkeypatch.setattr(live, "BuildQueue",
                        lambda *a, **k: BuildQueue(queue_dir / "q.json"))
    monkeypatch.setattr(
        live, "_preflight",
        lambda **k: (registry, 0, original_preflight(allow_prompt=False)[2]))

    def _run(**overrides):
        namespace = argparse.Namespace(
            url=f"{base}/builder_wizard.html", headed=False, polymer_id="t",
            rationale="fixture", dp=12, system_type="single", variant=None,
            dry_run=False, monomer="Polylactic acid")
        for key, value in overrides.items():
            setattr(namespace, key, value)
        return live.build(namespace), queue_dir / "q.json", registry

    return _run


def test_the_full_flow_captures_a_real_job_id(run) -> None:
    code, queue_file, registry = run(variant="atactic")
    assert code == 0
    entry = json.loads(queue_file.read_text())["entries"][0]
    assert entry["job_id"] == "5557799"
    assert entry["state"] == "SUBMITTED"
    assert "SINGLE_CHAIN_BUILD_VERIFIED" in registry.verified


def test_dry_run_verifies_everything_and_clicks_nothing(run) -> None:
    code, queue_file, registry = run(variant="atactic", dry_run=True)
    assert code == 0
    # No submission happened: the queue entry, if any, carries no job id.
    if queue_file.exists():
        entries = json.loads(queue_file.read_text())["entries"]
        assert all(e["job_id"] is None for e in entries)
    assert "SINGLE_CHAIN_BUILD_VERIFIED" not in registry.verified


def test_a_class_shape_monomer_resolves_by_its_own_name(run) -> None:
    code, queue_file, _registry = run(monomer="Polyethylene terephthalate")
    assert code == 0
    entry = json.loads(queue_file.read_text())["entries"][0]
    assert entry["spec"]["monomer"] == "Polyethylene terephthalate"


def test_an_absent_monomer_is_refused_before_any_click(run) -> None:
    code, queue_file, registry = run(monomer="Polybogusene")
    assert code == 5
    assert not queue_file.exists()
    assert registry.verified == []


def test_the_skeleton_duplicate_does_not_make_resolution_ambiguous(run) -> None:
    """Every menu entry exists twice (live + hidden clone); one choice, not two."""
    code, _qf, _reg = run(variant="isotactic (R)")
    assert code == 0


def test_resolution_logic_refuses_genuine_ambiguity() -> None:
    live = _live()
    choices = {"groups": [
        {"group": "A", "options": [{"index": 0, "text": "Polyfoo"}]},
        {"group": "B", "options": [{"index": 1, "text": "Polyfoo"}]},
    ]}
    index, _g, reason = live._resolve_handler_option(choices, "Polyfoo", None)
    assert index is None
    assert "refusing to choose" in reason


def test_a_hidden_only_match_fails_fast_with_a_diagnosis(site) -> None:
    """The live failure, pinned: one element matched and it was the hidden template.

    The worker used to accept a single match without checking visibility and then hang
    for the full Playwright timeout trying to click something display:none. It must
    refuse immediately and say that what matched was a template clone.
    """
    import time

    from polymer_engine.browser.credentials import Credentials
    from polymer_engine.browser.session import Session
    from polymer_engine.core.config import Secret

    with Session(credentials=Credentials(email="a", password=Secret("b")),
                 base_url=site) as session:
        session.navigate(f"{site}/builder_wizard.html")
        started = time.time()
        result = session.driver.send(
            "type", key="dp", text="12", delay_ms=1,
            locators=[{"strategy": "css",
                       "value": "#skel input[name='subtext[1]']"}])
        elapsed = time.time() - started
        assert not result.get("ok")
        assert elapsed < 5.0, f"took {elapsed:.1f}s: the hidden element was accepted"
        assert "hidden template" in str(result.get("error"))
