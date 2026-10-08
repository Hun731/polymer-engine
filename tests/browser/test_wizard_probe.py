"""Mapping the Polymer Builder wizard, and refusing to run its simulations.

The tutorial (charmm-gui.org/download/polymer_builder_tutorial.pdf) documents two very
different paths from the same first click:

* single chain -- one step, then download.tgz is offered directly;
* solution / melt -- several steps, one of which ("Next Step: Generate Equilibrium")
  launches a coarse-grained OpenMM run on CHARMM-GUI's own servers.

The probe walks the wizard and records each step from evidence. It stops at the first
download, or at the first step whose only way forward crosses a compute boundary, and it
never clicks such a button -- spending the shared service's compute is a person's call.
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

pytestmark = pytest.mark.skipif(not WorkerDriver.available(), reason="no .browserenv")

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeRecord:
    def __init__(self) -> None:
        self.state = type("S", (), {"proven_live": False})()


class _FakeRecords(dict):
    def __missing__(self, key):
        self[key] = _FakeRecord()
        return self[key]


class _FakeRegistry:
    """Faithful to the real registry's surface: records with proven_live, so the
    dependency-ordered verification helper runs rather than being swallowed."""

    def __init__(self) -> None:
        self.verified: list[str] = []
        self.records = _FakeRecords()

    def verify(self, name, **kwargs):
        self.verified.append(name)
        self.records[name].state.proven_live = True

    def save(self) -> None:
        pass


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


@pytest.fixture
def probe(site, monkeypatch):
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

    tmp = Path(tempfile.mkdtemp())

    registry = _FakeRegistry()
    original_preflight = live._preflight
    monkeypatch.setattr(live, "Session", Local)
    monkeypatch.setattr(live, "ACQUISITION_ROOT", tmp)
    monkeypatch.setattr(
        live, "_preflight",
        lambda **k: (registry, 0, original_preflight(allow_prompt=False)[2]))

    def _run(system_type, to_generation=False):
        code = live.wizard_probe(argparse.Namespace(
            url=f"{base}/wizard_steps.html", headed=False,
            monomer="Polylactic acid", variant="atactic", dp=10,
            system_type=system_type, max_steps=8, advance=True,
            to_generation=to_generation, step_settle_ms=400))
        wizard_map = json.loads((tmp / "wizard_map" / "wizard_map.json").read_text())
        return code, wizard_map, registry

    return _run


def test_single_chain_download_with_no_further_step_is_terminal(probe) -> None:
    """The minimal single-chain fixture offers download.tgz and no Next Step: that is a
    terminal download, reached in one step."""
    code, wizard_map, _registry = probe("single")
    assert code == 0
    assert wizard_map["n_steps_seen"] == 1
    step = wizard_map["steps"][0]
    assert step["download_available"] is True
    assert step["next_steps"] == []
    assert step.get("is_terminal_download") is True


def test_melt_maps_its_steps_and_stops_at_the_compute_boundary(probe) -> None:
    code, wizard_map, registry = probe("melt", to_generation=False)
    assert code == 0
    # It advanced through the safe solvation step and stopped at equilibration.
    assert wizard_map["n_steps_seen"] == 2
    last = wizard_map["steps"][-1]
    assert last["compute_boundaries"], "the equilibration step must be flagged"
    assert "Generate Equilibrium" in last["compute_boundaries"][0]
    # It did NOT reach a terminal download, and did NOT cross the boundary.
    assert not any(s.get("is_terminal_download") for s in wizard_map["steps"])
    assert "SYSTEM_GENERATION_VERIFIED" not in registry.verified


class TestBoundaryClassification:
    """The list that decides what a probe will never click."""

    def test_compute_launching_buttons_are_recognised(self) -> None:
        live = _live()
        for text in ("Next Step: Generate Equilibrium",
                     "Next Step: Replace into All-atom",
                     "Next Step: Input Generation"):
            assert live._boundary_hit(text) is not None, text

    def test_safe_navigation_buttons_are_not_boundaries(self) -> None:
        live = _live()
        for text in ("Next Step: Build Polymer Chains",
                     "Next Step: Solvate", "Next Step: Determine System Size",
                     "show system size info"):
            assert live._boundary_hit(text) is None, text


class TestWalkToTerminal:
    """Reaching the final input-generation download, not stopping at the built chain.

    A single chain offers download.tgz at step 1 -- but that is the built structure, not
    the production-ready GROMACS inputs. Those come after force-field assignment and
    input generation, several steps on, past an equilibration that runs on CHARMM-GUI's
    servers. The intermediate download is noted and passed; the terminal one (no 'Next
    Step' remaining) is the goal.
    """

    @pytest.fixture
    def walk(self, site, monkeypatch):
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

        tmp = Path(tempfile.mkdtemp())
        monkeypatch.setattr(live, "Session", Local)
        monkeypatch.setattr(live, "ACQUISITION_ROOT", tmp)
        original_preflight = live._preflight
        monkeypatch.setattr(
            live, "_preflight",
            lambda **k: (_FakeRegistry(), 0, original_preflight(allow_prompt=False)[2]))

        def _run(to_generation):
            live.wizard_probe(argparse.Namespace(
                url=f"{base}/wizard_full.html", headed=False,
                monomer="Polylactic acid", variant="atactic", dp=10,
                system_type="single", max_steps=8, advance=True,
                to_generation=to_generation, step_settle_ms=400))
            return json.loads((tmp / "wizard_map" / "wizard_map.json").read_text())

        return _run

    def test_unauthorized_walk_stops_at_the_first_generation_step(self, walk) -> None:
        wizard_map = walk(to_generation=False)
        assert not any(s.get("is_terminal_download") for s in wizard_map["steps"])
        assert wizard_map["steps"][-1]["compute_boundaries"]

    def test_authorized_walk_reaches_the_terminal_input_generation_download(
        self, walk
    ) -> None:
        wizard_map = walk(to_generation=True)
        terminal = [s for s in wizard_map["steps"] if s.get("is_terminal_download")]
        assert len(terminal) == 1, "exactly one terminal download must be found"
        assert terminal[0]["download_available"] is True
        assert terminal[0]["next_steps"] == [], "terminal means no way forward"
        # The intermediate built-chain download was seen but not treated as terminal.
        intermediate = [s for s in wizard_map["steps"]
                        if s["download_available"] and not s.get("is_terminal_download")]
        assert intermediate, "the step-1 built-chain download should be recorded"
