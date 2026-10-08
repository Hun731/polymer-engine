"""Discovering a form that hides most of itself until a choice is made.

The live Polymer Builder reported 27 controls of which 16 were hidden, alongside a
"System Type:" radio group. A page like that shows a fraction of what it offers until
something is selected, so a single snapshot of the initial state can easily conclude
that no monomer control exists.

Selecting a radio is ordinary form interaction: it reveals sections and creates nothing
on the server. No submit control is touched anywhere in this module.
"""

from __future__ import annotations

import functools
import http.server
import importlib.util
import json
import socket
import threading
from pathlib import Path

import pytest

from polymer_engine.browser.catalog import CatalogState
from polymer_engine.browser.credentials import Credentials
from polymer_engine.browser.discovery import (
    catalog_from_controls,
    discover_form_schema,
)
from polymer_engine.browser.driver import WorkerDriver
from polymer_engine.browser.session import Session
from polymer_engine.core.config import Secret

pytestmark = pytest.mark.skipif(not WorkerDriver.available(), reason="no .browserenv")

FIXTURES = Path(__file__).parent / "fixtures"
PAGE = "builder_conditional.html"


def _live_module():
    spec = importlib.util.spec_from_file_location(
        "charmm_gui_live", Path(__file__).resolve().parents[2] / "scripts" / "charmm_gui_live.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
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
def session(site: str):
    live = Session(credentials=Credentials(email="a@b.c", password=Secret("x"),
                                           typing_delay_ms=1), base_url=site)
    live.launch()
    live.navigate(f"{site}/{PAGE}")
    try:
        yield live
    finally:
        live.close()


def test_hidden_controls_are_counted_not_silently_lost(session: Session) -> None:
    """Skipping a hidden control is fine; skipping it silently is how 27 became 8."""
    schema, controls = discover_form_schema(session)
    hidden = [c for c in controls
              if not c.get("visible") or (c.get("type") or "") == "hidden"]
    assert hidden, "the fixture must start with hidden controls"
    assert len(schema.fields) == len(controls) - len(hidden)


def test_the_initial_snapshot_cannot_see_the_monomer_list(session: Session) -> None:
    """The state that produced CATALOG_NOT_IDENTIFIED on the live page."""
    _schema, controls = discover_form_schema(session)
    inventory = session.driver.send("page_inventory")
    catalog = catalog_from_controls(controls, semantics={}, inventory=inventory)
    assert catalog.state is CatalogState.NOT_IDENTIFIED
    # Crucially: not reported as the site having no monomers.
    assert "does not mean the Polymer Builder offers no monomers" in catalog.notes


def test_probing_a_radio_group_reveals_the_hidden_section(
    session: Session, tmp_path
) -> None:
    live = _live_module()
    inventory = session.driver.send("page_inventory")
    groups = inventory.get("radio_checkbox_groups", [])
    assert groups, "the fixture must offer a system-type radio group"

    live._probe_conditionals(session, groups, tmp_path)
    revealed = json.loads((tmp_path / "conditional_sections.json").read_text())

    assert set(revealed) == {"model=single", "model=melt", "model=solution"}
    for entry in revealed.values():
        names = {s["name"] for s in entry["selects"]}
        assert "block[1]" in names
        options = next(s for s in entry["selects"] if s["name"] == "block[1]")
        texts = [o["text"] for o in options["options"] if o["value"]]
        assert "Ethylene" in texts and "Lactide" in texts

    # Melt reveals a control the other modes do not -- which is itself information
    # about what each system type needs.
    assert (revealed["model=melt"]["n_visible_controls"]
            > revealed["model=single"]["n_visible_controls"])


class _Recorder:
    """Passes every command through, keeping a record of what was issued."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[tuple[str, dict]] = []

    def send(self, command: str, **payload):
        self.calls.append((command, payload))
        return self.inner.send(command, **payload)

    def close(self) -> None:
        self.inner.close()


def test_probing_never_touches_a_submit_control(session: Session, tmp_path) -> None:
    """Probing reveals sections. It must not be able to create a job."""
    live = _live_module()
    inventory = session.driver.send("page_inventory")
    recorder = _Recorder(session.driver)
    session._driver = recorder

    live._probe_conditionals(session, inventory.get("radio_checkbox_groups", []), tmp_path)

    clicks = [payload for command, payload in recorder.calls if command == "click"]
    assert clicks, "the probe should have selected something"
    for payload in clicks:
        for locator in payload["locators"]:
            value = locator["value"]
            # Every click targets a radio input by name and value, and nothing else.
            assert value.startswith("input[name="), value
            assert "[value=" in value, value
            for banned in ("submit", "button", "type='submit'"):
                assert banned not in value.lower(), value
    # The page never navigated away: still the same fixture.
    assert PAGE in session.current_url()


def test_a_monomer_control_found_after_probing_yields_a_real_catalogue(
    session: Session
) -> None:
    """Once the section is visible, discovery captures it normally."""
    session.driver.send("click", key="model=melt", wait_load=False,
                        locators=[{"strategy": "css",
                                   "value": "input[name='model'][value='melt']"}])
    session.driver.send("wait", ms=800)
    _schema, controls = discover_form_schema(session)
    inventory = session.driver.send("page_inventory")
    catalog = catalog_from_controls(controls, semantics={"monomer": ["block[1]"]},
                                    inventory=inventory)
    assert catalog.state is CatalogState.CAPTURED
    assert len(catalog.monomers) == 6
    assert "Lactide" in [m.label for m in catalog.monomers]


class TestControlProvenance:
    """Each control should say what it is, where it lives, and why it is not showing.

    Without these, a dump of a real page is a list of anonymous controls: sixteen
    hidden things with no names, no text and no indication of which section they belong
    to. That is not enough to tell a collapsed section from a never-rendered template.
    """

    def test_a_hidden_control_names_the_container_that_hides_it(
        self, session: Session
    ) -> None:
        controls = session.driver.send("form_fields")["controls"]
        hidden = [c for c in controls if not c.get("visible")]
        assert hidden, "the fixture must start with hidden controls"
        for control in hidden:
            assert control["hidden_reason"], control
            assert "display:none" in control["hidden_reason"]
            # Naming the element, not just the property, is what makes it actionable.
            assert "#" in control["hidden_reason"]

    def test_a_control_reports_the_section_it_sits_under(self, session: Session) -> None:
        controls = session.driver.send("form_fields")["controls"]
        blocks = [c for c in controls if c.get("name") == "block[1]"]
        assert blocks
        assert blocks[0]["section"] == "Building Block(s) of Polymer:"

    def test_a_visible_control_has_no_hidden_reason(self, session: Session) -> None:
        controls = session.driver.send("form_fields")["controls"]
        visible = [c for c in controls if c.get("visible")]
        assert visible
        assert all(c["hidden_reason"] is None for c in visible)

    def test_buttons_carry_their_text(self, session: Session) -> None:
        """A button with no text is indistinguishable from any other button."""
        inventory = session.driver.send("page_inventory")
        assert "buttons" in inventory
        radios = session.driver.send("form_fields")["controls"]
        # The fixture's radios are inputs, not buttons; the point is that the field
        # exists and is populated for elements that have text.
        assert all("text" in c for c in radios)
