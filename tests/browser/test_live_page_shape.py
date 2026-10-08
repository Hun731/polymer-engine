"""Discovery against the structural shape the live Polymer Builder actually has.

The first `dump` of the authenticated page reported 27 form controls and 8 schema
fields. Nineteen controls were being dropped silently, including two `<select>` elements
with no `name`, `id` or `label` -- exactly the kind of control a monomer list could live
in. The same run mapped `n_chains` onto a button reading "Next Step: Build Polymer
Chains", because the label contains the word "chains".

The fixture reproduces those structural features. It is not a copy of the live page and
makes no claim about its chemistry.
"""

from __future__ import annotations

import functools
import http.server
import socket
import threading
from pathlib import Path

import pytest

from polymer_engine.browser.credentials import Credentials
from polymer_engine.browser.discovery import (
    catalog_from_controls,
    choice_sets,
    discover_form_schema,
    map_semantics,
)
from polymer_engine.browser.driver import WorkerDriver
from polymer_engine.browser.session import Session
from polymer_engine.core.config import Secret

pytestmark = pytest.mark.skipif(not WorkerDriver.available(), reason="no .browserenv")

FIXTURES = Path(__file__).parent / "fixtures"
PAGE = "builder_unnamed_selects.html"


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
def discovered(site: str):
    creds = Credentials(email="a@b.c", password=Secret("x"), typing_delay_ms=1)
    session = Session(credentials=creds, base_url=site)
    session.launch()
    session.navigate(f"{site}/{PAGE}")
    schema, controls = discover_form_schema(session)
    inventory = session.driver.send("page_inventory")
    try:
        yield session, schema, controls, inventory
    finally:
        session.close()


def test_no_visible_control_is_dropped_silently(discovered) -> None:
    """Every visible, non-hidden control must reach the schema."""
    _session, schema, controls, _inventory = discovered
    visible = [c for c in controls
               if c.get("visible") and (c.get("type") or "") != "hidden"]
    assert len(schema.fields) == len(visible), (
        f"{len(visible) - len(schema.fields)} visible control(s) missing from the "
        f"schema: this is how a monomer selector becomes invisible")


def test_selects_without_a_name_are_still_addressable(discovered) -> None:
    """The two unnamed selects are the shape a monomer list could live in."""
    session, schema, controls, _inventory = discovered
    unnamed = [c for c in controls
               if c.get("tag") == "select" and not c.get("name") and not c.get("id")]
    assert len(unnamed) == 2

    positional = [key for key, spec in schema.fields.items()
                  if spec.control == "select" and key.startswith("select@")]
    assert len(positional) == 2, sorted(schema.fields)

    # And each positional key really resolves to exactly one element.
    for key in positional:
        result = session.driver.send(
            "locate", key=key,
            locators=[loc.as_dict() for loc in schema.require(key).ordered()])
        assert result.get("ok"), f"{key} does not resolve: {result.get('error')}"


def test_repeated_indexed_names_both_survive(discovered) -> None:
    """capf[1] and capl[1] are distinct controls and must not collide."""
    _session, schema, _controls, _inventory = discovered
    assert "capf[1]" in schema.fields
    assert "capl[1]" in schema.fields


def test_a_button_is_never_mapped_to_a_scientific_quantity(discovered) -> None:
    """'Next Step: Build Polymer Chains' is an action, not the chain count."""
    _session, _schema, controls, _inventory = discovered
    semantics = map_semantics(controls)
    for semantic, keys in semantics.items():
        for key in keys:
            assert "next" not in key.lower(), (
                f"{semantic} was mapped onto a button ({key})")
    assert "n_chains" not in semantics or semantics["n_chains"] != ["next"]


def test_every_choice_set_is_seen_including_unnamed_ones(discovered) -> None:
    _session, _schema, controls, inventory = discovered
    sets = choice_sets(controls, inventory)
    kinds = {entry["kind"] for entry in sets.values()}
    assert "radio" in kinds, "the System Type radio group must be visible as a choice set"
    # Four selects on the page, two of them unnamed.
    assert sum(1 for e in sets.values() if e["kind"] == "select") >= 2


def test_the_catalogue_is_not_identified_rather_than_empty(discovered) -> None:
    """Nothing maps to 'monomer' here, so the honest answer is 'we did not find it'."""
    from polymer_engine.browser.catalog import CatalogState

    _session, _schema, controls, inventory = discovered
    catalog = catalog_from_controls(controls, semantics=map_semantics(controls),
                                    inventory=inventory)
    assert catalog.state is CatalogState.NOT_IDENTIFIED
    assert catalog.inspected["n_choice_sets"] >= 3
    assert "does not mean the Polymer Builder offers no monomers" in catalog.notes
