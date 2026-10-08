"""Integration tests against a real Chromium driving local fixture pages.

These exercise the parts a fake cannot vouch for: that Playwright is really installed in
the isolated environment, that ``keyboard.type`` dispatches genuine key events, that the
DOM-reading JavaScript is valid, and that the sanitiser runs over real page markup.

The login fixture **rejects paste and counts keydown events**, so a test that passes it
could not have been satisfied by ``fill()`` or a clipboard write. That is the closest
thing to a proof of §16 that can be obtained without the live site.

Skipped when the browser environment is absent, so ordinary CI never needs it.
"""

from __future__ import annotations

import functools
import http.server
import json
import socket
import threading
from pathlib import Path

import pytest

from polymer_engine.browser.credentials import Credentials
from polymer_engine.browser.discovery import (
    catalog_from_controls,
    discover_form_schema,
    map_semantics,
)
from polymer_engine.browser.driver import WorkerDriver
from polymer_engine.browser.session import Session
from polymer_engine.browser.states import BrowserState
from polymer_engine.browser.workflows import (
    BuilderForm,
    fill_builder_form,
    submit_build,
    verify_before_submit,
)
from polymer_engine.core.config import Secret
from polymer_engine.simulation.charmm_gui_spec import PolymerBuilderSpec, SystemType

pytestmark = pytest.mark.skipif(
    not WorkerDriver.available(),
    reason="no .browserenv; see docs/CHARMM_GUI_BROWSER.md for the install",
)

FIXTURES = Path(__file__).parent / "fixtures"
PASSWORD = "correct-horse"


@pytest.fixture(scope="module")
def site() -> str:
    """Serve the fixture pages on a free port for the duration of the module."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(FIXTURES))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()


@pytest.fixture
def creds() -> Credentials:
    return Credentials(email="a@b.c", password=Secret(PASSWORD), typing_delay_ms=5)


def test_the_browser_environment_is_real_and_isolated() -> None:
    capabilities = WorkerDriver.capabilities()
    assert capabilities["available"], capabilities.get("error")
    assert capabilities["playwright"]
    assert capabilities["chromium"]
    # Chromium lives under the tooling environment, not beside the scientific stack.
    assert ".browserenv" in capabilities["env"]


def test_login_succeeds_against_a_form_that_rejects_paste(site: str, creds: Credentials) -> None:
    with Session(credentials=creds, base_url=site) as session:
        result = session.login(url=f"{site}/login.html", settle_s=0.3)
        assert result.state is BrowserState.OK, result.detail
        assert session.authenticated
        text = session.page_text(300)
        assert "Welcome back" in text
        # The fixture only signs in after more than five real keydown events.
        assert "keystrokes=" in text
        assert "paste rejected" not in text


def test_a_wrong_password_is_rejected_and_not_retried(site: str) -> None:
    creds = Credentials(email="a@b.c", password=Secret("wrong"), typing_delay_ms=1)
    with Session(credentials=creds, base_url=site) as session:
        result = session.login(url=f"{site}/login.html", settle_s=0.3)
        assert result.state is BrowserState.AUTHENTICATION_FAILED
        assert not result.state.retryable


def test_the_password_never_reaches_a_snapshot(
    site: str, creds: Credentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Type the real password into a real field, then prove the artifacts are clean."""
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", PASSWORD)
    with Session(credentials=creds, base_url=site) as session:
        session.navigate(f"{site}/login.html")
        typed = session.type_secret("password", "CHARMM_GUI_PASSWORD")
        assert typed["ok"]
        assert typed["chars_received"] == len(PASSWORD)
        assert "value" not in typed, "a secret field is never echoed back"

        snapshot = session.snapshot(50000)
        assert PASSWORD not in snapshot["html"]
        fields = session.driver.send("form_fields")
        assert PASSWORD not in json.dumps(fields)


def test_a_real_form_is_discovered_without_any_hard_coded_selector(
    site: str, creds: Credentials
) -> None:
    with Session(credentials=creds, base_url=site) as session:
        session.navigate(f"{site}/builder.html")
        schema, controls = discover_form_schema(session)
        assert schema.discovered
        assert {"monomer", "dp", "nchain", "tacticity"} <= set(schema.fields)
        semantics = map_semantics(controls)
        catalog = catalog_from_controls(controls, semantics=semantics)
        assert sorted(m.value for m in catalog.monomers) == ["PE", "PLA"]
        assert catalog.completeness == "unknown"
        assert catalog.tacticity_options == ["Atactic", "Isotactic"]


def _spec(**overrides) -> PolymerBuilderSpec:
    return PolymerBuilderSpec(
        polymer_id="pe", name="Polyethylene", repeat_unit_smiles="CC",
        degree_of_polymerization=30, n_chains=4, force_field="CHARMM36",
        temperature_k=300.0, pressure_bar=1.0, system_type=SystemType.MELT,
        **overrides)


def test_a_real_clamping_form_blocks_a_real_submission(site: str, creds: Credentials) -> None:
    """The whole point, exercised end to end in a real browser."""
    with Session(credentials=creds, base_url=site) as session:
        session.navigate(f"{site}/builder_clamping.html")
        schema, controls = discover_form_schema(session)
        semantics = map_semantics(controls)
        catalog = catalog_from_controls(controls, semantics=semantics)
        form = BuilderForm(schema=schema, semantics=semantics)
        spec = _spec()
        report = verify_before_submit(
            session, spec, form, fill_builder_form(session, spec, form, catalog), catalog)

        assert not report.safe_to_submit
        submission = submit_build(session, spec, form, report, submit_key="next")
        assert submission.state is BrowserState.REFUSED
        assert submission.job_id is None
        # The page never advanced, so no job was created.
        assert "Job submitted" not in session.page_text(200)
