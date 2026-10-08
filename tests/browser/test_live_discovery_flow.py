"""Discovery, end to end, with no job anywhere in it (§11-§14).

These exist because the verification graph once required downstream evidence for
upstream discovery: proving a form schema appeared to need a catalogue, and proving a
catalogue appeared to need a download. Every test here runs against a real browser and
a fixture page, and none of them involves a job id, an archive or a submission.
"""

from __future__ import annotations

import functools
import http.server
import socket
import threading
from pathlib import Path

import pytest

from polymer_engine.browser.catalog import CatalogState
from polymer_engine.browser.credentials import Credentials
from polymer_engine.browser.discovery import (
    catalog_from_controls,
    discover_form_schema,
    map_semantics,
)
from polymer_engine.browser.driver import WorkerDriver
from polymer_engine.browser.session import Session
from polymer_engine.browser.spec_mapping import default_probe_spec, map_spec_to_form
from polymer_engine.browser.verification import VerificationRegistry, VerificationState
from polymer_engine.core.config import Secret

pytestmark = pytest.mark.skipif(not WorkerDriver.available(), reason="no .browserenv")

FIXTURES = Path(__file__).parent / "fixtures"
PASSWORD = "correct-horse"


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
def registry(tmp_path) -> VerificationRegistry:
    return VerificationRegistry(tmp_path / "verification.json")


def _discover(site: str, page: str, monkeypatch: pytest.MonkeyPatch):
    """Log in for real, reach the builder, read the page. No job, no archive."""
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", PASSWORD)
    creds = Credentials(email="a@b.c", password=Secret(PASSWORD), typing_delay_ms=3)
    session = Session(credentials=creds, base_url=site)
    session.launch()
    login = session.login(url=f"{site}/login.html", settle_s=0.3)
    assert login.ok, login.detail

    session.navigate(f"{site}/{page}")
    schema, controls = discover_form_schema(session)
    inventory = session.driver.send("page_inventory")
    semantics = map_semantics(controls)
    catalog = catalog_from_controls(controls, source_url=session.current_url(),
                                    semantics=semantics, inventory=inventory)
    return session, schema, semantics, catalog


# -- §13 complete discovery ----------------------------------------------
def test_complete_discovery_proves_three_states_with_no_job(
    site: str, registry: VerificationRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, schema, _semantics, catalog = _discover(site, "builder_full.html", monkeypatch)
    try:
        assert len(schema.fields) == 8, sorted(schema.fields)
        assert catalog.state is CatalogState.CAPTURED
        assert sorted(m.value for m in catalog.monomers) == ["PE", "PLA", "PS"]
        assert catalog.system_modes == ["Single Chain", "Melt", "Solution"]
        assert catalog.tacticity_options == ["Atactic", "Isotactic"]

        registry.verify("LIVE_LOGIN_VERIFIED", evidence={"url": site})
        registry.verify("POLYMER_BUILDER_REACHED", evidence={"url": session.current_url()})
        registry.verify("FORM_SCHEMA_VERIFIED", evidence={"n_fields": len(schema.fields)})
        registry.verify("CATALOG_VERIFIED",
                        evidence={"fingerprint": catalog.fingerprint(),
                                  "n_monomers": len(catalog.monomers)})
        assert set(registry.live_verified()) == {
            "LIVE_LOGIN_VERIFIED", "POLYMER_BUILDER_REACHED", "FORM_SCHEMA_VERIFIED",
            "CATALOG_VERIFIED"}
        # Nothing downstream was touched.
        assert registry.records["DOWNLOAD_VERIFIED"].state is (
            VerificationState.NOT_IMPLEMENTED)
        assert not registry.system_generation_verified
    finally:
        session.close()


# -- §11 the monomer control is not a select -----------------------------
def test_radio_button_monomers_are_discovered(
    site: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact shape that produced '0 monomers' from working code."""
    session, schema, _semantics, catalog = _discover(
        site, "builder_radio_monomer.html", monkeypatch)
    try:
        assert catalog.state is CatalogState.CAPTURED, catalog.notes
        assert sorted(m.value for m in catalog.monomers) == ["PE", "PLA", "PS"]
        assert {m.kind for m in catalog.monomers} == {"radio"}
        assert schema.discovered
    finally:
        session.close()


# -- §12 partial discovery ------------------------------------------------
def test_a_form_verifies_even_when_the_monomer_control_cannot_be_found(
    site: str, registry: VerificationRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§12: do not downgrade the form schema because the catalogue failed."""
    session, schema, _semantics, catalog = _discover(
        site, "builder_no_monomer.html", monkeypatch)
    try:
        assert schema.discovered
        assert catalog.state is CatalogState.NOT_IDENTIFIED
        assert catalog.state.needs_human

        registry.verify("LIVE_LOGIN_VERIFIED", evidence={"url": site})
        registry.verify("POLYMER_BUILDER_REACHED", evidence={"url": session.current_url()})
        registry.verify("FORM_SCHEMA_VERIFIED", evidence={"n_fields": len(schema.fields)})
        registry.needs_review("CATALOG_VERIFIED", "monomer control not identified")

        assert registry.records["FORM_SCHEMA_VERIFIED"].state is (
            VerificationState.LIVE_VERIFIED)
        assert registry.records["CATALOG_VERIFIED"].state is (
            VerificationState.REQUIRES_HUMAN_REVIEW)
        # And the catalogue does not claim the site has no monomers.
        assert "does not mean the Polymer Builder offers no monomers" in catalog.notes
    finally:
        session.close()


def test_spec_mapping_is_reached_without_any_job(
    site: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, schema, semantics, catalog = _discover(site, "builder_full.html", monkeypatch)
    try:
        spec = default_probe_spec(catalog.monomers[0].label)
        mapping = map_spec_to_form(spec, schema, semantics)
        assert mapping.schema_discovered
        assert mapping.counts().get("MAPPED", 0) >= 2
    finally:
        session.close()


# -- §14 downstream stays downstream --------------------------------------
def test_downstream_states_need_a_job_and_discovery_does_not(
    registry: VerificationRegistry
) -> None:
    for name in ("LIVE_LOGIN_VERIFIED", "POLYMER_BUILDER_REACHED",
                 "FORM_SCHEMA_VERIFIED", "CATALOG_VERIFIED", "SPEC_MAPPING_VERIFIED"):
        registry.verify(name, evidence={"proof": name})

    with pytest.raises(ValueError):
        registry.verify("DOWNLOAD_VERIFIED", evidence={"sha256": "abc"})

    registry.verify("MELT_BUILD_VERIFIED", evidence={"job_id": "J-1"})
    registry.verify("JOB_ID_VERIFIED", evidence={"job_id": "J-1"})
    registry.verify("JOB_MONITORING_VERIFIED", evidence={"status": "done"})
    registry.verify("DOWNLOAD_VERIFIED", evidence={"sha256": "abc"})
    assert registry.records["DOWNLOAD_VERIFIED"].state is VerificationState.LIVE_VERIFIED
