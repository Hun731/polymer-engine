"""The opt-in live-website test (§55, §56, §69).

Skipped unless **both** ``CHARMM_GUI_LIVE_TEST=1`` and real credentials are present, so
ordinary CI never touches the live service and never needs an account.

What it does is deliberately minimal: one login, one catalogue capture, and -- only when
``CHARMM_GUI_LIVE_SUBMIT=1`` is *additionally* set -- one tiny build. There is no bulk
path here and no loop. CHARMM-GUI is a shared academic service, and a test suite that
submits jobs in quantity is an abuse of it regardless of whether the account is ours.

Everything written goes under ``campaign/charmm_gui/live_test``, separate from
production research data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from polymer_engine.browser.credentials import from_environment, live_test_enabled
from polymer_engine.browser.discovery import (
    catalog_from_controls,
    discover_form_schema,
    find_polymer_builder,
    map_semantics,
)
from polymer_engine.browser.driver import WorkerDriver
from polymer_engine.browser.session import Session
from polymer_engine.browser.states import BrowserState

LIVE_ROOT = Path("campaign/charmm_gui/live_test")

_credentials = from_environment()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not live_test_enabled(), reason="set CHARMM_GUI_LIVE_TEST=1"),
    pytest.mark.skipif(not _credentials.complete,
                       reason="set CHARMM_GUI_EMAIL and CHARMM_GUI_PASSWORD"),
    pytest.mark.skipif(not WorkerDriver.available(), reason="no .browserenv"),
]


@pytest.fixture(scope="module")
def session():
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    with Session(credentials=from_environment(), headless=True,
                 downloads_dir=LIVE_ROOT / "downloads") as live:
        yield live


def test_login(session: Session) -> None:
    """One login. A failure here is reported, never retried with the same password."""
    result = session.login()
    if result.state is BrowserState.HUMAN_INTERVENTION_REQUIRED:
        pytest.skip("CHARMM-GUI presented a human-verification challenge; "
                    "a person must complete it. It is not bypassed.")
    assert result.state is BrowserState.OK, result.detail
    assert session.authenticated


def test_catalog_discovery(session: Session) -> None:
    """Capture what Polymer Builder actually offers, and write it down."""
    if not session.authenticated:
        assert session.login().ok
    reached = find_polymer_builder(session)
    if not reached.ok:
        pytest.skip(f"Polymer Builder not reachable: {reached.detail}")

    schema, controls = discover_form_schema(session)
    assert schema.discovered
    assert controls, "the Polymer Builder page exposed no form controls"

    semantics = map_semantics(controls)
    catalog = catalog_from_controls(controls, source_url=session.current_url(),
                                    semantics=semantics)
    written = catalog.write(Path("data/charmm_gui"))
    assert written["json"].exists()
    # The catalogue records what was seen; it never claims that is everything.
    assert catalog.completeness == "unknown"


@pytest.mark.skipif(
    os.environ.get("CHARMM_GUI_LIVE_SUBMIT") != "1",
    reason="set CHARMM_GUI_LIVE_SUBMIT=1 to permit ONE real build submission",
)
def test_one_tiny_build(session: Session) -> None:
    """A single smallest-possible build, gated behind a second explicit opt-in.

    The specification comes from the environment rather than this file, because the only
    honest source for "a monomer Polymer Builder supports" is the catalogue captured
    above -- not a name written into a test months earlier.
    """
    from polymer_engine.browser.acquisition import CharmmGuiAcquisition
    from polymer_engine.browser.catalog import Catalog
    from polymer_engine.browser.matching import resolve_monomer
    from polymer_engine.simulation.charmm_gui_spec import PolymerBuilderSpec, SystemType

    catalog = Catalog.read(Path("data/charmm_gui"))
    if catalog is None or not catalog.monomers:
        pytest.skip("no catalogue has been discovered yet; run catalog discovery first")

    requested = os.environ.get("CHARMM_GUI_LIVE_MONOMER", "")
    if not requested:
        pytest.skip("set CHARMM_GUI_LIVE_MONOMER to a monomer from the captured "
                    "catalogue; one is not chosen for you")
    match = resolve_monomer(catalog, requested)
    assert match.resolved, match.reason

    submit_key = os.environ.get("CHARMM_GUI_LIVE_SUBMIT_FIELD", "")
    if not submit_key:
        pytest.skip("set CHARMM_GUI_LIVE_SUBMIT_FIELD to the discovered control that "
                    "starts the build; it is not guessed")

    spec = PolymerBuilderSpec(
        polymer_id="live-test", name=requested, repeat_unit_smiles="",
        degree_of_polymerization=int(os.environ.get("CHARMM_GUI_LIVE_DP", "10")),
        n_chains=1, force_field="CHARMM36", temperature_k=300.0, pressure_bar=1.0,
        system_type=SystemType.SINGLE_CHAIN,
        notes="smallest possible system, submitted once by the live test",
    )
    from polymer_engine.browser.queue import BuildQueue

    acquisition = CharmmGuiAcquisition(
        session=session, queue=BuildQueue(LIVE_ROOT / "build_queue.json"),
        root=LIVE_ROOT,
    )
    result = acquisition.acquire(
        spec, rationale="live integration test: smallest supported system",
        submit_key=submit_key, poll_timeout_s=1800.0, parameterize=False,
    )
    # Any of these is a legitimate outcome; a fabricated job id is not.
    assert result.state in {
        BrowserState.OK, BrowserState.UI_SCHEMA_MISMATCH,
        BrowserState.STRUCTURE_MISMATCH, BrowserState.HUMAN_INTERVENTION_REQUIRED,
        BrowserState.REQUIRES_HUMAN_REVIEW, BrowserState.SUBMISSION_FAILED,
    }, result.reason
    if result.state is BrowserState.OK:
        assert result.job_id, "an OK submission must carry a real job id"
