"""Fill/verify/submit, job capture, the queue and diagnostics (§23-§30, §51-§53, §68)."""

from __future__ import annotations

import json

import pytest
from tests.browser.conftest import FakeDriver, builder_controls

from polymer_engine.browser import diagnostics as diag
from polymer_engine.browser.credentials import Credentials
from polymer_engine.browser.discovery import (
    catalog_from_controls,
    discover_form_schema,
    map_semantics,
)
from polymer_engine.browser.queue import BuildEntry, BuildQueue
from polymer_engine.browser.session import Session
from polymer_engine.browser.states import BrowserState, JobState
from polymer_engine.browser.workflows import (
    BuilderForm,
    capture_job_id,
    fill_builder_form,
    submit_build,
    verify_before_submit,
)
from polymer_engine.core.config import Secret
from polymer_engine.core.logging import register_secret
from polymer_engine.simulation.charmm_gui_spec import (
    Comonomer,
    PolymerBuilderSpec,
    SystemType,
)

BASE = {"polymer_id": "pe", "name": "Polyethylene", "repeat_unit_smiles": "CC",
        "degree_of_polymerization": 30, "n_chains": 4, "force_field": "CHARMM36",
        "temperature_k": 300.0, "pressure_bar": 1.0}


def _spec(**overrides) -> PolymerBuilderSpec:
    return PolymerBuilderSpec(**{**BASE, **overrides})


def _prepare(pages: dict) -> tuple[Session, BuilderForm, object]:
    driver = FakeDriver(pages, start="builder")
    session = Session(driver, credentials=Credentials(email="a", password=Secret("b"),
                                                      typing_delay_ms=0),
                      base_url="https://example.invalid")
    schema, controls = discover_form_schema(session)
    semantics = map_semantics(controls)
    catalog = catalog_from_controls(controls, semantics=semantics)
    return session, BuilderForm(schema=schema, semantics=semantics), catalog


# -- the specification refuses contradictions ----------------------------
def test_a_single_chain_system_cannot_have_twenty_chains() -> None:
    problems = _spec(system_type=SystemType.SINGLE_CHAIN, n_chains=20).composition_problems()
    assert any("single-chain" in p for p in problems)


def test_a_composition_is_never_normalised_silently() -> None:
    spec = _spec(comonomers=[Comonomer("A", 0.7), Comonomer("B", 0.2)], sequence="random")
    assert any("not 1" in p and "not be normalised silently" in p
               for p in spec.composition_problems())


def test_a_copolymer_needs_an_explicit_sequence() -> None:
    spec = _spec(comonomers=[Comonomer("A", 0.5), Comonomer("B", 0.5)])
    assert any("explicit sequence" in p for p in spec.composition_problems())


def test_a_solution_needs_a_solvent() -> None:
    assert any("needs a solvent" in p
               for p in _spec(system_type=SystemType.SOLUTION).composition_problems())


def test_the_fingerprint_excludes_bookkeeping_and_tracks_chemistry() -> None:
    a, b = _spec(), _spec(charmm_gui_job_id="JOB-1", notes="anything")
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != _spec(degree_of_polymerization=31).fingerprint()
    assert a.fingerprint() != _spec(system_type=SystemType.SINGLE_CHAIN,
                                    n_chains=1).fingerprint()


# -- fill and verify -----------------------------------------------------
def test_a_well_behaved_form_verifies_and_submits(pages: dict) -> None:
    session, form, catalog = _prepare(pages)
    spec = _spec()
    report = verify_before_submit(session, spec, form,
                                  fill_builder_form(session, spec, form, catalog), catalog)
    assert report.safe_to_submit
    assert {o.semantic for o in report.outcomes} == {
        "monomer", "degree_of_polymerization", "n_chains"}
    assert all(o.agrees for o in report.outcomes)

    submission = submit_build(session, spec, form, report, submit_key="next")
    assert submission.state is BrowserState.OK
    assert submission.job_id == "1234567"


def test_a_form_that_clamps_a_value_blocks_submission(pages: dict) -> None:
    # The failure that would otherwise produce a valid-looking system for DP=3.
    pages["builder"].rewrite = lambda key, value: value[:1] if key == "dp" else value
    session, form, catalog = _prepare(pages)
    spec = _spec()
    report = verify_before_submit(session, spec, form,
                                  fill_builder_form(session, spec, form, catalog), catalog)
    assert not report.safe_to_submit

    submission = submit_build(session, spec, form, report, submit_key="next")
    assert submission.state is BrowserState.REFUSED
    assert submission.job_id is None
    # Nothing was clicked.
    assert not any(cmd == "click" for cmd, _ in session.driver.calls)


def test_a_silently_swapped_value_is_caught_by_reading_the_page_back(pages: dict) -> None:
    # The form accepts the keystrokes but stores something else entirely.
    pages["builder"].rewrite = lambda key, value: "7" if key == "nchain" else value
    session, form, catalog = _prepare(pages)
    spec = _spec()
    report = verify_before_submit(session, spec, form,
                                  fill_builder_form(session, spec, form, catalog), catalog)
    assert report.state is BrowserState.STRUCTURE_MISMATCH
    assert not report.safe_to_submit
    assert any(o.semantic == "n_chains" and o.readback == "7"
               for o in report.disagreements)


def test_an_unrequestable_monomer_stops_before_any_field_is_filled(pages: dict) -> None:
    session, form, catalog = _prepare(pages)
    report = fill_builder_form(session, _spec(name="Polypropylene"), form, catalog)
    assert report.state is BrowserState.MONOMER_NOT_FOUND
    assert not report.safe_to_submit


def test_a_missing_required_control_is_a_schema_mismatch(pages: dict) -> None:
    pages["builder"].controls = [c for c in builder_controls() if c["id"] != "nchain"]
    session, form, catalog = _prepare(pages)
    report = fill_builder_form(session, _spec(), form, catalog)
    assert report.state is BrowserState.UI_SCHEMA_MISMATCH
    assert "n_chains" in report.reason


def test_an_undiscovered_schema_can_never_drive_a_submission(pages: dict) -> None:
    from polymer_engine.browser.selectors import builder_schema_placeholder

    session, _form, catalog = _prepare(pages)
    form = BuilderForm(schema=builder_schema_placeholder(), semantics={})
    report = fill_builder_form(session, _spec(), form, catalog)
    assert report.state is BrowserState.UI_SCHEMA_MISMATCH
    assert "not been discovered" in report.reason


def test_an_ambiguous_semantic_mapping_requires_a_person(pages: dict) -> None:
    session, form, catalog = _prepare(pages)
    form.semantics["n_chains"] = ["nchain", "dp"]
    report = fill_builder_form(session, _spec(), form, catalog)
    assert report.state is BrowserState.REQUIRES_HUMAN_REVIEW


def test_an_unoffered_tacticity_is_a_structure_mismatch(pages: dict) -> None:
    session, form, catalog = _prepare(pages)
    report = fill_builder_form(session, _spec(tacticity="syndiotactic"), form, catalog)
    assert report.state is BrowserState.STRUCTURE_MISMATCH


def test_a_contradictory_request_is_refused_before_the_browser_is_touched(pages: dict) -> None:
    session, form, catalog = _prepare(pages)
    before = len(session.driver.calls)
    report = fill_builder_form(session, _spec(system_type=SystemType.SINGLE_CHAIN,
                                              n_chains=9), form, catalog)
    assert report.state is BrowserState.REFUSED
    assert len(session.driver.calls) == before


# -- job identifiers -----------------------------------------------------
def test_two_candidate_job_ids_yield_no_job_id(pages: dict) -> None:
    pages["submitted"].text = "jobid=111111 ... jobid=222222"
    session, _form, _catalog = _prepare(pages)
    session.navigate("https://example.invalid/done")
    submission = capture_job_id(session, "fp")
    assert submission.state is BrowserState.REQUIRES_HUMAN_REVIEW
    assert submission.job_id is None
    assert submission.candidates == ["111111", "222222"]


def test_no_job_id_is_never_invented(pages: dict) -> None:
    pages["submitted"].text = "Thank you."
    session, _form, _catalog = _prepare(pages)
    session.navigate("https://example.invalid/done")
    submission = capture_job_id(session, "fp")
    assert submission.state is BrowserState.SUBMISSION_FAILED
    assert submission.job_id is None
    assert "must be found by a person" in submission.reason


def test_a_captcha_at_submission_stops_the_workflow(pages: dict) -> None:
    pages["submitted"].captcha = True
    session, form, catalog = _prepare(pages)
    spec = _spec()
    report = verify_before_submit(session, spec, form,
                                  fill_builder_form(session, spec, form, catalog), catalog)
    submission = submit_build(session, spec, form, report, submit_key="next")
    assert submission.state is BrowserState.HUMAN_INTERVENTION_REQUIRED
    assert submission.job_id is None


# -- the queue -----------------------------------------------------------
@pytest.fixture
def queue(tmp_path) -> BuildQueue:
    return BuildQueue(tmp_path / "queue.json")


def test_an_in_flight_job_is_monitored_not_resubmitted(queue: BuildQueue) -> None:
    entry = queue.add(BuildEntry("pe", "PE", "fp", "melt", "baseline"))
    entry.advance(JobState.RUNNING, "submitted")
    check = queue.may_submit("fp")
    assert not check.may_submit
    assert check.action == "monitor"


def test_an_unknown_submission_outcome_blocks_rather_than_retries(queue: BuildQueue) -> None:
    entry = queue.add(BuildEntry("pe", "PE", "fp", "melt", "baseline"))
    queue.record_submission_failure(entry, BrowserState.SUBMISSION_FAILED,
                                    "no job id on the page")
    assert entry.state is JobState.BLOCKED
    assert not queue.may_submit("fp").may_submit


def test_a_pre_click_refusal_stays_retryable(queue: BuildQueue) -> None:
    entry = queue.add(BuildEntry("pe", "PE", "fp", "melt", "baseline"))
    queue.record_submission_failure(entry, BrowserState.UI_SCHEMA_MISMATCH, "field gone")
    assert entry.state is JobState.QUEUED
    assert queue.may_submit("fp").may_submit


def test_a_finished_build_is_reused(queue: BuildQueue) -> None:
    entry = queue.add(BuildEntry("pe", "PE", "fp", "melt", "baseline"))
    entry.job_id = "J1"
    entry.advance(JobState.VALIDATED, "validated")
    assert queue.may_submit("fp").action == "reuse"


def test_a_queued_build_needs_a_stated_rationale(queue: BuildQueue) -> None:
    with pytest.raises(ValueError, match="rationale"):
        queue.add(BuildEntry("pe", "PE", "fp", "melt", "   "))


def test_only_one_build_runs_at_a_time(queue: BuildQueue) -> None:
    a = queue.add(BuildEntry("a", "A", "fpa", "melt", "first", priority=10))
    queue.add(BuildEntry("b", "B", "fpb", "melt", "second", priority=90))
    assert queue.next_ready().polymer_id == "b"  # priority order
    a.advance(JobState.RUNNING, "submitted")
    assert queue.next_ready() is None  # nothing else starts while one is in flight


def test_the_queue_survives_a_round_trip(queue: BuildQueue) -> None:
    entry = queue.add(BuildEntry("pe", "PE", "fp", "melt", "baseline"))
    entry.job_id = "J9"
    entry.advance(JobState.DOWNLOADED, "got it")
    queue.save()
    reloaded = BuildQueue(queue.path)
    assert reloaded.by_job("J9") is not None
    assert reloaded.find("fp").state is JobState.DOWNLOADED


# -- diagnostics ---------------------------------------------------------
def test_a_diagnostic_carries_no_credential(pages: dict, tmp_path) -> None:
    register_secret("s3cret-passphrase")
    session, _form, _catalog = _prepare(pages)
    captured = diag.capture(
        session, name="t", state="UI_SCHEMA_MISMATCH",
        reason="failed while typing s3cret-passphrase",
        payload={"url": "u", "cookie": "abc", "password": "p",
                 "error": "Bearer eyJhbGciOiJIUzI1NiJ9"},
        directory=tmp_path,
    )
    text = (captured.directory / "diagnostic.json").read_text()
    assert "s3cret-passphrase" not in text
    assert "eyJhbGciOiJIUzI1NiJ9" not in text
    for banned in ("cookie", '"password"'):
        assert banned not in text
    assert json.loads(text)["state"] == "UI_SCHEMA_MISMATCH"


def test_no_screenshot_is_taken_of_a_page_holding_a_password(pages: dict, tmp_path) -> None:
    driver = FakeDriver(pages, start="login")
    session = Session(driver, credentials=Credentials(email="a", password=Secret("b")),
                      base_url="https://example.invalid")
    captured = diag.capture(session, name="login-fail", state="AUTHENTICATION_FAILED",
                            reason="rejected", directory=tmp_path)
    assert "page.png" not in captured.files
    assert "cannot be sanitised" in captured.screenshot_skipped


def test_a_screenshot_is_taken_when_no_password_control_is_present(
    pages: dict, tmp_path
) -> None:
    session, _form, _catalog = _prepare(pages)
    captured = diag.capture(session, name="builder-fail", state="UI_SCHEMA_MISMATCH",
                            reason="x", directory=tmp_path)
    assert "page.png" in captured.files
