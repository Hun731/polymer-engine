"""Credential handling and the login flow (§15, §16, §17, §68)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.browser.conftest import FakeDriver, login_page

from polymer_engine.browser.credentials import (
    DEFAULT_TYPING_DELAY_MS,
    Credentials,
    from_environment,
    live_test_enabled,
)
from polymer_engine.browser.selectors import login_schema
from polymer_engine.browser.session import Session
from polymer_engine.browser.states import BrowserState
from polymer_engine.core.config import Secret

SECRET = "s3cret-passphrase"


@pytest.fixture
def creds() -> Credentials:
    return Credentials(email="user@example.invalid", password=Secret(SECRET),
                       typing_delay_ms=1)


@pytest.fixture
def driver(pages: dict) -> FakeDriver:
    return FakeDriver(pages, start="login")


def _session(driver: FakeDriver, creds: Credentials) -> Session:
    return Session(driver, credentials=creds, base_url="https://example.invalid")


# -- credentials ---------------------------------------------------------
def test_password_never_renders_itself(creds: Credentials) -> None:
    assert SECRET not in repr(creds)
    assert SECRET not in str(creds)
    assert SECRET not in json.dumps(creds.as_dict())
    assert creds.as_dict()["password_set"] is True


def test_credentials_come_only_from_the_environment() -> None:
    creds = from_environment({"CHARMM_GUI_EMAIL": "a@b.c", "CHARMM_GUI_PASSWORD": SECRET})
    assert creds.complete
    assert creds.password.reveal() == SECRET
    # The value reaches the worker through its environment, never through a message.
    assert creds.worker_env({})["CHARMM_GUI_PASSWORD"] == SECRET


def test_missing_credentials_are_a_state_not_an_exception() -> None:
    creds = from_environment({})
    assert not creds.complete
    assert creds.missing == ["CHARMM_GUI_EMAIL", "CHARMM_GUI_PASSWORD"]


@pytest.mark.parametrize(("raw", "expected"), [
    ("", DEFAULT_TYPING_DELAY_MS), ("nonsense", DEFAULT_TYPING_DELAY_MS),
    ("15", 15), ("-5", 0), ("999999", 1000),
])
def test_typing_delay_is_bounded(raw: str, expected: int) -> None:
    assert from_environment({"CHARMM_GUI_TYPING_DELAY_MS": raw}).typing_delay_ms == expected


def test_live_tests_are_off_unless_switched_on() -> None:
    assert not live_test_enabled({})
    assert not live_test_enabled({"CHARMM_GUI_LIVE_TEST": "0"})
    assert live_test_enabled({"CHARMM_GUI_LIVE_TEST": "1"})


# -- login ---------------------------------------------------------------
def test_login_types_the_password_by_keyboard_and_never_carries_its_value(
    driver: FakeDriver, creds: Credentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", SECRET)
    result = _session(driver, creds).login(settle_s=0)
    assert result.state is BrowserState.OK

    secret_calls = [p for cmd, p in driver.calls if cmd == "type_secret"]
    assert len(secret_calls) == 1
    # Only the *name* of the variable crosses the boundary.
    assert secret_calls[0]["env_var"] == "CHARMM_GUI_PASSWORD"
    assert SECRET not in json.dumps(driver.calls, default=str)
    # And nothing paste-like was ever issued.
    assert not any(cmd in {"fill", "paste"} for cmd, _ in driver.calls)


def test_login_stops_at_a_captcha_and_does_not_try_to_solve_it(
    pages: dict, creds: Credentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", SECRET)
    pages["login"] = login_page(captcha=True)
    driver = FakeDriver(pages, start="login")
    result = _session(driver, creds).login(settle_s=0)

    assert result.state is BrowserState.HUMAN_INTERVENTION_REQUIRED
    assert result.state.needs_human
    # The password was never typed: the flow stopped before touching the form.
    assert not any(cmd == "type_secret" for cmd, _ in driver.calls)


def test_login_stops_at_an_mfa_prompt(
    pages: dict, creds: Credentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", SECRET)
    pages["home"].text = "Enter the verification code from your authenticator app"
    pages["home"].controls = []
    driver = FakeDriver(pages, start="login")
    result = _session(driver, creds).login(settle_s=0)
    assert result.state is BrowserState.HUMAN_INTERVENTION_REQUIRED


def test_a_rejected_password_is_not_retryable(
    pages: dict, creds: Credentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", SECRET)
    # Still on a page with a visible password field after submitting.
    pages["home"] = login_page()
    pages["home"].text = "Incorrect password"
    driver = FakeDriver(pages, start="login")
    result = _session(driver, creds).login(settle_s=0)

    assert result.state is BrowserState.AUTHENTICATION_FAILED
    assert not result.state.retryable, "retrying a rejected password risks a lockout"
    assert "not retried" in result.detail.lower()


def test_a_still_present_login_form_is_a_failure_even_without_an_error_message(
    pages: dict, creds: Credentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", SECRET)
    pages["home"] = login_page()
    pages["home"].text = "Sign in"  # no error text at all
    driver = FakeDriver(pages, start="login")
    assert _session(driver, creds).login(settle_s=0).state is BrowserState.AUTHENTICATION_FAILED


def test_login_without_credentials_never_launches_a_browser(driver: FakeDriver) -> None:
    result = Session(driver, credentials=Credentials(),
                     base_url="https://example.invalid").login()
    assert result.state is BrowserState.CREDENTIALS_MISSING
    assert driver.calls == []


def test_a_missing_password_field_is_a_schema_mismatch_not_a_login_failure(
    pages: dict, creds: Credentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", SECRET)
    pages["login"].controls = [c for c in pages["login"].controls
                               if c.get("type") != "password"]
    driver = FakeDriver(pages, start="login")
    result = _session(driver, creds).login(settle_s=0)
    assert result.state is BrowserState.UI_SCHEMA_MISMATCH


def test_a_field_that_truncates_the_password_is_reported(
    pages: dict, creds: Credentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", SECRET)
    pages["login"].rewrite = lambda _key, value: value[:4]
    driver = FakeDriver(pages, start="login")
    result = _session(driver, creds).login(settle_s=0)
    assert result.state is BrowserState.UI_SCHEMA_MISMATCH
    assert "characters" in result.detail
    # The diagnostic payload reports counts, never the value.
    assert SECRET not in json.dumps(result.as_dict(), default=str)


class TestSubmitControlSelection:
    """Regression for a real defect found on the live CHARMM-GUI sign-in page.

    That page carries a navigation control outside the login form::

        <button onclick="window.location.href='./?doc=sign'">Login</button>

    while the form's own submit is ``<input type="submit" value="Submit">``. Matching a
    button by the word "login" clicked the navigation control, which reloaded the page
    and discarded everything typed. The symptom was a login that failed with *correct*
    credentials, leaving the form present -- indistinguishable, from the outside, from a
    rejected password.

    The fixture reproduces that structure exactly.
    """

    FIXTURE = "login_navbutton.html"

    def test_the_submit_locator_is_scoped_to_the_login_form(self) -> None:
        from polymer_engine.browser.selectors import LOGIN_FORM, login_schema

        ordered = login_schema().require("submit").ordered()
        first = ordered[0]
        assert first.strategy == "css"
        assert first.value.startswith(LOGIN_FORM), (
            "the first submit locator must be scoped to the form holding the password")
        assert "input[type='submit']" in first.value

    def test_name_based_matches_come_last(self) -> None:
        """They are the ones that match navigation controls, so they must not win."""
        ordered = login_schema().require("submit").ordered()
        role_positions = [i for i, loc in enumerate(ordered) if loc.strategy == "role"]
        css_positions = [i for i, loc in enumerate(ordered) if loc.strategy == "css"]
        assert role_positions, "name-based fallbacks should still exist"
        assert min(role_positions) > max(css_positions)

    def test_strict_order_is_honoured_rather_than_re_sorted(self) -> None:
        """Without this the strategy preference would put `role` back in front."""
        spec = login_schema().require("submit")
        assert spec.strict_order
        assert spec.ordered() == spec.locators

    def test_no_login_locator_matches_a_control_outside_the_form(self) -> None:
        """Parsed from the fixture, so the check does not need a browser."""
        import re

        html = (Path(__file__).parent / "fixtures" / self.FIXTURE).read_text()
        outside = re.search(r"<button[^>]*>Login</button>", html)
        assert outside, "the fixture must contain the decoy navigation control"
        # Every leading submit locator is scoped, so none can reach outside the form.
        scoped = list(login_schema().require("submit").ordered()[:3])
        assert all("form:has" in loc.value for loc in scoped)
