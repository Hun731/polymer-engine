"""An authenticated browser session, and the login flow that establishes it.

The login flow (§17) is the part of this subsystem with the strictest rules, and they
are enforced structurally rather than by convention:

* the password is typed with genuine keystrokes, one character at a time, through
  :meth:`Session.type_secret`, which names an environment variable instead of carrying
  a value. There is no code path in this module -- or in the worker -- that accepts a
  password as an argument;
* no clipboard is touched. There is no ``navigator.clipboard`` call, no ``Control+V``,
  and ``fill()`` is not used on any field;
* a CAPTCHA or MFA prompt ends the flow as ``HUMAN_INTERVENTION_REQUIRED``. It is
  detected so a person can be asked, and never solved, worked around or retried;
* a rejected password is **not** retried. Repeating a rejected credential is how an
  account gets locked, and it will not have changed by itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.browser.credentials import (
    PASSWORD_VAR,
    Credentials,
    from_environment,
)
from polymer_engine.browser.driver import PageDriver, WorkerDriver
from polymer_engine.browser.selectors import (
    HUMAN_VERIFICATION_MARKERS,
    MFA_TEXT_MARKERS,
    FormSchema,
    login_schema,
)
from polymer_engine.browser.states import BrowserState
from polymer_engine.core.logging import get_logger

logger = get_logger("browser.session")

CHARMM_GUI_BASE = "https://charmm-gui.org"
#: Path of the sign-in page, relative to whatever base the session was given. Kept
#: relative so a session pointed at a staging or fixture host logs in against *that*
#: host rather than silently reaching for the production site.
LOGIN_PATH = "/?doc=sign"
LOGIN_URL = f"{CHARMM_GUI_BASE}{LOGIN_PATH}"

#: Text that, on a page we have just submitted a login to, means the credentials were
#: rejected. Checked only *after* submission, and only to distinguish a rejected
#: password from a network failure -- the two need opposite responses.
REJECTION_MARKERS: tuple[str, ...] = (
    "incorrect password", "invalid password", "login failed",
    "authentication failed", "wrong password", "user not found",
    "invalid email", "please try again",
)


@dataclass
class ActionResult:
    """The outcome of one browser action, with the state that decides what happens next."""

    state: BrowserState
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.state.ok

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "ok": self.ok, "detail": self.detail,
                "needs_human": self.state.needs_human, "data": dict(self.data)}


class Session:
    """One browser, one login, one page at a time (§54).

    Deliberately single-session: CHARMM-GUI is a shared academic service, and running
    many concurrent authenticated sessions against it is the behaviour that gets an
    account rate-limited. One session, one build at a time.
    """

    def __init__(
        self,
        driver: PageDriver | None = None,
        *,
        credentials: Credentials | None = None,
        headless: bool = True,
        base_url: str = CHARMM_GUI_BASE,
        downloads_dir: str | Path | None = None,
        timeout_ms: float = 30000.0,
    ) -> None:
        self.credentials = credentials if credentials is not None else from_environment()
        self.headless = headless
        self.base_url = base_url.rstrip("/")
        self.downloads_dir = Path(downloads_dir) if downloads_dir else None
        self.timeout_ms = timeout_ms
        self._driver = driver
        self._owns_driver = driver is None
        self._launched = False
        self._authenticated = False
        self.login_schema: FormSchema = login_schema()

    # -- lifecycle ------------------------------------------------------
    @property
    def driver(self) -> PageDriver:
        if self._driver is None:
            self._driver = WorkerDriver(env=self.credentials.worker_env())
        return self._driver

    def launch(self) -> ActionResult:
        if self._launched:
            return ActionResult(BrowserState.OK, "already launched")
        payload: dict[str, Any] = {"headless": self.headless, "timeout_ms": self.timeout_ms}
        if self.downloads_dir is not None:
            self.downloads_dir.mkdir(parents=True, exist_ok=True)
            payload["downloads_dir"] = str(self.downloads_dir)
        response = self.driver.send("launch", **payload)
        if not response.get("ok"):
            return ActionResult(BrowserState.BROWSER_CRASHED,
                                response.get("error", "browser did not launch"))
        self._launched = True
        logger.info("browser launched (headless=%s, chromium %s)",
                    self.headless, response.get("browser_version"))
        return ActionResult(BrowserState.OK, "launched", response)

    def close(self) -> None:
        if self._driver is not None and self._owns_driver:
            self._driver.close()
        self._driver = None if self._owns_driver else self._driver
        self._launched = False
        self._authenticated = False

    def __enter__(self) -> Session:
        self.launch()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- observation ----------------------------------------------------
    def navigate(self, url: str) -> ActionResult:
        if not self._launched:
            launched = self.launch()
            if not launched.ok:
                return launched
        target = url if url.startswith("http") else f"{self.base_url}{url}"
        response = self.driver.send("goto", url=target)
        if not response.get("ok"):
            return ActionResult(_state_of(response, BrowserState.NAVIGATION_FAILED),
                                response.get("error", "navigation failed"), response)
        return ActionResult(BrowserState.OK, f"at {response.get('url')}", response)

    def current_url(self) -> str:
        return str(self.driver.send("url").get("url", ""))

    def page_text(self, max_chars: int = 20000) -> str:
        return str(self.driver.send("text", max_chars=max_chars).get("text", ""))

    def snapshot(self, max_chars: int = 400000) -> dict[str, Any]:
        """A sanitised structural snapshot.  Values are redacted in the worker."""
        return self.driver.send("snapshot", max_chars=max_chars)

    def screenshot(self, path: str | Path) -> ActionResult:
        response = self.driver.send("screenshot", path=str(path))
        return ActionResult(BrowserState.OK if response.get("ok")
                            else BrowserState.NAVIGATION_FAILED,
                            response.get("error", ""), response)

    def human_verification_present(self) -> tuple[bool, list[str]]:
        """Whether a CAPTCHA or MFA challenge is on the page.  Detection only."""
        response = self.driver.send(
            "human_verification",
            captcha_markers=list(HUMAN_VERIFICATION_MARKERS),
            mfa_markers=list(MFA_TEXT_MARKERS),
        )
        return bool(response.get("detected")), list(response.get("markers", []))

    # -- typing ---------------------------------------------------------
    def type_text(self, key: str, value: str) -> dict[str, Any]:
        spec = self.login_schema.require(key)
        return self.driver.send(
            "type", key=key, text=value,
            delay_ms=self.credentials.typing_delay_ms,
            locators=[loc.as_dict() for loc in spec.ordered()],
        )

    def type_secret(self, key: str, env_var: str, schema: FormSchema | None = None) -> dict[str, Any]:
        """Type a credential the engine never holds as a string in this call.

        Only the *name* of an environment variable crosses the process boundary. The
        worker reads the value from its own environment and types it character by
        character; what comes back is a character count, not the value.
        """
        spec = (schema or self.login_schema).require(key)
        return self.driver.send(
            "type_secret", key=key, env_var=env_var,
            delay_ms=self.credentials.typing_delay_ms,
            locators=[loc.as_dict() for loc in spec.ordered()],
        )

    # -- login ----------------------------------------------------------
    @property
    def authenticated(self) -> bool:
        return self._authenticated

    def login(self, *, url: str | None = None, settle_s: float = 1.5) -> ActionResult:
        """Establish an authenticated session, or explain exactly why not."""
        if self._authenticated:
            return ActionResult(BrowserState.OK, "already authenticated")
        if not self.credentials.complete:
            return ActionResult(
                BrowserState.CREDENTIALS_MISSING,
                "set " + " and ".join(self.credentials.missing)
                + " in the environment; they are never read from a file or an argument",
                {"missing": self.credentials.missing},
            )

        arrived = self.navigate(url or LOGIN_PATH)
        if not arrived.ok:
            return arrived

        # Before touching the form: is a human challenge already on the page?
        blocked, markers = self.human_verification_present()
        if blocked:
            return ActionResult(
                BrowserState.HUMAN_INTERVENTION_REQUIRED,
                "the login page presents a human-verification challenge; a person must "
                "complete it. It is not bypassed.",
                {"markers": markers, "url": self.current_url()},
            )

        typed_email = self.type_text("email", self.credentials.email or "")
        if not typed_email.get("ok"):
            return ActionResult(_state_of(typed_email, BrowserState.UI_SCHEMA_MISMATCH),
                                typed_email.get("error", "could not enter the email"),
                                _sanitise(typed_email))

        typed_password = self.type_secret("password", PASSWORD_VAR)
        if not typed_password.get("ok"):
            return ActionResult(
                _state_of(typed_password, BrowserState.UI_SCHEMA_MISMATCH),
                typed_password.get("error", "could not enter the password"),
                _sanitise(typed_password),
            )
        logger.info("password entered by keyboard (%s characters accepted by the field)",
                    typed_password.get("chars_received"))

        submitted = self.driver.send(
            "click", key="submit",
            locators=[loc.as_dict() for loc in self.login_schema.require("submit").ordered()],
        )
        if not submitted.get("ok"):
            return ActionResult(_state_of(submitted, BrowserState.UI_SCHEMA_MISMATCH),
                                submitted.get("error", "could not submit the login form"),
                                _sanitise(submitted))

        if settle_s:
            time.sleep(settle_s)
        return self._verify_authenticated()

    def _verify_authenticated(self) -> ActionResult:
        """Decide whether the session is really logged in.

        A page that still shows a password field has not accepted the login, whatever
        else it says. That structural check is the primary signal; the text markers only
        distinguish *why*, so that a rejected credential is never retried as if it were
        a network blip.
        """
        blocked, markers = self.human_verification_present()
        if blocked:
            return ActionResult(
                BrowserState.HUMAN_INTERVENTION_REQUIRED,
                "a human-verification challenge appeared after submitting the login",
                {"markers": markers},
            )
        fields = self.driver.send("form_fields")
        controls = fields.get("controls", []) if fields.get("ok") else []
        password_visible = any(
            (c.get("type") or "") == "password" and c.get("visible") for c in controls
        )
        text = self.page_text(6000).lower()
        rejected = [marker for marker in REJECTION_MARKERS if marker in text]
        if password_visible or rejected:
            return ActionResult(
                BrowserState.AUTHENTICATION_FAILED,
                "CHARMM-GUI did not accept the credentials"
                + (f" ({rejected[0]})" if rejected else
                   " (the login form is still present)")
                + ". Not retried: repeating a rejected password risks locking the account.",
                {"markers": rejected, "url": self.current_url()},
            )
        self._authenticated = True
        logger.info("CHARMM-GUI session authenticated")
        return ActionResult(BrowserState.OK, "authenticated", {"url": self.current_url()})


def _state_of(response: dict[str, Any], default: BrowserState) -> BrowserState:
    raw = response.get("state")
    if isinstance(raw, str):
        try:
            return BrowserState(raw)
        except ValueError:
            return default
    return default


def _sanitise(response: dict[str, Any]) -> dict[str, Any]:
    """Drop anything that could carry a typed value out of a diagnostic payload."""
    return {k: v for k, v in response.items()
            if k not in {"value", "text", "traceback"}}


__all__ = [
    "CHARMM_GUI_BASE", "LOGIN_PATH", "LOGIN_URL", "REJECTION_MARKERS",
    "ActionResult", "Session",
]
