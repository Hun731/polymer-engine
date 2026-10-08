"""CHARMM-GUI credentials: from the environment, into process memory, nowhere else.

The rules this module exists to enforce (§15, §16):

* the password is read from the environment and never from a CLI argument, a config
  file, or a prompt whose contents could be shell-history;
* it is wrapped in :class:`Secret`, so interpolating it into a log line or an f-string
  yields a mask rather than the value;
* it is registered with the log redactor the moment it is read, so even a third-party
  library that echoes it gets scrubbed;
* it is **never serialised** -- not into the worker protocol, not into provenance, not
  into a manifest. The worker subprocess reads it from its own inherited environment,
  so the password never appears in a JSON message at all.

The last point is the reason :meth:`worker_env` exists instead of a ``password`` field
on any request object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from polymer_engine.core.config import Secret
from polymer_engine.core.logging import get_logger, register_secret

logger = get_logger("browser.credentials")

EMAIL_VAR = "CHARMM_GUI_EMAIL"
PASSWORD_VAR = "CHARMM_GUI_PASSWORD"
TYPING_DELAY_VAR = "CHARMM_GUI_TYPING_DELAY_MS"
LIVE_TEST_VAR = "CHARMM_GUI_LIVE_TEST"

#: Milliseconds between keystrokes. Modest by default: fast enough to be usable, slow
#: enough that a form with per-character validation or a JS-driven state machine keeps
#: up. Typing a password instantly is one of the ways a paste-like injection is
#: detected and rejected.
DEFAULT_TYPING_DELAY_MS = 40


@dataclass
class Credentials:
    """An email and a password that refuses to render itself."""

    email: str | None = None
    password: Secret = field(default_factory=lambda: Secret(None))
    typing_delay_ms: int = DEFAULT_TYPING_DELAY_MS

    @property
    def complete(self) -> bool:
        return bool(self.email) and bool(self.password)

    @property
    def missing(self) -> list[str]:
        absent = []
        if not self.email:
            absent.append(EMAIL_VAR)
        if not self.password:
            absent.append(PASSWORD_VAR)
        return absent

    def worker_env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """Environment for the browser worker subprocess.

        This is the *only* channel the password travels on. It goes into the child's
        environment, which means it never becomes part of a JSON message that could be
        logged, echoed in a traceback, or written to a diagnostic snapshot.
        """
        env = dict(base if base is not None else os.environ)
        if self.email:
            env[EMAIL_VAR] = self.email
        secret = self.password.reveal()
        if secret:
            env[PASSWORD_VAR] = secret
        env[TYPING_DELAY_VAR] = str(self.typing_delay_ms)
        return env

    def as_dict(self) -> dict[str, Any]:
        """Safe to log, by construction: reports presence, never content."""
        return {
            "email_set": bool(self.email),
            "password_set": bool(self.password),
            "typing_delay_ms": self.typing_delay_ms,
            "complete": self.complete,
            "missing": self.missing,
        }

    def __repr__(self) -> str:
        return f"Credentials(email_set={bool(self.email)}, password_set={bool(self.password)})"

    __str__ = __repr__


def _typing_delay(env: dict[str, str]) -> int:
    raw = env.get(TYPING_DELAY_VAR, "").strip()
    if not raw:
        return DEFAULT_TYPING_DELAY_MS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using the %d ms default",
            TYPING_DELAY_VAR, raw, DEFAULT_TYPING_DELAY_MS,
        )
        return DEFAULT_TYPING_DELAY_MS
    # A negative delay is meaningless; an enormous one would hang a login silently.
    return max(0, min(value, 1000))


def from_environment(env: dict[str, str] | None = None) -> Credentials:
    """Read credentials from the environment, registering the password for redaction.

    Returns an incomplete :class:`Credentials` rather than raising when the variables
    are unset: "no credentials" is a legitimate state that the engine reports as
    ``CREDENTIALS_MISSING``, not an error in the code path that asked.
    """
    source = dict(os.environ if env is None else env)
    email = (source.get(EMAIL_VAR) or "").strip() or None
    raw_password = source.get(PASSWORD_VAR) or None
    password = Secret(raw_password)
    if raw_password:
        # Registered before anything else can touch it.
        register_secret(password)
    creds = Credentials(email=email, password=password, typing_delay_ms=_typing_delay(source))
    logger.debug("CHARMM-GUI credentials: %s", creds.as_dict())
    return creds


class NoTerminal(RuntimeError):
    """There is no terminal to prompt on, so no secure way to ask for a password."""


def prompt(
    env: dict[str, str] | None = None, *, allow_prompt: bool = True,
) -> Credentials:
    """Take credentials from the environment, asking on the terminal for what is absent.

    The password is read with :func:`getpass.getpass`, which turns off terminal echo and
    reads straight from the controlling terminal -- so it never reaches the shell, never
    enters history, and never becomes visible in ``ps``.

    If there is no terminal, this **raises** rather than falling back to
    :func:`input`. A silent fallback would echo the password to the screen and into
    whatever is capturing the session, which is precisely the outcome the prompt exists
    to avoid. A non-interactive caller must supply the environment variables instead.
    """
    import getpass
    import sys

    source = dict(os.environ if env is None else env)
    email = (source.get(EMAIL_VAR) or "").strip() or None
    raw_password = source.get(PASSWORD_VAR) or None

    if (email is None or raw_password is None) and allow_prompt:
        if not sys.stdin.isatty():
            raise NoTerminal(
                "no terminal is attached, so there is no secure way to ask for a "
                f"password. Run this from an interactive shell, or set {EMAIL_VAR} and "
                f"{PASSWORD_VAR} in the environment. Falling back to an echoing prompt "
                f"would print the password to the screen"
            )
        if email is None:
            email = input("CHARMM-GUI email: ").strip() or None
        if raw_password is None:
            # Echo off, read from the controlling terminal, never from a pipe.
            raw_password = getpass.getpass("CHARMM-GUI password (not echoed): ") or None

    password = Secret(raw_password)
    if raw_password:
        register_secret(password)
    return Credentials(email=email, password=password,
                       typing_delay_ms=_typing_delay(source))


def live_test_enabled(env: dict[str, str] | None = None) -> bool:
    """Whether the opt-in live-website tests may run.

    Off unless explicitly switched on, so ordinary CI never needs credentials and never
    touches the live service.
    """
    source = os.environ if env is None else env
    return (source.get(LIVE_TEST_VAR) or "").strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "DEFAULT_TYPING_DELAY_MS", "EMAIL_VAR", "LIVE_TEST_VAR", "PASSWORD_VAR",
    "TYPING_DELAY_VAR", "Credentials", "NoTerminal", "from_environment",
    "live_test_enabled", "prompt",
]
