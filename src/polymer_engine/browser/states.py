"""Outcome states for browser-driven acquisition.

These are deliberately separate from the parameterization state machine. A browser
workflow can fail in ways that have nothing to do with chemistry -- a changed form
field, a CAPTCHA, an expired session -- and collapsing those into ``FAILED`` would lose
the one piece of information that decides what to do next: whether a person is needed,
whether the site changed, or whether retrying is pointless.

The distinction that matters most is between:

``UI_SCHEMA_MISMATCH``
    The page is not what the automation expects. **Never guess.** The correct response
    is to stop, save a sanitised snapshot, and let a person re-derive the schema. A
    guessed selector that happens to match the wrong field submits the wrong science.

``HUMAN_INTERVENTION_REQUIRED``
    A CAPTCHA, MFA prompt or other human-verification control. These are access
    controls. They are reported, never solved, and never worked around.
"""

from __future__ import annotations

from enum import Enum


class BrowserState(str, Enum):
    """What happened during a browser interaction."""

    OK = "OK"
    NOT_LAUNCHED = "NOT_LAUNCHED"
    CREDENTIALS_MISSING = "CREDENTIALS_MISSING"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    #: A CAPTCHA/MFA/verification control. Reported, never bypassed.
    HUMAN_INTERVENTION_REQUIRED = "HUMAN_INTERVENTION_REQUIRED"
    #: The page no longer matches the expected schema. Stop rather than guess.
    UI_SCHEMA_MISMATCH = "UI_SCHEMA_MISMATCH"
    #: What the site shows does not match what was requested.
    STRUCTURE_MISMATCH = "STRUCTURE_MISMATCH"
    #: The catalogue does not contain the requested monomer, or contains it ambiguously.
    MONOMER_NOT_FOUND = "MONOMER_NOT_FOUND"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"
    NAVIGATION_FAILED = "NAVIGATION_FAILED"
    SUBMISSION_FAILED = "SUBMISSION_FAILED"
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    BROWSER_CRASHED = "BROWSER_CRASHED"
    #: Refused locally before touching the network.
    REFUSED = "REFUSED"

    @property
    def ok(self) -> bool:
        return self is BrowserState.OK

    @property
    def needs_human(self) -> bool:
        """Whether a person must act before this can progress."""
        return self in {
            BrowserState.HUMAN_INTERVENTION_REQUIRED,
            BrowserState.UI_SCHEMA_MISMATCH,
            BrowserState.REQUIRES_HUMAN_REVIEW,
            BrowserState.CREDENTIALS_MISSING,
            BrowserState.STRUCTURE_MISMATCH,
        }

    @property
    def retryable(self) -> bool:
        """Whether trying again unchanged could plausibly succeed.

        Authentication failure is *not* retryable: repeating a rejected password is how
        an account gets locked, and the password will not have changed by itself.
        """
        return self in {
            BrowserState.TIMEOUT, BrowserState.NETWORK_ERROR,
            BrowserState.BROWSER_CRASHED, BrowserState.NAVIGATION_FAILED,
        }


class JobState(str, Enum):
    """Lifecycle of one CHARMM-GUI build job, from our side of the wire."""

    QUEUED = "QUEUED"
    SUBMITTED = "SUBMITTED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    DOWNLOAD_PENDING = "DOWNLOAD_PENDING"
    DOWNLOADED = "DOWNLOADED"
    VALIDATED = "VALIDATED"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    BLOCKED = "BLOCKED"

    @property
    def terminal(self) -> bool:
        return self in {
            JobState.ERROR, JobState.VALIDATED, JobState.FAILED_VALIDATION,
            JobState.BLOCKED,
        }

    @property
    def in_flight(self) -> bool:
        """Work exists on CHARMM-GUI's side. Never resubmit while this is true."""
        return self in {
            JobState.SUBMITTED, JobState.PENDING, JobState.RUNNING,
            JobState.DOWNLOAD_PENDING,
        }


#: Documented ``/api/check_status`` values mapped to our lifecycle. An unrecognised
#: status maps to nothing and is polled again -- it is never read as completion.
API_STATUS_TO_JOB_STATE: dict[str, JobState] = {
    "pending": JobState.PENDING,
    "running": JobState.RUNNING,
    "done": JobState.DOWNLOAD_PENDING,
    "error": JobState.ERROR,
}

__all__ = ["API_STATUS_TO_JOB_STATE", "BrowserState", "JobState"]
