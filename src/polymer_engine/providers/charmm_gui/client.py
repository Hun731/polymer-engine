"""CHARMM-GUI authenticated job API.

Documented contract (verified against https://charmm-gui.org/?doc=api):

======  ==========================================  ====================================
method  path                                         purpose
======  ==========================================  ====================================
POST    ``/api/login``                               email+password -> JWT
GET     ``/api/check_status?jobid=<ID>``             ``status`` in pending|running|done|error
GET     ``/api/download?jobid=<ID>``                 ``.tgz`` archive of job results
======  ==========================================  ====================================

All endpoints except login require ``Authorization: Bearer <JWT>``.  The token is
valid for up to 12 hours.

**Job submission is deliberately absent.**  CHARMM-GUI publishes no submission
endpoint, and Polymer Builder is not mentioned in the API documentation at all.
:meth:`submit_module` therefore raises :class:`UnsupportedCapability` rather than
guessing a path.  That is a real limitation of the upstream service, and the engine
reports it as one instead of pretending the capability exists.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from polymer_engine.core.config import Secret
from polymer_engine.core.errors import (
    AuthenticationError,
    CorruptArchive,
    CredentialsMissing,
    ProviderError,
    ResponseFormatError,
    UnsupportedCapability,
)
from polymer_engine.core.logging import get_logger, register_secret
from polymer_engine.providers.base import Capability, Provider, ProviderResult
from polymer_engine.providers.http import HttpClient

logger = get_logger("providers.charmm_gui")

JobState = Literal["pending", "running", "done", "error", "unknown"]

#: Documented lifecycle values.  Anything else maps to "unknown" rather than being
#: optimistically read as success.
KNOWN_STATES: frozenset[str] = frozenset({"pending", "running", "done", "error"})

TERMINAL_STATES: frozenset[str] = frozenset({"done", "error"})

#: CHARMM-GUI states a JWT is valid for at most 12 hours.  We refresh early.
TOKEN_LIFETIME_S = 12 * 3600
TOKEN_REFRESH_MARGIN_S = 300


@dataclass(slots=True)
class JobStatus:
    """Normalised view of ``/api/check_status``."""

    job_id: str
    state: JobState
    raw_status: str | None
    last_out_file: str | None = None
    last_out_time: str | None = None
    last_out_lines: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.last_out_lines is None:
            self.last_out_lines = []

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def succeeded(self) -> bool:
        return self.state == "done"

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "raw_status": self.raw_status,
            "last_out_file": self.last_out_file,
            "last_out_time": self.last_out_time,
            "last_out_lines": self.last_out_lines[-30:],
        }


class CHARMMGUIProvider(Provider):
    """Client for the three documented CHARMM-GUI API endpoints."""

    name = "charmm_gui"
    capabilities = frozenset({Capability.JOB_LOGIN, Capability.JOB_STATUS, Capability.JOB_DOWNLOAD})

    #: Capabilities the upstream service does not publish an endpoint for.
    unsupported_capabilities = frozenset({Capability.JOB_SUBMISSION})

    BASE = "https://charmm-gui.org/api"

    def __init__(
        self,
        client: HttpClient | None = None,
        *,
        email: str | None = None,
        password: str | Secret | None = None,
        token: str | Secret | None = None,
        base_url: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(client)
        self.base = (base_url or self.BASE).rstrip("/")
        self.email = email
        self._password = password if isinstance(password, Secret) else Secret(password)
        self._token = token if isinstance(token, Secret) else Secret(token)
        self._token_acquired_at: float | None = None if not self._token else clock()
        self._clock = clock
        self._sleeper = sleeper
        if self._token:
            register_secret(self._token)
        if self._password:
            register_secret(self._password)

    # -- configuration --------------------------------------------------
    def configured(self) -> bool:
        return bool(self._token) or bool(self.email and self._password)

    @property
    def has_token(self) -> bool:
        return bool(self._token)

    @property
    def token_expired(self) -> bool:
        """Whether the cached token is past its refresh window.

        Treated as expired when we hold a token but do not know when we got it.
        """
        if not self._token:
            return True
        if self._token_acquired_at is None:
            return True
        age = self._clock() - self._token_acquired_at
        return age >= (TOKEN_LIFETIME_S - TOKEN_REFRESH_MARGIN_S)

    def health(self) -> ProviderResult:
        if not self.configured():
            return ProviderResult(
                self.name,
                "health",
                False,
                error="No CHARMM-GUI credentials or token configured",
                error_type="CredentialsMissing",
                provenance={"provider": self.name, "endpoint": self.base},
            )
        return ProviderResult(
            self.name,
            "health",
            True,
            data={
                "configured": True,
                "has_token": self.has_token,
                "can_login": bool(self.email and self._password),
                "supported": sorted(c.value for c in self.capabilities),
                "unsupported": sorted(c.value for c in self.unsupported_capabilities),
            },
            provenance={"provider": self.name, "endpoint": self.base},
        )

    # -- login ----------------------------------------------------------
    def login(self, *, force: bool = False) -> ProviderResult:
        """Exchange email+password for a JWT.

        Returns a result whose ``data`` reports only *whether* authentication
        succeeded.  The token is never placed in the result, in provenance, or in a
        log line.
        """
        self.require(Capability.JOB_LOGIN)
        return self._guard("login", lambda: self._login(force))

    def _login(self, force: bool) -> ProviderResult:
        if self._token and not force and not self.token_expired:
            return ProviderResult(
                self.name, "login", True, data={"authenticated": True, "reused_cached_token": True},
                provenance={"provider": self.name, "endpoint": f"{self.base}/login"},
            )
        if not self.email or not self._password:
            raise CredentialsMissing(
                "CHARMM-GUI login needs both an email and a password",
                hint="set CHARMM_GUI_EMAIL and CHARMM_GUI_PASSWORD",
            )
        url = f"{self.base}/login"
        payload = self.client.post_json(
            url, {"email": self.email, "password": self._password.reveal()}, use_cache=False
        )
        token = self._extract_token(payload, url)
        self._token = Secret(token)
        self._token_acquired_at = self._clock()
        register_secret(self._token)
        logger.info("CHARMM-GUI authentication succeeded")
        return ProviderResult(
            self.name, "login", True, data={"authenticated": True, "reused_cached_token": False},
            provenance={"provider": self.name, "endpoint": url},
        )

    @staticmethod
    def _extract_token(payload: Any, url: str) -> str:
        if not isinstance(payload, dict):
            raise ResponseFormatError("CHARMM-GUI login response is not a JSON object", url=url)
        for key in ("token", "jwt", "access_token"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # A 200 with no token means the service rejected the credentials without
        # using an HTTP error status.  That is an authentication failure.
        raise AuthenticationError(
            "CHARMM-GUI login returned no token",
            url=url,
            response_keys=sorted(payload),
        )

    def _auth_headers(self) -> dict[str, str]:
        """Return the bearer header, logging in first if we can and must."""
        if not self._token or self.token_expired:
            if self.email and self._password:
                self._login(force=True)
            elif not self._token:
                raise CredentialsMissing(
                    "No CHARMM-GUI token available and no credentials to obtain one",
                    hint="set CHARMM_GUI_TOKEN, or CHARMM_GUI_EMAIL and CHARMM_GUI_PASSWORD",
                )
        token = self._token.reveal()
        if not token:  # pragma: no cover - defensive
            raise CredentialsMissing("CHARMM-GUI token is empty")
        return {"Authorization": f"Bearer {token}"}

    # -- status ---------------------------------------------------------
    def job_status(self, job_id: str) -> ProviderResult:
        self.require(Capability.JOB_STATUS)
        return self._guard("job_status", lambda: self._job_status(job_id))

    def _job_status(self, job_id: str) -> ProviderResult:
        job_id = _require_job_id(job_id)
        url = f"{self.base}/check_status"
        payload = self.client.get_json(
            url, params={"jobid": job_id}, headers=self._auth_headers(), use_cache=False
        )
        status = self._normalise_status(job_id, payload, url)
        return ProviderResult(
            self.name,
            "job_status",
            True,
            records=[status.as_dict()],
            data=payload if isinstance(payload, dict) else {},
            provenance={"provider": self.name, "endpoint": url, "job_id": job_id,
                        **({"request": p.as_dict()} if (p := self.client.last_provenance) else {})},
        )

    @staticmethod
    def _normalise_status(job_id: str, payload: Any, url: str) -> JobStatus:
        if not isinstance(payload, dict):
            raise ResponseFormatError("CHARMM-GUI status response is not a JSON object", url=url, job_id=job_id)
        raw = payload.get("status")
        raw_text = raw.strip().lower() if isinstance(raw, str) else None
        # An unrecognised status is "unknown", never assumed to be success.
        state: JobState = raw_text if raw_text in KNOWN_STATES else "unknown"  # type: ignore[assignment]
        lines = payload.get("lastOutLines")
        if isinstance(lines, str):
            lines = lines.splitlines()
        elif not isinstance(lines, list):
            lines = []
        return JobStatus(
            job_id=job_id,
            state=state,
            raw_status=raw if isinstance(raw, str) else None,
            last_out_file=payload.get("lastOutFile") if isinstance(payload.get("lastOutFile"), str) else None,
            last_out_time=payload.get("lastOutTime") if isinstance(payload.get("lastOutTime"), str) else None,
            last_out_lines=[str(x) for x in lines],
        )

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval_s: float = 30.0,
        timeout_s: float = 7200.0,
        max_polls: int | None = None,
    ) -> ProviderResult:
        """Poll ``check_status`` until the job reaches a terminal state.

        Returns ``ok=False`` on timeout rather than raising, so a campaign can record
        "still running" as a legitimate, resumable outcome.  An ``unknown`` status is
        polled again -- it is never treated as completion.
        """
        self.require(Capability.JOB_STATUS)
        return self._guard("wait_for_job", lambda: self._wait(job_id, poll_interval_s, timeout_s, max_polls))

    def _wait(self, job_id: str, poll_interval_s: float, timeout_s: float, max_polls: int | None) -> ProviderResult:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        started = self._clock()
        polls = 0
        last: JobStatus | None = None
        while True:
            result = self._job_status(job_id)
            last = self._normalise_status(job_id, result.data, f"{self.base}/check_status")
            polls += 1
            if last.terminal:
                return ProviderResult(
                    self.name,
                    "wait_for_job",
                    last.succeeded,
                    records=[last.as_dict()],
                    error=None if last.succeeded else f"CHARMM-GUI job {job_id} finished in state {last.state!r}",
                    error_type=None if last.succeeded else "JobFailed",
                    provenance={"provider": self.name, "job_id": job_id, "polls": polls},
                )
            elapsed = self._clock() - started
            if elapsed >= timeout_s or (max_polls is not None and polls >= max_polls):
                return ProviderResult(
                    self.name,
                    "wait_for_job",
                    False,
                    records=[last.as_dict()],
                    error=f"CHARMM-GUI job {job_id} did not finish within {timeout_s}s (state={last.state!r})",
                    error_type="JobStillRunning",
                    provenance={"provider": self.name, "job_id": job_id, "polls": polls, "elapsed_s": elapsed},
                )
            self._sleeper(poll_interval_s)

    # -- download -------------------------------------------------------
    def download_job(
        self,
        job_id: str,
        destination: str | Path,
        *,
        expected_sha256: str | None = None,
        verify_archive: bool = True,
    ) -> ProviderResult:
        """Download the job archive and confirm it is a readable ``.tgz``.

        A truncated or HTML-error payload saved under a ``.tgz`` name is a classic way
        for a broken download to look like a valid scientific system.  With
        ``verify_archive`` the file is opened as a tar before we claim success.
        """
        self.require(Capability.JOB_DOWNLOAD)
        return self._guard(
            "job_download", lambda: self._download(job_id, destination, expected_sha256, verify_archive)
        )

    def _download(
        self, job_id: str, destination: str | Path, expected_sha256: str | None, verify_archive: bool
    ) -> ProviderResult:
        job_id = _require_job_id(job_id)
        url = f"{self.base}/download"
        destination = Path(destination)
        path = self.client.download(
            f"{url}?jobid={job_id}",
            destination,
            headers=self._auth_headers(),
            expected_sha256=expected_sha256,
        )
        if verify_archive:
            self._verify_tar(path, job_id)
        from polymer_engine.core.provenance import sha256_file

        return ProviderResult(
            self.name,
            "job_download",
            True,
            artifacts=[str(path)],
            data={"size_bytes": path.stat().st_size, "sha256": sha256_file(path)},
            provenance={
                "provider": self.name,
                "endpoint": url,
                "job_id": job_id,
                "sha256": sha256_file(path),
                **({"request": p.as_dict()} if (p := self.client.last_provenance) else {}),
            },
        )

    @staticmethod
    def _verify_tar(path: Path, job_id: str) -> None:
        import tarfile

        if path.stat().st_size == 0:
            path.unlink(missing_ok=True)
            raise CorruptArchive("CHARMM-GUI download is empty", job_id=job_id, path=str(path))
        try:
            with tarfile.open(path, "r:*") as tf:
                if tf.next() is None:
                    raise CorruptArchive("CHARMM-GUI archive contains no members", job_id=job_id, path=str(path))
        except tarfile.TarError as exc:
            raise CorruptArchive(
                f"CHARMM-GUI download is not a readable tar archive: {exc}", job_id=job_id, path=str(path)
            ) from exc

    # -- explicitly unsupported ----------------------------------------
    def submit_module(self, *_args: Any, **_kwargs: Any) -> ProviderResult:
        """Not available.  CHARMM-GUI publishes no job-submission endpoint.

        Build the system in the web interface (or with CHARMM-GUI's own scripts), then
        hand the engine the job id.
        """
        raise UnsupportedCapability(
            "CHARMM-GUI does not publish a job-submission endpoint; "
            "create the job in the web interface and supply its job id",
            provider=self.name,
            capability=Capability.JOB_SUBMISSION.value,
            documented_endpoints=["/api/login", "/api/check_status", "/api/download"],
        )


def _require_job_id(job_id: str) -> str:
    cleaned = (job_id or "").strip()
    if not cleaned:
        raise ProviderError("A CHARMM-GUI job id is required")
    if not cleaned.isalnum():
        # Job ids are numeric/alphanumeric; anything else risks query injection.
        raise ProviderError("CHARMM-GUI job id must be alphanumeric", job_id=cleaned)
    return cleaned


__all__ = ["KNOWN_STATES", "TERMINAL_STATES", "TOKEN_LIFETIME_S", "CHARMMGUIProvider", "JobStatus"]
