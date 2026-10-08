"""CHARMM-GUI adapter: authentication, polling, download, and honest limits."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from polymer_engine.core.errors import (
    UnsupportedCapability,
)
from polymer_engine.providers.base import Capability
from polymer_engine.providers.charmm_gui import CHARMMGUIProvider
from polymer_engine.providers.charmm_gui.client import TOKEN_LIFETIME_S
from polymer_engine.providers.http import HttpClient
from polymer_engine.providers.testing import FixtureTransport, StubResponse

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "providers"
TOKEN = json.loads((FIXTURES / "charmm_gui_login.json").read_text())["token"]


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def make_provider(client: HttpClient, **kwargs) -> CHARMMGUIProvider:
    kwargs.setdefault("email", "chem@example.org")
    kwargs.setdefault("password", "correct-horse-battery")
    return CHARMMGUIProvider(client, **kwargs)


# ==========================================================================
# Configuration and capability honesty
# ==========================================================================
def test_unconfigured_provider_reports_missing_credentials(client: HttpClient) -> None:
    provider = CHARMMGUIProvider(client)
    assert provider.configured() is False
    health = provider.health()
    assert health.ok is False
    assert health.error_type == "CredentialsMissing"


def test_token_only_configuration_is_valid(client: HttpClient) -> None:
    provider = CHARMMGUIProvider(client, token=TOKEN)
    assert provider.configured() is True
    assert provider.has_token is True


def test_job_submission_is_declared_unsupported(client: HttpClient) -> None:
    provider = make_provider(client)
    assert provider.supports(Capability.JOB_SUBMISSION) is False
    assert Capability.JOB_SUBMISSION in provider.unsupported_capabilities


def test_submit_module_raises_rather_than_guessing_an_endpoint(
    client: HttpClient, transport: FixtureTransport
) -> None:
    with pytest.raises(UnsupportedCapability) as excinfo:
        make_provider(client).submit_module("polymer_builder", {})
    assert transport.call_count == 0, "must not attempt an undocumented request"
    assert excinfo.value.context["documented_endpoints"] == [
        "/api/login",
        "/api/check_status",
        "/api/download",
    ]


# ==========================================================================
# Login
# ==========================================================================
def test_successful_login(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add_json("/api/login", fixture("charmm_gui_login.json"))
    provider = make_provider(client)
    result = provider.login()
    assert result.ok
    assert result.data == {"authenticated": True, "reused_cached_token": False}
    assert provider.has_token


def test_login_never_leaks_the_token(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add_json("/api/login", fixture("charmm_gui_login.json"))
    result = make_provider(client).login()
    serialised = json.dumps(result.as_dict(), default=str)
    assert TOKEN not in serialised
    assert "authenticated" in serialised


def test_invalid_credentials_return_401(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("/api/login", StubResponse.json({"error": "invalid"}, status=401))
    result = make_provider(client).login()
    assert result.ok is False
    assert result.error_type == "AuthenticationError"


def test_forbidden_login_returns_403(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("/api/login", StubResponse.json({"error": "forbidden"}, status=403))
    result = make_provider(client).login()
    assert result.error_type == "AuthorizationError"


def test_200_without_a_token_is_an_authentication_failure(
    client: HttpClient, transport: FixtureTransport
) -> None:
    transport.add_json("/api/login", {"message": "ok"})
    result = make_provider(client).login()
    assert result.ok is False
    assert result.error_type == "AuthenticationError"


def test_malformed_login_json(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("/api/login", StubResponse.text("<html>oops</html>"))
    result = make_provider(client).login()
    assert result.ok is False
    assert result.error_type == "ResponseFormatError"


def test_login_without_credentials_fails_before_any_request(
    client: HttpClient, transport: FixtureTransport
) -> None:
    result = CHARMMGUIProvider(client, email="a@b.c").login()
    assert result.ok is False
    assert result.error_type == "CredentialsMissing"
    assert transport.call_count == 0


def test_login_rate_limited(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("/api/login", StubResponse.json({}, status=429))
    assert make_provider(client).login().error_type == "RateLimitError"


def test_login_server_error(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("/api/login", StubResponse.json({}, status=502))
    assert make_provider(client).login().error_type == "HttpStatusError"


def test_cached_token_is_reused(client: HttpClient, transport: FixtureTransport) -> None:
    provider = CHARMMGUIProvider(client, email="a@b.c", password="pw", token=TOKEN)
    result = provider.login()
    assert result.data["reused_cached_token"] is True
    assert transport.call_count == 0


def test_expired_token_triggers_a_fresh_login(transport: FixtureTransport) -> None:
    now = {"t": 0.0}
    http = HttpClient(transport=transport, sleeper=lambda s: None)
    provider = CHARMMGUIProvider(
        http, email="a@b.c", password="pw", token="stale-token", clock=lambda: now["t"]
    )
    assert provider.token_expired is False
    now["t"] = TOKEN_LIFETIME_S
    assert provider.token_expired is True

    transport.add_json("/api/login", fixture("charmm_gui_login.json"))
    transport.add_json("/api/check_status", fixture("charmm_gui_status_done.json"))
    result = provider.job_status("1234")
    assert result.ok
    assert any("/api/login" in u for u in transport.urls()), "an expired token must be refreshed"


def test_token_without_a_known_acquisition_time_is_treated_as_expired(transport: FixtureTransport) -> None:
    http = HttpClient(transport=transport, sleeper=lambda s: None)
    provider = CHARMMGUIProvider(http, token=TOKEN)
    provider._token_acquired_at = None
    assert provider.token_expired is True


# ==========================================================================
# Job status
# ==========================================================================
@pytest.mark.parametrize(
    "name,expected,terminal",
    [
        ("charmm_gui_status_pending.json", "pending", False),
        ("charmm_gui_status_running.json", "running", False),
        ("charmm_gui_status_done.json", "done", True),
        ("charmm_gui_status_error.json", "error", True),
    ],
)
def test_documented_job_states(
    client: HttpClient, transport: FixtureTransport, name: str, expected: str, terminal: bool
) -> None:
    transport.add_json("/api/check_status", fixture(name))
    result = CHARMMGUIProvider(client, token=TOKEN).job_status("1234")
    assert result.ok
    assert result.records[0]["state"] == expected
    from polymer_engine.providers.charmm_gui.client import TERMINAL_STATES

    assert (expected in TERMINAL_STATES) is terminal


def test_unrecognised_status_becomes_unknown_not_success(
    client: HttpClient, transport: FixtureTransport
) -> None:
    transport.add_json("/api/check_status", {"status": "finalising"})
    result = CHARMMGUIProvider(client, token=TOKEN).job_status("1234")
    assert result.records[0]["state"] == "unknown"


def test_status_401_is_reported_as_authentication(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("/api/check_status", StubResponse.json({}, status=401))
    result = CHARMMGUIProvider(client, token=TOKEN).job_status("1234")
    assert result.error_type == "AuthenticationError"


def test_status_403_is_reported_as_authorization(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("/api/check_status", StubResponse.json({}, status=403))
    result = CHARMMGUIProvider(client, token=TOKEN).job_status("1234")
    assert result.error_type == "AuthorizationError"


def test_status_429(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("/api/check_status", StubResponse.json({}, status=429))
    assert CHARMMGUIProvider(client, token=TOKEN).job_status("1234").error_type == "RateLimitError"


def test_status_5xx(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("/api/check_status", StubResponse.json({}, status=503))
    assert CHARMMGUIProvider(client, token=TOKEN).job_status("1234").error_type == "HttpStatusError"


def test_status_malformed_json(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("/api/check_status", StubResponse.text("not json at all"))
    assert CHARMMGUIProvider(client, token=TOKEN).job_status("1234").error_type == "ResponseFormatError"


def test_status_requires_a_job_id(client: HttpClient, transport: FixtureTransport) -> None:
    result = CHARMMGUIProvider(client, token=TOKEN).job_status("")
    assert result.ok is False
    assert transport.call_count == 0


def test_non_alphanumeric_job_id_is_rejected(client: HttpClient, transport: FixtureTransport) -> None:
    result = CHARMMGUIProvider(client, token=TOKEN).job_status("1234&admin=1")
    assert result.ok is False
    assert transport.call_count == 0


def test_authorization_header_is_sent(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add_json("/api/check_status", fixture("charmm_gui_status_done.json"))
    CHARMMGUIProvider(client, token=TOKEN).job_status("1234")
    assert transport.requests[0].headers["Authorization"] == f"Bearer {TOKEN}"


def test_status_without_any_credential_fails_closed(client: HttpClient, transport: FixtureTransport) -> None:
    result = CHARMMGUIProvider(client).job_status("1234")
    assert result.ok is False
    assert result.error_type == "CredentialsMissing"
    assert transport.call_count == 0


# ==========================================================================
# Polling
# ==========================================================================
def test_wait_polls_until_done(transport: FixtureTransport) -> None:
    slept: list[float] = []
    http = HttpClient(transport=transport, sleeper=lambda s: None)
    transport.add(
        "/api/check_status",
        [
            StubResponse.json(fixture("charmm_gui_status_pending.json")),
            StubResponse.json(fixture("charmm_gui_status_running.json")),
            StubResponse.json(fixture("charmm_gui_status_done.json")),
        ],
    )
    provider = CHARMMGUIProvider(http, token=TOKEN, sleeper=slept.append)
    result = provider.wait_for_job("1234", poll_interval_s=5.0)
    assert result.ok
    assert result.provenance["polls"] == 3
    assert slept == [5.0, 5.0]


def test_wait_reports_job_error_as_failure(transport: FixtureTransport) -> None:
    http = HttpClient(transport=transport, sleeper=lambda s: None)
    transport.add_json("/api/check_status", fixture("charmm_gui_status_error.json"))
    result = CHARMMGUIProvider(http, token=TOKEN, sleeper=lambda s: None).wait_for_job("1234")
    assert result.ok is False
    assert result.error_type == "JobFailed"


def test_wait_times_out_without_claiming_completion(transport: FixtureTransport) -> None:
    now = {"t": 0.0}

    def sleeper(seconds: float) -> None:
        now["t"] += seconds

    http = HttpClient(transport=transport, sleeper=lambda s: None)
    transport.add_json("/api/check_status", fixture("charmm_gui_status_running.json"))
    provider = CHARMMGUIProvider(http, token=TOKEN, clock=lambda: now["t"], sleeper=sleeper)
    result = provider.wait_for_job("1234", poll_interval_s=10.0, timeout_s=30.0)
    assert result.ok is False
    assert result.error_type == "JobStillRunning"
    assert result.records[0]["state"] == "running"


def test_wait_never_treats_unknown_as_complete(transport: FixtureTransport) -> None:
    http = HttpClient(transport=transport, sleeper=lambda s: None)
    transport.add_json("/api/check_status", {"status": "who-knows"})
    provider = CHARMMGUIProvider(http, token=TOKEN, sleeper=lambda s: None)
    result = provider.wait_for_job("1234", poll_interval_s=1.0, max_polls=3)
    assert result.ok is False
    assert result.error_type == "JobStillRunning"


# ==========================================================================
# Download
# ==========================================================================
def make_tgz(path: Path) -> bytes:
    source = path / "src"
    source.mkdir(parents=True, exist_ok=True)
    (source / "step5_input.gro").write_text("system\n1\n    1POL C1 1 0.0 0.0 0.0\n2.0 2.0 2.0\n")
    archive = path / "job.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(source, arcname="charmm-gui-1234")
    return archive.read_bytes()


def test_successful_download_verifies_the_archive(
    client: HttpClient, transport: FixtureTransport, tmp_path: Path
) -> None:
    payload = make_tgz(tmp_path)
    transport.add("/api/download", StubResponse(body=payload))
    target = tmp_path / "downloaded.tgz"
    result = CHARMMGUIProvider(client, token=TOKEN).download_job("1234", target)
    assert result.ok
    assert target.exists()
    assert result.data["sha256"]
    assert result.provenance["sha256"] == result.data["sha256"]


def test_download_failure_is_reported(client: HttpClient, transport: FixtureTransport, tmp_path: Path) -> None:
    transport.add("/api/download", StubResponse.json({}, status=500))
    result = CHARMMGUIProvider(client, token=TOKEN).download_job("1234", tmp_path / "x.tgz")
    assert result.ok is False
    assert result.error_type == "HttpStatusError"
    assert not (tmp_path / "x.tgz").exists()


def test_corrupt_archive_is_rejected(client: HttpClient, transport: FixtureTransport, tmp_path: Path) -> None:
    transport.add("/api/download", StubResponse.text("<html>Session expired</html>"))
    result = CHARMMGUIProvider(client, token=TOKEN).download_job("1234", tmp_path / "x.tgz")
    assert result.ok is False
    assert result.error_type == "CorruptArchive"


def test_empty_download_is_rejected(client: HttpClient, transport: FixtureTransport, tmp_path: Path) -> None:
    transport.add("/api/download", StubResponse(body=b""))
    result = CHARMMGUIProvider(client, token=TOKEN).download_job("1234", tmp_path / "x.tgz")
    assert result.ok is False
    assert result.error_type == "CorruptArchive"
    assert not (tmp_path / "x.tgz").exists()


def test_download_digest_mismatch_is_rejected(
    client: HttpClient, transport: FixtureTransport, tmp_path: Path
) -> None:
    transport.add("/api/download", StubResponse(body=make_tgz(tmp_path)))
    result = CHARMMGUIProvider(client, token=TOKEN).download_job(
        "1234", tmp_path / "x.tgz", expected_sha256="0" * 64
    )
    assert result.ok is False
    assert result.error_type == "ChecksumMismatch"
