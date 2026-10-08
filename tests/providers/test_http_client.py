"""Transport-layer behaviour: retries, classification, rate limiting, caching."""

from __future__ import annotations

import pytest

from polymer_engine.core.errors import (
    AuthenticationError,
    AuthorizationError,
    HttpStatusError,
    RateLimitError,
    ResponseFormatError,
    TimeoutErrorProvider,
    TransportError,
)
from polymer_engine.providers.http import HttpClient, HttpRequest, RateLimiter, ResponseCache
from polymer_engine.providers.testing import FixtureTransport, RecordingSleeper, StubResponse


def test_successful_json_request(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add_json("example.org/data", {"value": 42})
    assert client.get_json("https://example.org/data") == {"value": 42}
    assert transport.call_count == 1


def test_params_are_appended_to_the_query_string(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add_json("example.org/search", {"ok": True})
    client.get_json("https://example.org/search", params={"q": "poly(ethylene)", "rows": 5})
    url = transport.urls()[0]
    assert "q=poly%28ethylene%29" in url
    assert "rows=5" in url


def test_none_params_are_dropped() -> None:
    request = HttpRequest(url="https://x.test/a", params={"keep": 1, "drop": None})
    assert "drop" not in request.full_url()
    assert "keep=1" in request.full_url()


def test_401_raises_authentication_error(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("example.org/p", StubResponse.json({"error": "bad"}, status=401))
    with pytest.raises(AuthenticationError):
        client.get_json("https://example.org/p")


def test_403_raises_authorization_error(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("example.org/p", StubResponse.json({"error": "nope"}, status=403))
    with pytest.raises(AuthorizationError):
        client.get_json("https://example.org/p")


def test_404_raises_http_status_error_and_is_not_retried(
    client: HttpClient, transport: FixtureTransport, sleeper: RecordingSleeper
) -> None:
    transport.add("example.org/p", StubResponse.json({}, status=404))
    with pytest.raises(HttpStatusError) as excinfo:
        client.get_json("https://example.org/p")
    assert excinfo.value.status == 404
    assert excinfo.value.retryable is False
    assert transport.call_count == 1, "a 404 must not consume retry budget"
    assert sleeper.calls == []


def test_429_is_retried_then_raises_rate_limit_error(
    client: HttpClient, transport: FixtureTransport, sleeper: RecordingSleeper
) -> None:
    transport.add("example.org/p", StubResponse.json({}, status=429))
    with pytest.raises(RateLimitError):
        client.get_json("https://example.org/p")
    assert transport.call_count == 3, "initial attempt plus two retries"
    assert len(sleeper.calls) == 2


def test_retry_after_header_controls_the_delay(transport: FixtureTransport) -> None:
    sleeper = RecordingSleeper()
    client = HttpClient(transport=transport, max_retries=1, sleeper=sleeper, jitter=lambda: 1.0)
    transport.add(
        "example.org/p",
        [StubResponse(status=429, body=b"{}", headers={"Retry-After": "7"}), StubResponse.json({"ok": 1})],
    )
    assert client.get_json("https://example.org/p") == {"ok": 1}
    assert sleeper.calls == [7.0]


def test_5xx_retries_and_then_succeeds(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add(
        "example.org/p",
        [StubResponse.json({}, status=503), StubResponse.json({}, status=500), StubResponse.json({"ok": True})],
    )
    assert client.get_json("https://example.org/p") == {"ok": True}
    assert transport.call_count == 3


def test_5xx_exhausts_retries_and_raises(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("example.org/p", StubResponse.json({}, status=500))
    with pytest.raises(HttpStatusError) as excinfo:
        client.get_json("https://example.org/p")
    assert excinfo.value.status == 500
    assert excinfo.value.retryable is True


def test_timeout_is_retried_then_surfaces_as_timeout(
    client: HttpClient, transport: FixtureTransport
) -> None:
    transport.add("example.org/p", StubResponse.error(TimeoutErrorProvider("timed out", url="x")))
    with pytest.raises(TimeoutErrorProvider):
        client.get_json("https://example.org/p")
    assert transport.call_count == 3


def test_connection_failure_surfaces_as_transport_error(
    client: HttpClient, transport: FixtureTransport
) -> None:
    transport.add("example.org/p", StubResponse.error(TransportError("connection refused", url="x")))
    with pytest.raises(TransportError):
        client.get_json("https://example.org/p")


def test_transport_recovers_after_transient_failure(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add(
        "example.org/p",
        [StubResponse.error(TransportError("reset", url="x")), StubResponse.json({"ok": 1})],
    )
    assert client.get_json("https://example.org/p") == {"ok": 1}


def test_malformed_json_raises_response_format_error(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("example.org/p", StubResponse.text("<html>not json</html>"))
    with pytest.raises(ResponseFormatError):
        client.get_json("https://example.org/p")


def test_error_body_is_redacted_in_the_exception(client: HttpClient, transport: FixtureTransport) -> None:
    transport.add("example.org/p", StubResponse.text('{"api_key": "hunter2hunter2"}', status=400))
    with pytest.raises(HttpStatusError) as excinfo:
        client.get_json("https://example.org/p")
    assert "hunter2hunter2" not in excinfo.value.body


def test_offline_mode_refuses_uncached_requests(transport: FixtureTransport) -> None:
    client = HttpClient(transport=transport, offline=True)
    with pytest.raises(TransportError, match="Offline mode"):
        client.get_json("https://example.org/p")
    assert transport.call_count == 0


# -- caching -------------------------------------------------------------
def test_cache_miss_then_hit(caching_client: HttpClient, transport: FixtureTransport) -> None:
    transport.add_json("example.org/c", {"n": 1})
    assert caching_client.get_json("https://example.org/c") == {"n": 1}
    assert transport.call_count == 1
    assert caching_client.last_provenance is not None
    assert caching_client.last_provenance.from_cache is False

    assert caching_client.get_json("https://example.org/c") == {"n": 1}
    assert transport.call_count == 1, "second call must be served from cache"
    assert caching_client.last_provenance.from_cache is True
    assert caching_client.cache is not None
    assert caching_client.cache.hits == 1


def test_cache_distinguishes_different_params(caching_client: HttpClient, transport: FixtureTransport) -> None:
    transport.add_json("example.org/c", {"n": 1})
    caching_client.get_json("https://example.org/c", params={"a": 1})
    caching_client.get_json("https://example.org/c", params={"a": 2})
    assert transport.call_count == 2


def test_cache_does_not_store_error_responses(tmp_path, transport: FixtureTransport) -> None:
    cache = ResponseCache(tmp_path / "c", ttl_s=3600, clock=lambda: 0.0)
    client = HttpClient(transport=transport, cache=cache, max_retries=0, sleeper=lambda s: None)
    transport.add("example.org/e", StubResponse.json({}, status=404))
    with pytest.raises(HttpStatusError):
        client.get_json("https://example.org/e")
    assert list((tmp_path / "c").rglob("*.json")) == []


def test_expired_cache_entry_is_a_miss(tmp_path, transport: FixtureTransport) -> None:
    now = {"t": 0.0}
    cache = ResponseCache(tmp_path / "c", ttl_s=10.0, clock=lambda: now["t"])
    client = HttpClient(transport=transport, cache=cache, sleeper=lambda s: None)
    transport.add_json("example.org/c", {"n": 1})
    client.get_json("https://example.org/c")
    now["t"] = 100.0
    client.get_json("https://example.org/c")
    assert transport.call_count == 2


def test_cache_key_ignores_authorization_header() -> None:
    a = HttpRequest(url="https://x.test/j", headers={"Authorization": "Bearer aaa"})
    b = HttpRequest(url="https://x.test/j", headers={"Authorization": "Bearer bbb"})
    assert a.cache_key() == b.cache_key()


def test_corrupt_cache_entry_is_treated_as_a_miss(tmp_path, transport: FixtureTransport) -> None:
    cache = ResponseCache(tmp_path / "c", ttl_s=3600, clock=lambda: 0.0)
    client = HttpClient(transport=transport, cache=cache, sleeper=lambda s: None)
    transport.add_json("example.org/c", {"n": 1})
    client.get_json("https://example.org/c")
    for path in (tmp_path / "c").rglob("*.json"):
        path.write_text("{ this is not json")
    assert client.get_json("https://example.org/c") == {"n": 1}
    assert transport.call_count == 2


# -- rate limiting -------------------------------------------------------
def test_rate_limiter_allows_burst_then_throttles() -> None:
    now = {"t": 0.0}
    slept: list[float] = []

    def sleeper(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    limiter = RateLimiter(2.0, burst=2, clock=lambda: now["t"], sleeper=sleeper)
    assert limiter.acquire() == 0.0
    assert limiter.acquire() == 0.0
    assert limiter.acquire() == pytest.approx(0.5)
    assert slept == [pytest.approx(0.5)]


def test_rate_limiter_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError):
        RateLimiter(0.0)


# -- downloads -----------------------------------------------------------
def test_download_writes_file_and_verifies_digest(client: HttpClient, transport: FixtureTransport, tmp_path) -> None:
    from polymer_engine.core.provenance import sha256_bytes

    payload = b"binary-content"
    transport.add("example.org/f", StubResponse(body=payload))
    target = tmp_path / "out.bin"
    client.download("https://example.org/f", target, expected_sha256=sha256_bytes(payload))
    assert target.read_bytes() == payload


def test_download_with_wrong_digest_raises_and_leaves_no_file(
    client: HttpClient, transport: FixtureTransport, tmp_path
) -> None:
    from polymer_engine.core.errors import ChecksumMismatch

    transport.add("example.org/f", StubResponse(body=b"tampered"))
    target = tmp_path / "out.bin"
    with pytest.raises(ChecksumMismatch):
        client.download("https://example.org/f", target, expected_sha256="0" * 64)
    assert not target.exists()
