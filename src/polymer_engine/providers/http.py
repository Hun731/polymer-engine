"""HTTP transport for external data providers.

Design notes:

* **Transport is injectable.**  Unit tests supply a fixture transport; no test in
  the core suite touches the network.
* **Errors are classified, not swallowed.**  A 401 is an
  :class:`AuthenticationError`, a socket timeout is a :class:`TimeoutErrorProvider`.
  Callers can therefore tell "the network is down" from "the data says no".
* **Retries are selective.**  429 and 5xx retry with exponential backoff, honouring
  ``Retry-After``.  400/401/403/404 never retry -- retrying them just wastes the
  provider's quota and hides the real problem.
* **Every response carries provenance**: URL, status, timestamp, digest, and whether
  it came from cache.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit, urlunsplit

from polymer_engine.core.errors import (
    AuthenticationError,
    AuthorizationError,
    HttpStatusError,
    RateLimitError,
    ResponseFormatError,
    TimeoutErrorProvider,
    TransportError,
)
from polymer_engine.core.logging import get_logger, redact
from polymer_engine.core.provenance import canonical_hash, sha256_bytes

logger = get_logger("providers.http")

RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class HttpRequest:
    url: str
    method: str = "GET"
    params: Mapping[str, Any] | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes | None = None
    timeout_s: float = 30.0

    def full_url(self) -> str:
        if not self.params:
            return self.url
        parts = urlsplit(self.url)
        merged = urlencode({k: v for k, v in self.params.items() if v is not None}, doseq=True)
        query = f"{parts.query}&{merged}" if parts.query else merged
        return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))

    def cache_key(self) -> str:
        """Cache identity.  Authorization headers are excluded on purpose so a cache
        entry is never keyed by, or able to leak, a credential."""
        return canonical_hash(
            {
                "method": self.method.upper(),
                "url": self.full_url(),
                "body": self.body.decode("utf-8", "replace") if self.body else None,
            }
        )


@dataclass(slots=True)
class HttpResponse:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)
    url: str = ""
    elapsed_s: float = 0.0
    from_cache: bool = False
    fetched_at: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def digest(self) -> str:
        return sha256_bytes(self.body)

    def text(self, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding, errors="replace")

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ResponseFormatError(
                f"Response body is not valid JSON: {exc}",
                url=self.url,
                status=self.status,
                body_preview=redact(self.text()[:200]),
            ) from exc

    def header(self, name: str) -> str | None:
        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return None


class Transport(Protocol):
    """Anything that can turn an :class:`HttpRequest` into an :class:`HttpResponse`."""

    def send(self, request: HttpRequest) -> HttpResponse: ...


class UrllibTransport:
    """Default transport built on the standard library.

    Non-2xx responses are returned rather than raised so the client owns the
    classification and retry policy in one place.
    """

    def send(self, request: HttpRequest) -> HttpResponse:
        url = request.full_url()
        req = urllib.request.Request(url, data=request.body, headers=dict(request.headers), method=request.method.upper())
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=request.timeout_s) as response:
                body = response.read()
                return HttpResponse(
                    status=response.status,
                    body=body,
                    headers=dict(response.headers.items()),
                    url=url,
                    elapsed_s=time.monotonic() - started,
                    fetched_at=_now_iso(),
                )
        except urllib.error.HTTPError as exc:
            body = exc.read() if hasattr(exc, "read") else b""
            return HttpResponse(
                status=exc.code,
                body=body,
                headers=dict((exc.headers or {}).items()),
                url=url,
                elapsed_s=time.monotonic() - started,
                fetched_at=_now_iso(),
            )
        except TimeoutError as exc:
            raise TimeoutErrorProvider(f"Request timed out after {request.timeout_s}s", url=url) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise TimeoutErrorProvider(f"Request timed out after {request.timeout_s}s", url=url) from exc
            raise TransportError(f"Network failure: {reason}", url=url) from exc
        except OSError as exc:
            raise TransportError(f"Network failure: {exc}", url=url) from exc


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


class RateLimiter:
    """Token bucket, one instance per provider host.

    ``clock``/``sleeper`` are injectable so rate-limit behaviour is testable without
    real delays.
    """

    def __init__(
        self,
        rate_per_s: float,
        *,
        burst: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_s <= 0:
            raise ValueError("rate_per_s must be positive")
        self.rate = rate_per_s
        self.capacity = float(burst if burst is not None else max(1.0, rate_per_s))
        self._tokens = self.capacity
        self._clock = clock
        self._sleeper = sleeper
        self._last = clock()

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available; returns the time spent waiting."""
        waited = 0.0
        while True:
            now = self._clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return waited
            deficit = tokens - self._tokens
            delay = deficit / self.rate
            self._sleeper(delay)
            waited += delay


class ResponseCache:
    """Content-addressed on-disk cache with a TTL.

    Only successful GET-style responses are cached; error bodies are never served
    from cache.
    """

    def __init__(self, directory: str | Path, ttl_s: float = 86400.0, *, clock: Callable[[], float] = time.time) -> None:
        self.directory = Path(directory)
        self.ttl_s = ttl_s
        self._clock = clock
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> HttpResponse | None:
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt cache entry is a miss, not a failure.  Drop it.
            path.unlink(missing_ok=True)
            self.misses += 1
            return None
        if self.ttl_s and self._clock() - record.get("stored_at", 0.0) > self.ttl_s:
            self.misses += 1
            return None
        self.hits += 1
        return HttpResponse(
            status=record["status"],
            body=bytes.fromhex(record["body_hex"]),
            headers=record.get("headers", {}),
            url=record.get("url", ""),
            from_cache=True,
            fetched_at=record.get("fetched_at", ""),
        )

    def put(self, key: str, response: HttpResponse) -> None:
        if not response.ok:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stored_at": self._clock(),
            "status": response.status,
            "headers": response.headers,
            "url": response.url,
            "fetched_at": response.fetched_at,
            "body_hex": response.body.hex(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)

    def clear(self) -> int:
        removed = 0
        if self.directory.exists():
            for path in self.directory.rglob("*.json"):
                path.unlink()
                removed += 1
        return removed


@dataclass(slots=True)
class RequestProvenance:
    """What we recorded about one outbound call."""

    url: str
    method: str
    status: int
    fetched_at: str
    response_sha256: str
    from_cache: bool
    attempts: int
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "method": self.method,
            "status": self.status,
            "fetched_at": self.fetched_at,
            "response_sha256": self.response_sha256,
            "from_cache": self.from_cache,
            "attempts": self.attempts,
            "elapsed_s": round(self.elapsed_s, 4),
        }


class HttpClient:
    """Retrying, rate-limited, caching HTTP client with classified errors."""

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        user_agent: str = "polymer-autonomous-engine",
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 30.0,
        rate_limiter: RateLimiter | None = None,
        cache: ResponseCache | None = None,
        offline: bool = False,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.transport = transport or UrllibTransport()
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.rate_limiter = rate_limiter
        self.cache = cache
        self.offline = offline
        self._sleeper = sleeper
        self._jitter = jitter
        self.last_provenance: RequestProvenance | None = None

    # -- public API -----------------------------------------------------
    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
        data: bytes | None = None,
        timeout_s: float | None = None,
        use_cache: bool | None = None,
    ) -> HttpResponse:
        request_headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if headers:
            request_headers.update(headers)
        body = data
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"

        request = HttpRequest(
            url=url,
            method=method,
            params=params,
            headers=request_headers,
            body=body,
            timeout_s=timeout_s if timeout_s is not None else self.timeout_s,
        )
        cacheable = (use_cache if use_cache is not None else True) and method.upper() == "GET" and self.cache is not None
        key = request.cache_key()

        if cacheable and self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                self.last_provenance = RequestProvenance(
                    url=request.full_url(),
                    method=method.upper(),
                    status=cached.status,
                    fetched_at=cached.fetched_at,
                    response_sha256=cached.digest,
                    from_cache=True,
                    attempts=0,
                    elapsed_s=0.0,
                )
                return cached

        if self.offline:
            raise TransportError(
                "Offline mode is enabled and this request is not cached",
                url=request.full_url(),
            )

        response = self._send_with_retries(request)
        if cacheable and self.cache is not None:
            self.cache.put(key, response)
        return response

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return self.request(url, method="GET", **kwargs).json()

    def post_json(self, url: str, json_body: Any, **kwargs: Any) -> Any:
        return self.request(url, method="POST", json_body=json_body, **kwargs).json()

    def download(
        self,
        url: str,
        destination: str | Path,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        expected_sha256: str | None = None,
    ) -> Path:
        """Fetch ``url`` to ``destination``, verifying the digest when one is given.

        Written via a temporary file and renamed, so a failed download never leaves
        a truncated file that looks valid.
        """
        from polymer_engine.core.errors import ChecksumMismatch

        response = self.request(url, method="GET", headers=headers, timeout_s=timeout_s, use_cache=False)
        digest = response.digest
        if expected_sha256 and digest != expected_sha256:
            raise ChecksumMismatch(
                "Downloaded payload does not match the expected digest",
                url=url,
                expected=expected_sha256,
                actual=digest,
            )
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".part")
        tmp.write_bytes(response.body)
        tmp.replace(destination)
        return destination

    # -- internals ------------------------------------------------------
    def _send_with_retries(self, request: HttpRequest) -> HttpResponse:
        attempt = 0
        last_exception: Exception | None = None
        started = time.monotonic()
        while attempt <= self.max_retries:
            attempt += 1
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()
            try:
                response = self.transport.send(request)
            except (TransportError, TimeoutErrorProvider) as exc:
                last_exception = exc
                if attempt > self.max_retries:
                    break
                self._sleeper(self._backoff(attempt))
                continue

            if response.ok:
                self.last_provenance = RequestProvenance(
                    url=request.full_url(),
                    method=request.method.upper(),
                    status=response.status,
                    fetched_at=response.fetched_at,
                    response_sha256=response.digest,
                    from_cache=False,
                    attempts=attempt,
                    elapsed_s=time.monotonic() - started,
                )
                return response

            if response.status in RETRYABLE_STATUSES and attempt <= self.max_retries:
                delay = self._retry_after(response) or self._backoff(attempt)
                logger.warning(
                    "Retrying %s after HTTP %s in %.2fs (attempt %d/%d)",
                    request.url,
                    response.status,
                    delay,
                    attempt,
                    self.max_retries,
                )
                self._sleeper(delay)
                continue

            raise self._classify(request, response)

        assert last_exception is not None
        raise last_exception

    def _backoff(self, attempt: int) -> float:
        base = min(self.backoff_max_s, self.backoff_base_s * (2 ** (attempt - 1)))
        return base * (0.5 + 0.5 * self._jitter())

    @staticmethod
    def _retry_after(response: HttpResponse) -> float | None:
        raw = response.header("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    @staticmethod
    def _classify(request: HttpRequest, response: HttpResponse) -> Exception:
        url = request.full_url()
        preview = redact(response.text()[:300])
        if response.status == 401:
            return AuthenticationError("Authentication failed (HTTP 401)", url=url, status=401)
        if response.status == 403:
            return AuthorizationError("Not authorized for this resource (HTTP 403)", url=url, status=403)
        if response.status == 429:
            return RateLimitError(
                "Rate limited by the provider (HTTP 429)",
                retry_after=HttpClient._retry_after(response),
                url=url,
                status=429,
            )
        return HttpStatusError(
            f"HTTP {response.status} from provider",
            status=response.status,
            url=url,
            body=preview,
        )


__all__ = [
    "RETRYABLE_STATUSES",
    "HttpClient",
    "HttpRequest",
    "HttpResponse",
    "RateLimiter",
    "RequestProvenance",
    "ResponseCache",
    "Transport",
    "UrllibTransport",
]
