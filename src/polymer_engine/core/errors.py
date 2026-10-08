"""Typed error hierarchy.

Every failure mode in the engine maps to one of these classes so that callers can
distinguish an *infrastructure* problem (network down, tool missing) from a
*scientific* problem (system failed validation, PMF did not converge).  Collapsing
the two is the single most common way an autonomous pipeline fabricates results.
"""

from __future__ import annotations

from typing import Any


class PolymerEngineError(Exception):
    """Base class for every error raised deliberately by this package."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.context:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({rendered})"


# --------------------------------------------------------------------------
# Configuration / environment
# --------------------------------------------------------------------------
class ConfigError(PolymerEngineError):
    """Configuration is missing, malformed, or internally inconsistent."""


class CredentialsMissing(ConfigError):
    """An operation needs a credential that has not been supplied."""


class ToolNotFound(PolymerEngineError):
    """A required local executable could not be discovered."""


class ToolVersionIncompatible(PolymerEngineError):
    """A local executable exists but its version is outside the supported range."""


# --------------------------------------------------------------------------
# Providers / transport
# --------------------------------------------------------------------------
class ProviderError(PolymerEngineError):
    """Base class for external-data-provider failures."""


class TransportError(ProviderError):
    """Network-level failure: DNS, connection refused, TLS, socket timeout."""


class TimeoutErrorProvider(TransportError):
    """The request exceeded the configured timeout."""


class HttpStatusError(ProviderError):
    """The server returned a non-success HTTP status."""

    def __init__(self, message: str, *, status: int, url: str, body: str = "", **context: Any) -> None:
        super().__init__(message, status=status, url=url, **context)
        self.status = status
        self.url = url
        self.body = body

    @property
    def retryable(self) -> bool:
        return self.status == 429 or 500 <= self.status < 600


class AuthenticationError(ProviderError):
    """HTTP 401 / invalid or expired credentials."""


class AuthorizationError(ProviderError):
    """HTTP 403 / the credential is valid but not permitted."""


class RateLimitError(ProviderError):
    """HTTP 429 or a locally enforced rate limit."""

    def __init__(self, message: str, *, retry_after: float | None = None, **context: Any) -> None:
        super().__init__(message, retry_after=retry_after, **context)
        self.retry_after = retry_after


class ResponseFormatError(ProviderError):
    """The response could not be parsed or did not match the expected schema."""


class UnsupportedCapability(ProviderError):
    """The requested capability is not part of this provider's supported contract.

    Raised instead of guessing an undocumented endpoint.
    """


# --------------------------------------------------------------------------
# Archives / filesystem safety
# --------------------------------------------------------------------------
class ArchiveError(PolymerEngineError):
    """Base class for archive handling failures."""


class UnsafeArchiveMember(ArchiveError):
    """An archive member would escape the extraction root or is otherwise unsafe."""


class CorruptArchive(ArchiveError):
    """The archive could not be opened or read."""


class ChecksumMismatch(ArchiveError):
    """A file's digest did not match the expected value."""


# --------------------------------------------------------------------------
# Scientific validation
# --------------------------------------------------------------------------
class ScientificError(PolymerEngineError):
    """Base class for domain/scientific failures (as opposed to infrastructure)."""


class SystemValidationError(ScientificError):
    """A simulation system is not usable as a scientific input."""


class ParameterValidationError(ScientificError):
    """A simulation or analysis parameter is outside a physically sensible range."""


class InsufficientDataError(ScientificError):
    """Not enough samples/replicas to compute the requested quantity honestly."""


class ConvergenceNotEstablished(ScientificError):
    """A quantity was requested whose convergence has not been demonstrated."""


class IllegalStateTransition(PolymerEngineError):
    """An execution-state transition is not permitted by the lifecycle model."""


class ChemistryError(ScientificError):
    """A molecular structure is invalid or cannot be interpreted."""


__all__ = [
    "ArchiveError",
    "AuthenticationError",
    "AuthorizationError",
    "ChecksumMismatch",
    "ChemistryError",
    "ConfigError",
    "ConvergenceNotEstablished",
    "CorruptArchive",
    "CredentialsMissing",
    "HttpStatusError",
    "IllegalStateTransition",
    "InsufficientDataError",
    "ParameterValidationError",
    "PolymerEngineError",
    "ProviderError",
    "RateLimitError",
    "ResponseFormatError",
    "ScientificError",
    "SystemValidationError",
    "TimeoutErrorProvider",
    "ToolNotFound",
    "ToolVersionIncompatible",
    "TransportError",
    "UnsafeArchiveMember",
    "UnsupportedCapability",
]
