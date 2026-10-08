"""Provider registry.

Capabilities are read from each provider class rather than restated here, so the
registry cannot claim something the implementation does not do.  Adding a provider
means adding one entry; the orchestrator never changes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, ClassVar

from polymer_engine.core.config import EngineConfig, load_config
from polymer_engine.core.errors import PolymerEngineError
from polymer_engine.providers.base import Capability, Provider, ProviderResult
from polymer_engine.providers.charmm_gui import CHARMMGUIProvider
from polymer_engine.providers.crossref import CrossrefProvider
from polymer_engine.providers.europe_pmc import EuropePMCProvider
from polymer_engine.providers.http import HttpClient, RateLimiter, ResponseCache, Transport
from polymer_engine.providers.materials_project import MaterialsProjectProvider
from polymer_engine.providers.openalex import OpenAlexProvider
from polymer_engine.providers.pubchem import PubChemProvider
from polymer_engine.providers.rcsb_pdb import RCSBProvider

#: Per-host courtesy limits.  PubChem publishes a hard 5 requests/second cap; the
#: others are conservative defaults that keep us inside each service's fair-use ask.
DEFAULT_RATE_LIMITS: dict[str, float] = {
    "pubchem": 4.0,
    "crossref": 5.0,
    "europe_pmc": 5.0,
    "openalex": 8.0,
    "rcsb_pdb": 5.0,
    "materials_project": 5.0,
    "charmm_gui": 0.5,
}


def build_http_client(
    config: EngineConfig,
    *,
    provider: str,
    transport: Transport | None = None,
) -> HttpClient:
    """Create a client with this provider's cache namespace and rate limit."""
    cache = None
    if config.http.cache_enabled:
        cache = ResponseCache(config.paths.resolved("cache_dir") / "http" / provider, ttl_s=config.http.cache_ttl_s)
    rate = DEFAULT_RATE_LIMITS.get(provider, config.http.rate_limit_per_s)
    return HttpClient(
        transport=transport,
        user_agent=f"{config.http.user_agent} ({provider})",
        timeout_s=config.http.timeout_s,
        max_retries=config.http.max_retries,
        backoff_base_s=config.http.backoff_base_s,
        backoff_max_s=config.http.backoff_max_s,
        rate_limiter=RateLimiter(rate),
        cache=cache,
        offline=config.http.offline or not config.safety.allow_network,
    )


class ProviderRegistry:
    """Lazily constructs providers from configuration."""

    #: name -> factory taking (config, http client)
    FACTORIES: ClassVar[dict[str, Callable[[EngineConfig, HttpClient], Provider]]] = {
        "pubchem": lambda cfg, http: PubChemProvider(http),
        "crossref": lambda cfg, http: CrossrefProvider(http, mailto=cfg.credentials.crossref_mailto),
        "europe_pmc": lambda cfg, http: EuropePMCProvider(http),
        "openalex": lambda cfg, http: OpenAlexProvider(http, mailto=cfg.credentials.openalex_mailto),
        "rcsb_pdb": lambda cfg, http: RCSBProvider(http),
        "materials_project": lambda cfg, http: MaterialsProjectProvider(
            http, api_key=cfg.credentials.materials_project_api_key
        ),
        "charmm_gui": lambda cfg, http: CHARMMGUIProvider(
            http,
            email=cfg.credentials.charmm_gui_email,
            password=cfg.credentials.charmm_gui_password,
            token=cfg.credentials.charmm_gui_token,
        ),
    }

    def __init__(self, config: EngineConfig | None = None, *, transport: Transport | None = None) -> None:
        self.config = config or load_config()
        self._transport = transport
        self._instances: dict[str, Provider] = {}
        # Instance-level copy so registering a provider on one registry does not
        # mutate the class default shared by every other registry.
        self._factories: dict[str, Callable[[EngineConfig, HttpClient], Provider]] = dict(self.FACTORIES)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def get(self, name: str) -> Provider:
        if name not in self._factories:
            raise PolymerEngineError(
                "Unknown provider", provider=name, known=self.names()
            )
        if name not in self._instances:
            client = build_http_client(self.config, provider=name, transport=self._transport)
            self._instances[name] = self._factories[name](self.config, client)
        return self._instances[name]

    def register(self, name: str, factory: Callable[[EngineConfig, HttpClient], Provider]) -> None:
        """Add a provider without touching the orchestrator."""
        self._factories[name] = factory
        self._instances.pop(name, None)

    def __iter__(self) -> Iterator[Provider]:
        return (self.get(name) for name in self.names())

    def capabilities(self) -> dict[str, list[str]]:
        """Capabilities read from the implementations, not from a hand-kept list."""
        return {name: sorted(c.value for c in self.get(name).capabilities) for name in self.names()}

    def unsupported(self) -> dict[str, list[str]]:
        """Capabilities each provider explicitly does *not* offer."""
        out: dict[str, list[str]] = {}
        for name in self.names():
            declared: frozenset[Capability] = getattr(
                self.get(name), "unsupported_capabilities", frozenset()
            )
            if declared:
                out[name] = sorted(c.value for c in declared)
        return out

    def find(self, capability: Capability) -> list[Provider]:
        return [p for p in self if p.supports(capability)]

    def status(self) -> dict[str, dict[str, Any]]:
        """Configuration status for every provider.  Does not touch the network."""
        report: dict[str, dict[str, Any]] = {}
        for name in self.names():
            provider = self.get(name)
            report[name] = {
                "configured": provider.configured(),
                "capabilities": sorted(c.value for c in provider.capabilities),
                "unsupported": sorted(
                    c.value for c in getattr(provider, "unsupported_capabilities", frozenset())
                ),
                "requires_credentials": not provider.configured(),
            }
        return report

    def health(self, names: list[str] | None = None) -> dict[str, ProviderResult]:
        """Live reachability check.  Requires network; each provider returns, never raises."""
        return {name: self.get(name).health() for name in (names or self.names())}


__all__ = ["DEFAULT_RATE_LIMITS", "ProviderRegistry", "build_http_client"]
