"""Provider registry: capability honesty and extensibility."""

from __future__ import annotations

import pytest

from polymer_engine.core.config import load_config
from polymer_engine.core.errors import PolymerEngineError, TransportError
from polymer_engine.providers.base import Capability, Provider, ProviderResult
from polymer_engine.providers.registry import ProviderRegistry, build_http_client
from polymer_engine.providers.testing import FixtureTransport, StubResponse


@pytest.fixture
def registry(tmp_path):
    config = load_config(
        discover=False, use_env=False,
        overrides={"paths": {"root": str(tmp_path)}, "http": {"cache_enabled": False}},
    )
    return ProviderRegistry(config, transport=FixtureTransport())


class TestRegistry:
    def test_every_named_provider_can_be_constructed(self, registry: ProviderRegistry) -> None:
        for name in registry.names():
            assert isinstance(registry.get(name), Provider)

    def test_providers_are_cached_between_lookups(self, registry: ProviderRegistry) -> None:
        assert registry.get("pubchem") is registry.get("pubchem")

    def test_unknown_provider_raises_with_the_known_list(self, registry: ProviderRegistry) -> None:
        with pytest.raises(PolymerEngineError) as excinfo:
            registry.get("nonexistent")
        assert "pubchem" in excinfo.value.context["known"]

    def test_capabilities_come_from_the_implementations(self, registry: ProviderRegistry) -> None:
        """The registry must not be able to claim something the class does not declare."""
        capabilities = registry.capabilities()
        for name, declared in capabilities.items():
            assert declared == sorted(c.value for c in registry.get(name).capabilities)

    def test_charmm_gui_submission_is_declared_unsupported(self, registry: ProviderRegistry) -> None:
        assert registry.unsupported()["charmm_gui"] == ["job_submission"]
        assert "job_submission" not in registry.capabilities()["charmm_gui"]

    def test_find_by_capability(self, registry: ProviderRegistry) -> None:
        searchers = {p.name for p in registry.find(Capability.LITERATURE_SEARCH)}
        assert {"crossref", "europe_pmc", "openalex"} <= searchers
        assert "pubchem" not in searchers

    def test_status_reports_which_providers_need_credentials(self, registry: ProviderRegistry) -> None:
        status = registry.status()
        assert status["pubchem"]["configured"] is True
        assert status["materials_project"]["requires_credentials"] is True
        assert status["charmm_gui"]["requires_credentials"] is True

    def test_a_new_provider_can_be_added_without_touching_the_orchestrator(
        self, registry: ProviderRegistry
    ) -> None:
        class InHouseProvider(Provider):
            name = "in_house"
            capabilities = frozenset({Capability.COMPOUND_LOOKUP})

            def health(self) -> ProviderResult:
                return ProviderResult(self.name, "health", True)

        registry.register("in_house", lambda cfg, http: InHouseProvider(http))
        assert "in_house" in registry.names()
        assert registry.get("in_house").name == "in_house"
        assert registry.capabilities()["in_house"] == ["compound_lookup"]

    def test_registering_does_not_mutate_other_registries(self, registry: ProviderRegistry, tmp_path) -> None:
        """A per-instance registration must not leak into the class-level defaults."""
        class Temp(Provider):
            name = "temp"

            def health(self) -> ProviderResult:
                return ProviderResult(self.name, "health", True)

        registry.register("temp", lambda cfg, http: Temp(http))
        other = ProviderRegistry(
            load_config(discover=False, use_env=False, overrides={"paths": {"root": str(tmp_path)}})
        )
        assert "temp" not in other.names()

    def test_iteration_covers_every_provider(self, registry: ProviderRegistry) -> None:
        assert len(list(registry)) == len(registry.names())

    def test_health_never_raises_even_when_the_transport_fails(self, tmp_path) -> None:
        config = load_config(
            discover=False, use_env=False,
            overrides={"paths": {"root": str(tmp_path)}, "http": {"cache_enabled": False, "max_retries": 0}},
        )
        transport = FixtureTransport()
        transport.add("", StubResponse.error(TransportError("network down", url="x")))
        registry = ProviderRegistry(config, transport=transport)
        results = registry.health(["pubchem", "crossref"])
        assert all(result.ok is False for result in results.values())
        assert all(result.error_type for result in results.values())


class TestHttpClientConstruction:
    def test_offline_config_produces_an_offline_client(self, tmp_path) -> None:
        config = load_config(
            discover=False, use_env=False,
            overrides={"paths": {"root": str(tmp_path)}, "http": {"offline": True}},
        )
        assert build_http_client(config, provider="pubchem").offline is True

    def test_disallowing_network_also_produces_an_offline_client(self, tmp_path) -> None:
        config = load_config(
            discover=False, use_env=False,
            overrides={"paths": {"root": str(tmp_path)}, "safety": {"allow_network": False}},
        )
        assert build_http_client(config, provider="pubchem").offline is True

    def test_each_provider_gets_its_own_cache_namespace(self, tmp_path) -> None:
        config = load_config(
            discover=False, use_env=False,
            overrides={"paths": {"root": str(tmp_path)}, "http": {"cache_enabled": True}},
        )
        a = build_http_client(config, provider="pubchem")
        b = build_http_client(config, provider="crossref")
        assert a.cache is not None and b.cache is not None
        assert a.cache.directory != b.cache.directory

    def test_user_agent_identifies_the_provider(self, tmp_path) -> None:
        config = load_config(discover=False, use_env=False, overrides={"paths": {"root": str(tmp_path)}})
        assert "pubchem" in build_http_client(config, provider="pubchem").user_agent

    def test_pubchem_gets_a_conservative_rate_limit(self, tmp_path) -> None:
        """PubChem publishes a hard 5 requests/second cap."""
        config = load_config(discover=False, use_env=False, overrides={"paths": {"root": str(tmp_path)}})
        client = build_http_client(config, provider="pubchem")
        assert client.rate_limiter is not None
        assert client.rate_limiter.rate <= 5.0
