"""Provider behaviour against recorded offline fixtures.

Each provider is exercised for the same failure matrix: valid response, empty
result, malformed result, 429, 500, timeout, connection failure, duplicates, and
cache hit/miss.  The point is that a *transport* failure never looks like "the
literature contains nothing".
"""

from __future__ import annotations

import json
from pathlib import Path

from polymer_engine.core.errors import TimeoutErrorProvider, TransportError
from polymer_engine.providers.crossref import CrossrefProvider
from polymer_engine.providers.europe_pmc import EuropePMCProvider
from polymer_engine.providers.http import HttpClient
from polymer_engine.providers.materials_project import MaterialsProjectProvider
from polymer_engine.providers.openalex import OpenAlexProvider
from polymer_engine.providers.pubchem import PubChemProvider
from polymer_engine.providers.rcsb_pdb import RCSBProvider
from polymer_engine.providers.testing import FixtureTransport, StubResponse

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "providers"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ==========================================================================
# PubChem
# ==========================================================================
class TestPubChem:
    def test_valid_response_is_normalised(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("pubchem", fixture("pubchem_ethanol.json"))
        result = PubChemProvider(client).resolve_compound("ethanol")
        assert result.ok
        record = result.records[0]
        assert record["cid"] == 702
        assert record["smiles"] == "CCO"
        assert record["inchikey"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        assert record["heavy_atom_count"] == 3

    def test_no_match_is_an_empty_result_not_a_failure(
        self, client: HttpClient, transport: FixtureTransport
    ) -> None:
        transport.add("pubchem", StubResponse.json(fixture("pubchem_not_found.json"), status=404))
        result = PubChemProvider(client).resolve_compound("notarealcompound")
        assert result.ok is True
        assert result.empty is True
        assert result.records == []
        assert result.total_available == 0

    def test_retired_property_names_fall_back_to_the_legacy_set(
        self, client: HttpClient, transport: FixtureTransport
    ) -> None:
        transport.add(
            "pubchem",
            [StubResponse.json({"Fault": {}}, status=400), StubResponse.json(fixture("pubchem_legacy_ethanol.json"))],
        )
        result = PubChemProvider(client).resolve_compound("ethanol")
        assert result.ok
        assert result.records[0]["smiles"] == "CCO"

    def test_malformed_payload_is_reported_as_a_format_error(
        self, client: HttpClient, transport: FixtureTransport
    ) -> None:
        transport.add_json("pubchem", {"unexpected": "shape"})
        result = PubChemProvider(client).resolve_compound("ethanol")
        assert result.ok is False
        assert result.error_type == "ResponseFormatError"

    def test_non_json_body_is_reported_as_a_format_error(
        self, client: HttpClient, transport: FixtureTransport
    ) -> None:
        transport.add("pubchem", StubResponse.text("<html>maintenance</html>"))
        result = PubChemProvider(client).resolve_compound("ethanol")
        assert result.ok is False
        assert result.error_type == "ResponseFormatError"

    def test_rate_limit_is_reported_distinctly(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add("pubchem", StubResponse.json({}, status=429))
        result = PubChemProvider(client).resolve_compound("ethanol")
        assert result.ok is False
        assert result.error_type == "RateLimitError"
        assert result.empty is False, "a rate limit must never read as 'no records'"

    def test_server_error_is_reported_distinctly(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add("pubchem", StubResponse.json({}, status=500))
        result = PubChemProvider(client).resolve_compound("ethanol")
        assert result.ok is False
        assert result.error_type == "HttpStatusError"

    def test_timeout_is_reported_distinctly(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add("pubchem", StubResponse.error(TimeoutErrorProvider("slow", url="x")))
        result = PubChemProvider(client).resolve_compound("ethanol")
        assert result.ok is False
        assert result.error_type == "TimeoutErrorProvider"

    def test_connection_failure_is_reported_distinctly(
        self, client: HttpClient, transport: FixtureTransport
    ) -> None:
        transport.add("pubchem", StubResponse.error(TransportError("refused", url="x")))
        result = PubChemProvider(client).resolve_compound("ethanol")
        assert result.ok is False
        assert result.error_type == "TransportError"

    def test_empty_identifier_is_rejected_without_a_request(
        self, client: HttpClient, transport: FixtureTransport
    ) -> None:
        result = PubChemProvider(client).resolve_compound("   ")
        assert result.ok is False
        assert transport.call_count == 0

    def test_identifier_is_url_escaped(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("pubchem", fixture("pubchem_ethanol.json"))
        PubChemProvider(client).resolve_compound("N,N-dimethylformamide")
        assert "N%2CN-dimethylformamide" in transport.urls()[0]

    def test_cache_miss_then_hit(self, caching_client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("pubchem", fixture("pubchem_ethanol.json"))
        provider = PubChemProvider(caching_client)
        provider.resolve_compound("ethanol")
        assert transport.call_count == 1
        result = provider.resolve_compound("ethanol")
        assert transport.call_count == 1
        assert result.provenance["request"]["from_cache"] is True


# ==========================================================================
# Crossref
# ==========================================================================
class TestCrossref:
    def test_valid_search(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("api.crossref.org", fixture("crossref_search.json"))
        result = CrossrefProvider(client).search("polymer chain dynamics")
        assert result.ok
        assert result.total_available == 2
        first = result.records[0]
        assert first["doi"] == "10.1021/ma0001"
        assert first["year"] == 2019
        assert first["authors"] == ["A. Researcher", "B. Colleague"]

    def test_base_url_has_no_v1_segment(self) -> None:
        assert CrossrefProvider.BASE == "https://api.crossref.org"

    def test_empty_result(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("api.crossref.org", fixture("crossref_empty.json"))
        result = CrossrefProvider(client).search("nothing matches this")
        assert result.ok and result.empty
        assert result.total_available == 0

    def test_duplicate_dois_are_collapsed(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("api.crossref.org", fixture("crossref_duplicates.json"))
        result = CrossrefProvider(client).search("dup")
        assert [r["doi"] for r in result.records] == ["10.1/dup", "10.1/unique"]

    def test_malformed_result(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("api.crossref.org", {"status": "ok"})
        result = CrossrefProvider(client).search("x")
        assert result.ok is False
        assert result.error_type == "ResponseFormatError"

    def test_mailto_enters_the_polite_pool(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("api.crossref.org", fixture("crossref_search.json"))
        result = CrossrefProvider(client, mailto="lab@example.org").search("x")
        assert "mailto=lab%40example.org" in transport.urls()[0]
        assert result.provenance["polite_pool"] is True

    def test_rate_limit(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add("api.crossref.org", StubResponse.json({}, status=429))
        assert CrossrefProvider(client).search("x").error_type == "RateLimitError"

    def test_timeout(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add("api.crossref.org", StubResponse.error(TimeoutErrorProvider("t", url="x")))
        assert CrossrefProvider(client).search("x").error_type == "TimeoutErrorProvider"

    def test_missing_doi_lookup_is_empty_not_failed(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add("api.crossref.org", StubResponse.json({}, status=404))
        result = CrossrefProvider(client).work("10.0/missing")
        assert result.ok and result.empty

    def test_rows_out_of_range_is_rejected(self, client: HttpClient, transport: FixtureTransport) -> None:
        assert CrossrefProvider(client).search("x", rows=5000).ok is False
        assert transport.call_count == 0


# ==========================================================================
# Europe PMC
# ==========================================================================
class TestEuropePMC:
    def test_valid_search(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("europepmc", fixture("europe_pmc_search.json"))
        result = EuropePMCProvider(client).search("hydrogen bonding polyamide")
        assert result.ok
        assert result.total_available == 2
        assert result.records[0]["pmcid"] == "PMC9000001"
        assert result.records[0]["is_open_access"] is True
        assert result.records[0]["full_text_urls"] == ["https://europepmc.org/article/MED/38000001"]
        assert result.records[1]["is_open_access"] is False

    def test_empty_search(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("europepmc", fixture("europe_pmc_empty.json"))
        result = EuropePMCProvider(client).search("nothing")
        assert result.ok and result.empty

    def test_missing_result_list_is_treated_as_empty(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("europepmc", {"hitCount": 0})
        result = EuropePMCProvider(client).search("nothing")
        assert result.ok and result.empty

    def test_malformed_result(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("europepmc", {"resultList": {"result": "not-a-list"}})
        assert EuropePMCProvider(client).search("x").error_type == "ResponseFormatError"

    def test_server_error(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add("europepmc", StubResponse.json({}, status=503))
        assert EuropePMCProvider(client).search("x").error_type == "HttpStatusError"

    def test_connection_failure(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add("europepmc", StubResponse.error(TransportError("down", url="x")))
        assert EuropePMCProvider(client).search("x").error_type == "TransportError"


# ==========================================================================
# OpenAlex
# ==========================================================================
class TestOpenAlex:
    def test_valid_search(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("api.openalex.org", fixture("openalex_search.json"))
        result = OpenAlexProvider(client).search("glass transition")
        assert result.ok
        assert result.records[0]["doi"] == "10.1000/oa1"
        assert result.records[0]["journal"] == "Polymer Chemistry"
        assert result.records[0]["referenced_works"] == ["https://openalex.org/W0"]
        assert result.records[1]["doi"] is None
        assert result.records[1]["title"] == "Cohesive energy of PMMA"

    def test_empty_search(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("api.openalex.org", fixture("openalex_empty.json"))
        assert OpenAlexProvider(client).search("x").empty

    def test_malformed_result(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("api.openalex.org", {"results": {"not": "a list"}})
        assert OpenAlexProvider(client).search("x").error_type == "ResponseFormatError"

    def test_per_page_bounds(self, client: HttpClient, transport: FixtureTransport) -> None:
        assert OpenAlexProvider(client).search("x", per_page=500).ok is False
        assert transport.call_count == 0

    def test_rate_limit(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add("api.openalex.org", StubResponse.json({}, status=429))
        assert OpenAlexProvider(client).search("x").error_type == "RateLimitError"


# ==========================================================================
# RCSB PDB
# ==========================================================================
class TestRCSB:
    def test_valid_search(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("search.rcsb.org", fixture("rcsb_search.json"))
        result = RCSBProvider(client).search_text("polyethylene glycol")
        assert result.ok
        assert [r["entry_id"] for r in result.records] == ["1ABC", "2XYZ"]
        assert result.total_available == 2

    def test_http_204_means_no_hits_not_a_parse_error(
        self, client: HttpClient, transport: FixtureTransport
    ) -> None:
        transport.add("search.rcsb.org", StubResponse(status=204, body=b""))
        result = RCSBProvider(client).search_text("nothing at all")
        assert result.ok is True
        assert result.empty is True
        assert result.total_available == 0

    def test_entry_lookup(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("data.rcsb.org", fixture("rcsb_entry.json"))
        result = RCSBProvider(client).entry("1abc")
        assert result.records[0]["entry_id"] == "1ABC"
        assert result.records[0]["resolution_angstrom"] == 1.8
        assert result.records[0]["experimental_method"] == "X-RAY DIFFRACTION"

    def test_missing_entry_is_empty(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add("data.rcsb.org", StubResponse.json({}, status=404))
        assert RCSBProvider(client).entry("9ZZZ").empty

    def test_malformed_search_payload(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("search.rcsb.org", {"result_set": "nope"})
        assert RCSBProvider(client).search_text("x").error_type == "ResponseFormatError"

    def test_download_rejects_unknown_format(self, client: HttpClient, transport: FixtureTransport, tmp_path) -> None:
        result = RCSBProvider(client).download_structure("1ABC", tmp_path / "x.xyz", fmt="xyz")
        assert result.ok is False
        assert transport.call_count == 0

    def test_download_writes_the_file(self, client: HttpClient, transport: FixtureTransport, tmp_path) -> None:
        transport.add("files.rcsb.org", StubResponse.text("data_1ABC\n"))
        target = tmp_path / "1ABC.cif"
        result = RCSBProvider(client).download_structure("1abc", target)
        assert result.ok
        assert target.read_text().startswith("data_1ABC")
        assert "1ABC.cif" in transport.urls()[0]


# ==========================================================================
# Materials Project
# ==========================================================================
class TestMaterialsProject:
    def test_requires_an_api_key(self, client: HttpClient, transport: FixtureTransport) -> None:
        provider = MaterialsProjectProvider(client)
        assert provider.configured() is False
        result = provider.search(formula="Si")
        assert result.ok is False
        assert result.error_type == "CredentialsMissing"
        assert transport.call_count == 0

    def test_valid_search_labels_the_domain(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add_json("api.materialsproject.org", fixture("materials_project_search.json"))
        result = MaterialsProjectProvider(client, api_key="k" * 20).search(formula="Si")
        assert result.ok
        assert result.records[0]["material_id"] == "mp-149"
        assert result.records[0]["domain"] == "inorganic-crystalline"

    def test_api_key_never_appears_in_provenance(self, client: HttpClient, transport: FixtureTransport) -> None:
        secret = "mp-secret-key-abcdef"
        transport.add_json("api.materialsproject.org", fixture("materials_project_search.json"))
        result = MaterialsProjectProvider(client, api_key=secret).search(formula="Si")
        assert secret not in json.dumps(result.provenance)
        assert secret not in json.dumps(result.as_dict(), default=str)

    def test_requires_a_query_criterion(self, client: HttpClient, transport: FixtureTransport) -> None:
        result = MaterialsProjectProvider(client, api_key="k" * 20).search()
        assert result.ok is False
        assert transport.call_count == 0

    def test_401_is_an_authentication_error(self, client: HttpClient, transport: FixtureTransport) -> None:
        transport.add("api.materialsproject.org", StubResponse.json({}, status=401))
        result = MaterialsProjectProvider(client, api_key="bad").search(formula="Si")
        assert result.error_type == "AuthenticationError"
