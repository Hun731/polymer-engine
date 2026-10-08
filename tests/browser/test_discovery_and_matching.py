"""Catalogue discovery, versioning, form schema and monomer matching (§18-§22, §68)."""

from __future__ import annotations

import pytest
from tests.browser.conftest import FakeDriver

from polymer_engine.browser.catalog import Catalog, MonomerEntry, diff
from polymer_engine.browser.credentials import Credentials
from polymer_engine.browser.discovery import (
    catalog_from_controls,
    discover_form_schema,
    find_polymer_builder,
    map_semantics,
)
from polymer_engine.browser.matching import normalise, resolve_monomer, resolve_option
from polymer_engine.browser.selectors import builder_schema_placeholder, login_schema
from polymer_engine.browser.session import Session
from polymer_engine.browser.states import BrowserState
from polymer_engine.core.config import Secret


@pytest.fixture
def session(pages: dict) -> Session:
    driver = FakeDriver(pages, start="home")
    return Session(driver, credentials=Credentials(email="a@b.c", password=Secret("x")),
                   base_url="https://example.invalid")


# -- the schema refuses to be guessed ------------------------------------
def test_the_builder_schema_starts_empty_and_refuses_to_invent_a_field() -> None:
    schema = builder_schema_placeholder()
    assert not schema.discovered
    assert schema.fields == {}
    with pytest.raises(KeyError, match="discovered from the live page"):
        schema.require("degree_of_polymerization")


def test_only_universal_html_structure_is_hard_coded() -> None:
    # A password input is a platform guarantee; a Polymer Builder field name is not.
    values = [loc.value for loc in login_schema().require("password").ordered()]
    assert "input[type='password']" in values


def test_no_locator_anywhere_uses_screen_coordinates() -> None:
    for schema in (login_schema(), builder_schema_placeholder()):
        for spec in schema.fields.values():
            for locator in spec.locators:
                assert locator.strategy in {
                    "css", "label", "role", "placeholder", "text", "test_id"}


# -- discovery -----------------------------------------------------------
def test_navigation_follows_visible_links(session: Session) -> None:
    result = find_polymer_builder(session)
    assert result.state is BrowserState.OK
    assert "builder" in result.data["url"]


def test_a_missing_builder_link_is_reported_not_guessed(pages: dict) -> None:
    pages["home"].links = [{"text": "News", "href": "https://example.invalid/news"}]
    driver = FakeDriver(pages, start="home")
    session = Session(driver, credentials=Credentials(email="a", password=Secret("b")),
                      base_url="https://example.invalid")
    result = find_polymer_builder(session)
    assert result.state is BrowserState.NAVIGATION_FAILED
    assert "links_seen" in result.data


def test_form_schema_is_derived_from_the_page(session: Session) -> None:
    session.navigate("https://example.invalid/builder")
    schema, _controls = discover_form_schema(session)
    assert schema.discovered
    assert set(schema.fields) == {"monomer", "dp", "nchain", "tacticity", "next"}
    assert schema.require("dp").control == "number"
    assert schema.require("dp").required
    assert schema.require("monomer").control == "select"


def test_semantics_are_suggestions_carrying_every_candidate(session: Session) -> None:
    session.navigate("https://example.invalid/builder")
    _schema, controls = discover_form_schema(session)
    semantics = map_semantics(controls)
    assert semantics["degree_of_polymerization"] == ["dp"]
    assert semantics["n_chains"] == ["nchain"]
    assert semantics["monomer"] == ["monomer"]


def test_placeholder_options_never_become_monomers(session: Session) -> None:
    session.navigate("https://example.invalid/builder")
    _schema, controls = discover_form_schema(session)
    catalog = catalog_from_controls(controls, semantics=map_semantics(controls))
    assert [m.value for m in catalog.monomers] == ["PE", "PLA"]
    assert all(m.value for m in catalog.monomers)


def test_an_unidentified_monomer_control_is_not_an_empty_catalogue(
    session: Session
) -> None:
    """Zero monomers because we could not find the control is a different claim from
    zero monomers because the control offers none."""
    from polymer_engine.browser.catalog import CatalogState

    session.navigate("https://example.invalid/builder")
    _schema, controls = discover_form_schema(session)
    catalog = catalog_from_controls(controls, semantics={})

    assert catalog.monomers == []
    assert catalog.state is CatalogState.NOT_IDENTIFIED
    assert catalog.state.needs_human
    assert not catalog.state.usable
    # The notes must not read as a statement about the site's chemistry.
    assert "does not mean the Polymer Builder offers no monomers" in catalog.notes
    # And the choice sets that *were* seen are recorded, so the failure is actionable.
    assert catalog.inspected["n_choice_sets"] >= 1


def test_a_captured_catalogue_says_so(session: Session) -> None:
    from polymer_engine.browser.catalog import CatalogState

    session.navigate("https://example.invalid/builder")
    _schema, controls = discover_form_schema(session)
    catalog = catalog_from_controls(controls, semantics=map_semantics(controls))
    assert catalog.state is CatalogState.CAPTURED
    assert catalog.state.usable


def test_monomers_offered_as_radio_buttons_are_found(session: Session) -> None:
    """A page that lists monomers as radios must not report zero."""
    from polymer_engine.browser.catalog import CatalogState

    session.navigate("https://example.invalid/builder")
    _schema, controls = discover_form_schema(session)
    inventory = {"radio_checkbox_groups": [{
        "kind": "radio", "name": "mono",
        "options": [{"value": "PE", "label": "Polyethylene"},
                    {"value": "PLA", "label": "Poly(lactic acid)"}]}]}
    catalog = catalog_from_controls(controls, semantics={"monomer": ["mono"]},
                                    inventory=inventory)
    assert catalog.state is CatalogState.CAPTURED
    assert sorted(m.value for m in catalog.monomers) == ["PE", "PLA"]
    assert {m.kind for m in catalog.monomers} == {"radio"}


# -- versioning ----------------------------------------------------------
def test_a_catalogue_never_claims_to_be_complete() -> None:
    assert Catalog().completeness == "unknown"
    assert "never described as complete" in Catalog().to_markdown()


def test_fingerprint_ignores_capture_time_and_tracks_content() -> None:
    a = Catalog(monomers=[MonomerEntry("PE", "Polyethylene", "m")], captured_at="t1")
    b = Catalog(monomers=[MonomerEntry("PE", "Polyethylene", "m")], captured_at="t2")
    assert a.fingerprint() == b.fingerprint()
    c = Catalog(monomers=[MonomerEntry("PE", "Polyethylene", "m"),
                          MonomerEntry("PP", "Polypropylene", "m")])
    assert c.fingerprint() != a.fingerprint()


def test_a_relabelling_is_singled_out_from_an_add_remove_pair() -> None:
    before = Catalog(monomers=[MonomerEntry("PE", "Polyethylene", "m")])
    after = Catalog(monomers=[MonomerEntry("PE", "Polyethylene (PE)", "m")])
    changes = diff(before, after)
    assert changes.changed
    assert changes.relabelled == [{"key": "m:PE", "before": "Polyethylene",
                                   "after": "Polyethylene (PE)"}]
    assert not changes.added and not changes.removed


def test_round_trip_through_json_preserves_the_fingerprint(tmp_path) -> None:
    catalog = Catalog(monomers=[MonomerEntry("PLA", "Poly(lactic acid)", "m")],
                      tacticity_options=["Atactic"], source_url="u")
    catalog.write(tmp_path)
    reloaded = Catalog.read(tmp_path)
    assert reloaded is not None
    assert reloaded.fingerprint() == catalog.fingerprint()


# -- matching ------------------------------------------------------------
@pytest.fixture
def catalog() -> Catalog:
    return Catalog(monomers=[
        MonomerEntry("PLA", "Poly(lactic acid)", "m"),
        MonomerEntry("PGA", "Poly(glycolic acid)", "m"),
        MonomerEntry("PE", "Polyethylene", "m"),
    ], tacticity_options=["Atactic", "Isotactic"])


def test_punctuation_and_case_are_the_only_latitude() -> None:
    assert normalise("Poly(lactic acid)") == normalise("poly lactic ACID")
    assert normalise("Poly(lactic acid)") != normalise("Poly(glycolic acid)")


@pytest.mark.parametrize("query", ["PLA", "Poly(lactic acid)", "poly lactic acid"])
def test_exact_and_normalised_matches_resolve(catalog: Catalog, query: str) -> None:
    match = resolve_monomer(catalog, query)
    assert match.resolved
    assert match.entry is not None
    assert match.entry.value == "PLA"


def test_a_real_synonym_is_still_refused(catalog: Catalog) -> None:
    # "polylactide" *is* PLA to a chemist. It is not the catalogue's name for it, and
    # acting on that would be a substitution nobody recorded.
    match = resolve_monomer(catalog, "polylactide")
    assert match.state is BrowserState.MONOMER_NOT_FOUND
    assert not match.resolved
    assert [c["value"] for c in match.candidates]  # suggested to a person, not chosen


def test_an_alias_is_a_recorded_human_decision(catalog: Catalog) -> None:
    match = resolve_monomer(catalog, "polylactide", aliases={"polylactide": "PLA"})
    assert match.resolved
    assert match.matched_on == "alias"


def test_a_stale_alias_is_refused_rather_than_followed(catalog: Catalog) -> None:
    match = resolve_monomer(catalog, "x", aliases={"x": "NOT_IN_CATALOG"})
    assert match.state is BrowserState.REQUIRES_HUMAN_REVIEW


def test_duplicate_labels_require_a_person(catalog: Catalog) -> None:
    catalog.monomers.append(MonomerEntry("PLA2", "Poly(lactic acid)", "m"))
    match = resolve_monomer(catalog, "Poly(lactic acid)")
    assert match.state is BrowserState.REQUIRES_HUMAN_REVIEW
    assert len(match.candidates) == 2


def test_an_empty_catalogue_is_not_a_match(catalog: Catalog) -> None:
    assert resolve_monomer(Catalog(), "PE").state is BrowserState.MONOMER_NOT_FOUND


def test_an_unoffered_option_lists_what_is_offered(catalog: Catalog) -> None:
    value, why = resolve_option(catalog.tacticity_options, "syndiotactic",
                                what="tacticity")
    assert value is None
    assert "Atactic, Isotactic" in why


def test_not_requesting_an_option_is_not_an_error(catalog: Catalog) -> None:
    assert resolve_option(catalog.tacticity_options, None, what="tacticity") == (None, "")
