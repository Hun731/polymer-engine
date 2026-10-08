"""Capability verification, spec mapping and session recovery (§7, §23, §26, §27)."""

from __future__ import annotations

import pytest
from tests.browser.conftest import FakeDriver

from polymer_engine.browser.acquisition import CharmmGuiAcquisition
from polymer_engine.browser.credentials import Credentials
from polymer_engine.browser.discovery import discover_form_schema, map_semantics
from polymer_engine.browser.queue import BuildEntry, BuildQueue
from polymer_engine.browser.session import Session
from polymer_engine.browser.spec_mapping import (
    MappingVerdict,
    default_probe_spec,
    map_spec_to_form,
)
from polymer_engine.browser.states import BrowserState, JobState
from polymer_engine.browser.verification import (
    NEVER_GRANTED_HERE,
    VerificationRegistry,
    VerificationState,
)
from polymer_engine.core.config import Secret
from polymer_engine.simulation.charmm_gui_spec import PolymerBuilderSpec, SystemType


# -- §26 / §27 capability verification -----------------------------------
@pytest.fixture
def registry(tmp_path) -> VerificationRegistry:
    return VerificationRegistry(tmp_path / "verification.json")


def test_everything_starts_unproven(registry: VerificationRegistry) -> None:
    assert registry.live_verified() == []
    assert registry.next_unproven() == "LIVE_LOGIN_VERIFIED"
    assert not registry.system_generation_verified


def test_a_capability_cannot_be_verified_without_evidence(
    registry: VerificationRegistry
) -> None:
    with pytest.raises(ValueError, match="no evidence"):
        registry.verify("LIVE_LOGIN_VERIFIED", evidence={})


def test_a_capability_cannot_skip_its_prerequisites(registry: VerificationRegistry) -> None:
    with pytest.raises(ValueError, match="cannot be verified before"):
        registry.verify("DOWNLOAD_VERIFIED", evidence={"sha256": "abc"})


def test_fixture_verification_is_a_lower_claim_that_never_becomes_live(
    registry: VerificationRegistry
) -> None:
    registry.fixture_verified("LIVE_LOGIN_VERIFIED", "local fixtures only")
    record = registry.records["LIVE_LOGIN_VERIFIED"]
    assert record.state is VerificationState.FIXTURE_VERIFIED
    assert not record.state.proven_live
    assert registry.live_verified() == []


@pytest.mark.parametrize("claim", sorted(NEVER_GRANTED_HERE))
def test_qualification_can_never_be_granted_by_acquisition(
    registry: VerificationRegistry, claim: str
) -> None:
    """§27: a successful build is not a qualified force field."""
    with pytest.raises((ValueError, KeyError)):
        registry.verify(claim, evidence={"job_id": "J1"})


DISCOVERY = ("LIVE_LOGIN_VERIFIED", "POLYMER_BUILDER_REACHED", "FORM_SCHEMA_VERIFIED",
             "CATALOG_VERIFIED", "SPEC_MAPPING_VERIFIED")
POST_BUILD = ("JOB_ID_VERIFIED", "JOB_MONITORING_VERIFIED", "DOWNLOAD_VERIFIED",
              "SYSTEM_IMPORT_VERIFIED", "GROMACS_VALIDATION_VERIFIED")


def _discover(registry: VerificationRegistry) -> None:
    for name in DISCOVERY:
        registry.verify(name, evidence={"proof": name})


def test_the_dependency_graph_is_acyclic_and_well_formed() -> None:
    """A cycle would make a capability unreachable and its reason unreadable."""
    from polymer_engine.browser.verification import dependency_problems

    assert dependency_problems() == []


def test_no_discovery_state_depends_on_a_build_artifact() -> None:
    """The defect that shipped: a flat list made discovery depend on downloads."""
    from polymer_engine.browser.verification import ANY_OF, PREREQUISITES

    build_side = {"SINGLE_CHAIN_BUILD_VERIFIED", "MELT_BUILD_VERIFIED",
                  "JOB_ID_VERIFIED", "JOB_MONITORING_VERIFIED", "DOWNLOAD_VERIFIED",
                  "SYSTEM_IMPORT_VERIFIED", "GROMACS_VALIDATION_VERIFIED"}
    for name in DISCOVERY:
        reachable = set(PREREQUISITES[name]) | set(ANY_OF.get(name, ()))
        assert not (reachable & build_side), (
            f"{name} is a discovery state but depends on {reachable & build_side}")


def test_form_schema_and_catalogue_are_siblings(
    registry: VerificationRegistry
) -> None:
    """§9: a page with a valid form can prove its schema without a catalogue."""
    registry.verify("LIVE_LOGIN_VERIFIED", evidence={"url": "x"})
    registry.verify("POLYMER_BUILDER_REACHED", evidence={"url": "y"})

    registry.verify("FORM_SCHEMA_VERIFIED", evidence={"n_fields": 8})
    assert registry.records["FORM_SCHEMA_VERIFIED"].state is VerificationState.LIVE_VERIFIED
    assert registry.records["CATALOG_VERIFIED"].state is VerificationState.NOT_IMPLEMENTED

    # And the reverse: a catalogue does not need the form schema.
    fresh = VerificationRegistry(registry.path.with_name("other.json"))
    fresh.verify("LIVE_LOGIN_VERIFIED", evidence={"url": "x"})
    fresh.verify("POLYMER_BUILDER_REACHED", evidence={"url": "y"})
    fresh.verify("CATALOG_VERIFIED", evidence={"n_monomers": 3})
    assert fresh.records["CATALOG_VERIFIED"].state is VerificationState.LIVE_VERIFIED


def test_a_failure_names_the_edge_that_is_actually_missing(
    registry: VerificationRegistry
) -> None:
    """The old message talked about downloads whatever the capability was."""
    registry.verify("LIVE_LOGIN_VERIFIED", evidence={"url": "x"})
    with pytest.raises(ValueError) as excinfo:
        registry.verify("FORM_SCHEMA_VERIFIED", evidence={"n_fields": 8})
    message = str(excinfo.value)
    assert "POLYMER_BUILDER_REACHED" in message
    assert "download" not in message.lower()
    assert "job it came from" not in message


def test_nothing_downstream_of_a_build_can_be_proven_without_one(
    registry: VerificationRegistry
) -> None:
    _discover(registry)
    # There is no job, so there is nothing to monitor, download or import.
    for name in POST_BUILD:
        with pytest.raises(ValueError):
            registry.verify(name, evidence={"proof": name})


def test_system_generation_needs_every_prerequisite_and_one_build(
    registry: VerificationRegistry
) -> None:
    _discover(registry)
    with pytest.raises(ValueError, match="cannot be verified before"):
        registry.verify("SYSTEM_GENERATION_VERIFIED", evidence={"polymer": "PE"})

    # One build of either kind unlocks the rest of the chain.
    registry.verify("SINGLE_CHAIN_BUILD_VERIFIED", evidence={"job_id": "J1"})
    for name in POST_BUILD:
        registry.verify(name, evidence={"proof": name})
    registry.verify("SYSTEM_GENERATION_VERIFIED", evidence={"polymer": "PE"})

    assert registry.system_generation_verified
    # And a melt was never built, so that capability stays unproven.
    assert registry.records["MELT_BUILD_VERIFIED"].state is (
        VerificationState.NOT_IMPLEMENTED)


def test_the_registry_round_trips(registry: VerificationRegistry) -> None:
    registry.verify("LIVE_LOGIN_VERIFIED", evidence={"url": "https://charmm-gui.org/"})
    registry.save()
    reloaded = VerificationRegistry(registry.path)
    assert reloaded.records["LIVE_LOGIN_VERIFIED"].state is VerificationState.LIVE_VERIFIED
    assert reloaded.records["LIVE_LOGIN_VERIFIED"].evidence["url"]


def test_the_markdown_states_what_it_cannot_say(registry: VerificationRegistry) -> None:
    assert "does **not** state that the force field is qualified" in registry.to_markdown()


# -- §7 specification mapping --------------------------------------------
def _mapping(pages: dict, spec: PolymerBuilderSpec):
    driver = FakeDriver(pages, start="builder")
    session = Session(driver, credentials=Credentials(email="a", password=Secret("b")),
                      base_url="https://example.invalid")
    schema, controls = discover_form_schema(session)
    return map_spec_to_form(spec, schema, map_semantics(controls))


def test_fields_the_form_offers_are_mapped(pages: dict) -> None:
    spec = default_probe_spec("Polyethylene", system_type=SystemType.MELT)
    spec.n_chains = 4
    mapping = _mapping(pages, spec)
    verdicts = {f.spec_field: f.verdict for f in mapping.fields}
    assert verdicts["name"] is MappingVerdict.MAPPED
    assert verdicts["degree_of_polymerization"] is MappingVerdict.MAPPED
    assert verdicts["n_chains"] is MappingVerdict.MAPPED


def test_a_field_the_request_does_not_set_needs_no_mapping(pages: dict) -> None:
    spec = default_probe_spec("Polyethylene")
    mapping = _mapping(pages, spec)
    tacticity = next(f for f in mapping.fields if f.spec_field == "tacticity")
    assert tacticity.verdict is MappingVerdict.NOT_REQUESTED
    assert not tacticity.blocks_build


def test_a_requested_field_the_form_lacks_blocks_the_build(pages: dict) -> None:
    spec = default_probe_spec("Polyethylene")
    spec.solvent = "water"
    mapping = _mapping(pages, spec)
    solvent = next(f for f in mapping.fields if f.spec_field == "solvent")
    assert solvent.verdict is MappingVerdict.UNAVAILABLE
    assert solvent.blocks_build
    # And the reason does not claim the form cannot do it.
    assert "did not reach" in solvent.reason


def test_an_undiscovered_schema_maps_nothing_and_says_why(pages: dict) -> None:
    from polymer_engine.browser.selectors import builder_schema_placeholder

    mapping = map_spec_to_form(default_probe_spec("PE"),
                               builder_schema_placeholder(), {})
    assert not mapping.buildable
    assert not mapping.schema_discovered
    assert all("not evidence that the form lacks" in f.reason
               for f in mapping.fields if f.verdict is MappingVerdict.UNAVAILABLE)


def test_ambiguity_is_never_resolved_by_sort_order(pages: dict) -> None:
    driver = FakeDriver(pages, start="builder")
    session = Session(driver, credentials=Credentials(email="a", password=Secret("b")),
                      base_url="https://example.invalid")
    schema, controls = discover_form_schema(session)
    semantics = map_semantics(controls)
    semantics["degree_of_polymerization"] = ["dp", "nchain"]
    mapping = map_spec_to_form(default_probe_spec("Polyethylene"), schema, semantics)
    dp = next(f for f in mapping.fields if f.spec_field == "degree_of_polymerization")
    assert dp.verdict is MappingVerdict.AMBIGUOUS
    assert dp.blocks_build
    assert set(dp.control_keys) == {"dp", "nchain"}


def test_controls_the_specification_does_not_describe_are_surfaced(pages: dict) -> None:
    """A control nobody mapped is a setting left at the form's default."""
    driver = FakeDriver(pages, start="builder")
    session = Session(driver, credentials=Credentials(email="a", password=Secret("b")),
                      base_url="https://example.invalid")
    schema, controls = discover_form_schema(session)
    semantics = map_semantics(controls)
    # A control the semantic map does not claim -- e.g. an option we have no field for.
    semantics.pop("tacticity")
    mapping = map_spec_to_form(default_probe_spec("Polyethylene"), schema, semantics)

    assert "tacticity" in mapping.unmapped_controls
    assert "not errors" in mapping.to_markdown()


# -- §23 session recovery ------------------------------------------------
def _queued(queue: BuildQueue, **overrides) -> BuildEntry:
    spec = PolymerBuilderSpec(
        polymer_id="pe", name="Polyethylene", repeat_unit_smiles="CC",
        degree_of_polymerization=10, n_chains=1, force_field="CHARMM36",
        temperature_k=300.0, pressure_bar=1.0, system_type=SystemType.SINGLE_CHAIN)
    entry = queue.add(BuildEntry(
        polymer_id="pe", polymer_name="Polyethylene",
        request_fingerprint=spec.fingerprint(), system_type="single_chain",
        rationale="first live build", spec=spec.as_dict(), **overrides))
    return entry


def test_a_submitted_job_is_monitored_after_a_lost_session(tmp_path) -> None:
    queue = BuildQueue(tmp_path / "q.json")
    entry = _queued(queue)
    entry.job_id = "J-REAL"
    entry.advance(JobState.RUNNING, "submitted before the session died")

    # No provider configured: resume must still not resubmit.
    results = CharmmGuiAcquisition(queue=queue, root=tmp_path).resume()
    assert len(results) == 1
    assert results[0].job_id == "J-REAL"
    assert entry.state is not JobState.QUEUED, "a submitted job must never requeue"


def test_a_job_with_no_captured_id_blocks_rather_than_resubmitting(tmp_path) -> None:
    queue = BuildQueue(tmp_path / "q.json")
    entry = _queued(queue)
    entry.advance(JobState.SUBMITTED, "clicked, then the session died")

    results = CharmmGuiAcquisition(queue=queue, root=tmp_path).resume()
    assert entry.state is JobState.BLOCKED
    assert results[0].state is BrowserState.REQUIRES_HUMAN_REVIEW
    assert "may exist" in results[0].reason
    assert not queue.may_submit(entry.request_fingerprint).may_submit


def test_a_specification_that_does_not_round_trip_is_not_reconstructed(tmp_path) -> None:
    """Monitoring the right job with the wrong spec would validate the wrong request."""
    from polymer_engine.browser.acquisition import _spec_from_entry

    queue = BuildQueue(tmp_path / "q.json")
    entry = _queued(queue)
    entry.spec["degree_of_polymerization"] = 999  # tampered or stale
    assert _spec_from_entry(entry) is None


def test_finished_work_is_left_alone_by_resume(tmp_path) -> None:
    queue = BuildQueue(tmp_path / "q.json")
    entry = _queued(queue)
    entry.job_id = "J1"
    entry.advance(JobState.VALIDATED, "already done")
    assert CharmmGuiAcquisition(queue=queue, root=tmp_path).resume() == []
    assert entry.state is JobState.VALIDATED
