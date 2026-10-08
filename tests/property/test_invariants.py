"""Property-based tests.

These check invariants that must hold for *all* inputs, not just the examples a
hand-written test happens to pick.  Each one encodes a rule the engine relies on:

* an extracted archive path can never escape its root
* state-transition validity survives arbitrary command sequences
* unit conversion is lossless and round-trips
* descriptor and identity resolution is order-independent
* statistical estimators stay inside their mathematical bounds
"""

from __future__ import annotations

import math
import string

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polymer_engine.core.errors import ParameterValidationError, UnsafeArchiveMember
from polymer_engine.core.models import (
    ALLOWED_TRANSITIONS,
    ExecutionRecord,
    ExecutionState,
    GateReport,
    GateResult,
    GateStatus,
    can_transition,
)
from polymer_engine.core.units import CANONICAL, convert, dimension_of

FAST = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# ==========================================================================
# Archive safety
# ==========================================================================
path_segments = st.lists(
    st.one_of(
        st.text(alphabet=string.ascii_letters + string.digits + "._-", min_size=1, max_size=8),
        st.just(".."),
        st.just("."),
        st.just(""),
    ),
    min_size=1,
    max_size=6,
)


@FAST
@given(segments=path_segments)
def test_extracted_paths_never_escape_the_root(segments: list[str], tmp_path_factory) -> None:
    """No member name can resolve outside the extraction root -- it lands inside or raises."""
    from polymer_engine.simulation.archive import _check_member_path

    destination = tmp_path_factory.mktemp("extract")
    name = "/".join(segments)
    try:
        target = _check_member_path(name, destination)
    except UnsafeArchiveMember:
        return  # rejecting is always an acceptable outcome
    assert destination in target.parents or target == destination


@FAST
@given(depth=st.integers(min_value=1, max_value=10))
def test_any_number_of_parent_traversals_is_rejected(depth: int, tmp_path_factory) -> None:
    from polymer_engine.simulation.archive import _check_member_path

    destination = tmp_path_factory.mktemp("extract")
    name = "/".join([".."] * depth) + "/payload.txt"
    with pytest.raises(UnsafeArchiveMember):
        _check_member_path(name, destination)


@FAST
@given(prefix=st.text(alphabet=string.ascii_letters, min_size=0, max_size=5))
def test_absolute_members_are_always_rejected(prefix: str, tmp_path_factory) -> None:
    from polymer_engine.simulation.archive import _check_member_path

    destination = tmp_path_factory.mktemp("extract")
    with pytest.raises(UnsafeArchiveMember):
        _check_member_path(f"/{prefix}/payload.txt", destination)


# ==========================================================================
# Execution state machine
# ==========================================================================
states = st.sampled_from(list(ExecutionState))


@FAST
@given(targets=st.lists(states, min_size=1, max_size=25))
def test_state_machine_never_enters_an_illegal_state(targets: list[ExecutionState]) -> None:
    """However a caller drives the record, its state is always legally reachable."""
    from polymer_engine.core.errors import IllegalStateTransition

    record = ExecutionRecord(kind="test")
    for target in targets:
        previous = record.state
        try:
            record.transition(target, actor="test", reason="property test")
        except IllegalStateTransition:
            assert record.state is previous, "a refused transition must not change state"
            continue
        assert target in ALLOWED_TRANSITIONS[previous]
        assert record.state is target


@FAST
@given(targets=st.lists(states, min_size=1, max_size=25))
def test_history_length_matches_successful_transitions(targets: list[ExecutionState]) -> None:
    from polymer_engine.core.errors import IllegalStateTransition

    record = ExecutionRecord(kind="test")
    successful = 0
    for target in targets:
        try:
            record.transition(target, actor="test", reason="r")
        except IllegalStateTransition:
            continue
        successful += 1
    assert len(record.history) == successful


@FAST
@given(state=states)
def test_terminal_states_have_no_outgoing_transitions(state: ExecutionState) -> None:
    if state.terminal:
        assert ALLOWED_TRANSITIONS[state] == frozenset()


@FAST
@given(source=states, target=states)
def test_completed_can_never_go_back_to_running(source: ExecutionState, target: ExecutionState) -> None:
    if source is ExecutionState.COMPLETED and target is ExecutionState.RUNNING:
        assert not can_transition(source, target)


# ==========================================================================
# Units
# ==========================================================================
KNOWN_UNITS = [
    "nm", "angstrom", "pm", "m",
    "ps", "fs", "ns", "us", "s",
    "kJ/mol", "kcal/mol", "J/mol", "eV", "hartree",
    "K", "degC",
    "bar", "atm", "Pa", "kPa", "MPa",
    "kg/m^3", "g/cm^3",
    "deg", "rad",
]
finite_values = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@FAST
@given(value=finite_values, unit=st.sampled_from(KNOWN_UNITS))
def test_conversion_round_trips(value: float, unit: str) -> None:
    """Converting to canonical units and back must be lossless."""
    canonical = CANONICAL[dimension_of(unit)]
    there = convert(value, unit, canonical)
    back = convert(there, canonical, unit)
    assert back == pytest.approx(value, rel=1e-9, abs=1e-9)


@FAST
@given(value=finite_values, a=st.sampled_from(KNOWN_UNITS), b=st.sampled_from(KNOWN_UNITS))
def test_conversion_only_succeeds_within_a_dimension(value: float, a: str, b: str) -> None:
    if dimension_of(a) == dimension_of(b):
        assert math.isfinite(convert(value, a, b))
    else:
        with pytest.raises(ParameterValidationError):
            convert(value, a, b)


@FAST
@given(value=finite_values, a=st.sampled_from(KNOWN_UNITS))
def test_identity_conversion_is_exact(value: float, a: str) -> None:
    assert convert(value, a, a) == pytest.approx(value, rel=1e-12, abs=1e-12)


@FAST
@given(value=st.floats(min_value=0.1, max_value=1e4, allow_nan=False))
def test_temperature_conversion_preserves_intervals(value: float) -> None:
    """A degC interval and a K interval are the same size; only the origin differs."""
    a = convert(value, "degC", "K")
    b = convert(value + 1.0, "degC", "K")
    assert (b - a) == pytest.approx(1.0, rel=1e-9)


# ==========================================================================
# Polymer identity
# ==========================================================================
REPEAT_UNITS = ["*CC*", "*CCO*", "*CC(C)*", "*CC(*)c1ccccc1", "*NCCCCCC(=O)*", "*CC(Cl)*", "*OC(C)C(=O)*"]


@FAST
@given(smiles=st.sampled_from(REPEAT_UNITS))
def test_identity_resolution_is_idempotent(smiles: str) -> None:
    from polymer_engine.polymer.identity import make_identity

    pytest.importorskip("rdkit")
    first = make_identity(name="a", repeat_unit_smiles=smiles)
    second = make_identity(name="b", repeat_unit_smiles=first.canonical_repeat_unit)
    assert first.polymer_id == second.polymer_id


@FAST
@given(order=st.permutations(REPEAT_UNITS))
def test_deduplication_is_order_independent(order: list[str]) -> None:
    from polymer_engine.polymer.identity import deduplicate, make_identity

    pytest.importorskip("rdkit")
    report = deduplicate([make_identity(name=s, repeat_unit_smiles=s) for s in order])
    assert len({i.polymer_id for i in report.unique}) == len(set(REPEAT_UNITS))


@FAST
@given(smiles=st.sampled_from(REPEAT_UNITS))
def test_capping_removes_every_attachment_point(smiles: str) -> None:
    from polymer_engine.polymer.identity import capped_monomer_smiles, count_attachment_points

    assert count_attachment_points(capped_monomer_smiles(smiles)) == 0


@FAST
@given(smiles=st.sampled_from(REPEAT_UNITS))
def test_generated_candidates_are_never_the_parent(smiles: str) -> None:
    from polymer_engine.discovery.candidates import generate_candidates
    from polymer_engine.polymer.identity import canonical_repeat_unit, derive_polymer_id

    pytest.importorskip("rdkit")
    parent_canonical, _ = canonical_repeat_unit(smiles)
    parent_id = derive_polymer_id(parent_canonical)
    for generated in generate_candidates(smiles, max_candidates=12):
        assert generated.polymer_id != parent_id


# ==========================================================================
# Statistics
# ==========================================================================
series = st.lists(
    st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=300,
)


@FAST
@given(values=series)
def test_statistical_inefficiency_is_at_least_one(values: list[float]) -> None:
    from polymer_engine.analysis.statistics import statistical_inefficiency

    assert statistical_inefficiency(values) >= 1.0


@FAST
@given(values=series)
def test_effective_samples_never_exceed_raw_samples(values: list[float]) -> None:
    from polymer_engine.analysis.statistics import effective_sample_size

    assert effective_sample_size(values) <= len(values) + 1e-9


@FAST
@given(values=series)
def test_mean_lies_within_the_data_range(values: list[float]) -> None:
    from polymer_engine.analysis.statistics import describe

    measurement = describe(values)
    assert min(values) - 1e-6 <= measurement.value <= max(values) + 1e-6


@FAST
@given(values=series)
def test_uncertainty_is_never_negative(values: list[float]) -> None:
    from polymer_engine.analysis.statistics import describe

    measurement = describe(values)
    assert measurement.uncertainty is None or measurement.uncertainty >= 0.0


@FAST
@given(values=series, shift=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False))
def test_uncertainty_is_translation_invariant(values: list[float], shift: float) -> None:
    """Adding a constant moves the mean but must not change the spread."""
    from polymer_engine.analysis.statistics import describe

    base = describe(values)
    shifted = describe([v + shift for v in values])
    if base.uncertainty is None or shifted.uncertainty is None:
        return
    assert shifted.uncertainty == pytest.approx(base.uncertainty, rel=1e-6, abs=1e-9)


@FAST
@given(
    a=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=4, max_size=60),
)
def test_correlation_is_bounded(a: list[float]) -> None:
    from polymer_engine.analysis.statistics import pearson

    b = list(reversed(a))
    result = pearson(a, b)
    if result.statistic is not None:
        assert -1.0 - 1e-9 <= result.statistic <= 1.0 + 1e-9


@FAST
@given(n_pass=st.integers(0, 5), n_warn=st.integers(0, 5), n_fail=st.integers(0, 5), n_inc=st.integers(0, 5))
def test_gate_aggregation_is_worst_case(n_pass: int, n_warn: int, n_fail: int, n_inc: int) -> None:
    """A report is never better than its worst gate, and never passes while empty."""
    gates = (
        [GateResult(gate=f"p{i}", status=GateStatus.PASS, message="") for i in range(n_pass)]
        + [GateResult(gate=f"w{i}", status=GateStatus.WARN, message="") for i in range(n_warn)]
        + [GateResult(gate=f"f{i}", status=GateStatus.FAIL, message="") for i in range(n_fail)]
        + [GateResult(gate=f"i{i}", status=GateStatus.INCONCLUSIVE, message="") for i in range(n_inc)]
    )
    report = GateReport(name="t", gates=gates)
    if n_fail:
        assert report.status is GateStatus.FAIL
    elif n_inc:
        assert report.status is GateStatus.INCONCLUSIVE
    elif n_warn:
        assert report.status is GateStatus.WARN
    elif n_pass:
        assert report.status is GateStatus.PASS
    else:
        assert report.status is GateStatus.INCONCLUSIVE
    if n_fail or n_inc or not gates:
        assert report.promotable is False


# ==========================================================================
# Simulation parameters
# ==========================================================================
@FAST
@given(
    base_seed=st.integers(min_value=1, max_value=2**31 - 2),
    replicas=st.integers(min_value=1, max_value=32),
)
def test_replica_seeds_are_distinct_and_in_range(base_seed: int, replicas: int) -> None:
    from polymer_engine.simulation.mdp import replica_seed

    seeds = [replica_seed(base_seed, i, stage) for i in range(replicas) for stage in ("nvt", "npt", "prod")]
    assert len(set(seeds)) == len(seeds), "replicas sharing a seed would not be independent"
    assert all(1 <= s < 2**31 for s in seeds)


@FAST
@given(base_seed=st.integers(min_value=1, max_value=2**31 - 2), index=st.integers(0, 63))
def test_replica_seeds_are_reproducible(base_seed: int, index: int) -> None:
    from polymer_engine.simulation.mdp import replica_seed

    assert replica_seed(base_seed, index, "nvt") == replica_seed(base_seed, index, "nvt")


@FAST
@given(
    ns=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
    dt=st.floats(min_value=0.0001, max_value=0.005, allow_nan=False),
)
def test_step_count_is_non_negative_and_consistent(ns: float, dt: float) -> None:
    from polymer_engine.simulation.mdp import steps_for

    steps = steps_for(ns, dt)
    assert steps >= 0
    assert steps == pytest.approx(ns * 1000.0 / dt, abs=1.0)


@FAST
@given(
    k=st.floats(min_value=10.0, max_value=100_000.0, allow_nan=False),
    temperature=st.floats(min_value=100.0, max_value=600.0, allow_nan=False),
)
def test_recommended_spacing_always_overlaps(k: float, temperature: float) -> None:
    """The spacing the planner recommends must satisfy its own overlap criterion."""
    from polymer_engine.simulation.umbrella import (
        MAX_SPACING_SIGMA,
        recommend_spacing,
        restraint_sigma,
    )

    spacing = recommend_spacing(k, temperature)
    assert spacing <= MAX_SPACING_SIGMA * restraint_sigma(k, temperature) * (1 + 1e-9)
