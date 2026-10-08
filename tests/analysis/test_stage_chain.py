"""Every stage list must be derived, not written down twice.

Adding an optional annealing stage broke this in two places at once: the campaign driver
wrote an anneal.mdp and then never ran it, and the executor put `anneal` in its ordering
while its input table had no entry for it. Both were hard-coded tuples that had to be
kept in step with the generator by hand.

These tests assert the lists agree, so the next optional stage fails here rather than in
a campaign.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polymer_engine.core.config import SimulationDefaults
from polymer_engine.executors.gromacs import (
    OPTIONAL_STAGES,
    REQUIRED_STAGES,
    STAGE_INPUTS,
    stage_chain,
)
from polymer_engine.simulation.mdp import STAGE_ORDER, generate_stages


def _defaults(**overrides) -> SimulationDefaults:
    return SimulationDefaults(nvt_ns=0.1, npt_ns=1.0, production_ns=1.0, **overrides)


def test_every_ordered_stage_has_an_input_mapping() -> None:
    """The exact mismatch that raised KeyError('anneal') mid-run."""
    assert set(STAGE_ORDER) == set(STAGE_INPUTS)


def test_required_and_optional_stages_partition_the_ordering() -> None:
    assert set(REQUIRED_STAGES) | OPTIONAL_STAGES == set(STAGE_ORDER)
    assert not set(REQUIRED_STAGES) & OPTIONAL_STAGES


def test_the_generator_and_the_executor_agree_without_the_anneal(tmp_path: Path) -> None:
    generated = [s.name for s in generate_stages(_defaults(), replica_index=1)]
    for name in generated:
        (tmp_path / f"{name}.mdp").write_text("x")
    assert [row[0] for row in stage_chain(tmp_path)] == generated


def test_the_generator_and_the_executor_agree_with_the_anneal(tmp_path: Path) -> None:
    defaults = _defaults(anneal_ns=1.0, anneal_temperature_k=500.0)
    generated = [s.name for s in generate_stages(defaults, replica_index=1)]
    assert "anneal" in generated
    for name in generated:
        (tmp_path / f"{name}.mdp").write_text("x")
    assert [row[0] for row in stage_chain(tmp_path)] == generated


def test_the_chain_relinks_when_an_optional_stage_appears(tmp_path: Path) -> None:
    """npt must read anneal.gro when the anneal ran, and nvt.gro when it did not."""
    for name in ("em", "nvt", "npt", "prod"):
        (tmp_path / f"{name}.mdp").write_text("x")
    without = {row[0]: row[1] for row in stage_chain(tmp_path)}
    assert without["npt"] == "nvt.gro"

    (tmp_path / "anneal.mdp").write_text("x")
    with_anneal = {row[0]: row[1] for row in stage_chain(tmp_path)}
    assert with_anneal["anneal"] == "nvt.gro"
    assert with_anneal["npt"] == "anneal.gro"


def test_each_stage_reads_what_the_previous_one_wrote(tmp_path: Path) -> None:
    for name in STAGE_ORDER:
        (tmp_path / f"{name}.mdp").write_text("x")
    chain = stage_chain(tmp_path)
    assert chain[0][1] == "system.gro"
    for (_stage, structure, _mdp, _prefix), previous in zip(chain[1:], chain[:-1], strict=True):
        assert structure == f"{previous[3]}.gro"


def test_the_generator_chains_its_own_structures_the_same_way() -> None:
    stages = generate_stages(_defaults(anneal_ns=1.0), replica_index=1)
    for stage, previous in zip(stages[1:], stages[:-1], strict=True):
        assert stage.input_structure == f"{previous.output_prefix}.gro"


@pytest.mark.parametrize("anneal_ns", [0.0, 5.0])
def test_a_disabled_anneal_reproduces_the_original_protocol(anneal_ns: float) -> None:
    names = [s.name for s in generate_stages(_defaults(anneal_ns=anneal_ns),
                                             replica_index=1)]
    expected = (["em", "nvt", "npt", "prod"] if anneal_ns == 0
                else ["em", "nvt", "anneal", "npt", "prod"])
    assert names == expected
