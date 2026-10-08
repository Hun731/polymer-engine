"""Turning CHARMM-GUI's onclick monomer menu into a catalogue.

The live page presents two shapes through one mechanism, and confusing them silently
loses polymers:

    Polystyrene -> isotactic (R), isotactic (S), syndio (R), atactic
        one polymer, four conformations, all sharing the value STYR

    Amides -> Polyamide, Polyamide (inv), Nylon 3, Nylon 6
        a chemical class holding four polymers, each with its own value

Reading the second as conformations would drop Nylon 6, poly(ethylene terephthalate),
PTFE, polyketone and poly(ethylene oxide) while leaving a plausible catalogue behind.
"""

from __future__ import annotations

import pytest

from polymer_engine.browser.catalog import CatalogState
from polymer_engine.browser.discovery import (
    catalog_from_controls,
    monomers_from_handler_choices,
)

TACTICITIES = ["isotactic (R)", "isotactic (S)", "syndio (R)", "atactic"]


def _group(name: str, options: list[tuple[str, str]], *, duplicated: bool = True):
    """One menu group. Duplicated by default: the page keeps a hidden skeleton clone."""
    entries = [{"text": text, "value": value} for text, value in options]
    return {"group": name, "options": entries * (2 if duplicated else 1)}


def _choices(*groups):
    return {"ok": True, "handler": "set_monomer",
            "n_nodes": sum(len(g["options"]) for g in groups), "groups": list(groups)}


def test_shared_value_means_one_polymer_in_several_conformations() -> None:
    entries = monomers_from_handler_choices(_choices(
        _group("Polystyrene", [(t, "STYR") for t in TACTICITIES])))
    assert len(entries) == 1
    assert entries[0].label == "Polystyrene"
    assert entries[0].value == "STYR"
    assert list(entries[0].variants) == TACTICITIES


def test_distinct_values_mean_distinct_polymers() -> None:
    entries = monomers_from_handler_choices(_choices(
        _group("Amides", [("Polyamide", "AMIDU"), ("Nylon 3", "NYL3"),
                          ("Nylon 6", "NYL6")])))
    assert sorted(e.label for e in entries) == ["Nylon 3", "Nylon 6", "Polyamide"]
    assert all(e.variants == () for e in entries)
    # The class name is not itself a monomer.
    assert "Amides" not in {e.label for e in entries}


def test_a_class_holding_exactly_one_polymer_does_not_swallow_it() -> None:
    """`Halides -> Polytetrafluoroethylene` must yield PTFE, not "Halides"."""
    entries = monomers_from_handler_choices(_choices(
        _group("Polystyrene", [(t, "STYR") for t in TACTICITIES]),
        _group("Halides", [("Polytetrafluoroethylene", "TEFET")]),
    ))
    labels = {e.label for e in entries}
    assert "Polytetrafluoroethylene" in labels
    assert "Halides" not in labels


def test_a_lone_conformation_still_belongs_to_its_group() -> None:
    """The mirror case: one option that *is* a conformation seen elsewhere."""
    entries = monomers_from_handler_choices(_choices(
        _group("Polystyrene", [(t, "STYR") for t in TACTICITIES]),
        _group("Polyethylene", [("atactic", "ETHY")]),
    ))
    by_label = {e.label: e for e in entries}
    assert "Polyethylene" in by_label
    assert list(by_label["Polyethylene"].variants) == ["atactic"]
    assert "atactic" not in by_label


def test_the_skeleton_clone_is_not_counted_twice() -> None:
    entries = monomers_from_handler_choices(_choices(
        _group("Polystyrene", [(t, "STYR") for t in TACTICITIES], duplicated=True)))
    assert list(entries[0].variants) == TACTICITIES  # four, not eight


def test_conformation_groups_that_are_not_tacticity_still_work() -> None:
    entries = monomers_from_handler_choices(_choices(
        _group("Polybutadiene", [("trans", "13BD"), ("cis", "13BD")]),
        _group("Polyisoprene", [("trans", "14IP"), ("cis", "14IP")])))
    assert sorted(e.label for e in entries) == ["Polybutadiene", "Polyisoprene"]
    assert all(list(e.variants) == ["trans", "cis"] for e in entries)


@pytest.mark.parametrize("name", ["(ungrouped)", "#picker", "", "   "])
def test_an_unnamed_group_is_dropped_rather_than_invented(name: str) -> None:
    assert monomers_from_handler_choices(_choices(
        _group(name, [("atactic", "X")]))) == []


def test_a_handler_list_alone_reaches_captured() -> None:
    catalog = catalog_from_controls([], semantics={}, handler_choices=_choices(
        _group("Polystyrene", [(t, "STYR") for t in TACTICITIES]),
        _group("Amides", [("Nylon 6", "NYL6")])))
    assert catalog.state is CatalogState.CAPTURED
    assert sorted(m.label for m in catalog.monomers) == ["Nylon 6", "Polystyrene"]
    assert catalog.inspected["handler"]["n_groups"] == 2
