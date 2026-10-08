"""Polymer identity, canonicalisation, deduplication, and family classification."""

from __future__ import annotations

import pytest

from polymer_engine.core.errors import ChemistryError
from polymer_engine.core.models import Determination
from polymer_engine.polymer.identity import (
    build_oligomer,
    canonical_repeat_unit,
    capped_monomer_smiles,
    count_attachment_points,
    deduplicate,
    derive_polymer_id,
    make_identity,
    normalize_smiles,
    same_polymer,
    validate_repeat_unit,
)
from polymer_engine.polymer.taxonomy import PolymerFamily, classify
from tests.markers import requires_rdkit


class TestAttachmentPoints:
    @pytest.mark.parametrize(
        "smiles,expected",
        [("*CC*", 2), ("[*]CC[*]", 2), ("CCO", 0), ("*CC(*)C*", 3), ("[1*]CC[2*]", 2)],
    )
    def test_counting(self, smiles: str, expected: int) -> None:
        assert count_attachment_points(smiles) == expected

    def test_capping_replaces_every_attachment_point(self) -> None:
        assert count_attachment_points(capped_monomer_smiles("*CC*")) == 0

    def test_capping_a_closed_molecule_is_a_no_op(self) -> None:
        assert capped_monomer_smiles("CCO") == "CCO"

    def test_empty_smiles_is_rejected(self) -> None:
        with pytest.raises(ChemistryError):
            normalize_smiles("   ")


class TestValidation:
    def test_a_good_repeat_unit_has_no_problems(self) -> None:
        assert validate_repeat_unit("*CC*") == []

    def test_a_discrete_molecule_is_flagged(self) -> None:
        problems = validate_repeat_unit("CCO")
        assert any("no attachment point" in p for p in problems)

    def test_a_single_attachment_point_is_flagged(self) -> None:
        problems = validate_repeat_unit("*CCO")
        assert any("only one attachment point" in p for p in problems)

    @requires_rdkit
    def test_unparseable_structure_is_flagged(self) -> None:
        problems = validate_repeat_unit("*C(((C*")
        assert any("could not be parsed" in p for p in problems)


@requires_rdkit
class TestCanonicalisation:
    def test_notation_variants_resolve_identically(self) -> None:
        a, _ = canonical_repeat_unit("*CC*")
        b, _ = canonical_repeat_unit("[*]CC[*]")
        assert a == b

    def test_canonicalisation_is_marked_known_with_rdkit(self) -> None:
        _, determination = canonical_repeat_unit("*CCO*")
        assert determination is Determination.KNOWN

    def test_atom_ordering_does_not_change_identity(self) -> None:
        first = make_identity(name="a", repeat_unit_smiles="*CC(C)(*)C(=O)OC")
        second = make_identity(name="b", repeat_unit_smiles="*C(C(=O)OC)(C)C*")
        assert same_polymer(first, second)

    def test_unparseable_smiles_raises(self) -> None:
        with pytest.raises(ChemistryError):
            canonical_repeat_unit("*C(((C*")


class TestIdentity:
    def test_id_is_deterministic(self) -> None:
        assert derive_polymer_id("*CC*") == derive_polymer_id("*CC*")

    def test_architecture_changes_the_id(self) -> None:
        linear = derive_polymer_id("*CC*", architecture="linear")
        network = derive_polymer_id("*CC*", architecture="network")
        assert linear != network, "a crosslinked network is not the same material as a linear chain"

    def test_tacticity_changes_the_id(self) -> None:
        assert derive_polymer_id("*CC(C)*", tacticity="isotactic") != derive_polymer_id(
            "*CC(C)*", tacticity="atactic"
        )

    def test_issues_are_attached_not_raised_by_default(self) -> None:
        identity = make_identity(name="ethanol", repeat_unit_smiles="CCO")
        assert identity.issues
        assert identity.usable is False

    def test_strict_mode_raises(self) -> None:
        with pytest.raises(ChemistryError):
            make_identity(name="ethanol", repeat_unit_smiles="CCO", strict=True)


@requires_rdkit
class TestDeduplication:
    def test_notation_duplicates_collapse(self) -> None:
        identities = [
            make_identity(name="PE", repeat_unit_smiles="*CC*"),
            make_identity(name="polyethylene", repeat_unit_smiles="[*]CC[*]"),
            make_identity(name="PEO", repeat_unit_smiles="*CCO*"),
        ]
        report = deduplicate(identities)
        assert len(report.unique) == 2
        assert report.n_duplicates == 1
        assert "polyethylene" in next(iter(report.duplicates.values()))

    def test_empty_input(self) -> None:
        report = deduplicate([])
        assert report.unique == []
        assert report.n_duplicates == 0


@requires_rdkit
class TestOligomerConstruction:
    def test_trimer_has_three_times_the_backbone(self) -> None:
        built = build_oligomer("*CC*", 3)
        assert built is not None
        _, backbone = built
        assert len(backbone) == 6

    def test_amide_survives_oligomerisation(self) -> None:
        """Capping nylon-6 destroys its amide; joining units must not."""
        from rdkit import Chem

        built = build_oligomer("*NCCCCCC(=O)*", 3)
        assert built is not None
        mol, _ = built
        amide = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")
        assert len(mol.GetSubstructMatches(amide)) >= 2

        capped = Chem.MolFromSmiles(capped_monomer_smiles("*NCCCCCC(=O)*"))
        assert capped.GetSubstructMatches(amide) == (), "the capped monomer has no amide at all"

    def test_ether_oxygen_survives_and_no_false_hydroxyl_inside(self) -> None:
        from rdkit import Chem

        built = build_oligomer("*CCO*", 4)
        assert built is not None
        mol, _ = built
        hydroxyls = mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H1]"))
        assert len(hydroxyls) == 1, "only the terminal cap may be a hydroxyl"

    def test_branched_repeat_unit_has_no_unique_backbone(self) -> None:
        assert build_oligomer("*CC(*)C*", 3) is None

    def test_single_attachment_point_cannot_form_a_chain(self) -> None:
        assert build_oligomer("*CCO", 3) is None


@requires_rdkit
class TestClassification:
    @pytest.mark.parametrize(
        "smiles,family",
        [
            ("*CC*", PolymerFamily.POLYOLEFIN),
            ("*CC(C)*", PolymerFamily.POLYOLEFIN),
            ("*CC(*)c1ccccc1", PolymerFamily.POLYSTYRENIC),
            ("*CC(Cl)*", PolymerFamily.POLYVINYL_HALIDE),
            ("*C(F)(F)C(F)(F)*", PolymerFamily.POLYVINYL_HALIDE),
            ("*CC(O)*", PolymerFamily.POLYVINYL_ALCOHOL),
            ("*CCO*", PolymerFamily.POLYETHER),
            ("*NCCCCCC(=O)*", PolymerFamily.POLYAMIDE),
            ("*OCCOC(=O)c1ccc(cc1)C(=O)*", PolymerFamily.POLYESTER),
            ("*OC(C)C(=O)*", PolymerFamily.POLYESTER),
            ("*CC(C)(*)C(=O)OC", PolymerFamily.POLYACRYLATE),
            ("*CC(*)C(=O)N", PolymerFamily.POLYACRYLAMIDE),
            ("*CC(*)C#N", PolymerFamily.POLYNITRILE),
            ("*[Si](C)(C)O*", PolymerFamily.POLYSILOXANE),
            ("*OC(=O)Oc1ccc(cc1)C(C)(C)c1ccc(cc1)*", PolymerFamily.POLYCARBONATE),
            ("*OCCOC(=O)Nc1ccc(cc1)N*", PolymerFamily.POLYURETHANE),
        ],
    )
    def test_known_polymers(self, smiles: str, family: PolymerFamily) -> None:
        result = classify(smiles)
        assert result.family is family
        assert result.confident is True

    def test_in_chain_ester_and_pendant_ester_are_different_families(self) -> None:
        """PET and PMMA both contain C(=O)O and are not the same kind of material."""
        pet = classify("*OCCOC(=O)c1ccc(cc1)C(=O)*")
        pmma = classify("*CC(C)(*)C(=O)OC")
        assert pet.family is PolymerFamily.POLYESTER
        assert pmma.family is PolymerFamily.POLYACRYLATE

    def test_capping_artifact_does_not_create_a_false_alcohol(self) -> None:
        """Poly(ethylene oxide) has no hydroxyl; a naive capped classifier says it does."""
        assert classify("*CCO*").family is PolymerFamily.POLYETHER

    def test_unknown_chemistry_is_unclassified_not_guessed(self) -> None:
        result = classify("*[Se][Se]*")
        assert result.family is PolymerFamily.UNCLASSIFIED
        assert result.confident is False
        assert result.requires_review is True

    def test_unparseable_input_is_unclassified(self) -> None:
        result = classify("*C(((C*")
        assert result.family is PolymerFamily.UNCLASSIFIED
        assert result.confident is False
