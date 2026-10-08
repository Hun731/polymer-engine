"""OPLS-AA typing: what it will assign, and what it refuses to.

The refusals matter more than the assignments here. A typer that silently produces a
topology for a chemistry it has no charges for yields a simulation that runs, looks
plausible, and is wrong.
"""

from __future__ import annotations

import pytest
from tests.markers import requires_gromacs, requires_rdkit

from polymer_engine.core.errors import ChemistryError
from polymer_engine.simulation.opls_typing import (
    NEUTRALITY_TOLERANCE,
    TypingStatus,
    find_opls_directory,
    load_opls_types,
    type_repeat_unit,
)

pytestmark = requires_gromacs   # OPLS-AA ships with GROMACS


@pytest.fixture(scope="module")
def opls():
    directory = find_opls_directory()
    if directory is None:
        pytest.skip("no oplsaa.ff found beside the gmx executable; cannot test OPLS typing")
    return directory, load_opls_types(directory)


class TestForceFieldLoading:
    def test_the_installed_force_field_is_read_not_hard_coded(self, opls) -> None:
        directory, types = opls
        assert (directory / "ffnonbonded.itp").is_file()
        assert len(types) > 500, "OPLS-AA should define hundreds of atom types"

    def test_the_alkane_types_carry_the_published_charges(self, opls) -> None:
        """These four charges are what make hydrocarbon typing possible at all."""
        _, types = opls
        assert types["opls_135"].charge == pytest.approx(-0.18)   # CH3
        assert types["opls_136"].charge == pytest.approx(-0.12)   # CH2
        assert types["opls_137"].charge == pytest.approx(-0.06)   # CH
        assert types["opls_139"].charge == pytest.approx(0.00)    # C
        assert types["opls_140"].charge == pytest.approx(+0.06)   # H

    def test_every_alkane_group_is_neutral_on_its_own(self, opls) -> None:
        """The property the whole hydrocarbon route depends on."""
        _, types = opls
        h = types["opls_140"].charge
        for name, n_hydrogens in (("opls_135", 3), ("opls_136", 2), ("opls_137", 1), ("opls_139", 0)):
            group = types[name].charge + n_hydrogens * h
            assert group == pytest.approx(0.0, abs=1e-12), f"{name} group charge is {group}"

    def test_the_alkane_types_share_one_bonded_type(self, opls) -> None:
        """All CT, so bonds/angles/dihedrals come from the same published set."""
        _, types = opls
        assert {types[n].bonded_type for n in ("opls_135", "opls_136", "opls_137", "opls_139")} == {"CT"}

    def test_a_missing_force_field_raises_rather_than_returning_empty(self, tmp_path) -> None:
        with pytest.raises(ChemistryError, match=r"ffnonbonded\.itp"):
            load_opls_types(tmp_path)

    def test_discovery_returns_none_rather_than_guessing(self, tmp_path) -> None:
        assert find_opls_directory(tmp_path) is None


@requires_rdkit
class TestHydrocarbonsAreTypable:
    @pytest.mark.parametrize(
        ("name", "repeat_unit"),
        [("polyethylene", "*CC*"), ("polypropylene", "*CC(C)*"), ("polyisobutylene", "*CC(C)(C)*")],
    )
    def test_a_saturated_polyolefin_is_fully_typed_and_neutral(self, opls, name, repeat_unit) -> None:
        directory, types = opls
        result = type_repeat_unit(repeat_unit, types=types, force_field_source=str(directory))
        assert result.status is TypingStatus.SUPPORTED, result.reason
        assert result.untyped == []
        assert abs(result.net_charge) <= NEUTRALITY_TOLERANCE
        assert result.usable is True

    def test_every_assignment_names_a_real_type_from_the_installed_file(self, opls) -> None:
        directory, types = opls
        result = type_repeat_unit("*CC(C)*", types=types, force_field_source=str(directory))
        for _index, name, bonded, _charge in result.assignments:
            assert name in types
            assert types[name].bonded_type == bonded

    def test_a_branched_polyolefin_uses_the_ch_type(self, opls) -> None:
        _, types = opls
        assigned = {a[1] for a in type_repeat_unit("*CC(C)*", types=types).assignments}
        assert "opls_137" in assigned, "the methine carbon should be typed as alkane CH"

    def test_a_quaternary_polyolefin_uses_the_quaternary_type(self, opls) -> None:
        _, types = opls
        assigned = {a[1] for a in type_repeat_unit("*CC(C)(C)*", types=types).assignments}
        assert "opls_139" in assigned, "polyisobutylene has a quaternary backbone carbon"


@requires_rdkit
class TestNonHydrocarbonsAreRefused:
    """Each of these would be a wrong simulation, not a missing feature."""

    @pytest.mark.parametrize(
        ("name", "repeat_unit", "element"),
        [
            ("poly(vinyl chloride)", "*CC(Cl)*", "Cl"),
            ("polytetrafluoroethylene", "*C(F)(F)C(F)(F)*", "F"),
            ("poly(vinyl alcohol)", "*CC(O)*", "O"),
            ("nylon-6", "*NCCCCCC(=O)*", "N"),
            ("polyacrylonitrile", "*CC(*)C#N", "N"),
        ],
    )
    def test_a_heteroatom_polymer_is_unsupported(self, opls, name, repeat_unit, element) -> None:
        _, types = opls
        result = type_repeat_unit(repeat_unit, types=types)
        assert result.status is TypingStatus.UNSUPPORTED, f"{name} must not be typed"
        assert result.usable is False
        assert element in {e for _, e, _ in result.untyped}
        assert element in result.reason

    def test_polystyrene_is_refused_for_its_aromatic_atoms(self, opls) -> None:
        """Parameters exist in OPLS-AA; the benzylic-CH charge is what is missing."""
        _, types = opls
        result = type_repeat_unit("*CC(*)c1ccccc1", types=types)
        assert result.status is TypingStatus.UNSUPPORTED
        assert any("aromatic" in environment for _, _, environment in result.untyped)

    def test_the_reason_names_something_actionable(self, opls) -> None:
        _, types = opls
        result = type_repeat_unit("*CCO*", types=types)
        assert result.reason
        assert "O" in result.reason

    def test_a_refusal_still_reports_what_it_managed_to_type(self, opls) -> None:
        """A partial result is diagnostic; it must never be mistaken for a usable one."""
        _, types = opls
        result = type_repeat_unit("*CC(Cl)*", types=types)
        assert result.assignments, "the carbons and hydrogens should still be typed"
        assert result.usable is False


@requires_rdkit
class TestChargeNeutralityIsMeasuredNotAssumed:
    def test_the_peo_residual_is_what_the_docstring_claims(self, opls) -> None:
        """Regression for the documented reason polar polymers are excluded.

        Poly(ethylene oxide) built from the tabulated ether types comes to +0.120 e per
        repeat unit, not zero. This asserts that number against the installed force
        field, so the justification in the module docstring cannot quietly drift away
        from the parameters it describes.
        """
        _, types = opls
        ether_ch2 = types["opls_182"].charge     # C(H2OR): ethyl ether
        ether_o = types["opls_180"].charge       # O: dialkyl ether
        hydrogen = types["opls_140"].charge
        repeat_unit_charge = 2 * (ether_ch2 + 2 * hydrogen) + ether_o
        assert repeat_unit_charge == pytest.approx(0.120, abs=1e-9)
        assert abs(repeat_unit_charge) > NEUTRALITY_TOLERANCE

    def test_the_benzylic_series_is_missing_its_ch_member(self, opls) -> None:
        """Why polystyrene is refused: OPLS tabulates CH3 and CH2, not CH.

        The two tabulated members differ by exactly one hydrogen's charge, so the group
        charge is +0.115 in both cases -- matching the ipso carbon's -0.115. The
        convention makes the missing value obvious, which is precisely why writing it
        into the table would be fabricating a parameter rather than reading one.
        """
        _, types = opls
        hydrogen = types["opls_140"].charge
        toluene_ch3 = types["opls_148"].charge + 3 * hydrogen
        ethylbenzene_ch2 = types["opls_149"].charge + 2 * hydrogen
        assert toluene_ch3 == pytest.approx(0.115, abs=1e-9)
        assert ethylbenzene_ch2 == pytest.approx(0.115, abs=1e-9)
        assert types["opls_145"].charge == pytest.approx(-0.115)   # ipso, balances the group
        assert not any(
            "benzylic CH" in t.comment or "i-Pr benzene" in t.comment for t in types.values()
        )


@requires_rdkit
class TestDegradesWithoutGuessing:
    def test_a_repeat_unit_without_two_attachment_points_is_uncertain(self, opls) -> None:
        _, types = opls
        result = type_repeat_unit("CCCC", types=types)     # a molecule, not a repeat unit
        assert result.status is TypingStatus.UNCERTAIN
        assert result.usable is False

    def test_an_unparseable_repeat_unit_does_not_raise(self, opls) -> None:
        _, types = opls
        assert type_repeat_unit("not-a-smiles", types=types).usable is False


@requires_rdkit
class TestTopologyGeneration:
    """The topology writer, and the x2top failure that made it necessary."""

    @pytest.fixture
    def butane(self):
        from rdkit import Chem
        return Chem.AddHs(Chem.MolFromSmiles("CCCC"))

    def test_butane_gets_every_interaction_it_should(self, opls, butane, tmp_path) -> None:
        """n-butane: 13 bonds, 24 angles, 27 proper dihedrals, 27 one-four pairs.

        Regression for `gmx x2top`, which emits **three** dihedrals for this molecule,
        carrying placeholder Ryckaert-Bellemans coefficients (60, 5, 3, 60, 5, 3)
        instead of the published OPLS values. grompp accepts that topology and mdrun
        runs it; the torsional energy comes out near 300 kJ/mol against a true profile
        spanning 21 kJ/mol, and the MM minimum lands on the eclipsed conformer.
        """
        from polymer_engine.simulation.opls_typing import type_molecule, write_opls_topology

        _, types = opls
        result = type_molecule(butane, types=types)
        assert result.status is TypingStatus.SUPPORTED
        text = write_opls_topology(result, butane, tmp_path / "butane.top").read_text()

        def count(section: str) -> int:
            block = text.split(f"[ {section} ]")[1].split("[")[0]
            return sum(1 for line in block.splitlines() if line.strip()[:1].isdigit())

        assert count("atoms") == 14
        assert count("bonds") == 13
        assert count("angles") == 24
        assert count("dihedrals") == 27
        assert count("pairs") == 27

    def test_bonded_interactions_carry_no_inline_parameters(self, opls, butane, tmp_path) -> None:
        """GROMACS must resolve them from ffbonded.itp, so published values are used."""
        from polymer_engine.simulation.opls_typing import type_molecule, write_opls_topology

        _, types = opls
        text = write_opls_topology(
            type_molecule(butane, types=types), butane, tmp_path / "b.top"
        ).read_text()
        dihedrals = text.split("[ dihedrals ]")[1].split("[")[0]
        for line in dihedrals.splitlines():
            fields = line.split()
            if fields and fields[0].isdigit():
                assert len(fields) == 5, f"dihedral carries inline parameters: {line!r}"

    def test_the_topology_charges_sum_to_zero(self, opls, butane, tmp_path) -> None:
        from polymer_engine.simulation.opls_typing import type_molecule, write_opls_topology

        _, types = opls
        text = write_opls_topology(
            type_molecule(butane, types=types), butane, tmp_path / "b.top"
        ).read_text()
        block = text.split("[ atoms ]")[1].split("[")[0]
        total = sum(float(line.split()[6]) for line in block.splitlines()
                    if line.strip()[:1].isdigit())
        assert total == pytest.approx(0.0, abs=1e-9)

    def test_writing_refuses_an_unsupported_typing(self, opls, tmp_path) -> None:
        """A topology from a partial typing is the silent wrongness to avoid."""
        from rdkit import Chem

        from polymer_engine.simulation.opls_typing import type_molecule, write_opls_topology

        _, types = opls
        chloro = Chem.AddHs(Chem.MolFromSmiles("CCCl"))
        result = type_molecule(chloro, types=types)
        assert result.status is TypingStatus.UNSUPPORTED
        with pytest.raises(ChemistryError, match="not SUPPORTED"):
            write_opls_topology(result, chloro, tmp_path / "no.top")

    def test_every_dihedral_references_real_atoms(self, opls, butane, tmp_path) -> None:
        from polymer_engine.simulation.opls_typing import type_molecule, write_opls_topology

        _, types = opls
        text = write_opls_topology(
            type_molecule(butane, types=types), butane, tmp_path / "b.top"
        ).read_text()
        block = text.split("[ dihedrals ]")[1].split("[")[0]
        for line in block.splitlines():
            fields = line.split()
            if fields and fields[0].isdigit():
                indices = [int(v) for v in fields[:4]]
                assert len(set(indices)) == 4, f"repeated atom in dihedral: {line!r}"
                assert all(1 <= i <= butane.GetNumAtoms() for i in indices)


@requires_rdkit
class TestMeltBuilding:
    """Growing a chain and packing a box, including the ways packing goes wrong."""

    def test_box_edge_matches_the_requested_density(self) -> None:
        from polymer_engine.simulation.melt_builder import AVOGADRO, box_edge_for_density

        edge = box_edge_for_density(chain_mass_amu=843.636, n_chains=20, density_kg_m3=940.0)
        recovered = 20 * 843.636 / AVOGADRO * 1e-3 / ((edge**3) * 1e-27)
        assert recovered == pytest.approx(940.0, rel=1e-9)

    def test_a_non_positive_density_is_refused(self) -> None:
        from polymer_engine.simulation.melt_builder import box_edge_for_density

        with pytest.raises(ChemistryError, match="positive"):
            box_edge_for_density(chain_mass_amu=100.0, n_chains=1, density_kg_m3=0.0)

    def test_a_chain_needs_at_least_two_repeat_units(self) -> None:
        from polymer_engine.simulation.melt_builder import grow_chain

        with pytest.raises(ChemistryError, match="at least two"):
            grow_chain("*CC*", 1)

    def test_growing_preserves_the_backbone_length(self) -> None:
        """A DP-n polyethylene chain has 2n backbone carbons, not n capped fragments."""
        from polymer_engine.simulation.melt_builder import grow_chain

        for dp in (2, 5, 30):
            chain = grow_chain("*CC*", dp)
            carbons = sum(1 for a in chain.GetAtoms() if a.GetSymbol() == "C")
            assert carbons == 2 * dp, f"DP={dp} gave {carbons} carbons"

    def test_a_chemistry_the_force_field_cannot_type_is_refused(self, opls, tmp_path) -> None:
        """The force-field gate has to hold at build time, not just at analysis time."""
        from polymer_engine.simulation.melt_builder import build_melt

        _, types = opls

        class NeverCalled:
            def run(self, *a: object, **k: object) -> object:  # pragma: no cover
                raise AssertionError("packing must not start for an untypable chemistry")

        with pytest.raises(ChemistryError, match="cannot type"):
            build_melt(
                name="pvc", repeat_unit_smiles="*CC(Cl)*", degree_of_polymerization=10,
                n_chains=5, target_density_kg_m3=1380.0, directory=tmp_path,
                types=types, runner=NeverCalled(),
            )


class TestGpuFlags:
    """Regression: PME on the GPU is incompatible with the minimisation integrator."""

    def _runner(self, **capabilities):
        from polymer_engine.local.runner import GROMACSRunner

        class Status:
            def __init__(self, **caps: object) -> None:
                self.name, self.path, self.issues, self.usable = "gromacs", "/x/gmx", [], True
                self.capabilities = {"has_gpu": True, "has_plumed": True, **caps}

        return GROMACSRunner(Status(**capabilities), enabled=False)   # type: ignore[arg-type]

    def test_gpu_pme_is_requested_by_default(self, tmp_path) -> None:
        command = self._runner().mdrun(deffnm="npt", cwd=tmp_path, use_gpu=True).command
        assert "-pme" in command and command[command.index("-pme") + 1] == "gpu"

    def test_gpu_pme_can_be_withheld_for_minimisation(self, tmp_path) -> None:
        """`steep` + `-pme gpu` is a fatal GROMACS error, not a slow path."""
        command = self._runner().mdrun(
            deffnm="em", cwd=tmp_path, use_gpu=True, gpu_pme=False
        ).command
        assert "-nb" in command and command[command.index("-nb") + 1] == "gpu"
        assert "-pme" not in command
        assert "-bonded" not in command

    def test_a_cpu_build_refuses_gpu_offload_rather_than_aborting(self, tmp_path) -> None:
        result = self._runner(has_gpu=False).mdrun(deffnm="x", cwd=tmp_path, use_gpu=True)
        assert result.mode == "unavailable"
        assert "no GPU support" in (result.error or "")
