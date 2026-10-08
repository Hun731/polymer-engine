"""Bridging a CHARMM-GUI chain into our own GROMACS melt.

CHARMM-GUI hands back a chain in CHARMM format -- PSF, coordinates, CHARMM36 parameters.
Our density campaign works in GROMACS. This converts the one chain (ParmEd, isolated in
.paramenv) and packs N copies into a bulk melt our campaign can run, so the parameters
are CHARMM-GUI's curated ones while the density and its uncertainty are ours.

The conversion needs .paramenv and a real gmx, so the heavy tests skip without them; the
topology-rewriting logic is pure and always runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polymer_engine.simulation.charmm_import import (
    _chain_mass_amu,
    _count_from_gro,
    _write_melt_topology,
    available,
)

REPO = Path(__file__).resolve().parents[2]


class TestTopologyRewrite:
    """Turning a single-chain topology into a melt is setting one number."""

    def _single_top(self, tmp_path: Path) -> Path:
        top = tmp_path / "chain.top"
        top.write_text(
            "[ defaults ]\n1 2 yes 1 1\n\n"
            "[ atomtypes ]\nCG321 12.011 0.0 A 0.1 0.1\n\n"
            "[ moleculetype ]\nsystem1 3\n\n"
            "[ atoms ]\n"
            "     1 CG321 1 LACTS C1 1 0.0 12.011\n"
            "     2 OG302 1 LACTS O1 1 0.0 15.999\n\n"
            "[ system ]\nBuilt\n\n"
            "[ molecules ]\nsystem1                 1\n")
        return top

    def test_the_molecule_count_becomes_n(self, tmp_path: Path) -> None:
        single = self._single_top(tmp_path)
        melt = tmp_path / "melt.top"
        _write_melt_topology(single, melt, n_molecules=20)
        text = melt.read_text()
        assert "system1                 20" in text
        # The atoms section, which also has a "1" molecule index, is untouched.
        assert "     1 CG321 1 LACTS" in text

    def test_the_chain_mass_sums_the_atom_column(self, tmp_path: Path) -> None:
        single = self._single_top(tmp_path)
        # 12.011 + 15.999
        assert _chain_mass_amu(single) == pytest.approx(28.01, abs=0.01)


class TestChainCount:
    def test_the_packed_count_is_read_from_the_structure(self, tmp_path: Path) -> None:
        gro = tmp_path / "packed.gro"
        gro.write_text("melt\n" + "184\n" + "x\n")   # 184 atoms / 92 per chain = 2
        assert _count_from_gro(gro, atoms_per_chain=92) == 2

    def test_a_short_gro_is_zero_not_a_crash(self, tmp_path: Path) -> None:
        gro = tmp_path / "empty.gro"
        gro.write_text("title\n")
        assert _count_from_gro(gro, atoms_per_chain=92) == 0


@pytest.mark.skipif(not available(), reason="ParmEd not installed in .paramenv")
class TestRealConversion:
    """Against a real CHARMM-GUI archive, when one is present in the repo's downloads."""

    def _archive_dir(self) -> Path | None:
        systems = REPO / "campaign" / "charmm_gui" / "systems"
        for candidate in systems.glob("*/charmm-gui-*"):
            if (candidate / "psfcrdreader").is_dir():
                return candidate
        return None

    def test_a_real_pla_chain_converts_and_is_neutral(self) -> None:
        from polymer_engine.simulation.charmm_import import convert_chain

        archive = self._archive_dir()
        if archive is None:
            pytest.skip("no extracted CHARMM-GUI archive on disk")
        import tempfile

        out = Path(tempfile.mkdtemp())
        chain = convert_chain(archive, out / "chain")
        assert chain.n_atoms > 0
        assert abs(chain.net_charge) < 1e-3, "a homopolymer chain must be neutral"
        assert Path(chain.topology).exists()
        assert Path(chain.coordinates).exists()
