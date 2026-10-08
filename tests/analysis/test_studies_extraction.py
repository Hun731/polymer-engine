"""Geometric extraction helpers in run_studies: elements, contacts, hydrogen bonds.

These run on synthetic coordinates rather than a real trajectory, so they pin the geometry
(what counts as a contact, what counts as a hydrogen bond) without a slow MD read.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
pytest.importorskip("MDAnalysis")


def _studies():
    if "run_studies" in sys.modules:
        return sys.modules["run_studies"]
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "run_studies", ROOT / "scripts" / "run_studies.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_studies"] = module
    spec.loader.exec_module(module)
    return module


def test_element_symbol_from_gro_atom_names():
    rs = _studies()
    assert rs._element("CL2") == "CL"      # two-letter element wins over C
    assert rs._element("HG21") == "H"
    assert rs._element("C1") == "C"
    assert rs._element("OG302") == "O"


def test_intermolecular_contacts_counts_only_cross_chain_pairs():
    rs = _studies()
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float32)
    # chain 0 = atoms 0,1 ; chain 1 = atoms 2,3. atoms_per_chain = 2.
    # Place 0 and 2 within 0.6 nm (=6 A) of each other; keep the intramolecular
    # neighbours (0-1, 2-3) far so they are not what is being counted.
    pos = np.array([[0.0, 0.0, 0.0],     # chain 0
                    [50.0, 0.0, 0.0],    # chain 0 (far)
                    [3.0, 0.0, 0.0],     # chain 1 (3 A from atom 0 -> contact)
                    [60.0, 0.0, 0.0]],   # chain 1 (far)
                   dtype=np.float32)
    assert rs._intermolecular_contacts(pos, box, atoms_per_chain=2, cutoff_nm=0.6) == 1


def test_hydrogen_bond_geometry_accepts_linear_and_rejects_bent():
    rs = _studies()
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float32)
    # Linear donor: O(0)-H(1) ... O(2), with H..A = 2 A and a straight D-H..A angle.
    pos = np.array([[0.0, 0.0, 0.0],     # donor heavy O
                    [1.0, 0.0, 0.0],     # H, 1 A from donor
                    [3.0, 0.0, 0.0]],    # acceptor O, 2 A from H, angle 180 deg
                   dtype=np.float32)
    donor_pairs = [(1, 0)]               # (H index, donor-heavy index)
    acceptors = np.array([0, 2])
    assert rs._count_hbonds(pos, box, donor_pairs, acceptors) == 1

    # Bent geometry: move the acceptor so D-H..A collapses below 120 degrees.
    pos_bent = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.5, 1.8, 0.0]], dtype=np.float32)
    assert rs._count_hbonds(pos_bent, box, donor_pairs, acceptors) == 0
