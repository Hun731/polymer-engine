"""Offline import of a CHARMM-GUI single-chain archive.

A single-chain build's download.tgz is CHARMM format -- PSF/CRD/PDB plus a CGenFF
stream file -- not a GROMACS system. It is the parameterised building block the
engine's own melt builder needs, so the import reports the structure and the CGenFF
penalties rather than trying to construct or validate a GROMACS system it does not
contain. No browser and no credentials: it reads a local file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _live():
    if "charmm_gui_live" in sys.modules:
        return sys.modules["charmm_gui_live"]
    spec = importlib.util.spec_from_file_location(
        "charmm_gui_live", ROOT / "scripts" / "charmm_gui_live.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["charmm_gui_live"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def charmm_archive(tmp_path: Path) -> Path:
    """A single-chain-shaped archive: a named top dir, PDB, and a real CGenFF .str."""
    job = tmp_path / "src" / "8851425743"
    (job / "toppar").mkdir(parents=True)
    (job / "psfcrdreader").mkdir(parents=True)
    (job / "psfcrdreader" / "p1_raw.pdb").write_text(
        "ATOM      1  C1  LIG     1       0.000   0.000   0.000\nEND\n")
    (job / "toppar" / "pla.str").write_text(
        (ROOT / "tests" / "fixtures" / "charmm_gui" / "pla_cgenff.str").read_text())
    archive = tmp_path / "download.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(job, arcname="8851425743")
    return archive


@pytest.fixture
def run_import(tmp_path, monkeypatch):
    live = _live()
    monkeypatch.setattr(live, "ACQUISITION_ROOT", tmp_path / "cg")

    def _run(archive, polymer_id="pla-test"):
        code = live.import_archive(argparse.Namespace(
            archive=str(archive), polymer_id=polymer_id, smiles="CC(C(=O)O)"))
        result_file = tmp_path / "cg" / "imports" / f"{polymer_id}.json"
        return code, json.loads(result_file.read_text())

    return _run


def test_a_charmm_archive_yields_structure_and_penalties(run_import, charmm_archive):
    code, result = run_import(charmm_archive)
    assert code == 0
    assert any("p1_raw.pdb" in f for f in result["roles"]["structure"])
    assert any("pla.str" in f for f in result["roles"]["stream"])
    assert result["penalties"] is not None
    assert result["penalties"]["max_penalty"] == pytest.approx(64.0)


def test_a_high_penalty_is_requires_validation_not_failure(run_import, charmm_archive):
    _code, result = run_import(charmm_archive)
    # Max penalty 64 is "poor" analogy -- flagged for QM, not rejected outright.
    assert result["penalty_determination"] == "REQUIRES_VALIDATION"
    assert result["penalties"]["max_penalty"] > 50


def test_no_stream_file_is_inconclusive_not_a_pass(run_import, tmp_path):
    job = tmp_path / "src2" / "job"
    (job / "psfcrdreader").mkdir(parents=True)
    (job / "psfcrdreader" / "p1_raw.pdb").write_text("ATOM      1  C1  LIG\nEND\n")
    archive = tmp_path / "nopenalty.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(job, arcname="job")
    _code, result = run_import(archive, polymer_id="nopen")
    assert result["penalties"] is None
    assert result["penalty_determination"] == "INSUFFICIENT_DATA"


def test_a_reimport_does_not_double_count_files(run_import, charmm_archive):
    run_import(charmm_archive)
    _code, result = run_import(charmm_archive)  # second time into the same workdir
    structures = result["roles"]["structure"]
    assert len(structures) == len(set(structures))
    assert len(structures) == 1


def test_a_missing_archive_is_reported(run_import, tmp_path):
    code = _live().import_archive(argparse.Namespace(
        archive=str(tmp_path / "nope.tgz"), polymer_id="x", smiles=None))
    assert code == 2
