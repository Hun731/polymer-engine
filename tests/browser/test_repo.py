"""The polymer repository builder: serial, rate-limited, resumable.

CHARMM-GUI is a shared academic service, so the repo builds one polymer at a time with a
delay between them and resumes from a manifest rather than resubmitting. These test the
manifest, resume and offline-import logic; the live build loop needs credentials and is
exercised by the operator.
"""

from __future__ import annotations

import importlib.util
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


def _polymer_archive(tmp_path: Path, residues: list[str]) -> Path:
    """A single-chain-shaped archive whose p1_raw.str carries a residue sequence."""
    job = tmp_path / "src" / "job"
    (job / "psfcrdreader").mkdir(parents=True)
    (job / "toppar").mkdir(parents=True)
    (job / "psfcrdreader" / "p1_raw.str").write_text(
        "read sequence card\n* polymer sequence\n"
        f"{len(residues)}\n{' '.join(residues)}\ngenerate P1\n")
    (job / "psfcrdreader" / "p1_raw.psf").write_text("PSF\n\n     10 !NATOM\n")
    # curated residues, no penalties
    (job / "toppar" / "toppar_all36_synthetic_polymer.str").write_text(
        "".join(f"RESI {r}            0.000 \n" for r in set(residues)))
    archive = tmp_path / "download.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(job, arcname="job")
    return archive


def test_offline_import_reads_residues_and_marks_curated(tmp_path):
    live = _live()
    archive = _polymer_archive(tmp_path, ["LACTS", "LACTR", "LACTS"])
    result = live._import_offline(archive, "pla", tmp_path / "repo")
    assert result["ok"]
    assert result["residues"] == ["LACTR", "LACTS"]
    assert result["worst_residue_penalty"] == 0.0
    assert result["penalty_determination"] == "KNOWN"
    assert result["has_structure"]


def test_offline_import_reports_a_high_penalty_residue(tmp_path):
    live = _live()
    job = tmp_path / "src2" / "job"
    (job / "psfcrdreader").mkdir(parents=True)
    (job / "toppar").mkdir(parents=True)
    (job / "psfcrdreader" / "p1_raw.str").write_text(
        "read sequence card\n* polymer sequence\n1\nWEIRD\ngenerate\n")
    (job / "toppar" / "poly.str").write_text(
        "RESI WEIRD  0.000 ! param penalty=  60.0 ; charge penalty=  1.0\n")
    archive = tmp_path / "w.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(job, arcname="job")
    result = live._import_offline(archive, "weird", tmp_path / "repo2")
    assert result["worst_residue_penalty"] == pytest.approx(60.0)
    assert result["penalty_determination"] == "REQUIRES_VALIDATION"


def test_a_missing_archive_is_a_clean_failure(tmp_path):
    result = _live()._import_offline(tmp_path / "nope.tgz", "x", tmp_path / "r")
    assert not result["ok"]


def test_the_repo_summary_lists_imported_and_failed(tmp_path):
    live = _live()
    manifest = {"polymers": {
        "Polylactic acid": {"state": "imported", "job_id": "1", "residues": ["LACTS"],
                            "penalty_determination": "KNOWN",
                            "worst_residue_penalty": 0.0},
        "Broken": {"state": "failed", "reason": "no download after build"},
    }}
    live._write_repo_summary(manifest, tmp_path)
    text = (tmp_path / "REPO.md").read_text()
    assert "Polylactic acid" in text
    assert "KNOWN" in text
    assert "**failed**" in text


# --- Name resolution against the live set_monomer menu ---------------------------------
# Three ambiguities in the real menu each stranded a polymer at the "resolve" stage:
#   * an acid and its anion normalise to the same string (the '(-)' is punctuation);
#   * a diene offers trans/cis, never the 'atactic' the default assumes;
#   * Polyethylene is listed under two class groups (Olefins and Vinyls).
# The resolver breaks the first with an exact raw-label match and the third with a group
# hint; the diene is a variant-selection fix in the repo loop, not the resolver.

_MENU = {"groups": [
    {"group": "Poly(acrylic acid)", "options": [
        {"index": 10, "text": "atactic"}, {"index": 11, "text": "isotactic (R)"}]},
    {"group": "Poly(acrylic acid(-))", "options": [
        {"index": 20, "text": "atactic"}, {"index": 21, "text": "isotactic (R)"}]},
    {"group": "Polybutadiene", "options": [
        {"index": 30, "text": "trans"}, {"index": 31, "text": "cis"}]},
    {"group": "Olefins", "options": [
        {"index": 40, "text": "Polyethylene"}, {"index": 41, "text": "Polypropylene"}]},
    {"group": "Vinyls", "options": [
        {"index": 50, "text": "Polyethylene"}, {"index": 51, "text": "Poly(vinyl chloride)"}]},
]}


def test_an_acid_and_its_anion_are_told_apart_by_the_raw_label():
    live = _live()
    idx_acid, _g, _t = live._resolve_handler_option(_MENU, "Poly(acrylic acid)", "atactic")
    idx_anion, _g, _t = live._resolve_handler_option(_MENU, "Poly(acrylic acid(-))", "atactic")
    assert idx_acid == 10
    assert idx_anion == 20


def test_a_group_hint_resolves_a_monomer_listed_under_two_classes():
    live = _live()
    idx, group, _t = live._resolve_handler_option(
        _MENU, "Polyethylene", None, group="Olefins")
    assert idx == 40
    assert group == "Olefins"


def test_without_a_hint_a_two_class_monomer_still_refuses_rather_than_guesses():
    live = _live()
    idx, _g, reason = live._resolve_handler_option(_MENU, "Polyethylene", None)
    assert idx is None
    assert "refusing to choose" in reason


def test_a_diene_variant_is_selected_when_it_is_the_one_on_offer():
    live = _live()
    idx_trans, _g, txt = live._resolve_handler_option(_MENU, "Polybutadiene", "trans")
    assert idx_trans == 30
    assert txt == "trans"
