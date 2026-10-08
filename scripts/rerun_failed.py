#!/usr/bin/env python
"""Reset the failed CHARMM runs so the resumable runner redoes them with the fixed protocol.

The first campaign lost ~a third of the fleet to a barostat instability: production switched
to Parrinello-Rahman + Nose-Hoover, which resonates and fails LINCS a few ns in when the box
is not perfectly equilibrated (polypropylene, poly(acrylamide)); a few systems were also left
mid-anneal by the power cut. The fix is in the mdp defaults (stochastic C-rescale/V-rescale
for production too). This regenerates every failed system's mdps with that fix and deletes
exactly the stages that must re-run, keeping the good work:

  * production blow-up   -> keep npt.gro, redo production
  * anneal death         -> keep nvt.gro, redo anneal onward
  * one bad/missing replica -> redo only that replica's production

After this, `run_charmm_system.py --all-ready` resumes: finished stages are skipped, the
cleaned ones re-run under the stable couplings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_charmm_runs import _seed_for, _slug  # noqa: E402
from run_density_campaign import load_campaign_config, write_mdps  # noqa: E402

RUNS = REPO_ROOT / "campaign" / "charmm_gui" / "repo" / "runs"
CONFIG = REPO_ROOT / "configs" / "campaign_anneal.yaml"
STAGES = ("em", "nvt", "anneal", "npt", "prod")

# Failure classes -> the stage from which each replica must re-run. "prod" keeps the
# equilibrated npt.gro; "anneal" keeps nvt.gro and redoes the hot stages.
RESUME_FROM = {
    "prod": ["polypropylene", "poly_acrylamide"],
    "anneal": ["polyamide", "polyamide_inv", "poly_propylene_oxide",
               "poly_propylene_oxide_inv", "poly_propyl_methacrylate"],
}
# Systems with some good replicas: redo only replicas whose production is missing or whose
# final box is anomalous (dispersed). Judged per replica below.
PARTIAL = ["polyethylene_2_5-furandicarboxylate", "polyketone", "poly_ethylene_oxide_inv",
           "polystyrene", "polyvinylpyrrolidone"]


def _final_box_nm(gro: Path) -> float | None:
    if not gro.is_file():
        return None
    last = gro.read_text().splitlines()[-1].split()
    try:
        return float(last[0])
    except (ValueError, IndexError):
        return None


def _clean_from(replica: Path, stage: str) -> list[str]:
    """Delete every output of `stage` and the stages after it; report what was removed."""
    removed = []
    for st in STAGES[STAGES.index(stage):]:
        for f in replica.glob(f"{st}.*"):
            if f.suffix != ".mdp":       # keep mdps (regenerated separately)
                f.unlink()
                removed.append(f.name)
        for bak in replica.glob(f"#{st}.*#"):
            bak.unlink()
    return removed


def _regen_mdps(replica: Path, cfg: dict, seed: int) -> None:
    write_mdps(replica, cfg, seed, replica_index=int(replica.name.split("_")[-1]))


def main() -> int:
    cfg = load_campaign_config(CONFIG)
    manifest = json.loads((RUNS / "runs_manifest.json").read_text())
    names = {_slug(r["slug"]): r["name"] for r in manifest["runs"]}

    targets: dict[str, str] = {}
    for stage, slugs in RESUME_FROM.items():
        for s in slugs:
            targets[s] = stage

    for slug, stage in targets.items():
        name = names.get(slug, slug)
        print(f"{slug}: redo from {stage}")
        for rep in sorted((RUNS / slug).glob("replica_*")):
            seed = _seed_for(name, int(rep.name.split("_")[-1]))
            _regen_mdps(rep, cfg, seed)
            removed = _clean_from(rep, stage)
            print(f"  {rep.name}: cleaned {len(removed)} files from {stage}")

    for slug in PARTIAL:
        name = names.get(slug, slug)
        boxes = {rep.name: _final_box_nm(rep / "prod.gro")
                 for rep in sorted((RUNS / slug).glob("replica_*"))}
        good = [b for b in boxes.values() if b is not None]
        # A properly condensed replica has the SMALLEST box; a dispersed one is larger.
        # Compare to the minimum, not the median, or a 2-replica median hides a bad one.
        smallest = min(good) if good else None
        print(f"{slug}: boxes {boxes}")
        for rep in sorted((RUNS / slug).glob("replica_*")):
            box = boxes[rep.name]
            bad = box is None or (smallest is not None and box > 1.3 * smallest)
            if not bad:
                continue
            seed = _seed_for(name, int(rep.name.split("_")[-1]))
            _regen_mdps(rep, cfg, seed)
            removed = _clean_from(rep, "prod")
            print(f"  {rep.name}: redo production (box {box}); cleaned {len(removed)} files")

    print("\nDone. Resume with: setsid nohup .venv/bin/python scripts/run_charmm_system.py "
          "--all-ready > campaign/charmm_gui/repo/runs/campaign_resume.log 2>&1 < /dev/null &")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
