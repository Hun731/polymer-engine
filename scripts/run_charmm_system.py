#!/usr/bin/env python
"""Run a prepared CHARMM-GUI melt through the validated stage chain.

`prepare_charmm_runs.py` leaves each system as a grompp-ready directory (packed.gro,
topol.top, and the em/nvt/anneal/npt/prod mdps). This drives one or more of them through
those stages on the GPU using the campaign's own `run_stage`, so a CHARMM system runs by
exactly the protocol the SMILES campaign is validated against -- no second code path.

Production is long (tens of ns of NPT + production per replica). Use `--stages em` for a
fast readiness smoke test, or `--stages em,nvt,anneal,npt,prod` (the default) for a full
run. Nothing here launches a fleet on its own: it runs the systems you name, in series.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_density_campaign import gmx_env, run_stage  # noqa: E402

from polymer_engine.core.config import ToolConfig  # noqa: E402
from polymer_engine.core.logging import get_logger  # noqa: E402
from polymer_engine.local.discovery import discover_tool  # noqa: E402
from polymer_engine.local.runner import GROMACSRunner  # noqa: E402
from polymer_engine.simulation.mdp import STAGE_ORDER  # noqa: E402

logger = get_logger("run_charmm_system")

RUNS = REPO_ROOT / "campaign" / "charmm_gui" / "repo" / "runs"


def _chain_for(directory: Path) -> list[tuple[str, str]]:
    """(stage, previous-prefix) pairs from the mdps present, starting at packed.gro."""
    chain: list[tuple[str, str]] = []
    previous = "packed"
    for stage in STAGE_ORDER:
        if (directory / f"{stage}.mdp").exists():
            chain.append((stage, previous))
            previous = stage
    return chain


def _run_replica(slug: str, directory: Path, *, runner: GROMACSRunner,
                 stages: set[str], ntomp: int, stage_timeout_s: float) -> dict:
    """Run one replica's stage chain, resuming past stages already finished.

    A stage whose output .gro exists is skipped -- so a restart after a crash (or a
    machine reboot mid-campaign) continues where it stopped rather than re-running days
    of finished simulation. The same rule the main campaign uses to be interruptible.
    """
    out: dict = {"directory": str(directory.relative_to(REPO_ROOT)), "stages": []}
    chain = [(s, p) for s, p in _chain_for(directory) if s in stages]
    for stage, previous in chain:
        if (directory / f"{stage}.gro").is_file():
            out["stages"].append({"stage": stage, "resumed": True})
            print(f"    {slug:26s} {stage:7s} skip (already done)")
            continue
        t0 = time.time()
        ok, message = run_stage(runner, directory, stage, previous,
                                ntomp=ntomp, timeout_s=stage_timeout_s)
        out["stages"].append({"stage": stage, "ok": ok,
                              "seconds": round(time.time() - t0, 1), "message": message})
        print(f"    {slug:26s} {stage:7s} {'OK' if ok else 'FAIL'} "
              f"({time.time() - t0:.0f}s){'' if ok else ' :: ' + message}")
        if not ok:
            out["ok"] = False
            return out
    out["ok"] = True
    return out


def _run_one(slug: str, rec: dict, *, runner: GROMACSRunner, stages: set[str],
             ntomp: int, stage_timeout_s: float) -> dict:
    out: dict = {"slug": slug, "replicas": []}
    if not rec.get("run_ready"):
        out["skipped"] = f"not run-ready: {'; '.join(rec.get('caveats', []))}"
        print(f"    {slug:26s} SKIP :: {out['skipped']}")
        return out
    for rep in rec["replicas"]:
        directory = REPO_ROOT / rep["directory"]
        r = _run_replica(slug, directory, runner=runner, stages=stages,
                         ntomp=ntomp, stage_timeout_s=stage_timeout_s)
        r["replica"] = rep["replica"]
        out["replicas"].append(r)
    out["ok"] = all(r.get("ok") for r in out["replicas"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", nargs="*", default=None,
                    help="polymer slug(s) to run; default is all run-ready systems")
    ap.add_argument("--all-ready", action="store_true",
                    help="run every run-ready system (explicit opt-in to a long series)")
    ap.add_argument("--stages", default=",".join(STAGE_ORDER),
                    help="comma-separated stages to run (default: the full chain)")
    ap.add_argument("--ntomp", type=int, default=8, help="OpenMP threads per mdrun")
    ap.add_argument("--stage-timeout-h", type=float, default=12.0,
                    help="per-stage wall-clock cap in hours")
    ap.add_argument("--manifest", type=Path, default=RUNS / "runs_manifest.json")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    by_slug = {r["slug"]: r for r in manifest["runs"] if r.get("ok")}
    if args.slug:
        want = set(args.slug)
        selected = [by_slug[s] for s in want if s in by_slug]
        missing = want - set(by_slug)
        if missing:
            print(f"unknown slug(s): {sorted(missing)}")
    elif args.all_ready:
        selected = [r for r in by_slug.values() if r.get("run_ready")]
    else:
        print("Specify --slug <name> ... or --all-ready. Run-ready systems:")
        for s in sorted(r["slug"] for r in by_slug.values() if r.get("run_ready")):
            print(f"  {s}")
        return 0

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    runner = GROMACSRunner(discover_tool("gromacs", ToolConfig(executable="gmx")),
                           enabled=True, extra_env=gmx_env(),
                           default_timeout_s=args.stage_timeout_h * 3600.0)
    progress = RUNS / "campaign_progress.json"
    print(f"running {len(selected)} system(s), stages {sorted(stages)} "
          f"-> progress in {progress.relative_to(REPO_ROOT)}")
    results = []
    for i, rec in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] {rec['slug']}:")
        results.append(_run_one(rec["slug"], rec, runner=runner, stages=stages,
                                ntomp=args.ntomp,
                                stage_timeout_s=args.stage_timeout_h * 3600.0))
        # Persist after every system so a multi-day run is monitorable and a crash
        # loses at most the system in flight.
        n_ok = sum(1 for r in results if r.get("ok"))
        n_skip = sum(1 for r in results if r.get("skipped"))
        progress.write_text(json.dumps({
            "stages": sorted(stages), "n_selected": len(selected),
            "n_done": len(results), "n_ok": n_ok, "n_skipped": n_skip,
            "results": results}, indent=2))

    n_ok = sum(1 for r in results if r.get("ok"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    print(f"\n{n_ok}/{len(results)} completed the requested stages"
          + (f", {n_skip} skipped" if n_skip else ""))
    return 0 if n_ok == len(results) - n_skip else 1


if __name__ == "__main__":
    raise SystemExit(main())
