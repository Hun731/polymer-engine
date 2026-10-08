#!/usr/bin/env python
"""Turn imported CHARMM-GUI chains into run-ready GROMACS melt systems.

Each polymer in the repo manifest that imported cleanly (a structure on disk and a KNOWN
parameter provenance) is converted from CHARMM to GROMACS, packed into a bulk melt, given
the campaign's validated mdp stage chain, and grompp-verified. What comes out of here is a
directory a density run can start from without any further build step -- the same shape the
SMILES melt builder produces, but carrying CHARMM-GUI's curated parameters.

This does not run any MD. Preparation is cheap (seconds per polymer) and safe to repeat;
the long, GPU-bound production runs are a separate, deliberate step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_density_campaign import gmx_env, load_campaign_config, write_mdps  # noqa: E402

from polymer_engine.core.config import ToolConfig  # noqa: E402
from polymer_engine.core.logging import get_logger  # noqa: E402
from polymer_engine.local.discovery import discover_tool  # noqa: E402
from polymer_engine.local.runner import GROMACSRunner  # noqa: E402
from polymer_engine.simulation import charmm_import as ci  # noqa: E402

logger = get_logger("prepare_charmm_runs")

REPO = REPO_ROOT / "campaign" / "charmm_gui" / "repo"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "campaign_anneal.yaml"


def _slug(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "+-") else "_" for c in name.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_")


def _seed_for(name: str, replica: int) -> int:
    """The campaign's per-replica seed rule, so a prepared run matches the campaign's."""
    return 100003 + (sum(ord(c) for c in name) * 31) % 90000 + replica * 7919


def _prepare_one(
    entry: dict[str, Any], *, runner: GROMACSRunner, cfg: dict[str, Any],
    out_root: Path, replicas: int, box_scale: float, verify: bool,
) -> dict[str, Any]:
    name = entry["name"]
    slug = _slug(name)
    system_dir = REPO_ROOT / entry["workdir"]
    archive_dir = next(iter(sorted(system_dir.glob("charmm-gui-*"))), None)
    if archive_dir is None:
        return {"name": name, "slug": slug, "ok": False, "error": "no extracted archive dir"}

    sim = cfg["simulation"]
    target_density = float(entry.get("experimental_density_kg_m3") or 1000.0)
    run_dir = out_root / slug
    record: dict[str, Any] = {
        "name": name, "slug": slug, "value": entry.get("value"),
        "dp": entry.get("dp"), "penalty_determination": entry.get("penalty_determination"),
        "worst_residue_penalty": entry.get("worst_residue_penalty"),
        "source_archive": str(archive_dir.relative_to(REPO_ROOT)),
        "run_dir": str(run_dir.relative_to(REPO_ROOT)), "replicas": [],
    }

    try:
        chain = ci.convert_chain(archive_dir, run_dir / "chain")
    except ci.CharmmImportError as exc:
        record["ok"] = False
        record["error"] = f"conversion failed: {exc}"
        return record

    record["atoms_per_chain"] = chain.n_atoms
    record["net_charge"] = round(chain.net_charge, 4)
    record["param_files"] = chain.param_files

    # Run-readiness is more than "grompp accepted it". Two chemistries pass grompp but
    # are not valid *neat melts*: a net-charged chain (a polyelectrolyte) needs
    # counterions, and a high CGenFF analogy penalty means the parameters are assigned by
    # weak analogy. The first blocks a neat-melt run; the second is a confidence caveat.
    caveats: list[str] = []
    n_chains = int(cfg["simulation"]["chains_per_system"])
    system_charge = round(chain.net_charge * n_chains, 2)
    if abs(chain.net_charge) > 0.5:
        caveats.append(
            f"net charge {chain.net_charge:+.1f}/chain ({system_charge:+.0f} for "
            f"{n_chains} chains): a neat melt needs counterions and is not physical as "
            f"packed")
    penalty = entry.get("worst_residue_penalty") or 0.0
    if penalty > 10.0:
        caveats.append(
            f"CGenFF analogy penalty {penalty}: parameters assigned by weak analogy, "
            f"so the density is lower-confidence")
    record["system_charge"] = system_charge
    record["caveats"] = caveats
    # A neat-melt run is blocked by a net charge; a penalty is a caveat, not a blocker.
    record["run_ready"] = abs(chain.net_charge) <= 0.5

    all_ok = True
    for replica in range(1, replicas + 1):
        rep_dir = run_dir / f"replica_{replica:02d}"
        seed = _seed_for(name, replica)
        rep: dict[str, Any] = {"replica": replica, "seed": seed,
                               "directory": str(rep_dir.relative_to(REPO_ROOT))}
        try:
            melt = ci.pack_charmm_melt(
                chain, directory=rep_dir, n_chains=int(sim["chains_per_system"]),
                target_density_kg_m3=target_density, runner=runner, seed=seed,
                box_scale=box_scale)
            rep.update(n_chains_packed=melt.n_chains_packed, box_nm=round(melt.box_nm, 3),
                       total_atoms=melt.n_chains_packed * melt.atoms_per_chain)
            chain_pairs = write_mdps(rep_dir, cfg, seed, replica_index=replica)
            rep["stages"] = [s for s, _ in chain_pairs]
            if verify:
                first_stage, previous = chain_pairs[0]
                gr = runner.grompp(
                    mdp=f"{first_stage}.mdp", structure=f"{previous}.gro",
                    topology="topol.top", output=f"{first_stage}.tpr", cwd=rep_dir,
                    max_warnings=2)
                rep["grompp_ok"] = bool(gr.succeeded and (rep_dir / f"{first_stage}.tpr").exists())
                if not rep["grompp_ok"]:
                    rep["grompp_stderr"] = (gr.stderr or "")[-600:]
                    all_ok = False
            rep["ok"] = rep.get("grompp_ok", True)
        except Exception as exc:  # noqa: BLE001 - a prep failure is data, not a crash
            rep["ok"] = False
            rep["error"] = str(exc)
            all_ok = False
        record["replicas"].append(rep)

    record["ok"] = all_ok
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="campaign config supplying the mdp stage parameters")
    ap.add_argument("--out", type=Path, default=REPO / "runs",
                    help="where prepared run directories are written")
    ap.add_argument("--replicas", type=int, default=1,
                    help="replicas to pack per polymer (campaign default is 3)")
    ap.add_argument("--box-scale", type=float, default=2.0,
                    help="starting-box inflation so extended DP-30 chains all pack")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these polymer names or slugs")
    ap.add_argument("--limit", type=int, default=None, help="prepare at most N polymers")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the grompp check (faster, but readiness is unproven)")
    ap.add_argument("--list-only", action="store_true",
                    help="print what would be prepared and exit")
    args = ap.parse_args()

    manifest = json.loads((REPO / "manifest.json").read_text())
    polymers = list(manifest["polymers"].values())
    ready = [e for e in polymers
             if e.get("state") == "imported" and e.get("has_structure")
             and e.get("penalty_determination") == "KNOWN"]

    if args.only:
        want = {s.lower() for s in args.only}
        ready = [e for e in ready if e["name"].lower() in want or _slug(e["name"]) in want]
    if args.limit:
        ready = ready[: args.limit]

    print(f"{len(ready)} polymer(s) eligible (imported, has structure, KNOWN provenance)")
    if args.list_only:
        for e in ready:
            print(f"  {_slug(e['name']):40s} dp={e.get('dp')} "
                  f"penalty={e.get('worst_residue_penalty')}")
        return 0

    if not ci.available():
        print("ParmEd (.paramenv) is not available; cannot convert. Aborting.")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_campaign_config(args.config)
    runner = GROMACSRunner(discover_tool("gromacs", ToolConfig(executable="gmx")),
                           enabled=True, extra_env=gmx_env(), default_timeout_s=600)

    results: list[dict[str, Any]] = []
    t0 = time.time()
    for i, entry in enumerate(ready, 1):
        rec = _prepare_one(entry, runner=runner, cfg=cfg, out_root=args.out,
                           replicas=args.replicas, box_scale=args.box_scale,
                           verify=not args.no_verify)
        results.append(rec)
        status = "OK " if rec.get("ok") else "FAIL"
        detail = rec.get("error") or (
            f"{rec.get('atoms_per_chain')} atoms/chain, q={rec.get('net_charge')}, "
            f"{len(rec['replicas'])} replica(s)")
        print(f"[{i:2d}/{len(ready)}] {status} {rec['slug']:38s} {detail}")

    out_manifest = args.out / "runs_manifest.json"
    n_ok = sum(1 for r in results if r.get("ok"))
    n_run_ready = sum(1 for r in results if r.get("ok") and r.get("run_ready"))
    out_manifest.write_text(json.dumps({
        "schema": "charmm_runs/v1",
        "config": str(args.config.relative_to(REPO_ROOT)),
        "box_scale": args.box_scale, "replicas_per_polymer": args.replicas,
        "grompp_verified": not args.no_verify,
        "n_prepared": n_ok, "n_run_ready": n_run_ready, "n_total": len(results),
        "runs": results,
    }, indent=2))
    _write_prepared_md(args.out / "PREPARED.md", results, args)
    print(f"\n{n_ok}/{len(results)} prepared in {time.time() - t0:.0f}s "
          f"-> {out_manifest.relative_to(REPO_ROOT)}")
    return 0 if n_ok == len(results) else 1


def _write_prepared_md(path: Path, results: list[dict[str, Any]], args: Any) -> None:
    n_ready = sum(1 for r in results if r.get("ok") and r.get("run_ready"))
    lines = ["# CHARMM-GUI systems prepared for a density run", "",
             f"- Config: `{args.config}`  (validated anneal protocol)",
             f"- Replicas packed per polymer: {args.replicas}",
             f"- Starting-box scale: {args.box_scale}",
             f"- grompp-verified: {not args.no_verify}",
             f"- **Run-ready neat melts: {n_ready} / {len(results)}** "
             f"(the rest need counterions — see caveats)", "",
             "| polymer | atoms/chain | net charge | replicas | run-ready | caveats |",
             "|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: x["slug"]):
        if not r.get("ok"):
            lines.append(f"| {r['slug']} | — | — | — | ✗ | "
                         f"**FAIL**: {r.get('error', 'see manifest')} |")
            continue
        ready = sum(1 for rep in r["replicas"] if rep.get("ok"))
        flag = "✓" if r.get("run_ready") else "needs counterions"
        caveat = "; ".join(r.get("caveats", [])) or "—"
        lines.append(f"| {r['slug']} | {r.get('atoms_per_chain')} | "
                     f"{r.get('net_charge')} | {ready}/{len(r['replicas'])} | "
                     f"{flag} | {caveat} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
