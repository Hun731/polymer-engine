#!/usr/bin/env python
"""Cohesive energy density / solubility parameter for finished CHARMM runs (CPU only).

The solubility parameter needs one number the bulk trajectory does not contain: the
potential energy of a single chain *in isolation*. This runs that reference -- one chain,
no periodic images, on the CPU so it never competes with a GPU campaign -- reads the bulk
potential energy and volume from the production run, and forms

    CED = (E_gas - E_liquid_per_chain) / V_molar,   delta = sqrt(CED).

The gas and bulk energies must be treated consistently for the absolute value to mean
anything, so both use the same non-bonded cutoffs; the result is for comparing polymers
computed the same way, not against a handbook (the calculator says as much).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_density_campaign import gmx_env  # noqa: E402
from run_studies import read_xvg  # noqa: E402

from polymer_engine.core.config import ToolConfig  # noqa: E402
from polymer_engine.core.logging import get_logger  # noqa: E402
from polymer_engine.local.discovery import discover_tool  # noqa: E402
from polymer_engine.local.runner import GROMACSRunner  # noqa: E402
from polymer_engine.properties.cohesive import CohesiveEnergyDensity  # noqa: E402

logger = get_logger("run_cohesive")
RUNS = REPO_ROOT / "campaign" / "charmm_gui" / "repo" / "runs"
AVOGADRO = 6.02214076e23

# A single chain, effectively in vacuum. GROMACS 2026's Verlet scheme does not support
# pbc=no, so we use a box large enough that a chain never reaches its own periodic image
# within the cutoff (editconf pads by cutoff+2 nm), with a plain cut-off (no PME: there is
# nothing to sum over images). Non-bonded settings match the bulk so the energies subtract.
VACUUM_MDP = """\
integrator      = md
dt              = 0.001
nsteps          = {nsteps}
pbc             = xyz
cutoff-scheme   = Verlet
coulombtype     = cut-off
rcoulomb        = {cutoff}
vdwtype         = cut-off
rvdw            = {cutoff}
tcoupl          = v-rescale
tc-grps         = System
tau-t           = 1.0
ref-t           = {temperature}
gen-vel         = yes
gen-temp        = {temperature}
gen-seed        = {seed}
nstenergy       = 500
constraints     = h-bonds
constraint-algorithm = lincs
comm-mode       = linear
"""


def _energy_series(runner: GROMACSRunner, edr: Path, term: str, out: Path) -> np.ndarray:
    result = runner.run(["energy", "-f", str(edr), "-o", str(out)],
                        cwd=edr.parent, stdin=f"{term}\n")
    if not (result.succeeded and out.exists()):
        return np.empty(0)
    _t, y = read_xvg(out)
    return y


def _vacuum_energy(runner: GROMACSRunner, chain_gro: Path, chain_top: Path, work: Path,
                   *, temperature: float, ns: float, cutoff: float, seed: int) -> np.ndarray:
    """Run one isolated chain and return its potential-energy series (kJ/mol)."""
    work.mkdir(parents=True, exist_ok=True)
    # A box large enough that the (aperiodic) chain never reaches its own cutoff sphere.
    boxed = work / "chain_box.gro"
    runner.run(["editconf", "-f", str(chain_gro), "-o", str(boxed),
                "-c", "-d", f"{cutoff + 2.0}", "-bt", "cubic"], cwd=work)
    mdp = work / "vac.mdp"
    mdp.write_text(VACUUM_MDP.format(nsteps=int(ns * 1000 / 0.001), cutoff=cutoff,
                                     temperature=temperature, seed=seed))
    gr = runner.grompp(mdp="vac.mdp", structure=str(boxed), topology=str(chain_top),
                       output="vac.tpr", cwd=work, max_warnings=5)
    if not gr.succeeded:
        raise RuntimeError(f"vacuum grompp failed: {(gr.stderr or '')[-300:]}")
    md = runner.mdrun(deffnm="vac", cwd=work, use_gpu=False, ntomp=2,
                      timeout_s=3 * 3600.0)
    if not md.succeeded:
        raise RuntimeError(f"vacuum mdrun failed: {(md.stderr or '')[-300:]}")
    return _energy_series(runner, work / "vac.edr", "Potential", work / "vac_pot.xvg")


def _cohesive_for(slug: str, rec: dict, runner: GROMACSRunner, *, temperature: float,
                  ns: float, cutoff: float) -> dict[str, Any]:
    run_dir = REPO_ROOT / rec["run_dir"]
    chain_gro, chain_top = run_dir / "chain.gro", run_dir / "chain.top"
    if not (chain_gro.exists() and chain_top.exists()):
        return {"slug": slug, "ok": False, "error": "no converted single chain"}

    liquid_all, vol_all, n_chains = [], [], None
    for rep in rec["replicas"]:
        edr = REPO_ROOT / rep["directory"] / "prod.edr"
        if not edr.exists():
            continue
        pot = _energy_series(runner, edr, "Potential", edr.parent / "coh_pot.xvg")
        vol = _energy_series(runner, edr, "Volume", edr.parent / "coh_vol.xvg")
        if pot.size and vol.size:
            n_chains = rep["n_chains_packed"]
            liquid_all.append(pot[pot.size // 2:] / n_chains)   # per chain, prod half
            vol_all.append(vol[vol.size // 2:] / n_chains)
    if not liquid_all or not n_chains:
        return {"slug": slug, "ok": False, "error": "no production energy/volume"}
    liquid = np.concatenate(liquid_all)
    molar_volume_cm3 = float(np.mean(np.concatenate(vol_all))) * 1e-21 * AVOGADRO  # nm^3->cm^3

    try:
        gas = _vacuum_energy(runner, chain_gro, chain_top, run_dir / "vacuum",
                             temperature=temperature, ns=ns, cutoff=cutoff,
                             seed=100003 + sum(ord(c) for c in slug) % 9000)
    except Exception as exc:  # noqa: BLE001 - a run failure is data
        return {"slug": slug, "ok": False, "error": f"vacuum run: {exc}"}
    if gas.size < 10:
        return {"slug": slug, "ok": False, "error": "vacuum run produced no energy"}
    gas = gas[gas.size // 2:]

    result = CohesiveEnergyDensity().compute(
        liquid, gas, molar_volume_cm3, n_replicas=len(liquid_all),
        simulation_ns=ns, provenance={"cutoff_nm": cutoff, "n_chains": n_chains})
    payload = result.as_dict()
    return {"slug": slug, "ok": True, "cohesive_energy_density": payload}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", nargs="*", default=None, help="systems to process")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--ns", type=float, default=2.0, help="vacuum run length (ns)")
    ap.add_argument("--cutoff", type=float, default=1.2, help="non-bonded cutoff (nm)")
    ap.add_argument("--manifest", type=Path, default=RUNS / "runs_manifest.json")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    by_slug = {r["slug"]: r for r in manifest["runs"] if r.get("run_ready")}
    slugs = args.slug or [s for s, r in by_slug.items()
                          if (REPO_ROOT / r["replicas"][0]["directory"] / "prod.edr").exists()]
    # CUDA_VISIBLE_DEVICES="" hides every GPU from GROMACS, so the vacuum runs cannot land
    # on the GPU and compete with a running campaign -- use_gpu=False only omits "-nb gpu",
    # which lets GROMACS auto-select the GPU anyway. This keeps the cohesive runs on CPU.
    cpu_env = {**gmx_env(), "CUDA_VISIBLE_DEVICES": ""}
    runner = GROMACSRunner(discover_tool("gromacs", ToolConfig(executable="gmx")),
                           enabled=True, extra_env=cpu_env, default_timeout_s=3 * 3600.0)

    out_dir = RUNS / "studies"
    out_dir.mkdir(exist_ok=True)
    results = []
    for slug in slugs:
        if slug not in by_slug:
            print(f"  {slug}: not run-ready or unknown")
            continue
        rec = _cohesive_for(slug, by_slug[slug], runner, temperature=args.temperature,
                            ns=args.ns, cutoff=args.cutoff)
        results.append(rec)
        if rec["ok"]:
            p = rec["cohesive_energy_density"]["provenance"]
            print(f"  {slug:26s} delta = {p['solubility_parameter_mpa_half']:.2f} MPa^0.5 "
                  f"(CED {rec['cohesive_energy_density']['measurement']['value']:.0f} MPa)")
        else:
            print(f"  {slug:26s} FAILED: {rec['error']}")
    (out_dir / "cohesive.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"\nwrote {out_dir / 'cohesive.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
