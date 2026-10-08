#!/usr/bin/env python
"""Convert a CHARMM-GUI single-chain build to GROMACS, run inside .paramenv.

ParmEd reads the CHARMM PSF (topology + connectivity + atom types), loads the CHARMM36
parameter sets the chain uses, attaches coordinates from the PDB, and writes a GROMACS
topology and structure for the one chain. Our melt builder then packs N copies of it.

Invoked with one JSON request on argv, prints one JSON result. The engine's .venv never
imports ParmEd -- this is the only code that does, exactly as the OpenFF worker isolates
that dependency.

    .paramenv/bin/python scripts/charmm2gmx_worker.py '{"action":"convert", ...}'
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


def capabilities() -> dict[str, Any]:
    payload: dict[str, Any] = {"available": False}
    try:
        import parmed

        payload["parmed"] = parmed.__version__
        payload["available"] = True
        payload["ok"] = True
    except ImportError as exc:
        payload["error"] = f"parmed not importable: {exc}"
        payload["ok"] = False
    return payload


#: Parameter files a polymer chain needs, in load order. The synthetic-polymer sets
#: carry the monomer residues; cgenff is the general fallback. Others are skipped -- a
#: single organic chain does not need the lipid or nucleic-acid tables, and loading all
#: of them is slow and can raise on duplicate terms.
PARAM_PATTERNS = (
    "top_all36_cgenff.rtf", "par_all36_cgenff.prm",
    "toppar_all36_synthetic_polymer.str", "toppar_all36_synthetic_polymer_patch.str",
    "top_all36_prot.rtf", "par_all36m_prot.prm",
)


def _find_params(toppar: Path) -> list[str]:
    found: list[str] = []
    for name in PARAM_PATTERNS:
        path = toppar / name
        if path.exists():
            found.append(str(path))
    return found


def convert(request: dict[str, Any]) -> dict[str, Any]:
    """PSF + params + coordinates -> GROMACS .top and .gro for one chain."""
    import parmed as pmd

    psf_path = Path(request["psf"])
    coord_path = Path(request["coordinates"])   # pdb or crd
    toppar = Path(request["toppar"])
    prefix = request["prefix"]
    extra = request.get("extra_params", [])

    param_files = _find_params(toppar) + [str(p) for p in extra]
    if not param_files:
        raise ValueError(f"no CHARMM parameter files found under {toppar}")

    params = pmd.charmm.CharmmParameterSet(*param_files)
    psf = pmd.charmm.CharmmPsfFile(str(psf_path))
    psf.load_parameters(params)

    coords = pmd.load_file(str(coord_path))
    psf.coordinates = coords.coordinates

    # A generous box so a single chain is not wrapped; the melt builder sets the real
    # box when it packs. ParmEd needs a box to write a .gro.
    if psf.box is None:
        import numpy as np

        span = float(np.ptp(psf.coordinates, axis=0).max()) + 20.0
        psf.box = [span, span, span, 90.0, 90.0, 90.0]

    top_path = f"{prefix}.top"
    gro_path = f"{prefix}.gro"
    psf.save(top_path, format="gromacs", overwrite=True)
    psf.save(gro_path, format="gro", overwrite=True)

    return {
        "ok": True, "topology": top_path, "coordinates": gro_path,
        "n_atoms": len(psf.atoms), "n_bonds": len(psf.bonds),
        "n_angles": len(psf.angles), "n_dihedrals": len(psf.dihedrals),
        "n_residues": len(psf.residues),
        "param_files": [Path(p).name for p in param_files],
        "net_charge": round(sum(a.charge for a in psf.atoms), 6),
    }


ACTIONS = {"capabilities": lambda _r: capabilities(), "convert": convert}


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "error": "expected one JSON argument"}))
        return 2
    try:
        request = json.loads(sys.argv[1])
        action = request.get("action", "capabilities")
        handler = ACTIONS.get(action)
        if handler is None:
            print(json.dumps({"ok": False, "error": f"unknown action {action!r}"}))
            return 1
        print(json.dumps(handler(request)))
    except Exception as exc:  # noqa: BLE001 - the caller needs the reason
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()[-1500:]}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
