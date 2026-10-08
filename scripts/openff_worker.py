#!/usr/bin/env python
"""OpenFF parameterization worker, run inside the isolated ``.paramenv``.

The engine lives in ``.venv`` and OpenFF lives in ``.paramenv``, deliberately: the
running campaign's environment must not acquire a large new dependency tree mid-run.
This script is the bridge. It is invoked as a subprocess with one JSON request on the
command line and prints one JSON result to stdout.

Everything scientific is decided by the caller and passed in explicitly -- force field,
charge method, molecule. The worker chooses nothing.

    .paramenv/bin/python scripts/openff_worker.py '{"action": "capabilities"}'
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any


def capabilities() -> dict[str, Any]:
    """What this environment can actually do, measured rather than declared."""
    payload: dict[str, Any] = {"available": False, "charge_methods": []}
    try:
        import openff.interchange
        import openff.toolkit

        payload["openff_toolkit"] = openff.toolkit.__version__
        payload["openff_interchange"] = openff.interchange.__version__
        payload["available"] = True
    except ImportError as exc:
        payload["error"] = f"openff not importable: {exc}"
        return payload

    from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY

    payload["toolkits"] = [
        {"name": tk.__class__.__name__, "version": getattr(tk, "toolkit_version", None)}
        for tk in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits
    ]
    # AM1-BCC needs AmberTools' sqm or an OpenEye licence. NAGL is OpenFF's published
    # graph-network surrogate for it, and is what makes this route usable without either.
    try:
        from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper  # noqa: F401

        payload["charge_methods"].append("nagl")
        payload["nagl_model"] = NAGL_MODEL
    except ImportError:
        pass
    try:
        from openff.toolkit.utils.toolkits import AmberToolsToolkitWrapper

        if AmberToolsToolkitWrapper.is_available():
            payload["charge_methods"].append("am1bcc")
    except Exception:  # noqa: BLE001 - absence is the answer, not an error
        pass

    try:
        from openff.toolkit import ForceField

        payload["force_fields"] = []
        for name in ("openff-2.2.0.offxml", "openff-2.1.0.offxml", "openff-2.0.0.offxml"):
            try:
                ForceField(name)
                payload["force_fields"].append(name)
            except Exception:  # noqa: BLE001
                continue
    except ImportError:
        pass
    return payload


#: OpenFF's graph-network AM1-BCC surrogate. Named explicitly so the charge model used
#: is recorded in provenance rather than inherited from a library default.
NAGL_MODEL = "openff-gnn-am1bcc-0.1.0-rc.3.pt"


def assess(request: dict[str, Any]) -> dict[str, Any]:
    """Can this force field type this molecule?  Assigns nothing and runs no charges."""
    from openff.toolkit import ForceField, Molecule

    smiles = request["smiles"]
    force_field = request.get("force_field", "openff-2.2.0.offxml")
    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    ForceField(force_field).label_molecules(molecule.to_topology())
    return {
        "ok": True, "n_atoms": molecule.n_atoms, "force_field": force_field,
        "formula": molecule.to_hill_formula(),
    }


def parameterize(request: dict[str, Any]) -> dict[str, Any]:
    """Assign parameters and charges, and export a GROMACS system."""
    from openff.interchange import Interchange
    from openff.toolkit import ForceField, Molecule

    smiles = request["smiles"]
    force_field = request.get("force_field", "openff-2.2.0.offxml")
    charge_method = request.get("charge_method", "nagl")
    prefix = request["prefix"]
    box_nm = float(request.get("box_nm", 4.0))

    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.generate_conformers(n_conformers=1)

    if charge_method == "nagl":
        from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper

        NAGLToolkitWrapper().assign_partial_charges(
            molecule, partial_charge_method=NAGL_MODEL
        )
        charge_detail = NAGL_MODEL
    elif charge_method == "am1bcc":
        molecule.assign_partial_charges("am1bcc")
        charge_detail = "am1bcc"
    else:
        # Refused rather than silently substituted: Sage was fitted against AM1-BCC, so
        # pairing it with a different charge model is a change of force field.
        raise ValueError(
            f"charge method {charge_method!r} is not an AM1-BCC-equivalent model; "
            f"pairing it with Sage would silently change the force field"
        )

    net_charge = sum(float(c.m) for c in molecule.partial_charges)
    interchange = Interchange.from_smirnoff(
        ForceField(force_field), molecule.to_topology(),
        charge_from_molecules=[molecule],
    )
    interchange.box = [[box_nm, 0, 0], [0, box_nm, 0], [0, 0, box_nm]]
    interchange.to_gromacs(prefix)
    return {
        "ok": True, "topology": f"{prefix}.top", "coordinates": f"{prefix}.gro",
        "n_atoms": molecule.n_atoms, "net_charge": net_charge,
        "force_field": force_field, "charge_method": charge_detail,
        "formula": molecule.to_hill_formula(),
    }


ACTIONS = {"capabilities": lambda _r: capabilities(), "assess": assess,
           "parameterize": parameterize}


def _handler_for(action: str):
    handler = ACTIONS.get(action)
    if handler is None:
        raise ValueError(f"unknown action {action!r}; known: {sorted(ACTIONS)}")
    return handler


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "error": "expected one JSON argument"}))
        return 2
    try:
        request = json.loads(sys.argv[1])
        action = request.get("action", "capabilities")
        handler = _handler_for(action)
        print(json.dumps(handler(request)))
    except Exception as exc:  # noqa: BLE001 - the caller needs the reason, not a stack
        print(json.dumps({
            "ok": False, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
