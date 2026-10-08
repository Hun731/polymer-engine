"""Build tiny synthetic trajectories whose correct answers are known analytically.

These fixtures exist so trajectory tests can assert on *numbers*, not merely that the
code ran.  Each system is small enough to reason about by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build(root: Path) -> dict[str, Path]:
    import MDAnalysis as mda

    root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # System: a rigid 5-atom linear chain plus a single tracer atom.
    #   - chain atoms sit at x = 0, 1, 2, 3, 4 angstrom, all mass 1
    #     -> Rg^2 = mean((x - 2)^2) = (4+1+0+1+4)/5 = 2  -> Rg = sqrt(2) A
    #     -> end-to-end distance = 4 A = 0.4 nm
    #   - the tracer moves 1 A per frame along +x, so its MSD grows as (1 A * lag)^2
    # ------------------------------------------------------------------
    n_chain = 5
    n_atoms = n_chain + 1
    n_frames = 40
    box = np.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0], dtype=np.float32)

    universe = mda.Universe.empty(
        n_atoms,
        n_residues=2,
        atom_resindex=[0] * n_chain + [1],
        residue_segindex=[0, 0],
        trajectory=True,
    )
    # Real element symbols so MDAnalysis can guess sane masses from a PDB.
    universe.add_TopologyAttr("name", [f"C{i + 1}" for i in range(n_chain)] + ["O1"])
    universe.add_TopologyAttr("type", ["C"] * n_chain + ["O"])
    universe.add_TopologyAttr("resname", ["POL", "TRC"])
    universe.add_TopologyAttr("resid", [1, 2])
    universe.add_TopologyAttr("segid", ["SYS"])
    # Unit masses make Rg exactly analytic.
    universe.add_TopologyAttr("mass", [1.0] * n_atoms)  # exact in memory; PDB round-trip re-guesses

    coordinates = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for frame in range(n_frames):
        coordinates[frame, :n_chain, 0] = np.arange(n_chain, dtype=np.float32)
        coordinates[frame, :n_chain, 1] = 10.0
        coordinates[frame, :n_chain, 2] = 10.0
        coordinates[frame, n_chain] = (20.0 + frame, 20.0, 20.0)

    universe.load_new(coordinates, order="fac")
    for ts in universe.trajectory:
        ts.dimensions = box
        ts.time = ts.frame * 1.0  # 1 ps per frame

    topology = root / "chain.pdb"
    trajectory = root / "chain.xtc"
    universe.atoms.write(str(topology))
    with mda.Writer(str(trajectory), n_atoms=n_atoms) as writer:
        for _ in universe.trajectory:
            writer.write(universe.atoms)

    return {"topology": topology, "trajectory": trajectory}


if __name__ == "__main__":
    paths = build(Path(__file__).parent / "trajectories")
    print({k: str(v) for k, v in paths.items()})
