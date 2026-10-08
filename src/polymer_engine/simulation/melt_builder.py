"""Building amorphous polymer melt systems.

The engine has always validated systems and never built them, which meant real
candidates could not enter a campaign at all.  This module closes that gap for the
chemistries the force-field qualification actually covers, and refuses the rest.

The route is deliberately made of documented, checkable steps:

1. grow a single chain of the requested degree of polymerisation from the repeat unit;
2. embed and relax it with RDKit, so the starting conformer is not self-overlapping;
3. type it against the installed OPLS-AA (:mod:`polymer_engine.simulation.opls_typing`),
   which refuses any chemistry whose parameters are not tabulated;
4. pack copies into a periodic box with ``gmx insert-molecules``;
5. write a topology whose molecule count matches what was actually inserted.

Step 5 is where melt builders usually go wrong.  ``insert-molecules`` places *as many
as it can* and silently stops when it runs out of attempts, so a topology written from
the requested count describes a system that does not exist.  grompp then fails with an
atom-count mismatch -- or worse, succeeds against a differently-wrong structure.  The
builder therefore reads the count back from the packed structure and reports the
shortfall rather than assuming it got what it asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.errors import ChemistryError, PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.simulation.opls_typing import (
    OplsType,
    TypingStatus,
    type_molecule,
    write_opls_topology,
)

logger = get_logger("simulation.melt_builder")

#: Avogadro's number, for converting a target density into a box volume.
AVOGADRO = 6.02214076e23

#: ``insert-molecules`` gives up quietly; this many attempts per molecule is generous
#: for a melt at moderate density and still bounded.
INSERTION_TRIES = 200


@dataclass
class MeltSystem:
    """One packed, typed, ready-to-minimise melt."""

    name: str
    directory: str
    n_chains_requested: int
    n_chains_packed: int
    degree_of_polymerization: int
    atoms_per_chain: int
    total_atoms: int
    box_nm: float
    target_density_kg_m3: float
    packed_density_kg_m3: float
    chain_mass_amu: float
    structure: str
    topology: str
    force_field: str = "OPLS-AA"
    force_field_source: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Every requested chain was actually placed."""
        return self.n_chains_packed == self.n_chains_requested

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "directory": self.directory,
            "n_chains_requested": self.n_chains_requested,
            "n_chains_packed": self.n_chains_packed,
            "degree_of_polymerization": self.degree_of_polymerization,
            "atoms_per_chain": self.atoms_per_chain, "total_atoms": self.total_atoms,
            "box_nm": self.box_nm, "target_density_kg_m3": self.target_density_kg_m3,
            "packed_density_kg_m3": self.packed_density_kg_m3,
            "chain_mass_amu": self.chain_mass_amu,
            "structure": self.structure, "topology": self.topology,
            "force_field": self.force_field, "force_field_source": self.force_field_source,
            "complete": self.complete, "warnings": list(self.warnings),
        }


def grow_chain(repeat_unit_smiles: str, degree_of_polymerization: int) -> Any:
    """Join ``n`` repeat units into a single hydrogen-terminated chain.

    Reuses :func:`polymer_engine.polymer.identity.build_oligomer`, which joins units at
    their attachment points instead of capping each one -- capping would destroy the
    in-chain linkage and produce a different molecule from the polymer.
    """
    from polymer_engine.polymer.identity import build_oligomer

    if degree_of_polymerization < 2:
        raise ChemistryError(
            "A chain needs at least two repeat units", n=degree_of_polymerization
        )
    built = build_oligomer(repeat_unit_smiles, n_units=degree_of_polymerization)
    if built is None:
        raise ChemistryError(
            "Repeat unit does not have exactly two attachment points; cannot grow a chain",
            repeat_unit=repeat_unit_smiles,
        )
    return built[0]


def embed_chain(molecule: Any, *, seed: int, max_iterations: int = 4000) -> Any:
    """Add hydrogens, embed in 3D and relax with MMFF.

    The relaxation is a *starting geometry*, not a result: it exists so the packed box
    is not full of overlapping atoms that energy minimisation cannot recover from.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.useRandomCoords = True          # long chains fail deterministic embedding
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        raise ChemistryError("3D embedding failed for this chain", seed=seed)
    AllChem.MMFFOptimizeMolecule(molecule, maxIters=max_iterations)
    return molecule


def box_edge_for_density(
    chain_mass_amu: float, n_chains: int, density_kg_m3: float
) -> float:
    """Cubic box edge, in nm, that puts ``n_chains`` at the requested density."""
    if density_kg_m3 <= 0:
        raise ChemistryError("Target density must be positive", density=density_kg_m3)
    mass_kg = chain_mass_amu * n_chains / AVOGADRO * 1e-3
    volume_m3 = mass_kg / density_kg_m3
    return float((volume_m3 * 1e27) ** (1.0 / 3.0))


def _count_molecules(gro_path: Path, atoms_per_chain: int) -> int:
    lines = gro_path.read_text(encoding="utf-8").splitlines()
    total_atoms = int(lines[1].strip())
    if total_atoms % atoms_per_chain:
        raise PolymerEngineError(
            "Packed atom count is not a whole number of chains",
            total_atoms=total_atoms, atoms_per_chain=atoms_per_chain,
        )
    return total_atoms // atoms_per_chain


def build_melt(
    *,
    name: str,
    repeat_unit_smiles: str,
    degree_of_polymerization: int,
    n_chains: int,
    target_density_kg_m3: float,
    directory: str | Path,
    types: dict[str, OplsType],
    runner: Any,
    force_field_source: str = "",
    seed: int = 0,
    box_scale: float = 1.0,
    max_packing_attempts: int = 6,
    box_growth: float = 1.12,
) -> MeltSystem:
    """Build one amorphous melt, or raise with the reason it could not be built.

    ``box_scale`` inflates the box relative to the target density.  Packing directly at
    the final density tends to fail, so the usual route is to pack loose and let the NPT
    stage compress the system -- which is also what makes the resulting density a
    *measurement* rather than an input.

    If ``insert-molecules`` cannot place every chain, the box is grown by ``box_growth``
    and packing is retried, up to ``max_packing_attempts``.  Growing the box is
    preferable to accepting a short pack: a melt whose chain count varies between
    replicas is not the same system, and comparing densities across it would compare
    compositions rather than chemistries.
    """
    from rdkit import Chem
    from rdkit.Chem.Descriptors import MolWt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    chain = embed_chain(grow_chain(repeat_unit_smiles, degree_of_polymerization), seed=seed)
    atoms_per_chain = chain.GetNumAtoms()
    chain_mass = float(MolWt(chain))

    typing = type_molecule(chain, types=types, force_field_source=force_field_source)
    if typing.status is not TypingStatus.SUPPORTED:
        raise ChemistryError(
            "Refusing to build a melt from a chemistry this force field cannot type",
            polymer=name, status=typing.status.value, reason=typing.reason,
        )

    single = directory / "chain.pdb"
    Chem.MolToPDBFile(chain, str(single))
    # Take the names back out of the file the packer will actually read, so topology and
    # structure agree and grompp can run at -maxwarn 0.
    pdb_lines = [
        line for line in single.read_text(encoding="utf-8").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    atom_names = [line[12:16].strip() for line in pdb_lines]
    residue_name = pdb_lines[0][17:20].strip() if pdb_lines else "UNL"
    if len(atom_names) != atoms_per_chain:
        raise PolymerEngineError(
            "PDB atom count does not match the built chain",
            pdb_atoms=len(atom_names), chain_atoms=atoms_per_chain,
        )

    base_edge = box_edge_for_density(chain_mass, n_chains, target_density_kg_m3)
    packed = directory / "packed.gro"
    warnings: list[str] = []
    edge = base_edge * box_scale
    packed_chains = 0
    for attempt in range(1, max_packing_attempts + 1):
        result = runner.run(
            ["insert-molecules", "-ci", single.name, "-nmol", str(n_chains),
             "-box", f"{edge:.5f}", f"{edge:.5f}", f"{edge:.5f}",
             "-o", packed.name, "-try", str(INSERTION_TRIES), "-seed", str(seed or 1)],
            cwd=directory, artifacts=[packed],
        )
        if not result.succeeded:
            raise PolymerEngineError(
                "gmx insert-molecules failed while packing the melt",
                polymer=name, error=result.error, stderr=result.stderr[-600:],
            )
        if not packed.is_file():
            raise PolymerEngineError(
                "insert-molecules reported success but wrote no structure", polymer=name
            )
        packed_chains = _count_molecules(packed, atoms_per_chain)
        if packed_chains >= n_chains:
            break
        if attempt < max_packing_attempts:
            logger.info(
                "%s: packed %d/%d chains in a %.3f nm box; growing the box and retrying",
                name, packed_chains, n_chains, edge,
            )
            edge *= box_growth

    if packed_chains == 0:
        raise PolymerEngineError("insert-molecules packed no chains at all", polymer=name,
                                 box_nm=edge)
    if packed_chains != n_chains:
        warnings.append(
            f"insert-molecules placed {packed_chains} of {n_chains} requested chains "
            f"after {max_packing_attempts} attempts; the topology describes what was "
            f"actually packed, so this system is not composition-matched to its siblings"
        )
        logger.warning("%s: packed %d/%d chains", name, packed_chains, n_chains)

    topology = directory / "topol.top"
    write_opls_topology(typing, chain, topology, molecule_name="POL",
                        atom_names=atom_names, residue_name=residue_name)
    text = topology.read_text(encoding="utf-8").replace(
        "POL          1", f"POL      {packed_chains:5d}"
    )
    topology.write_text(text, encoding="utf-8")

    volume_nm3 = edge**3
    packed_density = packed_chains * chain_mass / AVOGADRO * 1e-3 / (volume_nm3 * 1e-27)
    system = MeltSystem(
        name=name, directory=str(directory),
        n_chains_requested=n_chains, n_chains_packed=packed_chains,
        degree_of_polymerization=degree_of_polymerization,
        atoms_per_chain=atoms_per_chain, total_atoms=packed_chains * atoms_per_chain,
        box_nm=edge, target_density_kg_m3=target_density_kg_m3,
        packed_density_kg_m3=packed_density, chain_mass_amu=chain_mass,
        structure=str(packed), topology=str(topology),
        force_field_source=force_field_source, warnings=warnings,
    )
    logger.info(
        "Built %s: %d chains x %d atoms = %d atoms in a %.3f nm box (%.1f kg/m^3 packed)",
        name, packed_chains, atoms_per_chain, system.total_atoms, edge, packed_density,
    )
    return system


__all__ = [
    "AVOGADRO", "INSERTION_TRIES", "MeltSystem", "box_edge_for_density",
    "build_melt", "embed_chain", "grow_chain",
]
