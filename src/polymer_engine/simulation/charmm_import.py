"""Convert a CHARMM-GUI single-chain build into a GROMACS chain we can pack.

CHARMM-GUI hands back a chain in CHARMM format -- a PSF, coordinates, and the CHARMM36
parameter sets the residues use. Our melt builder works in GROMACS. This bridges them:
it drives ParmEd (isolated in ``.paramenv``, never imported by the engine) to write a
GROMACS topology and structure for the one chain, which the melt builder then packs into
a bulk system.

The value of this path is that the chain carries **CHARMM-GUI's curated parameters** --
for PLA, validated residue parameters with no analogy penalty -- while the bulk system,
its density, and its uncertainty come from our own validated campaign rather than
CHARMM-GUI's coarse-grained equilibration.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.logging import get_logger

logger = get_logger("simulation.charmm_import")

REPO_ROOT = Path(__file__).resolve().parents[3]
PARAMENV = REPO_ROOT / ".paramenv" / "bin" / "python"
WORKER = REPO_ROOT / "scripts" / "charmm2gmx_worker.py"


@dataclass
class ConvertedChain:
    """A GROMACS chain converted from a CHARMM-GUI build, with its provenance."""

    topology: Path
    coordinates: Path
    n_atoms: int
    n_residues: int
    net_charge: float
    param_files: list[str] = field(default_factory=list)
    source_psf: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"topology": str(self.topology), "coordinates": str(self.coordinates),
                "n_atoms": self.n_atoms, "n_residues": self.n_residues,
                "net_charge": self.net_charge, "param_files": list(self.param_files),
                "source_psf": self.source_psf}


class CharmmImportError(RuntimeError):
    """Conversion could not be completed."""


def _call_worker(request: dict[str, Any], *, timeout: float = 600.0) -> dict[str, Any]:
    if not PARAMENV.exists():
        raise CharmmImportError(
            f"no .paramenv at {PARAMENV}; ParmEd is needed for CHARMM->GROMACS "
            f"conversion (install: .paramenv/bin/pip install parmed)")
    done = subprocess.run(
        [str(PARAMENV), str(WORKER), json.dumps(request)],
        capture_output=True, text=True, timeout=timeout, check=False)
    line = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""
    if not line:
        raise CharmmImportError(f"conversion worker produced no output; "
                                f"stderr: {done.stderr[-400:]}")
    result = json.loads(line)
    if not result.get("ok"):
        raise CharmmImportError(result.get("error", "conversion failed"))
    return result


def available() -> bool:
    """Whether ParmEd is importable in the isolated environment."""
    try:
        return bool(_call_worker({"action": "capabilities"}, timeout=60).get("available"))
    except Exception:  # noqa: BLE001 - absence is the answer
        return False


def convert_chain(archive_dir: Path, prefix: Path) -> ConvertedChain:
    """Find the PSF, coordinates and toppar in an extracted archive and convert.

    ``archive_dir`` is a directory a CHARMM-GUI archive was extracted into. The layout
    is discovered rather than assumed: the PSF and coordinates live under
    ``psfcrdreader/`` and the parameters under ``toppar/``, but they are located by
    search so a layout change is reported, not silently mis-handled.
    """
    psf = next(iter(sorted(archive_dir.rglob("*.psf"))), None)
    if psf is None:
        raise CharmmImportError(f"no PSF file under {archive_dir}")
    coords = (next(iter(sorted(archive_dir.rglob("*_raw.pdb"))), None)
              or next(iter(sorted(archive_dir.rglob("*.pdb"))), None)
              or next(iter(sorted(archive_dir.rglob("*.crd"))), None))
    if coords is None:
        raise CharmmImportError(f"no coordinate file under {archive_dir}")
    toppar = next(iter(sorted(p for p in archive_dir.rglob("toppar") if p.is_dir())),
                  None)
    if toppar is None:
        raise CharmmImportError(f"no toppar directory under {archive_dir}")

    prefix.parent.mkdir(parents=True, exist_ok=True)
    logger.info("converting %s -> GROMACS via ParmEd", psf.name)
    result = _call_worker({
        "action": "convert", "psf": str(psf), "coordinates": str(coords),
        "toppar": str(toppar), "prefix": str(prefix)})
    return ConvertedChain(
        topology=Path(result["topology"]), coordinates=Path(result["coordinates"]),
        n_atoms=result["n_atoms"], n_residues=result["n_residues"],
        net_charge=result["net_charge"], param_files=result.get("param_files", []),
        source_psf=str(psf))


@dataclass
class CharmmMelt:
    """A packed bulk melt built from a converted CHARMM-GUI chain."""

    topology: Path
    coordinates: Path
    n_chains_packed: int
    n_chains_requested: int
    atoms_per_chain: int
    box_nm: float
    target_density_kg_m3: float

    def as_dict(self) -> dict[str, Any]:
        return {"topology": str(self.topology), "coordinates": str(self.coordinates),
                "n_chains_packed": self.n_chains_packed,
                "n_chains_requested": self.n_chains_requested,
                "atoms_per_chain": self.atoms_per_chain, "box_nm": self.box_nm,
                "target_density_kg_m3": self.target_density_kg_m3}


def _chain_mass_amu(top: Path) -> float:
    """Total mass of the chain, from the [atoms] mass column of the topology."""
    masses: list[float] = []
    in_atoms = False
    for line in top.read_text().splitlines():
        stripped = line.split(";")[0].strip()
        if stripped.startswith("["):
            in_atoms = stripped.lower().startswith("[ atoms")
            continue
        if in_atoms and stripped:
            parts = stripped.split()
            if len(parts) >= 8:
                masses.append(float(parts[7]))
    return sum(masses)


def pack_charmm_melt(
    chain: ConvertedChain, *, directory: Path, n_chains: int,
    target_density_kg_m3: float, runner: Any, seed: int = 1,
    box_scale: float = 1.0, box_growth: float = 1.12, max_attempts: int = 6,
) -> CharmmMelt:
    """Pack N copies of a converted chain into a periodic box at a target density.

    The same discipline the SMILES melt builder uses: pack with insert-molecules, grow
    the box and retry if it cannot place every chain, and read the achieved count back
    from the structure -- a melt with fewer chains than requested, at an unrecorded
    density, is not the system that was asked for. The topology's molecule count is set
    to what was actually packed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    atoms_per_chain = chain.n_atoms
    chain_mass = _chain_mass_amu(chain.topology)

    # A cubic box that puts n_chains at the target density (SI: kg, m).
    avogadro = 6.02214076e23
    mass_kg = chain_mass * n_chains / avogadro * 1e-3
    base_edge_nm = float((mass_kg / target_density_kg_m3 * 1e27) ** (1.0 / 3.0))
    edge = base_edge_nm * box_scale

    packed = directory / "packed.gro"
    n_packed = 0
    for _ in range(max_attempts):
        result = runner.run(
            ["insert-molecules", "-ci", str(chain.coordinates), "-nmol", str(n_chains),
             "-box", f"{edge}", f"{edge}", f"{edge}", "-seed", str(seed),
             "-o", str(packed)], cwd=directory)
        if result.succeeded and packed.exists():
            n_packed = _count_from_gro(packed, atoms_per_chain)
            if n_packed >= n_chains:
                n_packed = n_chains
                break
        edge *= box_growth
    if n_packed == 0:
        raise CharmmImportError(
            f"insert-molecules could not pack any chain into the box for {chain.source_psf}")

    # Melt topology: the converted single-chain top, with the molecule count set to what
    # was packed. ParmEd names the molecule 'system1'; keep whatever it used.
    melt_top = directory / "topol.top"
    _write_melt_topology(chain.topology, melt_top, n_molecules=n_packed)
    logger.info("packed %d/%d chains of %s at %.1f kg/m^3 in a %.3f nm box",
                n_packed, n_chains, Path(chain.source_psf).name, target_density_kg_m3,
                edge)
    return CharmmMelt(
        topology=melt_top, coordinates=packed, n_chains_packed=n_packed,
        n_chains_requested=n_chains, atoms_per_chain=atoms_per_chain,
        box_nm=edge, target_density_kg_m3=target_density_kg_m3)


def _count_from_gro(gro: Path, atoms_per_chain: int) -> int:
    lines = gro.read_text().splitlines()
    if len(lines) < 3:
        return 0
    total_atoms = int(lines[1].strip())
    return total_atoms // atoms_per_chain if atoms_per_chain else 0


def _write_melt_topology(single_top: Path, melt_top: Path, *, n_molecules: int) -> None:
    """Copy a single-chain topology, setting the [molecules] count to n_molecules."""
    lines = single_top.read_text().splitlines()
    out: list[str] = []
    in_molecules = False
    for line in lines:
        stripped = line.split(";")[0].strip()
        if stripped.startswith("["):
            in_molecules = stripped.lower().startswith("[ molecules")
            out.append(line)
            continue
        if in_molecules and stripped:
            name = stripped.split()[0]
            out.append(f"{name}                 {n_molecules}")
            continue
        out.append(line)
    melt_top.write_text("\n".join(out) + "\n")


__all__ = ["CharmmImportError", "CharmmMelt", "ConvertedChain", "available",
           "convert_chain", "pack_charmm_melt"]
