#!/usr/bin/env python
"""Every study the finished trajectories already support, run per replica and combined.

The campaign extracts one number -- density -- from simulations that carry far more.
This reads the same ``prod.edr`` and ``prod.xtc`` files and computes, through the
engine's existing calculators and gates:

  from the energy file    volume, enthalpy, potential energy, temperature, pressure,
                          and the bulk modulus from NPT volume fluctuations
  from the trajectory     radius of gyration, end-to-end distance, inter-chain
                          carbon-carbon RDF, mean squared displacement, and a diffusion
                          coefficient issued only if the motion is genuinely diffusive

Read-only with respect to the campaign: it opens files GROMACS has finished writing and
never touches a replica whose production has not completed. Safe to run while a
campaign is live -- it is CPU work and the campaign's mdrun is on the GPU.

    .venv/bin/python scripts/run_studies.py campaign/run4
    .venv/bin/python scripts/run_studies.py campaign/run4 --candidate polypropylene

Every result carries the same four-state gate verdicts as the density does. A study
whose sampling requirement is not met says so; nothing is reported as a number without
its uncertainty and its gates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymer_engine.analysis import md as mdmod
from polymer_engine.analysis.chain_conformation import (
    bond_vector_correlation,
    orientation_autocorrelation,
)
from polymer_engine.analysis.scattering import (
    amorphous_halo,
    radial_distribution_from_counts,
    structure_factor,
)
from polymer_engine.core.logging import configure_logging, get_logger
from polymer_engine.properties.mechanical import BulkModulus
from polymer_engine.properties.structural import (
    ContactNumber,
    EndToEndDistance,
    FreeVolume,
    HydrogenBondCount,
    PersistenceLength,
    RadiusOfGyration,
)
from polymer_engine.properties.thermodynamic import (
    Enthalpy,
    PotentialEnergy,
    Pressure,
    Temperature,
    Volume,
)
from polymer_engine.properties.transport import (
    DiffusionCoefficient,
    MeanSquaredDisplacement,
    RelaxationTime,
)

logger = get_logger("studies")

GMX = "/mnt/data/biodesign/opt/gromacs-2026.3/bin/gmx"

#: Energy-file observables: (gmx energy term, calculator, output stem).
ENERGY_TERMS: tuple[tuple[str, Any, str], ...] = (
    ("Volume", Volume(), "volume"),
    ("Potential", PotentialEnergy(), "potential"),
    ("Enthalpy", Enthalpy(), "enthalpy"),
    ("Temperature", Temperature(), "temperature"),
    ("Pressure", Pressure(), "pressure"),
)

#: Analysis discards this leading fraction, matching the campaign's own convention so a
#: study and the density it sits beside describe the same window.
DISCARD_FRACTION = 0.5


def read_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times, values = [], []
    for line in path.read_text().splitlines():
        if line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            times.append(float(parts[0]))
            values.append(float(parts[1]))
    return np.asarray(times), np.asarray(values)


def extract_energy_term(replica: Path, term: str, stem: str) -> Path | None:
    """One term from prod.edr, cached beside it."""
    target = replica / f"study_{stem}.xvg"
    if target.exists():
        return target
    if not (replica / "prod.edr").exists():
        return None
    done = subprocess.run(
        [GMX, "energy", "-f", "prod.edr", "-o", target.name],
        cwd=replica, input=f"{term}\n", text=True,
        capture_output=True, timeout=300, check=False,
    )
    if done.returncode != 0 or not target.exists():
        logger.info("%s: no %s in prod.edr", replica.name, term)
        return None
    return target


def replica_finished(replica: Path) -> bool:
    """Production has finished: output exists and nothing is still writing the log.

    mdrun writes its final .gro and *then* appends performance statistics to the log, so
    "gro newer than log" is false by a couple of seconds on every successful run. The
    honest test is that the output exists and the log has gone quiet -- a rerun in
    progress has a young log, and an interrupted stage has no .gro at all.
    """
    import time

    log, gro = replica / "prod.log", replica / "prod.gro"
    if not (log.is_file() and gro.is_file()):
        return False
    log_age = time.time() - log.stat().st_mtime
    return log_age > 120.0 and gro.stat().st_mtime >= log.stat().st_mtime - 300.0


@dataclass
class ReplicaStudies:
    candidate: str
    replica: str
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def add(self, name: str, result: Any, extra: dict[str, Any] | None = None) -> None:
        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        if extra:
            payload["study_detail"] = extra
        self.results[name] = payload

    def as_dict(self) -> dict[str, Any]:
        return {"candidate": self.candidate, "replica": self.replica,
                "results": self.results, "skipped": self.skipped}


def production_window(values: np.ndarray) -> slice:
    return slice(int(len(values) * DISCARD_FRACTION), None)


def energy_studies(replica: Path, out: ReplicaStudies, temperature_k: float) -> None:
    volumes_window: np.ndarray | None = None
    ns_simulated: float | None = None

    for term, calculator, stem in ENERGY_TERMS:
        xvg = extract_energy_term(replica, term, stem)
        if xvg is None:
            out.skipped[stem] = f"{term} not present in prod.edr"
            continue
        times, values = read_xvg(xvg)
        if values.size < 20:
            out.skipped[stem] = "series too short"
            continue
        window = production_window(values)
        ns_simulated = float(times[-1] - times[0]) / 1000.0
        result = calculator.compute(
            values[window], times_ps=times[window],
            simulation_ns=ns_simulated,
            provenance={"source": str(xvg.relative_to(replica.parents[2])),
                        "discard_fraction": DISCARD_FRACTION},
        )
        out.add(stem, result)
        if stem == "volume":
            volumes_window = values[window]

    if volumes_window is not None:
        out.add("bulk_modulus", BulkModulus().compute(
            volumes_window, temperature_k,
            simulation_ns=ns_simulated,
            provenance={"ensemble": "NPT, c-rescale equilibration / "
                                    "Parrinello-Rahman production"},
        ))
    else:
        out.skipped["bulk_modulus"] = "no volume series"


def pbc_corrected(replica: Path, mode: str, stride: int) -> Path | None:
    """A PBC-corrected, strided copy of the production trajectory, cached.

    Two different corrections for two different questions. ``mol`` makes each chain
    whole again -- a chain wrapped across the boundary has a nonsense radius of
    gyration. ``nojump`` removes the wrapping jumps entirely, which is what a mean
    squared displacement needs: a wrapped centre of mass teleports by a box length and
    the MSD of that is the box, not the diffusion.

    trjconv reads the tpr, which GROMACS can always read even when MDAnalysis cannot.
    """
    target = replica / f"study_{mode}.xtc"
    if target.exists():
        return target
    if not (replica / "prod.xtc").exists() or not (replica / "prod.tpr").exists():
        return None
    done = subprocess.run(
        [GMX, "trjconv", "-f", "prod.xtc", "-s", "prod.tpr",
         "-pbc", mode, "-skip", str(stride), "-o", target.name],
        cwd=replica, input="System\n", text=True,
        capture_output=True, timeout=900, check=False,
    )
    if done.returncode != 0 or not target.exists():
        logger.warning("%s: trjconv -pbc %s failed: %s", replica.name, mode,
                       (done.stderr or "")[-200:])
        return None
    return target


# Bondi van der Waals radii (nm), for the geometric free-volume probe insertion.
_VDW_NM: dict[str, float] = {
    "H": 0.120, "C": 0.170, "N": 0.155, "O": 0.152, "F": 0.147,
    "P": 0.180, "S": 0.180, "CL": 0.175, "BR": 0.185, "I": 0.198,
}


def _element(name: str) -> str:
    """Element symbol from a gro atom name (e.g. 'CL2' -> 'CL', 'HG21' -> 'H')."""
    letters = "".join(ch for ch in name if ch.isalpha()).upper()
    if letters[:2] in _VDW_NM:
        return letters[:2]
    return letters[:1]


def _atom_radii_nm(universe: Any) -> np.ndarray:
    """Per-atom van der Waals radius in nm, defaulting unknown elements to carbon."""
    return np.array([_VDW_NM.get(_element(a.name), 0.170) for a in universe.atoms])


def _donor_h_and_acceptors(universe: Any) -> tuple[list[tuple[int, int]], np.ndarray]:
    """H-bond donors (H bonded to N/O) and acceptors (N/O), from covalent geometry.

    The gro carries no bond list, so a hydrogen is treated as donor-bound to the nearest
    N or O within 0.13 nm (a covalent X-H bond). Acceptors are every N and O. Returns the
    ``(h_index, heavy_donor_index)`` pairs and the acceptor indices; both are frame-
    independent (covalent bonds do not change) so this is computed once.
    """
    from MDAnalysis.lib.distances import capped_distance
    names = [_element(a.name) for a in universe.atoms]
    pos = universe.atoms.positions
    box = universe.dimensions
    h_idx = np.array([i for i, e in enumerate(names) if e == "H"])
    heavy_idx = np.array([i for i, e in enumerate(names) if e in ("N", "O")])
    if h_idx.size == 0 or heavy_idx.size == 0:
        return [], heavy_idx
    pairs, dists = capped_distance(pos[h_idx], pos[heavy_idx], max_cutoff=1.3,
                                   box=box, return_distances=True)
    # nearest heavy per hydrogen
    best: dict[int, tuple[float, int]] = {}
    for (hi, hj), d in zip(pairs, dists, strict=True):
        h_global, heavy_global = int(h_idx[hi]), int(heavy_idx[hj])
        if h_global not in best or d < best[h_global][0]:
            best[h_global] = (d, heavy_global)
    donor_pairs = [(h, heavy) for h, (_d, heavy) in best.items()]
    return donor_pairs, heavy_idx


def _count_hbonds(pos: np.ndarray, box: np.ndarray,
                  donor_pairs: list[tuple[int, int]], acceptors: np.ndarray) -> int:
    """Geometric hydrogen-bond count in one frame: H..A < 0.25 nm and D-H..A > 120 deg."""
    from MDAnalysis.lib.distances import calc_angles, capped_distance
    if not donor_pairs or acceptors.size == 0:
        return 0
    h_atoms = np.array([h for h, _ in donor_pairs])
    heavy_of_h = dict(donor_pairs)
    pairs, _dists = capped_distance(pos[h_atoms], pos[acceptors], max_cutoff=2.5,
                                    box=box, return_distances=True)
    count = 0
    for hi, aj in pairs:
        h = int(h_atoms[hi])
        a = int(acceptors[aj])
        donor = heavy_of_h[h]
        if a == donor:
            continue  # the H's own heavy atom is not an acceptor for it
        angle = calc_angles(pos[donor][None], pos[h][None], pos[a][None], box=box)[0]
        if np.degrees(angle) > 120.0:
            count += 1
    return count


def _intermolecular_contacts(pos: np.ndarray, box: np.ndarray, atoms_per_chain: int,
                             cutoff_nm: float = 0.6) -> int:
    """Count inter-chain atom pairs within a cutoff in one frame (Angstrom positions)."""
    from MDAnalysis.lib.distances import self_capped_distance
    pairs, _ = self_capped_distance(pos, max_cutoff=cutoff_nm * 10.0, box=box)
    if len(pairs) == 0:
        return 0
    chain_i = pairs[:, 0] // atoms_per_chain
    chain_j = pairs[:, 1] // atoms_per_chain
    return int(np.count_nonzero(chain_i != chain_j))


def _inter_chain_distances_nm(pos: np.ndarray, box: np.ndarray, atoms_per_chain: int,
                              r_max_nm: float) -> np.ndarray:
    """Cross-chain pair separations (nm) within r_max in one frame (Angstrom in).

    The cutoff is clamped below half the smallest box edge: beyond that the minimum-image
    convention breaks (and MDAnalysis refuses the search), so a small box simply reports a
    shorter RDF rather than crashing the whole study.
    """
    from MDAnalysis.lib.distances import self_capped_distance
    max_cut_ang = min(r_max_nm * 10.0, 0.49 * float(np.min(box[:3])))
    if max_cut_ang <= 0:
        return np.empty(0)
    pairs, dists = self_capped_distance(pos, max_cutoff=max_cut_ang, box=box,
                                        return_distances=True)
    if len(pairs) == 0:
        return np.empty(0)
    cross = (pairs[:, 0] // atoms_per_chain) != (pairs[:, 1] // atoms_per_chain)
    return np.asarray(dists)[cross] * 0.1


def _chains_of(universe: Any, atoms_per_chain: int | None) -> list[Any]:
    """Group atoms into whole chains.

    A SMILES-built melt names each chain as one residue, so ``universe.residues`` is the
    chain list. A CHARMM-GUI system names each *monomer* a residue, so that grouping would
    measure a single repeat unit -- Rg of 2.5 A, not of a 30-mer. When the caller knows
    the atom count per chain (from the run manifest) and it evenly divides the system into
    at least two chains, slice by it; otherwise fall back to residues.
    """
    n_atoms = len(universe.atoms)
    if (atoms_per_chain and atoms_per_chain > 0
            and n_atoms % atoms_per_chain == 0 and n_atoms // atoms_per_chain >= 2):
        n = n_atoms // atoms_per_chain
        return [universe.atoms[i * atoms_per_chain:(i + 1) * atoms_per_chain]
                for i in range(n)]
    return list(universe.residues)


def trajectory_studies(replica: Path, out: ReplicaStudies, *, stride: int,
                       atoms_per_chain: int | None = None,
                       free_volume: bool = False) -> None:
    """Chain-level observables, computed per chain in a single trajectory pass.

    The topology is ``prod.gro`` rather than ``prod.tpr``: MDAnalysis cannot read the
    2026 tpr format, and the gro carries what these studies need. Chains are grouped by
    :func:`_chains_of` -- by residue for a SMILES melt, by ``atoms_per_chain`` for a
    CHARMM-GUI system where a residue is one monomer. The selection-based helpers in
    ``analysis.md`` are deliberately not used for Rg: pooling twenty chains into one
    selection measures the size of the box contents, not of a chain.
    """
    if not mdmod.mdanalysis_available():
        out.skipped["trajectory"] = "MDAnalysis is not importable"
        return
    import warnings

    import MDAnalysis as mda

    topology = replica / "prod.gro"
    whole = pbc_corrected(replica, "mol", stride)
    nojump = pbc_corrected(replica, "nojump", stride)
    if not topology.exists() or whole is None:
        out.skipped["trajectory"] = "no PBC-corrected trajectory could be produced"
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        universe = mda.Universe(str(topology), str(whole))
    chains = _chains_of(universe, atoms_per_chain)
    n_frames = len(universe.trajectory)
    if len(chains) < 2 or n_frames < 10:
        out.skipped["trajectory"] = f"{len(chains)} chain(s), {n_frames} frame(s)"
        return

    # What extra per-frame observables are possible depends on the topology: contacts and
    # persistence need chains sliced by atom count; hydrogen bonds need N/O donors.
    apc = atoms_per_chain if (atoms_per_chain and atoms_per_chain > 0) else None
    donor_pairs, acceptors = _donor_h_and_acceptors(universe)
    do_hbonds = len(donor_pairs) > 0
    n_res = len(chains[0].residues)
    do_persist = n_res >= 3 and all(len(c.residues) == n_res for c in chains)

    # Rg and end-to-end are cheap and wanted at full time resolution (they carry the
    # statistics). Contacts, hydrogen bonds and the persistence correlation are expensive
    # per frame (pair enumeration, all-residue COMs) but are structural averages that
    # converge on a handful of frames, so they run on an evenly-spaced subsample drawn
    # from the production window (the second half), never the unequilibrated start.
    prod_start = int(n_frames * DISCARD_FRACTION)
    expensive_frames = set(np.linspace(prod_start, n_frames - 1,
                                       min(n_frames - prod_start, 15), dtype=int))

    # -- one pass: per-chain Rg, end-to-end (scalar + vector) every frame; contacts,
    #    hydrogen bonds and persistence on the subsample --------------------------------
    times, rg_mean, e2e_mean, box_min = [], [], [], []
    e2e_vectors: list[np.ndarray] = []
    contacts, hbonds, contact_times = [], [], []
    persist_sum: np.ndarray | None = None
    persist_n = 0
    bond_lengths: list[float] = []
    # Intermolecular RDF (chain-packing structure), accumulated as a histogram so we never
    # hold millions of pair distances at once.
    rdf_rmax, rdf_bins = 1.5, 150
    rdf_edges = np.linspace(0.0, rdf_rmax, rdf_bins + 1)
    rdf_counts = np.zeros(rdf_bins)
    box_volumes: list[float] = []
    for i, ts in enumerate(universe.trajectory):
        times.append(float(ts.time))
        rg_mean.append(float(np.mean([c.atoms.radius_of_gyration() for c in chains])))
        e2e_mean.append(float(np.mean([
            np.linalg.norm(c.atoms[0].position - c.atoms[-1].position)
            for c in chains])))
        e2e_vectors.append(np.array([c.atoms[-1].position - c.atoms[0].position
                                     for c in chains]))
        box_min.append(float(np.min(ts.dimensions[:3])))
        if i not in expensive_frames:
            continue
        contact_times.append(float(ts.time))
        pos, box = universe.atoms.positions, ts.dimensions
        if apc is not None:
            # Per chain, so the value is interpretable (a raw all-pairs count scales with
            # system size and reads as a meaningless six-figure number).
            contacts.append(_intermolecular_contacts(pos, box, apc) / len(chains))
            d_inter = _inter_chain_distances_nm(pos, box, apc, rdf_rmax)
            rdf_counts += np.histogram(d_inter, bins=rdf_edges)[0]
            box_volumes.append(float(box[0] * box[1] * box[2]) * 1e-3)  # A^3 -> nm^3
        if do_hbonds:
            hbonds.append(_count_hbonds(pos, box, donor_pairs, acceptors))
        if do_persist:
            # Vectorised: one COM per residue for the whole frame, reshaped to
            # (chains, monomers, 3). Residues are chain-contiguous because chains are
            # contiguous atom slices, so the reshape recovers each chain's backbone.
            res_com = universe.atoms.center_of_mass(compound="residues")
            beads = res_com.reshape(len(chains), n_res, 3)
            corr, bond = bond_vector_correlation(beads)
            persist_sum = corr if persist_sum is None else persist_sum + corr
            persist_n += 1
            bond_lengths.append(bond)
    times_ps = np.asarray(times)
    ns = float(times_ps[-1] - times_ps[0]) / 1000.0 if len(times_ps) > 1 else None
    window = production_window(times_ps)
    frame_note = {"n_chains": len(chains), "pbc": "mol (chains made whole)",
                  "stride": stride,
                  "averaging": "mean over chains per frame; statistics over time"}

    out.add("radius_of_gyration", RadiusOfGyration().compute(
        np.asarray(rg_mean)[window] * 0.1, times_ps=times_ps[window],
        simulation_ns=ns, provenance=frame_note))
    out.add("end_to_end_distance", EndToEndDistance().compute(
        np.asarray(e2e_mean)[window] * 0.1, times_ps=times_ps[window],
        simulation_ns=ns, provenance=frame_note))

    # -- finite-size check: a chain must not interact with its own periodic image, so the
    # box edge must exceed twice the chain's radius of gyration. Below 1.0 the density and
    # conformation are compromised; 1.0-1.5 is acceptable but tight; >=1.5 is comfortable.
    # This is the box-size axis of the size dependence (distinct from chain-length/DP).
    rg_nm = float(np.mean(np.asarray(rg_mean)[window])) * 0.1
    box_nm = float(np.mean(np.asarray(box_min)[window])) * 0.1
    ratio = box_nm / (2.0 * rg_nm) if rg_nm > 0 else float("nan")
    verdict = ("compromised: chain sees its periodic image" if ratio < 1.0
               else "acceptable but tight" if ratio < 1.5 else "comfortable")
    out.add("finite_size", {
        "box_edge_nm": round(box_nm, 3), "radius_of_gyration_nm": round(rg_nm, 3),
        "two_rg_nm": round(2.0 * rg_nm, 3), "box_over_2rg": round(ratio, 3),
        "criterion": "box_edge > 2*Rg", "verdict": verdict})

    # -- chain relaxation time from the end-to-end orientation autocorrelation ----------
    vectors = np.asarray(e2e_vectors)[window]          # (frames, chains, 3)
    if vectors.shape[0] >= 2:
        acf = orientation_autocorrelation(vectors)
        lag_ps = (times_ps[window] - times_ps[window][0])[: acf.size]
        out.add("relaxation_time", RelaxationTime().compute(
            lag_ps, acf, provenance={
                "observable": "end-to-end unit-vector autocorrelation C(t)=<u(0).u(t)>",
                **frame_note}))

    # -- persistence length from the monomer-COM backbone bond correlation --------------
    if do_persist and persist_sum is not None and persist_n > 0:
        mean_corr = persist_sum / persist_n
        mean_bond_nm = float(np.mean(bond_lengths)) * 0.1
        out.add("persistence_length", PersistenceLength().compute(
            mean_corr, mean_bond_nm, provenance={
                "observable": "bond-vector correlation along the monomer-COM backbone",
                "n_beads_per_chain": n_res, "n_frames_averaged": persist_n}))

    # -- intermolecular contacts and hydrogen bonds, on the production-window subsample --
    contact_times_arr = np.asarray(contact_times)
    if apc is not None and contacts:
        out.add("contact_number", ContactNumber().compute(
            np.asarray(contacts), times_ps=contact_times_arr, simulation_ns=ns,
            provenance={"observable": "inter-chain atom pairs within 0.6 nm, per chain",
                        "n_frames_sampled": len(contacts), **frame_note}))
    if do_hbonds and hbonds:
        out.add("hydrogen_bond_count", HydrogenBondCount().compute(
            np.asarray(hbonds), times_ps=contact_times_arr, simulation_ns=ns,
            provenance={"observable": "H..A < 0.25 nm and D-H..A > 120 deg, per frame",
                        "donors": len(donor_pairs), "n_frames_sampled": len(hbonds),
                        **frame_note}))

    # -- geometric free-volume fraction on the last production frame --------------------
    # Off by default: brute-force probe insertion is O(grid x atoms) and costs minutes on
    # a 10k-atom melt, which is not worth paying on every replica in a large fleet.
    if not free_volume:
        out.skipped["free_volume"] = "not requested (--free-volume); expensive on large melts"
    else:
        try:
            universe.trajectory[-1]
            radii = _atom_radii_nm(universe)
            d = universe.dimensions
            box_last = (float(d[0]) * 0.1, float(d[1]) * 0.1, float(d[2]) * 0.1)
            # 0.08 nm grid, not the 0.05 default: insertion is O(grid x atoms), so a fine
            # grid on a 10k-atom melt is minutes. 0.08 nm keeps the fraction meaningful at
            # a fraction of the cost -- and the probe radius, not the grid, sets the answer.
            out.add("free_volume", FreeVolume().compute(
                universe.atoms.positions * 0.1, radii, box_last,
                probe_radius_nm=0.11, grid_spacing_nm=0.08,
                provenance={"observable": "probe-sphere insertion on the final frame",
                            "probe_radius_nm": 0.11, "grid_spacing_nm": 0.08}))
        except Exception as exc:  # noqa: BLE001 - one failed study must not kill the rest
            out.skipped["free_volume"] = f"{type(exc).__name__}: {exc}"

    # -- RDF on the whole-molecule trajectory --------------------------------
    try:
        pair = mdmod.rdf(mdmod.AnalysisSpec(
            topology=str(topology), trajectory=str(whole),
            selection="name C*", selection_b="name C*",
            start=n_frames // 2))
        if pair is not None:
            out.add("rdf_carbon_carbon", {
                "r_nm": pair.r_nm.tolist(), "g_r": pair.g_r.tolist(),
                "n_frames": pair.n_frames,
                "note": "carbon-carbon pair distribution over the production half",
            })
    except Exception as exc:  # noqa: BLE001 - one failed study must not kill the rest
        out.skipped["rdf"] = f"{type(exc).__name__}: {exc}"

    # -- intermolecular RDF (chain packing) and structure factor S(q) -------------------
    # Unlike the carbon RDF, this excludes a chain's pairs with itself, so the structure
    # is how chains pack against each other; S(q) is its scattering-observable transform.
    if apc is not None and rdf_counts.sum() > 0 and box_volumes:
        n_atoms = len(universe.atoms)
        n_other = n_atoms - apc  # partner atoms on other chains, per reference atom
        r_nm, g_r = radial_distribution_from_counts(
            rdf_counts, rdf_edges, n_reference=n_atoms, n_partner=n_other,
            box_volume_nm3=float(np.mean(box_volumes)), n_frames=len(box_volumes))
        contact = float(g_r[np.argmax(g_r > 0.01)]) if np.any(g_r > 0.01) else 0.0
        out.add("rdf_intermolecular", {
            "r_nm": r_nm.tolist(), "g_r": g_r.tolist(),
            "n_frames": len(box_volumes), "correlation_hole_g_at_contact": round(contact, 3),
            "note": "inter-chain pair distribution (a chain's own pairs excluded). In a "
                    "dense melt g(r) < 1 at short range and rises to 1 -- the polymer "
                    "correlation hole, not a lattice peak."})
        rho = n_atoms / float(np.mean(box_volumes))
        q, s_q = structure_factor(r_nm, g_r, number_density_nm3=rho)
        halo_q, halo_s = amorphous_halo(q, s_q)
        out.add("structure_factor", {
            "q_inv_nm": q.tolist(), "s_q": s_q.tolist(),
            "amorphous_halo_q_inv_nm": round(halo_q, 2),
            "amorphous_halo_spacing_ang": round(2.0 * np.pi / halo_q * 10.0, 2),
            "amorphous_halo_s": round(halo_s, 3),
            "note": "structure factor from the intermolecular g(r); its peak is the "
                    "amorphous halo, the inter-chain packing distance seen in scattering."})

    # -- chain-COM MSD on the nojump trajectory ------------------------------
    if nojump is None:
        out.skipped["msd"] = "no nojump trajectory"
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        universe2 = mda.Universe(str(topology), str(nojump))
    chains2 = _chains_of(universe2, atoms_per_chain)
    coms, times2 = [], []
    for ts in universe2.trajectory:
        times2.append(float(ts.time))
        coms.append([c.atoms.center_of_mass() for c in chains2])
    positions = np.asarray(coms)              # (frames, chains, 3), angstrom
    times2_arr = np.asarray(times2)
    half = positions.shape[0] // 2
    if half < 5:
        out.skipped["msd"] = "too few frames for an MSD"
        return
    lags = np.arange(1, half)
    msd = np.empty(lags.size)
    for i, lag in enumerate(lags):
        displacement = positions[lag:] - positions[:-lag]
        msd[i] = float(np.mean(np.sum(displacement**2, axis=-1)))
    lag_ps = lags * float(times2_arr[1] - times2_arr[0])
    msd_nm2 = msd * 0.01                       # angstrom^2 -> nm^2

    msd_note = {"observable": "chain centre-of-mass MSD, averaged over chains and "
                              "time origins", "pbc": "nojump", "stride": stride}
    out.add("msd", MeanSquaredDisplacement().compute(
        lag_ps, msd_nm2, provenance=msd_note))
    out.add("diffusion_coefficient", DiffusionCoefficient().compute(
        lag_ps, msd_nm2, simulation_ns=ns,
        provenance={**msd_note,
                    "caveat": "no finite-size (Yeh-Hummer) correction"}))


def _verdict(payload: dict[str, Any]) -> str:
    return str(payload.get("gate_status") or payload.get("determination") or "?")


def _value(payload: dict[str, Any]) -> str:
    m = payload.get("measurement") or {}
    v, u = m.get("value"), m.get("uncertainty")
    unit = (payload.get("definition") or {}).get("units", "")
    if v is None:
        return "—"
    return f"{v:.4g} ± {u:.2g} {unit}" if u is not None else f"{v:.4g} {unit}"


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--stride", type=int, default=5,
                        help="trajectory frame stride (20 ps output x 5 = one sample "
                             "per 100 ps; chain observables decorrelate far slower)")
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--free-volume", action="store_true",
                        help="also compute the free-volume fraction (slow: grid insertion)")
    parser.add_argument("--skip-trajectory", action="store_true",
                        help="energy-file studies only (much faster)")
    args = parser.parse_args()

    # The density campaign nests replicas under root/experiments/<candidate>/; the CHARMM
    # run repo puts them directly under root/<candidate>/. Accept either: prefer an
    # experiments/ subdir when present, otherwise treat root itself as the candidate dir.
    experiments = args.root / "experiments"
    if not experiments.is_dir():
        if any(p.is_dir() and list(p.glob("replica_*")) for p in args.root.iterdir()):
            experiments = args.root
        else:
            print(f"no experiments under {args.root} (no experiments/ subdir and no "
                  f"<candidate>/replica_* directly under it)", file=sys.stderr)
            return 2

    studies_dir = args.root / "studies"
    studies_dir.mkdir(exist_ok=True)
    all_out: list[ReplicaStudies] = []

    # A CHARMM run manifest, when present, records atoms_per_chain per candidate -- needed
    # to group monomers-as-residues back into whole chains for Rg and the finite-size check.
    atoms_per_chain_by_slug: dict[str, int] = {}
    run_manifest = args.root / "runs_manifest.json"
    if run_manifest.is_file():
        man = json.loads(run_manifest.read_text())
        atoms_per_chain_by_slug = {r["slug"]: r.get("atoms_per_chain")
                                   for r in man.get("runs", []) if r.get("atoms_per_chain")}

    for candidate in sorted(p for p in experiments.iterdir() if p.is_dir()):
        if args.candidate and candidate.name != args.candidate:
            continue
        apc = atoms_per_chain_by_slug.get(candidate.name)
        for replica in sorted(candidate.glob("replica_*")):
            if not replica_finished(replica):
                logger.info("%s/%s: production not finished; skipped",
                            candidate.name, replica.name)
                continue
            out = ReplicaStudies(candidate.name, replica.name)
            energy_studies(replica, out, args.temperature_k)
            if not args.skip_trajectory:
                trajectory_studies(replica, out, stride=args.stride, atoms_per_chain=apc,
                                   free_volume=args.free_volume)
            all_out.append(out)
            (studies_dir / f"{candidate.name}_{replica.name}.json").write_text(
                json.dumps(out.as_dict(), indent=2, default=str) + "\n")

    # -- summary -------------------------------------------------------
    lines = ["# Studies from existing trajectories", "",
             f"Generated {datetime.now(UTC).isoformat()} · root `{args.root}`", "",
             "Same gates as the density: a value without enough independent samples is "
             "INCONCLUSIVE, not a smaller number.", "",
             "| Candidate | Replica | Study | Value | Verdict |", "|---|---|---|---|---|"]
    for out in all_out:
        for name, payload in sorted(out.results.items()):
            if name == "rdf_carbon_carbon":
                lines.append(f"| {out.candidate} | {out.replica[-2:]} | rdf (C-C) | "
                             f"{payload.get('n_frames')} frames | series |")
                continue
            lines.append(f"| {out.candidate} | {out.replica[-2:]} | {name} | "
                         f"{_value(payload)} | {_verdict(payload)} |")
        for name, why in sorted(out.skipped.items()):
            lines.append(f"| {out.candidate} | {out.replica[-2:]} | {name} | skipped | "
                         f"{why[:60]} |")
    (studies_dir / "STUDIES.md").write_text("\n".join(lines) + "\n")

    print(f"\n{len(all_out)} replica(s) analysed -> {studies_dir}/STUDIES.md")
    for out in all_out:
        n_pass = sum(1 for payload in out.results.values()
                     if _verdict(payload) == "pass")
        print(f"  {out.candidate}/{out.replica}: {len(out.results)} studies "
              f"({n_pass} pass), {len(out.skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
