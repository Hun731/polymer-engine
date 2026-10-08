"""Pure-numpy chain conformation and correlation math.

These functions take coordinate/vector arrays and return the reduced quantities the
property calculators consume -- an orientation autocorrelation for a relaxation time, a
bond-vector correlation along the backbone for a persistence length. They hold no MD
library dependency and no I/O, so they can be unit-tested against analytic cases; the
trajectory extraction that feeds them lives in the study script.
"""

from __future__ import annotations

import numpy as np


def orientation_autocorrelation(vectors: np.ndarray) -> np.ndarray:
    """First-order orientational autocorrelation C(t) = <u(0).u(t)>.

    ``vectors`` has shape ``(n_frames, n_chains, 3)`` and need not be normalised -- each
    vector is unit-normalised here, so C is the mean cosine between a chain's end-to-end
    direction at time origin and a lag later, averaged over chains and all time origins.
    C(0) = 1 by construction; the lag at which it decays measures reorientation.

    Returns C indexed by lag in frames, length ``n_frames`` (C[0] == 1).
    """
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 3 or v.shape[2] != 3 or v.shape[0] < 2:
        raise ValueError("vectors must be (n_frames, n_chains, 3) with >= 2 frames")
    norms = np.linalg.norm(v, axis=2, keepdims=True)
    norms[norms == 0.0] = 1.0
    u = v / norms
    n_frames = u.shape[0]
    c = np.empty(n_frames)
    c[0] = 1.0
    for lag in range(1, n_frames):
        dots = np.sum(u[lag:] * u[:-lag], axis=2)  # (n_origins, n_chains)
        c[lag] = float(np.mean(dots))
    return c


def bond_vector_correlation(beads: np.ndarray) -> tuple[np.ndarray, float]:
    """Backbone bond-vector correlation vs separation, for a persistence length.

    ``beads`` has shape ``(n_chains, n_beads, 3)`` -- an ordered coarse backbone, e.g. the
    centre of mass of each monomer along each chain. Bonds are consecutive differences;
    the correlation at separation ``s`` is the mean cosine between bonds ``i`` and
    ``i+s``, averaged over chains and starting positions. The persistence length follows
    from the geometric decay C(s) = exp(-s / (lp / l_bond)).

    Returns ``(correlations, mean_bond_length)`` with ``correlations[0] == 1`` and the
    bond length in the same length unit as ``beads``.
    """
    b = np.asarray(beads, dtype=float)
    if b.ndim != 3 or b.shape[2] != 3 or b.shape[1] < 3:
        raise ValueError("beads must be (n_chains, n_beads>=3, 3)")
    bonds = b[:, 1:, :] - b[:, :-1, :]                       # (chains, n_bonds, 3)
    lengths = np.linalg.norm(bonds, axis=2)                  # (chains, n_bonds)
    mean_bond = float(np.mean(lengths))
    unit = bonds / np.where(lengths[..., None] == 0.0, 1.0, lengths[..., None])
    n_bonds = unit.shape[1]
    corr = np.empty(n_bonds)
    for s in range(n_bonds):
        dots = np.sum(unit[:, s:, :] * unit[:, : n_bonds - s, :], axis=2)
        corr[s] = float(np.mean(dots))
    return corr, mean_bond


def persistence_length_from_correlation(
    correlations: np.ndarray, bond_length: float,
) -> float:
    """Persistence length from an exponential fit to the bond-vector correlation.

    C(s) = exp(-s * l_bond / lp) for an ideal worm-like chain, so a line fit to
    ln C(s) over the range where C stays positive gives lp = -l_bond / slope. Returns
    NaN if the correlation never provides two usable positive points.
    """
    c = np.asarray(correlations, dtype=float)
    s = np.arange(c.size)
    usable = c > 1e-3
    if int(np.count_nonzero(usable)) < 2:
        return float("nan")
    slope = np.polyfit(s[usable], np.log(c[usable]), 1)[0]
    if slope >= 0.0:
        return float("nan")
    return float(-bond_length / slope)


__all__ = ["bond_vector_correlation", "orientation_autocorrelation",
           "persistence_length_from_correlation"]
