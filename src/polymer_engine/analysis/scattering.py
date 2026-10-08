"""Pair-structure math: an intermolecular radial distribution and a structure factor.

The carbon-carbon RDF the study script computes over all pairs is dominated by the bonded
neighbours a chain has with itself, so its first peak is just a bond length. The structure
that says how chains *pack against each other* is the intermolecular g(r) -- pairs on
different chains only -- and its Fourier transform, the static structure factor S(q), is
what a scattering experiment measures. Both are pure functions of distances and a density
here; the trajectory extraction that supplies them lives in the study script.
"""

from __future__ import annotations

import numpy as np


def radial_distribution_from_counts(
    counts: np.ndarray, edges: np.ndarray, *, n_reference: int, n_partner: int,
    box_volume_nm3: float, n_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise histogrammed pair counts into g(r).

    ``counts`` are pair counts per bin summed over ``n_frames`` frames (for an
    intermolecular RDF, cross-molecule pairs each counted once); ``edges`` are the bin
    edges in nm. The ideal-gas expectation each shell is normalised against is
    ``0.5 * n_reference * (n_partner / V) * shell_volume`` -- the partner atoms a reference
    atom would see in that shell at uniform density, over references and halved for double
    counting. A uniform distribution gives g(r) = 1. Returns ``(r_centres_nm, g_r)``.
    """
    if box_volume_nm3 <= 0 or edges.size < 2:
        raise ValueError("box volume must be positive and edges must define >= 1 bin")
    per_frame = np.asarray(counts, dtype=float) / max(n_frames, 1)
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    ideal = 0.5 * n_reference * (n_partner / box_volume_nm3) * shell_vol
    ideal[ideal == 0.0] = np.nan
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, np.nan_to_num(per_frame / ideal)


def radial_distribution_from_distances(
    distances_nm: np.ndarray, *, n_reference: int, n_partner: int,
    box_volume_nm3: float, r_max_nm: float, bins: int, n_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram pair distances and normalise into g(r). See
    :func:`radial_distribution_from_counts` for the normalisation."""
    if r_max_nm <= 0 or bins < 1:
        raise ValueError("r_max and bins must be positive")
    edges = np.linspace(0.0, r_max_nm, bins + 1)
    counts, _ = np.histogram(distances_nm, bins=edges)
    return radial_distribution_from_counts(
        counts, edges, n_reference=n_reference, n_partner=n_partner,
        box_volume_nm3=box_volume_nm3, n_frames=n_frames)


def structure_factor(
    r_nm: np.ndarray, g_r: np.ndarray, *, number_density_nm3: float,
    q_min_inv_nm: float = 2.0, q_max_inv_nm: float = 40.0, n_q: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Static structure factor S(q) from a radial distribution function.

    S(q) = 1 + 4*pi*rho * integral[ (g(r) - 1) * sin(qr)/(qr) * r^2 dr ], the isotropic
    Fourier transform of the pair correlation. The integral runs over the tabulated g(r);
    truncation at the box-limited r_max introduces ripple at low q, so q_min defaults away
    from zero. Returns ``(q, S_q)``.
    """
    r = np.asarray(r_nm, dtype=float)
    h = np.asarray(g_r, dtype=float) - 1.0
    if r.size < 2 or r.size != h.size:
        raise ValueError("r and g(r) must be equal-length arrays of length >= 2")
    q = np.linspace(q_min_inv_nm, q_max_inv_nm, n_q)
    dr = float(r[1] - r[0])
    s = np.empty(n_q)
    for i, qi in enumerate(q):
        integrand = h * np.sin(qi * r) / (qi * r) * r ** 2
        s[i] = 1.0 + 4.0 * np.pi * number_density_nm3 * float(np.sum(integrand) * dr)
    return q, s


def amorphous_halo(
    q_inv_nm: np.ndarray, s_q: np.ndarray, *,
    q_window_inv_nm: tuple[float, float] = (8.0, 22.0),
) -> tuple[float, float]:
    """The amorphous halo: the S(q) peak in the nearest-neighbour packing window.

    A polymer melt's halo -- the inter-chain packing distance a scattering experiment
    sees -- sits near q = 10-16 /nm (real-space 4-6 A). A plain argmax over all q can
    instead land on a low-q feature of the correlation hole, reporting a spurious "spacing"
    of 10-20 A. Restricting the search to the physical window fixes that; if the window is
    empty the global maximum is returned rather than nothing. Returns ``(q_peak, S_peak)``.
    """
    q = np.asarray(q_inv_nm, dtype=float)
    s = np.asarray(s_q, dtype=float)
    lo, hi = q_window_inv_nm
    mask = (q >= lo) & (q <= hi)
    idx = np.where(mask)[0]
    if idx.size == 0:
        idx = np.arange(q.size)
    peak = idx[int(np.argmax(s[idx]))]
    return float(q[peak]), float(s[peak])


__all__ = ["amorphous_halo", "radial_distribution_from_counts",
           "radial_distribution_from_distances", "structure_factor"]
