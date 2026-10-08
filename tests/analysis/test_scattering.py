"""Intermolecular RDF and structure-factor math against analytic limits."""

from __future__ import annotations

import numpy as np
import pytest

from polymer_engine.analysis.scattering import (
    radial_distribution_from_distances,
    structure_factor,
)


def test_uniform_pairs_give_g_of_one():
    # Distances drawn uniformly in volume have the shell-weighted density prop to r^2, so
    # a matching set of pair distances should normalise to g(r) ~ 1.
    rng = np.random.default_rng(0)
    box = 5.0
    n_ref = n_part = 800
    # Sample distances with the ideal r^2 shell weighting out to r_max.
    r_max = 2.0
    u = rng.random(200_000)
    distances = r_max * u ** (1.0 / 3.0)   # pdf ~ r^2 on [0, r_max]
    # Expected total ideal pairs in [0, r_max]: 0.5 * n_ref * (n_part/V) * (4/3 pi r_max^3)
    ideal_total = 0.5 * n_ref * (n_part / box**3) * (4 / 3) * np.pi * r_max**3
    distances = distances[: int(ideal_total)]  # match the count g=1 implies
    _r, g = radial_distribution_from_distances(
        distances, n_reference=n_ref, n_partner=n_part, box_volume_nm3=box**3,
        r_max_nm=r_max, bins=20)
    # away from the noisy first bin, g should sit near 1
    assert np.mean(np.abs(g[3:] - 1.0)) < 0.1


def test_structure_factor_of_a_featureless_liquid_is_one():
    # g(r) == 1 everywhere -> h(r) == 0 -> S(q) == 1 at every q.
    r = np.linspace(0.01, 1.5, 150)
    g = np.ones_like(r)
    _q, s = structure_factor(r, g, number_density_nm3=30.0)
    assert np.allclose(s, 1.0, atol=1e-6)


def test_structure_factor_rises_with_a_correlation_shell():
    # A g(r) with a first-neighbour peak must push S(q) above 1 at some q.
    r = np.linspace(0.01, 1.5, 150)
    g = np.ones_like(r)
    g[(r > 0.4) & (r < 0.55)] = 2.5      # a packing shell near 0.47 nm
    _q, s = structure_factor(r, g, number_density_nm3=30.0)
    assert np.max(s) > 1.2


def test_degenerate_inputs_are_rejected():
    with pytest.raises(ValueError):
        radial_distribution_from_distances(
            np.array([0.1]), n_reference=1, n_partner=1, box_volume_nm3=0.0,
            r_max_nm=1.0, bins=10)
    with pytest.raises(ValueError):
        structure_factor(np.array([0.1]), np.array([1.0]), number_density_nm3=1.0)


def test_amorphous_halo_ignores_a_low_q_feature_and_finds_the_packing_peak():
    from polymer_engine.analysis.scattering import amorphous_halo
    q = np.linspace(2.0, 40.0, 200)
    s = np.ones_like(q)
    s[q < 4.0] = 3.0                       # a spurious low-q correlation-hole feature
    s[(q > 12.0) & (q < 15.0)] = 1.8       # the real amorphous halo near 13.5 /nm (~4.7 A)
    q_peak, s_peak = amorphous_halo(q, s)
    assert 12.0 <= q_peak <= 15.0          # picks the halo, not the taller low-q feature
    assert abs(s_peak - 1.8) < 1e-6


def test_amorphous_halo_falls_back_to_global_max_when_window_is_empty():
    from polymer_engine.analysis.scattering import amorphous_halo
    q = np.linspace(2.0, 6.0, 50)          # entirely below the 8-22 /nm window
    s = np.ones_like(q)
    s[10] = 2.0
    q_peak, _ = amorphous_halo(q, s)
    assert q_peak == q[10]
