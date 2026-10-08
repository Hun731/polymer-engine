"""Chain conformation math against analytic cases."""

from __future__ import annotations

import numpy as np
import pytest

from polymer_engine.analysis.chain_conformation import (
    bond_vector_correlation,
    orientation_autocorrelation,
    persistence_length_from_correlation,
)


def test_a_constant_orientation_has_correlation_one_at_every_lag():
    # A vector that never rotates stays perfectly correlated with itself.
    vectors = np.tile(np.array([0.0, 0.0, 2.0]), (10, 4, 1))
    c = orientation_autocorrelation(vectors)
    assert c[0] == pytest.approx(1.0)
    assert np.allclose(c, 1.0)


def test_orthogonal_flip_decorrelates_to_zero():
    # Alternating between two orthogonal directions: even lags correlate, odd lags do not.
    frames = []
    for i in range(20):
        v = np.array([1.0, 0.0, 0.0]) if i % 2 == 0 else np.array([0.0, 1.0, 0.0])
        frames.append(np.tile(v, (3, 1)))
    c = orientation_autocorrelation(np.asarray(frames))
    assert c[0] == pytest.approx(1.0)
    assert c[1] == pytest.approx(0.0, abs=1e-9)   # orthogonal one step apart
    assert c[2] == pytest.approx(1.0)             # same direction two steps apart


def test_a_straight_backbone_has_unit_correlation_and_the_right_bond_length():
    # Collinear, evenly spaced beads: every bond is parallel, correlation is 1 throughout.
    beads = np.zeros((5, 6, 3))
    beads[:, :, 0] = np.arange(6) * 0.25          # 0.25 nm spacing along x
    corr, bond = bond_vector_correlation(beads)
    assert bond == pytest.approx(0.25)
    assert np.allclose(corr, 1.0)


def test_persistence_length_recovers_an_exponential_decay():
    # Build C(s) = exp(-s * l_bond / lp) exactly and check the fit inverts it.
    l_bond, lp = 0.25, 1.0
    s = np.arange(8)
    corr = np.exp(-s * l_bond / lp)
    recovered = persistence_length_from_correlation(corr, l_bond)
    assert recovered == pytest.approx(lp, rel=1e-6)


def test_degenerate_inputs_are_rejected_not_guessed():
    with pytest.raises(ValueError):
        orientation_autocorrelation(np.zeros((1, 3, 3)))     # one frame
    with pytest.raises(ValueError):
        bond_vector_correlation(np.zeros((4, 2, 3)))          # < 3 beads
    assert np.isnan(persistence_length_from_correlation(np.array([1.0, -0.5]), 0.25))
