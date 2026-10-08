"""Cohesive energy density / solubility parameter calculator."""

from __future__ import annotations

import numpy as np

from polymer_engine.properties.cohesive import CohesiveEnergyDensity


def _series(mean: float, n: int = 200, sd: float = 3.0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, sd, n)


def test_solubility_parameter_matches_the_hand_calculation():
    # E_coh = E_gas - E_liquid = -350 - (-500) = 150 kJ/mol; Vm = 250 cm^3/mol.
    # CED = 1000 * 150 / 250 = 600 MPa; delta = sqrt(600) = 24.49 MPa^0.5.
    liquid = _series(-500.0, seed=1)
    gas = _series(-350.0, seed=2)
    result = CohesiveEnergyDensity().compute(liquid, gas, 250.0, n_replicas=3,
                                             simulation_ns=100.0)
    d = result.as_dict()
    assert abs(d["measurement"]["value"] - 600.0) < 5.0
    assert abs(d["provenance"]["solubility_parameter_mpa_half"] - 24.49) < 0.2


def test_no_net_cohesion_is_reported_as_insufficient_not_an_imaginary_delta():
    # Gas energy below the bulk energy -> no cohesion -> must not return sqrt of a negative.
    liquid = _series(-350.0, seed=3)
    gas = _series(-500.0, seed=4)
    result = CohesiveEnergyDensity().compute(liquid, gas, 250.0)
    assert result.as_dict()["measurement"]["value"] is None


def test_a_larger_molar_volume_lowers_the_density_at_fixed_cohesion():
    liquid, gas = _series(-500.0, seed=5), _series(-350.0, seed=6)
    small = CohesiveEnergyDensity().compute(liquid, gas, 200.0).as_dict()["measurement"]["value"]
    large = CohesiveEnergyDensity().compute(liquid, gas, 400.0).as_dict()["measurement"]["value"]
    assert small > large


def test_too_few_samples_are_rejected():
    result = CohesiveEnergyDensity().compute([1.0, 2.0], [3.0, 4.0], 250.0)
    assert result.as_dict()["measurement"]["value"] is None
