"""Cohesive energy density and the Hildebrand solubility parameter.

The cohesive energy is the energy released when the chains are brought from isolation into
the bulk -- the intermolecular attraction holding the melt together. Per mole of chains it
is ``E_gas - E_liquid``: the potential energy of one chain alone in vacuum minus its share
of the bulk potential energy. Divided by the molar volume it is the cohesive energy density
(CED), and its square root is the Hildebrand solubility parameter, the single number that
predicts what a polymer dissolves in and what it is miscible with.

This calculator is pure: it takes the two energy series and the molar volume and does the
bookkeeping. The vacuum single-chain simulation that supplies ``E_gas`` is run elsewhere.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from polymer_engine.core.models import Determination, GateReport, Measurement
from polymer_engine.properties.base import (
    PropertyCalculator,
    PropertyClass,
    PropertyDefinition,
    PropertyResult,
    SamplingRequirement,
    UncertaintyMethod,
)

# 1 kJ/mol per cm^3/mol = 1 kJ/cm^3... no: (kJ/mol)/(cm^3/mol) = kJ/cm^3 = 1000 J/cm^3 =
# 1000 MPa. So CED[MPa] = 1000 * E_coh[kJ/mol] / V_molar[cm^3/mol].
_KJ_PER_MOL_PER_CM3_TO_MPA = 1000.0


class CohesiveEnergyDensity(PropertyCalculator):
    """Cohesive energy density and Hildebrand solubility parameter from bulk vs gas energy."""

    definition = PropertyDefinition(
        name="cohesive_energy_density",
        property_class=PropertyClass.THERMODYNAMIC,
        units="MPa",
        observable="Bulk minus single-chain-in-vacuum potential energy, per molar volume",
        estimator="CED = (E_gas - E_liquid) / V_molar;  delta = sqrt(CED)",
        uncertainty_method=UncertaintyMethod.CORRELATION_AWARE_SE,
        sampling=SamplingRequirement(
            min_effective_samples=20.0,
            min_replicas=1,
            min_simulation_ns=5.0,
            notes="Both energies are means, so they converge quickly; the gas reference "
                  "must sample the isolated-chain conformations, not a single minimum.",
        ),
        description="Cohesive energy density; its square root is the solubility parameter.",
        caveats="A non-polarisable force field and a finite chain both shift the absolute "
                "value; compare like with like (same field, same DP) rather than to a "
                "handbook number.",
    )

    def compute(
        self,
        liquid_energy_kj_per_chain: Sequence[float] | np.ndarray,
        gas_energy_kj_per_chain: Sequence[float] | np.ndarray,
        molar_volume_cm3: float,
        *,
        n_replicas: int = 1,
        simulation_ns: float | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> PropertyResult:
        from polymer_engine.analysis.statistics import effective_sample_size

        liquid = np.asarray(liquid_energy_kj_per_chain, dtype=float).ravel()
        gas = np.asarray(gas_energy_kj_per_chain, dtype=float).ravel()
        if liquid.size < 10 or gas.size < 10:
            return self.unknown("at least ten energy samples are needed for each state")
        if molar_volume_cm3 <= 0:
            return self.unknown("molar volume must be positive",
                                determination=Determination.UNKNOWN)
        if not (np.all(np.isfinite(liquid)) and np.all(np.isfinite(gas))):
            return self.unknown("energy series contain non-finite values",
                                determination=Determination.UNKNOWN)

        e_liquid, e_gas = float(liquid.mean()), float(gas.mean())
        e_coh = e_gas - e_liquid  # kJ/mol per chain
        if e_coh <= 0:
            return self.unknown(
                "gas-phase energy is not above the bulk energy, so there is no net "
                "cohesion to report; check that the gas reference is a single isolated chain")

        ced_mpa = _KJ_PER_MOL_PER_CM3_TO_MPA * e_coh / molar_volume_cm3
        delta_mpa_half = math.sqrt(ced_mpa)

        # Propagate the two sampling errors of the means into the CED, then into delta.
        ess_l = max(effective_sample_size(liquid), 1.0)
        ess_g = max(effective_sample_size(gas), 1.0)
        se_liquid = float(liquid.std(ddof=1)) / math.sqrt(ess_l)
        se_gas = float(gas.std(ddof=1)) / math.sqrt(ess_g)
        se_coh = math.hypot(se_liquid, se_gas)
        ced_unc = _KJ_PER_MOL_PER_CM3_TO_MPA * se_coh / molar_volume_cm3
        delta_unc = ced_unc / (2.0 * delta_mpa_half) if delta_mpa_half > 0 else None

        measurement = Measurement(
            name=self.definition.name,
            value=ced_mpa,
            uncertainty=ced_unc,
            units="MPa",
            n_samples=int(min(liquid.size, gas.size)),
            effective_samples=min(ess_l, ess_g),
            method="CED = (E_gas - E_liquid) / V_molar",
            notes="solubility parameter delta = sqrt(CED); see provenance",
        )
        report = GateReport(name=f"property:{self.definition.name}")
        report.gates.extend(self.sampling_gates(
            measurement, n_replicas=n_replicas, equilibration_shown=True,
            simulation_ns=simulation_ns))
        return PropertyResult(
            definition=self.definition,
            measurement=measurement,
            report=report,
            n_replicas=n_replicas,
            provenance={
                "solubility_parameter_mpa_half": delta_mpa_half,
                "solubility_parameter_uncertainty": delta_unc,
                "cohesive_energy_kj_per_mol": e_coh,
                "e_liquid_kj_per_chain": e_liquid,
                "e_gas_kj_per_chain": e_gas,
                "molar_volume_cm3": molar_volume_cm3,
                **(provenance or {}),
            },
        )


__all__ = ["CohesiveEnergyDensity"]
