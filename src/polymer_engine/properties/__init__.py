"""Property calculations with declared units, estimators, uncertainty and sampling needs."""

from polymer_engine.core.config import AnalysisDefaults
from polymer_engine.properties.base import (
    PropertyCalculator,
    PropertyClass,
    PropertyDefinition,
    PropertyRegistry,
    PropertyResult,
    SamplingRequirement,
    UncertaintyMethod,
    combine_replica_property,
)
from polymer_engine.properties.mechanical import (
    BulkModulus,
    MechanicalInterpretation,
    MetricKind,
    StressStrainCurve,
    TensileAnalysis,
    mechanical_calculators,
    poisson_ratio,
)
from polymer_engine.properties.structural import (
    ContactNumber,
    EndToEndDistance,
    FreeVolume,
    HydrogenBondCount,
    PersistenceLength,
    RadiusOfGyration,
    structural_calculators,
)
from polymer_engine.properties.thermodynamic import (
    Density,
    Enthalpy,
    PotentialEnergy,
    Pressure,
    StatePoint,
    Temperature,
    ThermalExpansion,
    Volume,
    thermodynamic_calculators,
)
from polymer_engine.properties.transport import (
    DiffusionCoefficient,
    DiffusionRegime,
    MeanSquaredDisplacement,
    RelaxationTime,
    classify_regime,
    transport_calculators,
)


def default_registry(defaults: AnalysisDefaults | None = None) -> PropertyRegistry:
    """Every property the engine can compute."""
    registry = PropertyRegistry()
    for calculator in (
        *thermodynamic_calculators(defaults),
        *structural_calculators(defaults),
        *transport_calculators(defaults),
        *mechanical_calculators(defaults),
    ):
        registry.register(calculator)
    return registry


__all__ = [
    "BulkModulus",
    "ContactNumber",
    "Density",
    "DiffusionCoefficient",
    "DiffusionRegime",
    "EndToEndDistance",
    "Enthalpy",
    "FreeVolume",
    "HydrogenBondCount",
    "MeanSquaredDisplacement",
    "MechanicalInterpretation",
    "MetricKind",
    "PersistenceLength",
    "PotentialEnergy",
    "Pressure",
    "PropertyCalculator",
    "PropertyClass",
    "PropertyDefinition",
    "PropertyRegistry",
    "PropertyResult",
    "RadiusOfGyration",
    "RelaxationTime",
    "SamplingRequirement",
    "StatePoint",
    "StressStrainCurve",
    "Temperature",
    "TensileAnalysis",
    "ThermalExpansion",
    "UncertaintyMethod",
    "Volume",
    "classify_regime",
    "combine_replica_property",
    "default_registry",
    "mechanical_calculators",
    "poisson_ratio",
    "structural_calculators",
    "thermodynamic_calculators",
    "transport_calculators",
]
