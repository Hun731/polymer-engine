"""Polymer parameterization: from candidate to qualified force-field route.

The subsystem keeps three states apart, because conflating them is how a topology file
gets mistaken for a scientific result:

``PARAMETERIZED``  a topology exists · ``VALIDATED``  it was checked against independent
evidence · ``QUALIFIED``  it was checked well enough for a stated property class.

Backends are pluggable. Nothing above :mod:`polymer_engine.parameterization.backend`
names a force field.
"""

from polymer_engine.parameterization.capability import CapabilityState, discover_tools
from polymer_engine.parameterization.engine import ParameterizationEngine
from polymer_engine.parameterization.models import (
    ForceFieldAssessment,
    ParameterizationRequest,
    ParameterizationResult,
    ParameterizationState,
    ParameterizationValidation,
    PropertyClass,
    QMPriority,
)

__all__ = [
    "CapabilityState", "ForceFieldAssessment", "ParameterizationEngine",
    "ParameterizationRequest", "ParameterizationResult", "ParameterizationState",
    "ParameterizationValidation", "PropertyClass", "QMPriority", "discover_tools",
]
