"""Explicit unit handling.

Rule 9 of the engine charter: never silently change units.  Every numeric quantity
that crosses a module boundary carries its unit string, and conversions go through
:func:`convert` which raises rather than guessing.

The canonical internal unit system is GROMACS-native:

===============  ==========
quantity         unit
===============  ==========
length           nm
time             ps
energy           kJ/mol
temperature      K
pressure         bar
mass             amu
density          kg/m^3
force            kJ/mol/nm
angle            deg
===============  ==========
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from polymer_engine.core.errors import ParameterValidationError

# dimension -> canonical unit
CANONICAL: Final[dict[str, str]] = {
    "length": "nm",
    "time": "ps",
    "energy": "kJ/mol",
    "temperature": "K",
    "pressure": "bar",
    "mass": "amu",
    "density": "kg/m^3",
    "force": "kJ/mol/nm",
    "angle": "deg",
    "area": "nm^2",
    "volume": "nm^3",
    "diffusivity": "nm^2/ps",
    "inverse_temperature": "1/K",
    "charge": "e",
    "dimensionless": "1",
}

# unit -> (dimension, factor to canonical, offset to canonical)
# value_canonical = value * factor + offset
_UNITS: Final[dict[str, tuple[str, float, float]]] = {
    # length
    "nm": ("length", 1.0, 0.0),
    "angstrom": ("length", 0.1, 0.0),
    "A": ("length", 0.1, 0.0),
    "pm": ("length", 1e-3, 0.0),
    "m": ("length", 1e9, 0.0),
    # time
    "ps": ("time", 1.0, 0.0),
    "fs": ("time", 1e-3, 0.0),
    "ns": ("time", 1e3, 0.0),
    "us": ("time", 1e6, 0.0),
    "s": ("time", 1e12, 0.0),
    # energy (molar)
    "kJ/mol": ("energy", 1.0, 0.0),
    "J/mol": ("energy", 1e-3, 0.0),
    "kcal/mol": ("energy", 4.184, 0.0),
    "eV": ("energy", 96.48533212331, 0.0),
    "hartree": ("energy", 2625.4996394799, 0.0),
    # temperature
    "K": ("temperature", 1.0, 0.0),
    "degC": ("temperature", 1.0, 273.15),
    # pressure
    "bar": ("pressure", 1.0, 0.0),
    "atm": ("pressure", 1.01325, 0.0),
    "Pa": ("pressure", 1e-5, 0.0),
    "kPa": ("pressure", 1e-2, 0.0),
    "MPa": ("pressure", 10.0, 0.0),
    # mass
    "amu": ("mass", 1.0, 0.0),
    "g/mol": ("mass", 1.0, 0.0),
    # density
    "kg/m^3": ("density", 1.0, 0.0),
    "g/cm^3": ("density", 1000.0, 0.0),
    "g/mL": ("density", 1000.0, 0.0),
    # force
    "kJ/mol/nm": ("force", 1.0, 0.0),
    "kcal/mol/A": ("force", 41.84, 0.0),
    # angle
    "deg": ("angle", 1.0, 0.0),
    "rad": ("angle", 57.29577951308232, 0.0),
    # area
    "nm^2": ("area", 1.0, 0.0),
    "angstrom^2": ("area", 1e-2, 0.0),
    "A^2": ("area", 1e-2, 0.0),
    # volume
    "nm^3": ("volume", 1.0, 0.0),
    "angstrom^3": ("volume", 1e-3, 0.0),
    "A^3": ("volume", 1e-3, 0.0),
    "cm^3": ("volume", 1e21, 0.0),
    "L": ("volume", 1e24, 0.0),
    "m^3": ("volume", 1e27, 0.0),
    # diffusivity: 1 cm^2/s = 1e14 nm^2 / 1e12 ps = 1e2 nm^2/ps
    "nm^2/ps": ("diffusivity", 1.0, 0.0),
    "angstrom^2/ps": ("diffusivity", 1e-2, 0.0),
    "cm^2/s": ("diffusivity", 1e2, 0.0),
    "m^2/s": ("diffusivity", 1e6, 0.0),
    # inverse temperature (thermal expansion coefficients)
    "1/K": ("inverse_temperature", 1.0, 0.0),
    "K^-1": ("inverse_temperature", 1.0, 0.0),
    # charge: force-field partial charges are in elementary charges
    "e": ("charge", 1.0, 0.0),
    "C": ("charge", 6.241509074e18, 0.0),
    # dimensionless
    "1": ("dimensionless", 1.0, 0.0),
    "": ("dimensionless", 1.0, 0.0),
}

# Physical constants in the canonical system.
GAS_CONSTANT_KJ_PER_MOL_K: Final[float] = 0.008314462618
AVOGADRO: Final[float] = 6.02214076e23


def dimension_of(unit: str) -> str:
    """Return the physical dimension of ``unit``.

    Raises ``ParameterValidationError`` for units the engine does not know, rather
    than assuming dimensionless.
    """
    try:
        return _UNITS[unit][0]
    except KeyError:
        raise ParameterValidationError(f"Unknown unit {unit!r}", unit=unit) from None


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert ``value`` between two units of the same dimension.

    Cross-dimension conversion is an error, never a silent pass-through.
    """
    src_dim, src_f, src_o = _lookup(from_unit)
    dst_dim, dst_f, dst_o = _lookup(to_unit)
    if src_dim != dst_dim:
        raise ParameterValidationError(
            "Refusing to convert across physical dimensions",
            from_unit=from_unit,
            to_unit=to_unit,
            from_dimension=src_dim,
            to_dimension=dst_dim,
        )
    canonical = value * src_f + src_o
    return (canonical - dst_o) / dst_f


def to_canonical(value: float, unit: str) -> tuple[float, str]:
    """Convert ``value`` to the canonical unit of its dimension."""
    dim = dimension_of(unit)
    target = CANONICAL[dim]
    return convert(value, unit, target), target


def _lookup(unit: str) -> tuple[str, float, float]:
    try:
        return _UNITS[unit]
    except KeyError:
        raise ParameterValidationError(f"Unknown unit {unit!r}", unit=unit) from None


def kT(temperature_k: float) -> float:
    """Thermal energy RT in kJ/mol at ``temperature_k``."""
    if temperature_k <= 0:
        raise ParameterValidationError("Temperature must be positive", temperature_k=temperature_k)
    return GAS_CONSTANT_KJ_PER_MOL_K * temperature_k


@dataclass(frozen=True, slots=True)
class Quantity:
    """A number that knows its unit.

    Arithmetic is deliberately *not* implemented: this type exists to make units
    explicit at module boundaries and in provenance records, not to be a full
    dimensional-analysis library.
    """

    value: float
    unit: str

    def __post_init__(self) -> None:
        dimension_of(self.unit)  # validates

    @property
    def dimension(self) -> str:
        return dimension_of(self.unit)

    def to(self, unit: str) -> Quantity:
        return Quantity(convert(self.value, self.unit, unit), unit)

    def canonical(self) -> Quantity:
        value, unit = to_canonical(self.value, self.unit)
        return Quantity(value, unit)

    def as_dict(self) -> dict[str, object]:
        return {"value": self.value, "unit": self.unit, "dimension": self.dimension}


__all__ = [
    "AVOGADRO",
    "CANONICAL",
    "GAS_CONSTANT_KJ_PER_MOL_K",
    "Quantity",
    "convert",
    "dimension_of",
    "kT",
    "to_canonical",
]
