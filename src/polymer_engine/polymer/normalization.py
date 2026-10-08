"""Property-name and unit normalisation.

Experimental polymer data arrives with the same quantity spelled a dozen ways and
reported in whichever unit the original author preferred.  Glass transition
temperature in particular shows up in both K and degC, and a 273-degree error is
easy to make and hard to notice.

This module maps a free-text property name onto a canonical name with a known unit,
and converts values explicitly.  A property it does not recognise is returned
``UNKNOWN`` rather than passed through with an assumed unit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from polymer_engine.core.errors import ParameterValidationError
from polymer_engine.core.models import Determination, Measurement
from polymer_engine.core.units import convert, dimension_of


@dataclass(frozen=True, slots=True)
class PropertySpec:
    """A canonical property: its unit and the physically plausible range."""

    name: str
    canonical_unit: str
    description: str
    minimum: float | None = None
    maximum: float | None = None

    def plausible(self, value: float) -> bool:
        below = self.minimum is not None and value < self.minimum
        above = self.maximum is not None and value > self.maximum
        return not (below or above)


PROPERTIES: dict[str, PropertySpec] = {
    "glass_transition_temperature": PropertySpec(
        "glass_transition_temperature", "K", "Glass transition temperature (Tg)", minimum=4.0, maximum=1500.0
    ),
    "melting_temperature": PropertySpec(
        "melting_temperature", "K", "Crystalline melting temperature (Tm)", minimum=4.0, maximum=2000.0
    ),
    "decomposition_temperature": PropertySpec(
        "decomposition_temperature", "K", "Onset of thermal decomposition", minimum=100.0, maximum=2500.0
    ),
    "density": PropertySpec("density", "kg/m^3", "Bulk mass density", minimum=1.0, maximum=10000.0),
    "youngs_modulus": PropertySpec("youngs_modulus", "MPa", "Tensile (Young's) modulus", minimum=0.0, maximum=1.0e6),
    "tensile_strength": PropertySpec("tensile_strength", "MPa", "Ultimate tensile strength", minimum=0.0, maximum=1.0e5),
    "elongation_at_break": PropertySpec("elongation_at_break", "1", "Strain at break (fraction)", minimum=0.0, maximum=100.0),
    "cohesive_energy_density": PropertySpec(
        "cohesive_energy_density", "MPa", "Cohesive energy density", minimum=0.0, maximum=1.0e4
    ),
    "solubility_parameter": PropertySpec(
        "solubility_parameter", "MPa", "Hildebrand solubility parameter (as MPa^0.5, stored numerically)",
        minimum=0.0, maximum=100.0,
    ),
    "refractive_index": PropertySpec("refractive_index", "1", "Refractive index", minimum=1.0, maximum=3.0),
    "dielectric_constant": PropertySpec("dielectric_constant", "1", "Relative permittivity", minimum=1.0, maximum=1000.0),
    "number_average_molar_mass": PropertySpec("number_average_molar_mass", "g/mol", "Mn", minimum=1.0),
    "weight_average_molar_mass": PropertySpec("weight_average_molar_mass", "g/mol", "Mw", minimum=1.0),
    "polydispersity_index": PropertySpec("polydispersity_index", "1", "Mw/Mn", minimum=1.0, maximum=100.0),
    "degree_of_crystallinity": PropertySpec("degree_of_crystallinity", "1", "Crystalline fraction", minimum=0.0, maximum=1.0),
    "water_contact_angle": PropertySpec("water_contact_angle", "deg", "Static water contact angle", minimum=0.0, maximum=180.0),
    "gas_permeability": PropertySpec("gas_permeability", "1", "Permeability (Barrer, stored numerically)", minimum=0.0),
}

#: Free-text spellings that map onto a canonical property.
ALIASES: dict[str, str] = {
    "tg": "glass_transition_temperature",
    "glass transition": "glass_transition_temperature",
    "glass transition temp": "glass_transition_temperature",
    "glass_transition_temp": "glass_transition_temperature",
    "glass transition temperature": "glass_transition_temperature",
    "glasstransitiontemperature": "glass_transition_temperature",
    "tm": "melting_temperature",
    "melting point": "melting_temperature",
    "melt temperature": "melting_temperature",
    "melting temperature": "melting_temperature",
    "td": "decomposition_temperature",
    "decomposition temperature": "decomposition_temperature",
    "rho": "density",
    "density": "density",
    "bulk density": "density",
    "mass density": "density",
    "e": "youngs_modulus",
    "youngs modulus": "youngs_modulus",
    "young s modulus": "youngs_modulus",
    "young's modulus": "youngs_modulus",
    "tensile modulus": "youngs_modulus",
    "elastic modulus": "youngs_modulus",
    "tensile strength": "tensile_strength",
    "ultimate tensile strength": "tensile_strength",
    "uts": "tensile_strength",
    "elongation at break": "elongation_at_break",
    "strain at break": "elongation_at_break",
    "ced": "cohesive_energy_density",
    "cohesive energy density": "cohesive_energy_density",
    "solubility parameter": "solubility_parameter",
    "hildebrand parameter": "solubility_parameter",
    "delta": "solubility_parameter",
    "refractive index": "refractive_index",
    "n": "refractive_index",
    "dielectric constant": "dielectric_constant",
    "relative permittivity": "dielectric_constant",
    "mn": "number_average_molar_mass",
    "number average molecular weight": "number_average_molar_mass",
    "mw": "weight_average_molar_mass",
    "weight average molecular weight": "weight_average_molar_mass",
    "pdi": "polydispersity_index",
    "dispersity": "polydispersity_index",
    "crystallinity": "degree_of_crystallinity",
    "degree of crystallinity": "degree_of_crystallinity",
    "water contact angle": "water_contact_angle",
    "contact angle": "water_contact_angle",
    "permeability": "gas_permeability",
}

#: Unit spellings found in the wild, mapped onto core.units names.  ``None`` marks a
#: spelling that needs a scale factor and is resolved through :data:`UNIT_SCALE`.
UNIT_ALIASES: dict[str, str | None] = {
    "k": "K", "kelvin": "K",
    "c": "degC", "degc": "degC", "°c": "degC", "celsius": "degC", "oc": "degC",
    "g/cm3": "g/cm^3", "g/cm^3": "g/cm^3", "g cm-3": "g/cm^3", "g/cc": "g/cm^3", "g/ml": "g/mL",
    "kg/m3": "kg/m^3", "kg/m^3": "kg/m^3", "kg m-3": "kg/m^3",
    "mpa": "MPa", "gpa": None, "kpa": "kPa", "pa": "Pa", "bar": "bar", "atm": "atm",
    "g/mol": "g/mol", "da": "g/mol", "dalton": "g/mol", "kda": None,
    "%": None, "percent": None,
    "deg": "deg", "degree": "deg", "degrees": "deg", "rad": "rad",
    "nm": "nm", "a": "A", "angstrom": "A", "å": "A",
    "kj/mol": "kJ/mol", "kcal/mol": "kcal/mol",
    "": "1", "none": "1", "dimensionless": "1", "-": "1",
}

#: Units needing a scale factor rather than a rename (handled before conversion).
UNIT_SCALE: dict[str, tuple[str, float]] = {
    "gpa": ("MPa", 1000.0),
    "kda": ("g/mol", 1000.0),
    "%": ("1", 0.01),
    "percent": ("1", 0.01),
}


def normalize_property_name(raw: str) -> str | None:
    """Map a free-text property name onto a canonical one, or ``None``."""
    if not raw:
        return None
    key = re.sub(r"[\s_\-]+", " ", raw.strip().lower())
    key = key.replace("(", "").replace(")", "").strip()
    if key in PROPERTIES:
        return key
    collapsed = key.replace(" ", "_")
    if collapsed in PROPERTIES:
        return collapsed
    return ALIASES.get(key) or ALIASES.get(collapsed)


def normalize_unit(raw: str | None) -> tuple[str | None, float]:
    """Map a free-text unit onto a core unit plus a scale factor.

    Returns ``(None, 1.0)`` for an unrecognised unit -- the caller must then treat
    the value as ``UNKNOWN`` rather than assume a unit.
    """
    if raw is None:
        return "1", 1.0
    key = raw.strip().lower()
    if key in UNIT_SCALE:
        unit, scale = UNIT_SCALE[key]
        return unit, scale
    mapped = UNIT_ALIASES.get(key, raw.strip())
    if mapped is None:
        return None, 1.0
    try:
        dimension_of(mapped)
    except ParameterValidationError:
        return None, 1.0
    return mapped, 1.0


def normalize_property(
    raw_name: str,
    value: Any,
    unit: str | None = None,
    *,
    uncertainty: float | None = None,
    source: str | None = None,
) -> Measurement:
    """Normalise one reported property into a canonical :class:`Measurement`.

    Anything that cannot be normalised confidently -- unknown property, unknown
    unit, non-numeric value, or a value outside the physically plausible range --
    comes back as a non-``KNOWN`` measurement carrying the reason.
    """
    canonical = normalize_property_name(raw_name)
    if canonical is None:
        return Measurement.unknown(
            raw_name.strip() or "unnamed_property",
            reason=f"unrecognised property name {raw_name!r}",
            determination=Determination.REQUIRES_VALIDATION,
        )
    spec = PROPERTIES[canonical]

    numeric = _to_float(value)
    if numeric is None:
        return Measurement.unknown(
            canonical, units=spec.canonical_unit, reason=f"non-numeric value {value!r}"
        )

    source_unit, scale = normalize_unit(unit)
    if source_unit is None:
        return Measurement.unknown(
            canonical,
            units=spec.canonical_unit,
            reason=f"unrecognised unit {unit!r}; refusing to assume {spec.canonical_unit}",
            determination=Determination.REQUIRES_VALIDATION,
        )

    scaled = numeric * scale
    try:
        converted = convert(scaled, source_unit, spec.canonical_unit)
    except ParameterValidationError as exc:
        return Measurement.unknown(
            canonical,
            units=spec.canonical_unit,
            reason=f"cannot convert {source_unit} to {spec.canonical_unit}: {exc.message}",
            determination=Determination.REQUIRES_VALIDATION,
        )

    converted_uncertainty: float | None = None
    if uncertainty is not None:
        # Uncertainty scales with the multiplicative part of the conversion only;
        # an additive offset (K <-> degC) does not shift an interval width.
        span = convert(scaled + uncertainty * scale, source_unit, spec.canonical_unit)
        converted_uncertainty = abs(span - converted)

    if not spec.plausible(converted):
        return Measurement.unknown(
            canonical,
            units=spec.canonical_unit,
            reason=(
                f"value {converted:.6g} {spec.canonical_unit} is outside the plausible range "
                f"[{spec.minimum}, {spec.maximum}]"
            ),
            determination=Determination.REQUIRES_VALIDATION,
        )

    return Measurement(
        name=canonical,
        value=converted,
        uncertainty=converted_uncertainty,
        units=spec.canonical_unit,
        method=source or "reported",
        notes=None if source_unit == spec.canonical_unit else f"converted from {unit}",
    )


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        # Tolerate a leading comparison or approximation marker.
        text = re.sub(r"^[~<>≈]+\s*", "", text)
        try:
            return float(text)
        except ValueError:
            # "80-100" style ranges: take the midpoint and let the caller see the note.
            # The en dash in the pattern is intentional: literature tables use it for ranges.
            match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)", text)  # noqa: RUF001
            if match:
                return (float(match.group(1)) + float(match.group(2))) / 2.0
            return None
    return None


__all__ = [
    "ALIASES",
    "PROPERTIES",
    "UNIT_ALIASES",
    "UNIT_SCALE",
    "PropertySpec",
    "normalize_property",
    "normalize_property_name",
    "normalize_unit",
]
