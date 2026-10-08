"""Which parts of our build specification the live form can actually express (§7).

`PolymerBuilderSpec` is the engine's single description of a requested system. The live
Polymer Builder form is somebody else's description of the same thing. This module
compares the two and returns a verdict per field, without inventing a correspondence
between them.

Three verdicts, and the middle one is the important one:

``MAPPED``
    Exactly one discovered control corresponds to this specification field.

``AMBIGUOUS``
    Several controls could. **Not resolved here.** Picking one by sort order or label
    similarity would silently decide which box a degree of polymerisation goes into,
    and getting that wrong builds a different polymer while looking successful.

``UNAVAILABLE``
    No discovered control corresponds. Whether that means the form cannot express the
    field, or merely that the capture did not reach the control that does, is not
    distinguishable from a page -- so the reason says which of those it might be rather
    than asserting the form lacks the capability.

A specification field the form cannot express is not automatically a blocker either: a
tacticity nobody asked for is irrelevant, and the report separates fields the *request*
actually needs from fields that merely exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from polymer_engine.browser.selectors import FormSchema
from polymer_engine.core.logging import get_logger
from polymer_engine.simulation.charmm_gui_spec import PolymerBuilderSpec, SystemType

logger = get_logger("browser.spec_mapping")


class MappingVerdict(str, Enum):
    MAPPED = "MAPPED"
    AMBIGUOUS = "AMBIGUOUS"
    UNAVAILABLE = "UNAVAILABLE"
    #: The specification does not set this field, so nothing needs to map.
    NOT_REQUESTED = "NOT_REQUESTED"


#: Specification field -> the semantic name discovery uses for it. This table is the one
#: place the two vocabularies meet, and it is deliberately explicit rather than derived
#: from attribute names: `n_chains` and "Number of chains" are the same idea only
#: because a person says so.
SPEC_TO_SEMANTIC: dict[str, str] = {
    "name": "monomer",
    "degree_of_polymerization": "degree_of_polymerization",
    "n_chains": "n_chains",
    "tacticity": "tacticity",
    "system_type": "system_type",
    "comonomers": "composition",
    "sequence": "sequence",
    "solvent": "solvent",
    "water_model": "solvent",
    "temperature_k": "temperature",
    "target_density_kg_m3": "density",
    "box_nm": "box",
    "terminal_groups": "terminal",
}

#: Fields the engine tracks that Polymer Builder has no reason to expose, because they
#: describe what *we* will do with the system rather than how it is built.
ENGINE_ONLY: frozenset[str] = frozenset({
    "polymer_id", "repeat_unit_smiles", "force_field", "pressure_bar", "notes",
    "charmm_gui_job_id", "solvate",
})


@dataclass
class FieldMapping:
    """One specification field and what the live form offers for it."""

    spec_field: str
    semantic: str
    verdict: MappingVerdict
    requested_value: Any = None
    control_keys: list[str] = field(default_factory=list)
    reason: str = ""
    #: True when the request actually sets this field, so an UNAVAILABLE blocks.
    required_by_request: bool = False

    @property
    def blocks_build(self) -> bool:
        return (self.required_by_request
                and self.verdict in {MappingVerdict.UNAVAILABLE,
                                     MappingVerdict.AMBIGUOUS})

    def as_dict(self) -> dict[str, Any]:
        return {"spec_field": self.spec_field, "semantic": self.semantic,
                "verdict": self.verdict.value, "requested_value": self.requested_value,
                "control_keys": list(self.control_keys), "reason": self.reason,
                "required_by_request": self.required_by_request,
                "blocks_build": self.blocks_build}


@dataclass
class SpecMapping:
    """The whole comparison, and whether a build can proceed from it."""

    polymer: str
    system_type: str
    schema_name: str
    schema_discovered: bool
    fields: list[FieldMapping] = field(default_factory=list)
    engine_only: list[str] = field(default_factory=list)
    #: Controls the form offers that no specification field describes.
    unmapped_controls: list[str] = field(default_factory=list)

    @property
    def blockers(self) -> list[FieldMapping]:
        return [f for f in self.fields if f.blocks_build]

    @property
    def buildable(self) -> bool:
        return self.schema_discovered and not self.blockers

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for mapping in self.fields:
            counts[mapping.verdict.value] = counts.get(mapping.verdict.value, 0) + 1
        return dict(sorted(counts.items()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer": self.polymer, "system_type": self.system_type,
            "schema": self.schema_name, "schema_discovered": self.schema_discovered,
            "buildable": self.buildable, "counts": self.counts(),
            "blockers": [f.spec_field for f in self.blockers],
            "fields": [f.as_dict() for f in self.fields],
            "engine_only": list(self.engine_only),
            "unmapped_controls": list(self.unmapped_controls),
        }

    def to_markdown(self) -> str:
        lines = [f"# Specification mapping — {self.polymer}", "",
                 f"System type `{self.system_type}` · schema `{self.schema_name}` "
                 f"({'discovered' if self.schema_discovered else '**not discovered**'})",
                 "", "| Spec field | Semantic | Verdict | Requested | Controls |",
                 "|---|---|---|---|---|"]
        for mapping in self.fields:
            lines.append(
                f"| `{mapping.spec_field}` | {mapping.semantic} | "
                f"{mapping.verdict.value}{' **(blocks)**' if mapping.blocks_build else ''} "
                f"| {mapping.requested_value if mapping.requested_value is not None else '—'} "
                f"| {', '.join(f'`{k}`' for k in mapping.control_keys) or '—'} |")
        if self.engine_only:
            lines += ["", "## Engine-only fields", "",
                      "Tracked by the engine, not asked for by the form, because they "
                      "describe what happens to the system afterwards: "
                      + ", ".join(f"`{f}`" for f in self.engine_only), ""]
        if self.unmapped_controls:
            lines += ["## Form controls the specification does not describe", "",
                      ", ".join(f"`{c}`" for c in self.unmapped_controls),
                      "", "These are not errors. They are settings the specification "
                      "currently leaves at whatever the form defaults to, which is "
                      "worth reviewing before a build is trusted.", ""]
        return "\n".join(lines)


def _requested_values(spec: PolymerBuilderSpec) -> dict[str, Any]:
    """What the specification actually sets, in specification terms."""
    values: dict[str, Any] = {
        "name": spec.name,
        "degree_of_polymerization": spec.degree_of_polymerization,
        "tacticity": spec.tacticity,
        "system_type": spec.system_type.value,
        "sequence": spec.sequence,
        "solvent": spec.solvent,
        "water_model": spec.water_model,
        "temperature_k": spec.temperature_k,
        "target_density_kg_m3": spec.target_density_kg_m3,
        "box_nm": spec.box_nm,
        "terminal_groups": spec.terminal_groups,
        "comonomers": ([f"{c.monomer}:{c.fraction}" for c in spec.comonomers]
                       if spec.comonomers else None),
    }
    # A single chain is one chain by definition, so the count is not a request.
    values["n_chains"] = spec.n_chains if spec.system_type.n_chains_meaningful else None
    return values


def map_spec_to_form(
    spec: PolymerBuilderSpec, schema: FormSchema, semantics: dict[str, list[str]],
) -> SpecMapping:
    """Compare our specification against a discovered form.  Guesses nothing."""
    requested = _requested_values(spec)
    mapping = SpecMapping(
        polymer=spec.name, system_type=spec.system_type.value,
        schema_name=schema.name, schema_discovered=schema.discovered,
        engine_only=sorted(ENGINE_ONLY),
    )

    for spec_field, semantic in SPEC_TO_SEMANTIC.items():
        value = requested.get(spec_field)
        candidates = [key for key in semantics.get(semantic, []) if key in schema.fields]
        is_requested = value not in (None, "", [], ())

        if not is_requested:
            verdict, reason = MappingVerdict.NOT_REQUESTED, (
                "the specification does not set this field, so nothing needs to map")
        elif len(candidates) == 1:
            verdict, reason = MappingVerdict.MAPPED, (
                f"one control ({candidates[0]}) corresponds to {semantic}")
        elif len(candidates) > 1:
            verdict, reason = MappingVerdict.AMBIGUOUS, (
                f"{len(candidates)} controls could be {semantic}: "
                f"{', '.join(candidates)}. A person must say which; choosing by sort "
                f"order would decide the chemistry silently")
        elif not schema.discovered:
            verdict, reason = MappingVerdict.UNAVAILABLE, (
                "no schema has been discovered from the live page, so nothing can be "
                "mapped yet -- this is not evidence that the form lacks the field")
        else:
            verdict, reason = MappingVerdict.UNAVAILABLE, (
                f"no discovered control corresponds to {semantic}. Either the form "
                f"cannot express it, or the capture did not reach the control that "
                f"does; a page cannot distinguish those")

        mapping.fields.append(FieldMapping(
            spec_field=spec_field, semantic=semantic, verdict=verdict,
            requested_value=value, control_keys=candidates, reason=reason,
            required_by_request=is_requested,
        ))

    described = {key for keys in semantics.values() for key in keys}
    mapping.unmapped_controls = sorted(
        key for key, spec_field in schema.fields.items()
        if key not in described and spec_field.control != "button"
    )
    logger.info("spec mapping for %s: %s", spec.name, mapping.counts())
    return mapping


def default_probe_spec(monomer: str, *, system_type: SystemType = SystemType.SINGLE_CHAIN,
                       degree_of_polymerization: int = 10) -> PolymerBuilderSpec:
    """The smallest sensible specification, for probing a form (§8).

    Small on purpose: a first live build should cost the shared service as little as
    possible, and a ten-mer single chain exercises the whole route without asking for
    anything expensive.
    """
    return PolymerBuilderSpec(
        polymer_id="probe", name=monomer, repeat_unit_smiles="",
        degree_of_polymerization=degree_of_polymerization, n_chains=1,
        force_field="CHARMM36", temperature_k=300.0, pressure_bar=1.0,
        system_type=system_type,
        notes="smallest sensible configuration for a first live build",
    )


__all__ = [
    "ENGINE_ONLY", "SPEC_TO_SEMANTIC", "FieldMapping", "MappingVerdict", "SpecMapping",
    "default_probe_spec", "map_spec_to_form",
]
