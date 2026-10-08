"""Preparing a CHARMM-GUI Polymer Builder job, and qualifying what comes back.

CHARMM-GUI has no job-submission API.  It publishes login, status and download
endpoints and nothing else, so the build itself is done by a person in the web
interface.  That step cannot be automated without reverse-engineering an undocumented
form, which this engine does not do.

What *can* be automated is everything on either side of it:

* **Before** — derive the exact chemistry and box the job needs from the polymer record,
  record it, and hand the operator an unambiguous specification instead of leaving them
  to re-derive a degree of polymerisation and a box edge by hand.
* **After** — download the finished job through the documented API (or take a local
  archive), extract it safely, validate the system, read the CGenFF penalties, and
  decide whether the parameters may enter a campaign.

The specification is deliberately a *statement of the required inputs*, not a
click-by-click walkthrough. Field labels in the web interface change between releases,
and inventing them here would be the same class of mistake as inventing an endpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from polymer_engine.core.errors import ChemistryError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination
from polymer_engine.core.provenance import canonical_hash

logger = get_logger("simulation.charmm_gui_spec")

#: Documented CHARMM-GUI endpoints.  Submission is absent because it does not exist.
DOCUMENTED_ENDPOINTS = ("/api/login", "/api/check_status", "/api/download")

#: The Polymer Builder module, for the operator's reference.
POLYMER_BUILDER_URL = "https://www.charmm-gui.org/?doc=input/polymer"

#: Avogadro's number, for sizing a box from a target density.
AVOGADRO = 6.02214076e23


class SystemType(str, Enum):
    """Which kind of system Polymer Builder is being asked for.

    These are the *modes the module documents*; whether a given mode is offered for a
    given chemistry is discovered from the live page, never assumed from this list.
    """

    SINGLE_CHAIN = "single_chain"
    MELT = "melt"
    SOLUTION = "solution"

    @property
    def needs_solvent(self) -> bool:
        return self is SystemType.SOLUTION

    @property
    def n_chains_meaningful(self) -> bool:
        """A single chain is one chain; asking for twenty of it is a contradiction."""
        return self is not SystemType.SINGLE_CHAIN


@dataclass
class Comonomer:
    """One component of a copolymer, and how much of it there is."""

    monomer: str
    #: Mole fraction in [0, 1].  Fractions across a spec must sum to 1.
    fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {"monomer": self.monomer, "fraction": self.fraction}


@dataclass
class PolymerBuilderSpec:
    """Everything a person needs to reproduce one Polymer Builder job.

    Every field is either taken from the polymer record or derived from an explicitly
    supplied campaign setting.  Nothing here is a default the engine invented.
    """

    polymer_id: str
    name: str
    repeat_unit_smiles: str
    degree_of_polymerization: int
    n_chains: int
    force_field: str
    temperature_k: float
    pressure_bar: float
    target_density_kg_m3: float | None = None
    box_nm: float | None = None
    tacticity: str | None = None
    terminal_groups: str = "hydrogen"
    water_model: str | None = None
    solvate: bool = False
    #: Which Polymer Builder mode this is.  Defaults to a melt, which is what the
    #: density campaign needs; a single chain and a solution are different systems.
    system_type: SystemType = SystemType.MELT
    #: Additional monomers for a copolymer.  Empty for a homopolymer.
    comonomers: list[Comonomer] = field(default_factory=list)
    #: How the monomers are arranged: "random", "block", "alternating", or an explicit
    #: sequence string.  Meaningless -- and left None -- for a homopolymer.
    sequence: str | None = None
    solvent: str | None = None
    notes: str = ""
    #: Filled in once a person has actually run the job.
    charmm_gui_job_id: str | None = None

    def fingerprint(self) -> str:
        """Identifies the *chemistry and composition*, not the bookkeeping.

        The job id is excluded on purpose: two people building the same specification
        must produce the same fingerprint, which is what makes the manual step auditable.
        """
        return canonical_hash({
            "polymer_id": self.polymer_id,
            "repeat_unit_smiles": self.repeat_unit_smiles,
            "degree_of_polymerization": self.degree_of_polymerization,
            "n_chains": self.n_chains,
            "force_field": self.force_field,
            "temperature_k": self.temperature_k,
            "pressure_bar": self.pressure_bar,
            "tacticity": self.tacticity,
            "terminal_groups": self.terminal_groups,
            "water_model": self.water_model,
            "solvate": self.solvate,
            "system_type": self.system_type.value,
            "comonomers": [c.as_dict() for c in self.comonomers],
            "sequence": self.sequence,
            "solvent": self.solvent,
        })

    def composition_problems(self) -> list[str]:
        """Contradictions between the requested fields, before anything is submitted.

        Checked here rather than at the form, so an impossible request is refused
        locally instead of producing a job that builds something else.
        """
        issues: list[str] = []
        if self.degree_of_polymerization < 2:
            issues.append("degree of polymerisation must be at least 2")
        if self.n_chains < 1:
            issues.append("a system needs at least one chain")
        if self.system_type is SystemType.SINGLE_CHAIN and self.n_chains != 1:
            issues.append(
                f"a single-chain system has exactly one chain, but {self.n_chains} "
                f"were requested"
            )
        if self.system_type is SystemType.SOLUTION and not (self.solvent or self.water_model):
            issues.append("a solution system needs a solvent")
        if self.comonomers:
            fractions = [c.fraction for c in self.comonomers]
            if any(f < 0 or f > 1 for f in fractions):
                issues.append("every comonomer fraction must lie in [0, 1]")
            total = sum(fractions)
            if abs(total - 1.0) > 1e-6:
                issues.append(
                    f"comonomer fractions sum to {total:.6g}, not 1; the composition "
                    f"is not normalised and will not be normalised silently"
                )
            names = [c.monomer for c in self.comonomers]
            if len(set(names)) != len(names):
                issues.append("the same monomer appears twice in the composition")
            if self.sequence is None:
                issues.append(
                    "a copolymer needs an explicit sequence (random, block, "
                    "alternating, or a literal pattern); there is no safe default"
                )
        elif self.sequence is not None:
            issues.append("a sequence was given but there is only one monomer")
        return issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id, "name": self.name,
            "repeat_unit_smiles": self.repeat_unit_smiles,
            "degree_of_polymerization": self.degree_of_polymerization,
            "n_chains": self.n_chains, "force_field": self.force_field,
            "temperature_k": self.temperature_k, "pressure_bar": self.pressure_bar,
            "target_density_kg_m3": self.target_density_kg_m3, "box_nm": self.box_nm,
            "tacticity": self.tacticity, "terminal_groups": self.terminal_groups,
            "water_model": self.water_model, "solvate": self.solvate,
            "system_type": self.system_type.value,
            "comonomers": [c.as_dict() for c in self.comonomers],
            "sequence": self.sequence, "solvent": self.solvent,
            "composition_problems": self.composition_problems(),
            "notes": self.notes, "charmm_gui_job_id": self.charmm_gui_job_id,
            "fingerprint": self.fingerprint(),
            "submission": "manual — CHARMM-GUI publishes no job-submission endpoint",
            "documented_endpoints": list(DOCUMENTED_ENDPOINTS),
            "module_url": POLYMER_BUILDER_URL,
        }

    def instructions(self) -> str:
        """A human-readable brief for the person who will run the job."""
        lines = [
            f"# CHARMM-GUI Polymer Builder — {self.name}", "",
            f"Polymer id `{self.polymer_id}` · specification fingerprint "
            f"`{self.fingerprint()[:16]}`", "",
            "**The engine cannot submit this job.** CHARMM-GUI publishes login, status",
            "and download endpoints only; there is no documented way to create a job",
            "programmatically, so this step is done by a person at",
            f"<{POLYMER_BUILDER_URL}>.", "",
            "## Chemistry and composition", "",
            "| Input | Value |", "|---|---|",
            f"| Repeat unit (SMILES) | `{self.repeat_unit_smiles}` |",
            f"| Degree of polymerisation | {self.degree_of_polymerization} |",
            f"| Chains in the box | {self.n_chains} |",
            f"| Terminal groups | {self.terminal_groups} |",
        ]
        if self.tacticity:
            lines.append(f"| Tacticity | {self.tacticity} |")
        lines += [
            f"| Force field | {self.force_field} |",
            f"| Temperature | {self.temperature_k:g} K |",
            f"| Pressure | {self.pressure_bar:g} bar |",
        ]
        if self.box_nm:
            lines.append(f"| Cubic box edge | {self.box_nm:.4g} nm |")
        if self.target_density_kg_m3:
            lines.append(f"| Target density | {self.target_density_kg_m3:g} kg/m³ |")
        if self.solvate:
            lines.append(f"| Solvate | yes, {self.water_model or 'unspecified model'} |")
        else:
            lines.append("| Solvate | no — bulk melt |")

        lines += [
            "", "Field labels are not reproduced here on purpose: they change between",
            "CHARMM-GUI releases, and transcribing them would go stale silently. The",
            "table above is the information the module asks for, in its own terms.", "",
            "## When the job finishes", "",
            "Note the job id and hand it back to the engine:", "",
            "```bash",
            f"polymer-engine charmm-gui import <JOB_ID> --polymer {self.polymer_id} \\",
            f"    --force-field '{self.force_field}'",
            "```", "",
            "or, if you downloaded the archive yourself:", "",
            "```bash",
            f"polymer-engine charmm-gui import --archive <PATH.tgz> --polymer {self.polymer_id} \\",
            f"    --force-field '{self.force_field}'",
            "```", "",
            "## What the engine will check on return", "",
            "- the archive extracts safely (no traversal, links or decompression bombs);",
            "- coordinates and topology are present, finite and mutually consistent;",
            "- the atom count implied by the topology matches the coordinates;",
            "- **CGenFF penalty scores are read and reported, never suppressed.**", "",
            "A parameter set whose worst penalty is above the CGenFF \"good\" tier does",
            "not enter a campaign on its own: it comes back `REQUIRES_VALIDATION`, and",
            "either an explicit tolerance or a QM check has to justify it.", "",
        ]
        if self.notes:
            lines += ["## Notes", "", self.notes, ""]
        return "\n".join(lines)


def box_edge_for(
    chain_mass_amu: float, n_chains: int, density_kg_m3: float
) -> float:
    """Cubic box edge in nm that puts ``n_chains`` at the requested density."""
    if density_kg_m3 <= 0:
        raise ChemistryError("Target density must be positive", density=density_kg_m3)
    if n_chains < 1:
        raise ChemistryError("A box needs at least one chain", n_chains=n_chains)
    mass_kg = chain_mass_amu * n_chains / AVOGADRO * 1e-3
    return float((mass_kg / density_kg_m3 * 1e27) ** (1.0 / 3.0))


def _known_tacticity(record: Any) -> str | None:
    value = getattr(record, "tacticity", None)
    text = getattr(value, "value", value)
    if not text or str(text).lower() in {"unknown", "none", "unspecified"}:
        return None
    return str(text)


def spec_from_record(
    record: Any,
    *,
    force_field: str,
    degree_of_polymerization: int,
    n_chains: int,
    temperature_k: float,
    pressure_bar: float = 1.0,
    target_density_kg_m3: float | None = None,
    tacticity: str | None = None,
    water_model: str | None = None,
    solvate: bool = False,
    notes: str = "",
) -> PolymerBuilderSpec:
    """Derive a build specification from a :class:`PolymerRecord`.

    Every simulation setting is a required argument. None of them has a default here,
    because a force field, a chain length and a temperature each change the answer, and
    a specification that silently supplied them would make the manual step look more
    reproducible than it is.
    """
    if not force_field.strip():
        raise ChemistryError("A force field must be named", polymer_id=record.polymer_id)
    if degree_of_polymerization < 2:
        raise ChemistryError(
            "Degree of polymerisation must be at least 2",
            value=degree_of_polymerization,
        )
    if n_chains < 1:
        raise ChemistryError("At least one chain is required", value=n_chains)

    box = None
    if target_density_kg_m3:
        mass = record.descriptors.get("repeat_unit_mass") if record.descriptors else None
        if mass is not None and mass.determination is Determination.KNOWN and mass.value:
            box = box_edge_for(
                float(mass.value) * degree_of_polymerization, n_chains, target_density_kg_m3
            )

    spec = PolymerBuilderSpec(
        polymer_id=record.polymer_id,
        name=record.name,
        repeat_unit_smiles=record.canonical_repeat_unit or record.repeat_unit_smiles,
        degree_of_polymerization=degree_of_polymerization,
        n_chains=n_chains,
        force_field=force_field,
        temperature_k=temperature_k,
        pressure_bar=pressure_bar,
        target_density_kg_m3=target_density_kg_m3,
        box_nm=box,
        # An "unknown" tacticity is not information; leaving it out of the brief keeps
        # the operator from thinking the engine determined something it did not.
        tacticity=tacticity or _known_tacticity(record),
        water_model=water_model,
        solvate=solvate,
        notes=notes,
    )
    logger.info("Prepared Polymer Builder spec for %s (%s)", record.name, spec.fingerprint()[:16])
    return spec


def write_spec(spec: PolymerBuilderSpec, directory: str | Path) -> tuple[Path, Path]:
    """Write the machine-readable spec and the human brief side by side."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    slug = spec.name.replace("(", "").replace(")", "").replace(" ", "_").replace(",", "")
    payload = directory / f"{slug}_charmm_gui_spec.json"
    brief = directory / f"{slug}_charmm_gui_spec.md"
    payload.write_text(json.dumps(spec.as_dict(), indent=1), encoding="utf-8")
    brief.write_text(spec.instructions(), encoding="utf-8")
    return payload, brief


__all__ = [
    "AVOGADRO",
    "DOCUMENTED_ENDPOINTS",
    "POLYMER_BUILDER_URL",
    "Comonomer",
    "PolymerBuilderSpec",
    "SystemType",
    "box_edge_for",
    "spec_from_record",
    "write_spec",
]
