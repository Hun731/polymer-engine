"""Force-field selection as an auditable decision, not an automatic one.

Choosing a force field determines every number a simulation produces.  No code can
make that choice responsibly across all chemistry, so this module does something more
useful and more honest: it enumerates the candidates that are *plausible* for a given
polymer family, records what evidence supports each, and returns a confidence level.

The four confidence levels are the point:

``KNOWN``
    The force field was explicitly parameterised for this chemistry and validated
    against QM or experiment in this project.
``SUPPORTED``
    The chemistry is inside the force field's published coverage, but this specific
    polymer has not been validated here.
``UNCERTAIN``
    The chemistry is near the edge of coverage; QM validation is needed before use.
``REQUIRES_EXPERT_DECISION``
    The engine cannot narrow the choice. It says so and stops.

A strategy is only ``KNOWN`` when real validation evidence has been attached.  The
engine will not promote a guess to a fact by reasoning about it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from polymer_engine.core.errors import ScientificError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination
from polymer_engine.polymer.taxonomy import PolymerFamily

logger = get_logger("simulation.forcefield")


class Confidence(str, Enum):
    KNOWN = "KNOWN"
    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    REQUIRES_EXPERT_DECISION = "REQUIRES_EXPERT_DECISION"

    @property
    def usable_without_review(self) -> bool:
        return self is Confidence.KNOWN


@dataclass(frozen=True, slots=True)
class ForceFieldProfile:
    """What a force field covers, as published by its authors.

    ``coverage`` lists the polymer families the force field's own documentation claims
    to parameterise.  This is a record of published scope, not an endorsement, and it
    is deliberately conservative.
    """

    name: str
    description: str
    coverage: frozenset[PolymerFamily]
    water_models: tuple[str, ...] = ()
    reference: str = ""
    requires_parameter_generation: bool = False
    notes: str = ""

    def covers(self, family: PolymerFamily) -> bool:
        return family in self.coverage

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "coverage": sorted(f.value for f in self.coverage),
            "water_models": list(self.water_models),
            "reference": self.reference,
            "requires_parameter_generation": self.requires_parameter_generation,
            "notes": self.notes,
        }


_ORGANIC = frozenset(
    {
        PolymerFamily.POLYOLEFIN,
        PolymerFamily.POLYSTYRENIC,
        PolymerFamily.POLYESTER,
        PolymerFamily.POLYAMIDE,
        PolymerFamily.POLYETHER,
        PolymerFamily.POLYACRYLATE,
        PolymerFamily.POLYACRYLAMIDE,
        PolymerFamily.POLYVINYL_ALCOHOL,
        PolymerFamily.POLYVINYL_HALIDE,
        PolymerFamily.POLYNITRILE,
        PolymerFamily.POLYCARBONATE,
        PolymerFamily.POLYURETHANE,
        PolymerFamily.POLYUREA,
    }
)


def default_profiles() -> list[ForceFieldProfile]:
    """Force fields the engine knows the published scope of.

    Coverage is transcribed from each force field's own documentation.  Absence from
    this list means the engine has no scope information, not that a force field is
    unsuitable.
    """
    return [
        ForceFieldProfile(
            name="CHARMM36",
            description="CHARMM General Force Field / CHARMM36 for organic polymers and biomolecules",
            coverage=_ORGANIC,
            water_models=("TIP3P", "TIP3P-CHARMM", "TIP4P"),
            reference="Vanommeslaeghe et al., J. Comput. Chem. 2010 (CGenFF)",
            notes="CGenFF assigns parameters by analogy and reports a penalty score; "
                  "high-penalty assignments need QM validation before use.",
        ),
        ForceFieldProfile(
            name="OPLS-AA",
            description="OPLS all-atom force field for organic liquids and polymers",
            coverage=_ORGANIC,
            water_models=("TIP3P", "TIP4P", "SPC"),
            reference="Jorgensen et al., J. Am. Chem. Soc. 1996",
        ),
        ForceFieldProfile(
            name="GAFF2",
            description="General AMBER Force Field 2 for small organic molecules and polymers",
            coverage=_ORGANIC,
            water_models=("TIP3P", "OPC", "SPC/E"),
            reference="Wang et al., J. Comput. Chem. 2004",
            requires_parameter_generation=True,
            notes="Requires antechamber/parmchk parameter generation and partial charges "
                  "(AM1-BCC or RESP) derived per molecule.",
        ),
        ForceFieldProfile(
            name="OpenFF-2.x",
            description="Open Force Field small-molecule force field (SMIRNOFF)",
            coverage=_ORGANIC,
            water_models=("TIP3P", "OPC"),
            reference="Boothroyd et al., J. Chem. Theory Comput. 2023",
            requires_parameter_generation=True,
            notes="Direct SMILES-based typing; coverage of unusual valences is limited.",
        ),
        ForceFieldProfile(
            name="PCFF",
            description="Polymer Consistent Force Field",
            coverage=frozenset(
                {
                    PolymerFamily.POLYOLEFIN,
                    PolymerFamily.POLYSTYRENIC,
                    PolymerFamily.POLYESTER,
                    PolymerFamily.POLYAMIDE,
                    PolymerFamily.POLYETHER,
                    PolymerFamily.POLYACRYLATE,
                    PolymerFamily.POLYSILOXANE,
                }
            ),
            water_models=("SPC",),
            reference="Sun et al., J. Am. Chem. Soc. 1994",
            notes="Class-II functional form; requires a compatible engine build.",
        ),
    ]


@dataclass
class ForceFieldEvidence:
    """One piece of support for or against using a force field here."""

    kind: str
    description: str
    supports: bool
    source: str = ""
    metric: str | None = None
    value: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "supports": self.supports,
            "source": self.source,
            "metric": self.metric,
            "value": self.value,
        }


@dataclass
class ForceFieldStrategy:
    """The reasoning behind a force-field choice, kept with the campaign."""

    candidate_polymer_id: str
    family: PolymerFamily
    compatible_force_fields: list[ForceFieldProfile] = field(default_factory=list)
    evidence: list[ForceFieldEvidence] = field(default_factory=list)
    qm_validation: dict[str, Any] | None = None
    confidence: Confidence = Confidence.REQUIRES_EXPERT_DECISION
    selected_force_field: str | None = None
    justification: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def determination(self) -> Determination:
        if self.confidence is Confidence.KNOWN:
            return Determination.KNOWN
        if self.confidence is Confidence.REQUIRES_EXPERT_DECISION:
            return Determination.REQUIRES_VALIDATION
        return Determination.REQUIRES_VALIDATION

    @property
    def ready_to_simulate(self) -> bool:
        """Whether a campaign may proceed without a human confirming the choice."""
        return self.selected_force_field is not None and self.confidence is Confidence.KNOWN

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_polymer_id": self.candidate_polymer_id,
            "family": self.family.value,
            "compatible_force_fields": [p.name for p in self.compatible_force_fields],
            "compatible_detail": [p.as_dict() for p in self.compatible_force_fields],
            "evidence": [e.as_dict() for e in self.evidence],
            "qm_validation": self.qm_validation,
            "confidence": self.confidence.value,
            "determination": self.determination.value,
            "selected_force_field": self.selected_force_field,
            "justification": self.justification,
            "warnings": self.warnings,
            "ready_to_simulate": self.ready_to_simulate,
        }


class ForceFieldAdvisor:
    """Enumerates plausible force fields and records why one was chosen."""

    def __init__(self, profiles: Sequence[ForceFieldProfile] | None = None) -> None:
        self.profiles = list(profiles if profiles is not None else default_profiles())

    def profile(self, name: str) -> ForceFieldProfile | None:
        lowered = name.strip().lower()
        return next((p for p in self.profiles if p.name.lower() == lowered), None)

    def compatible(self, family: PolymerFamily) -> list[ForceFieldProfile]:
        return [p for p in self.profiles if p.covers(family)]

    def propose(
        self,
        polymer_id: str,
        family: PolymerFamily,
        *,
        requested: str | None = None,
        qm_validation: dict[str, Any] | None = None,
        evidence: Sequence[ForceFieldEvidence] = (),
    ) -> ForceFieldStrategy:
        """Assemble the decision record for one polymer.

        A requested force field is honoured; the engine's job is to say how much
        confidence that choice deserves, not to override it.
        """
        compatible = self.compatible(family)
        strategy = ForceFieldStrategy(
            candidate_polymer_id=polymer_id,
            family=family,
            compatible_force_fields=compatible,
            evidence=list(evidence),
            qm_validation=qm_validation,
        )

        if family is PolymerFamily.UNCLASSIFIED:
            strategy.confidence = Confidence.REQUIRES_EXPERT_DECISION
            strategy.justification = (
                "The polymer family could not be classified, so no coverage claim can be "
                "checked. A human must choose and justify the force field."
            )
            strategy.warnings.append("polymer family is unclassified")
            if requested:
                strategy.selected_force_field = requested
                strategy.warnings.append(
                    f"{requested} was requested but its coverage of this chemistry is unverified"
                )
            return strategy

        if requested:
            profile = self.profile(requested)
            strategy.selected_force_field = requested
            if profile is None:
                strategy.confidence = Confidence.UNCERTAIN
                strategy.justification = (
                    f"{requested} was requested but the engine holds no coverage information "
                    "for it. The choice is recorded and used as given; its suitability is "
                    "the operator's judgement."
                )
                strategy.warnings.append(f"no coverage profile for {requested}")
            elif not profile.covers(family):
                strategy.confidence = Confidence.UNCERTAIN
                strategy.justification = (
                    f"{profile.name} does not list {family.value} in its published coverage. "
                    "QM validation is needed before these parameters can be trusted here."
                )
                strategy.warnings.append(
                    f"{profile.name} does not claim coverage of {family.value}"
                )
            else:
                strategy.confidence = Confidence.SUPPORTED
                strategy.justification = (
                    f"{profile.name} lists {family.value} within its published coverage "
                    f"({profile.reference}). Coverage is not the same as validation for this "
                    "specific polymer."
                )
                if profile.requires_parameter_generation:
                    strategy.warnings.append(
                        f"{profile.name} requires per-molecule parameter generation; "
                        "the engine does not generate force-field parameters"
                    )
        elif not compatible:
            strategy.confidence = Confidence.REQUIRES_EXPERT_DECISION
            strategy.justification = (
                f"No force field known to the engine claims coverage of {family.value}. "
                "A human must select and justify one."
            )
            return strategy
        elif len(compatible) == 1:
            strategy.selected_force_field = compatible[0].name
            strategy.confidence = Confidence.SUPPORTED
            strategy.justification = (
                f"{compatible[0].name} is the only force field known to the engine that "
                f"claims coverage of {family.value}."
            )
        else:
            strategy.confidence = Confidence.REQUIRES_EXPERT_DECISION
            strategy.justification = (
                f"{len(compatible)} force fields claim coverage of {family.value} "
                f"({', '.join(p.name for p in compatible)}). They are not interchangeable and "
                "the engine will not pick between them."
            )
            return strategy

        strategy.confidence = self._apply_evidence(strategy)
        return strategy

    @staticmethod
    def _apply_evidence(strategy: ForceFieldStrategy) -> Confidence:
        """Promote to KNOWN only on real, passing validation evidence."""
        refuting = [e for e in strategy.evidence if not e.supports]
        if refuting:
            strategy.warnings.extend(f"contrary evidence: {e.description}" for e in refuting)
            return Confidence.UNCERTAIN

        validation_passed = bool(strategy.qm_validation and strategy.qm_validation.get("passed"))
        supporting = [e for e in strategy.evidence if e.supports]

        if validation_passed and supporting:
            strategy.justification += (
                " QM validation passed and supporting evidence was supplied, so this "
                "combination is validated for this polymer in this project."
            )
            return Confidence.KNOWN
        if validation_passed:
            strategy.justification += " QM validation passed for this chemistry."
            return Confidence.KNOWN
        if strategy.qm_validation is not None and not strategy.qm_validation.get("passed"):
            strategy.warnings.append("QM validation did not pass")
            return Confidence.UNCERTAIN
        return strategy.confidence


def require_ready_force_field(strategy: ForceFieldStrategy) -> None:
    """Raise unless the strategy may be used without human review.

    Called before a campaign commits compute to a parameter set.
    """
    if not strategy.ready_to_simulate:
        raise ScientificError(
            "Force-field choice is not validated for this polymer",
            polymer_id=strategy.candidate_polymer_id,
            family=strategy.family.value,
            confidence=strategy.confidence.value,
            selected=strategy.selected_force_field,
            justification=strategy.justification,
            warnings=strategy.warnings,
        )


__all__ = [
    "Confidence",
    "ForceFieldAdvisor",
    "ForceFieldEvidence",
    "ForceFieldProfile",
    "ForceFieldStrategy",
    "default_profiles",
    "require_ready_force_field",
]
