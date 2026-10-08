"""The parameter-quality gate, and deciding which terms deserve QM scrutiny.

Two ideas run through this module.

**A penalty is evidence, not a verdict.**  A high CGenFF penalty says the analogy was
weak.  It does not say the parameter is wrong, and it does not say it is right.  The
correct response is usually neither PASS nor FAIL but "validate this against QM before
using it", which is why :class:`ParameterQualityGate` can return
``REQUIRES_EXPERT_REVIEW`` and ``INCONCLUSIVE`` as first-class outcomes.

**Quality is relative to the question.**  A torsional barrier 5 kJ/mol off barely moves
a bulk density and ruins a conformer population, because populations depend
exponentially on relative energies.  So the gate is configured per
:class:`~polymer_engine.parameterization.models.PropertyClass`, and there is no
universal threshold anywhere in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination, GateReport, GateResult, GateStatus
from polymer_engine.parameterization.models import PropertyClass, QMPriority
from polymer_engine.simulation.cgenff import HIGH_PENALTY, MODERATE_PENALTY, PenaltyReport

logger = get_logger("parameterization.quality")


@dataclass(frozen=True)
class QualityStandard:
    """What one property class demands of a parameter set.

    Every field is a **convention** chosen for a stated reason, recorded in
    ``justification`` so a manifest names the standard it adopted rather than inheriting
    an invisible one.  None of these is a derivation.
    """

    property_class: PropertyClass
    justification: str
    #: Worst tolerable analogy penalty before QM validation becomes mandatory.
    max_penalty_without_qm: float
    #: Whether a torsional QM comparison is required at all.
    requires_torsion_qm: bool
    #: Tolerance on the MM-vs-QM torsion RMSE, when such a comparison is made.
    torsion_rmse_kj_mol: float | None
    #: Whether wildcard-matched bonded parameters are acceptable.
    allow_wildcards: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "property_class": self.property_class.value,
            "justification": self.justification,
            "max_penalty_without_qm": self.max_penalty_without_qm,
            "requires_torsion_qm": self.requires_torsion_qm,
            "torsion_rmse_kj_mol": self.torsion_rmse_kj_mol,
            "allow_wildcards": self.allow_wildcards,
        }


#: Standards per property class.  The ordering of strictness is the physics: observables
#: that depend exponentially on relative energies tolerate less torsional error than
#: observables that average over them.
STANDARDS: dict[PropertyClass, QualityStandard] = {
    PropertyClass.BULK_DENSITY: QualityStandard(
        property_class=PropertyClass.BULK_DENSITY,
        justification=(
            "Density is a packing average over many conformers, so a moderate torsional "
            "error largely averages out. Non-bonded parameters and molecular volume "
            "dominate; torsions matter mainly through chain stiffness."
        ),
        max_penalty_without_qm=MODERATE_PENALTY,
        requires_torsion_qm=False,
        torsion_rmse_kj_mol=None,
    ),
    PropertyClass.THERMODYNAMIC: QualityStandard(
        property_class=PropertyClass.THERMODYNAMIC,
        justification=(
            "Energies and their derivatives are sensitive to both non-bonded and "
            "torsional terms, so an analogy-assigned torsion needs checking."
        ),
        max_penalty_without_qm=MODERATE_PENALTY,
        requires_torsion_qm=True,
        torsion_rmse_kj_mol=2.0,
    ),
    PropertyClass.CONFORMATIONAL_FREE_ENERGY: QualityStandard(
        property_class=PropertyClass.CONFORMATIONAL_FREE_ENERGY,
        justification=(
            "Conformer populations depend exponentially on relative energies: at 300 K "
            "a 4 kJ/mol torsional error is about a factor of five in a population "
            "ratio, so torsional agreement must be tight and analogy is not enough."
        ),
        max_penalty_without_qm=0.0,
        requires_torsion_qm=True,
        torsion_rmse_kj_mol=1.0,
        allow_wildcards=False,
    ),
    PropertyClass.INTERFACIAL_FREE_ENERGY: QualityStandard(
        property_class=PropertyClass.INTERFACIAL_FREE_ENERGY,
        justification=(
            "Interfacial free energies are differences of large numbers and are "
            "dominated by non-bonded parameters and charges; a charge assigned by weak "
            "analogy propagates directly into the answer."
        ),
        max_penalty_without_qm=0.0,
        requires_torsion_qm=True,
        torsion_rmse_kj_mol=1.5,
        allow_wildcards=False,
    ),
    PropertyClass.TRANSPORT: QualityStandard(
        property_class=PropertyClass.TRANSPORT,
        justification=(
            "Diffusion depends on barriers to local rearrangement, so torsional "
            "barriers matter, though the observable is far noisier than a free energy."
        ),
        max_penalty_without_qm=MODERATE_PENALTY,
        requires_torsion_qm=True,
        torsion_rmse_kj_mol=2.0,
    ),
    PropertyClass.MECHANICAL: QualityStandard(
        property_class=PropertyClass.MECHANICAL,
        justification=(
            "Moduli depend on chain stiffness and packing; torsional stiffness enters "
            "directly, but MD strain rates dominate the systematic error anyway."
        ),
        max_penalty_without_qm=MODERATE_PENALTY,
        requires_torsion_qm=True,
        torsion_rmse_kj_mol=2.0,
    ),
    PropertyClass.STRUCTURAL: QualityStandard(
        property_class=PropertyClass.STRUCTURAL,
        justification=(
            "Radii of gyration and persistence lengths are set by the torsional "
            "distribution, so a weak torsional analogy shows up directly in the answer."
        ),
        max_penalty_without_qm=MODERATE_PENALTY,
        requires_torsion_qm=True,
        torsion_rmse_kj_mol=2.0,
    ),
}


def standard_for(property_class: PropertyClass) -> QualityStandard:
    return STANDARDS[property_class]


@dataclass
class SensitiveTerm:
    """A parameter worth spending QM time on, and why."""

    kind: str
    identifier: str
    reason: str
    penalty: float | None = None
    priority: QMPriority = QMPriority.MEDIUM

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "identifier": self.identifier, "reason": self.reason,
            "penalty": self.penalty, "priority": self.priority.value,
        }


def detect_sensitive_terms(
    penalties: PenaltyReport | None = None,
    *,
    functional_groups: list[str] | None = None,
    heteroatoms: list[str] | None = None,
) -> list[SensitiveTerm]:
    """Which parameters deserve QM scrutiny, and how urgently.

    Deliberately cheap and structural: this runs before any QM is scheduled, so it must
    not itself require a calculation. It errs toward flagging, because the cost of a
    needless torsion scan is an hour and the cost of a missed one is a wrong answer.
    """
    terms: list[SensitiveTerm] = []

    if penalties is not None:
        for parameter in penalties.parameters:
            if parameter.penalty <= MODERATE_PENALTY:
                continue
            # A weak torsional analogy is the single most consequential case, because
            # torsions set conformer populations.
            torsional = parameter.section in {"DIHEDRALS", "IMPROPERS", "IMPROPER"}
            priority = (
                QMPriority.HIGH
                if parameter.penalty >= HIGH_PENALTY or (torsional and parameter.penalty > 25)
                else QMPriority.MEDIUM
            )
            terms.append(SensitiveTerm(
                kind=parameter.section.lower(), identifier=parameter.atoms,
                penalty=parameter.penalty, priority=priority,
                reason=(f"analogy penalty {parameter.penalty:g} "
                        f"({parameter.tier}); assigned from {parameter.source[:60]}"),
            ))
        for atom, penalty in penalties.atom_charge_penalties.items():
            if penalty > MODERATE_PENALTY:
                terms.append(SensitiveTerm(
                    kind="charge", identifier=atom, penalty=penalty,
                    priority=QMPriority.HIGH if penalty >= HIGH_PENALTY else QMPriority.MEDIUM,
                    reason=(f"charge assigned by weak analogy (penalty {penalty:g}); "
                            f"charges propagate into every electrostatic observable"),
                ))

    # Chemistry that is known to need care even when no penalty was reported, because a
    # backend without a penalty scheme reports nothing at all.
    for group in functional_groups or []:
        if group in {"ester", "amide", "carbonyl"}:
            terms.append(SensitiveTerm(
                kind="linkage", identifier=group, priority=QMPriority.HIGH,
                reason=(f"{group} backbone linkage: the torsion about it sets chain "
                        f"conformation and is frequently the weakest analogy"),
            ))
        elif group in {"nitrile", "hydroxyl", "ether"}:
            terms.append(SensitiveTerm(
                kind="functional_group", identifier=group, priority=QMPriority.MEDIUM,
                reason=f"{group} is polar; its charges dominate local electrostatics",
            ))
    for element in heteroatoms or []:
        if element in {"F", "Cl", "Br", "I", "S", "P", "Si"}:
            terms.append(SensitiveTerm(
                kind="heteroatom", identifier=element, priority=QMPriority.MEDIUM,
                reason=(f"{element} is sparsely represented in general force fields; "
                        f"check the assignment rather than assuming coverage"),
            ))
    return terms


def overall_priority(terms: list[SensitiveTerm]) -> QMPriority:
    if any(t.priority is QMPriority.HIGH for t in terms):
        return QMPriority.HIGH
    if any(t.priority is QMPriority.MEDIUM for t in terms):
        return QMPriority.MEDIUM
    return QMPriority.LOW


@dataclass
class ParameterQualityGate:
    """Judge a parameter set against the standard for its intended use."""

    property_class: PropertyClass
    standard: QualityStandard = field(init=False)

    def __post_init__(self) -> None:
        self.standard = standard_for(self.property_class)

    def evaluate(
        self,
        *,
        force_field: str,
        force_field_version: str | None = None,
        penalties: PenaltyReport | None = None,
        qm_validated: bool | None = None,
        qm_torsion_rmse: float | None = None,
        wildcards: int = 0,
    ) -> tuple[GateReport, Determination]:
        """Return the gate report and what may be claimed about the parameters."""
        standard = self.standard
        gates = GateReport(name=f"parameter_quality:{self.property_class.value}")

        gates.gates.append(GateResult(
            gate="quality:standard", status=GateStatus.PASS,
            message=(f"judged for {self.property_class.value}: "
                     f"{standard.justification[:110]}"),
            evidence=standard.as_dict(),
        ))

        gates.gates.append(GateResult(
            gate="quality:force_field_identified",
            status=GateStatus.PASS if force_field_version else GateStatus.WARN,
            message=(f"{force_field} version {force_field_version}"
                     if force_field_version
                     else f"{force_field} version is not recorded; provenance is incomplete"),
            evidence={"force_field": force_field, "version": force_field_version},
        ))

        worst = penalties.max_penalty if penalties else None
        needs_qm = standard.requires_torsion_qm
        if worst is not None:
            over = worst > standard.max_penalty_without_qm
            needs_qm = needs_qm or over
            # A high penalty means "check this", never "this is wrong" -- so it is
            # INCONCLUSIVE while unchecked, and PASS once QM has actually checked it.
            # Leaving it INCONCLUSIVE after a successful validation would make the QM
            # run pointless: nothing could ever clear a weak analogy.
            if not over:
                status = GateStatus.PASS
                message = (f"worst analogy penalty {worst:g} is within the "
                           f"{standard.max_penalty_without_qm:g} ceiling for "
                           f"{self.property_class.value}")
            elif qm_validated:
                status = GateStatus.PASS
                message = (f"worst analogy penalty {worst:g} exceeds the "
                           f"{standard.max_penalty_without_qm:g} ceiling, and QM "
                           f"validation confirmed the parameters anyway")
            elif qm_validated is False:
                status = GateStatus.FAIL
                message = (f"worst analogy penalty {worst:g} exceeds the ceiling and QM "
                           f"validation did not confirm the parameters")
            else:
                status = GateStatus.INCONCLUSIVE
                message = (f"worst analogy penalty {worst:g} against a "
                           f"{standard.max_penalty_without_qm:g} ceiling for "
                           f"{self.property_class.value}; QM validation is required "
                           f"before these parameters may be used")
            gates.gates.append(GateResult(
                gate="quality:analogy_penalty", status=status, message=message,
                value=worst, threshold=standard.max_penalty_without_qm,
                evidence={"tier": penalties.tier if penalties else None,
                          "qm_validated": qm_validated},
            ))

        if wildcards and not standard.allow_wildcards:
            gates.gates.append(GateResult(
                gate="quality:wildcards", status=GateStatus.FAIL,
                message=(f"{wildcards} wildcard-matched parameter(s); "
                         f"{self.property_class.value} does not accept them"),
                value=float(wildcards),
            ))

        if needs_qm:
            if qm_validated is None:
                gates.gates.append(GateResult(
                    gate="quality:qm_validation", status=GateStatus.INCONCLUSIVE,
                    message=("QM validation is required for this property class and "
                             "has not been run"),
                ))
            elif not qm_validated:
                gates.gates.append(GateResult(
                    gate="quality:qm_validation", status=GateStatus.FAIL,
                    message="QM validation ran and the parameters did not agree",
                    value=qm_torsion_rmse, threshold=standard.torsion_rmse_kj_mol,
                ))
            else:
                gates.gates.append(GateResult(
                    gate="quality:qm_validation", status=GateStatus.PASS,
                    message=("QM validation passed"
                             + (f" (torsion RMSE {qm_torsion_rmse:.3f} kJ/mol)"
                                if qm_torsion_rmse is not None else "")),
                    value=qm_torsion_rmse, threshold=standard.torsion_rmse_kj_mol,
                    units="kJ/mol",
                ))
        else:
            gates.gates.append(GateResult(
                gate="quality:qm_validation", status=GateStatus.PASS,
                message=(f"{self.property_class.value} does not require torsional QM "
                         f"validation at this penalty level"),
            ))

        if gates.promotable:
            determination = Determination.KNOWN
        elif gates.status is GateStatus.FAIL:
            determination = Determination.INSUFFICIENT_DATA
        else:
            determination = Determination.REQUIRES_VALIDATION
        return gates, determination


__all__ = [
    "STANDARDS", "ParameterQualityGate", "QualityStandard", "SensitiveTerm",
    "detect_sensitive_terms", "overall_priority", "standard_for",
]
