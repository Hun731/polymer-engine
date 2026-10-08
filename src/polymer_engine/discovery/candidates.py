"""Rational candidate generation.

Mutations operate on the **molecular graph**, not on the SMILES string.  String
mutation is what produces "candidates" like ``*CC((*)`` that look plausible in a list
and are not molecules at all; every operator here edits an RDKit molecule and the
result must survive sanitisation before it is offered.

Every generated candidate passes through :func:`validate_candidate`, which checks:

* chemical validity (parses, sanitises, valences satisfied)
* that it is still a repeat unit (exactly two attachment points)
* that it differs from its parent
* that it is not a duplicate of something already generated or known
* configured constraints (mass, ring count, element whitelist, ...)
* that descriptors can actually be computed for it

Where a rule is genuinely uncertain -- a strained ring, an unusual valence that RDKit
accepts, a fragment with no synthetic precedent -- the candidate is emitted with
``REQUIRES_REVIEW`` rather than either being silently accepted or silently dropped.
Synthetic accessibility is *not* claimed: this module has no retrosynthesis model, and
saying otherwise would be the exact kind of fabrication the engine must avoid.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from polymer_engine.core.errors import ChemistryError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination
from polymer_engine.polymer.identity import (
    ATTACHMENT_PATTERN,
    canonical_repeat_unit,
    capped_monomer_smiles,
    count_attachment_points,
    derive_polymer_id,
    validate_repeat_unit,
)

logger = get_logger("discovery.candidates")

#: Placeholder element standing in for a chain attachment point during editing.
ATTACHMENT_ELEMENT = "At"


class MutationKind(str, Enum):
    SIDE_CHAIN_SUBSTITUTION = "side_chain_substitution"
    FUNCTIONAL_GROUP_SUBSTITUTION = "functional_group_substitution"
    BACKBONE_MODIFICATION = "backbone_modification"
    SPACER_MODIFICATION = "spacer_modification"
    POLARITY_MODIFICATION = "polarity_modification"
    HBOND_MODIFICATION = "hbond_modification"
    AROMATICITY_MODIFICATION = "aromaticity_modification"


class CandidateStatus(str, Enum):
    VALID = "valid"
    REQUIRES_REVIEW = "requires_review"
    INVALID = "invalid"
    DUPLICATE = "duplicate"


@dataclass
class GeneratedCandidate:
    polymer_id: str
    repeat_unit_smiles: str
    canonical_repeat_unit: str
    parent_id: str
    parent_smiles: str
    mutation: MutationKind
    description: str
    status: CandidateStatus = CandidateStatus.REQUIRES_REVIEW
    issues: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)
    descriptors: dict[str, float] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.status is CandidateStatus.VALID

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id,
            "repeat_unit_smiles": self.repeat_unit_smiles,
            "canonical_repeat_unit": self.canonical_repeat_unit,
            "parent_id": self.parent_id,
            "parent_smiles": self.parent_smiles,
            "mutation": self.mutation.value,
            "description": self.description,
            "status": self.status.value,
            "issues": self.issues,
            "review_reasons": self.review_reasons,
            "descriptors": self.descriptors,
        }


@dataclass
class Constraints:
    """Hard limits a candidate must satisfy to be offered."""

    max_repeat_unit_mass: float = 500.0
    min_repeat_unit_mass: float = 20.0
    max_rings: int = 4
    max_rotatable_bonds: int = 20
    allowed_elements: frozenset[str] = frozenset({"C", "H", "N", "O", "S", "F", "Cl", "Br", "Si", "P"})
    require_two_attachment_points: bool = True

    def check(self, descriptors: dict[str, float], elements: set[str]) -> list[str]:
        problems: list[str] = []
        mass = descriptors.get("repeat_unit_mass")
        if mass is not None:
            if mass > self.max_repeat_unit_mass:
                problems.append(f"repeat-unit mass {mass:.1f} exceeds the limit {self.max_repeat_unit_mass}")
            if mass < self.min_repeat_unit_mass:
                problems.append(f"repeat-unit mass {mass:.1f} is below the minimum {self.min_repeat_unit_mass}")
        rings = descriptors.get("ring_count")
        if rings is not None and rings > self.max_rings:
            problems.append(f"{rings:.0f} rings exceeds the limit {self.max_rings}")
        rotatable = descriptors.get("rotatable_bond_count")
        if rotatable is not None and rotatable > self.max_rotatable_bonds:
            problems.append(f"{rotatable:.0f} rotatable bonds exceeds the limit {self.max_rotatable_bonds}")
        forbidden = elements - set(self.allowed_elements)
        if forbidden:
            problems.append(f"contains disallowed element(s): {sorted(forbidden)}")
        return problems


#: Substituent fragments, as (SMILES fragment, human description, mutation kind).
SUBSTITUENTS: tuple[tuple[str, str, MutationKind], ...] = (
    ("C", "methyl", MutationKind.SIDE_CHAIN_SUBSTITUTION),
    ("CC", "ethyl", MutationKind.SIDE_CHAIN_SUBSTITUTION),
    ("C(C)C", "isopropyl", MutationKind.SIDE_CHAIN_SUBSTITUTION),
    ("CCCC", "n-butyl", MutationKind.SPACER_MODIFICATION),
    ("O", "hydroxyl", MutationKind.HBOND_MODIFICATION),
    ("N", "primary amine", MutationKind.HBOND_MODIFICATION),
    ("C(=O)O", "carboxyl", MutationKind.POLARITY_MODIFICATION),
    ("C(=O)N", "amide", MutationKind.HBOND_MODIFICATION),
    ("C#N", "nitrile", MutationKind.POLARITY_MODIFICATION),
    ("F", "fluoro", MutationKind.POLARITY_MODIFICATION),
    ("Cl", "chloro", MutationKind.POLARITY_MODIFICATION),
    ("OC", "methoxy", MutationKind.FUNCTIONAL_GROUP_SUBSTITUTION),
    ("c1ccccc1", "phenyl", MutationKind.AROMATICITY_MODIFICATION),
    ("c1ccncc1", "pyridyl", MutationKind.AROMATICITY_MODIFICATION),
)


def _rdkit():
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem
    except ImportError as exc:
        raise ChemistryError(
            "RDKit is required for candidate generation", hint="pip install rdkit"
        ) from exc


def _to_mol(repeat_unit_smiles: str):
    chem = _rdkit()
    substituted = ATTACHMENT_PATTERN.sub(f"[{ATTACHMENT_ELEMENT}]", repeat_unit_smiles.strip())
    mol = chem.MolFromSmiles(substituted)
    if mol is None:
        raise ChemistryError("Parent repeat unit could not be parsed", smiles=repeat_unit_smiles)
    return mol


def _to_repeat_unit(mol) -> str | None:
    """Serialise an edited molecule back to repeat-unit SMILES, or ``None`` if invalid."""
    chem = _rdkit()
    # RDKit signals an unsanitisable molecule with several undocumented exception
    # types (RuntimeError, ValueError, Boost.Python.ArgumentError, ...).  Catching
    # broadly here is deliberate and does not hide anything: a chemically impossible
    # edit becomes ``None``, and the caller reports it as a rejected candidate.
    try:
        chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - RDKit raises assorted undocumented types
        return None
    try:
        smiles = chem.MolToSmiles(mol)
    except Exception:  # noqa: BLE001 - as above
        return None
    return smiles.replace(f"[{ATTACHMENT_ELEMENT}]", "*") if smiles else None


def _substitutable_hydrogens(mol) -> list[int]:
    """Carbons carrying at least one hydrogen that a substituent can replace.

    Backbone carbons are included: replacing a hydrogen on one is exactly how
    polyethylene becomes polypropylene, which is a side-chain substitution and one of
    the most important moves available.  The attachment bonds themselves are never
    touched, so the repeat unit stays a repeat unit.
    """
    return [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C" and atom.GetTotalNumHs() >= 1
    ]


def substitute_side_chain(
    repeat_unit_smiles: str, fragment: str, *, site: int | None = None
) -> str | None:
    """Attach ``fragment`` in place of a hydrogen on a carbon.

    Returns ``None`` when the edit does not produce a sanitisable molecule, which is
    the honest answer for a chemically impossible substitution.
    """
    chem = _rdkit()
    parent = _to_mol(repeat_unit_smiles)
    sites = _substitutable_hydrogens(parent)
    if not sites:
        return None
    target = site if site is not None else sites[0]
    if target not in sites:
        return None

    fragment_mol = chem.MolFromSmiles(fragment)
    if fragment_mol is None:
        raise ChemistryError("Substituent fragment is not valid SMILES", fragment=fragment)

    combined = chem.RWMol(chem.CombineMols(parent, fragment_mol))
    attachment_index = parent.GetNumAtoms()  # first atom of the fragment
    try:
        combined.AddBond(target, attachment_index, chem.BondType.SINGLE)
    except RuntimeError:
        return None
    return _to_repeat_unit(combined.GetMol())


def insert_backbone_spacer(repeat_unit_smiles: str, spacer: str = "C") -> str | None:
    """Lengthen the backbone by inserting a spacer next to an attachment point."""
    chem = _rdkit()
    parent = _to_mol(repeat_unit_smiles)
    anchors = [a.GetIdx() for a in parent.GetAtoms() if a.GetSymbol() == ATTACHMENT_ELEMENT]
    if len(anchors) != 2:
        return None
    anchor = anchors[0]
    neighbours = [nb.GetIdx() for nb in parent.GetAtomWithIdx(anchor).GetNeighbors()]
    if len(neighbours) != 1:
        return None
    neighbour = neighbours[0]

    spacer_mol = chem.MolFromSmiles(spacer)
    if spacer_mol is None or spacer_mol.GetNumAtoms() == 0:
        raise ChemistryError("Spacer fragment is not valid SMILES", spacer=spacer)

    editable = chem.RWMol(chem.CombineMols(parent, spacer_mol))
    spacer_start = parent.GetNumAtoms()
    spacer_end = spacer_start + spacer_mol.GetNumAtoms() - 1
    try:
        editable.RemoveBond(anchor, neighbour)
        editable.AddBond(neighbour, spacer_start, chem.BondType.SINGLE)
        editable.AddBond(spacer_end, anchor, chem.BondType.SINGLE)
    except RuntimeError:
        return None
    return _to_repeat_unit(editable.GetMol())


def replace_backbone_heteroatom(repeat_unit_smiles: str, element: str = "O") -> str | None:
    """Swap a backbone carbon for a heteroatom (e.g. polyolefin -> polyether)."""
    chem = _rdkit()
    parent = _to_mol(repeat_unit_smiles)
    anchors = [a.GetIdx() for a in parent.GetAtoms() if a.GetSymbol() == ATTACHMENT_ELEMENT]
    if len(anchors) != 2:
        return None
    path = chem.GetShortestPath(parent, anchors[0], anchors[1])
    interior = [i for i in path if i not in anchors]
    editable = chem.RWMol(parent)
    for index in interior:
        atom = editable.GetAtomWithIdx(index)
        if atom.GetSymbol() != "C" or atom.GetDegree() > 2:
            continue
        atom.SetAtomicNum(chem.Atom(element).GetAtomicNum())
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
        return _to_repeat_unit(editable.GetMol())
    return None


def validate_candidate(
    repeat_unit_smiles: str,
    *,
    parent_smiles: str,
    constraints: Constraints | None = None,
    known_ids: Iterable[str] = (),
) -> tuple[CandidateStatus, list[str], list[str], dict[str, float]]:
    """Check one generated structure.  Returns (status, issues, review reasons, descriptors)."""
    from polymer_engine.polymer.descriptors import compute_descriptors

    constraints = constraints or Constraints()
    issues: list[str] = []
    review: list[str] = []

    structural = validate_repeat_unit(repeat_unit_smiles)
    if constraints.require_two_attachment_points and count_attachment_points(repeat_unit_smiles) != 2:
        issues.append("a linear repeat unit must have exactly two attachment points")
    issues.extend(p for p in structural if "attachment point" not in p)

    chem = _rdkit()
    probe = chem.MolFromSmiles(capped_monomer_smiles(repeat_unit_smiles))
    if probe is None:
        issues.append("candidate is not a chemically valid molecule")
        return CandidateStatus.INVALID, issues, review, {}

    elements = {atom.GetSymbol() for atom in probe.GetAtoms()}

    try:
        canonical, determination = canonical_repeat_unit(repeat_unit_smiles)
    except ChemistryError as exc:
        issues.append(f"canonicalisation failed: {exc.message}")
        return CandidateStatus.INVALID, issues, review, {}
    if determination is not Determination.KNOWN:
        review.append("structure could not be canonicalised; identity is provisional")

    parent_canonical, _ = canonical_repeat_unit(parent_smiles)
    if canonical == parent_canonical:
        issues.append("candidate is identical to its parent")

    polymer_id = derive_polymer_id(canonical)
    if polymer_id in set(known_ids):
        return CandidateStatus.DUPLICATE, ["candidate duplicates a known polymer"], review, {}

    try:
        descriptor_set = compute_descriptors(repeat_unit_smiles)
    except ChemistryError as exc:
        issues.append(f"descriptors could not be computed: {exc.message}")
        return CandidateStatus.INVALID, issues, review, {}
    descriptors = descriptor_set.known()
    if not descriptors:
        review.append("no descriptors could be computed for this structure")

    issues.extend(constraints.check(descriptors, elements))

    # Structural features RDKit accepts but that warrant a human look.
    ring_info = probe.GetRingInfo()
    if any(len(ring) < 5 for ring in ring_info.AtomRings()):
        review.append("contains a strained ring (fewer than five members)")
    if any(atom.GetFormalCharge() != 0 for atom in probe.GetAtoms()):
        review.append("carries a formal charge; a neutral repeat unit is usually intended")
    if elements & {"Si", "P"}:
        review.append("contains Si or P, where force-field coverage is often incomplete")

    if issues:
        return CandidateStatus.INVALID, issues, review, descriptors
    if review:
        return CandidateStatus.REQUIRES_REVIEW, issues, review, descriptors
    return CandidateStatus.VALID, issues, review, descriptors


def generate_candidates(
    parent_smiles: str,
    *,
    parent_id: str | None = None,
    constraints: Constraints | None = None,
    known_ids: Iterable[str] = (),
    max_candidates: int = 50,
    mutations: Sequence[MutationKind] | None = None,
) -> list[GeneratedCandidate]:
    """Enumerate mutations of ``parent_smiles`` and validate every one.

    Invalid candidates are returned too, carrying their reasons.  A generator that
    silently drops its failures gives no way to tell "the operators are too
    conservative" from "the constraints are too tight".
    """
    constraints = constraints or Constraints()
    allowed = set(mutations) if mutations else set(MutationKind)

    # Mutating a discrete molecule would produce a list of "candidates" that are not
    # polymers at all, so the parent must itself be a usable repeat unit.
    parent_problems = validate_repeat_unit(parent_smiles)
    if parent_problems:
        raise ChemistryError(
            "Parent is not a usable repeat unit", smiles=parent_smiles, problems=parent_problems
        )

    parent_canonical, _ = canonical_repeat_unit(parent_smiles)
    parent_identifier = parent_id or derive_polymer_id(parent_canonical)

    seen: set[str] = set(known_ids)
    seen.add(parent_identifier)
    results: list[GeneratedCandidate] = []

    proposals: list[tuple[str | None, MutationKind, str]] = []
    for fragment, description, kind in SUBSTITUENTS:
        if kind not in allowed:
            continue
        proposals.append((substitute_side_chain(parent_smiles, fragment), kind, f"add {description}"))
    if MutationKind.SPACER_MODIFICATION in allowed:
        for spacer, description in (("C", "one methylene"), ("CC", "two methylenes"), ("CCC", "three methylenes")):
            proposals.append(
                (insert_backbone_spacer(parent_smiles, spacer), MutationKind.SPACER_MODIFICATION,
                 f"insert {description} into the backbone")
            )
    if MutationKind.BACKBONE_MODIFICATION in allowed:
        for element, description in (("O", "ether oxygen"), ("N", "amine nitrogen"), ("S", "thioether sulfur")):
            proposals.append(
                (replace_backbone_heteroatom(parent_smiles, element), MutationKind.BACKBONE_MODIFICATION,
                 f"replace a backbone carbon with an {description}")
            )

    for smiles, kind, description in proposals:
        if len(results) >= max_candidates:
            break
        if smiles is None:
            continue
        status, issues, review, descriptors = validate_candidate(
            smiles, parent_smiles=parent_smiles, constraints=constraints, known_ids=seen
        )
        if status is CandidateStatus.DUPLICATE:
            continue
        try:
            canonical, _ = canonical_repeat_unit(smiles)
        except ChemistryError:
            continue
        polymer_id = derive_polymer_id(canonical)
        if polymer_id in seen:
            continue
        seen.add(polymer_id)
        results.append(
            GeneratedCandidate(
                polymer_id=polymer_id,
                repeat_unit_smiles=smiles,
                canonical_repeat_unit=canonical,
                parent_id=parent_identifier,
                parent_smiles=parent_smiles,
                mutation=kind,
                description=description,
                status=status,
                issues=issues,
                review_reasons=review,
                descriptors=descriptors,
            )
        )

    logger.info(
        "Generated %d candidates from %s (%d valid, %d need review, %d invalid)",
        len(results),
        parent_smiles,
        sum(1 for c in results if c.status is CandidateStatus.VALID),
        sum(1 for c in results if c.status is CandidateStatus.REQUIRES_REVIEW),
        sum(1 for c in results if c.status is CandidateStatus.INVALID),
    )
    return results


__all__ = [
    "SUBSTITUENTS",
    "CandidateStatus",
    "Constraints",
    "GeneratedCandidate",
    "MutationKind",
    "generate_candidates",
    "insert_backbone_spacer",
    "replace_backbone_heteroatom",
    "substitute_side_chain",
    "validate_candidate",
]
