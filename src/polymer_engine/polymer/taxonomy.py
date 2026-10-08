"""Polymer family classification.

Classification runs on a short **oligomer**, not on a hydrogen-capped repeat unit.
That choice is forced by chemistry: capping ``*NCCCCCC(=O)*`` (nylon-6) with
hydrogens yields an amine and an aldehyde with no amide bond anywhere, and capping
``*CCO*`` yields ethanol, complete with a hydroxyl the polymer does not have.
Joining repeat units makes the in-chain linkages real.

Two further refinements matter:

* **Only the interior of the oligomer counts.**  The two chain ends still carry
  capping artifacts, so matches are required to touch the interior region.
* **Backbone versus pendant is decisive.**  An ester *in* the chain is a polyester;
  the same ester *hanging off* the chain is an acrylate.  Poly(ethylene
  terephthalate) and poly(methyl methacrylate) both contain ``C(=O)O`` and are not
  remotely the same material, so each rule declares where its group must sit.

An unmatched repeat unit is ``UNCLASSIFIED``.  It is never forced into the nearest
family, because a wrong family label propagates into every family-conditioned model
downstream.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

#: How many repeat units to build.  Four gives two clean interior units.
OLIGOMER_UNITS = 4

Placement = Literal["in_chain", "pendant", "any"]


class PolymerFamily(str, Enum):
    POLYESTER = "polyester"
    POLYAMIDE = "polyamide"
    POLYURETHANE = "polyurethane"
    POLYUREA = "polyurea"
    POLYCARBONATE = "polycarbonate"
    POLYETHER = "polyether"
    POLYSILOXANE = "polysiloxane"
    POLYSULFONE = "polysulfone"
    POLYIMIDE = "polyimide"
    POLYACRYLATE = "polyacrylate"
    POLYACRYLAMIDE = "polyacrylamide"
    POLYNITRILE = "polynitrile"
    POLYVINYL_HALIDE = "polyvinyl_halide"
    POLYVINYL_ALCOHOL = "polyvinyl_alcohol"
    POLYSTYRENIC = "polystyrenic"
    POLYOLEFIN = "polyolefin"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True, slots=True)
class FamilyRule:
    family: PolymerFamily
    smarts: str
    basis: str
    #: ``in_chain`` needs >=2 matched atoms on the backbone; ``pendant`` needs the
    #: characteristic group off the backbone.
    placement: Placement = "any"
    #: For pendant rules, how many matched atoms may sit on the backbone (the
    #: attachment carbon itself usually does).
    max_backbone_atoms: int = 1


#: Evaluated in order; the first rule whose placement is satisfied wins.  Ordering
#: encodes chemical specificity -- a carbamate is matched before the ester and ether
#: fragments it contains.
FAMILY_RULES: tuple[FamilyRule, ...] = (
    FamilyRule(PolymerFamily.POLYIMIDE, "O=C1[#7]C(=O)c2ccccc21", "cyclic imide fused to an aromatic ring"),
    FamilyRule(PolymerFamily.POLYURETHANE, "[NX3][CX3](=O)[OX2H0]", "carbamate (urethane) linkage in the chain", "in_chain"),
    FamilyRule(PolymerFamily.POLYUREA, "[NX3][CX3](=O)[NX3]", "urea linkage in the chain", "in_chain"),
    FamilyRule(PolymerFamily.POLYCARBONATE, "[OX2H0][CX3](=O)[OX2H0]", "carbonate linkage in the chain", "in_chain"),
    FamilyRule(PolymerFamily.POLYSULFONE, "[#16X4](=[OX1])(=[OX1])", "sulfone group in the chain", "in_chain"),
    FamilyRule(PolymerFamily.POLYSILOXANE, "[Si][OX2][Si]", "siloxane backbone", "in_chain"),
    FamilyRule(PolymerFamily.POLYAMIDE, "[CX3](=[OX1])[NX3]", "amide linkage in the chain", "in_chain"),
    FamilyRule(PolymerFamily.POLYESTER, "[CX3](=[OX1])[OX2H0]", "carboxylic ester linkage in the chain", "in_chain"),
    FamilyRule(PolymerFamily.POLYACRYLAMIDE, "[CX3](=[OX1])[NX3]", "amide pendant on a saturated backbone", "pendant"),
    FamilyRule(PolymerFamily.POLYACRYLATE, "[CX3](=[OX1])[OX2H0][#6]", "carboxylic ester pendant on a saturated backbone", "pendant"),
    FamilyRule(PolymerFamily.POLYNITRILE, "[NX1]#[CX2]", "nitrile pendant", "pendant"),
    FamilyRule(PolymerFamily.POLYVINYL_HALIDE, "[CX4][F,Cl,Br,I]", "halogen on a backbone carbon", "pendant"),
    FamilyRule(PolymerFamily.POLYSTYRENIC, "[CX4][c]1[c][c][c][c][c]1", "phenyl pendant on a backbone carbon", "pendant"),
    FamilyRule(PolymerFamily.POLYVINYL_ALCOHOL, "[CX4][OX2H1]", "hydroxyl on a backbone carbon", "pendant"),
    FamilyRule(PolymerFamily.POLYETHER, "[#6][OX2H0][#6]", "ether oxygen in the chain", "in_chain"),
)


@dataclass(frozen=True, slots=True)
class Classification:
    family: PolymerFamily
    basis: str
    matched_groups: tuple[str, ...] = ()
    confident: bool = True
    notes: str = ""

    @property
    def requires_review(self) -> bool:
        return not self.confident

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "basis": self.basis,
            "matched_groups": list(self.matched_groups),
            "confident": self.confident,
            "requires_review": self.requires_review,
            "notes": self.notes,
        }


def _unclassified(basis: str, notes: str, matched: tuple[str, ...] = ()) -> Classification:
    return Classification(
        family=PolymerFamily.UNCLASSIFIED,
        basis=basis,
        matched_groups=matched,
        confident=False,
        notes=notes,
    )


def _interior_region(mol: Any, backbone: tuple[int, ...], n_units: int) -> set[int]:
    """Atoms belonging to the interior repeat units, side chains included.

    The first and last repeat unit are excluded because they carry the capping
    artifacts that would otherwise be classified as real functional groups.
    """
    if not backbone or n_units < 3:
        return set(range(mol.GetNumAtoms()))
    per_unit = max(1, len(backbone) // n_units)
    interior_backbone = set(backbone[per_unit : len(backbone) - per_unit])
    if not interior_backbone:
        interior_backbone = set(backbone)
    backbone_set = set(backbone)

    # Walk outward from the interior backbone into side chains, never crossing into
    # another backbone atom (that would leak back into the terminal units).
    region = set(interior_backbone)
    queue = deque(interior_backbone)
    while queue:
        idx = queue.popleft()
        for neighbour in mol.GetAtomWithIdx(idx).GetNeighbors():
            nb = neighbour.GetIdx()
            if nb in region or nb in backbone_set:
                continue
            region.add(nb)
            queue.append(nb)
    return region


def classify(repeat_unit_smiles: str, *, n_units: int = OLIGOMER_UNITS) -> Classification:
    """Assign a polymer family from a repeat-unit SMILES."""
    from polymer_engine.polymer.identity import build_oligomer, capped_monomer_smiles

    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return _unclassified(
            "RDKit unavailable",
            "Substructure classification needs RDKit; install it to enable family assignment.",
        )

    built = build_oligomer(repeat_unit_smiles, n_units)
    if built is not None:
        mol, backbone = built
        region = _interior_region(mol, backbone, n_units)
        backbone_set = set(backbone)
    else:
        # No unique chain (branched/network repeat unit, or an unparseable one).
        # Fall back to the capped monomer and mark the answer as needing review.
        capped = capped_monomer_smiles(repeat_unit_smiles)
        mol = Chem.MolFromSmiles(capped)
        if mol is None:
            return _unclassified(
                "unparseable repeat unit", f"Could not parse {repeat_unit_smiles!r} as a molecule."
            )
        region = set(range(mol.GetNumAtoms()))
        backbone_set = set()

    matched: list[str] = []
    chosen: FamilyRule | None = None

    for rule in FAMILY_RULES:
        pattern = Chem.MolFromSmarts(rule.smarts)
        if pattern is None:  # pragma: no cover - guards a malformed rule
            continue
        hits = [hit for hit in mol.GetSubstructMatches(pattern) if region & set(hit)]
        if not hits:
            continue
        if rule.family.value not in matched:
            matched.append(rule.family.value)
        if chosen is not None:
            continue
        if backbone_set and not any(_placement_ok(rule, hit, backbone_set) for hit in hits):
            continue
        chosen = rule

    if chosen is None:
        if all(atom.GetSymbol() in {"C", "H"} for atom in mol.GetAtoms()):
            return Classification(
                family=PolymerFamily.POLYOLEFIN,
                basis="saturated hydrocarbon repeat unit with no functional groups",
                matched_groups=tuple(matched),
            )
        return _unclassified(
            "no rule matched",
            "Repeat unit did not match any known family pattern; requires manual review.",
            tuple(matched),
        )

    others = [m for m in matched if m != chosen.family.value]
    return Classification(
        family=chosen.family,
        basis=chosen.basis,
        matched_groups=tuple(matched),
        confident=built is not None,
        notes=(
            "" if built is not None else "Classified from a capped monomer; no unique backbone was resolvable."
        )
        + (f" Also matched: {', '.join(others)}." if others else ""),
    )


def _placement_ok(rule: FamilyRule, hit: tuple[int, ...], backbone: set[int]) -> bool:
    overlap = len(set(hit) & backbone)
    if rule.placement == "in_chain":
        return overlap >= 2
    if rule.placement == "pendant":
        return overlap <= rule.max_backbone_atoms
    return True


def group_by_family(identities: list[tuple[str, str]]) -> dict[PolymerFamily, list[str]]:
    """Group ``(polymer_id, repeat_unit_smiles)`` pairs by classified family."""
    groups: dict[PolymerFamily, list[str]] = {}
    for polymer_id, smiles in identities:
        groups.setdefault(classify(smiles).family, []).append(polymer_id)
    return groups


__all__ = [
    "FAMILY_RULES",
    "OLIGOMER_UNITS",
    "Classification",
    "FamilyRule",
    "PolymerFamily",
    "classify",
    "group_by_family",
]
