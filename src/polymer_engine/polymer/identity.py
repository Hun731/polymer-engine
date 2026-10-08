"""Polymer identity and canonicalisation.

A polymer is not a molecule, and treating it as one is the root of most identity
bugs in polymer informatics.  What we canonicalise is the **repeat unit**, written
in the usual attachment-point convention:

    polyethylene           ``*CC*``
    poly(ethylene oxide)   ``*CCO*``
    poly(methyl acrylate)  ``*CC(*)C(=O)OC``

``*`` (or ``[*]``) marks a bond to the next repeat unit.  Two repeat units are the
same polymer when their canonical forms match *and* their attachment points are
equivalent, so ``*CC*`` and ``[*]CC[*]`` resolve to one identity.

RDKit is optional.  Without it, canonicalisation falls back to a conservative
textual normalisation and every derived identity is marked
``Determination.REQUIRES_VALIDATION`` rather than silently claimed as canonical.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from polymer_engine.core.errors import ChemistryError
from polymer_engine.core.models import Determination

ATTACHMENT_PATTERN = re.compile(r"\[\d*\*\d*\]|\*")

#: Placeholder used while canonicalising: a real element RDKit can parse, chosen
#: because astatine essentially never appears in polymer chemistry, so a collision
#: with genuine input is not a practical concern.
_ATTACHMENT_ISOTOPE = "[At]"


def _rdkit():
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem
    except ImportError:
        return None


def rdkit_available() -> bool:
    return _rdkit() is not None


def count_attachment_points(smiles: str) -> int:
    """Number of ``*`` / ``[*]`` attachment points in a repeat-unit SMILES."""
    return len(ATTACHMENT_PATTERN.findall(smiles))


def normalize_smiles(smiles: str) -> str:
    """Whitespace/format normalisation that does not require RDKit."""
    if not isinstance(smiles, str):
        raise ChemistryError("SMILES must be a string", got=type(smiles).__name__)
    cleaned = smiles.strip()
    if not cleaned:
        raise ChemistryError("SMILES is empty")
    return cleaned


def canonical_repeat_unit(smiles: str) -> tuple[str, Determination]:
    """Canonicalise a repeat-unit SMILES.

    Attachment points are substituted for a real atom so RDKit can canonicalise the
    graph, then restored.  Returns the canonical string and how much to trust it.
    """
    cleaned = normalize_smiles(smiles)
    chem = _rdkit()
    if chem is None:
        # Without RDKit we cannot canonicalise a molecular graph.  Say so.
        return cleaned, Determination.REQUIRES_VALIDATION

    substituted = ATTACHMENT_PATTERN.sub(_ATTACHMENT_ISOTOPE, cleaned)
    mol = chem.MolFromSmiles(substituted)
    if mol is None:
        raise ChemistryError("Repeat-unit SMILES could not be parsed", smiles=cleaned)
    canonical = chem.MolToSmiles(mol)
    restored = canonical.replace(_ATTACHMENT_ISOTOPE, "*")
    return restored, Determination.KNOWN


def build_oligomer(smiles: str, n_units: int = 3) -> tuple[Any, tuple[int, ...]] | None:
    """Build an ``n_units``-long oligomer from a repeat unit.

    This exists because hydrogen-capping a repeat unit *destroys the very linkage
    that defines its family*: ``*NCCCCCC(=O)*`` (nylon-6) capped becomes an amine
    plus an aldehyde with no amide bond anywhere, and ``*CCO*`` capped becomes
    ethanol, complete with a hydroxyl the polymer does not have.  Joining several
    repeat units makes the in-chain linkages chemically real, so substructure
    matching sees the polymer rather than a capping artifact.

    Returns ``(mol, backbone_atom_indices)``, or ``None`` when the repeat unit does
    not have exactly two attachment points (no unique chain to build) or RDKit is
    unavailable.
    """
    chem = _rdkit()
    if chem is None or n_units < 1:
        return None
    cleaned = normalize_smiles(smiles)
    if count_attachment_points(cleaned) != 2:
        return None

    unit = chem.MolFromSmiles(ATTACHMENT_PATTERN.sub(_ATTACHMENT_ISOTOPE, cleaned))
    if unit is None:
        return None

    combined = unit
    for _ in range(n_units - 1):
        combined = chem.CombineMols(combined, unit)
    editable = chem.RWMol(combined)
    unit_size = unit.GetNumAtoms()

    def anchors_of(copy_index: int) -> list[int] | None:
        offset = copy_index * unit_size
        found = [
            a.GetIdx()
            for a in editable.GetAtoms()
            if a.GetSymbol() == "At" and offset <= a.GetIdx() < offset + unit_size
        ]
        return found if len(found) == 2 else None

    # Record (anchor, neighbour) pairs before any edit, since indices shift on delete.
    per_copy: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for copy_index in range(n_units):
        anchors = anchors_of(copy_index)
        if anchors is None:
            return None
        pairs: list[tuple[int, int]] = []
        for anchor in anchors:
            neighbours = [nb.GetIdx() for nb in editable.GetAtomWithIdx(anchor).GetNeighbors()]
            if len(neighbours) != 1:
                return None
            pairs.append((anchor, neighbours[0]))
        per_copy.append((pairs[0], pairs[1]))

    # Join copy i's tail to copy i+1's head.
    for copy_index in range(n_units - 1):
        _, tail = per_copy[copy_index]
        head, _ = per_copy[copy_index + 1]
        try:
            editable.AddBond(tail[1], head[1], chem.BondType.SINGLE)
        except RuntimeError:
            return None

    # Drop every placeholder; RDKit restores the implicit hydrogens.
    doomed = sorted((a.GetIdx() for a in editable.GetAtoms() if a.GetSymbol() == "At"), reverse=True)
    surviving_terminals = [per_copy[0][0][1], per_copy[-1][1][1]]
    shift = {idx: idx for idx in range(editable.GetNumAtoms())}
    for anchor in doomed:
        editable.RemoveAtom(anchor)
        for original, current in list(shift.items()):
            if current > anchor:
                shift[original] = current - 1
            elif current == anchor:
                shift[original] = -1

    mol = editable.GetMol()
    try:
        chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - RDKit raises assorted undocumented types
        return None

    start, end = (shift.get(i, -1) for i in surviving_terminals)
    if start < 0 or end < 0:
        return mol, ()
    path = chem.GetShortestPath(mol, start, end)
    return mol, tuple(path)


def capped_monomer_smiles(smiles: str, *, cap: str = "[H]") -> str:
    """Replace attachment points with a cap so descriptors can be computed.

    Descriptors of a repeat unit are only meaningful on a closed-valence molecule.
    Hydrogen capping is the usual convention and is recorded in provenance so the
    choice is never implicit.
    """
    cleaned = normalize_smiles(smiles)
    if count_attachment_points(cleaned) == 0:
        return cleaned
    return ATTACHMENT_PATTERN.sub(cap, cleaned)


def inchikey_for_repeat_unit(smiles: str) -> tuple[str | None, Determination]:
    """InChIKey of the hydrogen-capped repeat unit.

    Returns ``(None, UNKNOWN)`` when RDKit is unavailable rather than fabricating a
    key from a hash, which would collide differently from a real InChIKey.
    """
    chem = _rdkit()
    if chem is None:
        return None, Determination.UNKNOWN
    capped = capped_monomer_smiles(smiles)
    mol = chem.MolFromSmiles(capped)
    if mol is None:
        raise ChemistryError("Capped repeat unit could not be parsed", smiles=capped)
    try:
        key = chem.MolToInchiKey(mol)
    # RDKit raises assorted types when the InChI toolkit is unavailable or rejects a
    # structure; re-raised as a typed engine error so callers can handle it.
    except Exception as exc:
        raise ChemistryError(f"InChIKey generation failed: {exc}", smiles=capped) from exc
    if not key:
        return None, Determination.UNKNOWN
    return key, Determination.KNOWN


def validate_repeat_unit(smiles: str) -> list[str]:
    """Return a list of problems with a repeat-unit SMILES.  Empty means usable."""
    problems: list[str] = []
    try:
        cleaned = normalize_smiles(smiles)
    except ChemistryError as exc:
        return [exc.message]

    n_attachments = count_attachment_points(cleaned)
    if n_attachments == 0:
        problems.append(
            "no attachment point '*' found; this looks like a discrete molecule, not a repeat unit"
        )
    elif n_attachments == 1:
        problems.append("only one attachment point; a linear repeat unit needs two")
    elif n_attachments > 4:
        problems.append(f"{n_attachments} attachment points is unusually high for a repeat unit")

    chem = _rdkit()
    if chem is not None:
        substituted = ATTACHMENT_PATTERN.sub(_ATTACHMENT_ISOTOPE, cleaned)
        mol = chem.MolFromSmiles(substituted)
        if mol is None:
            problems.append("SMILES could not be parsed as a valid molecular graph")
        elif mol.GetNumAtoms() <= n_attachments:
            problems.append("repeat unit contains no atoms besides its attachment points")
    return problems


@dataclass(frozen=True, slots=True)
class PolymerIdentity:
    """A resolved, hashable polymer identity.

    ``polymer_id`` is derived deterministically from the canonical repeat unit plus
    the architectural fields, so the same polymer described twice resolves to the
    same id without a central registry.
    """

    polymer_id: str
    name: str
    repeat_unit_smiles: str
    canonical_repeat_unit: str
    inchikey: str | None
    canonicalisation: Determination
    n_attachment_points: int
    architecture: Literal["linear", "branched", "network", "unknown"] = "linear"
    tacticity: Literal["atactic", "isotactic", "syndiotactic", "unknown"] = "unknown"
    degree_of_polymerization: int | None = None
    end_groups: tuple[str, ...] = ()
    copolymer_of: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id,
            "name": self.name,
            "repeat_unit_smiles": self.repeat_unit_smiles,
            "canonical_repeat_unit": self.canonical_repeat_unit,
            "inchikey": self.inchikey,
            "canonicalisation": self.canonicalisation.value,
            "n_attachment_points": self.n_attachment_points,
            "architecture": self.architecture,
            "tacticity": self.tacticity,
            "degree_of_polymerization": self.degree_of_polymerization,
            "end_groups": list(self.end_groups),
            "copolymer_of": list(self.copolymer_of),
            "issues": list(self.issues),
        }


def make_identity(
    *,
    name: str,
    repeat_unit_smiles: str,
    architecture: str = "linear",
    tacticity: str = "unknown",
    degree_of_polymerization: int | None = None,
    end_groups: tuple[str, ...] = (),
    copolymer_of: tuple[str, ...] = (),
    strict: bool = False,
) -> PolymerIdentity:
    """Resolve a polymer identity.

    With ``strict=True`` an invalid repeat unit raises; otherwise the problems are
    attached to the identity and ``usable`` is False.  Nothing is silently accepted.
    """
    problems = tuple(validate_repeat_unit(repeat_unit_smiles))
    if problems and strict:
        raise ChemistryError(
            "Repeat unit failed validation", smiles=repeat_unit_smiles, problems=list(problems)
        )

    cleaned = normalize_smiles(repeat_unit_smiles)
    try:
        canonical, determination = canonical_repeat_unit(cleaned)
    except ChemistryError:
        if strict:
            raise
        canonical, determination = cleaned, Determination.REQUIRES_VALIDATION

    inchikey: str | None = None
    if determination is Determination.KNOWN:
        try:
            inchikey, _ = inchikey_for_repeat_unit(cleaned)
        except ChemistryError:
            inchikey = None

    return PolymerIdentity(
        polymer_id=derive_polymer_id(
            canonical,
            architecture=architecture,
            tacticity=tacticity,
            copolymer_of=copolymer_of,
        ),
        name=name,
        repeat_unit_smiles=cleaned,
        canonical_repeat_unit=canonical,
        inchikey=inchikey,
        canonicalisation=determination,
        n_attachment_points=count_attachment_points(cleaned),
        architecture=architecture,  # type: ignore[arg-type]
        tacticity=tacticity,  # type: ignore[arg-type]
        degree_of_polymerization=degree_of_polymerization,
        end_groups=tuple(end_groups),
        copolymer_of=tuple(copolymer_of),
        issues=problems,
    )


def derive_polymer_id(
    canonical_smiles: str,
    *,
    architecture: str = "linear",
    tacticity: str = "unknown",
    copolymer_of: tuple[str, ...] = (),
) -> str:
    """Deterministic id.

    Deliberately includes architecture and tacticity: the same repeat unit as an
    isotactic linear chain and as a crosslinked network is not the same material,
    and merging them would corrupt any structure-property model built on top.
    """
    payload = "|".join(
        [canonical_smiles, architecture, tacticity, ",".join(sorted(copolymer_of))]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"pol_{digest}"


def same_polymer(a: PolymerIdentity, b: PolymerIdentity) -> bool:
    """Whether two identities denote the same material."""
    return a.polymer_id == b.polymer_id


@dataclass
class DeduplicationReport:
    unique: list[PolymerIdentity] = field(default_factory=list)
    duplicates: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n_duplicates(self) -> int:
        return sum(len(v) for v in self.duplicates.values())


def deduplicate(identities: list[PolymerIdentity]) -> DeduplicationReport:
    """Collapse identities that resolve to the same polymer, keeping the first.

    The report records which names were merged so a curation decision is auditable
    rather than invisible.
    """
    report = DeduplicationReport()
    seen: dict[str, PolymerIdentity] = {}
    for identity in identities:
        existing = seen.get(identity.polymer_id)
        if existing is None:
            seen[identity.polymer_id] = identity
            report.unique.append(identity)
        else:
            report.duplicates.setdefault(identity.polymer_id, []).append(identity.name)
    return report


__all__ = [
    "ATTACHMENT_PATTERN",
    "DeduplicationReport",
    "PolymerIdentity",
    "build_oligomer",
    "canonical_repeat_unit",
    "capped_monomer_smiles",
    "count_attachment_points",
    "deduplicate",
    "derive_polymer_id",
    "inchikey_for_repeat_unit",
    "make_identity",
    "normalize_smiles",
    "rdkit_available",
    "same_polymer",
    "validate_repeat_unit",
]
