"""Repeat-unit descriptors.

Every descriptor is returned as a :class:`Measurement` carrying its unit, so a
consumer cannot mistake TPSA (angstrom^2) for logP (dimensionless) or silently mix
g/mol with amu.

Descriptors are computed on the **hydrogen-capped repeat unit**.  That convention is
recorded in the result: a capped repeat unit is not the polymer, and descriptors
derived from it are per-repeat-unit quantities, not per-chain quantities.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from polymer_engine.core.errors import ChemistryError
from polymer_engine.core.models import Determination, Measurement
from polymer_engine.polymer.identity import capped_monomer_smiles, count_attachment_points

#: name -> (unit, description).  The unit strings must exist in core.units.
DESCRIPTOR_UNITS: dict[str, tuple[str, str]] = {
    "repeat_unit_mass": ("g/mol", "Molar mass of the hydrogen-capped repeat unit"),
    "heavy_atom_count": ("1", "Non-hydrogen atoms in the capped repeat unit"),
    "hbond_donor_count": ("1", "Lipinski hydrogen-bond donors"),
    "hbond_acceptor_count": ("1", "Lipinski hydrogen-bond acceptors"),
    "tpsa": ("1", "Topological polar surface area (angstrom^2, reported dimensionless)"),
    "logp": ("1", "Crippen octanol-water partition coefficient (log10)"),
    "molar_refractivity": ("1", "Crippen molar refractivity"),
    "fraction_csp3": ("1", "Fraction of carbons that are sp3"),
    "rotatable_bond_count": ("1", "Rotatable bonds in the capped repeat unit"),
    "ring_count": ("1", "Rings in the repeat unit"),
    "aromatic_ring_count": ("1", "Aromatic rings in the repeat unit"),
    "formal_charge": ("1", "Net formal charge"),
    "heteroatom_count": ("1", "Non-carbon, non-hydrogen atoms"),
    "backbone_atom_count": ("1", "Shortest path between the two attachment points"),
    "side_chain_heavy_atoms": ("1", "Heavy atoms not on the backbone path"),
}


@dataclass
class DescriptorSet:
    """Descriptors for one polymer, with the convention that produced them."""

    polymer_id: str
    smiles: str
    capped_smiles: str
    measurements: dict[str, Measurement] = field(default_factory=dict)
    convention: str = "hydrogen-capped repeat unit"
    backend: str = "rdkit"
    issues: list[str] = field(default_factory=list)

    def value(self, name: str) -> float | None:
        measurement = self.measurements.get(name)
        return measurement.value if measurement else None

    def known(self) -> dict[str, float]:
        """Only descriptors that were actually computed."""
        return {
            name: m.value
            for name, m in self.measurements.items()
            if m.determination is Determination.KNOWN and m.value is not None
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id,
            "smiles": self.smiles,
            "capped_smiles": self.capped_smiles,
            "convention": self.convention,
            "backend": self.backend,
            "issues": self.issues,
            "descriptors": {
                name: {
                    "value": m.value,
                    "units": m.units,
                    "determination": m.determination.value,
                }
                for name, m in self.measurements.items()
            },
        }

    def vector(self, names: list[str]) -> list[float | None]:
        """Descriptor values in a fixed order, ``None`` where unknown.

        Callers must decide how to handle missing values; imputing here would hide
        the difference between "zero" and "not computed".
        """
        return [self.value(name) for name in names]


def _unknown_set(polymer_id: str, smiles: str, capped: str, reason: str, backend: str) -> DescriptorSet:
    return DescriptorSet(
        polymer_id=polymer_id,
        smiles=smiles,
        capped_smiles=capped,
        backend=backend,
        issues=[reason],
        measurements={
            name: Measurement.unknown(name, units=unit, reason=reason)
            for name, (unit, _) in DESCRIPTOR_UNITS.items()
        },
    )


def compute_descriptors(smiles: str, *, polymer_id: str = "", cap: str = "[H]") -> DescriptorSet:
    """Compute repeat-unit descriptors.

    Without RDKit the result is a full set of ``UNKNOWN`` measurements rather than an
    exception, so a pipeline can proceed and record honestly that descriptors were
    unavailable.
    """
    capped = capped_monomer_smiles(smiles, cap=cap)
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return _unknown_set(polymer_id, smiles, capped, "RDKit is not installed", backend="none")

    mol = Chem.MolFromSmiles(capped)
    if mol is None:
        raise ChemistryError("Capped repeat unit is not a valid molecule", smiles=capped, source=smiles)

    computed: dict[str, Callable[[], float]] = {
        "repeat_unit_mass": lambda: float(Descriptors.MolWt(mol)),
        "heavy_atom_count": lambda: float(mol.GetNumHeavyAtoms()),
        "hbond_donor_count": lambda: float(Lipinski.NumHDonors(mol)),
        "hbond_acceptor_count": lambda: float(Lipinski.NumHAcceptors(mol)),
        "tpsa": lambda: float(rdMolDescriptors.CalcTPSA(mol)),
        "logp": lambda: float(Crippen.MolLogP(mol)),
        "molar_refractivity": lambda: float(Crippen.MolMR(mol)),
        "fraction_csp3": lambda: float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "rotatable_bond_count": lambda: float(Lipinski.NumRotatableBonds(mol)),
        "ring_count": lambda: float(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_ring_count": lambda: float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "formal_charge": lambda: float(Chem.GetFormalCharge(mol)),
        "heteroatom_count": lambda: float(
            sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in {"C", "H"})
        ),
    }

    measurements: dict[str, Measurement] = {}
    issues: list[str] = []
    for name, fn in computed.items():
        unit = DESCRIPTOR_UNITS[name][0]
        try:
            measurements[name] = Measurement(name=name, value=fn(), units=unit, method="rdkit")
        # A single descriptor failing must not discard the other fourteen.  The
        # failure is recorded on the measurement and surfaced in ``issues`` rather
        # than swallowed, and RDKit's exception types here are not documented.
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            issues.append(f"{name}: {exc}")
            measurements[name] = Measurement.unknown(name, units=unit, reason=str(exc))

    backbone, side_chain = _backbone_and_side_chain(smiles)
    for name, value in (("backbone_atom_count", backbone), ("side_chain_heavy_atoms", side_chain)):
        unit = DESCRIPTOR_UNITS[name][0]
        measurements[name] = (
            Measurement(name=name, value=float(value), units=unit, method="graph-path")
            if value is not None
            else Measurement.unknown(name, units=unit, reason="attachment points not resolvable")
        )

    return DescriptorSet(
        polymer_id=polymer_id,
        smiles=smiles,
        capped_smiles=capped,
        measurements=measurements,
        issues=issues,
    )


def _backbone_and_side_chain(smiles: str) -> tuple[int | None, int | None]:
    """Backbone length as the shortest path between the two attachment points.

    Returns ``(None, None)`` when there are not exactly two attachment points --
    a branched or network repeat unit has no single backbone path, and inventing one
    would be wrong.
    """
    if count_attachment_points(smiles) != 2:
        return None, None
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None, None

    from polymer_engine.polymer.identity import ATTACHMENT_PATTERN

    substituted = ATTACHMENT_PATTERN.sub("[At]", smiles.strip())
    mol = Chem.MolFromSmiles(substituted)
    if mol is None:
        return None, None
    anchors = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "At"]
    if len(anchors) != 2:
        return None, None
    path = Chem.GetShortestPath(mol, anchors[0], anchors[1])
    if not path:
        return None, None
    # Exclude the two placeholder atoms; what remains is the backbone.
    backbone = max(0, len(path) - 2)
    total_heavy = mol.GetNumHeavyAtoms() - 2
    return backbone, max(0, total_heavy - backbone)


def descriptor_matrix(
    descriptor_sets: list[DescriptorSet], names: list[str] | None = None
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Assemble a (polymer_ids, descriptor_names, matrix) triple.

    ``None`` entries are preserved so the caller decides on imputation explicitly.
    """
    names = names or sorted(DESCRIPTOR_UNITS)
    ids = [d.polymer_id for d in descriptor_sets]
    matrix = [d.vector(names) for d in descriptor_sets]
    return ids, names, matrix


__all__ = [
    "DESCRIPTOR_UNITS",
    "DescriptorSet",
    "compute_descriptors",
    "descriptor_matrix",
]
