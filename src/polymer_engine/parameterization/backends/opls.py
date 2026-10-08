"""OPLS-AA backend, built on the installed ``oplsaa.ff``.

This is the only backend that parameterizes locally today, and it is deliberately
narrow: it assigns *tabulated* OPLS-AA types and refuses anything else.  The scope is
saturated hydrocarbons, for the reason set out in
:mod:`polymer_engine.simulation.opls_typing` -- every alkane group is neutral on its own,
so an alkane is neutral by construction with no charge derivation, whereas a polar
repeat unit built from the same table does not sum to zero.

It does not use ``gmx x2top``.  That tool emits three dihedrals for n-butane, where the
molecule has 27, filled with placeholder Ryckaert-Bellemans coefficients; ``grompp``
accepts the result and ``mdrun`` runs it.  The topology written here lists every bonded
interaction without inline parameters, so GROMACS resolves the published values from
``ffbonded.itp``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polymer_engine.core.errors import PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination
from polymer_engine.core.provenance import sha256_file
from polymer_engine.parameterization.backend import ForceFieldBackend
from polymer_engine.parameterization.capability import CapabilityState
from polymer_engine.parameterization.charges import analyse_charges, charge_gates
from polymer_engine.parameterization.completeness import analyse_topology, completeness_gates
from polymer_engine.parameterization.models import (
    ForceFieldAssessment,
    ParameterizationRequest,
    ParameterizationResult,
    ParameterizationState,
    ParameterizationValidation,
    QMPriority,
)
from polymer_engine.parameterization.quality import detect_sensitive_terms, overall_priority
from polymer_engine.simulation.opls_typing import (
    TypingStatus,
    find_opls_directory,
    load_opls_types,
    type_molecule,
    type_repeat_unit,
    write_opls_topology,
)

logger = get_logger("parameterization.opls")


class OplsBackend(ForceFieldBackend):
    name = "opls_aa"
    force_field = "OPLS-AA"
    human_in_the_loop = False
    requires_credentials = False

    def __init__(self, force_field_dir: str | Path | None = None) -> None:
        self._directory = find_opls_directory(force_field_dir)
        self._types: dict[str, Any] | None = None
        self._version: str | None = None
        if self._directory is not None:
            doc = self._directory / "forcefield.doc"
            self._version = doc.read_text(encoding="utf-8").strip() if doc.is_file() else None

    def _load(self) -> dict[str, Any]:
        if self._types is None:
            if self._directory is None:
                raise PolymerEngineError("OPLS-AA force field is not installed")
            self._types = load_opls_types(self._directory)
        return self._types

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name, "force_field": self.force_field,
            "version": self._version, "source": str(self._directory) if self._directory else None,
            "available": self._directory is not None,
            "human_in_the_loop": False, "requires_credentials": False,
            "chemistry": "saturated hydrocarbons only (tabulated neutral alkane groups)",
            "provides": ["atom_typing", "topology_generation", "system_build"],
            "notes": ("gmx x2top is deliberately not used: it emits placeholder "
                      "dihedrals that grompp accepts"),
        }

    def assess(self, polymer: Any) -> ForceFieldAssessment:
        if self._directory is None:
            return self._assessment(
                polymer, CapabilityState.UNAVAILABLE,
                "no oplsaa.ff found beside the gmx executable or in GMXLIB",
            )

        smiles = str(getattr(polymer, "canonical_repeat_unit", None)
                     or getattr(polymer, "repeat_unit_smiles", ""))
        typing = type_repeat_unit(smiles, types=self._load(),
                                  force_field_source=str(self._directory))
        if typing.status is TypingStatus.SUPPORTED:
            return self._assessment(
                polymer, CapabilityState.SYSTEM_BUILD_AVAILABLE, typing.reason,
                force_field_version=self._version, qm_priority=QMPriority.LOW,
                estimated_cost="seconds",
                evidence={"net_charge": typing.net_charge,
                          "n_atoms_typed": len(typing.assignments)},
            )
        if typing.status is TypingStatus.REQUIRES_CALIBRATION:
            return self._assessment(
                polymer, CapabilityState.REQUIRES_EXPERT_REVIEW, typing.reason,
                force_field_version=self._version, qm_priority=QMPriority.HIGH,
                evidence={"net_charge": typing.net_charge},
            )
        return self._assessment(
            polymer, CapabilityState.BLOCKED, typing.reason,
            force_field_version=self._version,
            unsupported=[f"{element}: {environment}"
                         for _index, element, environment in typing.untyped[:6]],
            evidence={"n_untyped": len(typing.untyped)},
        )

    def parameterize(self, request: ParameterizationRequest) -> ParameterizationResult:
        problems = request.problems()
        if problems:
            return self._unavailable(request, "; ".join(problems))
        if self._directory is None:
            return self._unavailable(request, "OPLS-AA is not installed",
                                     actions=["install GROMACS with its shipped oplsaa.ff"])

        workdir = Path(request.workdir or "parameterization") / request.polymer_id
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            from rdkit import Chem

            from polymer_engine.simulation.melt_builder import embed_chain, grow_chain

            chain = embed_chain(
                grow_chain(request.repeat_unit_smiles, request.degree_of_polymerization),
                seed=1,
            )
            typing = type_molecule(chain, types=self._load(),
                                   force_field_source=str(self._directory))
            if typing.status is not TypingStatus.SUPPORTED:
                return self._unavailable(request, typing.reason)

            pdb = workdir / "chain.pdb"
            Chem.MolToPDBFile(chain, str(pdb))
            names = [line[12:16].strip()
                     for line in pdb.read_text(encoding="utf-8").splitlines()
                     if line.startswith(("ATOM", "HETATM"))]
            topology = write_opls_topology(
                typing, chain, workdir / "topol.top", molecule_name="POL",
                atom_names=names, residue_name="UNL",
            )
        except PolymerEngineError as exc:
            return self._unavailable(request, f"topology generation failed: {exc}")

        return ParameterizationResult(
            backend=self.name, request=request,
            state=ParameterizationState.PARAMETERIZED,
            force_field=self.force_field, force_field_version=self._version,
            parameter_source=str(self._directory),
            topology_path=str(topology), coordinate_path=str(pdb),
            n_atoms=len(typing.assignments), net_charge=typing.net_charge,
            artifacts={str(topology): sha256_file(topology), str(pdb): sha256_file(pdb)},
            provenance={
                "force_field_dir": str(self._directory),
                "typing": typing.as_dict(),
                "method": "tabulated OPLS-AA atom types; bonded terms resolved by GROMACS",
            },
        )

    def validate(self, result: ParameterizationResult) -> ParameterizationValidation:
        validation = ParameterizationValidation(
            backend=self.name, polymer_id=result.request.polymer_id,
            property_class=result.request.property_class,
            state=ParameterizationState.PARAMETERIZED,
        )
        if not result.topology_path:
            validation.diagnostics.append("no topology to validate")
            validation.state = ParameterizationState.FAILED
            return validation

        completeness = analyse_topology(result.topology_path)
        validation.completeness = completeness_gates(completeness)
        charges = analyse_charges(result.topology_path)
        validation.charges = charge_gates(charges)
        validation.metrics = {
            "completeness": completeness.as_dict(), "charges": charges.as_dict(),
        }
        terms = detect_sensitive_terms()
        validation.qm_priority = overall_priority(terms)

        if validation.promotable:
            validation.state = ParameterizationState.PARAMETER_COMPLETENESS_VALIDATED
            validation.determination = Determination.KNOWN
        else:
            validation.state = ParameterizationState.INCONCLUSIVE
            validation.determination = Determination.REQUIRES_VALIDATION
        return validation


__all__ = ["OplsBackend"]
