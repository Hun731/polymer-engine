"""OpenFF backend, driven through an isolated environment.

OpenFF lives in ``.paramenv`` rather than the engine's own ``.venv``, deliberately: a
running campaign's environment must not acquire a large new dependency tree mid-run.
This backend therefore shells out to :mod:`scripts.openff_worker`, which is the only
code that imports OpenFF.

Two scientific points shape the implementation.

**The charge model is part of the force field.** Sage was fitted against AM1-BCC
charges. AM1-BCC itself needs AmberTools' ``sqm`` or an OpenEye licence, neither of
which is installed here, so the worker uses NAGL -- OpenFF's published graph-network
surrogate for AM1-BCC. Pairing Sage with, say, Gasteiger charges would be a silent
change of force field, and the worker refuses it rather than falling back.

**Coverage is not validation.** Sage is fitted to small drug-like molecules. It types a
polymer oligomer cleanly, which is not the same as being validated for a bulk melt, so
this backend never reports better than ``PARAMETERIZATION_AVAILABLE`` on its own.
Qualification still has to come from the validation layer.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination
from polymer_engine.core.provenance import sha256_file
from polymer_engine.parameterization.backend import ForceFieldBackend
from polymer_engine.parameterization.capability import CapabilityState
from polymer_engine.parameterization.charges import analyse_charges, charge_gates
from polymer_engine.parameterization.completeness import (
    analyse_topology,
    completeness_gates,
)
from polymer_engine.parameterization.models import (
    ForceFieldAssessment,
    ParameterizationRequest,
    ParameterizationResult,
    ParameterizationState,
    ParameterizationValidation,
    QMPriority,
)
from polymer_engine.parameterization.quality import (
    detect_sensitive_terms,
    overall_priority,
)

logger = get_logger("parameterization.openff")

#: Where the isolated OpenFF environment lives, relative to the repository root.
PARAMENV = Path(__file__).resolve().parents[4] / ".paramenv" / "bin" / "python"
WORKER = Path(__file__).resolve().parents[4] / "scripts" / "openff_worker.py"

#: A parameterization is minutes at most; anything longer has hung.
WORKER_TIMEOUT_S = 900.0


def _call_worker(request: dict[str, Any], *, timeout_s: float = 60.0) -> dict[str, Any]:
    """Run one worker action in the isolated environment.

    Never raises for a scientific failure: the worker's own error text is returned so the
    caller can report *why* rather than merely that something went wrong.
    """
    if not PARAMENV.is_file() or not WORKER.is_file():
        return {"ok": False, "available": False,
                "error": f"isolated OpenFF environment not found at {PARAMENV}"}
    try:
        proc = subprocess.run(
            [str(PARAMENV), str(WORKER), json.dumps(request)],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "available": False, "error": f"worker failed: {exc}"}
    # The worker prints exactly one JSON object; OpenFF writes warnings to stderr.
    for line in reversed(proc.stdout.strip().splitlines()):
        try:
            return dict(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"ok": False, "available": False,
            "error": f"worker produced no JSON: {(proc.stderr or '')[-300:]}"}


class OpenFFBackend(ForceFieldBackend):
    name = "openff"
    force_field = "OpenFF Sage"
    human_in_the_loop = False
    requires_credentials = False

    def __init__(self, force_field: str = "openff-2.2.0.offxml") -> None:
        self.offxml = force_field
        self._capabilities: dict[str, Any] | None = None

    def _probe(self) -> dict[str, Any]:
        if self._capabilities is None:
            self._capabilities = _call_worker({"action": "capabilities"})
        return self._capabilities

    @property
    def _ready(self) -> bool:
        probe = self._probe()
        return bool(probe.get("available")) and bool(probe.get("charge_methods"))

    def capabilities(self) -> dict[str, Any]:
        probe = self._probe()
        return {
            "backend": self.name, "force_field": self.force_field,
            "version": probe.get("openff_toolkit"),
            "interchange_version": probe.get("openff_interchange"),
            "available": self._ready,
            "environment": str(PARAMENV.parent.parent),
            "human_in_the_loop": False, "requires_credentials": False,
            "charge_methods": probe.get("charge_methods", []),
            "charge_model": probe.get("nagl_model"),
            "force_fields": probe.get("force_fields", []),
            "missing": [] if self._ready else ["openff-toolkit", "openff-interchange"],
            "provides": ["molecule_construction", "parameter_assignment",
                         "charge_assignment", "gromacs_export"],
            "install_hint": ("mamba create -p .paramenv -c conda-forge python=3.11 "
                             "openff-toolkit openff-interchange packmol openbabel rdkit"),
            "caveat": ("Sage is fitted to small drug-like molecules; typing an oligomer "
                       "is not validation for a bulk melt"),
            "error": probe.get("error"),
        }

    def assess(self, polymer: Any) -> ForceFieldAssessment:
        if not self._ready:
            probe = self._probe()
            return self._assessment(
                polymer, CapabilityState.UNAVAILABLE,
                probe.get("error") or "OpenFF environment is not usable",
                evidence={"install_hint": self.capabilities()["install_hint"]},
            )

        smiles = _oligomer_smiles(polymer)
        if smiles is None:
            return self._assessment(
                polymer, CapabilityState.BLOCKED,
                "repeat unit could not be expanded into an oligomer to type",
            )
        result = _call_worker(
            {"action": "assess", "smiles": smiles, "force_field": self.offxml}
        )
        if not result.get("ok"):
            # A typing failure is a real chemistry limit, not an outage.
            return self._assessment(
                polymer, CapabilityState.BLOCKED,
                f"{self.offxml} cannot type this chemistry: {result.get('error', '')[:150]}",
                force_field_version=self._probe().get("openff_toolkit"),
                evidence={"smiles": smiles},
            )
        return self._assessment(
            polymer, CapabilityState.PARAMETERIZATION_AVAILABLE,
            (f"{self.offxml} types the oligomer ({result.get('n_atoms')} atoms); charges "
             f"from {self._probe().get('nagl_model')}. Typing is not melt validation."),
            force_field_version=self._probe().get("openff_toolkit"),
            qm_priority=QMPriority.MEDIUM, estimated_cost="seconds to minutes",
            evidence={"smiles": smiles, "formula": result.get("formula"),
                      "n_atoms": result.get("n_atoms")},
        )

    def parameterize(self, request: ParameterizationRequest) -> ParameterizationResult:
        problems = request.problems()
        if problems:
            return self._unavailable(request, "; ".join(problems))
        if not self._ready:
            return self._unavailable(
                request, self._probe().get("error") or "OpenFF is not available",
                actions=[self.capabilities()["install_hint"]],
            )

        smiles = _oligomer_smiles_from_repeat(
            request.repeat_unit_smiles, request.degree_of_polymerization
        )
        if smiles is None:
            return self._unavailable(
                request, "repeat unit could not be expanded into an oligomer")

        workdir = Path(request.workdir or "parameterization") / request.polymer_id
        workdir.mkdir(parents=True, exist_ok=True)
        prefix = workdir / "openff"
        result = _call_worker(
            {"action": "parameterize", "smiles": smiles, "force_field": self.offxml,
             "charge_method": "nagl", "prefix": str(prefix)},
            timeout_s=WORKER_TIMEOUT_S,
        )
        if not result.get("ok"):
            return self._unavailable(request, result.get("error", "parameterization failed"))

        topology, coordinates = Path(result["topology"]), Path(result["coordinates"])
        artifacts = {
            str(p): sha256_file(p) for p in (topology, coordinates) if p.is_file()
        }
        return ParameterizationResult(
            backend=self.name, request=request,
            state=ParameterizationState.PARAMETERIZED,
            force_field=f"{self.force_field} ({self.offxml})",
            force_field_version=self._probe().get("openff_toolkit"),
            parameter_source=f"openff-interchange {self._probe().get('openff_interchange')}",
            topology_path=str(topology), coordinate_path=str(coordinates),
            n_atoms=result.get("n_atoms"), net_charge=result.get("net_charge"),
            artifacts=artifacts,
            provenance={
                "offxml": self.offxml,
                "charge_method": result.get("charge_method"),
                "oligomer_smiles": smiles,
                "degree_of_polymerization": request.degree_of_polymerization,
                "environment": str(PARAMENV.parent.parent),
                "note": ("charges come from OpenFF's NAGL graph model, the published "
                         "AM1-BCC surrogate Sage expects"),
            },
        )

    def validate(self, result: ParameterizationResult) -> ParameterizationValidation:
        validation = ParameterizationValidation(
            backend=self.name, polymer_id=result.request.polymer_id,
            property_class=result.request.property_class, state=result.state,
        )
        if result.state is not ParameterizationState.PARAMETERIZED or not result.topology_path:
            validation.diagnostics.append(
                f"nothing to validate: parameterization is {result.state.value}")
            return validation

        completeness = analyse_topology(result.topology_path)
        validation.completeness = completeness_gates(completeness)
        charges = analyse_charges(result.topology_path)
        validation.charges = charge_gates(charges)
        validation.metrics = {"completeness": completeness.as_dict(),
                              "charges": charges.as_dict()}
        # OpenFF reports no analogy penalty, so chemistry alone decides QM priority.
        # Silence is not evidence of quality.
        validation.qm_priority = overall_priority(detect_sensitive_terms(
            None, functional_groups=result.provenance.get("functional_groups", []),
        )) if result.provenance.get("functional_groups") else QMPriority.MEDIUM

        if validation.promotable:
            validation.state = ParameterizationState.PARAMETER_COMPLETENESS_VALIDATED
            validation.determination = Determination.KNOWN
        else:
            validation.state = ParameterizationState.INCONCLUSIVE
            validation.determination = Determination.REQUIRES_VALIDATION
        return validation


def _oligomer_smiles(polymer: Any) -> str | None:
    repeat = getattr(polymer, "canonical_repeat_unit", None) or getattr(
        polymer, "repeat_unit_smiles", None)
    return _oligomer_smiles_from_repeat(str(repeat), 3) if repeat else None


def _oligomer_smiles_from_repeat(repeat_unit: str, n_units: int) -> str | None:
    """Expand a repeat unit into a hydrogen-terminated oligomer.

    Reuses the engine's own oligomer builder so the chemistry matches every other part
    of the pipeline; capping a single repeat unit would type a different molecule.
    """
    try:
        from rdkit import Chem

        from polymer_engine.polymer.identity import build_oligomer
    except ImportError:
        return None
    built = build_oligomer(repeat_unit, n_units=max(2, min(n_units, 4)))
    if built is None:
        return None
    return str(Chem.MolToSmiles(built[0]))


__all__ = ["PARAMENV", "WORKER", "OpenFFBackend"]
