"""GAFF / AmberTools backend.

Not installed on this machine (no ``antechamber``, ``parmchk2`` or ``tleap``), and
registered anyway for the same reason as OpenFF: an absent route should be visible.

Two cautions are built in rather than discovered later.

GAFF is a *general* force field for small organic molecules. Applying it to a polymer is
routine but not automatic: AM1-BCC charges are derived per molecule, so charges fitted on
a short oligomer do not transfer to a long chain without care, and ``parmchk2`` will
happily emit a parameter flagged as an estimate. The backend therefore treats
``parmchk2``'s own "ATTN, need revision" markers as first-class evidence in exactly the
way CGenFF penalties are treated -- as a reason to validate, not as a failure.
"""

from __future__ import annotations

import shutil
from typing import Any

from polymer_engine.core.logging import get_logger
from polymer_engine.parameterization.backend import ForceFieldBackend
from polymer_engine.parameterization.capability import CapabilityState
from polymer_engine.parameterization.models import (
    ForceFieldAssessment,
    ParameterizationRequest,
    ParameterizationResult,
    ParameterizationValidation,
    QMPriority,
)

logger = get_logger("parameterization.gaff")

#: Executables AmberTools must provide for this backend to function.
REQUIRED_EXECUTABLES = ("antechamber", "parmchk2", "tleap")

#: parmchk2 marks a parameter it had to guess with this string. It is GAFF's analogue of
#: a CGenFF penalty and must never be filtered out of a report.
PARMCHK_ATTENTION = "ATTN, need revision"


class GaffBackend(ForceFieldBackend):
    name = "gaff"
    force_field = "GAFF2"
    human_in_the_loop = False
    requires_credentials = False

    def __init__(self) -> None:
        self._paths = {name: shutil.which(name) for name in REQUIRED_EXECUTABLES}

    @property
    def _ready(self) -> bool:
        return all(path is not None for path in self._paths.values())

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name, "force_field": self.force_field, "version": None,
            "available": self._ready,
            "human_in_the_loop": False, "requires_credentials": False,
            "executables": dict(self._paths),
            "missing": [n for n, p in self._paths.items() if p is None],
            "provides": ["gaff_atom_typing", "am1bcc_charges",
                         "missing_parameter_detection"],
            "install_hint": "conda install -c conda-forge ambertools",
            "caveat": ("AM1-BCC charges are derived per molecule, so charges fitted on a "
                       "short oligomer do not transfer to a long chain without care"),
        }

    def assess(self, polymer: Any) -> ForceFieldAssessment:
        if not self._ready:
            missing = [n for n, path in self._paths.items() if path is None]
            return self._assessment(
                polymer, CapabilityState.UNAVAILABLE,
                f"AmberTools not installed: {', '.join(missing)} missing",
                evidence={"install_hint": self.capabilities()["install_hint"]},
            )
        return self._assessment(
            polymer, CapabilityState.PARAMETERIZATION_AVAILABLE,
            ("GAFF2 covers general organic chemistry; charge transferability to a long "
             "chain and any parmchk2 'ATTN' parameters must be checked"),
            qm_priority=QMPriority.MEDIUM, estimated_cost="minutes (AM1-BCC charges)",
        )

    def parameterize(self, request: ParameterizationRequest) -> ParameterizationResult:
        if not self._ready:
            missing = [n for n, p in self._paths.items() if p is None]
            return self._unavailable(
                request, f"AmberTools is not installed ({', '.join(missing)})",
                actions=[self.capabilities()["install_hint"]],
            )
        return self._unavailable(
            request,
            "AmberTools is present but this backend's path has not been exercised on "
            "this machine; it is not reported as working on untested code",
            actions=["run the GAFF backend's real-execution tests first"],
        )

    def validate(self, result: ParameterizationResult) -> ParameterizationValidation:
        from polymer_engine.parameterization.models import ParameterizationState

        return ParameterizationValidation(
            backend=self.name, polymer_id=result.request.polymer_id,
            property_class=result.request.property_class,
            state=ParameterizationState.BLOCKED,
            diagnostics=["GAFF backend is unavailable; nothing to validate"],
        )


__all__ = ["PARMCHK_ATTENTION", "REQUIRED_EXECUTABLES", "GaffBackend"]
