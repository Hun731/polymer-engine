"""What a force-field backend can actually do, and how far a polymer has got with it.

Capability is not a boolean.  A backend can build a structure but have no parameters
for it; it can produce parameters that have never been validated; it can be perfectly
capable and simply not installed.  Collapsing those into "available: true/false" is how
a pipeline ends up simulating a chemistry nobody checked.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from polymer_engine.core.logging import get_logger

logger = get_logger("parameterization.capability")


class CapabilityState(str, Enum):
    """How far a backend can carry a given polymer.

    The order matters: each state presupposes the ones above it, and
    :meth:`at_least` relies on that.  ``BLOCKED`` and ``REQUIRES_EXPERT_REVIEW`` sit
    outside the ladder because they are verdicts, not progress.
    """

    #: The backend is not installed, or its credentials are absent.
    UNAVAILABLE = "UNAVAILABLE"
    #: It can represent the molecule, but cannot assign parameters.
    STRUCTURE_SUPPORTED = "STRUCTURE_SUPPORTED"
    #: It can assign parameters for this chemistry.
    PARAMETERIZATION_AVAILABLE = "PARAMETERIZATION_AVAILABLE"
    #: It can also produce a simulation-ready system.
    SYSTEM_BUILD_AVAILABLE = "SYSTEM_BUILD_AVAILABLE"
    #: The parameters have been checked against an independent reference.
    PARAMETERS_VALIDATED = "PARAMETERS_VALIDATED"
    #: Validated *and* qualified for a stated property class.
    QUALIFIED = "QUALIFIED"
    #: This chemistry is out of scope for this backend, and saying so is the answer.
    BLOCKED = "BLOCKED"
    #: A person has to decide; the engine will not.
    REQUIRES_EXPERT_REVIEW = "REQUIRES_EXPERT_REVIEW"


#: The progress ladder, in order.  Verdict states are deliberately excluded.
CAPABILITY_LADDER: tuple[CapabilityState, ...] = (
    CapabilityState.UNAVAILABLE,
    CapabilityState.STRUCTURE_SUPPORTED,
    CapabilityState.PARAMETERIZATION_AVAILABLE,
    CapabilityState.SYSTEM_BUILD_AVAILABLE,
    CapabilityState.PARAMETERS_VALIDATED,
    CapabilityState.QUALIFIED,
)


def rank(state: CapabilityState) -> int:
    """Position on the ladder; verdict states rank below everything."""
    try:
        return CAPABILITY_LADDER.index(state)
    except ValueError:
        return -1


def at_least(state: CapabilityState, floor: CapabilityState) -> bool:
    """True when ``state`` is at or above ``floor`` on the ladder.

    A verdict state is never "at least" anything: ``BLOCKED`` is not progress.
    """
    return rank(state) >= 0 and rank(state) >= rank(floor)


@dataclass
class ToolStatus:
    """One external tool, as actually found on this machine."""

    name: str
    path: str | None = None
    version: str | None = None
    backend: str = ""
    capabilities: list[str] = field(default_factory=list)
    kind: str = "executable"
    notes: str = ""

    @property
    def available(self) -> bool:
        return self.path is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name, "path": self.path, "version": self.version,
            "backend": self.backend, "capabilities": list(self.capabilities),
            "kind": self.kind, "available": self.available, "notes": self.notes,
        }


#: Executables that a parameterization backend might need, and what they belong to.
EXECUTABLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("gmx", "gromacs", ("system_build", "energy_evaluation", "topology_check")),
    ("orca", "orca", ("qm_reference", "torsion_scan", "geometry_optimisation")),
    ("plumed", "plumed", ("enhanced_sampling",)),
    ("obabel", "openbabel", ("format_conversion", "3d_generation")),
    ("packmol", "packmol", ("melt_packing",)),
    ("tleap", "ambertools", ("system_build",)),
    ("antechamber", "ambertools", ("gaff_atom_typing", "am1bcc_charges")),
    ("parmchk2", "ambertools", ("missing_parameter_detection",)),
    ("cgenff", "cgenff", ("cgenff_atom_typing", "penalty_scores")),
    ("charmm", "charmm", ("system_build",)),
    ("acpype", "acpype", ("amber_to_gromacs",)),
)

#: Importable modules that a backend might need.
MODULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("rdkit", "rdkit", ("molecule_construction", "descriptors", "conformers")),
    ("openff.toolkit", "openff", ("molecule_construction", "parameter_assignment")),
    ("openff.interchange", "openff", ("system_export",)),
    ("openmm", "openmm", ("energy_evaluation",)),
    ("parmed", "parmed", ("topology_conversion",)),
    ("MDAnalysis", "mdanalysis", ("trajectory_analysis",)),
)


def _executable_version(path: str, name: str) -> str | None:
    """Best-effort version probe.  Never raises; an unknown version is not a failure."""
    attempts: tuple[tuple[str, ...], ...]
    if name == "gmx":
        attempts = (("--version",),)
    elif name == "orca":
        attempts = ((),)
    elif name == "plumed":
        attempts = (("info", "--version"),)
    else:
        attempts = (("--version",), ("-v",), ("--help",))
    for args in attempts:
        try:
            proc = subprocess.run([path, *args], capture_output=True, text=True,
                                  timeout=20, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        text = f"{proc.stdout}\n{proc.stderr}"
        import re

        for pattern in (r"GROMACS version:\s*(\S+)", r"Program Version\s+([0-9][0-9.]*)",
                        r"\bv?(\d+\.\d+(?:\.\d+)?)"):
            match = re.search(pattern, text)
            if match:
                return match.group(1)
    return None


def _module_version(module: str) -> str | None:
    import importlib

    try:
        loaded = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - a broken optional import is "not available"
        return None
    for attribute in ("__version__", "version"):
        value = getattr(loaded, attribute, None)
        if isinstance(value, str):
            return value
    return "unknown"


def discover_tools() -> list[ToolStatus]:
    """Probe the machine for every tool a backend might use.

    Everything here is measured, not read from configuration: a package listed in a
    requirements file that is not importable is not available.
    """
    found: list[ToolStatus] = []
    for name, backend, capabilities in EXECUTABLES:
        path = shutil.which(name)
        found.append(ToolStatus(
            name=name, path=path, backend=backend, capabilities=list(capabilities),
            version=_executable_version(path, name) if path else None,
        ))
    for module, backend, capabilities in MODULES:
        version = _module_version(module)
        found.append(ToolStatus(
            name=module, path=module if version else None, version=version,
            backend=backend, capabilities=list(capabilities), kind="python-module",
        ))
    found.extend(_browser_tools())
    logger.info("Discovered %d of %d parameterization tools",
                sum(1 for t in found if t.available), len(found))
    return found


#: Capabilities the browser subsystem exposes to the autonomous engine (§60). They are
#: reported only when the tooling is genuinely present -- a declared capability that
#: cannot execute is worse than an absent one, because the engine would plan around it.
BROWSER_CAPABILITIES: tuple[str, ...] = (
    "CHARMM_GUI_CATALOG", "CHARMM_GUI_BUILD_SPEC", "CHARMM_GUI_BROWSER_BUILD",
    "CHARMM_GUI_JOB_STATUS", "CHARMM_GUI_DOWNLOAD", "CHARMM_GUI_IMPORT",
)


def _browser_tools() -> list[ToolStatus]:
    """Probe the isolated browser environment by actually launching Chromium.

    Importing Playwright is not enough: the Python package installs without the browser
    binaries, and a session that discovers this at login time has already spent a
    credential on it.
    """
    from polymer_engine.browser.driver import BROWSER_ENV, WorkerDriver

    probe = WorkerDriver.capabilities()
    usable = bool(probe.get("available"))
    note = "" if usable else str(probe.get("error", "browser tooling not installed"))
    return [
        ToolStatus(
            name="playwright", path=str(BROWSER_ENV) if probe.get("playwright") else None,
            version=probe.get("playwright"), backend="charmm_gui",
            capabilities=["browser-automation"], kind="python-module",
            notes=note or f"isolated environment at {BROWSER_ENV}",
        ),
        ToolStatus(
            name="chromium", path=str(BROWSER_ENV / "browsers") if usable else None,
            version=probe.get("chromium"), backend="charmm_gui",
            capabilities=list(BROWSER_CAPABILITIES), kind="browser",
            notes=note or "drives the visible Polymer Builder interface; the documented "
                          "API is used for job status and download",
        ),
    ]


def summarise(tools: list[ToolStatus]) -> dict[str, Any]:
    by_backend: dict[str, dict[str, Any]] = {}
    for tool in tools:
        entry = by_backend.setdefault(
            tool.backend, {"available": [], "missing": [], "complete": False}
        )
        (entry["available"] if tool.available else entry["missing"]).append(tool.name)
    for entry in by_backend.values():
        entry["complete"] = not entry["missing"]
    return {
        "n_tools": len(tools),
        "n_available": sum(1 for t in tools if t.available),
        "backends": by_backend,
        "tools": [t.as_dict() for t in tools],
    }


__all__ = [
    "BROWSER_CAPABILITIES", "CAPABILITY_LADDER", "EXECUTABLES", "MODULES",
    "CapabilityState", "ToolStatus",
    "at_least", "discover_tools", "rank", "summarise",
]
