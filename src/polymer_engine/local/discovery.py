"""Local executable discovery.

The engine must never assume the developer's machine matches the CI machine, so
every tool is looked up at runtime and reported with:

* the path we resolved (configured path wins over ``PATH``)
* whether it exists *and* is executable -- those are different failures
* a parsed version, or an explicit ``UNKNOWN`` when the banner cannot be parsed
* build capabilities that change scientific behaviour (GPU support, MPI, precision)
* a compatibility verdict against the configured version range

Nothing here raises on a missing tool: absence is a reportable state, not a crash.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.config import EngineConfig, ToolConfig
from polymer_engine.core.errors import ToolNotFound, ToolVersionIncompatible
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination

logger = get_logger("local.discovery")

VERSION_PROBE_TIMEOUT_S = 30.0


@dataclass(frozen=True, slots=True)
class Version:
    """A dotted numeric version with a tolerant parser and total ordering."""

    parts: tuple[int, ...]
    raw: str

    @classmethod
    def parse(cls, text: str) -> Version | None:
        match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", text)
        if not match:
            return None
        parts = tuple(int(g) for g in match.groups() if g is not None)
        return cls(parts, text.strip())

    def _padded(self, length: int) -> tuple[int, ...]:
        return self.parts + (0,) * (length - len(self.parts))

    def __lt__(self, other: Version) -> bool:
        n = max(len(self.parts), len(other.parts))
        return self._padded(n) < other._padded(n)

    def __le__(self, other: Version) -> bool:
        return self == other or self < other

    def __str__(self) -> str:
        return ".".join(str(p) for p in self.parts)


@dataclass
class ToolStatus:
    """What we know about one local executable."""

    name: str
    requested: str
    path: str | None = None
    found: bool = False
    executable: bool = False
    version: str | None = None
    version_raw: str | None = None
    version_determination: Determination = Determination.UNKNOWN
    capabilities: dict[str, Any] = field(default_factory=dict)
    compatible: Determination = Determination.UNKNOWN
    min_version: str | None = None
    max_version: str | None = None
    source: str = "unresolved"
    issues: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Safe to run: present, executable, and not known-incompatible."""
        return self.found and self.executable and self.compatible is not Determination.REQUIRES_VALIDATION

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requested": self.requested,
            "path": self.path,
            "found": self.found,
            "executable": self.executable,
            "version": self.version,
            "version_determination": self.version_determination.value,
            "capabilities": self.capabilities,
            "compatible": self.compatible.value,
            "min_version": self.min_version,
            "max_version": self.max_version,
            "source": self.source,
            "issues": self.issues,
            "usable": self.usable,
        }

    def require(self) -> str:
        """Return the path, or raise with a specific reason."""
        if not self.found:
            raise ToolNotFound(
                f"{self.name} was not found",
                requested=self.requested,
                hint=f"set local_tools.{self.name}.path or add it to PATH",
            )
        if not self.executable:
            raise ToolNotFound(f"{self.name} exists but is not executable", path=self.path)
        if self.compatible is Determination.REQUIRES_VALIDATION:
            raise ToolVersionIncompatible(
                f"{self.name} version {self.version} is outside the supported range",
                path=self.path,
                min_version=self.min_version,
                max_version=self.max_version,
            )
        assert self.path is not None
        return self.path


# --------------------------------------------------------------------------
# Per-tool probes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolProbe:
    """How to ask one tool for its version and what to read out of the answer."""

    args: tuple[str, ...]
    parse: Callable[[str], tuple[str | None, dict[str, Any]]]
    #: Some tools (ORCA) print their banner and then exit non-zero when handed a
    #: flag they do not understand.  A parsed version is the success signal.
    tolerate_nonzero_exit: bool = False


def _parse_gromacs(text: str) -> tuple[str | None, dict[str, Any]]:
    """GROMACS prints a labelled block: ``GROMACS version:  2026.3`` etc."""
    capabilities: dict[str, Any] = {}
    version: str | None = None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if not value:
            continue
        if key == "gromacs version":
            version = value
        elif key == "precision":
            capabilities["precision"] = value
        elif key == "mpi library":
            capabilities["mpi"] = value
            capabilities["has_mpi"] = value.lower() not in {"none", "thread_mpi"}
            capabilities["thread_mpi"] = "thread_mpi" in value.lower()
        elif key == "gpu support":
            capabilities["gpu_support"] = value
            capabilities["has_gpu"] = value.lower() not in {"disabled", "none"}
        elif key == "simd instructions":
            capabilities["simd"] = value
        elif key == "gpu fft library":
            capabilities["gpu_fft"] = value
        elif key == "plumed support":
            # A build without this cannot run `mdrun -plumed` at all, however well
            # PLUMED itself is installed.
            capabilities["plumed_support"] = value
            capabilities["has_plumed"] = value.lower() not in {"disabled", "none", "off"}
    if version is None:
        # Fall back to the banner line ":-) GROMACS - gmx, 2026.3 (-:"
        match = re.search(r"GROMACS\s*-\s*\S+,\s*([0-9][^\s(]*)", text)
        if match:
            version = match.group(1)
    return version, capabilities


def _parse_orca(text: str) -> tuple[str | None, dict[str, Any]]:
    """ORCA prints ``Program Version 6.1.1  -  RELEASE`` inside its banner."""
    capabilities: dict[str, Any] = {}
    match = re.search(r"Program Version\s+([0-9][0-9.]*)", text)
    version = match.group(1) if match else None
    if re.search(r"\bMPI\b", text):
        capabilities["mpi_mentioned"] = True
    libxc = re.search(r"libXC version:\s*([0-9][0-9.]*)", text)
    if libxc:
        capabilities["libxc"] = libxc.group(1)
    return version, capabilities


def _parse_plumed(text: str) -> tuple[str | None, dict[str, Any]]:
    """PLUMED prints e.g. ``v2.9.0`` (sometimes with a git suffix).

    Probed through ``plumed info --version``.  Plain ``--version`` is not used: the
    conda 2.9.x builds answer it with ``ERROR: Unknown option --version`` **and exit 0**,
    so the probe looks successful while yielding no version at all.
    """
    if "Unknown option" in text:
        return None, {"version_probe_rejected": True}
    match = re.search(r"\bv?(\d+\.\d+(?:\.\d+)?)", text)
    return (match.group(1) if match else None), {}


def _parse_python(text: str) -> tuple[str | None, dict[str, Any]]:
    match = re.search(r"Python\s+([0-9][0-9.]*)", text)
    return (match.group(1) if match else None), {}


def _parse_generic(text: str) -> tuple[str | None, dict[str, Any]]:
    """Fallback probe: take the first dotted number we can find."""
    parsed = Version.parse(text)
    return (parsed.raw if parsed else None), {}


PROBES: dict[str, ToolProbe] = {
    "gromacs": ToolProbe(("--version",), _parse_gromacs),
    "orca": ToolProbe(("--version",), _parse_orca, tolerate_nonzero_exit=True),
    "plumed": ToolProbe(("info", "--long-version"), _parse_plumed),
    "python": ToolProbe(("--version",), _parse_python),
}


def _run_probe(path: str, args: Sequence[str], timeout_s: float) -> tuple[int, str]:
    proc = subprocess.run(
        [path, *args],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        # Keep a stray interactive prompt from hanging discovery.
        stdin=subprocess.DEVNULL,
    )
    return proc.returncode, f"{proc.stdout}\n{proc.stderr}"


def resolve_executable(tool: ToolConfig, *, env: dict[str, str] | None = None) -> tuple[str | None, str]:
    """Resolve a tool to an absolute path.

    A configured ``path`` wins over ``PATH`` unconditionally -- including when the
    configured path does not exist, so a typo surfaces instead of silently falling
    back to a different binary.
    """
    if tool.path is not None:
        return str(tool.path), "configured-path"
    env = env if env is not None else dict(os.environ)
    found = shutil.which(tool.executable, path=env.get("PATH"))
    if found:
        return found, "PATH"
    # An explicit relative/absolute executable name that is not on PATH.
    candidate = Path(tool.executable)
    if candidate.is_absolute() or os.sep in tool.executable:
        return str(candidate), "literal"
    return None, "unresolved"


def discover_tool(
    name: str,
    tool: ToolConfig,
    *,
    env: dict[str, str] | None = None,
    timeout_s: float = VERSION_PROBE_TIMEOUT_S,
    probe: bool = True,
) -> ToolStatus:
    """Locate one tool and describe it.  Never raises for a missing tool."""
    status = ToolStatus(
        name=name,
        requested=str(tool.path) if tool.path else tool.executable,
        min_version=tool.min_version,
        max_version=tool.max_version,
    )
    path, source = resolve_executable(tool, env=env)
    status.source = source
    if path is None:
        status.issues.append(f"{tool.executable!r} was not found on PATH and no path is configured")
        return status
    status.path = path

    candidate = Path(path)
    if not candidate.exists():
        status.issues.append(f"configured path does not exist: {path}")
        return status
    status.found = True
    if candidate.is_dir():
        status.issues.append(f"configured path is a directory, not an executable: {path}")
        return status
    if not os.access(path, os.X_OK):
        status.issues.append(f"file exists but is not executable (check permissions): {path}")
        return status
    status.executable = True

    if not probe:
        return status

    spec = PROBES.get(name, ToolProbe(("--version",), _parse_generic))
    try:
        returncode, output = _run_probe(path, spec.args, timeout_s)
    except subprocess.TimeoutExpired:
        status.issues.append(f"version probe timed out after {timeout_s}s")
        return status
    except OSError as exc:
        status.issues.append(f"could not execute {path}: {exc}")
        status.executable = False
        return status

    status.version_raw = output.strip()[:4000]
    version_text, capabilities = spec.parse(output)
    status.capabilities = capabilities

    if version_text is None:
        # Malformed or unrecognised banner: say so instead of inventing a version.
        status.version_determination = Determination.UNKNOWN
        status.issues.append("could not parse a version from the tool's output")
        if returncode != 0 and not spec.tolerate_nonzero_exit:
            status.issues.append(f"version probe exited with code {returncode}")
        return status

    status.version = version_text
    status.version_determination = Determination.KNOWN
    if returncode != 0 and not spec.tolerate_nonzero_exit:
        status.issues.append(f"version probe exited with code {returncode} but a version was parsed")

    status.compatible = _check_range(version_text, tool.min_version, tool.max_version, status.issues)
    return status


def _check_range(
    version_text: str, min_version: str | None, max_version: str | None, issues: list[str]
) -> Determination:
    if min_version is None and max_version is None:
        return Determination.KNOWN
    parsed = Version.parse(version_text)
    if parsed is None:
        issues.append(f"version {version_text!r} could not be compared against the supported range")
        return Determination.UNKNOWN
    if min_version is not None:
        floor = Version.parse(min_version)
        if floor is not None and parsed < floor:
            issues.append(f"version {version_text} is below the minimum supported {min_version}")
            return Determination.REQUIRES_VALIDATION
    if max_version is not None:
        ceiling = Version.parse(max_version)
        if ceiling is not None and ceiling < parsed:
            issues.append(f"version {version_text} is above the maximum supported {max_version}")
            return Determination.REQUIRES_VALIDATION
    return Determination.KNOWN


def discover_all(
    config: EngineConfig,
    *,
    env: dict[str, str] | None = None,
    probe: bool = True,
) -> dict[str, ToolStatus]:
    """Discover every configured tool."""
    tools = config.local_tools
    return {
        name: discover_tool(name, getattr(tools, name), env=env, probe=probe)
        for name in ("gromacs", "orca", "plumed", "python")
    }


def find_all_installations(executable: str, *, env: dict[str, str] | None = None) -> list[str]:
    """Every match for ``executable`` on ``PATH``, in precedence order.

    Multiple installations are a reproducibility hazard; the CLI surfaces them so an
    operator can pin one explicitly.
    """
    env = env if env is not None else dict(os.environ)
    seen: list[str] = []
    for directory in (env.get("PATH") or "").split(os.pathsep):
        if not directory:
            continue
        candidate = Path(directory) / executable
        if candidate.is_file() and os.access(candidate, os.X_OK):
            resolved = str(candidate)
            if resolved not in seen:
                seen.append(resolved)
    return seen


__all__ = [
    "PROBES",
    "ToolProbe",
    "ToolStatus",
    "Version",
    "discover_all",
    "discover_tool",
    "find_all_installations",
    "resolve_executable",
]
