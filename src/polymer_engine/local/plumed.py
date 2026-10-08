"""PLUMED kernel discovery.

GROMACS does not link PLUMED at build time.  ``gmx mdrun -plumed`` dlopens the kernel
at run time from the path in ``PLUMED_KERNEL``, and if that variable is unset it aborts
with an *internal error* rather than a clean diagnostic::

    Internal error (bug):
    An error occurred while initializing the PLUMED force provider:
    You are trying to use plumed, but it is not available.
    Check your PLUMED_KERNEL environment variable.

That message is produced *after* grompp has succeeded and mdrun has started, so without
the checks in this module a campaign discovers the problem only once it has already
spent the setup cost -- and the failure looks like a GROMACS bug rather than a missing
environment variable.

This module locates the kernel before anything runs, so a PLUMED job that cannot
possibly work is refused up front with a reason a person can act on.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.logging import get_logger

logger = get_logger("local.plumed")

#: The environment variable GROMACS reads to find the kernel.
PLUMED_KERNEL_ENV = "PLUMED_KERNEL"

#: Shared-library names, in the order they are usually installed.
KERNEL_NAMES = ("libplumedKernel.so", "libplumedKernel.dylib")

#: Directories to try relative to the ``plumed`` executable's prefix.
_LIB_DIRS = ("lib", "lib64", "lib/plumed", "lib64/plumed")

#: ``plumed --version`` is not accepted by every build (the conda 2.9.x packages reject
#: it outright), but ``plumed info --version`` is stable across 2.5+.
VERSION_ARGS = ("info", "--version")
LONG_VERSION_ARGS = ("info", "--long-version")


@dataclass
class PlumedKernel:
    """Where the kernel is, and whether it can actually be used."""

    path: str | None = None
    found: bool = False
    readable: bool = False
    source: str = "unresolved"
    version: str | None = None
    executable: str | None = None
    is_installed: bool | None = None
    has_dlopen: bool | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Safe to hand to ``gmx mdrun -plumed``."""
        return self.found and self.readable

    def environment(self) -> dict[str, str]:
        """The environment overlay GROMACS needs.  Empty when unusable."""
        return {PLUMED_KERNEL_ENV: self.path} if self.usable and self.path else {}

    def reason(self) -> str:
        """Why this kernel cannot be used, or an empty string when it can."""
        if self.usable:
            return ""
        return "; ".join(self.issues) or "the PLUMED kernel could not be located"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "found": self.found,
            "readable": self.readable,
            "source": self.source,
            "version": self.version,
            "executable": self.executable,
            "is_installed": self.is_installed,
            "has_dlopen": self.has_dlopen,
            "usable": self.usable,
            "issues": list(self.issues),
        }


def _probe(executable: str, args: tuple[str, ...], timeout_s: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [executable, *args], capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - environment specific
        return 1, str(exc)
    return proc.returncode, f"{proc.stdout}\n{proc.stderr}"


def plumed_version(executable: str, *, timeout_s: float = 20.0) -> str | None:
    """Version via ``plumed info --version``.

    ``--version`` alone is deliberately not used: some builds reject it with
    ``ERROR: Unknown option --version`` and exit 0, which parses as "no version" while
    looking like a successful probe.
    """
    for args in (LONG_VERSION_ARGS, VERSION_ARGS):
        code, text = _probe(executable, args, timeout_s)
        if code != 0 or "Unknown option" in text:
            continue
        match = re.search(r"\bv?(\d+\.\d+(?:\.\d+)?)", text)
        if match:
            return match.group(1)
    return None


def _flag(executable: str, flag: str, timeout_s: float) -> bool | None:
    """``plumed --is-installed`` / ``--has-dlopen`` answer through the exit code."""
    code, text = _probe(executable, (flag,), timeout_s)
    if "Unknown option" in text:
        return None
    return code == 0


def _candidate_directories(executable: str | None) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    if executable:
        prefix = Path(executable).resolve().parent.parent
        candidates.extend((prefix / d, "executable-prefix") for d in _LIB_DIRS)
        code, text = _probe(executable, ("info", "--root"), 20.0)
        if code == 0:
            root = text.strip().splitlines()[0].strip() if text.strip() else ""
            if root:
                # ``--root`` points at <prefix>/lib/plumed; the kernel sits beside it.
                candidates.append((Path(root), "plumed-root"))
                candidates.append((Path(root).parent, "plumed-root-parent"))
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        candidates.extend((Path(conda) / d, "conda-prefix") for d in _LIB_DIRS)
    return candidates


def discover_plumed_kernel(
    *,
    executable: str | None = None,
    configured_path: str | Path | None = None,
    timeout_s: float = 20.0,
) -> PlumedKernel:
    """Locate ``libplumedKernel.so``.

    Search order, most explicit first: a configured path, then ``PLUMED_KERNEL`` in the
    environment, then directories derived from the ``plumed`` executable, then
    ``CONDA_PREFIX``.  An explicit setting that points at a missing file is an error,
    never a reason to fall through to a guess -- silently using a different kernel from
    the one that was asked for is how a run becomes unreproducible.
    """
    result = PlumedKernel()
    result.executable = executable or shutil.which("plumed")

    if result.executable:
        result.version = plumed_version(result.executable, timeout_s=timeout_s)
        result.is_installed = _flag(result.executable, "--is-installed", timeout_s)
        result.has_dlopen = _flag(result.executable, "--has-dlopen", timeout_s)
        if result.is_installed is False:
            result.issues.append("plumed --is-installed reports the installation is incomplete")
        if result.has_dlopen is False:
            result.issues.append(
                "plumed --has-dlopen reports no dlopen support, so GROMACS cannot load the kernel"
            )
    else:
        result.issues.append("no 'plumed' executable on PATH")

    explicit: tuple[str, str] | None = None
    if configured_path:
        explicit = (str(configured_path), "configured")
    elif os.environ.get(PLUMED_KERNEL_ENV):
        explicit = (os.environ[PLUMED_KERNEL_ENV], "environment")

    if explicit is not None:
        candidate = Path(explicit[0])
        result.path, result.source = str(candidate), explicit[1]
        result.found = candidate.is_file()
        result.readable = result.found and os.access(candidate, os.R_OK)
        if not result.found:
            result.issues.append(f"{explicit[1]} PLUMED kernel does not exist: {candidate}")
        elif not result.readable:
            result.issues.append(f"PLUMED kernel is not readable: {candidate}")
        return result

    for directory, source in _candidate_directories(result.executable):
        for name in KERNEL_NAMES:
            candidate = directory / name
            if candidate.is_file():
                result.path, result.source = str(candidate), source
                result.found = True
                result.readable = os.access(candidate, os.R_OK)
                if not result.readable:
                    result.issues.append(f"PLUMED kernel is not readable: {candidate}")
                logger.info("Found PLUMED kernel at %s (via %s)", candidate, source)
                return result

    result.issues.append(
        f"no {KERNEL_NAMES[0]} found near the plumed executable or CONDA_PREFIX; "
        f"set {PLUMED_KERNEL_ENV} or local_tools.plumed.kernel_path"
    )
    return result


__all__ = [
    "KERNEL_NAMES",
    "PLUMED_KERNEL_ENV",
    "PlumedKernel",
    "discover_plumed_kernel",
    "plumed_version",
]
