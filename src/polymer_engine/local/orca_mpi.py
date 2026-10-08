"""ORCA's MPI environment.

ORCA does not link MPI.  A job with ``%pal nprocs > 1`` shells out to ``mpirun``, so
without the matching OpenMPI on ``PATH`` the run fails *inside* ORCA, after the input
has been written and the parent process has started::

    ORCA finished by error termination in PROPERTIES
    Calling Command: mpirun -np 8 .../orca_prop_mpi ...

That is a configuration problem wearing a chemistry error's clothes, and it costs the
whole setup before it shows up.  This module finds the OpenMPI that ORCA was built
against so a parallel job either gets a working environment or is refused up front.

The layout it looks for is the one ORCA's own installer writes: an ``orca_paths.env``
beside the executable naming ``ORCA_MPI_PREFIX``.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.logging import get_logger

logger = get_logger("local.orca_mpi")


@dataclass
class OrcaMpi:
    """Where ORCA's MPI lives, and whether a parallel job can actually start."""

    mpirun: str | None = None
    prefix: str | None = None
    source: str = "unresolved"
    issues: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.mpirun)

    def environment(self) -> dict[str, str]:
        """PATH/LD_LIBRARY_PATH overlay so ORCA can find ``mpirun`` and its libraries."""
        if not self.usable or not self.prefix:
            return {}
        prefix = Path(self.prefix)
        return {
            "PATH": os.pathsep.join([str(prefix / "bin"), os.environ.get("PATH", "")]),
            "LD_LIBRARY_PATH": os.pathsep.join(
                [str(prefix / "lib"), os.environ.get("LD_LIBRARY_PATH", "")]
            ),
        }

    def reason(self) -> str:
        if self.usable:
            return ""
        return "; ".join(self.issues) or "no mpirun could be located for ORCA"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mpirun": self.mpirun, "prefix": self.prefix, "source": self.source,
            "usable": self.usable, "issues": list(self.issues),
        }


def _prefix_from_env_file(orca_executable: str) -> str | None:
    """Read ``ORCA_MPI_PREFIX`` from the ``orca_paths.env`` an ORCA install ships."""
    orca_dir = Path(orca_executable).resolve().parent
    for candidate in (orca_dir / "orca_paths.env", orca_dir.parent / "orca_paths.env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "ORCA_MPI_PREFIX" and value.strip():
                return value.strip()
    return None


def discover_orca_mpi(orca_executable: str | None = None) -> OrcaMpi:
    """Locate the OpenMPI ORCA should use.

    Order: ``mpirun`` already on ``PATH``, then ``ORCA_MPI_PREFIX``, then the
    ``orca_paths.env`` beside the executable.
    """
    result = OrcaMpi()
    executable = orca_executable or shutil.which("orca")

    on_path = shutil.which("mpirun")
    if on_path:
        result.mpirun, result.source = on_path, "PATH"
        result.prefix = str(Path(on_path).resolve().parent.parent)
        return result

    prefix = os.environ.get("ORCA_MPI_PREFIX") or (
        _prefix_from_env_file(executable) if executable else None
    )
    if prefix:
        candidate = Path(prefix) / "bin" / "mpirun"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            result.mpirun, result.prefix = str(candidate), str(prefix)
            result.source = "orca_paths.env" if not os.environ.get("ORCA_MPI_PREFIX") else "env"
            logger.info("Found ORCA's mpirun at %s", candidate)
            return result
        result.issues.append(f"ORCA_MPI_PREFIX is set to {prefix} but {candidate} is not executable")
    else:
        result.issues.append(
            "no mpirun on PATH, no ORCA_MPI_PREFIX, and no orca_paths.env beside the executable"
        )
    return result


__all__ = ["OrcaMpi", "discover_orca_mpi"]
