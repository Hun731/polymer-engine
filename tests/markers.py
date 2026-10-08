"""Skip helpers that state *why* a test was skipped.

Charter rule: hardware/optional-dependency tests are skipped with an explicit
reason, never silently.
"""

from __future__ import annotations

import importlib.util
import shutil

import pytest


def _module_missing(name: str) -> bool:
    return importlib.util.find_spec(name) is None


def _tool_missing(name: str) -> bool:
    return shutil.which(name) is None


requires_rdkit = pytest.mark.skipif(
    _module_missing("rdkit"),
    reason="RDKit is not installed; chemistry descriptor tests cannot run in this environment",
)
requires_mdanalysis = pytest.mark.skipif(
    _module_missing("MDAnalysis"),
    reason="MDAnalysis is not installed; trajectory analysis tests cannot run in this environment",
)
requires_sklearn = pytest.mark.skipif(
    _module_missing("sklearn"),
    reason="scikit-learn is not installed; surrogate-model tests cannot run in this environment",
)
requires_scipy = pytest.mark.skipif(
    _module_missing("scipy"),
    reason="SciPy is not installed; tests needing its statistical routines cannot run",
)
requires_gromacs = pytest.mark.skipif(
    _tool_missing("gmx"),
    reason="GROMACS 'gmx' is not on PATH; real-execution tests cannot run in this environment",
)
requires_orca = pytest.mark.skipif(
    _tool_missing("orca"),
    reason="ORCA is not on PATH; real-execution tests cannot run in this environment",
)
requires_plumed = pytest.mark.skipif(
    _tool_missing("plumed"),
    reason="PLUMED is not on PATH; real-execution tests cannot run in this environment",
)
