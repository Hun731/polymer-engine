"""Local tool discovery across every environment shape we care about."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from polymer_engine.core.config import ToolConfig, load_config
from polymer_engine.core.errors import ToolNotFound, ToolVersionIncompatible
from polymer_engine.core.models import Determination
from polymer_engine.local.discovery import (
    Version,
    discover_all,
    discover_tool,
    find_all_installations,
    resolve_executable,
)

GROMACS_BANNER = """                         :-) GROMACS - gmx, 2026.3 (-:

GROMACS version:     2026.3
Precision:           mixed
MPI library:         thread_mpi
GPU support:         CUDA
SIMD instructions:   AVX_512
GPU FFT library:     cuFFT
"""

ORCA_BANNER = """
                                 *****************
                                 * O   R   C   A *
                                 *****************
                         Program Version 6.1.1  -  RELEASE  -
 Your ORCA version has been built with support for libXC version: 7.0.0
"""


def make_fake_tool(directory: Path, name: str, stdout: str, *, exit_code: int = 0, executable: bool = True) -> Path:
    path = directory / name
    path.write_text(f"#!/bin/sh\ncat <<'BANNER'\n{stdout}\nBANNER\nexit {exit_code}\n")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR if executable else mode & ~stat.S_IXUSR)
    return path


# ==========================================================================
# Version parsing and comparison
# ==========================================================================
class TestVersion:
    @pytest.mark.parametrize(
        "text,expected",
        [("2026.3", (2026, 3)), ("v2.9.0", (2.9 and 2, 9, 0)), ("6.1.1", (6, 1, 1)), ("3", (3,))],
    )
    def test_parse(self, text: str, expected: tuple) -> None:
        parsed = Version.parse(text)
        assert parsed is not None
        assert parsed.parts == expected

    def test_unparseable_returns_none(self) -> None:
        assert Version.parse("no numbers here") is None

    def test_ordering_pads_missing_components(self) -> None:
        assert Version.parse("2.9") < Version.parse("2.9.1")
        assert Version.parse("2021") < Version.parse("2026.3")
        assert not (Version.parse("2026.3") < Version.parse("2026.3"))

    def test_le(self) -> None:
        assert Version.parse("1.0") <= Version.parse("1.0")
        assert Version.parse("1.0") <= Version.parse("1.1")


# ==========================================================================
# Resolution
# ==========================================================================
class TestResolution:
    def test_path_lookup(self, tmp_path: Path) -> None:
        make_fake_tool(tmp_path, "gmx", GROMACS_BANNER)
        path, source = resolve_executable(ToolConfig(executable="gmx"), env={"PATH": str(tmp_path)})
        assert path == str(tmp_path / "gmx")
        assert source == "PATH"

    def test_configured_path_overrides_path_lookup(self, tmp_path: Path) -> None:
        on_path = tmp_path / "bin"
        on_path.mkdir()
        make_fake_tool(on_path, "gmx", GROMACS_BANNER)
        pinned = make_fake_tool(tmp_path, "gmx-pinned", GROMACS_BANNER)

        status = discover_tool(
            "gromacs", ToolConfig(executable="gmx", path=pinned), env={"PATH": str(on_path)}
        )
        assert status.path == str(pinned)
        assert status.source == "configured-path"

    def test_missing_configured_path_does_not_fall_back_to_path(self, tmp_path: Path) -> None:
        on_path = tmp_path / "bin"
        on_path.mkdir()
        make_fake_tool(on_path, "gmx", GROMACS_BANNER)

        status = discover_tool(
            "gromacs",
            ToolConfig(executable="gmx", path=tmp_path / "does-not-exist"),
            env={"PATH": str(on_path)},
        )
        assert status.found is False
        assert "does not exist" in status.issues[0]

    def test_absent_executable(self, tmp_path: Path) -> None:
        status = discover_tool("plumed", ToolConfig(executable="plumed"), env={"PATH": str(tmp_path)})
        assert status.found is False
        assert status.usable is False
        assert "was not found" in status.issues[0]

    def test_present_but_not_executable(self, tmp_path: Path) -> None:
        path = tmp_path / "gmx"
        path.write_text("#!/bin/sh\necho hi\n")
        path.chmod(0o644)
        status = discover_tool("gromacs", ToolConfig(executable="gmx", path=path))
        assert status.found is True
        assert status.executable is False
        assert "not executable" in status.issues[0]

    def test_directory_is_not_an_executable(self, tmp_path: Path) -> None:
        directory = tmp_path / "gmx"
        directory.mkdir()
        status = discover_tool("gromacs", ToolConfig(executable="gmx", path=directory))
        assert status.executable is False
        assert "directory" in status.issues[0]

    def test_multiple_installations_are_all_reported(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        make_fake_tool(first, "gmx", GROMACS_BANNER)
        make_fake_tool(second, "gmx", GROMACS_BANNER)
        found = find_all_installations("gmx", env={"PATH": f"{first}{os.pathsep}{second}"})
        assert found == [str(first / "gmx"), str(second / "gmx")]
        assert found[0].startswith(str(first)), "PATH precedence order must be preserved"


# ==========================================================================
# Version probing
# ==========================================================================
class TestProbing:
    def test_gromacs_version_and_capabilities(self, tmp_path: Path) -> None:
        make_fake_tool(tmp_path, "gmx", GROMACS_BANNER)
        status = discover_tool("gromacs", ToolConfig(executable="gmx"), env={"PATH": str(tmp_path)})
        assert status.version == "2026.3"
        assert status.version_determination is Determination.KNOWN
        assert status.capabilities["has_gpu"] is True
        assert status.capabilities["gpu_support"] == "CUDA"
        assert status.capabilities["precision"] == "mixed"
        assert status.capabilities["thread_mpi"] is True
        assert status.capabilities["has_mpi"] is False
        assert status.usable is True

    def test_gromacs_without_gpu_support(self, tmp_path: Path) -> None:
        make_fake_tool(tmp_path, "gmx", GROMACS_BANNER.replace("GPU support:         CUDA", "GPU support:         disabled"))
        status = discover_tool("gromacs", ToolConfig(executable="gmx"), env={"PATH": str(tmp_path)})
        assert status.capabilities["has_gpu"] is False

    def test_orca_banner_with_nonzero_exit_is_tolerated(self, tmp_path: Path) -> None:
        make_fake_tool(tmp_path, "orca", ORCA_BANNER, exit_code=1)
        status = discover_tool("orca", ToolConfig(executable="orca"), env={"PATH": str(tmp_path)})
        assert status.version == "6.1.1"
        assert status.capabilities["libxc"] == "7.0.0"
        assert status.issues == [], "ORCA exiting non-zero for --version is expected, not an issue"

    def test_plumed_version(self, tmp_path: Path) -> None:
        make_fake_tool(tmp_path, "plumed", "v2.9.0")
        status = discover_tool("plumed", ToolConfig(executable="plumed", min_version="2.7"), env={"PATH": str(tmp_path)})
        assert status.version == "2.9.0"
        assert status.compatible is Determination.KNOWN

    def test_malformed_version_output_is_unknown_not_invented(self, tmp_path: Path) -> None:
        make_fake_tool(tmp_path, "gmx", "corrupted binary garbage")
        status = discover_tool("gromacs", ToolConfig(executable="gmx"), env={"PATH": str(tmp_path)})
        assert status.version is None
        assert status.version_determination is Determination.UNKNOWN
        assert "could not parse a version" in status.issues[0]

    def test_probe_can_be_skipped(self, tmp_path: Path) -> None:
        make_fake_tool(tmp_path, "gmx", GROMACS_BANNER)
        status = discover_tool("gromacs", ToolConfig(executable="gmx"), env={"PATH": str(tmp_path)}, probe=False)
        assert status.found and status.executable
        assert status.version is None

    def test_probe_timeout_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "gmx"
        path.write_text("#!/bin/sh\nsleep 10\n")
        path.chmod(0o755)
        status = discover_tool("gromacs", ToolConfig(executable="gmx", path=path), timeout_s=0.3)
        assert status.version is None
        assert "timed out" in status.issues[0]


# ==========================================================================
# Version compatibility
# ==========================================================================
class TestCompatibility:
    def test_below_minimum_is_incompatible(self, tmp_path: Path) -> None:
        make_fake_tool(tmp_path, "gmx", GROMACS_BANNER.replace("2026.3", "2018.1"))
        status = discover_tool(
            "gromacs", ToolConfig(executable="gmx", min_version="2021"), env={"PATH": str(tmp_path)}
        )
        assert status.compatible is Determination.REQUIRES_VALIDATION
        assert status.usable is False
        assert "below the minimum" in status.issues[0]

    def test_above_maximum_is_incompatible(self, tmp_path: Path) -> None:
        make_fake_tool(tmp_path, "gmx", GROMACS_BANNER)
        status = discover_tool(
            "gromacs", ToolConfig(executable="gmx", max_version="2024"), env={"PATH": str(tmp_path)}
        )
        assert status.compatible is Determination.REQUIRES_VALIDATION
        assert "above the maximum" in status.issues[0]

    def test_no_declared_range_is_compatible(self, tmp_path: Path) -> None:
        make_fake_tool(tmp_path, "orca", ORCA_BANNER)
        status = discover_tool("orca", ToolConfig(executable="orca"), env={"PATH": str(tmp_path)})
        assert status.compatible is Determination.KNOWN


# ==========================================================================
# require()
# ==========================================================================
class TestRequire:
    def test_require_raises_tool_not_found(self, tmp_path: Path) -> None:
        status = discover_tool("plumed", ToolConfig(executable="plumed"), env={"PATH": str(tmp_path)})
        with pytest.raises(ToolNotFound):
            status.require()

    def test_require_raises_on_incompatible_version(self, tmp_path: Path) -> None:
        make_fake_tool(tmp_path, "gmx", GROMACS_BANNER.replace("2026.3", "2018.1"))
        status = discover_tool(
            "gromacs", ToolConfig(executable="gmx", min_version="2021"), env={"PATH": str(tmp_path)}
        )
        with pytest.raises(ToolVersionIncompatible):
            status.require()

    def test_require_returns_the_path_when_healthy(self, tmp_path: Path) -> None:
        make_fake_tool(tmp_path, "gmx", GROMACS_BANNER)
        status = discover_tool("gromacs", ToolConfig(executable="gmx"), env={"PATH": str(tmp_path)})
        assert status.require() == str(tmp_path / "gmx")


def test_discover_all_covers_every_configured_tool(tmp_path: Path) -> None:
    config = load_config(discover=False, use_env=False)
    statuses = discover_all(config, env={"PATH": str(tmp_path)}, probe=False)
    assert set(statuses) == {"gromacs", "orca", "plumed", "python"}
