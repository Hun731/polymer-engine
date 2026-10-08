"""Runner behaviour, especially the rule that a dry run is never a success."""

from __future__ import annotations

import stat
from pathlib import Path

from polymer_engine.core.config import ToolConfig
from polymer_engine.local.discovery import ToolStatus, discover_tool
from polymer_engine.local.runner import GROMACSRunner, PLUMEDRunner
from tests.markers import requires_gromacs, requires_orca


def script(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def usable_status(path: Path, *, gpu: bool = False) -> ToolStatus:
    return ToolStatus(
        name="gromacs",
        requested=str(path),
        path=str(path),
        found=True,
        executable=True,
        version="2026.3",
        capabilities={"has_gpu": gpu},
    )


class TestDryRunSemantics:
    def test_disabled_runner_does_not_execute(self, tmp_path: Path) -> None:
        marker = tmp_path / "ran"
        path = script(tmp_path, "gmx", f"touch {marker}")
        result = GROMACSRunner(usable_status(path), enabled=False).run(["mdrun"])
        assert result.mode == "dry_run"
        assert result.succeeded is False, "a dry run must never report success"
        assert result.executed is False
        assert result.returncode is None
        assert not marker.exists(), "nothing may actually run while disabled"

    def test_enabled_runner_executes(self, tmp_path: Path) -> None:
        marker = tmp_path / "ran"
        path = script(tmp_path, "gmx", f"touch {marker}\nexit 0")
        result = GROMACSRunner(usable_status(path), enabled=True).run(["mdrun"], cwd=tmp_path)
        assert result.mode == "real"
        assert result.succeeded is True
        assert marker.exists()

    def test_nonzero_exit_is_not_success(self, tmp_path: Path) -> None:
        path = script(tmp_path, "gmx", "echo boom >&2\nexit 3")
        result = GROMACSRunner(usable_status(path), enabled=True).run(["grompp"], cwd=tmp_path)
        assert result.returncode == 3
        assert result.succeeded is False
        assert "boom" in result.stderr

    def test_unusable_tool_reports_unavailable_not_success(self, tmp_path: Path) -> None:
        status = ToolStatus(name="plumed", requested="plumed", issues=["not installed"])
        result = PLUMEDRunner(status, enabled=True).version()
        assert result.mode == "unavailable"
        assert result.succeeded is False
        assert "not installed" in (result.error or "")

    def test_timeout_is_flagged_and_not_success(self, tmp_path: Path) -> None:
        path = script(tmp_path, "gmx", "sleep 10")
        result = GROMACSRunner(usable_status(path), enabled=True).run(["mdrun"], cwd=tmp_path, timeout_s=0.3)
        assert result.timed_out is True
        assert result.succeeded is False
        assert result.mode == "real"


class TestGromacsCommands:
    def test_grompp_builds_the_expected_command(self, tmp_path: Path) -> None:
        path = script(tmp_path, "gmx", "exit 0")
        runner = GROMACSRunner(usable_status(path), enabled=False)
        result = runner.grompp(
            mdp="em.mdp", structure="system.gro", topology="topol.top", output="em.tpr", cwd=tmp_path
        )
        assert result.command[1:] == [
            "grompp", "-f", "em.mdp", "-c", "system.gro", "-p", "topol.top", "-o", "em.tpr",
        ]
        assert "-maxwarn" not in result.command, "warnings must not be suppressed by default"

    def test_maxwarn_is_opt_in(self, tmp_path: Path) -> None:
        path = script(tmp_path, "gmx", "exit 0")
        runner = GROMACSRunner(usable_status(path), enabled=False)
        result = runner.grompp(
            mdp="a.mdp", structure="b.gro", topology="c.top", output="d.tpr", cwd=tmp_path, max_warnings=2
        )
        assert result.command[-2:] == ["-maxwarn", "2"]

    def test_mdrun_without_gpu_flags_by_default(self, tmp_path: Path) -> None:
        path = script(tmp_path, "gmx", "exit 0")
        result = GROMACSRunner(usable_status(path), enabled=False).mdrun(deffnm="nvt", cwd=tmp_path)
        assert result.command[1:] == ["mdrun", "-deffnm", "nvt"]

    def test_mdrun_gpu_request_is_refused_on_a_cpu_only_build(self, tmp_path: Path) -> None:
        path = script(tmp_path, "gmx", "exit 0")
        runner = GROMACSRunner(usable_status(path, gpu=False), enabled=True)
        result = runner.mdrun(deffnm="nvt", cwd=tmp_path, use_gpu=True)
        assert result.mode == "unavailable"
        assert "no GPU support" in (result.error or "")

    def test_mdrun_gpu_flags_appear_on_a_gpu_build(self, tmp_path: Path) -> None:
        path = script(tmp_path, "gmx", "exit 0")
        runner = GROMACSRunner(usable_status(path, gpu=True), enabled=False)
        result = runner.mdrun(deffnm="prod", cwd=tmp_path, use_gpu=True)
        assert "-nb" in result.command and "gpu" in result.command

    def test_checkpoint_continuation_uses_noappend_by_default(self, tmp_path: Path) -> None:
        path = script(tmp_path, "gmx", "exit 0")
        runner = GROMACSRunner(usable_status(path), enabled=False)
        result = runner.mdrun(deffnm="prod", cwd=tmp_path, checkpoint="prod.cpt")
        assert "-cpi" in result.command
        assert "-noappend" in result.command

    def test_energy_pipes_terms_on_stdin(self, tmp_path: Path) -> None:
        captured = tmp_path / "stdin.txt"
        path = script(tmp_path, "gmx", f"cat > {captured}\nexit 0")
        runner = GROMACSRunner(usable_status(path), enabled=True)
        runner.energy(edr="npt.edr", terms=["Density", "Temperature"], output="density.xvg", cwd=tmp_path)
        assert captured.read_text() == "Density\nTemperature\n\n"

    def test_plumed_flag_is_forwarded(self, tmp_path: Path) -> None:
        path = script(tmp_path, "gmx", "exit 0")
        runner = GROMACSRunner(usable_status(path), enabled=False)
        result = runner.mdrun(deffnm="w0", cwd=tmp_path, plumed="plumed.dat")
        assert result.command[-2:] == ["-plumed", "plumed.dat"]


class TestSerialisation:
    def test_as_dict_truncates_streams(self, tmp_path: Path) -> None:
        path = script(tmp_path, "gmx", "exit 0")
        runner = GROMACSRunner(usable_status(path), enabled=False)
        result = runner.run(["--version"])
        result.stdout = "x" * 10_000
        payload = result.as_dict(tail=100)
        assert len(payload["stdout_tail"]) == 100
        assert payload["succeeded"] is False


# --------------------------------------------------------------------------
# Optional hardware tests -- skipped with an explicit reason when unavailable
# --------------------------------------------------------------------------
@requires_gromacs
def test_real_gromacs_version(tmp_path: Path) -> None:

    status = discover_tool("gromacs", ToolConfig(executable="gmx"))
    result = GROMACSRunner(status, enabled=True).version()
    assert result.succeeded
    assert "GROMACS" in result.stdout + result.stderr


@requires_orca
def test_real_orca_reports_a_version() -> None:

    status = discover_tool("orca", ToolConfig(executable="orca"))
    assert status.found
    assert status.version is not None, "ORCA banner should yield a parseable version"


class TestOutputCapture:
    """Regression: a long ORCA log must not lose the header the parser needs.

    ``LocalRunner`` truncates to the tail by default, which is right for GROMACS (errors
    are at the end) and wrong for ORCA, whose banner, version and level of theory are at
    the front. A relaxed surface scan produces close to a megabyte of log, so the
    default limit silently removed everything that identified it.
    """

    def test_default_capture_keeps_the_tail(self, tmp_path: Path) -> None:
        from polymer_engine.local.runner import CAPTURE_LIMIT

        path = script(tmp_path, "gmx", f"python3 -c \"print('X'*{CAPTURE_LIMIT + 5000})\"")
        result = GROMACSRunner(usable_status(path), enabled=True).run(["x"], cwd=tmp_path)
        assert len(result.stdout) == CAPTURE_LIMIT

    def test_orca_runner_captures_the_whole_log(self, tmp_path: Path) -> None:
        from polymer_engine.local.discovery import ToolStatus
        from polymer_engine.local.runner import CAPTURE_LIMIT
        from polymer_engine.qm.orca_runner import ORCARunner

        marker = "HEADER-MARKER"
        path = script(
            tmp_path, "orca",
            f"python3 -c \"print('{marker}'); print('Y'*{CAPTURE_LIMIT + 5000})\"",
        )
        status = ToolStatus(
            name="orca", requested=str(path), path=str(path), found=True,
            executable=True, version="6.1.1",
        )
        result = ORCARunner(status, enabled=True).run(["x"], cwd=tmp_path)
        assert marker in result.stdout, "the header must survive; the parser needs it"
        assert len(result.stdout) > CAPTURE_LIMIT

    def test_capture_limit_is_configurable(self, tmp_path: Path) -> None:
        path = script(tmp_path, "gmx", "python3 -c \"print('Z'*10000)\"")
        runner = GROMACSRunner(usable_status(path), enabled=True, capture_limit=100)
        assert len(runner.run(["x"], cwd=tmp_path).stdout) == 100
