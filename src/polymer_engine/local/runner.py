"""Running local scientific executables.

The single most important rule here (charter rule 7): **a dry run is not a success.**
:class:`CommandResult` carries an explicit ``mode``, and ``ok`` is false for a dry
run.  Callers that want "did the command actually run and succeed" ask for
:attr:`CommandResult.succeeded`, which cannot be satisfied without real execution.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from polymer_engine.core.errors import ToolNotFound
from polymer_engine.core.logging import get_logger
from polymer_engine.local.discovery import ToolStatus
from polymer_engine.local.plumed import (
    PLUMED_KERNEL_ENV,
    PlumedKernel,
    discover_plumed_kernel,
)

logger = get_logger("local.runner")

ExecutionMode = Literal["real", "dry_run", "unavailable"]

#: stdout/stderr kept per stream; enough for diagnosis without unbounded logs.
#: Truncation keeps the *tail*, which is where errors appear -- but a tool whose
#: output is itself the scientific record (ORCA) must pass ``capture_limit=None``,
#: because its identifying header and level of theory are at the front.
CAPTURE_LIMIT = 200_000


@dataclass(slots=True)
class CommandResult:
    """What happened when we asked a local tool to do something."""

    command: list[str]
    mode: ExecutionMode
    returncode: int | None
    stdout: str
    stderr: str
    cwd: str
    duration_s: float = 0.0
    timed_out: bool = False
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """True only when the command really ran and exited zero.

        A dry run is never ``succeeded``.  Note this is a statement about the
        *process*, not about the science: a zero exit code is necessary, never
        sufficient, for a trustworthy simulation.
        """
        return self.mode == "real" and self.returncode == 0 and not self.timed_out

    @property
    def executed(self) -> bool:
        return self.mode == "real"

    def as_dict(self, *, tail: int = 4000) -> dict[str, Any]:
        return {
            "command": self.command,
            "mode": self.mode,
            "returncode": self.returncode,
            "succeeded": self.succeeded,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
            "cwd": self.cwd,
            "stdout_tail": self.stdout[-tail:],
            "stderr_tail": self.stderr[-tail:],
            "artifacts": self.artifacts,
            "error": self.error,
        }


class LocalRunner:
    """Base runner.  Refuses to execute unless explicitly enabled."""

    tool_name = "tool"

    def __init__(
        self,
        status: ToolStatus,
        *,
        enabled: bool = False,
        default_timeout_s: float | None = None,
        extra_env: Mapping[str, str] | None = None,
        capture_limit: int | None = CAPTURE_LIMIT,
    ) -> None:
        self.status = status
        self.enabled = enabled
        self.default_timeout_s = default_timeout_s
        self.extra_env = dict(extra_env or {})
        self.capture_limit = capture_limit

    @property
    def executable(self) -> str:
        return self.status.path or self.status.requested

    def available(self) -> bool:
        return self.status.usable

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout_s: float | None = None,
        stdin: str | None = None,
        artifacts: Sequence[str | Path] = (),
        env_overlay: Mapping[str, str] | None = None,
    ) -> CommandResult:
        command = [self.executable, *(str(a) for a in args)]
        workdir = str(cwd or Path.cwd())

        if not self.enabled:
            # Explicitly not a success: the command was never run.
            return CommandResult(
                command=command,
                mode="dry_run",
                returncode=None,
                stdout="",
                stderr="",
                cwd=workdir,
                error="Local execution is disabled (safety.execution_enabled is false)",
            )

        if not self.status.usable:
            reason = "; ".join(self.status.issues) or f"{self.tool_name} is not usable"
            return CommandResult(
                command=command,
                mode="unavailable",
                returncode=None,
                stdout="",
                stderr="",
                cwd=workdir,
                error=reason,
            )

        env = {**os.environ, **self.extra_env, **dict(env_overlay or {})}
        timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        started = time.monotonic()
        logger.info("Running %s: %s", self.tool_name, " ".join(command))
        try:
            proc = subprocess.run(
                command,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
                input=stdin,
                stdin=None if stdin is not None else subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=command,
                mode="real",
                returncode=None,
                stdout=_decode(exc.stdout),
                stderr=_decode(exc.stderr),
                cwd=workdir,
                duration_s=time.monotonic() - started,
                timed_out=True,
                error=f"Timed out after {timeout}s",
            )
        except OSError as exc:
            raise ToolNotFound(
                f"Could not execute {self.tool_name}: {exc}", path=self.executable
            ) from exc

        duration = time.monotonic() - started
        existing = [str(p) for p in artifacts if Path(p).exists()]
        return CommandResult(
            command=command,
            mode="real",
            returncode=proc.returncode,
            stdout=self._capture(proc.stdout),
            stderr=self._capture(proc.stderr),
            cwd=workdir,
            duration_s=duration,
            artifacts=existing,
        )


    def _capture(self, stream: str | None) -> str:
        text = stream or ""
        if self.capture_limit is None:
            return text
        return text[-self.capture_limit :]


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


class GROMACSRunner(LocalRunner):
    tool_name = "gromacs"

    def __init__(self, *args: Any, plumed_kernel: PlumedKernel | None = None, **kwargs: Any) -> None:
        """``plumed_kernel`` is resolved once per campaign and reused.

        Left as ``None``, each ``mdrun -plumed`` call discovers it, which is correct but
        re-probes the ``plumed`` executable every time.
        """
        super().__init__(*args, **kwargs)
        self.plumed_kernel = plumed_kernel

    def version(self) -> CommandResult:
        return self.run(["--version"])

    def grompp(
        self,
        *,
        mdp: str | Path,
        structure: str | Path,
        topology: str | Path,
        output: str | Path,
        cwd: str | Path,
        index: str | Path | None = None,
        restraint_structure: str | Path | None = None,
        max_warnings: int = 0,
        include_dirs: Sequence[str | Path] = (),
    ) -> CommandResult:
        """Preprocess into a ``.tpr``.

        ``max_warnings`` defaults to 0 on purpose: ``-maxwarn`` suppresses exactly the
        diagnostics that catch a broken system, so raising it must be a deliberate,
        recorded choice rather than a default.
        """
        args: list[str] = [
            "grompp", "-f", str(mdp), "-c", str(structure), "-p", str(topology), "-o", str(output),
        ]
        if restraint_structure is not None:
            args += ["-r", str(restraint_structure)]
        if index is not None:
            args += ["-n", str(index)]
        for directory in include_dirs:
            args += ["-I", str(directory)]
        if max_warnings:
            args += ["-maxwarn", str(max_warnings)]
        return self.run(args, cwd=cwd, artifacts=[Path(cwd) / str(output)])

    def mdrun(
        self,
        *,
        deffnm: str,
        cwd: str | Path,
        use_gpu: bool = False,
        gpu_pme: bool = True,
        ntomp: int | None = None,
        ntmpi: int | None = None,
        plumed: str | Path | None = None,
        checkpoint: str | Path | None = None,
        append: bool = False,
        timeout_s: float | None = None,
    ) -> CommandResult:
        """Run MD.

        GPU offload is opt-in and driven by discovered capability, never assumed:
        passing ``-nb gpu`` on a CPU-only build makes GROMACS abort.  ``gpu_pme`` must
        be ``False`` for energy minimisation, which uses a non-dynamical integrator that
        the PME GPU kernel does not support.

        ``ntmpi`` defaults to 1 whenever ``ntomp`` is set on a GPU-capable build. That is
        not a preference -- GROMACS makes it a *fatal error* to set the thread count
        without also fixing the rank count when GPUs are present::

            When using GPUs, setting the number of OpenMP threads without specifying the
            number of ranks can lead to conflicting demands. Please specify the number of
            thread-MPI ranks as well (option -ntmpi).

        So on any GPU machine, every ``ntomp`` call would abort before doing any work.
        """
        args: list[str] = ["mdrun", "-deffnm", deffnm]
        if use_gpu:
            if not self.status.capabilities.get("has_gpu"):
                return CommandResult(
                    command=[self.executable, "mdrun"],
                    mode="unavailable",
                    returncode=None,
                    stdout="",
                    stderr="",
                    cwd=str(cwd),
                    error="GPU offload requested but this GROMACS build reports no GPU support",
                )
            args += ["-nb", "gpu"]
            if gpu_pme:
                # PME on the GPU requires a dynamical integrator. Energy minimisation
                # uses `steep`, and GROMACS refuses the combination outright:
                #   Cannot compute PME interactions on a GPU, because:
                #     PME GPU does not support: Non-dynamical integrator
                # so a minimisation must pass gpu_pme=False rather than discover this
                # after the run has been dispatched.
                args += ["-pme", "gpu", "-bonded", "gpu"]
        if ntomp is not None:
            args += ["-ntomp", str(ntomp)]
            if ntmpi is None and self.status.capabilities.get("has_gpu"):
                ntmpi = 1
        if ntmpi is not None:
            args += ["-ntmpi", str(ntmpi)]
        env_overlay: dict[str, str] = {}
        if plumed is not None:
            if not self.status.capabilities.get("has_plumed", True):
                return CommandResult(
                    command=[self.executable, "mdrun"],
                    mode="unavailable",
                    returncode=None,
                    stdout="",
                    stderr="",
                    cwd=str(cwd),
                    error=(
                        "PLUMED biasing requested but this GROMACS build reports "
                        f"'Plumed support: {self.status.capabilities.get('plumed_support')}'"
                    ),
                )
            kernel = self.plumed_kernel if self.plumed_kernel is not None else discover_plumed_kernel()
            if not kernel.usable:
                # GROMACS would otherwise abort mid-run with an *internal error*, after
                # grompp and setup have already been paid for. Refuse before starting.
                return CommandResult(
                    command=[self.executable, "mdrun"],
                    mode="unavailable",
                    returncode=None,
                    stdout="",
                    stderr="",
                    cwd=str(cwd),
                    error=f"PLUMED biasing requested but the kernel is unusable: {kernel.reason()}",
                )
            env_overlay[PLUMED_KERNEL_ENV] = kernel.path or ""
            args += ["-plumed", str(plumed)]
        if checkpoint is not None:
            args += ["-cpi", str(checkpoint)]
            if append:
                args += ["-append"]
            else:
                args += ["-noappend"]
        return self.run(args, cwd=cwd, timeout_s=timeout_s, env_overlay=env_overlay)

    def energy(
        self,
        *,
        edr: str | Path,
        terms: Sequence[str],
        output: str | Path,
        cwd: str | Path,
        begin_ps: float | None = None,
    ) -> CommandResult:
        """Extract energy terms to an ``.xvg``.

        ``gmx energy`` reads its term selection from stdin; the terms are piped in
        rather than left for an interactive prompt that would hang a batch job.
        """
        args: list[str] = ["energy", "-f", str(edr), "-o", str(output)]
        if begin_ps is not None:
            args += ["-b", str(begin_ps)]
        selection = "\n".join(terms) + "\n\n"
        return self.run(args, cwd=cwd, stdin=selection, artifacts=[Path(cwd) / str(output)])

    def trjconv(
        self,
        *,
        trajectory: str | Path,
        tpr: str | Path,
        output: str | Path,
        cwd: str | Path,
        selection: str = "System",
        pbc: str | None = "mol",
        begin_ps: float | None = None,
        skip: int | None = None,
    ) -> CommandResult:
        args: list[str] = ["trjconv", "-f", str(trajectory), "-s", str(tpr), "-o", str(output)]
        if pbc:
            args += ["-pbc", pbc]
        if begin_ps is not None:
            args += ["-b", str(begin_ps)]
        if skip is not None:
            args += ["-skip", str(skip)]
        return self.run(args, cwd=cwd, stdin=f"{selection}\n", artifacts=[Path(cwd) / str(output)])


class ORCARunner(LocalRunner):
    tool_name = "orca"

    def run_input(self, input_file: str | Path, *, cwd: str | Path, timeout_s: float | None = None) -> CommandResult:
        """ORCA takes a single input file and writes its output to stdout."""
        return self.run([Path(input_file).name], cwd=cwd, timeout_s=timeout_s)


class PLUMEDRunner(LocalRunner):
    tool_name = "plumed"

    def version(self) -> CommandResult:
        return self.run(["--no-mpi", "--version"])

    def driver(
        self,
        *,
        plumed_input: str | Path,
        trajectory: str | Path,
        cwd: str | Path,
        trajectory_flag: str = "--mf_xtc",
        timeout_s: float | None = None,
    ) -> CommandResult:
        return self.run(
            ["driver", "--plumed", str(plumed_input), trajectory_flag, str(trajectory)],
            cwd=cwd,
            timeout_s=timeout_s,
        )


__all__ = [
    "CommandResult",
    "GROMACSRunner",
    "LocalRunner",
    "ORCARunner",
    "PLUMEDRunner",
]
