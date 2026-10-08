"""Executing ORCA and turning the result into a validated QM record.

The runner never reports success from an exit status.  It runs ORCA, parses the
output, and lets :mod:`polymer_engine.qm.orca_parser` decide what happened.  A job
that exits 0 while failing to converge comes back as
:attr:`~polymer_engine.qm.orca_parser.QMStatus.FAILED_SCIENTIFICALLY`.

Like every other local tool in the engine, execution is off unless explicitly enabled,
and a disabled runner produces a ``dry_run`` record that is not a success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.errors import ScientificError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination
from polymer_engine.core.provenance import ProvenanceGraph
from polymer_engine.local.discovery import ToolStatus
from polymer_engine.local.runner import CommandResult, LocalRunner
from polymer_engine.qm.orca_input import write_input
from polymer_engine.qm.orca_parser import OrcaOutput, QMStatus, parse_orca_output
from polymer_engine.qm.spec import QMJobSpec, Structure

logger = get_logger("qm.orca")

#: ORCA scratch files that are large and not worth keeping by default.
SCRATCH_SUFFIXES = (".tmp", ".gbw.tmp", ".densities", ".cpcm", ".bibtex")


@dataclass
class QMRun:
    """The full record of one attempted QM calculation."""

    spec: QMJobSpec
    status: QMStatus
    executed: bool
    workdir: str
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    duration_s: float = 0.0
    output: OrcaOutput | None = None
    input_path: str | None = None
    output_path: str | None = None
    artifacts: list[str] = field(default_factory=list)
    artifact_id: str | None = None
    error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def succeeded(self) -> bool:
        """True only when ORCA really ran and the science converged."""
        return self.executed and self.status is QMStatus.COMPLETED

    @property
    def final_energy_hartree(self) -> float | None:
        return self.output.final_energy_hartree if self.output else None

    def optimized_structure(self) -> Structure | None:
        """The final geometry as a :class:`Structure`, if the job produced one.

        Returns ``None`` for anything that did not complete -- handing back a geometry
        from a failed optimisation is how a bad structure enters a force field.
        """
        if not self.succeeded or self.output is None or not self.output.final_geometry:
            return None
        from polymer_engine.qm.spec import Atom

        atoms = [Atom(symbol, x, y, z) for symbol, x, y, z in self.output.final_geometry]
        return Structure(
            atoms=atoms,
            charge=self.spec.structure.charge,
            multiplicity=self.spec.structure.multiplicity,
            name=f"{self.spec.label}_optimized",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.spec.label,
            "spec": self.spec.as_dict(),
            "status": self.status.value,
            "executed": self.executed,
            "succeeded": self.succeeded,
            "workdir": self.workdir,
            "command": self.command,
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 3),
            "input_path": self.input_path,
            "output_path": self.output_path,
            "artifacts": self.artifacts,
            "artifact_id": self.artifact_id,
            "error": self.error,
            "result": self.output.as_dict() if self.output else None,
        }


class ORCARunner(LocalRunner):
    """Runs ORCA jobs from a :class:`QMJobSpec`.

    ORCA writes its entire log to stdout, and that log *is* the scientific record: the
    banner, version and level of theory are at the front, while convergence and results
    are at the end.  Truncating either end would make a long job unparseable, so this
    runner captures the whole stream.
    """

    tool_name = "orca"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("capture_limit", None)
        super().__init__(*args, **kwargs)

    def run_job(
        self,
        spec: QMJobSpec,
        workdir: str | Path,
        *,
        graph: ProvenanceGraph | None = None,
        parents: list[str] | None = None,
        keep_scratch: bool = False,
    ) -> QMRun:
        """Write the input, run ORCA, parse the output, and classify the outcome."""
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        input_path = write_input(spec, workdir)
        output_path = workdir / f"{spec.label}.out"

        run = QMRun(
            spec=spec,
            status=QMStatus.INCOMPLETE,
            executed=False,
            workdir=str(workdir),
            input_path=str(input_path),
        )

        if not self.enabled:
            run.error = "Local execution is disabled (safety.execution_enabled is false)"
            run.command = [self.executable, input_path.name]
            logger.info("ORCA execution disabled; wrote %s but ran nothing", input_path.name)
            return run

        if not self.status.usable:
            run.error = "; ".join(self.status.issues) or "ORCA is not usable"
            return run

        # ORCA insists on being given the input file relative to its working directory.
        result: CommandResult = self.run(
            [input_path.name], cwd=workdir, timeout_s=spec.timeout_s
        )
        run.executed = result.mode == "real"
        run.command = result.command
        run.returncode = result.returncode
        run.duration_s = result.duration_s
        run.stdout_tail = result.stdout[-4000:]
        run.stderr_tail = result.stderr[-4000:]

        if result.timed_out:
            run.status = QMStatus.INCOMPLETE
            run.error = f"ORCA timed out after {spec.timeout_s}s"
            output_path.write_text(result.stdout, encoding="utf-8")
            run.output_path = str(output_path)
            return run

        # ORCA writes its log to stdout; persist it before parsing so a failure is
        # inspectable afterwards.
        output_path.write_text(result.stdout, encoding="utf-8")
        run.output_path = str(output_path)

        run.output = parse_orca_output(
            result.stdout,
            expect_geometry=spec.requires_geometry_convergence,
            expect_frequencies=spec.requires_frequencies,
        )
        run.status = run.output.status
        if not run.succeeded:
            run.error = "; ".join(run.output.diagnostics) or f"ORCA job finished as {run.status.value}"
            logger.warning("ORCA job %s: %s", spec.label, run.error)

        if not keep_scratch:
            _remove_scratch(workdir)

        run.artifacts = sorted(
            str(p) for p in workdir.iterdir() if p.is_file() and p.suffix not in {".tmp"}
        )

        if graph is not None:
            artifact = graph.register_file(
                output_path,
                kind="qm_output",
                parents=parents or [],
                source="orca",
                source_version=run.output.orca_version if run.output else None,
                software={"orca": run.output.orca_version or "unknown"} if run.output else {},
                command=run.command,
                parameters=spec.as_dict(),
                units={"energy": "kJ/mol", "length": "nm"},
                validation_state=(
                    Determination.KNOWN if run.succeeded else Determination.REQUIRES_VALIDATION
                ),
                notes=run.status.value,
            )
            run.artifact_id = artifact.artifact_id
        return run

    def run_torsion_scan(
        self, spec: QMJobSpec, workdir: str | Path, **kwargs: Any
    ) -> QMRun:
        """Run a relaxed torsional scan and require that it produced a profile."""
        if spec.torsion is None:
            raise ScientificError("A torsion scan needs a TorsionSpec", label=spec.label)
        run = self.run_job(spec, workdir, **kwargs)
        if run.succeeded and run.output is not None and len(run.output.scan_points) < 2:
            run.status = QMStatus.FAILED_SCIENTIFICALLY
            run.error = "scan completed but produced fewer than two usable points"
            run.output.diagnostics.append(run.error)
        return run


def _remove_scratch(workdir: Path) -> None:
    """Delete ORCA's large intermediate files, keeping inputs, logs and geometries."""
    for path in workdir.iterdir():
        if not path.is_file():
            continue
        if path.name.endswith(SCRATCH_SUFFIXES) or path.suffix in {".tmp"}:
            path.unlink(missing_ok=True)


def build_orca_runner(config: Any, *, tool: ToolStatus | None = None) -> ORCARunner:
    """Construct an :class:`ORCARunner` from engine configuration."""
    from polymer_engine.local.discovery import discover_tool

    status = tool or discover_tool("orca", config.local_tools.orca)
    return ORCARunner(
        status,
        enabled=config.safety.execution_enabled,
        default_timeout_s=config.resources.job_timeout_s,
    )


__all__ = ["SCRATCH_SUFFIXES", "ORCARunner", "QMRun", "build_orca_runner"]
