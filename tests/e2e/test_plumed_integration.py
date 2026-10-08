"""Real PLUMED + GROMACS integration.

Everything in ``test_research_pipeline.py::TestUmbrellaPipeline`` samples analytically,
which proves the *estimator* and says nothing about whether the engine drives GROMACS
and PLUMED correctly.  This module closes that gap: it runs ``gmx grompp`` and
``gmx mdrun -plumed`` for real, reads the COLVAR PLUMED actually wrote, and checks the
collective variable against an independent geometric calculation.

The independent check applies the minimum-image convention explicitly.  A naive
``|r_i - r_j|`` disagrees with PLUMED by up to ~2 nm in a 3 nm box whenever the molecule
straddles a periodic boundary -- which happened in 331 of 501 frames the first time this
was measured.  The naive calculation is the wrong one, and a test that used it would
"discover" a PLUMED bug that does not exist.

Skipped with an explicit reason wherever GROMACS or PLUMED is missing.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from polymer_engine.core.config import ToolConfig, load_config
from polymer_engine.core.errors import InsufficientDataError
from polymer_engine.core.models import GateStatus
from polymer_engine.core.provenance import ProvenanceGraph
from polymer_engine.local.discovery import discover_tool
from polymer_engine.local.plumed import (
    KERNEL_NAMES,
    PLUMED_KERNEL_ENV,
    PlumedKernel,
    discover_plumed_kernel,
    plumed_version,
)
from polymer_engine.local.runner import GROMACSRunner
from polymer_engine.simulation.mdp import (
    generate_minimization,
    generate_npt,
    generate_nvt,
    generate_production,
)
from polymer_engine.simulation.umbrella import ReactionCoordinate
from polymer_engine.simulation.umbrella_execution import (
    UmbrellaCampaign,
    UmbrellaJustification,
    UmbrellaStatus,
    make_gromacs_plumed_runner,
    read_colvar,
)
from tests.markers import requires_gromacs, requires_plumed

VALID_SYSTEM = Path(__file__).resolve().parents[1] / "fixtures" / "systems" / "valid_system"

#: The fixture chain is 8 united-atom carbons; atoms 9-68 are water.
CHAIN_HEAD, CHAIN_TAIL = 1, 8

#: float32 trajectory storage; PLUMED and numpy should agree to round-off.
CV_TOLERANCE_NM = 1.0e-5


def equilibration_config(root: Path):
    """Small but real: enough steps to relax the fixture, few enough to run in a test."""
    return load_config(
        discover=False,
        use_env=False,
        overrides={
            "paths": {"root": str(root)},
            "http": {"cache_enabled": False, "offline": True},
            "simulation": {
                "replicas": 1,
                "minimization_steps": 5000,
                "nvt_ns": 0.02,
                "npt_ns": 0.02,
                "production_ns": 0.01,
                "force_field": "test-UA",
                "water_model": "SPC-like",
                "trajectory_output_ps": 0.1,
                "energy_output_ps": 0.1,
                "log_output_ps": 0.5,
            },
            "resources": {"gpu_available": False},
        },
    )


def gromacs_runner(kernel: PlumedKernel | None = None) -> GROMACSRunner:
    return GROMACSRunner(
        discover_tool("gromacs", ToolConfig(executable="gmx")),
        enabled=True,
        plumed_kernel=kernel,
    )


def minimum_image_distance(a: np.ndarray, b: np.ndarray, box_nm: np.ndarray) -> float:
    """Distance under the minimum-image convention, which is what PLUMED computes."""
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    delta -= box_nm * np.round(delta / box_nm)
    return float(np.linalg.norm(delta))


# ==========================================================================
# Discovery: the failure modes that must be reported, not guessed around
# ==========================================================================
class TestKernelDiscovery:
    def test_a_configured_kernel_that_does_not_exist_is_an_error(self, tmp_path: Path) -> None:
        """Never fall through to a different kernel than the one that was asked for."""
        kernel = discover_plumed_kernel(configured_path=tmp_path / "absent.so")
        assert kernel.usable is False
        assert kernel.found is False
        assert "does not exist" in kernel.reason()
        assert kernel.environment() == {}

    def test_an_unreadable_kernel_is_refused(self, tmp_path: Path) -> None:
        blocked = tmp_path / KERNEL_NAMES[0]
        blocked.write_bytes(b"\x7fELF")
        blocked.chmod(0o000)
        try:
            kernel = discover_plumed_kernel(configured_path=blocked)
            if kernel.readable:  # running as root: the mode bits do not bind
                pytest.skip("process can read mode-000 files; the readability check cannot be exercised")
            assert kernel.usable is False
            assert "not readable" in kernel.reason()
        finally:
            blocked.chmod(0o644)

    def test_a_missing_plumed_executable_is_reported(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.delenv(PLUMED_KERNEL_ENV, raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        kernel = discover_plumed_kernel()
        assert kernel.usable is False
        assert "no 'plumed' executable on PATH" in kernel.reason()

    def test_the_environment_overlay_is_empty_when_unusable(self, tmp_path: Path) -> None:
        assert discover_plumed_kernel(configured_path=tmp_path / "nope.so").environment() == {}

    def test_an_explicit_kernel_wins_over_discovery(self, tmp_path: Path) -> None:
        explicit = tmp_path / KERNEL_NAMES[0]
        explicit.write_bytes(b"\x7fELF")
        kernel = discover_plumed_kernel(configured_path=explicit)
        assert kernel.usable is True
        assert kernel.source == "configured"
        assert kernel.environment() == {PLUMED_KERNEL_ENV: str(explicit)}


class TestRunnerRefusals:
    """``mdrun -plumed`` must refuse *before* running, not abort mid-simulation."""

    @requires_gromacs
    def test_mdrun_refuses_when_the_kernel_is_unusable(self, tmp_path: Path) -> None:
        broken = PlumedKernel(path=None, found=False, issues=["no kernel for this test"])
        result = gromacs_runner(broken).mdrun(deffnm="x", cwd=tmp_path, plumed="plumed.dat")
        assert result.succeeded is False
        assert result.executed is False
        assert result.mode == "unavailable"
        assert "kernel is unusable" in (result.error or "")

    @requires_gromacs
    def test_a_gromacs_build_without_plumed_support_is_refused(self, tmp_path: Path) -> None:
        runner = gromacs_runner(discover_plumed_kernel())
        runner.status.capabilities["has_plumed"] = False
        runner.status.capabilities["plumed_support"] = "disabled"
        result = runner.mdrun(deffnm="x", cwd=tmp_path, plumed="plumed.dat")
        assert result.succeeded is False
        assert result.mode == "unavailable"
        assert "Plumed support: disabled" in (result.error or "")

    @requires_gromacs
    def test_a_plumed_free_mdrun_is_unaffected_by_a_broken_kernel(self, tmp_path: Path) -> None:
        """A missing PLUMED kernel must not block ordinary MD."""
        broken = PlumedKernel(path=None, found=False, issues=["absent"])
        result = gromacs_runner(broken).mdrun(deffnm="x", cwd=tmp_path, timeout_s=30)
        assert result.mode != "unavailable"   # it ran and failed on the missing .tpr instead


# ==========================================================================
# The real thing
# ==========================================================================
@requires_gromacs
@requires_plumed
@pytest.mark.slow
class TestRealPlumedExecution:
    """gmx grompp -> gmx mdrun -plumed -> COLVAR -> parsed -> independently verified."""

    @pytest.fixture(scope="class")
    @classmethod
    def equilibrated(cls, tmp_path_factory) -> Path:
        """EM -> NVT -> NPT on the validated fixture system.

        The raw fixture cannot be used directly: started cold it reports
        ``LJ (SR) = 2.9e+05 kJ/mol`` and SETTLE fails on step 0.  That is a property of
        the fixture, not of PLUMED, and skipping minimisation to save test time would
        have manufactured a PLUMED failure that does not exist.
        """
        work = tmp_path_factory.mktemp("equilibrated")
        for name in ("system.gro", "topol.top", "polymer.itp", "water.itp"):
            (work / name).write_bytes((VALID_SYSTEM / name).read_bytes())
        defaults = equilibration_config(work).simulation
        gmx = gromacs_runner()

        stages = [
            ("em", generate_minimization(defaults), "system.gro"),
            ("nvt", generate_nvt(defaults, seed=1), "em.gro"),
            ("npt", generate_npt(defaults, seed=1), "nvt.gro"),
        ]
        for name, stage, structure in stages:
            (work / f"{name}.mdp").write_text(stage.text, encoding="utf-8")
            grompp = gmx.grompp(
                mdp=f"{name}.mdp", structure=structure, topology="topol.top",
                output=f"{name}.tpr", cwd=work,
            )
            assert grompp.succeeded, f"{name} grompp failed: {grompp.stderr[-800:]}"
            md = gmx.mdrun(deffnm=name, cwd=work, ntomp=4, timeout_s=900)
            assert md.succeeded, f"{name} mdrun failed: {md.stderr[-800:]}"
        assert (work / "npt.gro").exists()
        return work

    # ---- B1: the installation itself -------------------------------------
    def test_the_installation_reports_itself_correctly(self) -> None:
        kernel = discover_plumed_kernel()
        assert kernel.usable is True
        assert kernel.version is not None, "plumed info --version produced no version"
        assert kernel.is_installed is True
        assert kernel.has_dlopen is True, "GROMACS cannot dlopen a kernel without dlopen support"
        assert Path(kernel.path or "").is_file()

    def test_the_version_probe_survives_a_build_that_rejects_bare_version(self) -> None:
        """Regression: `plumed --version` exits 0 with 'Unknown option' on conda 2.9.x.

        The old probe parsed that as "no version" while looking like a clean run, so the
        engine reported `version: null` for a perfectly good PLUMED.
        """
        assert plumed_version(discover_plumed_kernel().executable or "plumed") is not None

    def test_gromacs_reports_plumed_support(self) -> None:
        capabilities = gromacs_runner().status.capabilities
        assert capabilities.get("has_plumed") is True, capabilities.get("plumed_support")

    # ---- B4/B5/B6: run it and check the number ---------------------------
    def test_a_real_plumed_run_produces_a_verifiable_collective_variable(
        self, equilibrated: Path, tmp_path: Path
    ) -> None:
        work = tmp_path / "cv"
        work.mkdir()
        for name in ("topol.top", "polymer.itp", "water.itp", "npt.gro", "npt.cpt"):
            (work / name).write_bytes((equilibrated / name).read_bytes())

        defaults = equilibration_config(work).simulation
        stage = generate_production(defaults, seed=7, plumed=True)
        # Verification run only: force full-precision positions into the .trr at the
        # COLVAR stride. Production writes compressed positions at 0.001 nm, which is
        # 100x coarser than the agreement being measured here.
        text = stage.text.replace("nstxout                  = 0", "nstxout                  = 10")
        assert "nstxout                  = 10" in text
        (work / "prod.mdp").write_text(text, encoding="utf-8")
        (work / "plumed.dat").write_text(
            "UNITS LENGTH=nm ENERGY=kj/mol TIME=ps\n"
            f"d: DISTANCE ATOMS={CHAIN_HEAD},{CHAIN_TAIL}\n"
            "PRINT ARG=d FILE=COLVAR STRIDE=10\n",
            encoding="utf-8",
        )

        gmx = gromacs_runner(discover_plumed_kernel())
        grompp = gmx.grompp(
            mdp="prod.mdp", structure="npt.gro", topology="topol.top", output="prod.tpr", cwd=work
        )
        assert grompp.succeeded, grompp.stderr[-800:]
        assert grompp.returncode == 0
        assert "WARNING" not in grompp.stderr.upper() or "0 WARNING" in grompp.stderr.upper()

        md = gmx.mdrun(deffnm="prod", cwd=work, plumed="plumed.dat", ntomp=4, timeout_s=1800)
        assert md.succeeded, f"{md.error} {md.stderr[-1500:]}"
        assert md.returncode == 0
        assert md.mode == "real"

        # PLUMED initialised, and said so in its own log.
        plumed_log = (work / "PLUMED.OUT").read_text(errors="replace")
        assert "PLUMED is starting" in plumed_log
        assert "Molecular dynamics engine: gromacs" in plumed_log
        assert "Number of atoms: 68" in plumed_log
        assert "PLUMED: Action DISTANCE" in plumed_log
        for bad in ("not available", "Check your PLUMED_KERNEL", "PLUMED error"):
            assert bad not in plumed_log, bad

        colvar_path = work / "COLVAR"
        assert colvar_path.exists(), "mdrun completed but PLUMED wrote no COLVAR"
        raw = [row for row in colvar_path.read_text().splitlines() if not row.startswith("#")]
        assert len(raw) > 100, f"COLVAR has only {len(raw)} samples"

        # Parsed by the engine's own reader, not by the test.
        values = read_colvar(colvar_path)
        assert values.size == len(raw)
        assert np.all(np.isfinite(values)), "COLVAR contains a non-finite CV"
        times = np.array([float(row.split()[0]) for row in raw])
        assert np.all(np.diff(times) > 0), "COLVAR time is not monotonically increasing"

        # ---- the independent check ----
        mda = pytest.importorskip("MDAnalysis", reason="MDAnalysis is needed to re-read the trajectory")
        universe = mda.Universe(str(work / "system.gro") if (work / "system.gro").exists()
                                else str(equilibrated / "system.gro"), str(work / "prod.trr"))
        colvar = {round(float(t), 4): float(v) for t, v in zip(times, values, strict=True)}
        differences, wrapped = [], 0
        for frame in universe.trajectory:
            if not frame.has_positions:
                continue
            stamp = round(float(frame.time), 4)
            if stamp not in colvar:
                continue
            positions = universe.atoms.positions / 10.0          # MDAnalysis reports angstrom
            box = np.asarray(frame.dimensions[:3], dtype=float) / 10.0
            head, tail = positions[CHAIN_HEAD - 1], positions[CHAIN_TAIL - 1]
            independent = minimum_image_distance(head, tail, box)
            if not math.isclose(independent, float(np.linalg.norm(head - tail)), abs_tol=1e-6):
                wrapped += 1
            differences.append(abs(independent - colvar[stamp]))

        assert len(differences) > 100, f"only {len(differences)} frames could be cross-checked"
        worst = max(differences)
        assert worst < CV_TOLERANCE_NM, (
            f"PLUMED CV disagrees with the independent calculation by {worst:.3e} nm over "
            f"{len(differences)} frames ({wrapped} of which straddled a periodic boundary)"
        )
        # Whether any frame wraps depends on where the chain diffuses, so it is not
        # asserted here. `TestMinimumImage` covers the PBC path deterministically.

    # ---- B7: through the engine's umbrella execution layer ----------------
    def test_umbrella_windows_run_through_the_engine_and_are_judged(
        self, equilibrated: Path, tmp_path: Path
    ) -> None:
        """campaign -> executor -> mdrun -plumed -> COLVAR -> gates -> provenance."""
        graph = ProvenanceGraph()
        defaults = equilibration_config(tmp_path).simulation
        stage = generate_production(defaults, seed=11, plumed=True)
        gmx = gromacs_runner(discover_plumed_kernel())

        coordinate = ReactionCoordinate(
            name="d", kind="distance", units="nm",
            justification="End-to-end extension of the fixture chain, an intramolecular coordinate.",
            group_a=str(CHAIN_HEAD), group_b=str(CHAIN_TAIL),
        )
        justification = UmbrellaJustification(
            question="What is the free-energy profile for extending the fixture chain?",
            reaction_coordinate="distance between the first and last chain carbon",
            physical_interpretation="reversible work of intramolecular extension",
            expected_observable="curvature of the PMF about the equilibrium extension",
            reason_for_method="the extension is stiff and is poorly sampled in plain MD",
            starting_state="the fixture system after EM, NVT and NPT equilibration",
            endpoint_definition="0.85 nm, beyond the equilibrium extension",
            author="integration-test",
        )

        def seed_and_run(window, directory: Path):
            for name in ("topol.top", "polymer.itp", "water.itp"):
                (directory / name).write_bytes((equilibrated / name).read_bytes())
            (directory / "start.gro").write_bytes((equilibrated / "npt.gro").read_bytes())
            (directory / "umbrella.mdp").write_text(stage.text, encoding="utf-8")
            runner = make_gromacs_plumed_runner(
                gmx, structure_for=lambda _w: "start.gro", timeout_s=1800.0
            )
            return runner(window, directory)

        root = tmp_path / "umbrella"
        result = UmbrellaCampaign(coordinate, temperature_k=300.0, graph=graph).run(
            minimum=0.55, maximum=0.85, root=root,
            justification=justification, runner=seed_and_run,
        )

        assert result.runs, "no window ran"
        for run in result.runs:
            assert run.error is None, run.error
            window_dir = root / f"window_{run.index:03d}"
            assert (window_dir / "COLVAR").exists(), f"window {run.index} produced no COLVAR"
            assert (window_dir / "plumed.dat").read_text().count("RESTRAINT") == 1
            samples = read_colvar(window_dir / "COLVAR")
            assert samples.size > 0 and np.all(np.isfinite(samples))

        assert result.status not in {
            UmbrellaStatus.FAILED,
            UmbrellaStatus.NOT_EXECUTED,
            UmbrellaStatus.REQUIRES_EXPERT_DECISION,
        }, result.status

        # The gates ran and reached a verdict -- whatever that verdict is. A 10 ps window
        # is not expected to yield a publishable PMF, and the gates are expected to say so.
        assert result.report.gates
        justification_gate = next(
            g for g in result.report.gates if g.gate == "umbrella:justification"
        )
        assert justification_gate.status is GateStatus.PASS

        # Provenance: the graph is internally consistent and no recorded file has been
        # altered since it was hashed. `verify_all` returns a per-artifact status map;
        # derived artifacts (the PMF itself) are legitimately "not-file-backed".
        assert graph.dangling_parents() == []
        statuses = graph.verify_all()
        assert statuses, "the umbrella campaign recorded no provenance at all"
        assert not [a for a, s in statuses.items() if s.startswith("MISMATCH")], statuses
        file_backed = [a for a, s in statuses.items() if s == "ok"]
        assert file_backed, f"no file-backed artifact was recorded: {statuses}"


# ==========================================================================
# B8 regression tests: every failure mode, without needing the real software
# ==========================================================================
class _FakeStatus:
    """A ToolStatus stand-in, so failure modes are testable without GROMACS."""

    def __init__(self, *, usable: bool = True, **capabilities: object) -> None:
        self.name = "gromacs"
        self.path = "/nonexistent/gmx"
        self.issues: list[str] = [] if usable else ["gromacs was not found on this machine"]
        self.capabilities = {"has_gpu": False, "has_plumed": True, **capabilities}
        self._usable = usable

    @property
    def usable(self) -> bool:
        return self._usable


class TestFailureModes:
    """Each of these is a way a PLUMED run can go wrong. None may read as success."""

    def test_gromacs_unavailable_is_not_a_failed_run(self, tmp_path: Path) -> None:
        runner = GROMACSRunner(_FakeStatus(usable=False), enabled=True)  # type: ignore[arg-type]
        result = runner.mdrun(deffnm="x", cwd=tmp_path, plumed="plumed.dat")
        assert result.succeeded is False
        assert result.executed is False
        assert result.mode == "unavailable"
        assert "not found" in (result.error or "")

    def test_execution_disabled_is_not_a_success(self, tmp_path: Path) -> None:
        runner = GROMACSRunner(_FakeStatus(), enabled=False)  # type: ignore[arg-type]
        result = runner.mdrun(deffnm="x", cwd=tmp_path, plumed="plumed.dat")
        assert result.succeeded is False
        assert result.mode == "dry_run"

    def test_a_grompp_failure_stops_the_window(self, tmp_path: Path) -> None:
        from polymer_engine.local.runner import CommandResult

        class FailingGrompp:
            def grompp(self, **kw: object) -> CommandResult:
                return CommandResult(
                    command=["gmx", "grompp"], mode="real", returncode=1,
                    stdout="", stderr="Fatal error: number of coordinates does not match topology",
                    cwd=str(tmp_path),
                )

            def mdrun(self, **kw: object) -> CommandResult:   # pragma: no cover - must not run
                raise AssertionError("mdrun must not be reached after grompp fails")

        window = _window(0, 0.5)
        outcome = make_gromacs_plumed_runner(FailingGrompp())(window, tmp_path)
        assert outcome.succeeded is False
        assert "grompp failed" in (outcome.error or "")

    def test_an_mdrun_failure_is_reported_not_swallowed(self, tmp_path: Path) -> None:
        runner = _stub_runner(tmp_path, mdrun_ok=False, stderr="Fatal error: LINCS warnings")
        outcome = make_gromacs_plumed_runner(runner)(_window(0, 0.5), tmp_path)
        assert outcome.succeeded is False
        assert "mdrun failed" in (outcome.error or "")

    def test_a_plumed_initialisation_failure_is_reported(self, tmp_path: Path) -> None:
        """The exact GROMACS message when PLUMED_KERNEL is unset."""
        runner = _stub_runner(
            tmp_path, mdrun_ok=False,
            stderr=("Internal error (bug):\nAn error occurred while initializing the PLUMED "
                    "force provider:\nYou are trying to use plumed, but it is not available.\n"
                    "Check your PLUMED_KERNEL environment variable."),
        )
        outcome = make_gromacs_plumed_runner(runner)(_window(0, 0.5), tmp_path)
        assert outcome.succeeded is False
        assert "PLUMED_KERNEL" in (outcome.error or "")

    def test_a_missing_colvar_is_a_failure_even_when_mdrun_exits_zero(self, tmp_path: Path) -> None:
        """The headline rule, in the PLUMED path: exit 0 is not success."""
        runner = _stub_runner(tmp_path, mdrun_ok=True, write_colvar=None)
        outcome = make_gromacs_plumed_runner(runner)(_window(0, 0.5), tmp_path)
        assert outcome.succeeded is False
        assert "no COLVAR" in (outcome.error or "")

    def test_an_empty_colvar_is_refused_not_returned_as_zero_samples(self, tmp_path: Path) -> None:
        (tmp_path / "COLVAR").write_text("#! FIELDS time d\n", encoding="utf-8")
        with pytest.raises(InsufficientDataError, match="no usable data"):
            read_colvar(tmp_path / "COLVAR")

    def test_a_header_only_colvar_does_not_become_a_window(self, tmp_path: Path) -> None:
        runner = _stub_runner(tmp_path, mdrun_ok=True, write_colvar="#! FIELDS time d\n")
        outcome = make_gromacs_plumed_runner(runner)(_window(0, 0.5), tmp_path)
        # The file exists, so the runner reports success. Reading it is what refuses,
        # and the campaign turns that refusal into a dropped window rather than a
        # zero-sample window that would silently weight the WHAM histogram.
        assert outcome.succeeded is True
        with pytest.raises(InsufficientDataError):
            read_colvar(outcome.colvar_path or (tmp_path / "COLVAR"))

    def test_a_malformed_colvar_raises_rather_than_dropping_the_row(self, tmp_path: Path) -> None:
        """Regression: unparseable data rows used to be skipped with `continue`.

        A COLVAR truncated by a full disk loses its tail, so silently dropping the bad
        rows keeps exactly the early, least-equilibrated samples and reports them as a
        complete -- merely shorter -- series.
        """
        (tmp_path / "COLVAR").write_text(
            "#! FIELDS time d\n0.0 not-a-number\n0.5 0.31\n", encoding="utf-8"
        )
        with pytest.raises(InsufficientDataError, match="numeric"):
            read_colvar(tmp_path / "COLVAR")

    def test_a_repeated_plumed_header_is_still_accepted(self, tmp_path: Path) -> None:
        """PLUMED re-emits `#! FIELDS` when appending; that is not corruption."""
        (tmp_path / "COLVAR").write_text(
            "#! FIELDS time d\n0.0 0.31\n#! FIELDS time d\n0.5 0.32\n", encoding="utf-8"
        )
        assert read_colvar(tmp_path / "COLVAR").tolist() == [0.31, 0.32]

    def test_a_non_finite_cv_never_reaches_a_free_energy(self, tmp_path: Path) -> None:
        (tmp_path / "COLVAR").write_text(
            "#! FIELDS time d\n0.0 0.31\n0.5 nan\n1.0 inf\n1.5 0.33\n", encoding="utf-8"
        )
        values = read_colvar(tmp_path / "COLVAR")
        # NaN/Inf parse as floats, so they must survive to the finiteness gate rather
        # than being quietly filtered out here.
        assert not np.all(np.isfinite(values)), "NaN/Inf was silently dropped or replaced"
        assert values.size == 4

    def test_a_truncated_colvar_column_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "COLVAR").write_text("#! FIELDS time d\n0.0\n0.5\n", encoding="utf-8")
        with pytest.raises(InsufficientDataError, match="too few columns"):
            read_colvar(tmp_path / "COLVAR")


class TestGpuThreadRegression:
    """Regression: `mdrun -ntomp N` without `-ntmpi` is fatal on a GPU build."""

    def test_ntmpi_is_supplied_automatically_on_a_gpu_build(self, tmp_path: Path) -> None:
        runner = GROMACSRunner(_FakeStatus(has_gpu=True), enabled=False)  # type: ignore[arg-type]
        result = runner.mdrun(deffnm="x", cwd=tmp_path, ntomp=8)
        assert "-ntmpi" in result.command
        assert result.command[result.command.index("-ntmpi") + 1] == "1"

    def test_an_explicit_ntmpi_is_respected(self, tmp_path: Path) -> None:
        runner = GROMACSRunner(_FakeStatus(has_gpu=True), enabled=False)  # type: ignore[arg-type]
        result = runner.mdrun(deffnm="x", cwd=tmp_path, ntomp=8, ntmpi=4)
        assert result.command[result.command.index("-ntmpi") + 1] == "4"

    def test_a_cpu_only_build_is_left_alone(self, tmp_path: Path) -> None:
        """Adding -ntmpi where GROMACS did not need it would change CPU scheduling."""
        runner = GROMACSRunner(_FakeStatus(has_gpu=False), enabled=False)  # type: ignore[arg-type]
        result = runner.mdrun(deffnm="x", cwd=tmp_path, ntomp=8)
        assert "-ntmpi" not in result.command


class TestMinimumImage:
    """The PBC path, deterministically.

    The real-trajectory check above only exercises wrapping when the chain happens to
    straddle a boundary -- 331 of 501 frames in one run, none in another. These cases
    make the requirement explicit, so the independent calculation cannot silently
    degrade into the naive ``|r_i - r_j|`` that disagrees with PLUMED by whole box
    lengths.
    """

    BOX = np.array([3.0, 3.0, 3.0])

    def test_a_wrapped_pair_is_nearer_than_it_looks(self) -> None:
        a, b = np.array([0.1, 1.5, 1.5]), np.array([2.9, 1.5, 1.5])
        naive = float(np.linalg.norm(a - b))
        assert naive == pytest.approx(2.8)
        assert minimum_image_distance(a, b, self.BOX) == pytest.approx(0.2)

    def test_the_distance_is_invariant_under_a_whole_box_translation(self) -> None:
        a, b = np.array([0.4, 1.1, 2.2]), np.array([1.9, 0.3, 0.7])
        reference = minimum_image_distance(a, b, self.BOX)
        for shift in (self.BOX * [1, 0, 0], self.BOX * [0, -1, 0], self.BOX * [2, 1, -3]):
            assert minimum_image_distance(a + shift, b, self.BOX) == pytest.approx(reference)

    def test_an_unwrapped_pair_matches_the_naive_distance(self) -> None:
        a, b = np.array([1.4, 1.5, 1.5]), np.array([1.6, 1.5, 1.5])
        assert minimum_image_distance(a, b, self.BOX) == pytest.approx(float(np.linalg.norm(a - b)))

    def test_no_distance_can_exceed_half_the_box_diagonal(self) -> None:
        limit = float(np.linalg.norm(self.BOX / 2.0))
        rng = np.random.default_rng(0)
        for _ in range(200):
            a, b = rng.uniform(0, 3, 3), rng.uniform(0, 3, 3)
            assert minimum_image_distance(a, b, self.BOX) <= limit + 1e-12


def _window(index: int, center: float, force_constant: float = 1000.0):
    from polymer_engine.simulation.umbrella import Window

    return Window(index=index, center=center, force_constant=force_constant, units="nm")


def _stub_runner(directory: Path, *, mdrun_ok: bool, stderr: str = "", write_colvar: str | None = ""):
    """A GROMACS stand-in whose grompp succeeds and whose mdrun is scripted."""
    from polymer_engine.local.runner import CommandResult

    class Stub:
        def grompp(self, **kw: object) -> CommandResult:
            Path(str(kw.get("cwd", directory)), str(kw.get("output", "x.tpr"))).write_bytes(b"tpr")
            return CommandResult(command=["gmx", "grompp"], mode="real", returncode=0,
                                 stdout="", stderr="", cwd=str(directory))

        def mdrun(self, **kw: object) -> CommandResult:
            if write_colvar is not None and mdrun_ok:
                (directory / "COLVAR").write_text(
                    write_colvar or "#! FIELDS time d\n0.0 0.50\n0.5 0.51\n", encoding="utf-8"
                )
            return CommandResult(
                command=["gmx", "mdrun"], mode="real", returncode=0 if mdrun_ok else 1,
                stdout="", stderr=stderr, cwd=str(directory),
            )

    return Stub()
