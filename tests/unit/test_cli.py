"""CLI surface: commands, output shape, and exit codes.

Exit codes are part of the contract, so CI can distinguish "the tool broke" (1),
"you used it wrong" (2), and "the science did not pass" (3).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from polymer_engine.cli.main import EXIT_ERROR, EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE, app
from tests.markers import requires_rdkit

VALID_SYSTEM = Path(__file__).resolve().parents[1] / "fixtures" / "systems" / "valid_system"
DATASET = Path(__file__).resolve().parents[1] / "fixtures" / "datasets" / "polymers.csv"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("POLYMER_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYMER_HTTP_OFFLINE", "true")
    monkeypatch.delenv("POLYMER_CONFIG", raising=False)
    return tmp_path


def payload(result) -> dict:
    return json.loads(result.stdout)


class TestEngineCommands:
    def test_help_lists_every_command_group(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == EXIT_OK
        for group in ("engine", "polymer", "provider", "system", "campaign", "umbrella", "model", "design", "evidence"):
            assert group in result.stdout

    def test_init_creates_the_workspace_and_database(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["engine", "init"])
        assert result.exit_code == EXIT_OK, result.stdout
        body = payload(result)
        assert body["initialised"] is True
        assert Path(body["database"]).exists()
        assert body["objective_id"].startswith("obj_")

    def test_status_without_init_fails_with_a_hint(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["engine", "status"])
        assert result.exit_code == EXIT_ERROR
        assert "engine init" in result.stderr

    def test_status_after_init(self, runner: CliRunner, workspace: Path) -> None:
        runner.invoke(app, ["engine", "init"])
        result = runner.invoke(app, ["engine", "status"])
        assert result.exit_code == EXIT_OK
        assert payload(result)["counts"]["objectives"] == 1

    def test_config_output_is_redacted(self, runner: CliRunner, workspace: Path, monkeypatch) -> None:
        monkeypatch.setenv("CHARMM_GUI_TOKEN", "super-secret-token-value")
        result = runner.invoke(app, ["engine", "config"])
        assert result.exit_code == EXIT_OK
        assert "super-secret-token-value" not in result.stdout
        assert payload(result)["credentials"]["charmm_gui_token"] == "configured"

    def test_providers_reports_capabilities_and_gaps(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["engine", "providers"])
        assert result.exit_code == EXIT_OK
        body = payload(result)
        assert "job_submission" in body["unsupported"]["charmm_gui"]
        assert body["status"]["pubchem"]["configured"] is True

    def test_discover_tools_reports_each_tool(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["engine", "discover-tools"])
        assert result.exit_code in {EXIT_OK, EXIT_GATE_FAILED}
        body = payload(result)
        assert set(body) == {"gromacs", "orca", "plumed", "python"}
        for entry in body.values():
            assert "found" in entry and "usable" in entry


@requires_rdkit
class TestPolymerCommands:
    def test_descriptors(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["polymer", "descriptors", "*CC(*)c1ccccc1"])
        assert result.exit_code == EXIT_OK
        body = payload(result)
        assert body["family"]["family"] == "polystyrenic"
        assert body["descriptors"]["descriptors"]["repeat_unit_mass"]["value"] == pytest.approx(106.17, abs=0.02)

    def test_descriptors_rejects_a_non_polymer_with_a_hint(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["polymer", "descriptors", "not-a-smiles("])
        assert result.exit_code == EXIT_ERROR
        assert "attachment points" in result.stderr

    def test_ingest_reports_reconciled_counts(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["polymer", "ingest", str(DATASET)])
        assert result.exit_code == EXIT_OK
        body = payload(result)
        assert body["rows_read"] == body["accepted"] + body["rejected"] + body["duplicates"]
        assert body["reconciles"] is True

    def test_ingest_missing_file_is_a_usage_error(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["polymer", "ingest", "/nonexistent.csv"])
        assert result.exit_code == EXIT_USAGE

    def test_cluster_groups_by_family(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["polymer", "cluster", str(DATASET)])
        assert result.exit_code == EXIT_OK
        assert "polyolefin" in payload(result)["families"]


class TestSystemCommands:
    def test_validate_a_good_system(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["system", "validate", str(VALID_SYSTEM)])
        assert result.exit_code == EXIT_OK
        assert payload(result)["promotable"] is True

    def test_validate_a_broken_system_exits_with_the_gate_code(
        self, runner: CliRunner, workspace: Path, tmp_path: Path
    ) -> None:
        broken = tmp_path / "broken"
        shutil.copytree(VALID_SYSTEM, broken)
        (broken / "topol.top").unlink()
        result = runner.invoke(app, ["system", "validate", str(broken)])
        assert result.exit_code == EXIT_GATE_FAILED
        assert payload(result)["promotable"] is False

    def test_validate_a_non_directory_is_a_usage_error(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["system", "validate", str(VALID_SYSTEM / "system.gro")])
        assert result.exit_code == EXIT_USAGE

    def test_import_a_directory(self, runner: CliRunner, workspace: Path) -> None:
        runner.invoke(app, ["engine", "init"])
        result = runner.invoke(app, ["system", "import", str(VALID_SYSTEM)])
        assert result.exit_code == EXIT_OK
        assert payload(result)["validation"]["promotable"] is True

    def test_import_an_unsafe_archive_fails(self, runner: CliRunner, workspace: Path, tmp_path: Path) -> None:
        import io
        import tarfile

        runner.invoke(app, ["engine", "init"])
        archive = tmp_path / "evil.tgz"
        with tarfile.open(archive, "w:gz") as tf:
            info = tarfile.TarInfo("../escaped.txt")
            info.size = 3
            tf.addfile(info, io.BytesIO(b"bad"))
        result = runner.invoke(app, ["system", "import", str(archive)])
        assert result.exit_code == EXIT_ERROR
        assert not (tmp_path / "escaped.txt").exists()


class TestCampaignCommands:
    def test_full_lifecycle(self, runner: CliRunner, workspace: Path) -> None:
        assert runner.invoke(app, ["engine", "init"]).exit_code == EXIT_OK

        created = runner.invoke(
            app,
            ["campaign", "create", "c1", "--polymer", "pol_x", "--system", str(VALID_SYSTEM),
             "--question", "What is the density?"],
        )
        assert created.exit_code == EXIT_OK, created.stdout
        assert payload(created)["status"] == "system_validated"

        planned = runner.invoke(app, ["campaign", "plan", "c1"])
        assert planned.exit_code == EXIT_OK, planned.stdout
        body = payload(planned)
        assert body["replicas"] == 3
        assert body["seeds_distinct"] is True
        assert Path(body["manifest"]).exists()

        ran = runner.invoke(app, ["campaign", "run", "c1"])
        assert ran.exit_code == EXIT_OK
        for entry in payload(ran)["executed"]:
            assert entry["execution_mode"] == "dry_run"
            assert entry["scientifically_usable"] is False

        status = runner.invoke(app, ["campaign", "status", "c1"])
        assert status.exit_code == EXIT_OK
        assert payload(status)["decisions"]

    def test_status_of_an_unknown_campaign_is_a_usage_error(self, runner: CliRunner, workspace: Path) -> None:
        runner.invoke(app, ["engine", "init"])
        result = runner.invoke(app, ["campaign", "status", "nope"])
        assert result.exit_code == EXIT_USAGE

    def test_plan_before_create_is_a_usage_error(self, runner: CliRunner, workspace: Path) -> None:
        runner.invoke(app, ["engine", "init"])
        result = runner.invoke(app, ["campaign", "plan", "nope"])
        assert result.exit_code == EXIT_USAGE
        assert "campaign create" in result.stderr

    def test_manifest_is_reproducible_and_complete(self, runner: CliRunner, workspace: Path) -> None:
        runner.invoke(app, ["engine", "init"])
        runner.invoke(app, ["campaign", "create", "c2", "--polymer", "pol_x", "--system", str(VALID_SYSTEM)])
        runner.invoke(app, ["campaign", "plan", "c2"])
        result = runner.invoke(app, ["evidence", "manifest", "c2"])
        assert result.exit_code == EXIT_OK
        manifest = payload(result)
        assert manifest["fingerprint"]
        assert manifest["random_seeds"]
        assert manifest["specification"]["simulation"]["barostat"]
        assert manifest["replicas"]["seeds_distinct"] is True


@requires_rdkit
class TestModelAndDesignCommands:
    def test_model_evaluate_builds_a_dataset_and_scores_it(self, runner: CliRunner, workspace: Path) -> None:
        """Regression: the feature matrix must be built in feature_names order."""
        dataset = Path(__file__).resolve().parents[2] / "examples" / "polyethylene_density" / "polymers.csv"
        result = runner.invoke(
            app, ["model", "evaluate", str(dataset), "--target", "glass transition temperature"]
        )
        assert result.exit_code == EXIT_OK, result.stdout + result.stderr
        body = payload(result)
        assert body["cross_validation"]["n_samples"] >= 10
        assert body["duplicate_polymer_ids"] == []

    def test_model_evaluate_rejects_an_unknown_property(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["model", "evaluate", str(DATASET), "--target", "vibes"])
        assert result.exit_code == EXIT_USAGE
        assert "known properties" in result.stderr

    def test_design_generate_reports_candidate_statuses(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["design", "generate", "*CC*", "--max", "6"])
        assert result.exit_code == EXIT_OK
        body = payload(result)
        assert body["n_generated"] == 6
        assert sum(body["counts"].values()) == 6

    def test_design_generate_rejects_a_bad_parent(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["design", "generate", "CCO"])
        assert result.exit_code == EXIT_ERROR
        assert "attachment points" in result.stderr

    def test_design_rank_scores_candidates_against_a_surrogate(
        self, runner: CliRunner, workspace: Path
    ) -> None:
        dataset = Path(__file__).resolve().parents[2] / "examples" / "polyethylene_density" / "polymers.csv"
        result = runner.invoke(
            app,
            ["design", "rank", str(dataset), "--parent", "*CC*",
             "--target", "glass transition temperature", "--batch-size", "3"],
        )
        assert result.exit_code == EXIT_OK, result.stdout + result.stderr
        body = payload(result)
        assert len(body["selected"]) <= 3
        assert "weights" in body["normalisation"]


class TestUmbrellaCommands:
    def test_plan_writes_windows(self, runner: CliRunner, workspace: Path, tmp_path: Path) -> None:
        output = tmp_path / "windows"
        result = runner.invoke(
            app,
            ["umbrella", "plan", str(output), "--min", "0.4", "--max", "1.2",
             "--justification", "Interchain separation probes cohesion."],
        )
        assert result.exit_code == EXIT_OK, result.stdout
        body = payload(result)
        assert body["expected_to_overlap"] is True
        assert (output / "umbrella_plan.json").exists()
        assert len(list(output.glob("window_*"))) == body["n_windows"]

    def test_analyze_with_too_few_windows_fails_clearly(
        self, runner: CliRunner, workspace: Path, tmp_path: Path
    ) -> None:
        empty = tmp_path / "nothing"
        empty.mkdir()
        result = runner.invoke(app, ["umbrella", "analyze", str(empty)])
        assert result.exit_code == EXIT_ERROR
        assert "window.json" in result.stderr


class TestEvidenceCommands:
    def test_report_on_an_empty_ledger(self, runner: CliRunner, workspace: Path) -> None:
        runner.invoke(app, ["engine", "init"])
        result = runner.invoke(app, ["evidence", "report"])
        assert result.exit_code == EXIT_OK
        assert payload(result)["n_claims"] == 0

    def test_strategies_are_listed_with_their_records(self, runner: CliRunner, workspace: Path) -> None:
        runner.invoke(app, ["engine", "init"])
        result = runner.invoke(app, ["evidence", "strategies"])
        assert result.exit_code == EXIT_OK
        strategies = payload(result)["strategies"]
        assert len(strategies) >= 5
        assert all("score" in s and "determination" in s for s in strategies)

    def test_lineage_for_an_unknown_artifact_is_a_usage_error(self, runner: CliRunner, workspace: Path) -> None:
        runner.invoke(app, ["engine", "init"])
        result = runner.invoke(app, ["evidence", "lineage", "art_nope"])
        assert result.exit_code == EXIT_USAGE


@requires_rdkit
class TestNewCommandGroups:
    def test_property_list_reports_every_contract(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["property", "list"])
        assert result.exit_code == EXIT_OK
        definitions = payload(result)
        assert len(definitions) >= 15
        for name, definition in definitions.items():
            assert definition["units"], name
            assert definition["uncertainty_method"], name
            assert definition["sampling"]["min_replicas"] >= 1, name

    def test_qm_parse_exits_three_for_a_job_that_exited_zero_but_failed(
        self, runner: CliRunner, workspace: Path
    ) -> None:
        """The headline case: ORCA returned 0, the science did not."""
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "orca" / "scan_nonconverged.out"
        result = runner.invoke(app, ["qm", "parse", str(fixture), "--expect-geometry"])
        assert result.exit_code == EXIT_GATE_FAILED
        assert payload(result)["status"] == "FAILED_TERMINATION"

    def test_qm_parse_succeeds_for_a_good_job(self, runner: CliRunner, workspace: Path) -> None:
        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "orca" / "opt.out.gz"
        result = runner.invoke(app, ["qm", "parse", str(fixture), "--expect-geometry"])
        assert result.exit_code == EXIT_OK
        assert payload(result)["geometry_converged"] is True

    def test_qm_run_requires_a_method_and_basis(self, runner: CliRunner, workspace: Path, tmp_path: Path) -> None:
        """The engine will not choose a level of theory."""
        xyz = tmp_path / "h2.xyz"
        xyz.write_text("2\nh2\nH 0 0 0\nH 0 0 0.74\n")
        result = runner.invoke(app, ["qm", "run", str(xyz)])
        assert result.exit_code != EXIT_OK

    def test_qm_run_rejects_an_unknown_job_kind(self, runner: CliRunner, workspace: Path, tmp_path: Path) -> None:
        xyz = tmp_path / "h2.xyz"
        xyz.write_text("2\nh2\nH 0 0 0\nH 0 0 0.74\n")
        result = runner.invoke(
            app, ["qm", "run", str(xyz), "--kind", "telepathy", "--method", "HF", "--basis", "STO-3G"]
        )
        assert result.exit_code == EXIT_USAGE

    def test_qm_run_writes_an_input_even_when_execution_is_disabled(
        self, runner: CliRunner, workspace: Path, tmp_path: Path
    ) -> None:
        xyz = tmp_path / "h2.xyz"
        xyz.write_text("2\nh2\nH 0 0 0\nH 0 0 0.74\n")
        runner.invoke(app, ["engine", "init"])
        result = runner.invoke(
            app,
            ["qm", "run", str(xyz), "--method", "HF", "--basis", "STO-3G",
             "--workdir", str(tmp_path / "qm")],
        )
        assert result.exit_code != EXIT_OK  # nothing ran, so nothing succeeded
        assert (tmp_path / "qm" / "h2.inp").exists()

    def test_research_failures_reports_an_empty_ledger(self, runner: CliRunner, workspace: Path) -> None:
        runner.invoke(app, ["engine", "init"])
        result = runner.invoke(app, ["research", "failures"])
        assert result.exit_code == EXIT_OK
        assert payload(result)["n_failures"] == 0

    def test_research_state_without_a_run_is_a_usage_error(self, runner: CliRunner, workspace: Path) -> None:
        result = runner.invoke(app, ["research", "state", str(workspace / "absent.json")])
        assert result.exit_code == EXIT_USAGE

    def test_research_correlate_reports_the_multiplicity(self, runner: CliRunner, workspace: Path) -> None:
        dataset = (
            Path(__file__).resolve().parents[2] / "examples" / "benchmark_campaign" / "candidates.csv"
        )
        result = runner.invoke(app, ["research", "correlate", str(dataset), "--target", "density"])
        assert result.exit_code == EXIT_OK
        body = payload(result)
        assert body["n_tests"] > 1
        assert "significant_after_correction" in body

    def test_property_compute_gates_an_undersampled_series(
        self, runner: CliRunner, workspace: Path, tmp_path: Path
    ) -> None:
        xvg = tmp_path / "density.xvg"
        xvg.write_text(
            '@    title "Density"\n' + "\n".join(f"{i * 10.0} {1000 + i * 0.01}" for i in range(30))
        )
        result = runner.invoke(app, ["property", "compute", "density", str(xvg), "--replicas", "1"])
        assert result.exit_code == EXIT_GATE_FAILED
        assert payload(result)["usable"] is False

    def test_property_compute_rejects_an_unknown_property(
        self, runner: CliRunner, workspace: Path, tmp_path: Path
    ) -> None:
        xvg = tmp_path / "x.xvg"
        xvg.write_text("0 1\n1 2\n")
        result = runner.invoke(app, ["property", "compute", "vibes", str(xvg)])
        assert result.exit_code == EXIT_USAGE
