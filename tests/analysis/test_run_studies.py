"""The post-campaign study runner: connectors, not calculators.

The calculators are tested where they live. What broke in practice was everything
around them: detecting that a replica finished (mdrun appends to the log *after*
writing the final .gro), reading the verdict out of a PropertyResult dict, and the two
PBC corrections without which Rg and MSD are quietly wrong.
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _studies():
    import sys

    # Registered in sys.modules before execution: @dataclass resolves its own module
    # through sys.modules at class-creation time, and an unregistered module makes
    # every dataclass in the script fail to define.
    if "run_studies" in sys.modules:
        return sys.modules["run_studies"]
    spec = importlib.util.spec_from_file_location(
        "run_studies", ROOT / "scripts" / "run_studies.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_studies"] = module
    spec.loader.exec_module(module)
    return module


class TestReplicaFinished:
    def _replica(self, tmp_path: Path, *, log_age: float, gro: bool,
                 gro_before_log: float = 2.0) -> Path:
        replica = tmp_path / "replica_01"
        replica.mkdir()
        now = time.time()
        (replica / "prod.log").write_text("done")
        os.utime(replica / "prod.log", (now - log_age, now - log_age))
        if gro:
            (replica / "prod.gro").write_text("coords")
            t = now - log_age - gro_before_log
            os.utime(replica / "prod.gro", (t, t))
        return replica

    def test_a_finished_run_counts_even_though_the_log_is_newer(self, tmp_path) -> None:
        """mdrun writes the .gro, then appends statistics to the log."""
        replica = self._replica(tmp_path, log_age=600, gro=True)
        assert _studies().replica_finished(replica)

    def test_a_run_still_writing_its_log_is_not_finished(self, tmp_path) -> None:
        replica = self._replica(tmp_path, log_age=5, gro=True)
        assert not _studies().replica_finished(replica)

    def test_an_interrupted_run_has_no_output(self, tmp_path) -> None:
        replica = self._replica(tmp_path, log_age=600, gro=False)
        assert not _studies().replica_finished(replica)

    def test_a_gro_left_by_a_much_earlier_attempt_does_not_count(self, tmp_path) -> None:
        replica = self._replica(tmp_path, log_age=600, gro=True, gro_before_log=9000)
        assert not _studies().replica_finished(replica)


class TestVerdictExtraction:
    """PropertyResult.as_dict has gate_status; the first version read a key that
    does not exist and printed '?' for every study."""

    def test_gate_status_is_read(self) -> None:
        assert _studies()._verdict({"gate_status": "pass"}) == "pass"

    def test_a_result_with_no_gates_is_not_invented(self) -> None:
        assert _studies()._verdict({}) == "?"

    def test_real_results_never_show_question_marks(self) -> None:
        """Against the study output actually produced from run4, if present."""
        import json

        produced = sorted((ROOT / "campaign/run4/studies").glob("*_replica_*.json"))
        if not produced:
            pytest.skip("no study output on disk")
        payload = json.loads(produced[0].read_text())
        module = _studies()
        for name, result in payload["results"].items():
            if name == "rdf_carbon_carbon":
                continue
            assert module._verdict(result) != "?", name


class TestProductionWindow:
    def test_half_is_discarded_like_the_campaign_does(self) -> None:
        module = _studies()
        values = np.arange(100)
        assert values[module.production_window(values)][0] == 50

    def test_the_discard_fraction_matches_the_campaign_convention(self) -> None:
        assert _studies().DISCARD_FRACTION == 0.5


class TestScientificRefusals:
    """The runner must inherit the calculators' refusals, not soften them."""

    def test_a_subdiffusive_msd_never_yields_a_diffusion_coefficient(self) -> None:
        from polymer_engine.properties.transport import DiffusionCoefficient

        lag = np.linspace(1, 1000, 200)
        msd = 0.01 * lag**0.5          # Rouse-like, alpha = 0.5
        result = DiffusionCoefficient().compute(lag, msd)
        assert result.measurement is None or not result.usable

    def test_bulk_modulus_refuses_a_constant_volume(self) -> None:
        from polymer_engine.properties.mechanical import BulkModulus

        result = BulkModulus().compute(np.full(100, 50.0), 300.0)
        assert not result.usable
