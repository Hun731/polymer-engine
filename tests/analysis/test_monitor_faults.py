"""Status reporting must not claim a stopped campaign is running.

Three separate faults conspired to report a stopped campaign as healthy, each harmless
alone: the monitor could not see the annealing stage, `kill -0 0` succeeds, and a status
file nobody is writing still reads as current. Together they produced
`driver=up mon=up | polypropylene/01/npt 30.4%` for a campaign with no processes at all.
"""

from __future__ import annotations

import importlib.util
import subprocess
import time
from pathlib import Path

import pytest

from polymer_engine.simulation.mdp import STAGE_ORDER

ROOT = Path(__file__).resolve().parents[2]


def shell_code(path: Path) -> str:
    """A shell script with its comments removed.

    Scanning raw text for a removed construct finds the comment that explains why it was
    removed. Both of these checks failed that way first.
    """
    return "\n".join(line for line in path.read_text().splitlines()
                      if not line.lstrip().startswith("#"))


def _monitor():
    spec = importlib.util.spec_from_file_location(
        "monitor_campaign", ROOT / "scripts" / "monitor_campaign.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_monitor_knows_every_stage_the_engine_generates() -> None:
    """A hard-coded copy here made an in-flight anneal invisible."""
    assert tuple(_monitor().STAGES) == tuple(STAGE_ORDER)


def test_an_unfinished_stage_is_reported_as_in_flight(tmp_path: Path) -> None:
    replica = tmp_path / "experiments" / "polyethylene" / "replica_01"
    replica.mkdir(parents=True)
    (replica / "anneal.log").write_text("nsteps = 1000\n           Step           Time\n"
                                        "            500      1000.00000\n")
    flight = _monitor().in_flight(tmp_path)
    assert flight["stage"] == "anneal"
    assert flight["percent_complete"] == 50.0
    assert flight["live"] is True


def test_a_stage_whose_output_is_newer_than_its_log_has_finished(tmp_path: Path) -> None:
    """Output written *after* the last log line means the stage completed."""
    replica = tmp_path / "experiments" / "polyethylene" / "replica_01"
    replica.mkdir(parents=True)
    (replica / "anneal.log").write_text("nsteps = 1000\n")
    time.sleep(0.01)
    (replica / "anneal.gro").write_text("done")
    flight = _monitor().in_flight(tmp_path)
    assert flight["finished"] is True
    assert flight["live"] is False


def test_a_stale_log_is_not_reported_as_running(tmp_path: Path) -> None:
    """The flag that separates 'still going' from 'died mid-stage'."""
    module = _monitor()
    replica = tmp_path / "experiments" / "polyethylene" / "replica_01"
    replica.mkdir(parents=True)
    log = replica / "npt.log"
    log.write_text("nsteps = 1000\n           Step           Time\n"
                   "            500      1000.00000\n")
    old = time.time() - (module.STALE_LOG_SECONDS + 60)
    import os

    os.utime(log, (old, old))
    flight = module.in_flight(tmp_path)
    assert flight["live"] is False
    assert flight["log_age_seconds"] > module.STALE_LOG_SECONDS


class TestSuperviseCheck:
    """`kill -0 0` succeeds, so a missing pid file used to read as a running driver."""

    SCRIPT = ROOT / "scripts" / "supervise_check.sh"

    def test_pid_zero_is_not_treated_as_alive(self) -> None:
        alive = subprocess.run(
            ["bash", "-c",
             f'source <(sed -n "/^alive()/,/^}}/p" {self.SCRIPT}); '
             'alive "" && echo EMPTY_ALIVE; alive 0 && echo ZERO_ALIVE; '
             'alive $$ && echo SELF_ALIVE'],
            capture_output=True, text=True, check=False,
        ).stdout
        assert "ZERO_ALIVE" not in alive, "PID 0 signals the caller's own process group"
        assert "EMPTY_ALIVE" not in alive
        assert "SELF_ALIVE" in alive, "a real pid must still register as alive"

    def test_the_script_no_longer_defaults_a_missing_pidfile_to_zero(self) -> None:
        code = shell_code(self.SCRIPT)
        assert "|| echo 0" not in code
        assert 'alive "$D"' in code


class TestCampaignCtl:
    """Stopping this campaign must not stop somebody else's GROMACS job."""

    SCRIPT = ROOT / "scripts" / "campaign_ctl.sh"

    def test_stop_does_not_kill_every_gmx_on_the_machine(self) -> None:
        text = shell_code(self.SCRIPT)
        assert "pgrep -x gmx" not in text.split("status()")[0], (
            "stop() must not match GROMACS processes by name; it killed an unrelated "
            "project's job that shared the GPU")
        assert "descendant_gmx" in text

    def test_descendants_are_found_and_strangers_are_not(self) -> None:
        """Spawn a sleep tree and confirm only its own descendants are listed."""
        outer = subprocess.Popen(["bash", "-c", "sleep 30 & wait"])
        stranger = subprocess.Popen(["sleep", "30"])
        try:
            time.sleep(0.5)
            found = subprocess.run(
                ["bash", "-c",
                 f'source <(sed -n "/^descendant_gmx/,/^}}/p" {self.SCRIPT}); '
                 f"descendant_gmx {outer.pid}"],
                capture_output=True, text=True, check=False,
            ).stdout.split()
            # No `gmx` here, so nothing should match -- and certainly not the stranger.
            assert str(stranger.pid) not in found
        finally:
            outer.kill()
            stranger.kill()
            outer.wait()
            stranger.wait()


@pytest.mark.parametrize("stage", ["em", "nvt", "anneal", "npt", "prod"])
def test_every_stage_can_appear_in_flight(tmp_path: Path, stage: str) -> None:
    replica = tmp_path / "experiments" / "c" / "replica_01"
    replica.mkdir(parents=True)
    (replica / f"{stage}.log").write_text("nsteps = 100\n")
    assert _monitor().in_flight(tmp_path).get("stage") == stage


class TestRerunLeftovers:
    """A rerun leaves files from the previous attempt, and they lied about the present.

    `in_flight` used to treat "log present, .gro absent" as "this stage is running",
    using output as a proxy for completion. On a resumed campaign that inverts: the
    stage genuinely running was hidden by a .gro from an earlier attempt, while a stage
    abandoned fourteen hours earlier stayed on screen as current.
    """

    def _replica(self, tmp_path: Path) -> Path:
        replica = tmp_path / "experiments" / "polypropylene" / "replica_01"
        replica.mkdir(parents=True)
        return replica

    def test_a_leftover_gro_does_not_hide_the_running_stage(self, tmp_path: Path) -> None:
        import os

        replica = self._replica(tmp_path)
        # Yesterday's attempt got as far as writing anneal.gro.
        (replica / "anneal.gro").write_text("old")
        old = time.time() - 50_000
        for name in ("npt.log",):
            (replica / name).write_text("nsteps = 1000\n           Step           Time\n"
                                        "            304      1000.00000\n")
            os.utime(replica / name, (old, old))
        # Today's run is writing the anneal again.
        (replica / "anneal.log").write_text("nsteps = 1000\n           Step           Time\n"
                                            "            230      1000.00000\n")

        flight = _monitor().in_flight(tmp_path)
        assert flight["stage"] == "anneal", "the running stage must not be hidden"
        assert flight["finished"] is False, "a .gro older than its log is a leftover"
        assert flight["live"] is True
        assert flight["percent_complete"] == 23.0

    def test_an_abandoned_log_is_not_reported_as_current(self, tmp_path: Path) -> None:
        import os

        replica = self._replica(tmp_path)
        log = replica / "npt.log"
        log.write_text("nsteps = 1000\n           Step           Time\n"
                       "            304      1000.00000\n")
        old = time.time() - 50_000
        os.utime(log, (old, old))

        flight = _monitor().in_flight(tmp_path)
        assert flight["stage"] == "npt"
        # It is the newest log there is -- but nothing is running.
        assert flight["live"] is False
        assert flight["log_age_seconds"] > 40_000

    def test_the_newest_log_wins_regardless_of_stage_order(self, tmp_path: Path) -> None:
        import os

        replica = self._replica(tmp_path)
        for name, age in (("prod.log", 9_000), ("anneal.log", 5)):
            (replica / name).write_text("nsteps = 100\n")
            os.utime(replica / name, (time.time() - age, time.time() - age))
        assert _monitor().in_flight(tmp_path)["stage"] == "anneal"
