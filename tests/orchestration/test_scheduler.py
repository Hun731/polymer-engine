"""Scheduler: resource limits, dependencies, retries, cancellation, priority.

The central guarantee is that configured resources are never oversubscribed. Several
of these tests assert that by counting how many handlers are inside the critical
section at once, not merely by inspecting the report afterwards.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from polymer_engine.core.errors import PolymerEngineError, ScientificError
from polymer_engine.orchestrator.scheduler import (
    Job,
    JobState,
    LocalExecutor,
    ResourcePool,
    ResourceRequest,
    Scheduler,
    build_pool,
)


class ConcurrencyProbe:
    """Counts how many handlers run simultaneously, and how much they hold."""

    def __init__(self, delay: float = 0.03) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.active_gpus = 0
        self.max_gpus = 0
        self.order: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, job: Job) -> str:
        with self._lock:
            self.active += 1
            self.active_gpus += job.resources.gpus
            self.max_active = max(self.max_active, self.active)
            self.max_gpus = max(self.max_gpus, self.active_gpus)
            self.order.append(job.job_id)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
            self.active_gpus -= job.resources.gpus
        return f"result:{job.job_id}"


def pool(**kwargs) -> ResourcePool:
    base = {
        "total_cpus": 8, "total_gpus": 1, "total_memory_mb": 16000,
        "total_disk_mb": 100_000, "vram_mb_per_gpu": 24000,
    }
    base.update(kwargs)
    return ResourcePool(**base)


# ==========================================================================
# Resource accounting
# ==========================================================================
class TestResourcePool:
    def test_reserve_and_release_round_trip(self) -> None:
        p = pool()
        demand = ResourceRequest(cpus=4, gpus=1, memory_mb=2000)
        p.reserve(demand)
        assert p.free_cpus == 4 and p.free_gpus == 0
        p.release(demand)
        assert p.free_cpus == 8 and p.free_gpus == 1

    def test_cannot_reserve_more_than_is_free(self) -> None:
        p = pool()
        p.reserve(ResourceRequest(cpus=8))
        assert p.can_reserve(ResourceRequest(cpus=1)) is False
        with pytest.raises(PolymerEngineError):
            p.reserve(ResourceRequest(cpus=1))

    def test_double_release_does_not_manufacture_capacity(self) -> None:
        p = pool()
        demand = ResourceRequest(cpus=4)
        p.reserve(demand)
        p.release(demand)
        p.release(demand)
        assert p.used_cpus == 0
        assert p.free_cpus == p.total_cpus

    def test_a_request_larger_than_the_machine_is_rejected(self) -> None:
        assert "CPUs" in pool(total_cpus=4).exceeds_capacity(ResourceRequest(cpus=64))

    def test_gpu_demand_beyond_the_machine_is_rejected(self) -> None:
        assert "GPU" in pool(total_gpus=1).exceeds_capacity(ResourceRequest(gpus=4))

    def test_vram_cannot_be_pooled_across_cards(self) -> None:
        """Two 10 GB cards cannot run one 15 GB job."""
        p = pool(total_gpus=2, vram_mb_per_gpu=10_000)
        reason = p.exceeds_capacity(ResourceRequest(gpus=1, vram_mb=15_000))
        assert "VRAM" in reason and "cannot be pooled" in reason

    def test_a_fitting_request_is_accepted(self) -> None:
        assert pool().exceeds_capacity(ResourceRequest(cpus=2, gpus=1, memory_mb=1000)) == ""

    def test_zero_cpus_is_rejected(self) -> None:
        with pytest.raises(ScientificError):
            ResourceRequest(cpus=0)

    def test_negative_demand_is_rejected(self) -> None:
        with pytest.raises(ScientificError):
            ResourceRequest(cpus=1, gpus=-1)

    def test_pool_needs_at_least_one_cpu(self) -> None:
        with pytest.raises(ScientificError):
            ResourcePool(total_cpus=0)


# ==========================================================================
# Never oversubscribe
# ==========================================================================
class TestNoOversubscription:
    def test_four_gpu_jobs_on_one_gpu_run_one_at_a_time(self) -> None:
        probe = ConcurrencyProbe()
        scheduler = Scheduler(pool(total_gpus=1), LocalExecutor(probe))
        scheduler.submit_all(
            [Job(job_id=f"replica{i}", kind="md", resources=ResourceRequest(cpus=2, gpus=1))
             for i in range(4)]
        )
        report = scheduler.run()
        assert probe.max_gpus == 1, "the GPU was oversubscribed"
        assert probe.max_active == 1
        assert len(report.by_state(JobState.SUCCEEDED)) == 4

    def test_two_gpu_jobs_on_two_gpus_run_together(self) -> None:
        probe = ConcurrencyProbe()
        scheduler = Scheduler(pool(total_gpus=2), LocalExecutor(probe))
        scheduler.submit_all(
            [Job(job_id=f"r{i}", kind="md", resources=ResourceRequest(cpus=1, gpus=1)) for i in range(2)]
        )
        scheduler.run()
        assert probe.max_gpus == 2

    def test_cpu_limits_are_respected(self) -> None:
        probe = ConcurrencyProbe()
        scheduler = Scheduler(pool(total_cpus=4, total_gpus=0), LocalExecutor(probe))
        scheduler.submit_all(
            [Job(job_id=f"c{i}", kind="cpu", resources=ResourceRequest(cpus=2)) for i in range(6)]
        )
        scheduler.run()
        assert probe.max_active <= 2

    def test_memory_limits_are_respected(self) -> None:
        probe = ConcurrencyProbe()
        scheduler = Scheduler(
            pool(total_cpus=32, total_gpus=0, total_memory_mb=4000), LocalExecutor(probe)
        )
        scheduler.submit_all(
            [Job(job_id=f"m{i}", kind="mem", resources=ResourceRequest(cpus=1, memory_mb=1500))
             for i in range(6)]
        )
        scheduler.run()
        assert probe.max_active <= 2

    def test_max_concurrent_jobs_caps_below_the_resource_limit(self) -> None:
        probe = ConcurrencyProbe()
        scheduler = Scheduler(
            pool(total_cpus=32, total_gpus=0), LocalExecutor(probe), max_concurrent_jobs=2
        )
        scheduler.submit_all(
            [Job(job_id=f"j{i}", kind="x", resources=ResourceRequest(cpus=1)) for i in range(8)]
        )
        scheduler.run()
        assert probe.max_active <= 2

    def test_an_impossible_job_is_rejected_not_queued(self) -> None:
        """Queuing an unsatisfiable job would deadlock the scheduler."""
        scheduler = Scheduler(pool(total_gpus=1), LocalExecutor(ConcurrencyProbe()))
        job = scheduler.submit(Job(job_id="huge", kind="md", resources=ResourceRequest(cpus=1, gpus=8)))
        assert job.state is JobState.REJECTED
        assert "cannot be satisfied" in job.error

    def test_a_rejected_job_does_not_stall_the_queue(self) -> None:
        probe = ConcurrencyProbe()
        scheduler = Scheduler(pool(total_gpus=1), LocalExecutor(probe))
        scheduler.submit(Job(job_id="huge", kind="md", resources=ResourceRequest(gpus=8)))
        scheduler.submit(Job(job_id="ok", kind="md", resources=ResourceRequest(cpus=1)))
        scheduler.run()
        assert scheduler.get("ok").state is JobState.SUCCEEDED
        assert scheduler.get("huge").state is JobState.REJECTED

    def test_resources_are_released_when_a_job_fails(self) -> None:
        """A leaked reservation would silently shrink the machine."""
        def failing(job: Job) -> Any:
            raise RuntimeError("boom")

        p = pool(total_cpus=4, total_gpus=1)
        scheduler = Scheduler(p, LocalExecutor(failing))
        scheduler.submit_all(
            [Job(job_id=f"f{i}", kind="x", resources=ResourceRequest(cpus=4, gpus=1)) for i in range(3)]
        )
        scheduler.run()
        assert p.used_cpus == 0
        assert p.used_gpus == 0


# ==========================================================================
# Dependencies
# ==========================================================================
class TestDependencies:
    def test_a_dependent_job_runs_after_its_dependency(self) -> None:
        probe = ConcurrencyProbe(delay=0.01)
        scheduler = Scheduler(pool(total_cpus=16, total_gpus=0), LocalExecutor(probe))
        scheduler.submit(Job(job_id="first", kind="a"))
        scheduler.submit(Job(job_id="second", kind="b", depends_on=("first",)))
        scheduler.run()
        assert probe.order.index("first") < probe.order.index("second")

    def test_a_failed_dependency_blocks_its_dependents(self) -> None:
        """Running on missing inputs is how a pipeline produces nonsense."""
        def selective(job: Job) -> Any:
            if job.job_id == "first":
                raise RuntimeError("upstream failed")
            return "ok"

        scheduler = Scheduler(pool(), LocalExecutor(selective))
        scheduler.submit(Job(job_id="first", kind="a"))
        scheduler.submit(Job(job_id="second", kind="b", depends_on=("first",)))
        scheduler.run()
        assert scheduler.get("first").state is JobState.FAILED
        assert scheduler.get("second").state is JobState.BLOCKED
        assert "dependency" in scheduler.get("second").error

    def test_a_missing_dependency_blocks(self) -> None:
        scheduler = Scheduler(pool(), LocalExecutor(ConcurrencyProbe(delay=0.0)))
        scheduler.submit(Job(job_id="orphan", kind="b", depends_on=("never_submitted",)))
        scheduler.run()
        assert scheduler.get("orphan").state is JobState.BLOCKED

    def test_a_chain_of_dependencies_runs_in_order(self) -> None:
        probe = ConcurrencyProbe(delay=0.01)
        scheduler = Scheduler(pool(total_cpus=16, total_gpus=0), LocalExecutor(probe))
        scheduler.submit(Job(job_id="em", kind="em"))
        scheduler.submit(Job(job_id="nvt", kind="nvt", depends_on=("em",)))
        scheduler.submit(Job(job_id="npt", kind="npt", depends_on=("nvt",)))
        scheduler.submit(Job(job_id="prod", kind="prod", depends_on=("npt",)))
        scheduler.run()
        assert probe.order == ["em", "nvt", "npt", "prod"]

    def test_a_fan_in_dependency_waits_for_all_parents(self) -> None:
        probe = ConcurrencyProbe(delay=0.01)
        scheduler = Scheduler(pool(total_cpus=16, total_gpus=0), LocalExecutor(probe))
        for i in range(3):
            scheduler.submit(Job(job_id=f"rep{i}", kind="md"))
        scheduler.submit(
            Job(job_id="analyse", kind="analysis", depends_on=("rep0", "rep1", "rep2"))
        )
        scheduler.run()
        assert probe.order[-1] == "analyse"


# ==========================================================================
# Priority, retry, cancellation
# ==========================================================================
class TestPolicies:
    def test_higher_priority_runs_first(self) -> None:
        probe = ConcurrencyProbe(delay=0.01)
        scheduler = Scheduler(
            pool(total_cpus=1, total_gpus=0), LocalExecutor(probe), max_concurrent_jobs=1
        )
        scheduler.submit(Job(job_id="low", kind="x", priority=200))
        scheduler.submit(Job(job_id="high", kind="x", priority=1))
        scheduler.run()
        assert probe.order[0] == "high"

    def test_a_transient_failure_is_retried(self) -> None:
        attempts = {"n": 0}

        def flaky(job: Job) -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            return "ok"

        scheduler = Scheduler(pool(), LocalExecutor(flaky))
        scheduler.submit(Job(job_id="flaky", kind="x", max_retries=3))
        scheduler.run()
        job = scheduler.get("flaky")
        assert job.state is JobState.SUCCEEDED
        assert job.attempts == 3

    def test_retries_are_bounded(self) -> None:
        def always_fails(job: Job) -> Any:
            raise RuntimeError("permanent")

        scheduler = Scheduler(pool(), LocalExecutor(always_fails))
        scheduler.submit(Job(job_id="doomed", kind="x", max_retries=2))
        scheduler.run()
        job = scheduler.get("doomed")
        assert job.state is JobState.FAILED
        assert job.attempts == 3  # the original plus two retries
        assert "permanent" in job.error

    def test_a_pending_job_can_be_cancelled(self) -> None:
        scheduler = Scheduler(pool(), LocalExecutor(ConcurrencyProbe(delay=0.0)))
        scheduler.submit(Job(job_id="doomed", kind="x"))
        scheduler.cancel("doomed")
        scheduler.run()
        assert scheduler.get("doomed").state is JobState.CANCELLED

    def test_a_cancelled_job_blocks_its_dependents(self) -> None:
        scheduler = Scheduler(pool(), LocalExecutor(ConcurrencyProbe(delay=0.0)))
        scheduler.submit(Job(job_id="first", kind="a"))
        scheduler.submit(Job(job_id="second", kind="b", depends_on=("first",)))
        scheduler.cancel("first")
        scheduler.run()
        assert scheduler.get("second").state is JobState.BLOCKED

    def test_duplicate_job_ids_are_rejected(self) -> None:
        scheduler = Scheduler(pool(), LocalExecutor(ConcurrencyProbe(delay=0.0)))
        scheduler.submit(Job(job_id="dup", kind="x"))
        with pytest.raises(PolymerEngineError, match="Duplicate"):
            scheduler.submit(Job(job_id="dup", kind="x"))

    def test_an_unknown_job_id_raises(self) -> None:
        scheduler = Scheduler(pool(), LocalExecutor(ConcurrencyProbe(delay=0.0)))
        with pytest.raises(PolymerEngineError):
            scheduler.get("nope")


# ==========================================================================
# Reporting and persistence
# ==========================================================================
class TestReporting:
    def test_report_counts_every_outcome(self) -> None:
        def selective(job: Job) -> Any:
            if job.kind == "bad":
                raise RuntimeError("no")
            return "ok"

        scheduler = Scheduler(pool(total_cpus=16, total_gpus=0), LocalExecutor(selective))
        scheduler.submit(Job(job_id="good", kind="good"))
        scheduler.submit(Job(job_id="bad", kind="bad"))
        scheduler.submit(Job(job_id="blocked", kind="good", depends_on=("bad",)))
        report = scheduler.run()
        summary = report.summary()
        assert summary["counts"]["SUCCEEDED"] == 1
        assert summary["counts"]["FAILED"] == 1
        assert summary["counts"]["BLOCKED"] == 1
        assert report.all_terminal is True

    def test_events_reach_the_store(self, tmp_path) -> None:
        from polymer_engine.db.store import Store

        with Store(tmp_path / "e.sqlite") as store:
            scheduler = Scheduler(pool(), LocalExecutor(ConcurrencyProbe(delay=0.0)), store=store)
            scheduler.submit(Job(job_id="tracked", kind="x", campaign_id="c1"))
            scheduler.run()
            kinds = {event["kind"] for event in store.list_events()}
            assert "job_started" in kinds
            assert "job_succeeded" in kinds

    def test_state_snapshot_is_serialisable(self) -> None:
        import json

        scheduler = Scheduler(pool(), LocalExecutor(ConcurrencyProbe(delay=0.0)))
        scheduler.submit(Job(job_id="a", kind="x"))
        snapshot = scheduler.state_snapshot()
        assert json.dumps(snapshot)
        assert snapshot["pool"]["total_cpus"] == 8
        assert len(snapshot["jobs"]) == 1


class TestPoolConstruction:
    def test_pool_from_config_honours_a_forced_cpu_only_setting(self) -> None:
        """Pinning gpu_available=false must win over detection, for reproducibility."""
        from polymer_engine.core.config import load_config

        config = load_config(
            discover=False, use_env=False, overrides={"resources": {"gpu_available": False}}
        )
        assert build_pool(config).total_gpus == 0

    def test_pool_leaves_headroom_for_the_engine(self) -> None:
        from polymer_engine.core.config import load_config
        from polymer_engine.local.resources import inspect_resources

        config = load_config(discover=False, use_env=False)
        report = inspect_resources(config, probe_tools=False)
        built = build_pool(config, report=report)
        detected = report.cpu_affinity_count or report.cpu_count or 1
        assert built.total_cpus < detected or detected == 1
