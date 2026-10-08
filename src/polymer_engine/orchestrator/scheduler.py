"""Resource-aware job scheduling.

The scheduler exists to make one guarantee: **configured resources are never
oversubscribed.**  Four GPU replicas on a one-GPU machine do not all start; three
queue.  A job whose demand exceeds the machine's total capacity is rejected outright
rather than queued forever.

Design notes:

* Resources are *reserved* before a job starts and released when it finishes,
  including when it fails or is cancelled.  A leaked reservation would silently
  shrink the machine.
* Dependencies are respected: a job runs only when everything it depends on has
  succeeded.  A failed dependency cancels its dependents rather than letting them run
  on missing inputs.
* Execution is pluggable.  :class:`LocalExecutor` runs jobs in a thread pool here;
  the same interface admits a SLURM/PBS backend later without the scheduler changing.
* State is serialisable, so a queue can be checkpointed and resumed.
"""

from __future__ import annotations

import heapq
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from polymer_engine.core.errors import PolymerEngineError, ScientificError
from polymer_engine.core.logging import get_logger, log_event
from polymer_engine.core.models import utc_now

logger = get_logger("orchestrator.scheduler")


class JobState(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"
    #: A dependency failed, so this job can never run.
    BLOCKED = "BLOCKED"
    #: Demands more than the machine has; queuing it would deadlock.
    REJECTED = "REJECTED"

    @property
    def terminal(self) -> bool:
        return self in {
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.BLOCKED,
            JobState.REJECTED,
        }


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """What one job needs while it runs."""

    cpus: int = 1
    gpus: int = 0
    memory_mb: int = 1024
    disk_mb: int = 1024
    vram_mb: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("cpus", self.cpus), ("gpus", self.gpus), ("memory_mb", self.memory_mb),
            ("disk_mb", self.disk_mb), ("vram_mb", self.vram_mb),
        ):
            if value < 0:
                raise ScientificError(f"{name} cannot be negative", **{name: value})
        if self.cpus < 1:
            raise ScientificError("A job needs at least one CPU", cpus=self.cpus)

    def as_dict(self) -> dict[str, int]:
        return {
            "cpus": self.cpus, "gpus": self.gpus, "memory_mb": self.memory_mb,
            "disk_mb": self.disk_mb, "vram_mb": self.vram_mb,
        }


@dataclass
class ResourcePool:
    """The machine's capacity, and what is currently reserved.

    ``vram_mb`` is per-GPU rather than aggregate: a job needing 20 GB cannot be spread
    across two 10 GB cards.
    """

    total_cpus: int
    total_gpus: int = 0
    total_memory_mb: int = 0
    total_disk_mb: int = 0
    vram_mb_per_gpu: int = 0

    used_cpus: int = 0
    used_gpus: int = 0
    used_memory_mb: int = 0
    used_disk_mb: int = 0

    def __post_init__(self) -> None:
        if self.total_cpus < 1:
            raise ScientificError("A resource pool needs at least one CPU", cpus=self.total_cpus)

    @property
    def free_cpus(self) -> int:
        return self.total_cpus - self.used_cpus

    @property
    def free_gpus(self) -> int:
        return self.total_gpus - self.used_gpus

    @property
    def free_memory_mb(self) -> int:
        return self.total_memory_mb - self.used_memory_mb

    @property
    def free_disk_mb(self) -> int:
        return self.total_disk_mb - self.used_disk_mb

    def exceeds_capacity(self, demand: ResourceRequest) -> str:
        """Whether a request can *never* be satisfied.  Empty string means it can."""
        if demand.cpus > self.total_cpus:
            return f"needs {demand.cpus} CPUs but the machine has {self.total_cpus}"
        if demand.gpus > self.total_gpus:
            return f"needs {demand.gpus} GPU(s) but the machine has {self.total_gpus}"
        if self.total_memory_mb and demand.memory_mb > self.total_memory_mb:
            return f"needs {demand.memory_mb} MB RAM but the machine has {self.total_memory_mb} MB"
        if self.total_disk_mb and demand.disk_mb > self.total_disk_mb:
            return f"needs {demand.disk_mb} MB disk but only {self.total_disk_mb} MB is available"
        if demand.vram_mb and self.vram_mb_per_gpu and demand.vram_mb > self.vram_mb_per_gpu:
            return (
                f"needs {demand.vram_mb} MB VRAM but each GPU has {self.vram_mb_per_gpu} MB; "
                "VRAM cannot be pooled across cards"
            )
        return ""

    def can_reserve(self, demand: ResourceRequest) -> bool:
        if demand.cpus > self.free_cpus or demand.gpus > self.free_gpus:
            return False
        if self.total_memory_mb and demand.memory_mb > self.free_memory_mb:
            return False
        return not (self.total_disk_mb and demand.disk_mb > self.free_disk_mb)

    def reserve(self, demand: ResourceRequest) -> None:
        if not self.can_reserve(demand):
            raise PolymerEngineError(
                "Cannot reserve resources that are not free",
                demand=demand.as_dict(),
                free={"cpus": self.free_cpus, "gpus": self.free_gpus,
                      "memory_mb": self.free_memory_mb, "disk_mb": self.free_disk_mb},
            )
        self.used_cpus += demand.cpus
        self.used_gpus += demand.gpus
        self.used_memory_mb += demand.memory_mb
        self.used_disk_mb += demand.disk_mb

    def release(self, demand: ResourceRequest) -> None:
        # Clamped at zero: a double release must not manufacture capacity.
        self.used_cpus = max(0, self.used_cpus - demand.cpus)
        self.used_gpus = max(0, self.used_gpus - demand.gpus)
        self.used_memory_mb = max(0, self.used_memory_mb - demand.memory_mb)
        self.used_disk_mb = max(0, self.used_disk_mb - demand.disk_mb)

    def snapshot(self) -> dict[str, int]:
        return {
            "total_cpus": self.total_cpus, "used_cpus": self.used_cpus, "free_cpus": self.free_cpus,
            "total_gpus": self.total_gpus, "used_gpus": self.used_gpus, "free_gpus": self.free_gpus,
            "total_memory_mb": self.total_memory_mb, "free_memory_mb": self.free_memory_mb,
            "total_disk_mb": self.total_disk_mb, "free_disk_mb": self.free_disk_mb,
            "vram_mb_per_gpu": self.vram_mb_per_gpu,
        }

    @classmethod
    def from_report(cls, report: Any, *, reserve_cpus: int = 1) -> ResourcePool:
        """Build a pool from an inspected machine, leaving headroom for the engine."""
        cpus = max(1, (report.cpu_affinity_count or report.cpu_count or 2) - reserve_cpus)
        memory_mb = int((report.memory_available_kb or 0) / 1024)
        disk_mb = int((report.disk_free_bytes or 0) / (1024 * 1024))
        vram = max((g.memory_mib or 0) for g in report.gpus) if report.gpus else 0
        return cls(
            total_cpus=cpus,
            total_gpus=len(report.gpus),
            total_memory_mb=memory_mb,
            total_disk_mb=disk_mb,
            vram_mb_per_gpu=vram,
        )


@dataclass
class Job:
    """One unit of schedulable work."""

    job_id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    priority: int = 100
    depends_on: tuple[str, ...] = ()
    max_retries: int = 0
    campaign_id: str | None = None
    label: str = ""

    state: JobState = JobState.PENDING
    attempts: int = 0
    result: Any = None
    error: str | None = None
    submitted_at: str = field(default_factory=lambda: utc_now().isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    assigned_gpus: tuple[int, ...] = ()

    @property
    def terminal(self) -> bool:
        return self.state.terminal

    @property
    def succeeded(self) -> bool:
        return self.state is JobState.SUCCEEDED

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "label": self.label,
            "campaign_id": self.campaign_id,
            "state": self.state.value,
            "priority": self.priority,
            "depends_on": list(self.depends_on),
            "resources": self.resources.as_dict(),
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "error": self.error,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "assigned_gpus": list(self.assigned_gpus),
        }


class JobExecutor(ABC):
    """Runs a job.  The scheduler owns *when*; an executor owns *how*."""

    name: str = "executor"

    @abstractmethod
    def submit(self, job: Job) -> Future:
        """Start the job and return a future resolving to its result."""

    def shutdown(self, wait: bool = True) -> None:  # noqa: B027 - optional hook
        """Release any resources the executor holds.

        Deliberately not abstract: most executors have nothing to release, and forcing
        every backend to write an empty override adds noise rather than safety.
        """


class LocalExecutor(JobExecutor):
    """Runs jobs in a local thread pool.

    A thread pool is right here because the work is subprocess-bound (GROMACS, ORCA):
    the GIL is released while waiting on a child process.
    """

    name = "local"

    def __init__(self, handler: Callable[[Job], Any], *, max_workers: int = 8) -> None:
        self.handler = handler
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="polymer-job")

    def submit(self, job: Job) -> Future:
        return self._pool.submit(self.handler, job)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


@dataclass
class SchedulerReport:
    """What a scheduler run did."""

    jobs: list[Job] = field(default_factory=list)
    peak_concurrency: int = 0
    peak_gpu_usage: int = 0
    wall_time_s: float = 0.0

    def by_state(self, state: JobState) -> list[Job]:
        return [j for j in self.jobs if j.state is state]

    @property
    def all_terminal(self) -> bool:
        return all(j.terminal for j in self.jobs)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for job in self.jobs:
            counts[job.state.value] = counts.get(job.state.value, 0) + 1
        return {
            "n_jobs": len(self.jobs),
            "counts": counts,
            "peak_concurrency": self.peak_concurrency,
            "peak_gpu_usage": self.peak_gpu_usage,
            "wall_time_s": round(self.wall_time_s, 3),
            "all_terminal": self.all_terminal,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.summary(), "jobs": [j.as_dict() for j in self.jobs]}


class Scheduler:
    """A dependency-aware, resource-limited job queue."""

    def __init__(
        self,
        pool: ResourcePool,
        executor: JobExecutor,
        *,
        max_concurrent_jobs: int | None = None,
        poll_interval_s: float = 0.02,
        store: Any = None,
    ) -> None:
        self.pool = pool
        self.executor = executor
        self.max_concurrent_jobs = max_concurrent_jobs
        self.poll_interval_s = poll_interval_s
        self.store = store
        self._jobs: dict[str, Job] = {}
        self._order: list[tuple[int, int, str]] = []
        self._sequence = 0
        self._lock = threading.RLock()
        self._cancelled: set[str] = set()
        self.peak_concurrency = 0
        self.peak_gpu_usage = 0

    # -- queue management ----------------------------------------------
    def submit(self, job: Job) -> Job:
        """Add a job.  A job the machine can never run is rejected immediately."""
        with self._lock:
            if job.job_id in self._jobs:
                raise PolymerEngineError("Duplicate job id", job_id=job.job_id)
            reason = self.pool.exceeds_capacity(job.resources)
            if reason:
                job.state = JobState.REJECTED
                job.error = f"resource request cannot be satisfied by this machine: {reason}"
                job.finished_at = utc_now().isoformat()
                self._jobs[job.job_id] = job
                logger.warning("Rejected job %s: %s", job.job_id, job.error)
                return job
            self._jobs[job.job_id] = job
            self._sequence += 1
            heapq.heappush(self._order, (job.priority, self._sequence, job.job_id))
            return job

    def submit_all(self, jobs: Iterable[Job]) -> list[Job]:
        return [self.submit(job) for job in jobs]

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise PolymerEngineError("Unknown job", job_id=job_id) from None

    def jobs(self) -> list[Job]:
        return list(self._jobs.values())

    def cancel(self, job_id: str) -> Job:
        """Cancel a job that has not started; a running job is left to finish."""
        with self._lock:
            job = self.get(job_id)
            self._cancelled.add(job_id)
            if job.state in {JobState.PENDING, JobState.READY, JobState.RETRYING}:
                job.state = JobState.CANCELLED
                job.finished_at = utc_now().isoformat()
            return job

    # -- readiness ------------------------------------------------------
    def _dependency_state(self, job: Job) -> JobState | None:
        """``None`` if the job may run; otherwise the state it should take."""
        for dependency_id in job.depends_on:
            dependency = self._jobs.get(dependency_id)
            if dependency is None:
                return JobState.BLOCKED
            if dependency.state in {JobState.FAILED, JobState.CANCELLED, JobState.BLOCKED, JobState.REJECTED}:
                return JobState.BLOCKED
            if not dependency.succeeded:
                return JobState.PENDING
        return None

    def _next_runnable(self, running: int) -> Job | None:
        """The highest-priority job whose dependencies and resources are satisfied."""
        if self.max_concurrent_jobs is not None and running >= self.max_concurrent_jobs:
            return None
        deferred: list[tuple[int, int, str]] = []
        chosen: Job | None = None
        while self._order:
            entry = heapq.heappop(self._order)
            job = self._jobs[entry[2]]
            if job.terminal or job.state is JobState.RUNNING:
                continue
            if job.job_id in self._cancelled:
                job.state = JobState.CANCELLED
                job.finished_at = utc_now().isoformat()
                continue
            blocked = self._dependency_state(job)
            if blocked is JobState.BLOCKED:
                job.state = JobState.BLOCKED
                job.error = "a dependency failed or was cancelled"
                job.finished_at = utc_now().isoformat()
                continue
            if blocked is JobState.PENDING:
                deferred.append(entry)
                continue
            if not self.pool.can_reserve(job.resources):
                deferred.append(entry)
                continue
            chosen = job
            break
        for entry in deferred:
            heapq.heappush(self._order, entry)
        return chosen

    # -- the loop --------------------------------------------------------
    def run(self, *, timeout_s: float | None = None) -> SchedulerReport:
        """Run every submitted job to a terminal state, respecting resource limits."""
        started = time.monotonic()
        running: dict[str, tuple[Future, Job]] = {}

        while True:
            with self._lock:
                while True:
                    job = self._next_runnable(len(running))
                    if job is None:
                        break
                    self.pool.reserve(job.resources)
                    job.state = JobState.RUNNING
                    job.attempts += 1
                    job.started_at = utc_now().isoformat()
                    if job.resources.gpus:
                        job.assigned_gpus = tuple(
                            range(self.pool.used_gpus - job.resources.gpus, self.pool.used_gpus)
                        )
                    running[job.job_id] = (self.executor.submit(job), job)
                    self.peak_concurrency = max(self.peak_concurrency, len(running))
                    self.peak_gpu_usage = max(self.peak_gpu_usage, self.pool.used_gpus)
                    self._record(job, "job_started")

            if not running:
                with self._lock:
                    if self._next_runnable(0) is None:
                        break
                    continue

            done = [job_id for job_id, (future, _) in running.items() if future.done()]
            for job_id in done:
                future, job = running.pop(job_id)
                with self._lock:
                    self.pool.release(job.resources)
                    self._settle(job, future)

            if timeout_s is not None and time.monotonic() - started > timeout_s:
                with self._lock:
                    for _, job in running.values():
                        job.state = JobState.FAILED
                        job.error = f"scheduler timed out after {timeout_s}s"
                        job.finished_at = utc_now().isoformat()
                        self.pool.release(job.resources)
                break

            if running:
                time.sleep(self.poll_interval_s)

        return SchedulerReport(
            jobs=list(self._jobs.values()),
            peak_concurrency=self.peak_concurrency,
            peak_gpu_usage=self.peak_gpu_usage,
            wall_time_s=time.monotonic() - started,
        )

    def _settle(self, job: Job, future: Future) -> None:
        """Record the outcome and requeue for retry where the policy allows."""
        job.finished_at = utc_now().isoformat()
        exception = future.exception()
        if exception is None:
            job.result = future.result()
            job.state = JobState.SUCCEEDED
            self._record(job, "job_succeeded")
            return

        job.error = f"{type(exception).__name__}: {exception}"
        if job.attempts <= job.max_retries and job.job_id not in self._cancelled:
            job.state = JobState.RETRYING
            self._sequence += 1
            heapq.heappush(self._order, (job.priority, self._sequence, job.job_id))
            logger.warning(
                "Job %s failed (attempt %d/%d), retrying: %s",
                job.job_id, job.attempts, job.max_retries + 1, job.error,
            )
            self._record(job, "job_retrying")
            return

        job.state = JobState.FAILED
        logger.error("Job %s failed: %s", job.job_id, job.error)
        self._record(job, "job_failed")

    def _record(self, job: Job, event: str) -> None:
        payload = {"event": event, **job.as_dict(), "resources": self.pool.snapshot()}
        log_event(logger, event, payload)
        if self.store is not None:
            self.store.log_event(event, payload)

    # -- persistence -----------------------------------------------------
    def state_snapshot(self) -> dict[str, Any]:
        """A serialisable view of the queue, for checkpointing."""
        return {
            "pool": self.pool.snapshot(),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "jobs": [job.as_dict() for job in self._jobs.values()],
        }


def build_pool(config: Any, *, report: Any = None) -> ResourcePool:
    """Construct a :class:`ResourcePool` from configuration and the real machine.

    Configuration wins over detection: an operator who pins ``gpu_available: false``
    gets a CPU-only pool even on a machine with a GPU, which is what reproducibility
    requires.
    """
    from polymer_engine.local.resources import inspect_resources

    report = report or inspect_resources(config, probe_tools=False)
    pool = ResourcePool.from_report(report)
    if config.resources.gpu_available is False:
        pool.total_gpus = 0
        pool.vram_mb_per_gpu = 0
    if config.resources.cpus_per_job:
        pool.total_cpus = max(pool.total_cpus, config.resources.cpus_per_job)
    return pool


__all__ = [
    "Job",
    "JobExecutor",
    "JobState",
    "LocalExecutor",
    "ResourcePool",
    "ResourceRequest",
    "Scheduler",
    "SchedulerReport",
    "build_pool",
]
