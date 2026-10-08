"""Compute resource inspection.

Used by the scheduler to decide concurrency and by provenance to record what the
work actually ran on.  Every field is optional: on a platform where we cannot
determine something we report ``None`` rather than a plausible default.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.config import EngineConfig
from polymer_engine.local.discovery import discover_all


@dataclass(slots=True)
class GPUInfo:
    index: int
    name: str
    memory_mib: int | None = None


@dataclass(slots=True)
class ResourceReport:
    cpu_count: int | None
    cpu_affinity_count: int | None
    memory_total_kb: int | None
    memory_available_kb: int | None
    disk_free_bytes: int | None
    platform: str
    gpus: list[GPUInfo] = field(default_factory=list)
    tools: dict[str, Any] = field(default_factory=dict)
    max_concurrent_jobs: int = 1

    @property
    def gpu_available(self) -> bool:
        return bool(self.gpus)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cpu_count": self.cpu_count,
            "cpu_affinity_count": self.cpu_affinity_count,
            "memory_total_kb": self.memory_total_kb,
            "memory_available_kb": self.memory_available_kb,
            "disk_free_bytes": self.disk_free_bytes,
            "platform": self.platform,
            "gpu_available": self.gpu_available,
            "gpus": [asdict(g) for g in self.gpus],
            "tools": self.tools,
            "max_concurrent_jobs": self.max_concurrent_jobs,
        }


def _meminfo() -> tuple[int | None, int | None]:
    total = available = None
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1])
                if total is not None and available is not None:
                    break
    except (OSError, ValueError, IndexError):
        # Non-Linux or an unreadable procfs.  Unknown is the honest answer.
        return None, None
    return total, available


def _affinity_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None


def detect_gpus(*, timeout_s: float = 10.0) -> list[GPUInfo]:
    """Detect NVIDIA GPUs via ``nvidia-smi``.

    Absence of ``nvidia-smi`` means "we could not detect any", which is reported as
    an empty list -- not as proof that no accelerator exists.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    gpus: list[GPUInfo] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        memory = None
        if len(parts) > 2:
            try:
                memory = int(parts[2])
            except ValueError:
                memory = None
        gpus.append(GPUInfo(index=index, name=parts[1], memory_mib=memory))
    return gpus


def inspect_resources(config: EngineConfig, *, probe_tools: bool = True) -> ResourceReport:
    """Describe the machine and the tools available on it."""
    total_kb, available_kb = _meminfo()
    scratch = config.paths.resolved("scratch_dir")
    disk_free: int | None
    try:
        disk_free = shutil.disk_usage(scratch if scratch.exists() else Path.cwd()).free
    except OSError:
        disk_free = None

    configured_gpu = config.resources.gpu_available
    if configured_gpu is False:
        gpus: list[GPUInfo] = []
    elif configured_gpu is True:
        detected = detect_gpus()
        # Configuration asserts a GPU exists; record the detected ones if any, but
        # keep the assertion authoritative so runs stay reproducible.
        gpus = detected or [GPUInfo(index=i, name="configured") for i in (config.resources.gpu_device_ids or [0])]
    else:
        gpus = detect_gpus()

    tools = {name: status.as_dict() for name, status in discover_all(config, probe=probe_tools).items()}
    return ResourceReport(
        cpu_count=os.cpu_count(),
        cpu_affinity_count=_affinity_count(),
        memory_total_kb=total_kb,
        memory_available_kb=available_kb,
        disk_free_bytes=disk_free,
        platform=f"{platform.system()} {platform.release()}",
        gpus=gpus,
        tools=tools,
        max_concurrent_jobs=config.resources.max_concurrent_jobs,
    )


__all__ = ["GPUInfo", "ResourceReport", "detect_gpus", "inspect_resources"]
