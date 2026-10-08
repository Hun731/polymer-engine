"""Executor registry.

Actions are dispatched by kind.  There is **no fallback executor**: an unknown action
kind raises rather than quietly running a dry-run stub and reporting success, which is
how a framework ends up recording synthetic observations as science.
"""

from __future__ import annotations

from collections.abc import Callable

from polymer_engine.core.config import EngineConfig
from polymer_engine.core.errors import PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.executors.base import Executor
from polymer_engine.executors.gromacs import GromacsEquilibrateExecutor, ReplicaAnalysisExecutor
from polymer_engine.local.discovery import ToolStatus, discover_all
from polymer_engine.local.runner import GROMACSRunner

logger = get_logger("executors.registry")


class ExecutorRegistry:
    """Maps action kinds to executors, wired from configuration."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        tools: dict[str, ToolStatus] | None = None,
        probe_tools: bool = True,
    ) -> None:
        self.config = config
        self.tools = tools if tools is not None else discover_all(config, probe=probe_tools)
        self._executors: dict[str, Executor] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        enabled = self.config.safety.execution_enabled
        gromacs_status = self.tools["gromacs"]
        gpu = bool(
            self.config.resources.gpu_available
            and gromacs_status.capabilities.get("has_gpu", False)
        )
        runner = GROMACSRunner(
            gromacs_status,
            enabled=enabled,
            default_timeout_s=self.config.resources.job_timeout_s,
        )
        self.register(GromacsEquilibrateExecutor(runner, use_gpu=gpu))
        self.register(ReplicaAnalysisExecutor(self.config.analysis))

    def register(self, executor: Executor) -> Executor:
        self._executors[executor.kind] = executor
        return executor

    def register_factory(self, kind: str, factory: Callable[[], Executor]) -> None:
        self._executors[kind] = factory()

    def get(self, kind: str) -> Executor:
        try:
            return self._executors[kind]
        except KeyError:
            raise PolymerEngineError(
                "No executor is registered for this action kind",
                kind=kind,
                known=sorted(self._executors),
                hint="register an executor rather than letting the action run as a stub",
            ) from None

    def supports(self, kind: str) -> bool:
        return kind in self._executors

    def kinds(self) -> list[str]:
        return sorted(self._executors)

    def available_tools(self) -> set[str]:
        return {name for name, status in self.tools.items() if status.usable}


__all__ = ["ExecutorRegistry"]
