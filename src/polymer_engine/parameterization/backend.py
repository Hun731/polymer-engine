"""The backend interface and its registry.

A backend is anything that can turn a polymer into parameters: a local typing table, a
toolchain like AmberTools, or a human-in-the-loop acquisition route such as CHARMM-GUI.
The orchestrator never names one.  It asks the registry which backends exist, asks each
what it could do with a given polymer, and routes on the answers.

Adding a backend is subclassing :class:`ForceFieldBackend` and registering it.  Nothing
above this layer changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from polymer_engine.core.errors import PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.parameterization.capability import CapabilityState
from polymer_engine.parameterization.models import (
    ForceFieldAssessment,
    ParameterizationRequest,
    ParameterizationResult,
    ParameterizationState,
    ParameterizationValidation,
)

logger = get_logger("parameterization.backend")


class ForceFieldBackend(ABC):
    """One route from a polymer to parameters."""

    #: Stable identifier used in provenance and on the command line.
    name: str = "backend"
    #: The force field this backend assigns, when it has exactly one.
    force_field: str | None = None
    #: True when a person must act partway through (e.g. a web submission).
    human_in_the_loop: bool = False
    #: True when credentials are needed to complete the route.
    requires_credentials: bool = False

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """What this backend can do *on this machine*, measured rather than declared."""

    @abstractmethod
    def assess(self, polymer: Any) -> ForceFieldAssessment:
        """How far could this backend carry this polymer?  Must not do any work."""

    @abstractmethod
    def parameterize(self, request: ParameterizationRequest) -> ParameterizationResult:
        """Produce parameters, or a result explaining why it could not."""

    @abstractmethod
    def validate(self, result: ParameterizationResult) -> ParameterizationValidation:
        """Check a parameter set against independent evidence."""

    # -- shared helpers -------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self.capabilities().get("available"))

    def _assessment(
        self,
        polymer: Any,
        state: CapabilityState,
        reason: str,
        **extra: Any,
    ) -> ForceFieldAssessment:
        """Build an assessment with this backend's identity already filled in."""
        return ForceFieldAssessment(
            backend=self.name,
            polymer_id=str(getattr(polymer, "polymer_id", "unknown")),
            polymer_name=str(getattr(polymer, "name", "unknown")),
            state=state,
            force_field=self.force_field,
            reason=reason,
            requires_credentials=self.requires_credentials,
            requires_human_step=self.human_in_the_loop,
            **extra,
        )

    def _unavailable(
        self, request: ParameterizationRequest, reason: str, *, actions: Iterable[str] = ()
    ) -> ParameterizationResult:
        """A uniform 'could not' that is never mistaken for a 'did'."""
        return ParameterizationResult(
            backend=self.name, request=request, state=ParameterizationState.BLOCKED,
            force_field=self.force_field, diagnostics=[reason],
            required_actions=list(actions),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} available={self.available}>"


class BackendRegistry:
    """Every backend the engine knows about, available or not.

    Unavailable backends stay registered on purpose: "OpenFF is not installed" is a
    materially different answer from "OpenFF does not exist", and a router that silently
    omitted the former would look as though it had considered fewer routes than it did.
    """

    def __init__(self) -> None:
        self._backends: dict[str, ForceFieldBackend] = {}

    def register(self, backend: ForceFieldBackend) -> ForceFieldBackend:
        if backend.name in self._backends:
            raise PolymerEngineError("Backend already registered", backend=backend.name)
        self._backends[backend.name] = backend
        logger.debug("Registered backend %s", backend.name)
        return backend

    def get(self, name: str) -> ForceFieldBackend:
        try:
            return self._backends[name]
        except KeyError:
            raise PolymerEngineError(
                "Unknown parameterization backend", backend=name,
                known=sorted(self._backends),
            ) from None

    def names(self) -> list[str]:
        return sorted(self._backends)

    def all(self) -> list[ForceFieldBackend]:
        return [self._backends[name] for name in self.names()]

    def available(self) -> list[ForceFieldBackend]:
        return [b for b in self.all() if b.available]

    def assess_all(self, polymer: Any) -> list[ForceFieldAssessment]:
        """Ask every backend what it could do.  Unavailable ones answer too."""
        assessments = []
        for backend in self.all():
            try:
                assessments.append(backend.assess(polymer))
            except Exception as exc:  # noqa: BLE001 - one broken backend must not hide the rest
                logger.warning("Backend %s failed to assess: %s", backend.name, exc)
                assessments.append(ForceFieldAssessment(
                    backend=backend.name,
                    polymer_id=getattr(polymer, "polymer_id", "unknown"),
                    polymer_name=getattr(polymer, "name", "unknown"),
                    state=CapabilityState.UNAVAILABLE,
                    reason=f"assessment raised {type(exc).__name__}: {exc}",
                ))
        return assessments

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_backends": len(self._backends),
            "n_available": len(self.available()),
            "backends": {b.name: b.capabilities() for b in self.all()},
        }


def default_registry() -> BackendRegistry:
    """The backends this engine ships with, in preference order by construction.

    Imported lazily so that a backend whose optional dependency is missing cannot break
    registry construction for the others.
    """
    from polymer_engine.parameterization.backends.charmm_gui import CharmmGuiBackend
    from polymer_engine.parameterization.backends.gaff import GaffBackend
    from polymer_engine.parameterization.backends.openff import OpenFFBackend
    from polymer_engine.parameterization.backends.opls import OplsBackend

    registry = BackendRegistry()
    for backend in (OplsBackend(), CharmmGuiBackend(), OpenFFBackend(), GaffBackend()):
        registry.register(backend)
    return registry


__all__ = ["BackendRegistry", "ForceFieldBackend", "default_registry"]
