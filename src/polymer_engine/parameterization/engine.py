"""The parameterization engine: assess, route, parameterize, validate, qualify.

This is the layer the CLI and the campaign talk to.  It owns no chemistry of its own --
every decision is delegated to a backend, a gate, or the router -- and its job is to keep
the three states apart:

``PARAMETERIZED`` a topology exists · ``VALIDATED`` it was checked · ``QUALIFIED`` it was
checked well enough for a *stated* property class on a *stated* polymer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polymer_engine.core.logging import get_logger
from polymer_engine.parameterization.backend import BackendRegistry, default_registry
from polymer_engine.parameterization.models import (
    ForceFieldAssessment,
    ParameterizationRequest,
    ParameterizationResult,
    ParameterizationState,
    ParameterizationValidation,
    PropertyClass,
)
from polymer_engine.parameterization.quality import ParameterQualityGate
from polymer_engine.parameterization.registry import (
    ParameterizationRecord,
    ParameterizationRegistry,
)
from polymer_engine.parameterization.router import (
    ForceFieldComparison,
    RouteDecision,
    SystemBuildRouter,
    compare,
)
from polymer_engine.parameterization.state import TrackStore
from polymer_engine.simulation.cgenff import PenaltyReport

logger = get_logger("parameterization.engine")


class ParameterizationEngine:
    """Drive a polymer from candidate to qualified parameters, or to a stated refusal."""

    def __init__(
        self,
        backends: BackendRegistry | None = None,
        registry: ParameterizationRegistry | None = None,
        tracks: TrackStore | None = None,
    ) -> None:
        self.backends = backends or default_registry()
        self.registry = registry or ParameterizationRegistry()
        self.tracks = tracks or TrackStore()
        self.router = SystemBuildRouter()

    # -- questions the engine can answer --------------------------------
    def assess(self, polymer: Any) -> list[ForceFieldAssessment]:
        """Can this polymer be parameterized, and by what?"""
        return self.backends.assess_all(polymer)

    def compare(
        self, polymer: Any, *, property_class: PropertyClass
    ) -> ForceFieldComparison:
        """Every route side by side, with a routing decision attached."""
        return compare(self.assess(polymer), property_class=property_class)

    def route(
        self, polymer: Any, *, property_class: PropertyClass, prefer: str | None = None
    ) -> RouteDecision:
        return self.router.route(self.assess(polymer),
                                 property_class=property_class, prefer=prefer)

    # -- doing the work -------------------------------------------------
    def parameterize(
        self, request: ParameterizationRequest, *, backend: str
    ) -> ParameterizationResult:
        track = self.tracks.track(request.polymer_id, backend)
        if track.state is ParameterizationState.DISCOVERED:
            track.advance(ParameterizationState.BACKEND_SELECTED,
                          f"routed to {backend}")
        result = self.backends.get(backend).parameterize(request)
        if result.state is ParameterizationState.PARAMETERIZED:
            track.advance(ParameterizationState.PARAMETERIZED,
                          f"{backend} produced a topology",
                          evidence={"topology": result.topology_path})
        else:
            # A verdict is reachable from anywhere; record it rather than pretending.
            if result.state in {ParameterizationState.BLOCKED,
                                ParameterizationState.FAILED,
                                ParameterizationState.REQUIRES_EXPERT_REVIEW}:
                track.advance(result.state, "; ".join(result.diagnostics) or "refused")
        return result

    def validate(
        self, result: ParameterizationResult, *, penalties: PenaltyReport | None = None
    ) -> ParameterizationValidation:
        backend = self.backends.get(result.backend)
        validation = backend.validate(result)
        track = self.tracks.track(result.request.polymer_id, result.backend)

        if validation.state is ParameterizationState.PARAMETER_COMPLETENESS_VALIDATED:
            if track.state is ParameterizationState.PARAMETERIZED:
                track.advance(ParameterizationState.TOPOLOGY_VALIDATED,
                              "topology and charges are consistent")
            if track.state is ParameterizationState.TOPOLOGY_VALIDATED:
                track.advance(ParameterizationState.PARAMETER_COMPLETENESS_VALIDATED,
                              "every bonded interaction the connectivity implies is present")
        elif validation.state in {ParameterizationState.INCONCLUSIVE,
                                  ParameterizationState.FAILED}:
            if not track.terminal:
                track.advance(validation.state,
                              "; ".join(validation.diagnostics) or "validation refused")

        # Whether the parameters are good *enough* depends on the intended use.
        gate = ParameterQualityGate(result.request.property_class)
        report, determination = gate.evaluate(
            force_field=result.force_field or "unknown",
            force_field_version=result.force_field_version,
            penalties=penalties or self._penalties(validation),
            qm_validated=None,
        )
        validation.metrics["quality_gate"] = {
            "status": report.status.value, "promotable": report.promotable,
            "determination": determination.value,
            "standard": gate.standard.as_dict(),
            "gates": [g.model_dump(mode="json") for g in report.gates],
        }
        requires_qm = gate.standard.requires_torsion_qm or not report.promotable
        validation.metrics["qm_required"] = requires_qm
        if track.state is ParameterizationState.PARAMETER_COMPLETENESS_VALIDATED:
            if requires_qm:
                track.advance(
                    ParameterizationState.QM_VALIDATION_REQUIRED,
                    "quality gate requires QM validation before these parameters may "
                    "be used for this property class",
                )
            else:
                # Deliberately NOT QM_VALIDATED: no QM was run, and a state that says
                # otherwise is a claim nothing tested. The shortcut records the truth --
                # this property class does not need it.
                track.advance(
                    ParameterizationState.SYSTEM_VALIDATED,
                    f"{result.request.property_class.value} does not require torsional "
                    f"QM at this parameter quality; QM was not run",
                )
        return validation

    @staticmethod
    def _penalties(validation: ParameterizationValidation) -> PenaltyReport | None:
        raw = validation.metrics.get("penalties")
        if not isinstance(raw, dict):
            return None
        report = PenaltyReport()
        report.cgenff_version = raw.get("cgenff_version")
        report.residue_param_penalty = raw.get("max_penalty")
        return report

    def record(
        self,
        result: ParameterizationResult,
        validation: ParameterizationValidation,
        *,
        family: str,
    ) -> ParameterizationRecord:
        """Commit what happened to the registry, in whatever state it reached."""
        track = self.tracks.track(result.request.polymer_id, result.backend)
        penalties = validation.metrics.get("penalties") or {}
        record = ParameterizationRecord(
            polymer_id=result.request.polymer_id,
            polymer_name=result.request.polymer_name,
            family=family, backend=result.backend,
            force_field=result.force_field or "unknown",
            force_field_version=result.force_field_version,
            property_class=result.request.property_class,
            state=track.state,
            parameter_source=result.parameter_source,
            topology_path=result.topology_path,
            parameter_files=list(result.parameter_files),
            net_charge=result.net_charge,
            max_penalty=penalties.get("max_penalty") if isinstance(penalties, dict) else None,
            qm_priority=validation.qm_priority,
            validation_metrics=dict(validation.metrics),
            artifacts=dict(result.artifacts),
            provenance={**result.provenance, "track": track.as_dict()},
            diagnostics=list(result.diagnostics) + list(validation.diagnostics),
        )
        return self.registry.add(record)

    def save(self, directory: str | Path) -> dict[str, str]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return {
            "registry": str(self.registry.save(directory / "parameterization_registry.json")),
            "tracks": str(self.tracks.save(directory / "parameterization_tracks.json")),
        }


__all__ = ["ParameterizationEngine"]
