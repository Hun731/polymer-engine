"""Campaign construction, execution state, and the reproducibility manifest.

A campaign is the unit of scientific work: one polymer, one system, one parameter
set, N replicas, and the analysis and gates that decide whether anything came of it.

Two properties are load-bearing:

* **Resumable.**  Every stage writes its state to the store, so a campaign can be
  reloaded and continued.  Re-planning an existing campaign is idempotent.
* **Reproducible.**  :meth:`Campaign.manifest` emits everything needed to rebuild the
  campaign: objective, parameters, seeds, software versions, input digests, analysis
  settings, and gate outcomes.  :func:`campaign_fingerprint` hashes the deterministic
  subset, so two runs of the same specification can be compared exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from polymer_engine.core.config import AnalysisDefaults, EngineConfig, SimulationDefaults, UmbrellaDefaults
from polymer_engine.core.errors import PolymerEngineError, SystemValidationError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import (
    Action,
    CostEstimate,
    Determination,
    ExecutionRecord,
    GateReport,
    utc_now,
)
from polymer_engine.core.provenance import ProvenanceGraph, canonical_hash
from polymer_engine.simulation.mdp import validate_parameters
from polymer_engine.simulation.replicas import Replica, ReplicaSet, materialize_replicas
from polymer_engine.simulation.system import ImportedSystem, SystemManifest

logger = get_logger("orchestrator.campaign")

MANIFEST_VERSION = "1.0"


class CampaignStatus(str, Enum):
    CREATED = "created"
    SYSTEM_VALIDATED = "system_validated"
    PLANNED = "planned"
    RUNNING = "running"
    ANALYSED = "analysed"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class CampaignSpec:
    """Everything that defines a campaign, before anything is built.

    The scientific assumptions live in ``simulation``/``analysis``/``umbrella`` rather
    than being scattered as literals through the code, and they are copied verbatim
    into the manifest.
    """

    campaign_id: str
    polymer_id: str
    polymer_name: str = ""
    objective_id: str | None = None
    hypothesis_id: str | None = None
    question: str = ""
    workdir: Path = Path("workspaces/campaign")
    system_source: str | None = None
    simulation: SimulationDefaults = field(default_factory=SimulationDefaults)
    analysis: AnalysisDefaults = field(default_factory=AnalysisDefaults)
    umbrella: UmbrellaDefaults = field(default_factory=UmbrellaDefaults)
    strategy_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_config(
        cls, config: EngineConfig, *, campaign_id: str, polymer_id: str, **kwargs: Any
    ) -> CampaignSpec:
        """Build a spec whose defaults come from configuration, not from literals."""
        return cls(
            campaign_id=campaign_id,
            polymer_id=polymer_id,
            workdir=config.paths.resolved("workspace_dir") / campaign_id,
            simulation=config.simulation.model_copy(deep=True),
            analysis=config.analysis.model_copy(deep=True),
            umbrella=config.umbrella.model_copy(deep=True),
            **kwargs,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "polymer_id": self.polymer_id,
            "polymer_name": self.polymer_name,
            "objective_id": self.objective_id,
            "hypothesis_id": self.hypothesis_id,
            "question": self.question,
            "workdir": str(self.workdir),
            "system_source": self.system_source,
            "simulation": self.simulation.model_dump(mode="json"),
            "analysis": self.analysis.model_dump(mode="json"),
            "umbrella": self.umbrella.model_dump(mode="json"),
            "strategy_id": self.strategy_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CampaignSpec:
        return cls(
            campaign_id=payload["campaign_id"],
            polymer_id=payload["polymer_id"],
            polymer_name=payload.get("polymer_name", ""),
            objective_id=payload.get("objective_id"),
            hypothesis_id=payload.get("hypothesis_id"),
            question=payload.get("question", ""),
            workdir=Path(payload.get("workdir", "workspaces/campaign")),
            system_source=payload.get("system_source"),
            simulation=SimulationDefaults.model_validate(payload.get("simulation", {})),
            analysis=AnalysisDefaults.model_validate(payload.get("analysis", {})),
            umbrella=UmbrellaDefaults.model_validate(payload.get("umbrella", {})),
            strategy_id=payload.get("strategy_id"),
            metadata=payload.get("metadata", {}),
            created_at=datetime.fromisoformat(payload["created_at"])
            if payload.get("created_at")
            else utc_now(),
        )


def campaign_fingerprint(spec: CampaignSpec, *, system_digest: str | None = None) -> str:
    """Digest of the deterministic parts of a campaign specification.

    Excludes timestamps, ids and paths, so two campaigns that will produce the same
    inputs fingerprint identically.  This is what the reproducibility test compares.
    """
    return canonical_hash(
        {
            "polymer_id": spec.polymer_id,
            "simulation": spec.simulation.model_dump(mode="json"),
            "analysis": spec.analysis.model_dump(mode="json"),
            "umbrella": spec.umbrella.model_dump(mode="json"),
            "system_digest": system_digest,
        }
    )


@dataclass
class Campaign:
    """A campaign in progress, with its artifacts, gates and state."""

    spec: CampaignSpec
    status: CampaignStatus = CampaignStatus.CREATED
    system: SystemManifest | None = None
    system_artifact_id: str | None = None
    system_report: GateReport | None = None
    replicas: ReplicaSet | None = None
    actions: list[Action] = field(default_factory=list)
    records: list[ExecutionRecord] = field(default_factory=list)
    gate_reports: dict[str, GateReport] = field(default_factory=dict)
    software: dict[str, str] = field(default_factory=dict)
    blocked_reason: str = ""

    @property
    def campaign_id(self) -> str:
        return self.spec.campaign_id

    @property
    def workdir(self) -> Path:
        return self.spec.workdir

    def record_for(self, action_id: str) -> ExecutionRecord | None:
        return next((r for r in self.records if r.label == action_id), None)

    # -- manifest ------------------------------------------------------
    def manifest(self) -> dict[str, Any]:
        """Everything needed to reconstruct and audit this campaign."""
        return {
            "manifest_version": MANIFEST_VERSION,
            "campaign_id": self.campaign_id,
            "status": self.status.value,
            "objective_id": self.spec.objective_id,
            "hypothesis_id": self.spec.hypothesis_id,
            "question": self.spec.question,
            "polymer": {"polymer_id": self.spec.polymer_id, "name": self.spec.polymer_name},
            "specification": self.spec.as_dict(),
            "fingerprint": campaign_fingerprint(self.spec, system_digest=self._system_digest()),
            "force_field": self.spec.simulation.force_field,
            "water_model": self.spec.simulation.water_model,
            "software": self.software,
            # String keys: JSON has no integer keys, so an int-keyed dict would come
            # back as strings and make a stored manifest compare unequal to a fresh one.
            "random_seeds": (
                {str(r.index): r.seeds for r in self.replicas.replicas} if self.replicas else {}
            ),
            "replicas": {
                "requested": self.spec.simulation.replicas,
                "materialised": self.replicas.n_replicas if self.replicas else 0,
                "seeds_distinct": self.replicas.seeds_are_distinct() if self.replicas else None,
                "directories": [r.directory for r in self.replicas.replicas] if self.replicas else [],
            },
            "system": {
                "root": self.system.root if self.system else None,
                "coordinates": self.system.coordinates if self.system else None,
                "topology": self.system.topology if self.system else None,
                "digest": self._system_digest(),
                "artifact_id": self.system_artifact_id,
                "validation": self._report_dict(self.system_report),
            },
            "analysis_settings": self.spec.analysis.model_dump(mode="json"),
            "umbrella_settings": self.spec.umbrella.model_dump(mode="json"),
            "actions": [a.model_dump(mode="json") for a in self.actions],
            "execution_records": [r.model_dump(mode="json") for r in self.records],
            "gates": {name: self._report_dict(report) for name, report in self.gate_reports.items()},
            "blocked_reason": self.blocked_reason,
            "generated_at": utc_now().isoformat(),
        }

    def _system_digest(self) -> str | None:
        if self.system is None:
            return None
        return canonical_hash(sorted(a.sha256 for a in self.system.artifacts))

    @staticmethod
    def _report_dict(report: GateReport | None) -> dict[str, Any] | None:
        if report is None:
            return None
        return {
            "status": report.status.value,
            "promotable": report.promotable,
            "summary": report.summary(),
            "gates": [g.model_dump(mode="json") for g in report.gates],
        }

    def write_manifest(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.workdir / "campaign_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.manifest(), indent=2, default=str), encoding="utf-8")
        return target


class CampaignBuilder:
    """Builds campaigns from a validated system.  Never fabricates inputs."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        graph: ProvenanceGraph | None = None,
        software: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.graph = graph if graph is not None else ProvenanceGraph()
        self.software = software or {}

    def create(self, spec: CampaignSpec) -> Campaign:
        campaign = Campaign(spec=spec, software=dict(self.software))
        campaign.workdir.mkdir(parents=True, exist_ok=True)
        logger.info("Created campaign %s in %s", spec.campaign_id, spec.workdir)
        return campaign

    def attach_system(self, campaign: Campaign, imported: ImportedSystem) -> Campaign:
        """Attach an imported system, recording its validation verdict.

        A system that fails validation is attached anyway -- with the campaign marked
        ``BLOCKED`` -- so the failure is inspectable rather than lost to an exception.
        """
        campaign.system = imported.manifest
        campaign.system_artifact_id = imported.artifact_id
        campaign.system_report = imported.report
        campaign.gate_reports["system"] = imported.report
        if imported.report.promotable:
            campaign.status = CampaignStatus.SYSTEM_VALIDATED
        else:
            campaign.status = CampaignStatus.BLOCKED
            campaign.blocked_reason = (
                "system failed validation: "
                + "; ".join(g.message for g in imported.report.gates if g.status.blocks_promotion)
            )
            logger.warning("Campaign %s blocked: %s", campaign.campaign_id, campaign.blocked_reason)
        return campaign

    def plan(self, campaign: Campaign, *, gromacs_version: str | None = None) -> Campaign:
        """Materialise replicas and build the action graph.

        Idempotent: re-planning an already-planned campaign rebuilds the same inputs
        with the same seeds, which is what makes resume safe.
        """
        if campaign.system is None:
            raise PolymerEngineError(
                "Cannot plan a campaign with no system attached", campaign_id=campaign.campaign_id
            )
        if campaign.status is CampaignStatus.BLOCKED:
            raise SystemValidationError(
                "Refusing to plan a campaign whose system failed validation",
                campaign_id=campaign.campaign_id,
                reason=campaign.blocked_reason,
            )

        problems = validate_parameters(campaign.spec.simulation, gromacs_version=gromacs_version)
        if problems:
            campaign.status = CampaignStatus.BLOCKED
            campaign.blocked_reason = "invalid simulation parameters: " + "; ".join(problems)
            raise PolymerEngineError(campaign.blocked_reason, campaign_id=campaign.campaign_id)

        campaign.replicas = materialize_replicas(
            campaign.system,
            campaign.workdir / "replicas",
            campaign.spec.simulation,
            graph=self.graph,
            system_artifact_id=campaign.system_artifact_id,
            gromacs_version=gromacs_version,
            validate_first=False,  # already validated in attach_system
            artifact_prefix=campaign.campaign_id,
        )
        campaign.actions = self._build_actions(campaign)
        campaign.records = [
            ExecutionRecord(kind=action.kind, label=action.id) for action in campaign.actions
        ]
        campaign.status = CampaignStatus.PLANNED
        campaign.write_manifest()
        logger.info(
            "Planned campaign %s: %d replicas, %d actions",
            campaign.campaign_id,
            campaign.replicas.n_replicas,
            len(campaign.actions),
        )
        return campaign

    def _build_actions(self, campaign: Campaign) -> list[Action]:
        """The action graph: equilibrate each replica, then analyse them together."""
        assert campaign.replicas is not None
        spec = campaign.spec
        actions: list[Action] = []

        # Cost model: roughly proportional to simulated time.  Declared explicitly so
        # the planner is comparing real estimates rather than assuming everything is free.
        total_ns = spec.simulation.nvt_ns + spec.simulation.npt_ns + spec.simulation.production_ns
        per_replica_hours = total_ns * 0.5  # ~2 ns/hour on one GPU for a modest system

        for replica in campaign.replicas.replicas:
            actions.append(
                Action(
                    kind="gromacs_equilibrate",
                    title=f"Equilibrate and run replica {replica.index + 1}",
                    question=spec.question or f"What is the equilibrium behaviour of {spec.polymer_id}?",
                    campaign_id=campaign.campaign_id,
                    hypothesis_id=spec.hypothesis_id,
                    strategy_id=spec.strategy_id,
                    inputs={
                        "workdir": replica.directory,
                        "replica_index": replica.index,
                        "seeds": replica.seeds,
                        "requires_tool": "gromacs",
                    },
                    cost=CostEstimate(
                        gpu_hours=per_replica_hours,
                        determination=Determination.KNOWN,
                        basis=f"{total_ns:.1f} ns at ~2 ns/GPU-hour",
                    ),
                    expected_information_gain=0.6,
                    uncertainty_reduction=0.5,
                    design_relevance=0.7,
                    risk=0.2,
                )
            )

        analysis = Action(
            kind="analyze_replicas",
            title="Analyse replica convergence and agreement",
            question="Do the replicas agree, and has the observable converged?",
            campaign_id=campaign.campaign_id,
            hypothesis_id=spec.hypothesis_id,
            strategy_id=spec.strategy_id,
            inputs={
                "workdir": str(campaign.workdir),
                "replica_dirs": [r.directory for r in campaign.replicas.replicas],
                "metrics": ["density", "temperature", "pressure", "potential"],
            },
            cost=CostEstimate(cpu_hours=0.5, determination=Determination.KNOWN, basis="analysis only"),
            expected_information_gain=0.9,
            uncertainty_reduction=0.8,
            design_relevance=0.9,
            risk=0.05,
            depends_on=[a.id for a in actions],
        )
        actions.append(analysis)
        return actions


def load_campaign(store: Any, campaign_id: str) -> Campaign | None:
    """Reload a campaign from the store so it can be continued."""
    payload = store.get_campaign(campaign_id)
    if payload is None:
        return None
    spec = CampaignSpec.from_dict(payload["specification"])
    campaign = Campaign(spec=spec, status=CampaignStatus(payload.get("status", "created")))
    campaign.software = payload.get("software", {})
    campaign.blocked_reason = payload.get("blocked_reason", "")
    campaign.actions = store.list_actions(campaign_id=campaign_id)
    campaign.records = store.list_execution_records(campaign_id=campaign_id)
    system = payload.get("system") or {}
    if system.get("root"):
        manifest_path = Path(system["root"]) / "system_manifest.json"
        if manifest_path.exists():
            campaign.system = SystemManifest.load(manifest_path)
            campaign.system_artifact_id = system.get("artifact_id")

    # Restore the replica set from disk.  Without this, saving a reloaded campaign
    # would write a manifest with no seeds and silently destroy the reproducibility
    # record written at plan time.
    replicas_path = campaign.workdir / "replicas" / "replicas.json"
    if replicas_path.exists():
        campaign.replicas = _load_replica_set(replicas_path)
    return campaign


def _load_replica_set(path: Path) -> ReplicaSet:
    payload = json.loads(path.read_text(encoding="utf-8"))
    replica_set = ReplicaSet(
        root=payload.get("root", str(path.parent)),
        system_root=payload.get("system_root", ""),
        parameters=payload.get("parameters", {}),
    )
    for item in payload.get("replicas", []):
        replica_set.replicas.append(
            Replica(
                index=int(item["index"]),
                directory=item["directory"],
                seeds={k: int(v) for k, v in (item.get("seeds") or {}).items()},
                stages=list(item.get("stages") or []),
                coordinates=item.get("coordinates"),
                topology=item.get("topology"),
            )
        )
    return replica_set


def save_campaign(store: Any, campaign: Campaign) -> None:
    """Persist a campaign and everything hanging off it."""
    store.save_campaign(
        campaign.campaign_id,
        campaign.manifest(),
        status=campaign.status.value,
        objective_id=campaign.spec.objective_id,
        polymer_id=campaign.spec.polymer_id,
    )
    for action in campaign.actions:
        store.save_action(action)
    for record in campaign.records:
        store.save_execution_record(record, campaign_id=campaign.campaign_id, action_id=record.label)


__all__ = [
    "MANIFEST_VERSION",
    "Campaign",
    "CampaignBuilder",
    "CampaignSpec",
    "CampaignStatus",
    "campaign_fingerprint",
    "load_campaign",
    "save_campaign",
]
