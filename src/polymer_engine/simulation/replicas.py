"""Replica materialisation.

An "independent replica" is only independent if something about it differs.  This
module copies a validated system into per-replica directories and gives each one a
distinct velocity/thermostat seed, then records those seeds in the manifest so the
run can be reproduced exactly.

Replicas share the same starting coordinates by design: independence comes from
independent velocity generation, which is the standard construction.  What must not
happen -- and what the seed handling here prevents -- is replicas that are byte-identical,
because then replica agreement measures nothing.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymer_engine.core.config import SimulationDefaults
from polymer_engine.core.errors import ParameterValidationError, SystemValidationError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination
from polymer_engine.core.provenance import ProvenanceGraph, sha256_file
from polymer_engine.simulation.mdp import MdpStage, generate_stages
from polymer_engine.simulation.system import SystemManifest, validate_system

logger = get_logger("simulation.replicas")

#: Files copied into each replica directory from the validated system.
COPIED_KINDS = ("coordinates", "topology", "include_topology", "forcefield", "index")


@dataclass
class Replica:
    index: int
    directory: str
    seeds: dict[str, int] = field(default_factory=dict)
    stages: list[str] = field(default_factory=list)
    coordinates: str | None = None
    topology: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "directory": self.directory,
            "seeds": self.seeds,
            "stages": self.stages,
            "coordinates": self.coordinates,
            "topology": self.topology,
        }


@dataclass
class ReplicaSet:
    root: str
    replicas: list[Replica] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    system_root: str = ""

    @property
    def n_replicas(self) -> int:
        return len(self.replicas)

    def seeds_are_distinct(self) -> bool:
        """Every (replica, stage) seed must be unique, or replicas are not independent."""
        seen: set[int] = set()
        for replica in self.replicas:
            for seed in replica.seeds.values():
                if seed in seen:
                    return False
                seen.add(seed)
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "system_root": self.system_root,
            "n_replicas": self.n_replicas,
            "parameters": self.parameters,
            "seeds_distinct": self.seeds_are_distinct(),
            "replicas": [r.as_dict() for r in self.replicas],
        }

    def write(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else Path(self.root) / "replicas.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return target


def materialize_replicas(
    manifest: SystemManifest,
    workdir: str | Path,
    defaults: SimulationDefaults,
    *,
    graph: ProvenanceGraph | None = None,
    system_artifact_id: str | None = None,
    gromacs_version: str | None = None,
    plumed: bool = False,
    validate_first: bool = True,
    artifact_prefix: str | None = None,
) -> ReplicaSet:
    """Create ``defaults.replicas`` independent replica directories.

    Refuses to build anything from a system that does not pass validation -- generating
    inputs for a broken system just moves the failure further downstream, where it is
    more expensive to notice.
    """
    workdir = Path(workdir)
    if defaults.replicas < 1:
        raise ParameterValidationError("At least one replica is required", replicas=defaults.replicas)

    if validate_first:
        report = validate_system(manifest.root, manifest=manifest)
        if not report.promotable:
            raise SystemValidationError(
                "Refusing to materialise replicas from a system that failed validation",
                status=report.status.value,
                failures=[g.message for g in report.gates if g.status.blocks_promotion],
            )

    if not manifest.coordinates or not manifest.topology:
        raise SystemValidationError("System manifest has no coordinates or topology", root=manifest.root)

    workdir.mkdir(parents=True, exist_ok=True)
    # Artifact ids must be unique across campaigns.  Deriving them from the directory
    # name alone gives every campaign a "replica_01_replicas", so two campaigns would
    # silently share provenance records.
    prefix = artifact_prefix or workdir.parent.name or workdir.name
    replica_set = ReplicaSet(
        root=str(workdir),
        system_root=manifest.root,
        parameters={
            **defaults.model_dump(mode="json"),
            "gromacs_version": gromacs_version,
            "plumed": plumed,
        },
    )

    sources = [a for a in manifest.artifacts if a.kind in COPIED_KINDS]

    for index in range(defaults.replicas):
        directory = workdir / f"replica_{index + 1:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        replica = Replica(index=index, directory=str(directory))

        for artifact in sources:
            source = Path(artifact.path)
            # Flatten into the replica directory so relative #include paths resolve.
            target = directory / source.name
            shutil.copy2(source, target)
            if artifact.path == manifest.coordinates:
                canonical = directory / "system.gro"
                if target != canonical:
                    shutil.copy2(source, canonical)
                replica.coordinates = str(canonical)
            if artifact.path == manifest.topology:
                canonical = directory / "topol.top"
                if target != canonical:
                    shutil.copy2(source, canonical)
                replica.topology = str(canonical)

        stages: list[MdpStage] = generate_stages(
            defaults, replica_index=index, gromacs_version=gromacs_version, plumed=plumed
        )
        for stage in stages:
            stage.write(directory)
            replica.stages.append(stage.name)
            if stage.seed is not None:
                replica.seeds[stage.name] = stage.seed

        (directory / "replica.json").write_text(
            json.dumps(
                {
                    "index": index,
                    "seeds": replica.seeds,
                    "stages": [
                        {
                            "name": s.name,
                            "input_structure": s.input_structure,
                            "nsteps": s.nsteps,
                            "ns": s.ns,
                            "parameters": s.parameters,
                        }
                        for s in stages
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if graph is not None:
            graph.register_derived(
                artifact_id=f"{prefix}:replica_{index + 1:02d}",
                kind="replica_inputs",
                parents=[system_artifact_id] if system_artifact_id else [],
                parameters={"replica_index": index, "seeds": replica.seeds},
                payload=[sha256_file(directory / f"{s.name}.mdp") for s in stages],
                random_seed=replica.seeds.get("nvt"),
                validation_state=Determination.KNOWN,
                notes=f"Replica {index + 1} of {defaults.replicas}",
            )
        replica_set.replicas.append(replica)

    if not replica_set.seeds_are_distinct():  # pragma: no cover - guards the seed derivation
        raise ParameterValidationError(
            "Generated replicas share a random seed, so they would not be independent"
        )

    replica_set.write()
    logger.info("Materialised %d replicas in %s", replica_set.n_replicas, workdir)
    return replica_set


__all__ = ["COPIED_KINDS", "Replica", "ReplicaSet", "materialize_replicas"]
