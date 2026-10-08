"""Turning a downloaded archive into a validated simulation system.

The pipeline is:

    download -> checksum -> safe extraction -> artifact discovery -> classification
    -> topology/coordinate consistency -> manifest -> provenance registration

Validation returns a :class:`GateReport`, not a boolean.  A check that could not be
evaluated (a topology whose molecule types live in an unresolved library include, say)
is ``INCONCLUSIVE``, and inconclusive blocks promotion just as firmly as a failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from polymer_engine.core.errors import SystemValidationError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination, GateReport, GateResult, GateStatus
from polymer_engine.core.provenance import ProvenanceGraph, sha256_file
from polymer_engine.simulation.archive import ExtractionReport, safe_extract
from polymer_engine.simulation.formats import GroFile, Topology, read_gro, read_topology

logger = get_logger("simulation.system")

#: Extension -> artifact kind.  Order matters only for the ``.top`` special case.
EXTENSION_KINDS: dict[str, str] = {
    ".gro": "coordinates",
    ".g96": "coordinates",
    ".pdb": "structure",
    ".cif": "structure",
    ".top": "topology",
    ".itp": "include_topology",
    ".prm": "forcefield",
    ".rtf": "forcefield",
    ".str": "forcefield",
    ".par": "forcefield",
    ".mdp": "mdp",
    ".ndx": "index",
    ".tpr": "run_input",
    ".xtc": "trajectory",
    ".trr": "trajectory",
    ".edr": "energy",
    ".cpt": "checkpoint",
    ".xvg": "analysis",
    ".log": "log",
    ".dat": "plumed_or_data",
}


@dataclass(frozen=True, slots=True)
class SystemArtifact:
    kind: str
    path: str
    relative_path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def classify_artifact(path: Path) -> str:
    name = path.name.lower()
    if name == "topol.top":
        return "topology"
    return EXTENSION_KINDS.get(path.suffix.lower(), "other")


def discover_artifacts(root: str | Path) -> list[SystemArtifact]:
    """Hash and classify every file under ``root``."""
    root = Path(root)
    if not root.is_dir():
        raise SystemValidationError("System directory does not exist", path=str(root))
    artifacts: list[SystemArtifact] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        artifacts.append(
            SystemArtifact(
                kind=classify_artifact(path),
                path=str(path),
                relative_path=str(path.relative_to(root)),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return artifacts


@dataclass
class SystemManifest:
    """Everything we know about one simulation system on disk."""

    root: str
    artifacts: list[SystemArtifact] = field(default_factory=list)
    coordinates: str | None = None
    topology: str | None = None
    include_topologies: list[str] = field(default_factory=list)
    mdps: list[str] = field(default_factory=list)
    index_files: list[str] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)

    def by_kind(self, kind: str) -> list[SystemArtifact]:
        return [a for a in self.artifacts if a.kind == kind]

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "coordinates": self.coordinates,
            "topology": self.topology,
            "include_topologies": self.include_topologies,
            "mdps": self.mdps,
            "index_files": self.index_files,
            "source": self.source,
            "artifacts": [a.as_dict() for a in self.artifacts],
        }

    def write(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else Path(self.root) / "system_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> SystemManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls(root=data["root"], source=data.get("source", {}))
        manifest.artifacts = [SystemArtifact(**a) for a in data.get("artifacts", [])]
        manifest.coordinates = data.get("coordinates")
        manifest.topology = data.get("topology")
        manifest.include_topologies = data.get("include_topologies", [])
        manifest.mdps = data.get("mdps", [])
        manifest.index_files = data.get("index_files", [])
        return manifest


def _pick_coordinates(artifacts: list[SystemArtifact]) -> str | None:
    """Choose the system coordinate file.

    Prefers a conventional name, then the shallowest, then the largest.  Depth first
    matters because replica subdirectories contain per-stage ``.gro`` files that must
    not be mistaken for the system's input coordinates.
    """
    candidates = [a for a in artifacts if a.kind == "coordinates"]
    if not candidates:
        return None
    preferred = {"system.gro", "step5_input.gro", "conf.gro", "solvated.gro", "npt.gro"}
    ranked = sorted(
        candidates,
        key=lambda a: (
            Path(a.relative_path).name.lower() not in preferred,
            len(Path(a.relative_path).parts),
            -a.size_bytes,
        ),
    )
    return ranked[0].path


def _pick_topology(artifacts: list[SystemArtifact]) -> str | None:
    candidates = [a for a in artifacts if a.kind == "topology"]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda a: (
            Path(a.relative_path).name.lower() != "topol.top",
            len(Path(a.relative_path).parts),
        ),
    )
    return ranked[0].path


def build_manifest(root: str | Path, *, source: dict[str, Any] | None = None) -> SystemManifest:
    root = Path(root)
    artifacts = discover_artifacts(root)
    manifest = SystemManifest(root=str(root), artifacts=artifacts, source=source or {})
    manifest.coordinates = _pick_coordinates(artifacts)
    manifest.topology = _pick_topology(artifacts)
    manifest.include_topologies = [a.path for a in artifacts if a.kind == "include_topology"]
    manifest.mdps = [a.path for a in artifacts if a.kind == "mdp"]
    manifest.index_files = [a.path for a in artifacts if a.kind == "index"]
    return manifest


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
#: A coordinate value further than this from the origin is physically implausible for
#: a molecular system in nm and usually signals a corrupt or mis-parsed file.
MAX_PLAUSIBLE_COORDINATE_NM = 1.0e4


def validate_system(
    root: str | Path,
    *,
    manifest: SystemManifest | None = None,
    require_box: bool = True,
) -> GateReport:
    """Run every structural check on a system directory."""
    root = Path(root)
    report = GateReport(name="system_validation")
    if not root.is_dir():
        report.gates.append(
            GateResult(
                gate="system_directory",
                status=GateStatus.FAIL,
                message=f"System directory does not exist: {root}",
            )
        )
        return report

    manifest = manifest or build_manifest(root)

    # -- presence ------------------------------------------------------
    report.gates.append(_presence_gate("coordinates_present", manifest.coordinates, "coordinate (.gro) file"))
    report.gates.append(_presence_gate("topology_present", manifest.topology, "topology (.top) file"))

    if manifest.coordinates is None or manifest.topology is None:
        return report

    # -- coordinates ---------------------------------------------------
    try:
        gro = read_gro(manifest.coordinates)
    except SystemValidationError as exc:
        report.gates.append(
            GateResult(gate="coordinates_parse", status=GateStatus.FAIL, message=str(exc))
        )
        return report
    report.gates.append(
        GateResult(
            gate="coordinates_parse",
            status=GateStatus.PASS,
            message=f"Parsed {gro.n_atoms} atoms from {Path(manifest.coordinates).name}",
            value=float(gro.n_atoms),
            units="1",
        )
    )
    report.gates.extend(_coordinate_gates(gro, require_box=require_box))

    # -- topology ------------------------------------------------------
    try:
        topology = read_topology(manifest.topology, include_dirs=[Path(manifest.topology).parent, root])
    except SystemValidationError as exc:
        report.gates.append(
            GateResult(gate="topology_parse", status=GateStatus.FAIL, message=str(exc))
        )
        return report
    report.gates.extend(_topology_gates(topology))
    report.gates.append(_consistency_gate(gro, topology))
    return report


def _presence_gate(name: str, value: str | None, description: str) -> GateResult:
    if value:
        return GateResult(
            gate=name, status=GateStatus.PASS, message=f"Found {description}: {Path(value).name}"
        )
    return GateResult(gate=name, status=GateStatus.FAIL, message=f"No {description} found")


def _coordinate_gates(gro: GroFile, *, require_box: bool) -> list[GateResult]:
    gates: list[GateResult] = []

    if gro.n_atoms <= 0:
        gates.append(
            GateResult(
                gate="atom_count",
                status=GateStatus.FAIL,
                message="Coordinate file declares no atoms",
                value=float(gro.n_atoms),
                units="1",
            )
        )
    else:
        gates.append(
            GateResult(
                gate="atom_count",
                status=GateStatus.PASS,
                message=f"{gro.n_atoms} atoms",
                value=float(gro.n_atoms),
                units="1",
            )
        )

    if not gro.positions_finite:
        n_bad = int(np.count_nonzero(~np.isfinite(gro.positions)))
        gates.append(
            GateResult(
                gate="coordinates_finite",
                status=GateStatus.FAIL,
                message=f"{n_bad} coordinate components are NaN or infinite",
                value=float(n_bad),
                units="1",
            )
        )
    else:
        magnitude = float(np.abs(gro.positions).max()) if gro.n_atoms else 0.0
        status = GateStatus.PASS if magnitude <= MAX_PLAUSIBLE_COORDINATE_NM else GateStatus.FAIL
        gates.append(
            GateResult(
                gate="coordinates_finite",
                status=status,
                message=(
                    "All coordinates are finite and within a plausible range"
                    if status is GateStatus.PASS
                    else f"Largest coordinate magnitude {magnitude:.3g} nm is implausible"
                ),
                value=magnitude,
                threshold=MAX_PLAUSIBLE_COORDINATE_NM,
                units="nm",
            )
        )

    if not gro.has_box:
        gates.append(
            GateResult(
                gate="box_defined",
                status=GateStatus.FAIL if require_box else GateStatus.WARN,
                message="Coordinate file has no box vectors",
            )
        )
    else:
        lengths = gro.box[:3]
        if any(v <= 0 for v in lengths):
            gates.append(
                GateResult(
                    gate="box_defined",
                    status=GateStatus.FAIL,
                    message=f"Box has a non-positive dimension: {lengths}",
                )
            )
        else:
            volume = gro.box_volume_nm3 or 0.0
            gates.append(
                GateResult(
                    gate="box_defined",
                    status=GateStatus.PASS,
                    message=f"Box {lengths[0]:.3f} x {lengths[1]:.3f} x {lengths[2]:.3f} nm",
                    value=volume,
                    units="1",
                )
            )
            gates.append(_box_fit_gate(gro, lengths))
    return gates


def _box_fit_gate(gro: GroFile, lengths: tuple[float, ...]) -> GateResult:
    """Check the solute actually fits inside the declared box.

    A molecule wider than its periodic box interacts with its own image, which
    silently corrupts every energetic quantity derived from the run.
    """
    extent = gro.extent_nm()
    overflow = [i for i in range(3) if extent[i] > lengths[i] * 1.001]
    if overflow:
        axes = ", ".join("xyz"[i] for i in overflow)
        return GateResult(
            gate="contents_fit_box",
            status=GateStatus.FAIL,
            message=f"Atom coordinates exceed the box along {axes}",
            evidence={"extent_nm": list(extent), "box_nm": list(lengths[:3])},
            units="nm",
        )
    return GateResult(
        gate="contents_fit_box",
        status=GateStatus.PASS,
        message="All atoms lie within the declared box",
        evidence={"extent_nm": list(extent), "box_nm": list(lengths[:3])},
        units="nm",
    )


def _topology_gates(topology: Topology) -> list[GateResult]:
    gates: list[GateResult] = []

    if topology.missing_includes:
        gates.append(
            GateResult(
                gate="topology_includes",
                status=GateStatus.FAIL,
                message=f"Topology references {len(topology.missing_includes)} missing local include(s)",
                evidence={"missing": topology.missing_includes},
            )
        )
    else:
        library = [i for i in topology.includes if i not in topology.missing_includes]
        gates.append(
            GateResult(
                gate="topology_includes",
                status=GateStatus.PASS,
                message=f"All {len(library)} include(s) resolved or deferred to the GROMACS library",
                evidence={"resolved": topology.resolved_includes},
            )
        )

    if not topology.has_molecules_section:
        gates.append(
            GateResult(
                gate="molecules_section",
                status=GateStatus.FAIL,
                message="Topology has no [ molecules ] section, so it defines no system composition",
            )
        )
    elif not topology.molecules:
        gates.append(
            GateResult(
                gate="molecules_section",
                status=GateStatus.FAIL,
                message="[ molecules ] section is empty",
            )
        )
    else:
        total = sum(count for _, count in topology.molecules)
        gates.append(
            GateResult(
                gate="molecules_section",
                status=GateStatus.PASS,
                message=f"{len(topology.molecules)} molecule entries, {total} molecules total",
                value=float(total),
                units="1",
            )
        )

    unresolved = topology.unresolved_molecule_types
    if unresolved:
        gates.append(
            GateResult(
                gate="molecule_types_resolved",
                status=GateStatus.INCONCLUSIVE,
                message=(
                    f"{len(unresolved)} molecule type(s) are defined in unresolved includes; "
                    "atom counts cannot be verified locally"
                ),
                evidence={"unresolved": unresolved},
            )
        )
    elif topology.molecules:
        gates.append(
            GateResult(
                gate="molecule_types_resolved",
                status=GateStatus.PASS,
                message=f"All {len(topology.molecule_types)} molecule types resolved",
                value=float(len(topology.molecule_types)),
                units="1",
            )
        )
    return gates


def _consistency_gate(gro: GroFile, topology: Topology) -> GateResult:
    """The check that actually matters: do topology and coordinates describe one system?"""
    expected = topology.total_atoms
    if expected is None:
        return GateResult(
            gate="topology_matches_coordinates",
            status=GateStatus.INCONCLUSIVE,
            message=(
                "Topology atom count could not be computed (unresolved molecule types); "
                "consistency with the coordinates is unverified"
            ),
            evidence={
                "coordinate_atoms": gro.n_atoms,
                "unresolved_molecule_types": topology.unresolved_molecule_types,
            },
        )
    if expected != gro.n_atoms:
        return GateResult(
            gate="topology_matches_coordinates",
            status=GateStatus.FAIL,
            message=f"Topology implies {expected} atoms but the coordinate file has {gro.n_atoms}",
            value=float(gro.n_atoms),
            threshold=float(expected),
            units="1",
        )
    return GateResult(
        gate="topology_matches_coordinates",
        status=GateStatus.PASS,
        message=f"Topology and coordinates agree on {expected} atoms",
        value=float(expected),
        units="1",
    )


# --------------------------------------------------------------------------
# End-to-end import
# --------------------------------------------------------------------------
@dataclass
class ImportedSystem:
    manifest: SystemManifest
    report: GateReport
    extraction: ExtractionReport | None
    artifact_id: str | None = None

    @property
    def usable(self) -> bool:
        return self.report.promotable

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.as_dict(),
            "validation": {
                "status": self.report.status.value,
                "promotable": self.report.promotable,
                "gates": [g.model_dump(mode="json") for g in self.report.gates],
            },
            "extraction": self.extraction.as_dict() if self.extraction else None,
            "artifact_id": self.artifact_id,
        }


def import_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    graph: ProvenanceGraph | None = None,
    source: dict[str, Any] | None = None,
    max_total_bytes: int = 2 * 1024**3,
    max_members: int = 200_000,
    require_box: bool = True,
) -> ImportedSystem:
    """Extract, inspect, validate, and register a system archive."""
    archive = Path(archive)
    destination = Path(destination)
    archive_digest = sha256_file(archive)

    extraction = safe_extract(
        archive, destination, max_total_bytes=max_total_bytes, max_members=max_members
    )
    manifest = build_manifest(
        destination,
        source={
            "archive": str(archive),
            "archive_sha256": archive_digest,
            **(source or {}),
        },
    )
    manifest.write()
    report = validate_system(destination, manifest=manifest, require_box=require_box)

    artifact_id: str | None = None
    if graph is not None:
        archive_artifact = graph.register_file(
            archive,
            kind="provider_archive",
            source=str((source or {}).get("provider", "local")),
            parameters={"members": extraction.members_extracted},
        )
        system_artifact = graph.register_derived(
            artifact_id=f"sys_{archive_digest[:16]}",
            kind="simulation_system",
            parents=[archive_artifact.artifact_id],
            parameters={"destination": str(destination), "require_box": require_box},
            payload=[a.sha256 for a in manifest.artifacts],
            validation_state=(
                Determination.KNOWN if report.promotable else Determination.REQUIRES_VALIDATION
            ),
            notes=report.summary(),
        )
        artifact_id = system_artifact.artifact_id

    logger.info("Imported system from %s: %s", archive.name, report.summary())
    return ImportedSystem(manifest=manifest, report=report, extraction=extraction, artifact_id=artifact_id)


def import_directory(
    source_dir: str | Path,
    *,
    graph: ProvenanceGraph | None = None,
    source: dict[str, Any] | None = None,
    require_box: bool = True,
) -> ImportedSystem:
    """Validate a system that is already unpacked on disk."""
    source_dir = Path(source_dir)
    manifest = build_manifest(source_dir, source=source or {"origin": "local-directory"})
    report = validate_system(source_dir, manifest=manifest, require_box=require_box)
    artifact_id: str | None = None
    if graph is not None:
        artifact = graph.register_derived(
            artifact_id=f"sys_{sha256_file(manifest.coordinates)[:16]}"
            if manifest.coordinates
            else f"sys_{abs(hash(str(source_dir))):016x}",
            kind="simulation_system",
            parents=[],
            parameters={"source_dir": str(source_dir)},
            payload=[a.sha256 for a in manifest.artifacts],
            validation_state=(
                Determination.KNOWN if report.promotable else Determination.REQUIRES_VALIDATION
            ),
            notes=report.summary(),
        )
        artifact_id = artifact.artifact_id
    return ImportedSystem(manifest=manifest, report=report, extraction=None, artifact_id=artifact_id)


__all__ = [
    "EXTENSION_KINDS",
    "ImportedSystem",
    "SystemArtifact",
    "SystemManifest",
    "build_manifest",
    "classify_artifact",
    "discover_artifacts",
    "import_archive",
    "import_directory",
    "validate_system",
]
