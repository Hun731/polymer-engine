"""Provider-independent system construction.

The orchestrator asks for a system; it does not know or care which backend produces
one.  A backend takes a :class:`SystemBuildRequest` and returns a
:class:`SystemBuildResult` carrying coordinates, topology, parameters, box, force-field
metadata and complete provenance -- or an honest failure.

Backends declare what they can do.  ``CharmmGuiImportBackend`` can *import* a completed
CHARMM-GUI job but cannot *submit* one, because CHARMM-GUI publishes no submission
endpoint; that limitation is declared rather than worked around.

Nothing here fabricates force-field parameters.  A backend either obtains real
parameters from a real source or reports that it cannot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from polymer_engine.core.errors import PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination, GateReport
from polymer_engine.core.provenance import ProvenanceGraph, canonical_hash
from polymer_engine.simulation.system import ImportedSystem, SystemManifest, import_archive, import_directory

logger = get_logger("simulation.builder")


class BuildStatus(str, Enum):
    BUILT = "BUILT"
    IMPORTED = "IMPORTED"
    FAILED = "FAILED"
    #: The backend cannot do this at all (e.g. no submission endpoint exists).
    UNSUPPORTED = "UNSUPPORTED"
    #: A human must supply something the engine cannot decide or obtain.
    REQUIRES_EXPERT_DECISION = "REQUIRES_EXPERT_DECISION"
    #: Credentials or an external job id are missing.
    REQUIRES_INPUT = "REQUIRES_INPUT"


class BuildCapability(str, Enum):
    BUILD_FROM_SMILES = "build_from_smiles"
    BUILD_FROM_STRUCTURE = "build_from_structure"
    IMPORT_ARCHIVE = "import_archive"
    IMPORT_DIRECTORY = "import_directory"
    SOLVATE = "solvate"
    ADD_IONS = "add_ions"
    ASSIGN_PARAMETERS = "assign_parameters"
    SUBMIT_REMOTE_JOB = "submit_remote_job"


@dataclass
class SystemBuildRequest:
    """What the caller wants built.

    ``force_field`` is required and is never guessed: the choice determines every
    number the resulting simulation produces, and no code can make it responsibly.
    """

    polymer_id: str
    force_field: str
    label: str = "system"
    repeat_unit_smiles: str | None = None
    degree_of_polymerization: int | None = None
    n_chains: int = 1
    water_model: str | None = None
    solvate: bool = False
    box_nm: tuple[float, float, float] | None = None
    target_density_kg_m3: float | None = None
    temperature_k: float = 300.0
    neutralize: bool = False
    salt_concentration_m: float | None = None
    source_archive: str | None = None
    source_directory: str | None = None
    external_job_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.polymer_id.strip():
            problems.append("polymer_id is required")
        if not self.force_field.strip() or self.force_field.upper() == "UNSPECIFIED":
            problems.append(
                "force_field must name a real force field; the engine will not choose one"
            )
        if self.n_chains < 1:
            problems.append("n_chains must be at least 1")
        if self.degree_of_polymerization is not None and self.degree_of_polymerization < 1:
            problems.append("degree_of_polymerization must be at least 1")
        if self.solvate and not self.water_model:
            problems.append("solvation requires a water model")
        if self.box_nm is not None and any(v <= 0 for v in self.box_nm):
            problems.append("box dimensions must be positive")
        if self.target_density_kg_m3 is not None and self.target_density_kg_m3 <= 0:
            problems.append("target density must be positive")
        if self.salt_concentration_m is not None and self.salt_concentration_m < 0:
            problems.append("salt concentration cannot be negative")
        return problems

    def fingerprint(self) -> str:
        return canonical_hash(
            {
                "polymer_id": self.polymer_id,
                "force_field": self.force_field,
                "repeat_unit_smiles": self.repeat_unit_smiles,
                "degree_of_polymerization": self.degree_of_polymerization,
                "n_chains": self.n_chains,
                "water_model": self.water_model,
                "solvate": self.solvate,
                "box_nm": list(self.box_nm) if self.box_nm else None,
                "target_density_kg_m3": self.target_density_kg_m3,
                "neutralize": self.neutralize,
                "salt_concentration_m": self.salt_concentration_m,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id,
            "label": self.label,
            "force_field": self.force_field,
            "repeat_unit_smiles": self.repeat_unit_smiles,
            "degree_of_polymerization": self.degree_of_polymerization,
            "n_chains": self.n_chains,
            "water_model": self.water_model,
            "solvate": self.solvate,
            "box_nm": list(self.box_nm) if self.box_nm else None,
            "target_density_kg_m3": self.target_density_kg_m3,
            "temperature_k": self.temperature_k,
            "neutralize": self.neutralize,
            "salt_concentration_m": self.salt_concentration_m,
            "source_archive": self.source_archive,
            "source_directory": self.source_directory,
            "external_job_id": self.external_job_id,
            "metadata": self.metadata,
            "fingerprint": self.fingerprint(),
        }


@dataclass
class SystemBuildResult:
    """What a backend produced, or why it could not."""

    status: BuildStatus
    backend: str
    request: SystemBuildRequest
    manifest: SystemManifest | None = None
    validation: GateReport | None = None
    artifact_id: str | None = None
    force_field: str | None = None
    water_model: str | None = None
    parameter_source: str | None = None
    diagnostics: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """A system is usable only if it exists *and* passed structural validation."""
        return (
            self.status in {BuildStatus.BUILT, BuildStatus.IMPORTED}
            and self.manifest is not None
            and self.validation is not None
            and self.validation.promotable
        )

    @property
    def determination(self) -> Determination:
        if self.usable:
            return Determination.KNOWN
        if self.status is BuildStatus.UNSUPPORTED:
            return Determination.UNSUPPORTED
        if self.status is BuildStatus.REQUIRES_EXPERT_DECISION:
            return Determination.REQUIRES_VALIDATION
        return Determination.REQUIRES_VALIDATION

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "backend": self.backend,
            "usable": self.usable,
            "determination": self.determination.value,
            "request": self.request.as_dict(),
            "force_field": self.force_field,
            "water_model": self.water_model,
            "parameter_source": self.parameter_source,
            "system_root": self.manifest.root if self.manifest else None,
            "coordinates": self.manifest.coordinates if self.manifest else None,
            "topology": self.manifest.topology if self.manifest else None,
            "artifact_id": self.artifact_id,
            "validation": None
            if self.validation is None
            else {
                "status": self.validation.status.value,
                "promotable": self.validation.promotable,
                "summary": self.validation.summary(),
            },
            "diagnostics": self.diagnostics,
            "required_actions": self.required_actions,
            "provenance": self.provenance,
        }


class SystemBuilderBackend(ABC):
    """A source of simulation systems."""

    name: str = "backend"
    capabilities: frozenset[BuildCapability] = frozenset()

    @abstractmethod
    def build(
        self,
        request: SystemBuildRequest,
        destination: str | Path,
        *,
        graph: ProvenanceGraph | None = None,
    ) -> SystemBuildResult:
        """Produce a system, or explain why not.  Must not raise for expected failures."""

    def supports(self, capability: BuildCapability) -> bool:
        return capability in self.capabilities

    def can_handle(self, request: SystemBuildRequest) -> bool:  # noqa: ARG002
        """Whether this backend is applicable to the request at all."""
        return True

    def _reject(
        self, request: SystemBuildRequest, status: BuildStatus, reason: str, *actions: str
    ) -> SystemBuildResult:
        return SystemBuildResult(
            status=status,
            backend=self.name,
            request=request,
            diagnostics=[reason],
            required_actions=list(actions),
        )


class LocalDirectoryBackend(SystemBuilderBackend):
    """Adopt a system that already exists on disk.

    The simplest and most trustworthy backend: someone built the system with a tool
    that does it properly, and the engine validates and registers it.
    """

    name = "local_directory"
    capabilities = frozenset({BuildCapability.IMPORT_DIRECTORY, BuildCapability.IMPORT_ARCHIVE})

    def can_handle(self, request: SystemBuildRequest) -> bool:
        return bool(request.source_directory or request.source_archive)

    def build(
        self,
        request: SystemBuildRequest,
        destination: str | Path,
        *,
        graph: ProvenanceGraph | None = None,
    ) -> SystemBuildResult:
        problems = request.validate()
        if problems:
            return self._reject(
                request, BuildStatus.REQUIRES_EXPERT_DECISION,
                "; ".join(problems), "supply the missing build parameters",
            )
        if not self.can_handle(request):
            return self._reject(
                request, BuildStatus.REQUIRES_INPUT,
                "no source directory or archive was supplied",
                "set source_directory or source_archive on the request",
            )

        destination = Path(destination)
        try:
            if request.source_archive:
                source = Path(request.source_archive)
                if not source.is_file():
                    return self._reject(
                        request, BuildStatus.FAILED, f"archive not found: {source}"
                    )
                imported = import_archive(
                    source, destination, graph=graph,
                    source={"backend": self.name, "archive": str(source)},
                )
            else:
                source = Path(request.source_directory or "")
                if not source.is_dir():
                    return self._reject(
                        request, BuildStatus.FAILED, f"directory not found: {source}"
                    )
                imported = import_directory(
                    source, graph=graph, source={"backend": self.name, "directory": str(source)}
                )
        except PolymerEngineError as exc:
            return self._reject(request, BuildStatus.FAILED, str(exc))

        return _result_from_import(self.name, request, imported, parameter_source="pre-built system")


class CharmmGuiImportBackend(SystemBuilderBackend):
    """Import a completed CHARMM-GUI job by id.

    **Submission is not supported and is not attempted.**  CHARMM-GUI publishes login,
    status and download endpoints only; there is no documented way to create a job
    programmatically, and Polymer Builder is not mentioned in its API documentation at
    all.  Guessing an endpoint would be fabrication, so this backend declares the
    limitation and tells the operator what to do instead.
    """

    name = "charmm_gui_import"
    capabilities = frozenset({BuildCapability.IMPORT_ARCHIVE})

    def __init__(self, provider: Any = None) -> None:
        self.provider = provider

    def can_handle(self, request: SystemBuildRequest) -> bool:
        return bool(request.external_job_id or request.source_archive)

    def build(
        self,
        request: SystemBuildRequest,
        destination: str | Path,
        *,
        graph: ProvenanceGraph | None = None,
    ) -> SystemBuildResult:
        problems = request.validate()
        if problems:
            return self._reject(
                request, BuildStatus.REQUIRES_EXPERT_DECISION, "; ".join(problems)
            )

        destination = Path(destination)

        if request.source_archive:
            archive = Path(request.source_archive)
        elif request.external_job_id:
            if self.provider is None:
                return self._reject(
                    request, BuildStatus.REQUIRES_INPUT,
                    "a CHARMM-GUI job id was given but no authenticated provider is configured",
                    "configure CHARMM_GUI_EMAIL/PASSWORD or CHARMM_GUI_TOKEN",
                )
            destination.mkdir(parents=True, exist_ok=True)
            archive = destination / f"charmm_gui_{request.external_job_id}.tgz"
            download = self.provider.download_job(request.external_job_id, archive)
            if not download.ok:
                return self._reject(
                    request, BuildStatus.FAILED,
                    f"could not download CHARMM-GUI job {request.external_job_id}: {download.error}",
                )
        else:
            return self._reject(
                request,
                BuildStatus.UNSUPPORTED,
                "CHARMM-GUI publishes no job-submission endpoint, so a system cannot be "
                "created programmatically",
                "build the system in the CHARMM-GUI web interface",
                "then supply its job id as external_job_id, or the downloaded archive as source_archive",
            )

        try:
            imported = import_archive(
                archive, destination / "system", graph=graph,
                source={"backend": self.name, "provider": "charmm_gui",
                        "job_id": request.external_job_id},
            )
        except PolymerEngineError as exc:
            return self._reject(request, BuildStatus.FAILED, str(exc))

        return _result_from_import(
            self.name, request, imported,
            parameter_source=f"CHARMM-GUI job {request.external_job_id or archive.name}",
        )


def _result_from_import(
    backend: str, request: SystemBuildRequest, imported: ImportedSystem, *, parameter_source: str
) -> SystemBuildResult:
    result = SystemBuildResult(
        status=BuildStatus.IMPORTED if imported.report.promotable else BuildStatus.FAILED,
        backend=backend,
        request=request,
        manifest=imported.manifest,
        validation=imported.report,
        artifact_id=imported.artifact_id,
        force_field=request.force_field,
        water_model=request.water_model,
        parameter_source=parameter_source,
        provenance={
            "backend": backend,
            "request_fingerprint": request.fingerprint(),
            **(imported.manifest.source if imported.manifest else {}),
        },
    )
    if not imported.report.promotable:
        result.diagnostics = [
            g.message for g in imported.report.gates if g.status.blocks_promotion
        ]
        result.required_actions = ["fix the system so it passes structural validation"]
    return result


class SystemBuilder:
    """Chooses a backend for a request and delegates to it."""

    def __init__(self, backends: Sequence[SystemBuilderBackend] | None = None) -> None:
        self._backends: list[SystemBuilderBackend] = list(
            backends if backends is not None else [LocalDirectoryBackend()]
        )

    def register(self, backend: SystemBuilderBackend) -> SystemBuilderBackend:
        """Add a backend without the orchestrator needing to know about it."""
        self._backends.append(backend)
        return backend

    def backends(self) -> list[str]:
        return [b.name for b in self._backends]

    def capabilities(self) -> dict[str, list[str]]:
        return {b.name: sorted(c.value for c in b.capabilities) for b in self._backends}

    def get(self, name: str) -> SystemBuilderBackend:
        for backend in self._backends:
            if backend.name == name:
                return backend
        raise PolymerEngineError("Unknown builder backend", backend=name, known=self.backends())

    def select(self, request: SystemBuildRequest) -> SystemBuilderBackend | None:
        for backend in self._backends:
            if backend.can_handle(request):
                return backend
        return None

    def build(
        self,
        request: SystemBuildRequest,
        destination: str | Path,
        *,
        backend: str | None = None,
        graph: ProvenanceGraph | None = None,
    ) -> SystemBuildResult:
        """Build a system, choosing a backend if one was not named."""
        chosen = self.get(backend) if backend else self.select(request)
        if chosen is None:
            return SystemBuildResult(
                status=BuildStatus.REQUIRES_INPUT,
                backend="none",
                request=request,
                diagnostics=[
                    "no registered backend can satisfy this request; "
                    "the engine does not build polymer systems from scratch"
                ],
                required_actions=[
                    "build the system with an external tool (CHARMM-GUI, Packmol, OpenFF, ...)",
                    "then supply it as source_directory, source_archive or external_job_id",
                ],
            )
        result = chosen.build(request, destination, graph=graph)
        logger.info(
            "System build via %s: %s (usable=%s)", chosen.name, result.status.value, result.usable
        )
        return result


def default_builder(provider: Any = None) -> SystemBuilder:
    """The builder the engine ships with.

    Both backends *import* systems.  Building a polymer system from scratch --
    chain construction, packing, solvation, parameter assignment -- is not implemented,
    and a half-correct builder is worse than none.
    """
    return SystemBuilder([LocalDirectoryBackend(), CharmmGuiImportBackend(provider)])


__all__ = [
    "BuildCapability",
    "BuildStatus",
    "CharmmGuiImportBackend",
    "LocalDirectoryBackend",
    "SystemBuildRequest",
    "SystemBuildResult",
    "SystemBuilder",
    "SystemBuilderBackend",
    "default_builder",
]
