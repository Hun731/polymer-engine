"""CHARMM-GUI backend: an explicitly human-in-the-loop acquisition route.

CHARMM-GUI publishes ``/api/login``, ``/api/check_status`` and ``/api/download`` and no
submission endpoint.  The build itself is therefore done by a person in the web
interface, and this backend is honest about that at every level: :meth:`capabilities`
reports ``human_in_the_loop``, :meth:`assess` sets ``requires_human_step``, and
:meth:`parameterize` returns ``REQUIRES_EXPERT_REVIEW`` with a written brief rather than
pretending to have built anything.

No endpoint is guessed and no form is reverse-engineered.  The architecture is ready for
a documented submission API if one appears: only :meth:`parameterize` would change.

What the backend does automate is everything around the manual step -- deriving the
specification, importing the finished job, validating the system, and reading the CGenFF
penalties that say how far the parameters were assigned by analogy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polymer_engine.core.errors import PolymerEngineError
from polymer_engine.core.logging import get_logger
from polymer_engine.core.models import Determination
from polymer_engine.core.provenance import sha256_file
from polymer_engine.parameterization.backend import ForceFieldBackend
from polymer_engine.parameterization.capability import CapabilityState
from polymer_engine.parameterization.charges import analyse_charges, charge_gates
from polymer_engine.parameterization.completeness import analyse_topology, completeness_gates
from polymer_engine.parameterization.models import (
    ForceFieldAssessment,
    ParameterizationRequest,
    ParameterizationResult,
    ParameterizationState,
    ParameterizationValidation,
    QMPriority,
)
from polymer_engine.parameterization.quality import detect_sensitive_terms, overall_priority
from polymer_engine.simulation.cgenff import (
    find_stream_files,
    parse_stream_file,
    penalty_gates,
)
from polymer_engine.simulation.charmm_gui_spec import spec_from_record, write_spec

logger = get_logger("parameterization.charmm_gui")

#: Files a CHARMM-GUI archive is expected to carry, by role.
ARTIFACT_ROLES: dict[str, tuple[str, ...]] = {
    "coordinates": (".gro", ".pdb", ".crd"),
    "topology": (".top", ".psf"),
    "parameters": (".prm", ".par"),
    "stream": (".str",),
    "include": (".itp", ".rtf"),
    "index": (".ndx",),
    "mdp": (".mdp",),
}


class CharmmGuiBackend(ForceFieldBackend):
    name = "charmm_gui"
    force_field = "CHARMM36 + CGenFF"
    human_in_the_loop = True
    requires_credentials = True

    def __init__(self, provider: Any = None) -> None:
        self.provider = provider

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name, "force_field": self.force_field, "version": None,
            # Available as a *route*: the import half needs nothing installed locally.
            "available": True,
            "human_in_the_loop": True,
            "requires_credentials": self.provider is not None,
            "submission": "manual — CHARMM-GUI publishes no job-submission endpoint",
            "documented_endpoints": ["/api/login", "/api/check_status", "/api/download"],
            "provides": ["build_specification", "job_import", "cgenff_penalties",
                         "system_validation"],
            "chemistry": "broad organic coverage via CGenFF analogy assignment",
        }

    def assess(self, polymer: Any) -> ForceFieldAssessment:
        # CHARMM-GUI covers far more chemistry than the local table, but nothing is
        # parameterized until a person has actually run the job, so the ceiling here is
        # STRUCTURE_SUPPORTED rather than PARAMETERIZATION_AVAILABLE.
        return self._assessment(
            polymer, CapabilityState.STRUCTURE_SUPPORTED,
            ("CGenFF assigns parameters by analogy for broad organic chemistry, but the "
             "build is a manual step: CHARMM-GUI publishes no submission endpoint, so "
             "nothing exists until a person runs the job"),
            estimated_cost="minutes of human time, then a download",
            qm_priority=QMPriority.MEDIUM,
        )

    def parameterize(self, request: ParameterizationRequest) -> ParameterizationResult:
        """Import a finished job, or produce the brief for one that has not been run.

        With a job id or archive this imports and validates. Without one it does not
        pretend: it writes the specification and returns REQUIRES_EXPERT_REVIEW.
        """
        problems = request.problems()
        if problems:
            return self._unavailable(request, "; ".join(problems))

        workdir = Path(request.workdir or "parameterization") / request.polymer_id
        workdir.mkdir(parents=True, exist_ok=True)

        if not request.external_job_id and not request.source_archive:
            return self._await_human(request, workdir)
        return self._import_job(request, workdir)

    def _await_human(
        self, request: ParameterizationRequest, workdir: Path
    ) -> ParameterizationResult:
        from polymer_engine.polymer.records import build_record

        record = build_record(
            name=request.polymer_name, repeat_unit_smiles=request.repeat_unit_smiles,
            properties={}, source="parameterization",
        )
        spec = spec_from_record(
            record, force_field=request.force_field or self.force_field,
            degree_of_polymerization=request.degree_of_polymerization,
            n_chains=request.n_chains, temperature_k=request.temperature_k,
            pressure_bar=request.pressure_bar,
            target_density_kg_m3=request.target_density_kg_m3, notes=request.notes,
        )
        payload, brief = write_spec(spec, workdir)
        return ParameterizationResult(
            backend=self.name, request=request,
            state=ParameterizationState.REQUIRES_EXPERT_REVIEW,
            force_field=self.force_field,
            parameter_source="CHARMM-GUI Polymer Builder (manual)",
            diagnostics=["no job id or archive supplied; a person must run the build"],
            required_actions=[
                f"build the system at {spec.as_dict()['module_url']} using {brief}",
                "then re-run with --job-id <ID> or --archive <PATH>",
            ],
            artifacts={str(payload): sha256_file(payload), str(brief): sha256_file(brief)},
            provenance={"specification": spec.as_dict(),
                        "submission": "manual; no documented endpoint exists"},
        )

    def _import_job(
        self, request: ParameterizationRequest, workdir: Path
    ) -> ParameterizationResult:
        from polymer_engine.simulation.builder import SystemBuildRequest, default_builder

        build = default_builder(self.provider).build(
            SystemBuildRequest(
                polymer_id=request.polymer_id,
                force_field=request.force_field or self.force_field,
                external_job_id=request.external_job_id,
                source_archive=request.source_archive,
                degree_of_polymerization=request.degree_of_polymerization,
                n_chains=request.n_chains,
            ),
            workdir,
        )
        if not build.usable:
            return ParameterizationResult(
                backend=self.name, request=request,
                state=ParameterizationState.FAILED, force_field=self.force_field,
                diagnostics=list(build.diagnostics) or [f"import returned {build.status.value}"],
                required_actions=list(build.required_actions),
                provenance={"build_status": build.status.value},
            )

        artifacts, roles = self._catalogue(workdir)
        topology = roles.get("topology", [None])[0]
        coordinates = roles.get("coordinates", [None])[0]
        net_charge = None
        if topology and str(topology).endswith(".top"):
            try:
                net_charge = analyse_charges(topology).system_charge
            except PolymerEngineError:
                net_charge = None

        return ParameterizationResult(
            backend=self.name, request=request,
            state=ParameterizationState.PARAMETERIZED,
            force_field=self.force_field,
            force_field_version=build.provenance.get("force_field_version"),
            parameter_source=f"CHARMM-GUI job {request.external_job_id or request.source_archive}",
            topology_path=str(topology) if topology else None,
            coordinate_path=str(coordinates) if coordinates else None,
            parameter_files=[str(p) for p in roles.get("parameters", [])
                             + roles.get("stream", [])],
            net_charge=net_charge,
            artifacts=artifacts,
            provenance={
                "build_status": build.status.value,
                "roles": {k: [str(p) for p in v] for k, v in roles.items()},
                "job_id": request.external_job_id,
                "archive": request.source_archive,
                "submission": "manual; imported through the documented download API",
            },
        )

    @staticmethod
    def _catalogue(root: Path) -> tuple[dict[str, str], dict[str, list[Path]]]:
        """Hash every file and sort it by role, so the manifest is complete."""
        artifacts: dict[str, str] = {}
        roles: dict[str, list[Path]] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                artifacts[str(path)] = sha256_file(path)
            except OSError:
                continue
            suffix = path.suffix.lower()
            for role, suffixes in ARTIFACT_ROLES.items():
                if suffix in suffixes:
                    roles.setdefault(role, []).append(path)
                    break
        return artifacts, roles

    def validate(self, result: ParameterizationResult) -> ParameterizationValidation:
        validation = ParameterizationValidation(
            backend=self.name, polymer_id=result.request.polymer_id,
            property_class=result.request.property_class,
            state=result.state,
        )
        if result.state is not ParameterizationState.PARAMETERIZED:
            validation.diagnostics.append(
                f"nothing to validate: parameterization is {result.state.value}"
            )
            return validation

        if result.topology_path and str(result.topology_path).endswith(".top"):
            completeness = analyse_topology(result.topology_path)
            validation.completeness = completeness_gates(completeness)
            charges = analyse_charges(result.topology_path)
            validation.charges = charge_gates(charges)
            validation.metrics["completeness"] = completeness.as_dict()
            validation.metrics["charges"] = charges.as_dict()

        # CGenFF penalties: the number that says how far the analogy reached.
        worst: Any = None
        for candidate in result.parameter_files:
            try:
                report = parse_stream_file(candidate)
            except PolymerEngineError:
                continue
            if worst is None or report.max_penalty > worst.max_penalty:
                worst = report
        if worst is None and result.topology_path:
            for found in find_stream_files(Path(result.topology_path).parent):
                try:
                    report = parse_stream_file(found)
                except PolymerEngineError:
                    continue
                if worst is None or report.max_penalty > worst.max_penalty:
                    worst = report

        if worst is not None:
            validation.penalties = penalty_gates(worst)
            validation.metrics["penalties"] = worst.as_dict()
            terms = detect_sensitive_terms(worst)
            validation.qm_priority = overall_priority(terms)
            validation.metrics["sensitive_terms"] = [t.as_dict() for t in terms]
        else:
            validation.diagnostics.append(
                "no CGenFF stream file found; parameter provenance is incomplete"
            )
            validation.qm_priority = QMPriority.HIGH

        if validation.promotable:
            validation.state = ParameterizationState.PARAMETER_COMPLETENESS_VALIDATED
            validation.determination = Determination.KNOWN
        else:
            validation.state = ParameterizationState.INCONCLUSIVE
            validation.determination = Determination.REQUIRES_VALIDATION
        return validation


__all__ = ["ARTIFACT_ROLES", "CharmmGuiBackend"]
