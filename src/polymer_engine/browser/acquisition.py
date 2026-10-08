"""The whole route: a polymer request in, a validated system and a qualification out.

This is the module §78 and §79 describe -- the one that makes CHARMM-GUI behave like a
pluggable backend rather than a website somebody visits. It runs the browser workflow to
get a *real* job id, monitors that job through the documented status API, downloads it
through the documented download API, and then hands the archive to exactly the same
import, completeness, charge, penalty and QM machinery that OpenFF and OPLS-AA results
go through. There is no separate, easier path for CHARMM-GUI systems.

Two boundaries are load-bearing:

* **The browser drives the visible interface; the API does status and download.** No
  Polymer Builder endpoint is invented, and the browser is never described as an API.
* **Acquisition is not qualification.** What comes back from a successful run is a
  *parameterized* system with a validation report attached. Whether it is qualified for
  a property class is decided later, by evidence this module does not produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polymer_engine.browser import diagnostics as diag
from polymer_engine.browser.catalog import Catalog
from polymer_engine.browser.discovery import (
    catalog_from_controls,
    discover_form_schema,
    find_polymer_builder,
    map_semantics,
)
from polymer_engine.browser.matching import resolve_monomer
from polymer_engine.browser.queue import BuildEntry, BuildQueue
from polymer_engine.browser.session import Session
from polymer_engine.browser.states import API_STATUS_TO_JOB_STATE, BrowserState, JobState
from polymer_engine.browser.workflows import (
    BuilderForm,
    Submission,
    fill_builder_form,
    submit_build,
    verify_before_submit,
)
from polymer_engine.core.logging import get_logger
from polymer_engine.simulation.charmm_gui_spec import PolymerBuilderSpec, SystemType

logger = get_logger("browser.acquisition")

ACQUISITION_ROOT = Path("campaign/charmm_gui")

#: Polling, bounded and backed off (§31, §54). A polymer build takes minutes; polling
#: every few seconds would be rude to a shared service and would learn nothing extra.
INITIAL_POLL_S = 30.0
MAX_POLL_S = 300.0
POLL_BACKOFF = 1.5


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class BuildResult:
    """What one acquisition attempt produced (§71)."""

    polymer_id: str
    request_fingerprint: str
    state: BrowserState
    job_state: JobState
    job_id: str | None = None
    source: str = "charmm_gui_browser"
    artifacts: dict[str, str] = field(default_factory=dict)
    force_field: str | None = None
    parameterization_state: str | None = None
    validation_state: str | None = None
    catalog_version: str | None = None
    reason: str = ""
    diagnostics: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    @property
    def acquired(self) -> bool:
        """A real archive was downloaded.  Says nothing about whether it is any good."""
        return self.job_state in {JobState.DOWNLOADED, JobState.VALIDATED}

    def as_dict(self) -> dict[str, Any]:
        return {
            "polymer_id": self.polymer_id,
            "request_fingerprint": self.request_fingerprint,
            "state": self.state.value, "job_state": self.job_state.value,
            "job_id": self.job_id, "source": self.source, "acquired": self.acquired,
            "artifacts": dict(self.artifacts), "force_field": self.force_field,
            "parameterization_state": self.parameterization_state,
            "validation_state": self.validation_state,
            "catalog_version": self.catalog_version, "reason": self.reason,
            "diagnostics": list(self.diagnostics),
            "provenance": dict(self.provenance), "created_at": self.created_at,
        }


@dataclass
class SystemValidationResult:
    """The verdict on an imported system (§72)."""

    coordinate_valid: bool = False
    topology_valid: bool = False
    parameter_complete: bool = False
    charges_valid: bool = False
    box_valid: bool = False
    provenance_valid: bool = False
    diagnostics: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return all((self.coordinate_valid, self.topology_valid, self.parameter_complete,
                    self.charges_valid, self.box_valid, self.provenance_valid))

    def as_dict(self) -> dict[str, Any]:
        return {
            "coordinate_valid": self.coordinate_valid,
            "topology_valid": self.topology_valid,
            "parameter_complete": self.parameter_complete,
            "charges_valid": self.charges_valid, "box_valid": self.box_valid,
            "provenance_valid": self.provenance_valid, "valid": self.valid,
            "diagnostics": list(self.diagnostics),
        }


class CharmmGuiAcquisition:
    """Runs the browser build, then the documented API, then local validation."""

    def __init__(
        self,
        *,
        session: Session | None = None,
        provider: Any = None,
        queue: BuildQueue | None = None,
        root: str | Path = ACQUISITION_ROOT,
        semantics: dict[str, list[str]] | None = None,
    ) -> None:
        self.session = session
        self.provider = provider
        self.queue = queue if queue is not None else BuildQueue()
        self.root = Path(root)
        #: A human-confirmed semantic mapping, if one exists. Without it, discovery's
        #: suggestions are used and any ambiguity stops the workflow.
        self.semantics = semantics

    # -- discovery ------------------------------------------------------
    def discover(self, session: Session) -> tuple[Catalog | None, BuilderForm | None, str]:
        """Log in if needed, reach Polymer Builder, and read the page."""
        if not session.authenticated:
            login = session.login()
            if not login.ok:
                return None, None, f"{login.state.value}: {login.detail}"
        reached = find_polymer_builder(session)
        if not reached.ok:
            diag.capture(session, name="polymer-builder-navigation",
                         state=reached.state.value, reason=reached.detail,
                         payload=reached.data, directory=self.root / "diagnostics")
            return None, None, f"{reached.state.value}: {reached.detail}"

        schema, controls = discover_form_schema(session)
        semantics = self.semantics or map_semantics(controls)
        # The inventory sees choices a form-control scan cannot: radio groups, checkbox
        # groups, datalists. Without it a page that lists monomers as anything but a
        # <select> reports zero, which reads as a claim about the site's chemistry.
        inventory = session.driver.send("page_inventory")
        # And the onclick-driven list, which is how this page actually offers monomers.
        handlers = session.driver.send("handler_choices", handler="set_monomer")
        catalog = catalog_from_controls(
            controls, source_url=session.current_url(), semantics=semantics,
            inventory=inventory if inventory.get("ok") else None,
            handler_choices=handlers if handlers.get("ok") else None,
        )
        catalog.write(Path("data/charmm_gui"))
        _write_schema(schema, semantics, Path("data/charmm_gui"))
        return catalog, BuilderForm(schema=schema, semantics=semantics), ""

    # -- the full route -------------------------------------------------
    def acquire(
        self, spec: PolymerBuilderSpec, *, rationale: str,
        submit_key: str, poll_timeout_s: float = 7200.0,
        parameterize: bool = True,
    ) -> BuildResult:
        """Build, monitor, download and validate one system.

        Every failure returns a :class:`BuildResult` carrying the state that says what to
        do next. Nothing here raises past the caller for an expected condition -- a
        missing monomer, a changed form, an unfinished job -- because each of those is a
        legitimate outcome the engine must record rather than an error in the code.
        """
        fingerprint = spec.fingerprint()
        result = BuildResult(polymer_id=spec.polymer_id,
                             request_fingerprint=fingerprint,
                             state=BrowserState.OK, job_state=JobState.QUEUED)

        duplicate = self.queue.may_submit(fingerprint)
        if not duplicate.may_submit:
            existing = duplicate.existing
            result.state = BrowserState.OK if duplicate.action in {"monitor", "reuse"} \
                else BrowserState.REQUIRES_HUMAN_REVIEW
            result.job_state = existing.state if existing else JobState.BLOCKED
            result.job_id = existing.job_id if existing else None
            result.reason = duplicate.reason
            logger.info("not submitting %s: %s", spec.name, duplicate.reason)
            if duplicate.action == "monitor" and result.job_id:
                return self._monitor_and_import(result, spec, poll_timeout_s, parameterize)
            return result

        entry = self.queue.add(BuildEntry(
            polymer_id=spec.polymer_id, polymer_name=spec.name,
            request_fingerprint=fingerprint, system_type=spec.system_type.value,
            rationale=rationale, spec=spec.as_dict(),
            force_field=spec.force_field,
        ))
        self.queue.save()

        session = self.session
        if session is None:
            result.state = BrowserState.NOT_LAUNCHED
            result.reason = "no browser session was supplied"
            return result

        catalog, form, problem = self.discover(session)
        if catalog is None or form is None:
            result.state = BrowserState.UI_SCHEMA_MISMATCH
            result.reason = problem
            self.queue.record_submission_failure(entry, result.state, problem)
            self.queue.save()
            return result
        result.catalog_version = catalog.version

        match = resolve_monomer(catalog, spec.name)
        if not match.resolved:
            result.state = match.state
            result.reason = match.reason
            result.diagnostics = [f"candidates: {[c['label'] for c in match.candidates]}"]
            self.queue.record_submission_failure(entry, match.state, match.reason)
            self.queue.save()
            return result

        report = fill_builder_form(session, spec, form, catalog)
        report = verify_before_submit(session, spec, form, report, catalog)
        if not report.safe_to_submit:
            captured = diag.capture(
                session, name=f"fill-{spec.polymer_id}", state=report.state.value,
                reason=report.reason, payload=report.as_dict(),
                directory=self.root / "diagnostics",
            )
            result.state = report.state
            result.reason = report.reason or "the form did not hold what was requested"
            result.diagnostics = [str(captured.directory)]
            entry.diagnostics.append(str(captured.directory))
            self.queue.record_submission_failure(entry, report.state, result.reason)
            self.queue.save()
            return result

        submission: Submission = submit_build(session, spec, form, report,
                                              submit_key=submit_key)
        if submission.state is not BrowserState.OK or not submission.job_id:
            captured = diag.capture(
                session, name=f"submit-{spec.polymer_id}", state=submission.state.value,
                reason=submission.reason, payload=submission.as_dict(),
                directory=self.root / "diagnostics",
            )
            result.state = submission.state
            result.reason = submission.reason
            result.diagnostics = [str(captured.directory)]
            entry.diagnostics.append(str(captured.directory))
            self.queue.record_submission_failure(entry, submission.state, submission.reason)
            self.queue.save()
            return result

        result.job_id = submission.job_id
        entry.job_id = submission.job_id
        entry.advance(JobState.SUBMITTED, f"submitted through the web interface at "
                                          f"{submission.url}")
        self.queue.save()
        return self._monitor_and_import(result, spec, poll_timeout_s, parameterize)

    # -- monitoring and import ------------------------------------------
    def _monitor_and_import(
        self, result: BuildResult, spec: PolymerBuilderSpec,
        poll_timeout_s: float, parameterize: bool,
    ) -> BuildResult:
        entry = self.queue.by_job(result.job_id or "") or self.queue.find(
            result.request_fingerprint)
        if self.provider is None:
            result.job_state = JobState.SUBMITTED
            result.reason = ("job submitted, but no CHARMM-GUI API provider is "
                             "configured to monitor or download it")
            return result

        status = self.provider.wait_for_job(
            result.job_id, poll_interval_s=INITIAL_POLL_S, timeout_s=poll_timeout_s
        )
        raw = (status.records[0].get("state") if status.records else None) or "unknown"
        mapped = API_STATUS_TO_JOB_STATE.get(str(raw), None)
        if mapped is None:
            result.job_state = JobState.RUNNING
            result.state = BrowserState.OK
            result.reason = (f"CHARMM-GUI reported status {raw!r}, which is not a "
                             f"documented value; the job is left running rather than "
                             f"read as finished")
            return result
        if mapped is JobState.ERROR:
            result.job_state = JobState.ERROR
            result.reason = status.error or "CHARMM-GUI reported the job as failed"
            if entry:
                entry.advance(JobState.ERROR, result.reason)
                self.queue.save()
            return result
        if mapped is not JobState.DOWNLOAD_PENDING:
            result.job_state = mapped
            result.reason = (f"the job is {mapped.value} after {poll_timeout_s:g} s; "
                             f"this is resumable, not a failure")
            if entry:
                entry.advance(mapped, result.reason)
                self.queue.save()
            return result

        archive_dir = self.root / "archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f"{result.job_id}.tgz"
        download = self.provider.download_job(result.job_id, archive)
        if not download.ok:
            result.job_state = JobState.DOWNLOAD_PENDING
            result.state = BrowserState.NETWORK_ERROR
            result.reason = download.error or "download failed"
            if entry:
                entry.advance(JobState.DOWNLOAD_PENDING, result.reason)
                self.queue.save()
            return result

        result.job_state = JobState.DOWNLOADED
        result.artifacts[str(archive)] = str(download.data.get("sha256", ""))
        result.provenance["download"] = dict(download.provenance)
        if entry:
            entry.archive_path = str(archive)
            entry.archive_sha256 = str(download.data.get("sha256", ""))
            entry.advance(JobState.DOWNLOADED,
                          f"downloaded through the documented API "
                          f"({download.data.get('size_bytes')} bytes)")
            self.queue.save()

        if parameterize:
            self._parameterize(result, spec, archive, entry)
        return result

    def _parameterize(
        self, result: BuildResult, spec: PolymerBuilderSpec, archive: Path,
        entry: BuildEntry | None,
    ) -> None:
        """Hand the archive to the same pipeline every other backend feeds (§79)."""
        from polymer_engine.parameterization.backends.charmm_gui import CharmmGuiBackend
        from polymer_engine.parameterization.models import (
            ParameterizationRequest,
            PropertyClass,
        )

        backend = CharmmGuiBackend(self.provider)
        request = ParameterizationRequest(
            polymer_id=spec.polymer_id, polymer_name=spec.name,
            repeat_unit_smiles=spec.repeat_unit_smiles,
            property_class=PropertyClass.BULK_DENSITY,
            degree_of_polymerization=spec.degree_of_polymerization,
            n_chains=spec.n_chains, temperature_k=spec.temperature_k,
            pressure_bar=spec.pressure_bar,
            target_density_kg_m3=spec.target_density_kg_m3,
            force_field=spec.force_field, external_job_id=result.job_id,
            source_archive=str(archive),
            workdir=str(self.root / "systems"),
        )
        parameterized = backend.parameterize(request)
        result.parameterization_state = parameterized.state.value
        result.force_field = parameterized.force_field
        result.artifacts.update(parameterized.artifacts)
        result.provenance["parameterization"] = parameterized.as_dict()

        validation = backend.validate(parameterized)
        result.validation_state = validation.determination.value
        result.provenance["validation"] = validation.as_dict()
        result.diagnostics += list(validation.diagnostics)

        if entry:
            entry.parameterization_state = parameterized.state.value
            entry.validation_state = validation.determination.value
            if validation.promotable:
                entry.advance(JobState.VALIDATED,
                              "imported, complete, charge-neutral and penalty-checked")
                result.job_state = JobState.VALIDATED
            else:
                entry.advance(JobState.FAILED_VALIDATION,
                              "; ".join(validation.diagnostics)[:200]
                              or "local validation did not promote this system")
                result.job_state = JobState.FAILED_VALIDATION
            self.queue.save()


    # -- session recovery (§23) -----------------------------------------
    def resume(self, *, poll_timeout_s: float = 7200.0) -> list[BuildResult]:
        """Pick up every in-flight job after a crash or an expired session.

        The asymmetry from the queue applies here too, and this is where it earns its
        keep. A job that was *submitted* is monitored, never resubmitted -- an expired
        cookie says nothing about whether CHARMM-GUI is still building. A job whose
        submission outcome was never established is left BLOCKED for a person, because
        "we lost the session before we saw a job id" is not evidence that no job exists.

        Re-authentication happens through the normal login flow, so the password is
        typed by keyboard exactly as it was the first time.
        """
        results: list[BuildResult] = []
        for entry in self.queue.entries:
            if not entry.state.in_flight:
                continue
            if not entry.job_id:
                # In flight with no id: submission may or may not have happened.
                entry.advance(
                    JobState.BLOCKED,
                    "the session ended before a job id was captured. A job may exist on "
                    "CHARMM-GUI; a person must check before anything is resubmitted",
                )
                results.append(BuildResult(
                    polymer_id=entry.polymer_id,
                    request_fingerprint=entry.request_fingerprint,
                    state=BrowserState.REQUIRES_HUMAN_REVIEW,
                    job_state=JobState.BLOCKED,
                    reason=entry.history[-1]["reason"],
                ))
                continue

            logger.info("resuming job %s for %s (%s)", entry.job_id, entry.polymer_name,
                        entry.state.value)
            result = BuildResult(
                polymer_id=entry.polymer_id,
                request_fingerprint=entry.request_fingerprint,
                state=BrowserState.OK, job_state=entry.state, job_id=entry.job_id,
            )
            spec = _spec_from_entry(entry)
            results.append(
                self._monitor_and_import(result, spec, poll_timeout_s, parameterize=True)
                if spec is not None else result
            )
        self.queue.save()
        return results

    def reauthenticate(self, session: Session) -> BrowserState:
        """Establish a fresh authenticated session without touching any job state."""
        session._authenticated = False
        outcome = session.login()
        logger.info("re-authentication: %s", outcome.state.value)
        return outcome.state


def _spec_from_entry(entry: BuildEntry) -> PolymerBuilderSpec | None:
    """Rebuild the specification a queue entry was created from.

    Returns None rather than a partially-reconstructed specification: resuming with a
    spec that differs from the one submitted would validate the returned system against
    the wrong request.
    """
    payload = entry.spec
    required = ("polymer_id", "name", "repeat_unit_smiles", "degree_of_polymerization",
                "n_chains", "force_field", "temperature_k", "pressure_bar")
    if not payload or any(key not in payload for key in required):
        logger.warning("queue entry %s has no complete specification; it can be "
                       "monitored but not re-validated", entry.request_fingerprint[:12])
        return None
    spec = PolymerBuilderSpec(
        polymer_id=payload["polymer_id"], name=payload["name"],
        repeat_unit_smiles=payload["repeat_unit_smiles"],
        degree_of_polymerization=int(payload["degree_of_polymerization"]),
        n_chains=int(payload["n_chains"]), force_field=payload["force_field"],
        temperature_k=float(payload["temperature_k"]),
        pressure_bar=float(payload["pressure_bar"]),
        target_density_kg_m3=payload.get("target_density_kg_m3"),
        tacticity=payload.get("tacticity"),
        system_type=SystemType(payload.get("system_type", "melt")),
    )
    if spec.fingerprint() != entry.request_fingerprint:
        # The specification on disk does not describe the job that was submitted.
        logger.error("queue entry %s does not round-trip: rebuilt fingerprint %s",
                     entry.request_fingerprint[:12], spec.fingerprint()[:12])
        return None
    return spec


def _write_schema(schema: Any, semantics: dict[str, list[str]], directory: Path) -> Path:
    """Persist the discovered form schema alongside the catalogue (§20, §76)."""
    import json

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "builder_form_schema.json"
    payload = {
        "schema": "charmm_gui.builder_form_schema/1",
        **schema.as_dict(),
        "semantic_hints": semantics,
        "note": (
            "semantic_hints map a scientific quantity to a discovered control. They are "
            "suggestions produced by matching label text and must be confirmed by a "
            "person before they drive a submission: a control whose label merely "
            "contains 'chains' is not necessarily the chain count."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


__all__ = [
    "ACQUISITION_ROOT", "INITIAL_POLL_S", "MAX_POLL_S", "POLL_BACKOFF",
    "BuildResult", "CharmmGuiAcquisition", "SystemValidationResult",
]
