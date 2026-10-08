"""Filling the Polymer Builder form, checking it, and only then submitting.

The ordering here is the safety property. Between "the form has been filled" and "the
build has been requested" sits :func:`verify_before_submit`, which reads every control
back off the page and compares it to the normalised request. If any field disagrees --
or if any required field could not be located at all -- **nothing is submitted**.

That check is not defensive padding. Filling a form is an open-loop operation: a select
can silently reject a value, a numeric input can clamp, a JavaScript handler can rewrite
what was typed, and a changed layout can put the right value in the wrong box. Each of
those produces a job that builds a *different polymer* than the one requested, and the
resulting system would look entirely valid downstream -- correct topology, sensible
density, clean simulation -- while answering a question nobody asked.

Reading the page back is the only thing that distinguishes those cases from success.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from polymer_engine.browser.catalog import Catalog
from polymer_engine.browser.matching import resolve_monomer, resolve_option
from polymer_engine.browser.selectors import FormSchema
from polymer_engine.browser.session import Session
from polymer_engine.browser.states import BrowserState
from polymer_engine.core.logging import get_logger
from polymer_engine.simulation.charmm_gui_spec import PolymerBuilderSpec, SystemType

logger = get_logger("browser.workflows")

#: Semantic fields a build needs, per system type. A field listed here that the
#: discovered schema cannot supply is a ``UI_SCHEMA_MISMATCH``, not a default.
REQUIRED_SEMANTICS: dict[SystemType, tuple[str, ...]] = {
    SystemType.SINGLE_CHAIN: ("monomer", "degree_of_polymerization"),
    SystemType.MELT: ("monomer", "degree_of_polymerization", "n_chains"),
    SystemType.SOLUTION: ("monomer", "degree_of_polymerization", "n_chains", "solvent"),
}

#: Patterns that a job identifier is *allowed* to look like. Deliberately permissive
#: about content and strict about ambiguity: if a page yields several distinct
#: candidates, none is chosen. The engine never fabricates a job id, and never picks one
#: out of a set.
JOB_ID_PATTERNS: tuple[str, ...] = (
    r"\bjobid[=:\s]+([A-Za-z0-9_-]{4,40})\b",
    r"\bjob\s*id[=:\s]+([A-Za-z0-9_-]{4,40})\b",
    r"\bJob\s*ID\s*[:=]\s*([A-Za-z0-9_-]{4,40})\b",
)


@dataclass
class FieldOutcome:
    """What one field was asked to hold and what it actually holds."""

    semantic: str
    field_key: str | None
    requested: Any
    written: bool = False
    readback: Any = None
    agrees: bool = False
    problem: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"semantic": self.semantic, "field_key": self.field_key,
                "requested": self.requested, "written": self.written,
                "readback": self.readback, "agrees": self.agrees,
                "problem": self.problem}


@dataclass
class FillReport:
    """Every field the workflow touched, and whether the page agrees about it."""

    state: BrowserState
    outcomes: list[FieldOutcome] = field(default_factory=list)
    reason: str = ""
    url: str = ""

    @property
    def disagreements(self) -> list[FieldOutcome]:
        return [o for o in self.outcomes if o.written and not o.agrees]

    @property
    def unwritten(self) -> list[FieldOutcome]:
        return [o for o in self.outcomes if not o.written]

    @property
    def safe_to_submit(self) -> bool:
        """Every field written, every field read back, every read-back agreeing."""
        return (self.state.ok and bool(self.outcomes)
                and not self.disagreements and not self.unwritten)

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "reason": self.reason, "url": self.url,
                "safe_to_submit": self.safe_to_submit,
                "n_fields": len(self.outcomes),
                "n_disagreements": len(self.disagreements),
                "outcomes": [o.as_dict() for o in self.outcomes]}


@dataclass
class BuilderForm:
    """A discovered schema plus the semantic mapping a person has confirmed.

    The mapping is kept separate from the schema because it is a different kind of
    claim. The schema says "this page has a control named ``nchain``"; the mapping says
    "``nchain`` is the number of chains". The first is observed, the second is
    interpreted, and only the first can be re-derived automatically.
    """

    schema: FormSchema
    semantics: dict[str, list[str]] = field(default_factory=dict)

    def field_for(self, semantic: str) -> str | None:
        keys = self.semantics.get(semantic) or []
        # Exactly one candidate, or the mapping is not usable for automation.
        return keys[0] if len(keys) == 1 else None

    def ambiguous(self) -> dict[str, list[str]]:
        return {k: v for k, v in self.semantics.items() if len(v) > 1}

    def missing(self, system_type: SystemType) -> list[str]:
        needed = REQUIRED_SEMANTICS.get(system_type, ())
        return [semantic for semantic in needed if self.field_for(semantic) is None]


def _normalised_request(spec: PolymerBuilderSpec, monomer_value: str) -> dict[str, Any]:
    """The values that must end up on the page, in the page's own terms."""
    values: dict[str, Any] = {
        "monomer": monomer_value,
        "degree_of_polymerization": spec.degree_of_polymerization,
    }
    if spec.system_type.n_chains_meaningful:
        values["n_chains"] = spec.n_chains
    if spec.tacticity:
        values["tacticity"] = spec.tacticity
    if spec.solvent or spec.water_model:
        values["solvent"] = spec.solvent or spec.water_model
    if spec.sequence:
        values["sequence"] = spec.sequence
    return values


def fill_builder_form(
    session: Session, spec: PolymerBuilderSpec, form: BuilderForm, catalog: Catalog,
) -> FillReport:
    """Resolve the request against the catalogue and enter it into the page."""
    problems = spec.composition_problems()
    if problems:
        return FillReport(BrowserState.REFUSED,
                          reason="the request contradicts itself: " + "; ".join(problems))

    if not form.schema.discovered:
        return FillReport(
            BrowserState.UI_SCHEMA_MISMATCH,
            reason="the Polymer Builder schema has not been discovered from a live page; "
                   "no field can be filled from an assumed layout",
        )
    ambiguous = form.ambiguous()
    if ambiguous:
        return FillReport(
            BrowserState.REQUIRES_HUMAN_REVIEW,
            reason=("the semantic mapping is ambiguous and must be confirmed by a "
                    "person: " + "; ".join(f"{k} -> {v}" for k, v in ambiguous.items())),
        )
    absent = form.missing(spec.system_type)
    if absent:
        return FillReport(
            BrowserState.UI_SCHEMA_MISMATCH,
            reason=(f"a {spec.system_type.value} build needs "
                    f"{', '.join(absent)}, and the discovered form provides no control "
                    f"for {'them' if len(absent) > 1 else 'it'}"),
        )

    match = resolve_monomer(catalog, spec.name)
    if not match.resolved:
        return FillReport(match.state, reason=match.reason,
                          outcomes=[FieldOutcome("monomer", form.field_for("monomer"),
                                                 spec.name, problem=match.reason)])
    assert match.entry is not None
    logger.info("monomer %r resolved to catalogue value %r (%s)",
                spec.name, match.entry.value, match.matched_on)

    if spec.tacticity:
        chosen, why = resolve_option(catalog.tacticity_options, spec.tacticity,
                                     what="tacticity")
        if chosen is None:
            return FillReport(BrowserState.STRUCTURE_MISMATCH, reason=why)

    requested = _normalised_request(spec, match.entry.value)
    report = FillReport(BrowserState.OK, url=session.current_url())

    for semantic, value in requested.items():
        key = form.field_for(semantic)
        outcome = FieldOutcome(semantic=semantic, field_key=key, requested=value)
        if key is None:
            outcome.problem = f"no control is mapped to {semantic!r}"
            report.outcomes.append(outcome)
            continue
        spec_field = form.schema.get(key)
        if spec_field is None:
            outcome.problem = f"field {key!r} vanished from the schema"
            report.outcomes.append(outcome)
            continue
        locators = [loc.as_dict() for loc in spec_field.ordered()]
        command = "select" if spec_field.control == "select" else "type"
        payload = ({"value": str(value)} if command == "select"
                   else {"text": str(value),
                         "delay_ms": session.credentials.typing_delay_ms})
        response = session.driver.send(command, key=key, locators=locators, **payload)
        outcome.written = bool(response.get("ok"))
        if not outcome.written:
            outcome.problem = response.get("error", f"could not set {semantic}")
        report.outcomes.append(outcome)

    if any(o.problem for o in report.outcomes):
        report.state = BrowserState.UI_SCHEMA_MISMATCH
        report.reason = "; ".join(o.problem for o in report.outcomes if o.problem)
    return report


def verify_before_submit(
    session: Session, spec: PolymerBuilderSpec, form: BuilderForm,
    report: FillReport, catalog: Catalog,
) -> FillReport:
    """Read every field back off the page and compare it to the request (§29).

    Comparison is by *value*, with numeric fields compared numerically so that ``"30"``
    and ``30`` agree while ``"30"`` and ``"3"`` do not. Anything that cannot be read
    back counts as a disagreement: an unreadable field is an unverified one, and an
    unverified field is not submitted.
    """
    match = resolve_monomer(catalog, spec.name)
    expected = (_normalised_request(spec, match.entry.value)
                if match.resolved and match.entry else {})

    for outcome in report.outcomes:
        if not outcome.written or outcome.field_key is None:
            continue
        spec_field = form.schema.get(outcome.field_key)
        if spec_field is None:
            outcome.problem = "field vanished between filling and verification"
            continue
        response = session.driver.send(
            "read_value", key=outcome.field_key,
            locators=[loc.as_dict() for loc in spec_field.ordered()],
        )
        if not response.get("ok"):
            outcome.problem = response.get("error", "could not read the field back")
            outcome.agrees = False
            continue
        outcome.readback = response.get("value")
        want = expected.get(outcome.semantic, outcome.requested)
        outcome.agrees = _values_agree(want, outcome.readback)
        if not outcome.agrees:
            outcome.problem = (f"the page holds {outcome.readback!r} but "
                               f"{want!r} was requested")

    if report.disagreements:
        report.state = BrowserState.STRUCTURE_MISMATCH
        report.reason = ("the form does not hold what was requested: "
                         + "; ".join(f"{o.semantic}: {o.problem}"
                                     for o in report.disagreements))
        logger.error("refusing to submit: %s", report.reason)
    return report


def _values_agree(want: Any, got: Any) -> bool:
    if got is None:
        return False
    want_text, got_text = str(want).strip(), str(got).strip()
    if want_text == got_text:
        return True
    try:
        return abs(float(want_text) - float(got_text)) < 1e-9
    except ValueError:
        return want_text.lower() == got_text.lower()


@dataclass
class Submission:
    """The record of one actual build request."""

    state: BrowserState
    job_id: str | None = None
    request_fingerprint: str = ""
    url: str = ""
    reason: str = ""
    candidates: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "job_id": self.job_id,
                "request_fingerprint": self.request_fingerprint, "url": self.url,
                "reason": self.reason, "candidates": list(self.candidates)}


def submit_build(
    session: Session, spec: PolymerBuilderSpec, form: BuilderForm,
    report: FillReport, *, submit_key: str,
) -> Submission:
    """Click the build control -- but only if verification passed.

    The guard is first and unconditional. A caller that has not verified, or whose
    verification failed, gets a refusal rather than a submission.
    """
    fingerprint = spec.fingerprint()
    if not report.safe_to_submit:
        return Submission(
            BrowserState.REFUSED, request_fingerprint=fingerprint,
            reason=("not submitted: " + (report.reason or "the form was not verified "
                    "against the request")),
        )
    spec_field = form.schema.get(submit_key)
    if spec_field is None:
        return Submission(
            BrowserState.UI_SCHEMA_MISMATCH, request_fingerprint=fingerprint,
            reason=f"no control named {submit_key!r} on the discovered form",
        )
    logger.info("submitting Polymer Builder job for %s (fingerprint %s)",
                spec.name, fingerprint[:16])
    response = session.driver.send(
        "click", key=submit_key,
        locators=[loc.as_dict() for loc in spec_field.ordered()],
    )
    if not response.get("ok"):
        return Submission(BrowserState.SUBMISSION_FAILED,
                          request_fingerprint=fingerprint,
                          reason=response.get("error", "the build control did not respond"))

    blocked, markers = session.human_verification_present()
    if blocked:
        return Submission(
            BrowserState.HUMAN_INTERVENTION_REQUIRED, request_fingerprint=fingerprint,
            url=session.current_url(),
            reason="a human-verification challenge appeared at submission",
            candidates=markers,
        )
    return capture_job_id(session, fingerprint)


def capture_job_id(session: Session, fingerprint: str) -> Submission:
    """Read the real job identifier off the page.  Never invent or choose one.

    A single unambiguous candidate is the job id. Several distinct candidates, or none,
    mean the workflow does not know the job id -- which is reported as such, because a
    wrong job id would later download somebody else's system and validate it as ours.
    """
    url = session.current_url()
    text = session.page_text(8000)
    found: list[str] = []
    for pattern in JOB_ID_PATTERNS:
        found += re.findall(pattern, f"{url}\n{text}", flags=re.IGNORECASE)
    unique = sorted(set(found))

    if len(unique) == 1:
        logger.info("captured CHARMM-GUI job id %s", unique[0])
        return Submission(BrowserState.OK, job_id=unique[0],
                          request_fingerprint=fingerprint, url=url,
                          reason="job id read from the submitted page")
    if len(unique) > 1:
        return Submission(
            BrowserState.REQUIRES_HUMAN_REVIEW, request_fingerprint=fingerprint,
            url=url, candidates=unique,
            reason=(f"the page presents {len(unique)} possible job identifiers; none is "
                    f"chosen, because monitoring the wrong one would download a "
                    f"different system"),
        )
    return Submission(
        BrowserState.SUBMISSION_FAILED, request_fingerprint=fingerprint, url=url,
        reason=("the build appears to have been submitted but no job identifier could "
                "be read from the page; the job may exist and must be found by a person "
                "before anything is resubmitted"),
    )


__all__ = [
    "JOB_ID_PATTERNS", "REQUIRED_SEMANTICS", "BuilderForm", "FieldOutcome",
    "FillReport", "Submission", "capture_job_id", "fill_builder_form",
    "submit_build", "verify_before_submit",
]
