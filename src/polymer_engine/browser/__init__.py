"""Browser-driven acquisition of CHARMM-GUI Polymer Builder systems.

The subsystem is split so that the parts which can be tested without an account are
separate from the part that cannot:

``credentials``  reading a password from the environment and never letting go of it
``driver``       the process boundary to Playwright in an isolated environment
``session``      launch, login by real keystrokes, navigate, snapshot
``discovery``    reading the live page into a catalogue and a form schema
``catalog``      what Polymer Builder offers, fingerprinted and diffable
``matching``     resolving a requested polymer to exactly one catalogue entry
``workflows``    fill, verify, and only then submit
``queue``        one build at a time, and never the same build twice
``diagnostics``  saving enough to debug a failure and nothing that could leak
``acquisition``  the whole route, ending in the same validation every backend uses

Only ``acquisition`` and ``session`` need a live site. Everything else is exercised
against a real browser driving local fixtures, which tests the decisions rather than
the website.
"""

from polymer_engine.browser.acquisition import (
    BuildResult,
    CharmmGuiAcquisition,
    SystemValidationResult,
)
from polymer_engine.browser.catalog import Catalog, CatalogDiff, MonomerEntry, diff
from polymer_engine.browser.credentials import Credentials, from_environment, live_test_enabled
from polymer_engine.browser.driver import PageDriver, WorkerDriver
from polymer_engine.browser.matching import MonomerMatch, resolve_monomer
from polymer_engine.browser.queue import BuildEntry, BuildQueue
from polymer_engine.browser.selectors import FieldSpec, FormSchema, Locator
from polymer_engine.browser.session import ActionResult, Session
from polymer_engine.browser.states import BrowserState, JobState
from polymer_engine.browser.workflows import BuilderForm, FillReport, Submission

__all__ = [
    "ActionResult",
    "BrowserState",
    "BuildEntry",
    "BuildQueue",
    "BuildResult",
    "BuilderForm",
    "Catalog",
    "CatalogDiff",
    "CharmmGuiAcquisition",
    "Credentials",
    "FieldSpec",
    "FillReport",
    "FormSchema",
    "JobState",
    "Locator",
    "MonomerEntry",
    "MonomerMatch",
    "PageDriver",
    "Session",
    "Submission",
    "SystemValidationResult",
    "WorkerDriver",
    "diff",
    "from_environment",
    "live_test_enabled",
    "resolve_monomer",
]
