"""How a semantic field is located on a page, without ever guessing.

Two kinds of knowledge are kept strictly apart here, and the distinction is the whole
point of the module.

**Universal web structure** may be encoded as a default. A login form has an input of
type ``email`` (or an accessible name containing "email") and an input of type
``password``. That is not knowledge about CHARMM-GUI; it is knowledge about HTML, it is
enforced by browsers and assistive technology, and it is stable.

**Anything specific to the Polymer Builder form must be discovered from the live page**
and stored in ``data/charmm_gui/builder_form_schema.json``. Transcribing a field label
from a tutorial or a screenshot into source would go stale silently, and a stale
selector that still matches *something* is worse than one that matches nothing: it fills
the wrong field and submits a different polymer than the one requested.

So the only Polymer Builder selectors in this file are the empty set. When a workflow
needs one and the schema does not supply it, the answer is
:attr:`~polymer_engine.browser.states.BrowserState.UI_SCHEMA_MISMATCH`, not a guess.

Locators are expressed as ordered strategies. Each is a role/label/attribute
description that Playwright can resolve; **screen coordinates are never used**, because
they encode a window size rather than a meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Strategy = Literal["label", "role", "placeholder", "css", "test_id", "text"]
#: What kind of control a field is expected to be. Checked after a locator matches, so
#: that a locator resolving to the wrong control type is caught rather than typed into.
ControlKind = Literal["text", "number", "select", "checkbox", "radio", "button", "any"]

#: Strategies in preference order. Accessible name first, because it is the thing a
#: person reads off the page and the thing least likely to change for cosmetic reasons;
#: raw CSS last, because it is the most coupled to markup.
STRATEGY_PREFERENCE: tuple[Strategy, ...] = (
    "test_id", "label", "role", "placeholder", "text", "css",
)


@dataclass(frozen=True)
class Locator:
    """One way to find one element."""

    strategy: Strategy
    value: str
    #: For ``role``: the accessible name to disambiguate within that role.
    name: str | None = None
    exact: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "value": self.value,
                "name": self.name, "exact": self.exact}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Locator:
        return cls(strategy=data["strategy"], value=data["value"],
                   name=data.get("name"), exact=bool(data.get("exact", False)))


@dataclass(frozen=True)
class FieldSpec:
    """A semantic field and every way we know to locate it.

    ``required`` means the workflow cannot proceed without it. A required field with no
    matching locator is a ``UI_SCHEMA_MISMATCH``; an optional one is simply absent.
    """

    key: str
    description: str
    locators: tuple[Locator, ...]
    required: bool = True
    #: What kind of control this is expected to be, checked after matching so that a
    #: locator resolving to the wrong control type is caught rather than typed into.
    control: ControlKind = "any"
    #: Try the locators exactly as written instead of sorting by strategy.
    #:
    #: :data:`STRATEGY_PREFERENCE` is the right default for a *discovered* schema, whose
    #: locators are generated mechanically and are all equally trustworthy. It is wrong
    #: for a hand-written one, where the author knows that a structural match beats a
    #: match on a word: "Login" appears on navigation links as often as on submit
    #: buttons, and preferring the accessible name would click the wrong one.
    strict_order: bool = False

    def ordered(self) -> tuple[Locator, ...]:
        if self.strict_order:
            return tuple(self.locators)
        return tuple(sorted(
            self.locators,
            key=lambda loc: STRATEGY_PREFERENCE.index(loc.strategy)
            if loc.strategy in STRATEGY_PREFERENCE else len(STRATEGY_PREFERENCE),
        ))

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "description": self.description,
                "required": self.required, "control": self.control,
                "strict_order": self.strict_order,
                "locators": [loc.as_dict() for loc in self.ordered()]}


@dataclass
class FormSchema:
    """A set of semantic fields for one page, and where it came from.

    ``discovered`` distinguishes a schema read off a live page from one supplied as a
    structural default. Only a discovered schema may drive a Polymer Builder submission;
    see :meth:`require`.
    """

    name: str
    fields: dict[str, FieldSpec] = field(default_factory=dict)
    discovered: bool = False
    source_url: str | None = None
    discovered_at: str | None = None

    def add(self, spec: FieldSpec) -> None:
        self.fields[spec.key] = spec

    def get(self, key: str) -> FieldSpec | None:
        return self.fields.get(key)

    def require(self, key: str) -> FieldSpec:
        """Return a field or explain, precisely, why the automation must stop."""
        spec = self.fields.get(key)
        if spec is None:
            known = ", ".join(sorted(self.fields)) or "(none)"
            raise KeyError(
                f"the {self.name!r} schema has no field {key!r}; known fields: {known}. "
                f"This field must be discovered from the live page, not assumed"
            )
        return spec

    @property
    def missing_required(self) -> list[str]:
        return sorted(k for k, spec in self.fields.items()
                      if spec.required and not spec.locators)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "discovered": self.discovered,
            "source_url": self.source_url, "discovered_at": self.discovered_at,
            "n_fields": len(self.fields),
            "missing_required": self.missing_required,
            "fields": {k: spec.as_dict() for k, spec in sorted(self.fields.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FormSchema:
        schema = cls(name=data["name"], discovered=bool(data.get("discovered", False)),
                     source_url=data.get("source_url"),
                     discovered_at=data.get("discovered_at"))
        for key, raw in (data.get("fields") or {}).items():
            schema.add(FieldSpec(
                key=key, description=raw.get("description", ""),
                locators=tuple(Locator.from_dict(loc) for loc in raw.get("locators", [])),
                required=bool(raw.get("required", True)),
                control=raw.get("control", "any"),
                strict_order=bool(raw.get("strict_order", False)),
            ))
        return schema


# ---------------------------------------------------------------------------
# Universal structure only.  Nothing below is CHARMM-GUI-specific.
# ---------------------------------------------------------------------------

#: The form that contains a password input *is* the login form. Scoping every control
#: to it is what stops a locator matching something outside it that merely reads
#: "Login" -- a navigation link, a menu toggle, a header button.
#:
#: This is not a guess about any particular site. ``<form>`` grouping and
#: ``input[type=password]`` are platform guarantees, the same class of fact as the
#: password input itself, and browsers submit a form from the controls inside it.
LOGIN_FORM = "form:has(input[type='password'])"


def login_schema() -> FormSchema:
    """A generic login form, located structurally rather than by wording.

    ``input[type=password]`` is the one element on the web whose meaning is guaranteed
    by the platform: browsers mask it, password managers target it, and no site uses it
    for anything else. Encoding that is not the same as encoding a site's markup.

    The submit control gets the same treatment, and it has to. The real CHARMM-GUI
    sign-in page carries a ``<button onclick="location.href=...">Login</button>``
    *outside* the form, while the form's own submit is an ``<input type="submit">``
    labelled "Submit". Matching a button by the word "login" clicks the navigation
    control, which reloads the page and discards everything typed -- producing a login
    that fails with correct credentials and a form that is still present afterwards.

    So: structural submit controls inside the login form first, accessible names last,
    and ``strict_order`` so that ordering is honoured rather than re-sorted.
    """
    schema = FormSchema(name="login")
    schema.add(FieldSpec(
        key="email", description="account email address", control="text",
        strict_order=True,
        locators=(
            # Scoped to the login form first, so a search box named "email" elsewhere
            # on the page cannot win.
            Locator("css", f"{LOGIN_FORM} input[type='email']"),
            Locator("css", f"{LOGIN_FORM} input[name='email' i]"),
            Locator("css", f"{LOGIN_FORM} input[id='email' i]"),
            Locator("css", f"{LOGIN_FORM} input[name='username' i]"),
            Locator("css", f"{LOGIN_FORM} input[type='text']"),
            Locator("css", "input[type='email']"),
            Locator("css", "input[name='email']"),
            Locator("role", "textbox", name="email"),
        ),
    ))
    schema.add(FieldSpec(
        key="password", description="account password", control="text",
        strict_order=True,
        locators=(
            Locator("css", f"{LOGIN_FORM} input[type='password']"),
            Locator("css", "input[type='password']"),
            Locator("css", "input[name='password']"),
        ),
    ))
    schema.add(FieldSpec(
        key="submit", description="submit the login form", control="button",
        strict_order=True,
        locators=(
            # Structural: the browser submits a form from a submit control inside it.
            Locator("css", f"{LOGIN_FORM} input[type='submit']"),
            Locator("css", f"{LOGIN_FORM} button[type='submit']"),
            Locator("css", f"{LOGIN_FORM} button:not([type='button'])"),
            # Unscoped structural fallbacks, for a form we could not identify.
            Locator("css", "input[type='submit']"),
            Locator("css", "button[type='submit']"),
            # Name-based, last and deliberately so: these are the ones that match
            # navigation controls.
            Locator("role", "button", name="sign in"),
            Locator("role", "button", name="log in"),
        ),
    ))
    return schema


#: Controls that indicate a human-verification challenge. Their presence is *reported*,
#: never solved: a CAPTCHA is an access control, and defeating one is out of scope by
#: policy regardless of whether it would be technically possible.
HUMAN_VERIFICATION_MARKERS: tuple[str, ...] = (
    "iframe[src*='recaptcha']",
    "iframe[src*='hcaptcha']",
    "iframe[title*='captcha' i]",
    "div.g-recaptcha",
    "div.h-captcha",
    "iframe[src*='turnstile']",
    "[data-sitekey]",
)

#: Text that, when it appears on a page, means a second authentication factor is being
#: requested. Matched case-insensitively against visible text only.
MFA_TEXT_MARKERS: tuple[str, ...] = (
    "two-factor", "two factor", "2fa", "verification code",
    "one-time passcode", "one time password", "authenticator app",
    "multi-factor", "security code",
)


def builder_schema_placeholder() -> FormSchema:
    """The Polymer Builder schema before discovery: deliberately empty.

    There is no honest default. Every field name, option value and control type in the
    Polymer Builder form has to come from the live page, and until it does, any workflow
    that needs one must stop with ``UI_SCHEMA_MISMATCH``.
    """
    return FormSchema(name="polymer_builder", discovered=False)


__all__ = [
    "HUMAN_VERIFICATION_MARKERS", "LOGIN_FORM", "MFA_TEXT_MARKERS",
    "STRATEGY_PREFERENCE",
    "ControlKind", "FieldSpec", "FormSchema", "Locator", "Strategy",
    "builder_schema_placeholder", "login_schema",
]
