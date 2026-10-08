"""A scripted page, so the workflow logic can be tested without an account.

The fake implements the same JSON protocol the real Playwright worker speaks, over an
in-memory model of a page. That makes it possible to test the things that matter most
and are hardest to trigger on demand -- a CAPTCHA appearing, a select silently rejecting
a value, a form that changed since discovery -- deterministically.

What these tests prove is the engine's *decisions*. They do not prove anything about
CHARMM-GUI, and no test in this directory claims to.
"""

from __future__ import annotations

import re
from typing import Any

import pytest


class FakePage:
    """An in-memory page: controls, text, and the rules that govern them."""

    def __init__(
        self,
        controls: list[dict[str, Any]] | None = None,
        *,
        url: str = "https://example.invalid/builder",
        text: str = "",
        links: list[dict[str, str]] | None = None,
        captcha: bool = False,
        #: Rewrites a typed/selected value, standing in for a form that clamps,
        #: truncates or rejects what it is given.
        rewrite: Any = None,
    ) -> None:
        self.controls = controls if controls is not None else []
        self.url = url
        self.text = text
        self.links = links or []
        self.captcha = captcha
        self.rewrite = rewrite
        self.values: dict[str, Any] = {}

    def find(self, locators: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Resolve a locator the way the real worker does: uniquely or not at all."""
        for locator in locators:
            matches = [c for c in self.controls if _matches(c, locator)]
            if len(matches) == 1:
                return matches[0]
        return None


def _matches(control: dict[str, Any], locator: dict[str, Any]) -> bool:
    strategy, value = locator["strategy"], locator["value"]
    if strategy == "css":
        if m := re.match(r"^(\w+)?#(.+)$", value):
            return control.get("id") == m.group(2).replace("\\", "")
        if m := re.match(r"^(\w+)?\[name='(.+)'\]$", value):
            return control.get("name") == m.group(2)
        if m := re.match(r"^input\[type='(.+)'\]$", value):
            return control.get("type") == m.group(1)
        return False
    if strategy == "label":
        return (control.get("label") or "") == value
    if strategy == "test_id":
        return control.get("test_id") == value
    if strategy == "role":
        role = {"button": ("button", "submit"), "textbox": ("text", "email")}.get(value, ())
        name = (locator.get("name") or "").lower()
        return (control.get("type") in role
                and (not name or name in (control.get("label") or "").lower()))
    return False


class FakeDriver:
    """Speaks the worker protocol against a sequence of :class:`FakePage` objects."""

    def __init__(self, pages: dict[str, FakePage], *, start: str) -> None:
        self.pages = pages
        self.current = start
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        #: Every command the login flow is forbidden from issuing.
        self.forbidden = {"paste", "clipboard", "fill", "set_value"}

    @property
    def page(self) -> FakePage:
        return self.pages[self.current]

    def send(self, command: str, **payload: Any) -> dict[str, Any]:
        self.calls.append((command, payload))
        assert command not in self.forbidden, f"{command} is not permitted"
        handler = getattr(self, f"_{command}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown command {command}"}
        return handler(payload)

    def close(self) -> None:
        self.closed = True

    # -- protocol -------------------------------------------------------
    def _launch(self, _p: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "browser_version": "fake/1.0"}

    def _goto(self, p: dict[str, Any]) -> dict[str, Any]:
        target = p["url"]
        for name, page in self.pages.items():
            if page.url == target or name == target or target.endswith(name):
                self.current = name
                return {"ok": True, "url": page.url, "status": 200, "title": name}
        return {"ok": False, "state": "NAVIGATION_FAILED", "error": f"no page at {target}"}

    def _url(self, _p: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "url": self.page.url, "title": self.current}

    def _text(self, p: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "text": self.page.text[: p.get("max_chars", 20000)]}

    def _links(self, p: dict[str, Any]) -> dict[str, Any]:
        needle = (p.get("contains") or "").lower()
        found = [link for link in self.page.links
                 if not needle or needle in link["text"].lower()]
        return {"ok": True, "links": found}

    def _form_fields(self, _p: dict[str, Any]) -> dict[str, Any]:
        controls = []
        for i, control in enumerate(self.page.controls):
            entry = {"index": i, "tag": control.get("tag", "input"),
                     "type": control.get("type"), "name": control.get("name"),
                     "id": control.get("id"), "test_id": control.get("test_id"),
                     "label": control.get("label"), "placeholder": None,
                     "required": control.get("required", False), "disabled": False,
                     "visible": control.get("visible", True),
                     "options": control.get("options"), "checked": None}
            controls.append(entry)
        return {"ok": True, "controls": controls, "url": self.page.url,
                "title": self.current}

    def _human_verification(self, p: dict[str, Any]) -> dict[str, Any]:
        """Mirrors the real worker: a CAPTCHA element *or* MFA text in the body."""
        markers = ["element:div.g-recaptcha"] if self.page.captcha else []
        body = self.page.text.lower()
        markers += [f"text:{phrase}" for phrase in p.get("mfa_markers", [])
                    if phrase in body]
        return {"ok": True, "detected": bool(markers), "markers": markers}

    def _type(self, p: dict[str, Any]) -> dict[str, Any]:
        return self._write(p, p["text"], echo=True)

    def _type_secret(self, p: dict[str, Any]) -> dict[str, Any]:
        import os

        value = os.environ.get(p["env_var"], "")
        if not value:
            return {"ok": False, "state": "CREDENTIALS_MISSING",
                    "error": f"{p['env_var']} is unset"}
        return self._write(p, value, echo=False)

    def _write(self, p: dict[str, Any], value: str, *, echo: bool) -> dict[str, Any]:
        control = self.page.find(p["locators"])
        if control is None:
            return {"ok": False, "state": "UI_SCHEMA_MISMATCH",
                    "error": f"no unique element for {p.get('key')}"}
        stored = value
        if self.page.rewrite is not None:
            stored = self.page.rewrite(p.get("key", ""), value)
        self.page.values[control.get("id") or control.get("name") or "?"] = stored
        result: dict[str, Any] = {"ok": len(str(stored)) == len(value),
                                  "chars_expected": len(value),
                                  "chars_received": len(str(stored)),
                                  "locator": p["locators"][0]}
        if not result["ok"]:
            result["state"] = "UI_SCHEMA_MISMATCH"
            result["error"] = (f"field {p.get('key')!r} received "
                               f"{len(str(stored))} characters but {len(value)} "
                               f"were typed")
        if echo:
            result["value"] = stored
        return result

    def _select(self, p: dict[str, Any]) -> dict[str, Any]:
        control = self.page.find(p["locators"])
        if control is None:
            return {"ok": False, "state": "UI_SCHEMA_MISMATCH",
                    "error": f"no unique element for {p.get('key')}"}
        wanted = p["value"]
        options = [o["value"] for o in (control.get("options") or [])]
        if wanted not in options:
            return {"ok": False, "state": "STRUCTURE_MISMATCH",
                    "error": f"{wanted!r} is not an option"}
        stored = wanted
        if self.page.rewrite is not None:
            stored = self.page.rewrite(p.get("key", ""), wanted)
        self.page.values[control.get("id") or control.get("name") or "?"] = stored
        return {"ok": True, "selected": [stored], "locator": p["locators"][0]}

    def _read_value(self, p: dict[str, Any]) -> dict[str, Any]:
        key = p.get("key", "")
        if "password" in key.lower():
            return {"ok": False, "state": "REFUSED",
                    "error": "secret fields are never echoed"}
        control = self.page.find(p["locators"])
        if control is None:
            return {"ok": False, "state": "UI_SCHEMA_MISMATCH", "error": "gone"}
        ident = control.get("id") or control.get("name") or "?"
        return {"ok": True, "value": self.page.values.get(ident),
                "locator": p["locators"][0]}

    def _click(self, p: dict[str, Any]) -> dict[str, Any]:
        control = self.page.find(p["locators"])
        if control is None:
            return {"ok": False, "state": "UI_SCHEMA_MISMATCH",
                    "error": f"no unique element for {p.get('key')}"}
        target = control.get("navigates_to")
        if target:
            self.current = target
        return {"ok": True, "url": self.page.url, "locator": p["locators"][0]}

    def _snapshot(self, p: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "html": "<html>[fake]</html>", "url": self.page.url,
                "title": self.current}

    def _screenshot(self, p: dict[str, Any]) -> dict[str, Any]:
        from pathlib import Path

        path = Path(p["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return {"ok": True, "path": str(path)}


EMAIL_CONTROL = {"tag": "input", "type": "email", "id": "email", "name": "email",
                 "label": "Email"}
PASSWORD_CONTROL = {"tag": "input", "type": "password", "id": "pw", "name": "password",
                    "label": "Password"}
SUBMIT_CONTROL = {"tag": "input", "type": "submit", "id": "go", "label": "Log in",
                  "navigates_to": "home"}


def login_page(**kwargs: Any) -> FakePage:
    return FakePage(
        [dict(EMAIL_CONTROL), dict(PASSWORD_CONTROL), dict(SUBMIT_CONTROL)],
        url="https://example.invalid/?doc=sign", text="Sign in", **kwargs,
    )


def builder_controls() -> list[dict[str, Any]]:
    return [
        {"tag": "select", "id": "monomer", "name": "monomer", "label": "Monomer",
         "options": [{"value": "", "text": "-- choose --"},
                     {"value": "PE", "text": "Polyethylene"},
                     {"value": "PLA", "text": "Poly(lactic acid)"}]},
        {"tag": "input", "type": "number", "id": "dp", "name": "dp",
         "label": "Degree of polymerization", "required": True},
        {"tag": "input", "type": "number", "id": "nchain", "name": "nchain",
         "label": "Number of chains"},
        {"tag": "select", "id": "tacticity", "name": "tacticity", "label": "Tacticity",
         "options": [{"value": "atactic", "text": "Atactic"},
                     {"value": "isotactic", "text": "Isotactic"}]},
        {"tag": "button", "type": "button", "id": "next", "label": "Build",
         "navigates_to": "submitted"},
    ]


@pytest.fixture
def pages() -> dict[str, FakePage]:
    return {
        "login": login_page(),
        "home": FakePage([], url="https://example.invalid/", text="Welcome",
                         links=[{"text": "Input Generator",
                                 "href": "https://example.invalid/gen"},
                                {"text": "Polymer Builder",
                                 "href": "https://example.invalid/builder"}]),
        "builder": FakePage(builder_controls(), url="https://example.invalid/builder",
                            text="Polymer Builder"),
        "submitted": FakePage([], url="https://example.invalid/done",
                              text="Job submitted. jobid=1234567"),
    }
