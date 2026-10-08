"""Saving enough to debug a browser failure, and nothing that could leak a credential.

Every artifact written here passes through :func:`sanitise`, and the allow-list approach
is deliberate: a deny-list of secret-looking keys fails the moment a new field appears,
whereas dropping everything not explicitly permitted fails safe. Cookies, storage and
authorisation headers are not filtered out -- they are never collected, because nothing
in this subsystem reads them.

Screenshots are the one artifact that cannot be sanitised programmatically, so they are
taken only when the page holds no password control, and the manifest records why one is
absent when it is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polymer_engine.core.logging import get_logger, redact

logger = get_logger("browser.diagnostics")

DIAGNOSTIC_DIR = Path("campaign/charmm_gui/diagnostics")

#: Keys permitted in a saved diagnostic. Anything else is dropped without inspection.
ALLOWED_KEYS: frozenset[str] = frozenset({
    "state", "reason", "detail", "error", "url", "title", "key", "semantic",
    "field_key", "requested", "readback", "agrees", "problem", "locator",
    "strategy", "value", "name", "exact", "match_count", "tried", "markers",
    "chars_expected", "chars_received", "controls", "candidates", "n_fields",
    "safe_to_submit", "outcomes", "links_seen", "fingerprint", "job_id",
    "system_type", "polymer_id", "catalog_version", "schema",
})

#: Form-control keys whose *value* must never be recorded, even though the control's
#: presence is useful. Matched on the control's own metadata, not on content.
SECRET_CONTROL_TYPES = frozenset({"password"})

#: Patterns scrubbed from any free text before it is written.
BEARER_RE = re.compile(r"(bearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
COOKIE_RE = re.compile(r"((?:session|token|jwt|csrf)[=:]\s*)[A-Za-z0-9._~+/=-]{8,}",
                       re.IGNORECASE)


def scrub_text(text: str) -> str:
    """Remove credential-shaped substrings, then apply the log redactor.

    The redactor knows the actual password, because :func:`credentials.from_environment`
    registered it. The regexes catch tokens the redactor has never seen.
    """
    cleaned = BEARER_RE.sub(r"\1[redacted]", text)
    cleaned = COOKIE_RE.sub(r"\1[redacted]", cleaned)
    return redact(cleaned)


def sanitise(payload: Any, *, depth: int = 0) -> Any:
    """Keep only allow-listed keys, and scrub every string that survives."""
    if depth > 8:
        return "[truncated]"
    if isinstance(payload, dict):
        clean: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in ALLOWED_KEYS:
                continue
            if key == "controls" and isinstance(value, list):
                clean[key] = [_sanitise_control(c) for c in value if isinstance(c, dict)]
                continue
            clean[key] = sanitise(value, depth=depth + 1)
        return clean
    if isinstance(payload, list):
        return [sanitise(item, depth=depth + 1) for item in payload[:200]]
    if isinstance(payload, str):
        return scrub_text(payload)
    return payload


def _sanitise_control(control: dict[str, Any]) -> dict[str, Any]:
    """Describe a form control without recording what it holds."""
    kind = (control.get("type") or "").lower()
    clean = {
        "tag": control.get("tag"), "type": kind, "name": control.get("name"),
        "id": control.get("id"), "required": control.get("required"),
        "visible": control.get("visible"),
        "label": scrub_text(str(control.get("label") or ""))[:120] or None,
    }
    if kind not in SECRET_CONTROL_TYPES and control.get("options"):
        clean["options"] = [
            {"value": str(o.get("value", ""))[:80],
             "text": scrub_text(str(o.get("text", "")))[:120]}
            for o in control["options"][:200]
        ]
    return clean


@dataclass
class Diagnostic:
    """One saved failure, and the files that describe it."""

    name: str
    state: str
    reason: str
    url: str = ""
    directory: Path | None = None
    files: list[str] = field(default_factory=list)
    screenshot_skipped: str = ""
    captured_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "state": self.state, "reason": self.reason,
                "url": self.url, "captured_at": self.captured_at,
                "files": list(self.files),
                "screenshot_skipped": self.screenshot_skipped,
                "directory": str(self.directory) if self.directory else None}


def capture(
    session: Any, *, name: str, state: str, reason: str,
    payload: dict[str, Any] | None = None,
    directory: str | Path = DIAGNOSTIC_DIR,
    include_snapshot: bool = True,
) -> Diagnostic:
    """Save a sanitised record of a browser failure.

    Never raises: a diagnostic that fails to be written must not replace the failure it
    was documenting.
    """
    root = Path(directory) / f"{name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    diagnostic = Diagnostic(name=name, state=state, reason=scrub_text(reason),
                            directory=root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        try:
            diagnostic.url = session.current_url()
        except Exception:  # noqa: BLE001 - a dead page still deserves a record
            diagnostic.url = "(unavailable)"

        record: dict[str, Any] = {
            "name": name, "state": state, "reason": diagnostic.reason,
            "url": diagnostic.url, "captured_at": diagnostic.captured_at,
            **sanitise(payload or {}),
        }
        fields_response: dict[str, Any] = {}
        try:
            fields_response = session.driver.send("form_fields")
        except Exception:  # noqa: BLE001
            fields_response = {}
        if fields_response.get("ok"):
            record["controls"] = sanitise({"controls": fields_response["controls"]})["controls"]
        _write(root / "diagnostic.json", json.dumps(record, indent=2) + "\n", diagnostic)

        if include_snapshot:
            try:
                snapshot = session.snapshot(300000)
                if snapshot.get("ok"):
                    _write(root / "page.html", scrub_text(snapshot["html"]), diagnostic)
            except Exception as exc:  # noqa: BLE001
                logger.debug("snapshot unavailable: %s", exc)

        has_password = any(
            (c.get("type") or "") == "password"
            for c in fields_response.get("controls", [])
        )
        if has_password:
            # A screenshot of a login form shows a masked field, but it also shows the
            # email and any error text -- and masking is the page's choice, not ours.
            diagnostic.screenshot_skipped = (
                "the page carries a password control; a screenshot is not taken, "
                "because its contents cannot be sanitised after the fact"
            )
        else:
            try:
                shot = session.driver.send("screenshot", path=str(root / "page.png"))
                if shot.get("ok"):
                    diagnostic.files.append("page.png")
            except Exception as exc:  # noqa: BLE001
                logger.debug("screenshot unavailable: %s", exc)

        _write(root / "diagnostic_manifest.json",
               json.dumps(diagnostic.as_dict(), indent=2) + "\n", diagnostic,
               record_name=False)
        logger.warning("browser diagnostic saved to %s (%s)", root, state)
    except Exception as exc:
        logger.exception("could not save a diagnostic for %s: %s", name, exc)  # noqa: TRY401
    return diagnostic


def _write(path: Path, text: str, diagnostic: Diagnostic, *, record_name: bool = True) -> None:
    path.write_text(text)
    if record_name:
        diagnostic.files.append(path.name)


__all__ = [
    "ALLOWED_KEYS", "DIAGNOSTIC_DIR", "Diagnostic", "capture", "sanitise", "scrub_text",
]
