"""Structured logging with credential redaction.

Two things matter here:

* A token must never reach a log sink.  :class:`RedactingFilter` scrubs known
  secret values and common credential-shaped patterns from every record before it
  is formatted.
* Autonomous decisions must be reconstructable.  :func:`log_event` emits a
  machine-readable event that the observability layer also persists to the store.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from polymer_engine.core.config import EngineConfig, Secret

LOGGER_NAME = "polymer_engine"

# Patterns that look like credentials regardless of which value we happen to know.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|token|password|secret|passwd)\"?\s*[:=]\s*\"?)([^\s,\"'}]{4,})", re.IGNORECASE),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\b"),  # JWT
)

_REDACTED = "***REDACTED***"


class RedactingFilter(logging.Filter):
    """Scrub secrets from log records.

    ``literals`` holds exact secret values discovered from configuration; the
    regular expressions catch credential-shaped text we were never told about.
    """

    def __init__(self, literals: Iterable[str] = ()) -> None:
        super().__init__()
        self._literals = {value for value in literals if value and len(value) >= 4}

    def add_secret(self, value: str | Secret | None) -> None:
        raw = value.reveal() if isinstance(value, Secret) else value
        if raw and len(raw) >= 4:
            self._literals.add(raw)

    def scrub(self, text: str) -> str:
        for literal in self._literals:
            if literal in text:
                text = text.replace(literal, _REDACTED)
        text = _PATTERNS[0].sub(rf"\1{_REDACTED}", text)
        text = _PATTERNS[1].sub(rf"\1{_REDACTED}", text)
        text = _PATTERNS[2].sub(_REDACTED, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.scrub(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._scrub_value(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._scrub_value(v) for v in record.args)
        payload = getattr(record, "event_payload", None)
        if isinstance(payload, dict):
            record.event_payload = _scrub_mapping(payload, self.scrub)
        return True

    def _scrub_value(self, value: Any) -> Any:
        return self.scrub(value) if isinstance(value, str) else value


def _scrub_mapping(payload: dict[str, Any], scrub: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            out[key] = _scrub_mapping(value, scrub)
        elif isinstance(value, list):
            out[key] = [_scrub_mapping(v, scrub) if isinstance(v, dict) else (scrub(v) if isinstance(v, str) else v) for v in value]
        elif isinstance(value, str):
            out[key] = scrub(value)
        else:
            out[key] = value
    return out


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
        extra = getattr(record, "event_payload", None)
        if extra:
            payload["payload"] = extra
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_redactor = RedactingFilter()


def configure_logging(config: EngineConfig | None = None, *, stream: Any = None) -> logging.Logger:
    """Install handlers on the package logger.  Idempotent."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False

    level = config.logging.level if config else "INFO"
    fmt = config.logging.format if config else "text"
    logger.setLevel(getattr(logging, level))

    if config is not None:
        for secret in (
            config.credentials.charmm_gui_password,
            config.credentials.charmm_gui_token,
            config.credentials.materials_project_api_key,
            config.credentials.anthropic_api_key,
        ):
            _redactor.add_secret(secret)

    formatter: logging.Formatter = (
        JsonFormatter() if fmt == "json" else logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(formatter)
    handler.addFilter(_redactor)
    logger.addHandler(handler)

    if config is not None and config.logging.file:
        path = Path(config.logging.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(_redactor)
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")
    if not logging.getLogger(LOGGER_NAME).handlers:
        configure_logging()
    return logger


def register_secret(value: str | Secret | None) -> None:
    """Teach the redactor about a secret discovered at runtime."""
    _redactor.add_secret(value)


def redact(text: str) -> str:
    """Scrub a string using the active redaction rules."""
    return _redactor.scrub(text)


def log_event(logger: logging.Logger, event: str, payload: dict[str, Any], *, level: int = logging.INFO) -> None:
    """Emit a structured, machine-readable event."""
    logger.log(level, "%s", event, extra={"event": event, "event_payload": payload})


__all__ = [
    "JsonFormatter",
    "RedactingFilter",
    "configure_logging",
    "get_logger",
    "log_event",
    "redact",
    "register_secret",
]
