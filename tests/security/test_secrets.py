"""Credentials must never reach a log, a manifest, a provenance record, or a traceback."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from polymer_engine.core.config import Secret, load_config
from polymer_engine.core.errors import ConfigError
from polymer_engine.core.logging import configure_logging, get_logger, redact, register_secret
from polymer_engine.providers.charmm_gui import CHARMMGUIProvider
from polymer_engine.providers.http import HttpClient
from polymer_engine.providers.testing import FixtureTransport, StubResponse

TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.s3cr3t-signature-value"
PASSWORD = "correct-horse-battery-staple"
API_KEY = "mp-live-key-9f8e7d6c5b4a"


class TestSecretType:
    def test_repr_is_masked(self) -> None:
        assert "hunter2" not in repr(Secret("hunter2"))
        assert Secret.MASK in repr(Secret("hunter2"))

    def test_str_is_masked(self) -> None:
        assert "hunter2" not in str(Secret("hunter2"))

    def test_fstring_interpolation_is_masked(self) -> None:
        secret = Secret("hunter2")
        assert "hunter2" not in f"token={secret}"

    def test_reveal_returns_the_value(self) -> None:
        assert Secret("hunter2").reveal() == "hunter2"

    def test_empty_secret_is_falsy(self) -> None:
        assert not Secret(None)
        assert not Secret("")
        assert Secret("x")


class TestConfigRedaction:
    def test_redacted_config_hides_every_credential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHARMM_GUI_PASSWORD", PASSWORD)
        monkeypatch.setenv("CHARMM_GUI_TOKEN", TOKEN)
        monkeypatch.setenv("MP_API_KEY", API_KEY)
        monkeypatch.setenv("CHARMM_GUI_EMAIL", "scientist@lab.example.org")
        config = load_config(discover=False)
        rendered = json.dumps(config.redacted())
        for secret in (PASSWORD, TOKEN, API_KEY):
            assert secret not in rendered
        assert "configured" in rendered
        assert "scientist@lab.example.org" not in rendered

    def test_config_repr_does_not_dump_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHARMM_GUI_TOKEN", TOKEN)
        assert TOKEN not in repr(load_config(discover=False))

    def test_group_readable_token_file_is_refused(self, tmp_path: Path) -> None:
        token_file = tmp_path / "token"
        token_file.write_text(TOKEN)
        token_file.chmod(0o644)
        with pytest.raises(ConfigError, match="group- or world-accessible"):
            load_config(
                discover=False, use_env=False,
                overrides={"credentials": {"charmm_gui_token_file": str(token_file)}},
            )

    def test_private_token_file_is_read(self, tmp_path: Path) -> None:
        token_file = tmp_path / "token"
        token_file.write_text(TOKEN)
        token_file.chmod(0o600)
        config = load_config(
            discover=False, use_env=False,
            overrides={"credentials": {"charmm_gui_token_file": str(token_file)}},
        )
        assert config.credentials.charmm_gui_token.reveal() == TOKEN


class TestLogRedaction:
    def test_registered_secrets_are_scrubbed(self) -> None:
        buffer = io.StringIO()
        configure_logging(stream=buffer)
        register_secret(TOKEN)
        get_logger("t").info("Authorization: Bearer %s", TOKEN)
        assert TOKEN not in buffer.getvalue()
        assert "REDACTED" in buffer.getvalue()

    def test_unknown_bearer_tokens_are_scrubbed(self) -> None:
        assert "abcdefgh123456" not in redact("Authorization: Bearer abcdefgh123456")

    def test_jwt_shaped_strings_are_scrubbed(self) -> None:
        assert "s3cr3t" not in redact(f"got {TOKEN} back")

    def test_key_value_credentials_are_scrubbed(self) -> None:
        for text in ('api_key=abcd1234', '"token": "abcd1234"', "password = abcd1234"):
            assert "abcd1234" not in redact(text)

    def test_ordinary_messages_survive(self) -> None:
        message = "Completed replica_01 in 12.3s with density 1050 kg/m^3"
        assert redact(message) == message

    def test_structured_event_payloads_are_scrubbed(self) -> None:
        from polymer_engine.core.logging import log_event

        buffer = io.StringIO()
        configure_logging(stream=buffer)
        register_secret(API_KEY)
        log_event(get_logger("t"), "provider_call", {"headers": {"X-API-KEY": API_KEY}})
        assert API_KEY not in buffer.getvalue()


class TestProviderSecrets:
    def test_charmm_gui_token_never_appears_in_a_result(self) -> None:
        transport = FixtureTransport()
        transport.add_json("/api/login", {"token": TOKEN})
        provider = CHARMMGUIProvider(
            HttpClient(transport=transport, sleeper=lambda s: None),
            email="a@b.c", password=PASSWORD,
        )
        result = provider.login()
        rendered = json.dumps(result.as_dict(), default=str)
        assert TOKEN not in rendered
        assert PASSWORD not in rendered

    def test_password_is_not_echoed_on_failure(self) -> None:
        transport = FixtureTransport()
        transport.add("/api/login", StubResponse.json({"error": "bad"}, status=401))
        provider = CHARMMGUIProvider(
            HttpClient(transport=transport, sleeper=lambda s: None),
            email="a@b.c", password=PASSWORD,
        )
        result = provider.login()
        assert result.ok is False
        assert PASSWORD not in json.dumps(result.as_dict(), default=str)

    def test_error_bodies_containing_credentials_are_scrubbed(self) -> None:
        from polymer_engine.core.errors import HttpStatusError

        transport = FixtureTransport()
        transport.add("/api/x", StubResponse.text(f'{{"token": "{TOKEN}"}}', status=400))
        client = HttpClient(transport=transport, max_retries=0, sleeper=lambda s: None)
        with pytest.raises(HttpStatusError) as excinfo:
            client.get_json("https://charmm-gui.org/api/x")
        assert TOKEN not in excinfo.value.body

    def test_cache_keys_do_not_embed_credentials(self, tmp_path: Path) -> None:
        from polymer_engine.providers.http import HttpRequest

        request = HttpRequest(url="https://x.test/j", headers={"Authorization": f"Bearer {TOKEN}"})
        assert TOKEN not in request.cache_key()

    def test_cached_response_files_contain_no_authorization_header(self, tmp_path: Path) -> None:
        from polymer_engine.providers.http import ResponseCache

        transport = FixtureTransport()
        transport.add_json("example.org/x", {"ok": True})
        client = HttpClient(
            transport=transport,
            cache=ResponseCache(tmp_path / "cache", ttl_s=60, clock=lambda: 0.0),
            sleeper=lambda s: None,
        )
        client.get_json("https://example.org/x", headers={"Authorization": f"Bearer {TOKEN}"})
        for path in (tmp_path / "cache").rglob("*.json"):
            assert TOKEN not in path.read_text()


class TestRepositoryHygiene:
    def test_no_credentials_are_committed_in_source(self) -> None:
        """A literal secret in source is the one leak no runtime guard can fix."""
        import re

        root = Path(__file__).resolve().parents[2]
        patterns = [
            re.compile(r"CHARMM_GUI_PASSWORD\s*=\s*[\"'][^\"']{4,}"),
            re.compile(r"\bapi[_-]?key\s*=\s*[\"'][A-Za-z0-9]{16,}[\"']", re.IGNORECASE),
            re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
        ]
        offenders: list[str] = []
        for path in (root / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in patterns:
                if pattern.search(text):
                    offenders.append(f"{path}: {pattern.pattern}")
        assert offenders == [], f"possible committed credentials: {offenders}"

    def test_shipped_config_contains_no_secrets(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for name in ("configs/default.yaml", "polymer.yaml"):
            path = root / name
            if not path.exists():
                continue
            text = path.read_text().lower()
            for forbidden in ("password:", "api_key:", "token:"):
                assert forbidden not in text or "file" in text, f"{name} appears to hold a credential"


class TestBrowserCredentials:
    """The browser subsystem's credential rules, enforced by structure (§15, §16)."""

    def test_no_source_file_contains_a_password_cli_flag(self) -> None:
        """There must be no way to pass a password as an argument."""
        import re
        from pathlib import Path

        banned = re.compile(r"--password|password\s*:\s*str\s*=\s*typer|argv.*password",
                            re.IGNORECASE)
        for path in Path("src/polymer_engine").rglob("*.py"):
            text = path.read_text()
            assert not banned.search(text), f"{path} appears to accept a password argument"

    def test_the_worker_protocol_has_no_command_that_carries_a_password(self) -> None:
        """``type_secret`` names an environment variable; nothing carries a value."""
        import ast
        import inspect
        from pathlib import Path

        source = Path("scripts/browser_worker.py").read_text()
        tree = ast.parse(source)
        func = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "type_secret")
        body = ast.get_source_segment(source, func) or ""

        # The credential is read from this process's own environment, by variable name.
        assert 'os.environ.get(var)' in body
        assert 'req["env_var"]' in body
        # It is never read out of the request payload itself.
        assert 'req["password"]' not in body
        assert 'req.get("password"' not in body
        assert inspect.cleandoc(ast.get_docstring(func) or "")

    def test_a_secret_field_is_never_echoed_back(self) -> None:
        """``read_value`` refuses password-like keys instead of returning them."""
        import ast
        from pathlib import Path

        source = Path("scripts/browser_worker.py").read_text()
        tree = ast.parse(source)
        func = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "read_value")
        body = ast.get_source_segment(source, func) or ""
        assert "_is_secret_key(key)" in body
        assert "REFUSED" in body

    def test_no_clipboard_api_is_used_anywhere(self) -> None:
        """§16: the password path must be keyboard events, never a paste.

        Checks executable code only. The docstrings deliberately *name* these APIs to
        record that they are not used, and a naive text scan would flag its own
        documentation.
        """
        import ast
        from pathlib import Path

        banned = ("navigator.clipboard", "Control+V", "ControlOrMeta+v",
                  "insertFromPaste", "clipboardData")
        paths = [Path("scripts/browser_worker.py"),
                 *Path("src/polymer_engine/browser").rglob("*.py")]
        for path in paths:
            tree = ast.parse(path.read_text())
            # Every string literal that is not a docstring, plus every attribute chain.
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
                and node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            }
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and id(node) not in docstrings):
                    for marker in banned:
                        assert marker not in node.value, (
                            f"{path}: clipboard use found in code: {marker}")

    def test_the_secret_typing_path_uses_keyboard_type(self) -> None:
        from pathlib import Path

        worker = Path("scripts/browser_worker.py").read_text()
        typing_fn = worker.split("def _keyboard_type")[1].split("\n    def ")[0]
        assert "keyboard.type(" in typing_fn
        assert ".fill(" not in typing_fn

    def test_credentials_never_serialise(self) -> None:
        import json

        from polymer_engine.browser.credentials import Credentials
        from polymer_engine.core.config import Secret

        creds = Credentials(email="a@b.c", password=Secret("top-secret-value"))
        for rendered in (repr(creds), str(creds), json.dumps(creds.as_dict())):
            assert "top-secret-value" not in rendered

    def test_a_diagnostic_drops_unknown_keys_rather_than_inspecting_them(self) -> None:
        """An allow-list fails safe when a new field appears; a deny-list does not."""
        from polymer_engine.browser.diagnostics import sanitise

        cleaned = sanitise({"url": "u", "some_future_field": "sk-live-abcdef123456"})
        assert "some_future_field" not in cleaned
        assert cleaned == {"url": "u"}
