"""The `dump` command must actually print what it claims to.

This exists because a patch to the printing code silently did not apply -- `str.replace`
returns the original string when the pattern does not match, and the pattern had been
reformatted by a linter between writing it and using it. Lint passed, types passed, and
the unit tests passed, because they all covered the *worker* fields rather than the
command that displays them. The result was two live runs that produced byte-identical
output while appearing to have new capabilities.

So these assert on the rendered output, which is the only thing that would have caught it.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import importlib.util
import socket
import threading
from pathlib import Path

import pytest

from polymer_engine.browser.driver import WorkerDriver

pytestmark = pytest.mark.skipif(not WorkerDriver.available(), reason="no .browserenv")

FIXTURES = Path(__file__).parent / "fixtures"
PASSWORD = "correct-horse"


def _live_module():
    spec = importlib.util.spec_from_file_location(
        "charmm_gui_live",
        Path(__file__).resolve().parents[2] / "scripts" / "charmm_gui_live.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def site() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(FIXTURES))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()


@pytest.fixture
def rendered(site, tmp_path, monkeypatch, capsys) -> str:
    """Run the real `dump` command against a fixture and return what it printed."""
    monkeypatch.setenv("CHARMM_GUI_EMAIL", "a@b.c")
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", PASSWORD)
    live = _live_module()

    base = site
    original = live.Session

    class Local(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["base_url"] = base
            super().__init__(*args, **kwargs)

        def login(self, *args, **kwargs):
            return super().login(url=f"{base}/login.html", settle_s=0.3)

    live.Session = Local
    code = live.dump(argparse.Namespace(
        url=f"{base}/builder_conditional.html", out=str(tmp_path),
        headed=False, probe_conditionals=False))
    assert code == 0
    return capsys.readouterr().out


def test_every_advertised_section_is_printed(rendered: str) -> None:
    for heading in ("choice sets", "buttons", "tables", "hidden/invisible control(s)"):
        assert heading in rendered, f"the dump never printed a {heading!r} section"


def test_hidden_controls_report_why_and_where(rendered: str) -> None:
    """The fields that make an anonymous control list actionable."""
    assert "why=display:none" in rendered
    assert "section='Building Block(s)" in rendered


def test_option_text_is_printed_not_just_counts(rendered: str) -> None:
    """A select's name says little; its options identify a monomer list."""
    assert "Ethylene" in rendered
    assert "Lactide" in rendered


def test_the_artifact_files_are_written(rendered: str, tmp_path: Path) -> None:
    for name in ("dom_summary.json", "accessibility_summary.json",
                 "form_schema.json", "discovery_report.md"):
        assert (tmp_path / name).exists(), name


def test_no_credential_reaches_the_output_or_the_artifacts(
    rendered: str, tmp_path: Path
) -> None:
    assert PASSWORD not in rendered
    assert "a@b.c" not in rendered.split("email")[0]
    for path in tmp_path.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md"}:
            assert PASSWORD not in path.read_text(), path
