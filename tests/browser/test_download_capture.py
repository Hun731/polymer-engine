"""Capturing a server-generated download, which a fixed sleep races.

CHARMM-GUI's download.tgz is produced by download_project(this), which navigates to a
tarball the server builds on demand -- it can arrive seconds after the click. The first
implementation clicked and then slept a fixed interval, and captured nothing when the
file was still being generated. page.expect_download waits for the transfer to complete
instead.
"""

from __future__ import annotations

import io
import socket
import tarfile
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from polymer_engine.browser.credentials import Credentials
from polymer_engine.browser.driver import WorkerDriver
from polymer_engine.browser.session import Session
from polymer_engine.core.config import Secret

pytestmark = pytest.mark.skipif(not WorkerDriver.available(), reason="no .browserenv")


def _make_handler(delay: float):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/page"):
                body = (b"<html><body><h1>Result</h1>"
                        b"<span onclick=\"download_project(this)\">download.tgz</span>"
                        b"<script>function download_project(e)"
                        b"{window.location='/download';}</script></body></html>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/download":
                time.sleep(delay)   # the server builds the tarball on demand
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                    data = b"README\n"
                    info = tarfile.TarInfo("job/README")
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
                payload = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Disposition",
                                 'attachment; filename="download.tgz"')
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


@pytest.fixture
def slow_download_site():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = HTTPServer(("127.0.0.1", port), _make_handler(delay=2.0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()


def test_a_download_generated_after_the_click_is_still_captured(slow_download_site):
    downloads = Path(tempfile.mkdtemp())
    with Session(credentials=Credentials(email="a", password=Secret("b")),
                 base_url=slow_download_site, downloads_dir=downloads) as session:
        session.navigate(f"{slow_download_site}/page")
        result = session.driver.send("download_click", text="download.tgz",
                                     timeout_ms=30000)
    assert result["ok"], result.get("error")
    saved = Path(result["path"])
    assert saved.exists()
    assert tarfile.is_tarfile(saved), "the captured file must be a real archive"
    assert tarfile.open(saved).getnames() == ["job/README"]


def test_a_click_that_starts_no_download_reports_it(slow_download_site):
    """No hang: a control that does not download must fail, not wait forever."""
    with Session(credentials=Credentials(email="a", password=Secret("b")),
                 base_url=slow_download_site) as session:
        session.navigate(f"{slow_download_site}/page")
        result = session.driver.send("download_click", text="nonexistent-control",
                                     timeout_ms=3000)
    assert not result["ok"]
    assert "no download" in result["error"].lower()
