"""The interactive credential prompt (§2, §5, §6).

The prompt exists so a password can reach the browser without passing through a shell,
a history file, a command argument or a chat transcript. These tests check the two
properties that make that true: echo is off, and there is no fallback that would turn it
back on.
"""

from __future__ import annotations

import os
import pty
import select
import sys
from pathlib import Path

import pytest

from polymer_engine.browser.credentials import NoTerminal, prompt
from polymer_engine.browser.driver import WorkerDriver

FAKE_PASSWORD = "correct-horse"


def test_no_terminal_raises_rather_than_echoing() -> None:
    """A silent fallback to input() would print the password to the screen."""
    with pytest.raises(NoTerminal, match="no secure way"):
        prompt({})


def test_the_environment_is_used_without_prompting() -> None:
    creds = prompt({"CHARMM_GUI_EMAIL": "a@b.c", "CHARMM_GUI_PASSWORD": FAKE_PASSWORD})
    assert creds.complete
    assert creds.password.reveal() == FAKE_PASSWORD


def test_prompting_can_be_refused_outright() -> None:
    creds = prompt({}, allow_prompt=False)
    assert not creds.complete
    assert creds.missing == ["CHARMM_GUI_EMAIL", "CHARMM_GUI_PASSWORD"]


@pytest.mark.skipif(not hasattr(pty, "fork"), reason="no pty support")
def test_a_real_terminal_prompt_does_not_echo_the_password() -> None:
    """Run getpass on an actual PTY and confirm the characters never come back."""
    script = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from polymer_engine.browser.credentials import prompt\n"
        "c = prompt({'CHARMM_GUI_EMAIL': 'a@b.c'})\n"
        "print('LEN', len(c.password.reveal() or ''))\n"
        "print('REPR_CLEAN', 'correct-horse' not in repr(c))\n"
    )
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        os.chdir(Path(__file__).resolve().parents[2])
        os.execv(sys.executable, [sys.executable, "-c", script])

    output = b""
    typed = False
    while True:
        ready, _, _ = select.select([fd], [], [], 20)
        if not ready:
            break
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        output += chunk
        if not typed and b"password" in output.lower():
            os.write(fd, FAKE_PASSWORD.encode() + b"\n")
            typed = True
    os.waitpid(pid, 0)

    text = output.decode(errors="replace")
    assert typed, "the prompt never asked for a password"
    assert f"LEN {len(FAKE_PASSWORD)}" in text, text[-400:]
    assert "REPR_CLEAN True" in text
    # The whole point: the terminal never saw the characters.
    assert FAKE_PASSWORD not in text


@pytest.mark.skipif(not WorkerDriver.available(), reason="no .browserenv")
def test_a_prompted_password_reaches_the_browser_as_keystrokes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: prompt -> Secret -> worker environment -> real key events.

    Uses the fixture that rejects paste and counts keydowns, so passing it means the
    characters arrived as genuine keystrokes rather than an injected value.
    """
    import functools
    import http.server
    import socket
    import threading

    from polymer_engine.browser.session import Session

    creds = prompt({"CHARMM_GUI_EMAIL": "a@b.c",
                    "CHARMM_GUI_PASSWORD": FAKE_PASSWORD,
                    "CHARMM_GUI_TYPING_DELAY_MS": "3"})
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", FAKE_PASSWORD)

    fixtures = Path(__file__).parent / "fixtures"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(fixtures))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        with Session(credentials=creds, base_url=base) as session:
            result = session.login(url=f"{base}/login.html", settle_s=0.3)
            assert result.ok, result.detail
            text = session.page_text(300)
            assert "keystrokes=" in text
            assert "paste rejected" not in text
    finally:
        server.shutdown()
