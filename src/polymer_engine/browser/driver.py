"""The boundary between the engine and a real browser.

Playwright and its 115 MB Chromium live in ``.browserenv``, not in the engine's
``.venv``. That is the same isolation rule the OpenFF backend follows, and for the same
reason: a campaign that is running must not have its environment change underneath it.

Unlike the OpenFF worker, a browser session is *stateful* -- cookies, a logged-in
context, a current page -- so this bridge keeps one long-lived subprocess and speaks a
line-oriented JSON protocol to it, rather than starting a process per call.

:class:`PageDriver` is a protocol rather than a class so the workflows above it can be
exercised against a scripted fake. That is what makes it possible to test a login flow's
*logic* -- that it stops on a CAPTCHA, that it never pastes -- without a live account.
Those tests prove the decisions, not the site.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from polymer_engine.core.errors import PolymerEngineError, ToolNotFound
from polymer_engine.core.logging import get_logger

logger = get_logger("browser.driver")

REPO_ROOT = Path(__file__).resolve().parents[3]
BROWSER_ENV = REPO_ROOT / ".browserenv"
WORKER = REPO_ROOT / "scripts" / "browser_worker.py"
#: Chromium is installed under the tooling environment so it cannot be confused with,
#: or shared with, anything the scientific stack uses.
BROWSERS_PATH = BROWSER_ENV / "browsers"

#: Commands whose *request* must never be logged, because it names a credential source.
#: The value still never travels here -- only the variable name does -- but the log line
#: is suppressed anyway so a future edit cannot make it leak by accident.
UNLOGGED_COMMANDS = frozenset({"type_secret"})


class BrowserUnavailable(PolymerEngineError):
    """The browser tooling environment is absent or unusable."""


@runtime_checkable
class PageDriver(Protocol):
    """Everything the workflows need a page to do."""

    def send(self, command: str, **payload: Any) -> dict[str, Any]:
        """Issue one command and return the worker's response."""
        ...

    def close(self) -> None:
        ...


class WorkerDriver:
    """Drives the real Playwright worker as a subprocess."""

    def __init__(
        self,
        *,
        python: Path | None = None,
        worker: Path | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        self.python = Path(python) if python else BROWSER_ENV / "bin" / "python"
        self.worker = Path(worker) if worker else WORKER
        self.timeout_s = timeout_s
        self._env = env
        self._proc: subprocess.Popen[str] | None = None

    # -- availability ---------------------------------------------------
    @classmethod
    def available(cls) -> bool:
        return (BROWSER_ENV / "bin" / "python").exists() and WORKER.exists()

    @classmethod
    def capabilities(cls) -> dict[str, Any]:
        """What the browser tooling can actually do, measured rather than declared."""
        python = BROWSER_ENV / "bin" / "python"
        payload: dict[str, Any] = {
            "available": False, "env": str(BROWSER_ENV),
            "worker": str(WORKER), "worker_present": WORKER.exists(),
        }
        if not python.exists():
            payload["error"] = f"no browser environment at {BROWSER_ENV}"
            return payload
        probe = (
            "import json,importlib.metadata as m\n"
            "out={}\n"
            "try:\n"
            "    out['playwright']=m.version('playwright')\n"
            "except Exception as e:\n"
            "    out['error']=str(e); print(json.dumps(out)); raise SystemExit(0)\n"
            "try:\n"
            "    from playwright.sync_api import sync_playwright\n"
            "    with sync_playwright() as p:\n"
            "        b=p.chromium.launch(headless=True)\n"
            "        out['chromium']=b.version; b.close()\n"
            "    out['available']=True\n"
            "except Exception as e:\n"
            "    out['error']=f'{type(e).__name__}: {e}'\n"
            "print(json.dumps(out))\n"
        )
        try:
            done = subprocess.run(
                [str(python), "-c", probe], capture_output=True, text=True,
                timeout=120, env=_worker_env(None), check=False,
            )
            payload.update(json.loads(done.stdout.strip().splitlines()[-1]))
        except Exception as exc:  # noqa: BLE001 - absence is the answer, not a crash
            payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload

    # -- process --------------------------------------------------------
    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.python.exists():
            raise ToolNotFound(
                f"no browser environment at {BROWSER_ENV}",
                hint="python3 -m venv .browserenv && .browserenv/bin/pip install playwright "
                     "&& PLAYWRIGHT_BROWSERS_PATH=.browserenv/browsers "
                     ".browserenv/bin/playwright install chromium",
            )
        self._proc = subprocess.Popen(
            [str(self.python), str(self.worker)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=_worker_env(self._env), cwd=str(REPO_ROOT),
        )
        logger.debug("browser worker started (pid %s)", self._proc.pid)

    def send(self, command: str, **payload: Any) -> dict[str, Any]:
        self.start()
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise BrowserUnavailable("browser worker is not running")
        request = {"cmd": command, **payload}
        if command not in UNLOGGED_COMMANDS:
            logger.debug("browser <- %s", command)
        try:
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        except (BrokenPipeError, ValueError) as exc:
            raise BrowserUnavailable(f"browser worker died: {exc}") from exc
        if not line:
            stderr = ""
            if proc.stderr is not None:
                try:
                    stderr = proc.stderr.read()[-800:]
                except Exception:  # noqa: BLE001
                    stderr = ""
            raise BrowserUnavailable(
                f"browser worker produced no response to {command!r}"
                + (f"; stderr: {stderr}" if stderr else "")
            )
        return json.loads(line)

    def close(self) -> None:
        """Shut the worker down by talking to it, then by PID -- never by name.

        ``terminate()`` here signals exactly ``self._proc``, a handle this object
        created. No command-string matching is involved, so nothing else on the machine
        -- least of all a running campaign -- can be caught by it.
        """
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
            proc.wait(timeout=20)
        except Exception:  # noqa: BLE001 - fall through to the signal
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def __enter__(self) -> WorkerDriver:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _worker_env(extra: dict[str, str] | None) -> dict[str, str]:
    env = dict(extra if extra is not None else os.environ)
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(BROWSERS_PATH))
    # The worker must not import the engine, and must not inherit a PYTHONPATH that
    # would let it try: the two environments have different dependency trees.
    env.pop("PYTHONPATH", None)
    return env


def install_hint() -> str:
    return (
        "python3 -m venv .browserenv\n"
        ".browserenv/bin/pip install playwright\n"
        "PLAYWRIGHT_BROWSERS_PATH=.browserenv/browsers .browserenv/bin/playwright install chromium"
    )


def which_browsers() -> list[str]:
    """Any system browser, reported for diagnostics only.  Never used directly."""
    return [name for name in ("google-chrome", "chromium", "firefox")
            if shutil.which(name)]


__all__ = [
    "BROWSERS_PATH", "BROWSER_ENV", "REPO_ROOT", "WORKER", "BrowserUnavailable",
    "PageDriver", "WorkerDriver", "install_hint", "which_browsers",
]
