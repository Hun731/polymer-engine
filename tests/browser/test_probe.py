"""The `probe` command: reveal what a control opens, never start a build.

The live Polymer Builder chooses a monomer through a `<span onclick=...>` reading
"select unit" -- not a form control, and therefore invisible to every scan that looks
for one. `probe` clicks a named element and reports what appeared, so the fields a build
actually needs can be enumerated from the page instead of guessed.

The refusal list is the safety property. Discovery must never be one typo away from
creating a job on a shared academic service.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import importlib.util
import json
import socket
import threading
from pathlib import Path

import pytest

from polymer_engine.browser.driver import WorkerDriver

pytestmark = pytest.mark.skipif(not WorkerDriver.available(), reason="no .browserenv")

FIXTURES = Path(__file__).parent / "fixtures"


def _live():
    spec = importlib.util.spec_from_file_location(
        "charmm_gui_live",
        Path(__file__).resolve().parents[2] / "scripts" / "charmm_gui_live.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def site() -> str:
    with socket.socket() as probe_sock:
        probe_sock.bind(("127.0.0.1", 0))
        port = probe_sock.getsockname()[1]
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(FIXTURES))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()


# -- the refusal list ------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Next Step: Build Polymer Chains", "Submit", "next step", "BUILD POLYMER",
    "Generate", "Continue", "Proceed",
])
def test_build_shaped_text_is_refused_before_anything_launches(text: str, capsys) -> None:
    live = _live()
    assert live.probe(argparse.Namespace(
        click=text, url="unused", handler="set_monomer", index=None,
        out="unused", headed=False)) == 2
    assert "REFUSED" in capsys.readouterr().err
    # Refused before the browser or credentials were touched at all.
    assert "preflight" not in capsys.readouterr().out


def test_an_innocuous_label_is_not_refused() -> None:
    live = _live()
    for text in ("select unit", "Add monomer unit", "Add polymer chain"):
        assert live._refuses(text) is None, text


def test_the_refusal_list_covers_the_real_build_control() -> None:
    """The live page's build control, verbatim."""
    live = _live()
    assert live._refuses("Next Step:\nBuild Polymer Chains") is not None


# -- revealing what a control opens ----------------------------------------
@pytest.fixture
def probed(site: str, tmp_path, monkeypatch, capsys) -> str:
    monkeypatch.setenv("CHARMM_GUI_EMAIL", "a@b.c")
    monkeypatch.setenv("CHARMM_GUI_PASSWORD", "correct-horse")
    live = _live()
    base = site
    original = live.Session

    class Local(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["base_url"] = base
            super().__init__(*args, **kwargs)

        def login(self, *args, **kwargs):
            return super().login(url=f"{base}/login.html", settle_s=0.3)

    live.Session = Local
    code = live.probe(argparse.Namespace(
        click="select unit", url=f"{base}/builder_picker.html",
        handler="set_monomer", index=None, out=str(tmp_path), headed=False))
    assert code == 0
    return capsys.readouterr().out


def test_a_clickable_non_form_element_is_found(probed: str) -> None:
    """A span with an onclick is how the monomer is chosen; no form scan sees it."""
    assert "clickable non-form elements" in probed
    assert "unitpick" in probed
    assert "openPicker()" in probed


def test_clicking_reveals_the_hidden_monomer_list(probed: str) -> None:
    assert "newly visible:   3" in probed
    assert "unit[1]" in probed
    for monomer in ("Ethylene", "Propylene", "Styrene", "Lactide"):
        assert monomer in probed


def test_the_required_fields_are_summarised(probed: str) -> None:
    assert "fields a build would need" in probed
    assert "2 marked required" in probed
    # Both required controls named, and the optional ones not mislabelled.
    assert "REQUIRED input   name=subtext[1]" in probed
    assert "REQUIRED input   name=nrep" in probed


def test_the_result_is_written_for_later_reading(probed: str, tmp_path: Path) -> None:
    payload = json.loads((tmp_path / "probe_result.json").read_text())
    assert payload["clicked"] == "select unit"
    # Revealed, not "new": the select was always in the DOM, just hidden.
    assert payload["revealed_selects"]
    assert payload["revealed_selects"][0]["name"] == "unit[1]"
    assert {"after_click_inventory.json", "after_click_controls.json"} <= {
        p.name for p in tmp_path.iterdir()}


def test_no_credential_reaches_the_probe_output(probed: str, tmp_path: Path) -> None:
    assert "correct-horse" not in probed
    for path in tmp_path.rglob("*.json"):
        assert "correct-horse" not in path.read_text(), path


class TestHandlerDrivenMonomerList:
    """The live page lists monomers as `<li onclick="set_monomer(this)">`.

    No form control carries the choice, so every scan looking for one reported zero. The
    option's own text is the *tacticity variant* -- "atactic", "isotactic (R)" -- and the
    monomer name sits above the group, so reading the option text as the monomer would
    build a catalogue of four tacticities repeated many times.
    """

    PAGE = "builder_setmonomer.html"

    @pytest.fixture
    def extracted(self, site: str, tmp_path, monkeypatch, capsys) -> str:
        monkeypatch.setenv("CHARMM_GUI_EMAIL", "a@b.c")
        monkeypatch.setenv("CHARMM_GUI_PASSWORD", "correct-horse")
        live = _live()
        base = site
        original = live.Session

        class Local(original):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs["base_url"] = base
                super().__init__(*args, **kwargs)

            def login(self, *args, **kwargs):
                return super().login(url=f"{base}/login.html", settle_s=0.3)

        live.Session = Local
        assert live.probe(argparse.Namespace(
            click="select unit", url=f"{base}/{self.PAGE}", handler="set_monomer",
            index=None, out=str(tmp_path), headed=False)) == 0
        return capsys.readouterr().out

    def test_monomer_names_come_from_the_group_not_the_option(
        self, extracted: str
    ) -> None:
        for monomer in ("Ethylene", "Propylene", "Styrene"):
            assert f"group {monomer!r}" in extracted

    def test_tacticity_variants_are_kept_as_variants(self, extracted: str) -> None:
        assert "'isotactic (R)', 'isotactic (S)', 'syndio (R)', 'atactic'" in extracted

    def test_a_duplicate_text_in_a_hidden_template_does_not_block_the_click(
        self, extracted: str
    ) -> None:
        """Two elements read "select unit"; one is inside the skeleton and hidden."""
        assert "could not click" not in extracted
        assert "clicking 'select unit'" in extracted

    def test_the_catalogue_reaches_captured_from_a_handler_list(self) -> None:
        from polymer_engine.browser.catalog import CatalogState
        from polymer_engine.browser.discovery import catalog_from_controls

        choices = {"ok": True, "handler": "set_monomer", "n_nodes": 5, "groups": [
            {"group": "Ethylene", "options": [{"text": "atactic"}]},
            {"group": "Propylene", "options": [
                {"text": "isotactic (R)"}, {"text": "atactic"}]},
        ]}
        catalog = catalog_from_controls([], semantics={}, handler_choices=choices)
        assert catalog.state is CatalogState.CAPTURED
        assert sorted(m.label for m in catalog.monomers) == ["Ethylene", "Propylene"]
        assert dict(catalog.monomers[1].variants and
                    {"Propylene": list(catalog.monomers[1].variants)}) == {
            "Propylene": ["isotactic (R)", "atactic"]}

    def test_an_unnamed_group_is_not_invented_into_a_monomer(self) -> None:
        """A variant list with no monomer above it names no monomer."""
        from polymer_engine.browser.discovery import monomers_from_handler_choices

        entries = monomers_from_handler_choices({"groups": [
            {"group": "(ungrouped)", "options": [{"text": "atactic"}]},
            {"group": "#picker", "options": [{"text": "atactic"}]},
        ]})
        assert entries == []
