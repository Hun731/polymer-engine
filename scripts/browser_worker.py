#!/usr/bin/env python
"""Playwright worker, run inside the isolated ``.browserenv``.

The engine lives in ``.venv``; Playwright and a 115 MB Chromium live in ``.browserenv``.
This script is the bridge, and it holds the browser session for its whole lifetime --
unlike the OpenFF worker, which is one-shot, a browser session is stateful and must
survive across many commands.

Protocol: one JSON object per line on stdin, one JSON object per line on stdout.
Every response carries ``ok`` and, on failure, ``error`` and ``state``.

**The password never appears in this protocol.** ``type_secret`` names an environment
variable; the worker reads it from its own inherited environment, types it with real
keystrokes, and reports back only how many characters the field received. There is no
command that accepts a password as a value and none that returns one.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Field values are echoed back for verification (§29), except for these, whose contents
# must never leave the browser. Matched case-insensitively against the field key.
NEVER_ECHO = ("password", "passwd", "secret", "token", "session")

#: Attributes stripped from any DOM snapshot before it leaves this process (§53, §70).
SENSITIVE_ATTRS = ("value", "data-token", "data-session", "data-csrf",
                   "authorization", "cookie")


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in NEVER_ECHO)


class Worker:
    """Holds one browser, one context and one page."""

    def __init__(self) -> None:
        self._pw: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self.downloads: list[dict[str, Any]] = []
        self._downloads_dir: str | None = None

    # -- lifecycle ------------------------------------------------------
    def launch(self, req: dict[str, Any]) -> dict[str, Any]:
        from playwright.sync_api import sync_playwright

        if self._browser is not None:
            return {"ok": True, "already_running": True}
        headless = bool(req.get("headless", True))
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self._context = self._browser.new_context(
            viewport={"width": int(req.get("width", 1440)),
                      "height": int(req.get("height", 1000))},
            accept_downloads=True,
        )
        self._context.set_default_timeout(float(req.get("timeout_ms", 30000)))
        self._page = self._context.new_page()
        downloads_dir = req.get("downloads_dir")
        self._downloads_dir = downloads_dir
        if downloads_dir:
            Path(downloads_dir).mkdir(parents=True, exist_ok=True)
            # No auto "download" handler: downloads are captured explicitly by
            # download_url / download_click via expect_download, which name each file.
            # An auto-handler would save a redundant second copy under the site's own
            # fixed filename (charmm-gui.tgz), which then collides across builds.
        return {"ok": True, "headless": headless,
                "browser_version": self._browser.version,
                "playwright_version": _playwright_version()}

    def _save_download(self, download: Any, directory: str) -> None:
        target = str(Path(directory) / download.suggested_filename)
        download.save_as(target)
        self.downloads.append({"path": target, "url": download.url,
                               "filename": download.suggested_filename})

    def close(self, _req: dict[str, Any]) -> dict[str, Any]:
        for closer in (self._context, self._browser):
            # Shutting down: why a close failed cannot change what we do next.
            with contextlib.suppress(Exception):
                if closer is not None:
                    closer.close()
        if self._pw is not None:
            with contextlib.suppress(Exception):
                self._pw.stop()
        self._pw = self._browser = self._context = self._page = None
        return {"ok": True}

    def _hidden_note(self) -> str:
        note = getattr(self, "_last_hidden_only", None)
        self._last_hidden_only = None
        return f" (only hidden template elements matched: {note})" if note else ""

    @property
    def page(self) -> Any:
        if self._page is None:
            raise RuntimeError("browser is not launched")
        return self._page

    # -- locating -------------------------------------------------------
    def _resolve(self, locator: dict[str, Any]) -> Any:
        """Turn one locator description into a Playwright locator.

        Coordinates are deliberately unsupported: there is no branch here that accepts
        an (x, y), because a pixel position encodes a window size, not a meaning.
        """
        page = self.page
        strategy = locator["strategy"]
        value = locator["value"]
        exact = bool(locator.get("exact", False))
        if strategy == "css":
            return page.locator(value)
        if strategy == "test_id":
            return page.get_by_test_id(value)
        if strategy == "label":
            return page.get_by_label(value, exact=exact)
        if strategy == "placeholder":
            return page.get_by_placeholder(value, exact=exact)
        if strategy == "text":
            return page.get_by_text(value, exact=exact)
        if strategy == "role":
            name = locator.get("name")
            if name:
                return page.get_by_role(value, name=name, exact=exact)
            return page.get_by_role(value)
        raise ValueError(f"unknown locator strategy {strategy!r}")

    def _first_match(self, locators: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]] | None:
        """Return the first locator that matches exactly one visible element.

        A locator matching *several* elements is rejected rather than silently taking
        index 0: ambiguity about which control we are about to type into is exactly the
        situation where guessing submits the wrong science.
        """
        hidden_only: list[str] = []
        for locator in locators:
            try:
                resolved = self._resolve(locator)
                count = resolved.count()
            except Exception:  # noqa: BLE001 - a non-matching strategy is not an error
                continue
            if count == 0:
                continue
            # Only a *visible* element is a match. A page that keeps a hidden template
            # clone of its form (the Polymer Builder does) satisfies a bare name
            # selector with an element nobody can interact with, and accepting it
            # means a 30-second hang against something that will never become
            # clickable. If every candidate is hidden, the locator has not matched.
            visible = [i for i in range(min(count, 20))
                       if _safe_visible(resolved.nth(i))]
            if len(visible) == 1:
                return resolved.nth(visible[0]), {
                    **locator, "match_count": count,
                    "narrowed_to_visible": count > 1}
            if not visible:
                hidden_only.append(locator.get("value", "?"))
        if hidden_only:
            # Diagnosis for the "state" the caller reports: distinguishes "nothing
            # matched" from "everything that matched is a hidden template clone".
            self._last_hidden_only = hidden_only[:3]
        return None

    def locate(self, req: dict[str, Any]) -> dict[str, Any]:
        match = self._first_match(req["locators"])
        if match is None:
            return {"ok": False, "state": "UI_SCHEMA_MISMATCH",
                    "error": f"no locator matched a unique element for {req.get('key', '?')}"
                             + self._hidden_note(),
                    "tried": req["locators"]}
        _, used = match
        return {"ok": True, "locator": used}

    # -- interaction ----------------------------------------------------
    def type_text(self, req: dict[str, Any]) -> dict[str, Any]:
        """Type a non-secret value with real keystrokes."""
        return self._keyboard_type(req["locators"], req["text"],
                                   req.get("delay_ms", 40), req.get("key", "field"),
                                   echo=not _is_secret_key(req.get("key", "")))

    def type_secret(self, req: dict[str, Any]) -> dict[str, Any]:
        """Type a value read from this process's own environment (§15, §16).

        The value is never transported over the protocol, never returned, and never
        logged. The caller learns only how many characters the field ended up holding,
        which is enough to detect a field that silently rejected the input.
        """
        var = req["env_var"]
        value = os.environ.get(var)
        if not value:
            return {"ok": False, "state": "CREDENTIALS_MISSING",
                    "error": f"environment variable {var} is unset or empty"}
        return self._keyboard_type(req["locators"], value, req.get("delay_ms", 40),
                                   req.get("key", "secret"), echo=False)

    def _keyboard_type(self, locators: list[dict[str, Any]], value: str,
                       delay_ms: Any, key: str, *, echo: bool) -> dict[str, Any]:
        """Click the field, clear it, and type character by character.

        ``page.keyboard.type`` dispatches genuine keydown/keypress/keyup events, which
        is what a form with per-character validation or a JS-managed state machine
        expects. No clipboard is involved at any point: there is no ``navigator.
        clipboard`` call, no ``Control+V``, and no ``fill()`` on this path.
        """
        match = self._first_match(locators)
        if match is None:
            return {"ok": False, "state": "UI_SCHEMA_MISMATCH",
                    "error": f"no unique element matched for {key!r}"
                             + self._hidden_note(),
                    "tried": locators}
        element, used = match
        element.click()
        # Select-all + Delete rather than fill(""): still keyboard events only.
        self.page.keyboard.press("ControlOrMeta+a")
        self.page.keyboard.press("Delete")
        self.page.keyboard.type(value, delay=float(delay_ms))
        try:
            received = int(element.evaluate("el => (el.value || '').length"))
        except Exception:  # noqa: BLE001 - not an input; length is simply unknowable
            received = -1
        result: dict[str, Any] = {
            "ok": received in (-1, len(value)), "locator": used,
            "chars_expected": len(value), "chars_received": received,
        }
        if received not in (-1, len(value)):
            # The field did not take what was typed -- truncation, a maxlength, or an
            # input handler rewriting it. Reported, not retried blindly.
            result["state"] = "UI_SCHEMA_MISMATCH"
            result["error"] = (f"field {key!r} received {received} characters but "
                               f"{len(value)} were typed")
        if echo:
            result["value"] = value
        return result

    def click(self, req: dict[str, Any]) -> dict[str, Any]:
        match = self._first_match(req["locators"])
        if match is None:
            return {"ok": False, "state": "UI_SCHEMA_MISMATCH",
                    "error": f"no unique element matched for {req.get('key', 'click')!r}",
                    "tried": req["locators"]}
        element, used = match
        element.click()
        if req.get("wait_load", True):
            # A page that stays busy is not a failure; the click already landed.
            with contextlib.suppress(Exception):
                self.page.wait_for_load_state("networkidle", timeout=15000)
        return {"ok": True, "locator": used, "url": self.page.url}

    def click_text(self, req: dict[str, Any]) -> dict[str, Any]:
        """Click the element whose visible text matches, exactly and uniquely.

        For controls that a form-control scan cannot reach: a span, a table cell, a
        div with an onclick and nothing else identifying it. Refuses on more than one
        match, for the same reason every other locator here does -- clicking one of
        several similar things is how the wrong thing gets clicked.
        """
        wanted = req["text"].strip().lower()
        index = req.get("index")
        contains = bool(req.get("contains", False))
        counts = self.page.evaluate(
            """([wanted, contains]) => {
              // Contains-mode restricts to genuinely clickable elements; otherwise a
              // button's text also matches every ancestor that wraps it.
              const clickable = (el) =>
                el.tagName === 'BUTTON' || el.tagName === 'A' ||
                el.tagName === 'INPUT' || el.hasAttribute('onclick');
              const match = (el) => {
                const t = (el.innerText || el.value || '').trim().toLowerCase()
                  .replace(/\\s+/g, ' ');
                return contains ? (clickable(el) && t.includes(wanted)) : t === wanted;
              };
              let total = 0, visible = 0;
              document.querySelectorAll('*').forEach((el) => {
                if (el.children.length && !contains) return;   // innermost for exact
                if (!match(el)) return;
                total += 1;
                if (el.offsetParent || el.getClientRects().length) visible += 1;
              });
              return {total, visible};
            }""",
            [wanted.replace("\n", " "), contains],
        )
        # A page that keeps a hidden template row has two of everything. The visible one
        # is the real control; the clone inside the skeleton is not on screen at all.
        if index is None and counts["visible"] != 1:
            return {"ok": False, "state": "UI_SCHEMA_MISMATCH",
                    "error": (f"{counts['total']} element(s) have the exact text "
                              f"{req['text']!r} and {counts['visible']} of them are "
                              f"visible; exactly one visible match is required. Pass an "
                              f"index to choose deliberately."),
                    "total": counts["total"], "visible": counts["visible"]}
        clicked = self.page.evaluate(
            """([wanted, index, contains]) => {
              const clickable = (el) =>
                el.tagName === 'BUTTON' || el.tagName === 'A' ||
                el.tagName === 'INPUT' || el.hasAttribute('onclick');
              const match = (el) => {
                const t = (el.innerText || el.value || '').trim().toLowerCase()
                  .replace(/\\s+/g, ' ');
                return contains ? (clickable(el) && t.includes(wanted)) : t === wanted;
              };
              const all = [];
              for (const el of document.querySelectorAll('*')) {
                if (el.children.length && !contains) continue;
                if (!match(el)) continue;
                all.push(el);
              }
              const target = index === null
                ? all.find(e => e.offsetParent || e.getClientRects().length)
                : all[index];
              if (!target) return false;
              target.click();
              return true;
            }""",
            [wanted.replace("\n", " "), index, contains],
        )
        if not clicked:
            return {"ok": False, "state": "UI_SCHEMA_MISMATCH",
                    "error": f"no clickable element for {req['text']!r}"}
        self.page.wait_for_timeout(float(req.get("settle_ms", 1200)))
        return {"ok": True, "url": self.page.url, "text": req["text"]}

    def select(self, req: dict[str, Any]) -> dict[str, Any]:
        match = self._first_match(req["locators"])
        if match is None:
            return {"ok": False, "state": "UI_SCHEMA_MISMATCH",
                    "error": f"no unique element matched for {req.get('key', 'select')!r}",
                    "tried": req["locators"]}
        element, used = match
        chosen = element.select_option(req["value"])
        return {"ok": bool(chosen), "locator": used, "selected": chosen}

    def read_value(self, req: dict[str, Any]) -> dict[str, Any]:
        """Read back what a control currently holds, for pre-submission verification."""
        key = req.get("key", "field")
        if _is_secret_key(key):
            return {"ok": False, "state": "REFUSED",
                    "error": f"refusing to read back {key!r}: secret fields are never echoed"}
        match = self._first_match(req["locators"])
        if match is None:
            return {"ok": False, "state": "UI_SCHEMA_MISMATCH",
                    "error": f"no unique element matched for {key!r}"}
        element, used = match
        value = element.evaluate(
            "el => el.type === 'checkbox' || el.type === 'radio' ? String(el.checked)"
            " : (el.value !== undefined ? el.value : el.textContent)"
        )
        return {"ok": True, "locator": used, "value": value}

    # -- navigation and observation -------------------------------------
    def goto(self, req: dict[str, Any]) -> dict[str, Any]:
        response = self.page.goto(req["url"], wait_until=req.get("wait_until", "domcontentloaded"))
        return {"ok": True, "url": self.page.url,
                "status": response.status if response else None,
                "title": self.page.title()}

    def url(self, _req: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "url": self.page.url, "title": self.page.title()}

    def text(self, req: dict[str, Any]) -> dict[str, Any]:
        body = self.page.inner_text("body")
        limit = int(req.get("max_chars", 20000))
        return {"ok": True, "text": body[:limit], "truncated": len(body) > limit}

    def links(self, req: dict[str, Any]) -> dict[str, Any]:
        """Every anchor on the page, for navigating by what is actually there."""
        found = self.page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({text: (e.innerText||'').trim().slice(0,120),"
            " href: e.href, title: e.title || null}))",
        )
        needle = (req.get("contains") or "").lower()
        if needle:
            found = [a for a in found
                     if needle in a["text"].lower() or needle in a["href"].lower()]
        return {"ok": True, "links": found[: int(req.get("limit", 400))]}

    def form_fields(self, _req: dict[str, Any]) -> dict[str, Any]:
        """Discover every form control on the page, with its accessible name (§20).

        This is how the Polymer Builder schema is derived. Nothing about the form is
        assumed: labels, options and control types are read off the live DOM. Current
        values are **not** returned for password-like controls.
        """
        controls = self.page.evaluate(
            """() => {
              const named = (el) => {
                if (el.labels && el.labels.length)
                  return Array.from(el.labels).map(l => (l.innerText||'').trim()).join(' ');
                if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
                const id = el.getAttribute('id');
                if (id) { const l = document.querySelector(`label[for="${CSS.escape(id)}"]`);
                          if (l) return (l.innerText||'').trim(); }
                const p = el.closest('label');
                if (p) return (p.innerText||'').trim();
                // A two-column table often puts the label in the preceding cell --
                // but only when that cell is a label. A cell holding its own control
                // is a control cell, and borrowing its text names this element after
                // its neighbour.
                const cell = el.closest('td');
                const prev = cell && cell.previousElementSibling;
                if (prev && !prev.querySelector('input, select, textarea, button'))
                  return (prev.innerText||'').trim();
                return null;
              };
              // A unique CSS path, so a control with no name, id or label is still
              // addressable. Positional and therefore fragile -- the caller marks any
              // field that depends on one, rather than letting it look as solid as a
              // name-based match.
              const cssPath = (el) => {
                if (el.id) return `${el.tagName.toLowerCase()}#${CSS.escape(el.id)}`;
                const parts = [];
                let node = el;
                while (node && node.nodeType === 1 && parts.length < 8) {
                  let part = node.tagName.toLowerCase();
                  if (node.id) { parts.unshift(`${part}#${CSS.escape(node.id)}`); break; }
                  const siblings = node.parentElement
                    ? Array.from(node.parentElement.children).filter(
                        c => c.tagName === node.tagName)
                    : [];
                  if (siblings.length > 1)
                    part += `:nth-of-type(${siblings.indexOf(node) + 1})`;
                  parts.unshift(part);
                  node = node.parentElement;
                }
                return parts.join(' > ');
              };
              return Array.from(document.querySelectorAll('input, select, textarea, button'))
                .map((el, i) => ({
                  index: i,
                  css_path: cssPath(el),
                  tag: el.tagName.toLowerCase(),
                  type: (el.getAttribute('type') || '').toLowerCase() || null,
                  name: el.getAttribute('name'),
                  id: el.getAttribute('id'),
                  test_id: el.getAttribute('data-testid'),
                  label: named(el),
                  placeholder: el.getAttribute('placeholder'),
                  required: el.hasAttribute('required'),
                  disabled: el.disabled === true,
                  visible: !!(el.offsetParent !== null || el.getClientRects().length),
                  options: el.tagName === 'SELECT'
                    ? Array.from(el.options).map(o => ({value: o.value,
                        text: (o.text||'').trim(), selected: o.selected}))
                    : null,
                  checked: (el.type === 'checkbox' || el.type === 'radio')
                    ? el.checked : null,
                  // A button's text is the only thing that says what it does. Without
                  // it a discovered form is a list of anonymous controls.
                  text: (el.tagName === 'BUTTON'
                          ? (el.innerText || '').trim()
                          : (el.getAttribute('value') || '')).slice(0, 80) || null,
                  // Why an element is invisible distinguishes "this section is
                  // collapsed" from "this is a template that is never shown".
                  hidden_reason: (() => {
                    if (el.offsetParent !== null || el.getClientRects().length) return null;
                    let node = el;
                    while (node && node.nodeType === 1) {
                      const st = getComputedStyle(node);
                      if (st.display === 'none')
                        return `display:none on ${node.tagName.toLowerCase()}` +
                               (node.id ? `#${node.id}` : '');
                      if (st.visibility === 'hidden') return 'visibility:hidden';
                      node = node.parentElement;
                    }
                    return 'not rendered';
                  })(),
                  // Which section of the page this control sits under.
                  section: (() => {
                    let node = el;
                    while (node) {
                      let sib = node.previousElementSibling;
                      while (sib) {
                        if (/^H[1-6]$/.test(sib.tagName))
                          return (sib.innerText || '').trim().slice(0, 60);
                        sib = sib.previousElementSibling;
                      }
                      node = node.parentElement;
                    }
                    return null;
                  })(),
                }));
            }"""
        )
        for control in controls:
            # Defence in depth: a password control's presence is reported, its content
            # is not, and no branch above ever read one.
            if (control.get("type") or "") == "password":
                control["options"] = None
        return {"ok": True, "controls": controls, "url": self.page.url,
                "title": self.page.title()}

    def page_inventory(self, _req: dict[str, Any]) -> dict[str, Any]:
        """Everything on the page that could represent a choice, form control or not.

        ``form_fields`` answers "what can I type into?". This answers "how might this
        page be offering me a list of things?" -- which is a different question, and the
        one that matters when a catalogue is not a ``<select>``. Radio groups, checkbox
        groups, datalists, tables, fieldsets and repeated link/card structures are all
        ways a real site presents a list, and none of them is a ``<select>``.

        Reports structure only. No value is read from a password control, and cookies
        and storage are not touched by any branch here.
        """
        return self.page.evaluate(
            """() => {
              const text = (el) => (el ? (el.innerText || '').trim().slice(0, 160) : null);
              const named = (el) => {
                if (el.labels && el.labels.length)
                  return Array.from(el.labels).map(l => (l.innerText||'').trim()).join(' ');
                if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
                const id = el.getAttribute('id');
                if (id) { const l = document.querySelector(`label[for="${CSS.escape(id)}"]`);
                          if (l) return (l.innerText||'').trim(); }
                const p = el.closest('label');
                if (p) return (p.innerText||'').trim();
                const cell = el.closest('td');
                const prev = cell && cell.previousElementSibling;
                if (prev && !prev.querySelector('input, select, textarea, button'))
                  return text(prev);
                return null;
              };

              // Radio and checkbox groups, keyed by the name that makes them a group.
              const groups = {};
              document.querySelectorAll("input[type=radio], input[type=checkbox]")
                .forEach(el => {
                  const key = el.getAttribute('name') || `(unnamed:${el.type})`;
                  (groups[key] = groups[key] || {kind: el.type, name: key, options: []})
                    .options.push({value: el.value, label: named(el),
                                   id: el.getAttribute('id'), checked: el.checked,
                                   visible: !!(el.offsetParent || el.getClientRects().length)});
                });

              const datalists = Array.from(document.querySelectorAll('datalist')).map(d => ({
                id: d.getAttribute('id'),
                options: Array.from(d.options).map(o => ({value: o.value, text: o.text}))}));

              const selects = Array.from(document.querySelectorAll('select')).map(s => ({
                name: s.getAttribute('name'), id: s.getAttribute('id'), label: named(s),
                n_options: s.options.length,
                options: Array.from(s.options).map(o => ({value: o.value,
                  text: (o.text||'').trim(), selected: o.selected}))}));

              const tables = Array.from(document.querySelectorAll('table')).slice(0, 40)
                .map((t, i) => ({
                  index: i, id: t.getAttribute('id'), rows: t.rows.length,
                  headers: Array.from(t.querySelectorAll('th')).slice(0,12).map(h => text(h)),
                  first_rows: Array.from(t.rows).slice(0, 4).map(r =>
                    Array.from(r.cells).slice(0, 8).map(c => text(c))),
                  controls: t.querySelectorAll('input, select, button').length}));

              const fieldsets = Array.from(document.querySelectorAll('fieldset')).map(f => ({
                legend: text(f.querySelector('legend')),
                controls: f.querySelectorAll('input, select, textarea, button').length}));

              const headings = Array.from(document.querySelectorAll('h1,h2,h3,h4'))
                .slice(0, 60).map(h => ({level: h.tagName, text: text(h)}));

              // Repeated link/card structures: a JS widget's options often live here.
              const cards = Array.from(document.querySelectorAll(
                '[role=option], [role=radio], [role=listbox] *, .card, .option, li > a'))
                .slice(0, 200).map(e => ({tag: e.tagName.toLowerCase(),
                  role: e.getAttribute('role'), cls: e.getAttribute('class'),
                  data_value: e.getAttribute('data-value') || e.getAttribute('value'),
                  text: text(e)})).filter(e => e.text);

              // Elements that are clickable without being form controls. The thing
              // that opens a monomer picker can be a span with an onclick and no
              // name, id or role -- invisible to every form-control scan.
              const clickable = Array.from(document.querySelectorAll(
                '[onclick], a[href^="javascript"], [style*="cursor"], span, td, div'))
                .filter(e => {
                  const t = (e.innerText || '').trim();
                  if (!t || t.length > 60) return false;
                  if (e.querySelector('input, select, textarea, button')) return false;
                  const st = getComputedStyle(e);
                  return e.hasAttribute('onclick') || st.cursor === 'pointer';
                })
                .slice(0, 60).map(e => ({
                  tag: e.tagName.toLowerCase(), id: e.getAttribute('id'),
                  cls: (e.getAttribute('class') || '').slice(0, 40) || null,
                  text: (e.innerText || '').trim().slice(0, 60),
                  href: e.getAttribute('href'),
                  data_href: e.getAttribute('data-href'),
                  onclick: (e.getAttribute('onclick') || '').slice(0, 120) || null,
                  cursor: getComputedStyle(e).cursor,
                  visible: !!(e.offsetParent || e.getClientRects().length)}));

              // Buttons, including the ones whose visible label is drawn by CSS or a
              // background image rather than text. CHARMM-GUI's "Next Step" control is
              // one of these: innerText is empty, and the label lives in a ::after
              // pseudo-element or a class. Capture enough to identify and click it.
              const pseudo = (el) => {
                try {
                  const a = getComputedStyle(el, '::after').content;
                  const b = getComputedStyle(el, '::before').content;
                  const clean = (c) => (c && c !== 'none' && c !== 'normal')
                    ? c.replace(/^["']|["']$/g, '') : '';
                  return (clean(b) + ' ' + clean(a)).trim() || null;
                } catch (e) { return null; }
              };
              const buttons = Array.from(document.querySelectorAll(
                'button, input[type=button], input[type=submit], input[type=image], '
                + 'a.button, [role=button], .btn, [class*=next], [class*=nav], '
                + '[onclick*=submit], [onclick*=next]'))
                .slice(0, 120).map(b => ({
                  tag: b.tagName.toLowerCase(), id: b.getAttribute('id'),
                  name: b.getAttribute('name'), type: b.getAttribute('type'),
                  cls: (b.getAttribute('class') || '').slice(0, 60) || null,
                  title: b.getAttribute('title'),
                  aria: b.getAttribute('aria-label'),
                  alt: b.getAttribute('alt'),
                  value: b.getAttribute('value'),
                  pseudo_label: pseudo(b),
                  form_action: b.form ? (b.form.getAttribute('action') || null) : null,
                  text: ((b.innerText || b.getAttribute('value') || '').trim()).slice(0,80),
                  onclick: (b.getAttribute('onclick') || '').slice(0, 160) || null,
                  visible: !!(b.offsetParent || b.getClientRects().length)}));

              return {ok: true, url: location.href, title: document.title,
                      selects, radio_checkbox_groups: Object.values(groups), datalists,
                      tables, fieldsets, headings, cards, buttons, clickable,
                      n_forms: document.forms.length,
                      has_password_field: !!document.querySelector("input[type=password]")};
            }"""
        )

    def handler_choices(self, req: dict[str, Any]) -> dict[str, Any]:
        """Choice lists built from onclick handlers rather than form controls.

        The live Polymer Builder lists monomers as ``<li onclick="set_monomer(this)">``
        inside per-monomer groups. Nothing about that is a form control, so every scan
        that looks for one reports zero. This reads them together with the ancestor text
        that names the group, because the option's own text is only the tacticity
        variant -- "atactic" is not a monomer.
        """
        handler = req.get("handler", "set_monomer")
        return self.page.evaluate(
            """(handler) => {
              const nodes = Array.from(document.querySelectorAll('[onclick]'))
                .filter(e => (e.getAttribute('onclick') || '').includes(handler));

              const label = (el) => (el.innerText || '').trim().slice(0, 80);
              // The nearest ancestor text that is not just the option list itself.
              const groupOf = (el) => {
                let node = el.parentElement, hops = 0;
                while (node && hops < 6) {
                  // A heading or a non-list sibling above this group usually names it.
                  let sib = node.previousElementSibling;
                  while (sib) {
                    const t = (sib.innerText || '').trim();
                    if (t && t.length < 60 && !sib.querySelector('[onclick]'))
                      return t.slice(0, 60);
                    sib = sib.previousElementSibling;
                  }
                  if (node.hasAttribute && node.hasAttribute('data-name'))
                    return node.getAttribute('data-name');
                  if (node.id) return `#${node.id}`;
                  node = node.parentElement; hops += 1;
                }
                return null;
              };

              const groups = {};
              nodes.forEach((el, i) => {
                const g = groupOf(el) || '(ungrouped)';
                (groups[g] = groups[g] || {group: g, options: []}).options.push({
                  index: i, tag: el.tagName.toLowerCase(), text: label(el),
                  value: el.getAttribute('value') || el.getAttribute('data-value'),
                  title: el.getAttribute('title'),
                  onclick: (el.getAttribute('onclick') || '').slice(0, 100),
                  visible: !!(el.offsetParent || el.getClientRects().length),
                  parent_id: el.parentElement ? el.parentElement.getAttribute('id') : null,
                  parent_class: el.parentElement
                    ? (el.parentElement.getAttribute('class') || '').slice(0, 40) : null,
                });
              });
              return {ok: true, handler, n_nodes: nodes.length,
                      groups: Object.values(groups)};
            }""",
            handler,
        )

    def click_handler(self, req: dict[str, Any]) -> dict[str, Any]:
        """Click the Nth element wired to a named onclick handler.

        The Polymer Builder's monomer menu is ``<li onclick="set_monomer(this)">`` --
        no name, id, role or unique text, so no locator strategy reaches a specific
        option. ``handler_choices`` reports each option with its index in document
        order; this clicks that index. Firing ``el.click()`` runs the handler whether
        or not the hover menu is currently showing, which is what the site's own UI
        does through CSS hover.
        """
        handler = req["handler"]
        index = int(req["index"])
        clicked = self.page.evaluate(
            """([handler, index]) => {
              const nodes = Array.from(document.querySelectorAll('[onclick]'))
                .filter(e => (e.getAttribute('onclick') || '').includes(handler));
              if (index < 0 || index >= nodes.length) return null;
              const el = nodes[index];
              el.click();
              return {text: (el.innerText || '').trim().slice(0, 80),
                      n_nodes: nodes.length};
            }""",
            [handler, index],
        )
        if clicked is None:
            return {"ok": False, "state": "UI_SCHEMA_MISMATCH",
                    "error": f"handler {handler!r} has no option at index {index}"}
        self.page.wait_for_timeout(float(req.get("settle_ms", 800)))
        return {"ok": True, **clicked}

    def human_verification(self, req: dict[str, Any]) -> dict[str, Any]:
        """Detect a CAPTCHA or MFA challenge.  Detection only -- never a bypass."""
        markers: list[str] = []
        for css in req.get("captcha_markers", []):
            try:
                if self.page.locator(css).count() > 0:
                    markers.append(f"element:{css}")
            except Exception:  # noqa: BLE001
                continue
        try:
            body = self.page.inner_text("body").lower()
        except Exception:  # noqa: BLE001
            body = ""
        markers += [f"text:{phrase}" for phrase in req.get("mfa_markers", [])
                    if phrase in body]
        return {"ok": True, "detected": bool(markers), "markers": markers}

    def snapshot(self, req: dict[str, Any]) -> dict[str, Any]:
        """A sanitised structural snapshot for diagnostics (§53, §70).

        ``value`` attributes are stripped before the HTML leaves this process, so a
        snapshot taken on a half-filled login form cannot carry a password. Cookies and
        storage are never read at all -- there is no code path here that touches them.
        """
        html = self.page.evaluate(
            """(attrs) => {
              const clone = document.documentElement.cloneNode(true);
              clone.querySelectorAll('script, style').forEach(e => e.remove());
              clone.querySelectorAll('*').forEach(el => {
                for (const a of attrs) if (el.hasAttribute(a)) el.setAttribute(a, '[redacted]');
                if (el.tagName === 'INPUT' && (el.getAttribute('type')||'') === 'password')
                  el.setAttribute('value', '[redacted]');
              });
              return clone.outerHTML;
            }""",
            list(SENSITIVE_ATTRS),
        )
        limit = int(req.get("max_chars", 400000))
        return {"ok": True, "html": html[:limit], "truncated": len(html) > limit,
                "url": self.page.url, "title": self.page.title()}

    def screenshot(self, req: dict[str, Any]) -> dict[str, Any]:
        path = Path(req["path"]).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(path), full_page=bool(req.get("full_page", True)))
        return {"ok": True, "path": str(path)}

    def download_click(self, req: dict[str, Any]) -> dict[str, Any]:
        """Click an element and wait for the download it starts to finish.

        ``page.expect_download`` is the correct idiom: the file may be generated on the
        server and arrive seconds after the click, so a fixed sleep races it. The click
        happens inside the expectation, and the saved path is returned only once the
        transfer completes.

        ``download_project(this)`` on CHARMM-GUI navigates to a tarball served with an
        attachment disposition, which Playwright surfaces as a download event.
        """
        text = req["text"]
        timeout = float(req.get("timeout_ms", 120000))
        directory = self._downloads_dir or "."
        Path(directory).mkdir(parents=True, exist_ok=True)
        target_selector = req.get("selector")
        try:
            with self.page.expect_download(timeout=timeout) as info:
                if target_selector:
                    self.page.locator(target_selector).first.click()
                else:
                    self.page.get_by_text(text, exact=False).first.click()
            download = info.value
        except Exception as exc:  # noqa: BLE001 - the caller needs the reason
            return {"ok": False, "state": _classify(exc),
                    "error": f"no download after clicking {text!r}: {exc}"}
        name = req.get("save_as") or download.suggested_filename or "download.tgz"
        saved = str(Path(directory) / name)
        download.save_as(saved)
        self.downloads.append({"path": saved, "url": download.url, "filename": name})
        return {"ok": True, "path": saved, "filename": name}

    def download_url(self, req: dict[str, Any]) -> dict[str, Any]:
        """Download the file a URL serves, through the current authenticated session.

        The result page's download.tgz carries its target in ``data-href``
        (``?doc=input/download&jobid=...``), and there are several hidden duplicates of
        the link. Rather than fight to click the one visible copy, read the URL and
        fetch it: navigating to it in the logged-in browser sends the session cookies
        and returns the tarball with an attachment disposition, which Playwright
        surfaces as a download.
        """
        url = req["url"]
        timeout = float(req.get("timeout_ms", 180000))
        directory = self._downloads_dir or "."
        Path(directory).mkdir(parents=True, exist_ok=True)
        try:
            with self.page.expect_download(timeout=timeout) as info:  # noqa: SIM117
                # A download response aborts navigation, which Playwright raises; the
                # download itself still fires, so the interruption is expected. The two
                # contexts cannot merge -- the suppression must sit inside the download
                # expectation, or the download event is lost.
                with contextlib.suppress(Exception):
                    self.page.goto(url, wait_until="commit", timeout=timeout)
            download = info.value
        except Exception as exc:  # noqa: BLE001 - the caller needs the reason
            return {"ok": False, "state": _classify(exc),
                    "error": f"no download from {url}: {exc}"}
        name = req.get("save_as") or download.suggested_filename or "download.tgz"
        saved = str(Path(directory) / name)
        download.save_as(saved)
        self.downloads.append({"path": saved, "url": url, "filename": name})
        return {"ok": True, "path": saved, "filename": name}

    def downloads_seen(self, _req: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "downloads": list(self.downloads)}

    def wait(self, req: dict[str, Any]) -> dict[str, Any]:
        self.page.wait_for_timeout(float(req.get("ms", 500)))
        return {"ok": True}


def _safe_visible(locator: Any) -> bool:
    try:
        return bool(locator.is_visible())
    except Exception:  # noqa: BLE001
        return False


def _playwright_version() -> str:
    try:
        import importlib.metadata as metadata

        return metadata.version("playwright")
    except Exception:  # noqa: BLE001
        return "unknown"


COMMANDS = {
    "launch": "launch", "close": "close", "goto": "goto", "url": "url",
    "text": "text", "links": "links", "locate": "locate", "type": "type_text",
    "type_secret": "type_secret", "click": "click", "select": "select",
    "read_value": "read_value", "form_fields": "form_fields",
    "human_verification": "human_verification", "snapshot": "snapshot",
    "page_inventory": "page_inventory", "click_text": "click_text",
    "download_click": "download_click", "download_url": "download_url",
    "handler_choices": "handler_choices", "click_handler": "click_handler",
    "screenshot": "screenshot", "downloads": "downloads_seen", "wait": "wait",
}


def main() -> int:
    worker = Worker()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"ok": False, "error": f"malformed request: {exc}"})
            continue
        command = request.get("cmd", "")
        if command == "shutdown":
            worker.close({})
            _emit({"ok": True, "shutdown": True})
            return 0
        method = COMMANDS.get(command)
        if method is None:
            _emit({"ok": False, "error": f"unknown command {command!r}",
                   "known": sorted(COMMANDS)})
            continue
        try:
            _emit(getattr(worker, method)(request))
        except Exception as exc:  # noqa: BLE001 - the caller needs a reason, not a crash
            _emit({"ok": False, "state": _classify(exc),
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-1200:]})
    return 0


def _classify(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timeout" in text:
        return "TIMEOUT"
    if "closed" in text or "crash" in text or "disconnected" in text:
        return "BROWSER_CRASHED"
    if "net::" in text or "connection" in text or "dns" in text:
        return "NETWORK_ERROR"
    return "NAVIGATION_FAILED"


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
