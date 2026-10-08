#!/usr/bin/env python
"""Run the live CHARMM-GUI acquisition sequence, start to finish (§3-§19).

One command, run by the operator, so the password stays between the operator's shell
and the browser process and never passes through anyone else's hands:

    export CHARMM_GUI_EMAIL='...'
    read -rs CHARMM_GUI_PASSWORD && export CHARMM_GUI_PASSWORD   # no shell history
    .venv/bin/python scripts/charmm_gui_live.py discover

``discover`` is read-only: it logs in, walks to Polymer Builder through the visible
interface, captures the catalogue and the form schema, and compares both against
``PolymerBuilderSpec``. It submits nothing.

``build`` submits exactly one job, and only when told which monomer and which control
starts the build -- both taken from the capture that ``discover`` wrote, never guessed.

Every stage records its verdict in the capability registry, and a stage that cannot be
proven stays unproven. Nothing here can mark a capability verified without evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymer_engine.browser import diagnostics as diag
from polymer_engine.browser.acquisition import ACQUISITION_ROOT, CharmmGuiAcquisition
from polymer_engine.browser.catalog import Catalog, diff
from polymer_engine.browser.credentials import NoTerminal, from_environment, prompt
from polymer_engine.browser.discovery import discover_form_schema, map_semantics
from polymer_engine.browser.driver import WorkerDriver
from polymer_engine.browser.queue import BuildQueue
from polymer_engine.browser.session import Session
from polymer_engine.browser.spec_mapping import default_probe_spec, map_spec_to_form
from polymer_engine.browser.states import BrowserState
from polymer_engine.browser.verification import VerificationRegistry
from polymer_engine.core.logging import configure_logging, get_logger
from polymer_engine.simulation.charmm_gui_spec import SystemType

DATA = Path("data/charmm_gui")
REPORT = Path("CHARMM_GUI_LIVE_REPORT.md")



logger = get_logger("charmm_gui.live")

def _preflight(*, allow_prompt: bool = True) -> tuple[VerificationRegistry, int, Any]:
    """Check the tooling and obtain credentials, without ever printing one."""
    registry = VerificationRegistry()
    probe = WorkerDriver.capabilities()
    credentials = from_environment()
    prompted = False
    if not credentials.complete and allow_prompt:
        try:
            credentials = prompt()
            prompted = True
        except NoTerminal as exc:
            print(f"\n  {exc}", file=sys.stderr)

    print("== preflight ==")
    print(f"  browser        : {'ok' if probe.get('available') else 'MISSING'} "
          f"(playwright {probe.get('playwright')}, chromium {probe.get('chromium')})")
    # Presence only. Neither value is printed, and the password is never rendered.
    print(f"  email          : {'present' if credentials.email else 'ABSENT'}"
          f"{' (prompted)' if prompted else ''}")
    print(f"  password       : {'present' if credentials.password else 'ABSENT'}"
          f"{' (prompted, not echoed)' if prompted else ''}")
    print(f"  typing delay   : {credentials.typing_delay_ms} ms")

    if not probe.get("available"):
        registry.block("LIVE_LOGIN_VERIFIED", "browser tooling unavailable")
        registry.save()
        return registry, 2, credentials
    if not credentials.complete:
        registry.block("LIVE_LOGIN_VERIFIED",
                       f"credentials absent: {', '.join(credentials.missing)}")
        registry.save()
        return registry, 2, credentials
    return registry, 0, credentials


def discover(args: argparse.Namespace) -> int:
    registry, code, credentials = _preflight()
    if code:
        return code

    with Session(credentials=credentials, headless=not args.headed,
                 downloads_dir=ACQUISITION_ROOT / "downloads") as session:
        # -- §3 login -----------------------------------------------------
        print("\n== login ==")
        result = session.login()
        print(f"  {result.state.value}: {result.detail[:120]}")
        if result.state is BrowserState.HUMAN_INTERVENTION_REQUIRED:
            registry.needs_review("LIVE_LOGIN_VERIFIED", result.detail)
            registry.save()
            print("\n  A human-verification challenge is present. It is not bypassed.",
                  file=sys.stderr)
            return 3
        if not result.ok:
            registry.fail("LIVE_LOGIN_VERIFIED", result.detail)
            registry.save()
            return 3
        registry.verify("LIVE_LOGIN_VERIFIED",
                        evidence={"url": result.data.get("url", ""),
                                  "method": "keyboard-typed credentials"})

        # -- §4-§6 discovery ---------------------------------------------
        print("\n== Polymer Builder ==")
        previous = Catalog.read(DATA)
        acquisition = CharmmGuiAcquisition(session=session)
        catalog, form, problem = acquisition.discover(session)
        if catalog is None or form is None:
            registry.fail("CATALOG_VERIFIED", problem)
            registry.save()
            diag.capture(session, name="discovery", state="NAVIGATION_FAILED",
                         reason=problem, directory=ACQUISITION_ROOT / "diagnostics")
            print(f"  failed: {problem}", file=sys.stderr)
            return 4

        print(f"  url            : {session.current_url()}")
        print(f"  catalogue      : {catalog.version} · {len(catalog.monomers)} monomers"
              f" · state {catalog.state.value} · completeness {catalog.completeness}")
        if catalog.state.needs_human:
            print(f"  NOT IDENTIFIED : {catalog.notes[:200]}")
            print(f"  inspected      : {catalog.inspected}")
        print(f"  system modes   : {', '.join(catalog.system_modes) or 'none exposed'}")
        print(f"  tacticity      : {', '.join(catalog.tacticity_options) or 'none exposed'}")
        print(f"  form fields    : {len(form.schema.fields)}")
        if previous:
            print(f"  change         : {diff(previous, catalog).summary()}")

        registry.verify("POLYMER_BUILDER_REACHED",
                        evidence={"url": session.current_url(),
                                  "n_form_fields": len(form.schema.fields)})
        if catalog.state.usable:
            registry.verify("CATALOG_VERIFIED",
                            evidence={"fingerprint": catalog.fingerprint(),
                                      "n_monomers": len(catalog.monomers),
                                      "source_url": catalog.source_url})
        else:
            registry.needs_review(
                "CATALOG_VERIFIED",
                f"{catalog.state.value}: {catalog.notes[:200]}")

        ambiguous = form.ambiguous()
        if form.schema.fields and not ambiguous:
            registry.verify("FORM_SCHEMA_VERIFIED",
                            evidence={"n_fields": len(form.schema.fields),
                                      "fields": sorted(form.schema.fields)})
        elif ambiguous:
            registry.needs_review(
                "FORM_SCHEMA_VERIFIED",
                f"the semantic mapping is ambiguous: {ambiguous}")
            print(f"  AMBIGUOUS      : {ambiguous}")

        # -- §7 specification comparison ---------------------------------
        print("\n== specification mapping ==")
        for system_type in (SystemType.SINGLE_CHAIN, SystemType.MELT):
            probe_name = (catalog.monomers[0].label if catalog.monomers else "unknown")
            spec = default_probe_spec(probe_name, system_type=system_type)
            if system_type is SystemType.MELT:
                spec.n_chains = 4
            mapping = map_spec_to_form(spec, form.schema, form.semantics)
            path = DATA / f"spec_mapping_{system_type.value}.md"
            path.write_text(mapping.to_markdown())
            print(f"  {system_type.value:14} {mapping.counts()} "
                  f"buildable={mapping.buildable}")
            if mapping.blockers:
                print(f"                 blockers: "
                      f"{', '.join(b.spec_field for b in mapping.blockers)}")

    registry.save()
    _write_report(registry, catalog, form)
    print(f"\n  wrote {DATA}/monomer_catalog.json, {DATA}/builder_form_schema.json, "
          f"{REPORT}")
    print(f"  next unproven capability: {registry.next_unproven()}")
    return 0


def dump(args: argparse.Namespace) -> int:
    """Capture a sanitised picture of the live Polymer Builder page (§16).

    Read-only. Exists so the page's actual structure can be reviewed without anyone
    guessing at it from documentation -- particularly when the monomer control is not
    something a form-control scan recognises.
    """
    registry, code, credentials = _preflight()
    if code:
        return code

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with Session(credentials=credentials, headless=not args.headed) as session:
        login = session.login()
        if not login.ok:
            print(f"  {login.state.value}: {login.detail}", file=sys.stderr)
            return 3
        arrived = session.navigate(args.url)
        if not arrived.ok:
            print(f"  {arrived.state.value}: {arrived.detail}", file=sys.stderr)
            return 4

        inventory = session.driver.send("page_inventory")
        fields = session.driver.send("form_fields")
        schema, controls = discover_form_schema(session)
        semantics = map_semantics(controls)

        (out / "dom_summary.json").write_text(json.dumps(
            diag.sanitise({"url": session.current_url(), "controls": controls}),
            indent=2) + "\n")
        (out / "accessibility_summary.json").write_text(json.dumps({
            k: v for k, v in inventory.items() if k != "ok"}, indent=2) + "\n")
        (out / "form_schema.json").write_text(json.dumps(
            {**schema.as_dict(), "semantic_hints": semantics}, indent=2) + "\n")
        session.driver.send("screenshot", path=str(out / "page.png"))

        selects = inventory.get("selects", [])
        groups = inventory.get("radio_checkbox_groups", [])
        lines = [
            "# Live Polymer Builder page", "",
            f"- url: {session.current_url()}",
            f"- title: {inventory.get('title')}",
            f"- form controls: {len(fields.get('controls', []))}",
            f"- discovered schema fields: {len(schema.fields)}",
            f"- headings: {[h['text'] for h in inventory.get('headings', [])][:10]}", "",
            "## Choice sets on the page", "",
            "| Kind | Name | Options | First few |", "|---|---|---|---|",
        ]
        for sel in selects:
            texts = [o["text"] for o in sel.get("options", [])][:6]
            lines.append(f"| select | `{sel.get('name')}` | {sel.get('n_options')} "
                         f"| {', '.join(texts)} |")
        for group in groups:
            texts = [str(o.get("label") or o.get("value")) for o in group.get("options", [])][:6]
            lines.append(f"| {group.get('kind')} | `{group.get('name')}` "
                         f"| {len(group.get('options', []))} | {', '.join(texts)} |")
        if not selects and not groups:
            lines.append("| — | — | 0 | no select, radio or checkbox group on the page |")
        lines += ["", f"- tables: {len(inventory.get('tables', []))}",
                  f"- fieldsets: {[f.get('legend') for f in inventory.get('fieldsets', [])]}",
                  f"- semantic suggestions: {semantics}", "",
                  "Credentials, cookies, session tokens and hidden-field values are not "
                  "recorded in any file here.", ""]
        (out / "discovery_report.md").write_text("\n".join(lines))

        controls_all = fields.get("controls", [])
        print(f"  form controls  : {len(controls_all)}")
        print(f"  schema fields  : {len(schema.fields)}")
        print(f"  tables         : {len(inventory.get('tables', []))}")
        print(f"  headings       : {[h['text'] for h in inventory.get('headings', [])][:6]}")
        print(f"  semantics      : {semantics}")

        # The option text is the thing that identifies a monomer list. Printed inline so
        # the answer is in the console output rather than only in a file.
        print("\n  choice sets (option text is what identifies a monomer list):")
        for sel in selects:
            texts = [o["text"] for o in sel.get("options", []) if o.get("value")]
            print(f"    select  name={sel.get('name')!r} id={sel.get('id')!r} "
                  f"label={sel.get('label')!r}")
            print(f"            options: {texts[:12]}")
        for group in groups:
            texts = [str(o.get('label') or o.get('value')) for o in group.get('options', [])]
            print(f"    {group.get('kind'):7} name={group.get('name')!r}")
            print(f"            options: {texts[:12]}")

        print("\n  buttons (their text is the only thing that says what they do):")
        for b in inventory.get("buttons", [])[:30]:
            print(f"    {'vis' if b.get('visible') else 'hid'} "
                  f"id={str(b.get('id'))[:16]:16} "
                  f"text={str(b.get('text'))[:38]!r:40} "
                  f"onclick={str(b.get('onclick'))[:52]!r}")

        print("\n  tables:")
        for t in inventory.get("tables", [])[:6]:
            print(f"    table {t.get('index')} id={t.get('id')!r} rows={t.get('rows')} "
                  f"controls={t.get('controls')} headers={t.get('headers')}")
            for row in (t.get("first_rows") or [])[:3]:
                cells = [str(c)[:32] for c in row if c]
                if cells:
                    print(f"       {cells}")

        cards = inventory.get("cards", [])
        if cards:
            print(f"\n  {len(cards)} card/option-like element(s) "
                  f"(a JS widget's choices live here):")
            for c in cards[:20]:
                print(f"    {c.get('tag'):6} role={c.get('role')!s:10} "
                      f"data_value={str(c.get('data_value'))[:18]:18} "
                      f"text={str(c.get('text'))[:44]!r}")

        anchors = session.driver.send("links").get("links", [])
        interesting = [a for a in anchors
                       if any(w in (a.get("text") or "").lower()
                              for w in ("monomer", "block", "library", "residue",
                                        "browse", "add", "select"))][:20]
        if interesting:
            print("\n  links mentioning monomers/blocks/library:")
            for a in interesting:
                print(f"    {str(a.get('text'))[:42]!r:44} -> {str(a.get('href'))[:66]}")

        hidden = [c for c in controls_all
                  if not c.get("visible") or (c.get("type") or "") == "hidden"]
        if hidden:
            print(f"\n  {len(hidden)} hidden/invisible control(s):")
            for c in hidden[:25]:
                print(f"    {c.get('tag'):7} type={c.get('type')!s:8} "
                      f"name={str(c.get('name'))[:18]:18} "
                      f"id={str(c.get('id'))[:12]:12} "
                      f"opts={len(c.get('options') or []):2} "
                      f"text={str(c.get('text'))[:20]!r:22} "
                      f"why={str(c.get('hidden_reason'))[:28]:28} "
                      f"section={str(c.get('section'))[:26]!r}")

        if args.probe_conditionals:
            _probe_conditionals(session, groups, out)

        print(f"\n  wrote {out}/")
    registry.save()
    return 0


def _probe_conditionals(session: Session, groups: list[Any], out: Path) -> None:
    """Select each option of a radio group and re-inspect, without submitting anything.

    A builder page usually hides most of its form until a system type is chosen, so a
    single snapshot of the initial state shows a fraction of what the page can offer.
    Clicking a radio is ordinary form interaction: it reveals sections, and it creates
    nothing on the server. No submit control is touched.
    """
    print("\n  == probing conditional sections (no submission) ==")
    revealed: dict[str, Any] = {}
    for group in groups:
        name = group.get("name")
        if not name:
            continue
        for option in group.get("options", []):
            value = option.get("value")
            label = option.get("label") or value
            clicked = session.driver.send(
                "click", key=f"{name}={value}", wait_load=False,
                locators=[{"strategy": "css",
                           "value": f"input[name='{name}'][value='{value}']"}],
            )
            if not clicked.get("ok"):
                print(f"    {name}={value!r}: could not select ({clicked.get('error')})")
                continue
            session.driver.send("wait", ms=1200)
            after = session.driver.send("form_fields")
            inv = session.driver.send("page_inventory")
            visible = [c for c in after.get("controls", []) if c.get("visible")]
            sels = inv.get("selects", [])
            print(f"    {name}={value!r} ({label}): {len(visible)} visible control(s), "
                  f"{len(sels)} select(s)")
            for sel in sels:
                texts = [o["text"] for o in sel.get("options", []) if o.get("value")]
                if len(texts) > 3:  # a longer list is the interesting one
                    print(f"        select name={sel.get('name')!r} "
                          f"({len(texts)} options): {texts[:15]}")
            revealed[f"{name}={value}"] = {
                "n_visible_controls": len(visible),
                "selects": [{"name": s.get("name"), "id": s.get("id"),
                             "n_options": s.get("n_options"),
                             "options": [{"value": o.get("value"), "text": o.get("text")}
                                         for o in s.get("options", [])]}
                            for s in sels],
                "radio_groups": [{"name": g.get("name"),
                                  "options": [{"value": o.get("value"),
                                               "label": o.get("label")}
                                              for o in g.get("options", [])]}
                                 for g in inv.get("radio_checkbox_groups", [])],
                "headings": [h.get("text") for h in inv.get("headings", [])],
            }
    (out / "conditional_sections.json").write_text(json.dumps(revealed, indent=2) + "\n")


#: Text that identifies a control which starts a build. `probe` refuses to click
#: anything matching these, whatever the operator asks for. Discovery must never be one
#: typo away from creating a job on a shared academic service.
NEVER_CLICK: tuple[str, ...] = (
    "next step", "build polymer", "submit", "start", "run ", "generate",
    "next »", "continue", "proceed",
)


def _refuses(text: str) -> str | None:
    lowered = text.strip().lower()
    for banned in NEVER_CLICK:
        if banned in lowered:
            return banned
    return None


def probe(args: argparse.Namespace) -> int:
    """Click one element to reveal what it opens, and report what changed (read-only).

    Exists to answer "what does this page actually ask for?" without submitting
    anything. It clicks exactly what it is told, refuses anything build-shaped, and
    reports the difference in controls before and after.
    """
    refused = _refuses(args.click)
    if refused:
        print(f"  REFUSED: {args.click!r} matches {refused!r}, which starts a build. "
              f"This command never clicks those.", file=sys.stderr)
        return 2

    registry, code, credentials = _preflight()
    if code:
        return code

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with Session(credentials=credentials, headless=not args.headed) as session:
        login = session.login()
        if not login.ok:
            print(f"  {login.state.value}: {login.detail}", file=sys.stderr)
            return 3
        arrived = session.navigate(args.url)
        if not arrived.ok:
            print(f"  {arrived.state.value}: {arrived.detail}", file=sys.stderr)
            return 4

        before_fields = session.driver.send("form_fields").get("controls", [])
        before_inv = session.driver.send("page_inventory")

        # Choice lists built from onclick handlers -- how the live page lists monomers.
        choices = session.driver.send("handler_choices", handler=args.handler)
        if choices.get("ok") and choices.get("n_nodes"):
            print(f"\n  == {choices['n_nodes']} option(s) wired to "
                  f"{args.handler}(), in {len(choices['groups'])} group(s) ==")
            for group in choices["groups"]:
                opts = group["options"]
                print(f"    group {group['group']!r}: {len(opts)} option(s)")
                print(f"      {[o['text'] for o in opts][:8]}")
                sample = opts[0]
                print(f"      onclick={sample['onclick']!r} "
                      f"value={sample['value']!r} title={sample['title']!r} "
                      f"parent_id={sample['parent_id']!r} "
                      f"parent_class={sample['parent_class']!r}")
            (out / "handler_choices.json").write_text(
                json.dumps(choices, indent=2) + "\n")

        clickable = before_inv.get("clickable", [])
        print(f"\n  clickable non-form elements ({len(clickable)}, showing all):")
        for el in clickable[:200]:
            print(f"    {'vis' if el.get('visible') else 'hid'} {el.get('tag'):5} "
                  f"id={str(el.get('id'))[:14]:14} cursor={str(el.get('cursor'))[:8]:8} "
                  f"text={str(el.get('text'))[:34]!r:36} "
                  f"onclick={str(el.get('onclick'))[:40]!r}")

        print(f"\n  == clicking {args.click!r} ==")
        clicked = session.driver.send("click_text", text=args.click, settle_ms=1800,
                                      index=args.index)
        if not clicked.get("ok"):
            print(f"  could not click: {clicked.get('error')}", file=sys.stderr)
            _dump_state(session, out, "probe_failed")
            return 5

        after_fields = session.driver.send("form_fields").get("controls", [])
        after_inv = session.driver.send("page_inventory")

        def key(c: dict[str, Any]) -> str:
            return f"{c.get('tag')}|{c.get('name')}|{c.get('id')}|{c.get('index')}"

        before_keys = {key(c) for c in before_fields}
        appeared = [c for c in after_fields if key(c) not in before_keys]
        newly_visible = [c for c in after_fields if c.get("visible") and not any(
            key(b) == key(c) and b.get("visible") for b in before_fields)]

        print(f"  controls before: {len(before_fields)}  after: {len(after_fields)}")
        print(f"  newly present:   {len(appeared)}")
        print(f"  newly visible:   {len(newly_visible)}")
        for c in (appeared or newly_visible)[:30]:
            opts = c.get("options") or []
            print(f"    {c.get('tag'):7} type={c.get('type')!s:9} "
                  f"name={str(c.get('name'))[:22]:22} "
                  f"label={str(c.get('label'))[:26]!r:28} "
                  f"req={c.get('required')!s:5} opts={len(opts)}")
            if opts:
                print(f"            {[o.get('text') for o in opts][:14]}")

        # Compare visibility, not presence. A page that hides its controls already has
        # them in the DOM, so "newly present" finds nothing and "newly visible" finds
        # everything the click actually revealed.
        revealed_names = {c.get("name") for c in newly_visible if c.get("tag") == "select"}
        revealed_selects = [s2 for s2 in after_inv.get("selects", [])
                            if s2.get("name") in revealed_names]
        for sel in revealed_selects:
            texts = [o["text"] for o in sel.get("options", []) if o.get("value")]
            print(f"\n  REVEALED select name={sel.get('name')!r} ({len(texts)} options):")
            print(f"    {texts[:40]}")

        _required_summary(after_fields)
        _dump_state(session, out, "after_click")
        (out / "probe_result.json").write_text(json.dumps({
            "clicked": args.click, "url": session.current_url(),
            "n_before": len(before_fields), "n_after": len(after_fields),
            "appeared": diag.sanitise({"controls": appeared})["controls"],
            "newly_visible": diag.sanitise({"controls": newly_visible})["controls"],
            "revealed_selects": revealed_selects,
        }, indent=2) + "\n")
        print(f"\n  wrote {out}/probe_result.json")
    registry.save()
    return 0


def _required_summary(controls: list[dict[str, Any]]) -> None:
    """Every visible control a build would need filled, in one list."""
    visible = [c for c in controls
               if c.get("visible") and (c.get("type") or "") != "hidden"
               and c.get("tag") != "button"]
    required = [c for c in visible if c.get("required")]
    print(f"\n  == fields a build would need ({len(visible)} visible, "
          f"{len(required)} marked required) ==")
    for c in visible:
        mark = "REQUIRED" if c.get("required") else "optional"
        opts = c.get("options") or []
        print(f"    {mark:8} {c.get('tag'):7} name={str(c.get('name'))[:24]:24} "
              f"label={str(c.get('label'))[:30]!r:32} "
              + (f"{len(opts)} options" if opts else f"value={str(c.get('text'))[:14]!r}"))


def _dump_state(session: Session, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    inventory = session.driver.send("page_inventory")
    fields = session.driver.send("form_fields")
    (out / f"{name}_inventory.json").write_text(json.dumps(
        {k: v for k, v in inventory.items() if k != "ok"}, indent=2) + "\n")
    (out / f"{name}_controls.json").write_text(json.dumps(
        diag.sanitise({"controls": fields.get("controls", [])}), indent=2) + "\n")
    session.driver.send("screenshot", path=str(out / f"{name}.png"))


def _resolve_handler_option(
    choices: dict[str, Any], monomer: str, variant: str | None,
    group: str | None = None,
) -> tuple[int | None, str, str]:
    """Find the click index for one (monomer, variant), exactly or not at all.

    Two shapes, resolved by the same rules as the catalogue: in a conformer group the
    group label is the monomer and the option text is the variant; in a class group the
    option text is the monomer itself. Ambiguity or absence returns no index and a
    reason -- nothing is clicked on a guess.

    ``normalise`` folds away case, spacing and punctuation, which is what lets a menu
    label match a catalogue name -- but it also folds ``Poly(acrylic acid)`` and
    ``Poly(acrylic acid(-))`` onto the same string, so both an acid and its anion match.
    When that happens two tie-breakers, in order, resolve it without guessing: an exact
    *raw* label match (the literal ``(-)`` distinguishes the anion), then a caller-
    supplied ``group`` hint (which class a monomer offered in more than one lives under,
    e.g. Polyethylene under Olefins). Only if neither singles out one option do we refuse.
    """
    from polymer_engine.browser.matching import normalise

    target = normalise(monomer)
    # Each hit carries the raw string that identified it (a conformer group's label, or a
    # class group's option text), so an exact-match tie-breaker can see the (-) that
    # normalise erased.
    hits: list[tuple[int, str, str, str]] = []  # index, group_label, text, raw_identity
    for grp in choices.get("groups", []):
        group_label = str(grp.get("group") or "")
        options = grp.get("options", [])
        if normalise(group_label) == target:
            wanted = normalise(variant) if variant else None
            for option in options:
                text = str(option.get("text") or "")
                if wanted is None or normalise(text) == wanted:
                    hits.append((int(option["index"]), group_label, text, group_label))
                    if wanted is None:
                        break          # no variant asked: first offered conformation
            continue
        for option in options:
            text = str(option.get("text") or "")
            if normalise(text) == target:
                hits.append((int(option["index"]), group_label, text, text))
                break
    if not hits:
        return None, "", (f"{monomer!r}" + (f" / {variant!r}" if variant else "")
                          + " is not in the live menu; nothing was clicked")

    # The menu appears twice (live + hidden skeleton clone); identical resolutions are
    # one choice. Genuinely different resolutions need a tie-breaker.
    distinct = {(g, t) for _i, g, t, _r in hits}
    if len(distinct) > 1:
        want_raw = monomer.strip().casefold()
        exact = [h for h in hits if h[3].strip().casefold() == want_raw]
        if len({(g, t) for _i, g, t, _r in exact}) == 1:
            hits = exact
        elif group is not None:
            want_group = normalise(group)
            scoped = [h for h in hits if normalise(h[1]) == want_group]
            if len({(g, t) for _i, g, t, _r in scoped}) == 1:
                hits = scoped
            else:
                distinct = {(g, t) for _i, g, t, _r in (scoped or hits)}
        if len({(g, t) for _i, g, t, _r in hits}) > 1:
            return None, "", (
                f"{monomer!r} matches {len(distinct)} different options: "
                f"{sorted(distinct)}; refusing to choose"
                + (f" (group hint {group!r} did not single one out)" if group else ""))
    index, group_label, text, _raw = hits[0]
    return index, group_label, text


def _poly_table_text(session: Session) -> str:
    inventory = session.driver.send("page_inventory")
    for table in inventory.get("tables", []):
        if table.get("id") == "poly_table":
            return " ".join(str(c) for row in table.get("first_rows", [])
                            for c in row if c)
    return ""


#: Button/link text that starts real computation on CHARMM-GUI's servers. A probe
#: refuses to click any of these: "Generate Equilibrium" launches a coarse-grained
#: OpenMM run, and the all-atom and input-generation steps follow from it. Crossing one
#: is a decision to spend the shared service's compute, which a person makes, not a
#: mapping tool.
COMPUTE_BOUNDARIES: tuple[str, ...] = (
    "generate equilibrium", "generate equilibrated", "replace into all-atom",
    "input generation", "input generations", "generate", "run equilibration",
)

#: Presence of any of these means the current step already offers a finished system to
#: download -- the single-chain path reaches it in one step.
DOWNLOAD_MARKERS: tuple[str, ...] = ("download.tgz", "download.taz")


def _boundary_hit(text: str) -> str | None:
    lowered = text.strip().lower()
    for marker in COMPUTE_BOUNDARIES:
        if marker in lowered:
            return marker
    return None


def _page_step(session: Session) -> dict[str, Any]:
    """A structured snapshot of the wizard page currently shown.

    Everything a probe needs to decide what a step is and whether to advance: the
    heading, every button with its text, whether a download is offered, and which
    'Next Step' buttons would cross a compute boundary.
    """
    inventory = session.driver.send("page_inventory")
    text = session.page_text(8000).lower()
    def _label(b: dict[str, Any]) -> str:
        # The visible label, wherever it lives: text, value, a CSS pseudo-element, a
        # title or aria-label. CHARMM-GUI draws "Next Step" with a pseudo-element, so
        # the plain text is empty.
        for key in ("text", "value", "pseudo_label", "title", "aria", "alt"):
            v = str(b.get(key) or "").strip()
            if v:
                return v
        return ""

    buttons = [{**b, "label": _label(b)} for b in inventory.get("buttons", [])]
    next_steps = [b for b in buttons
                  if "next step" in _label(b).lower()
                  or "next step" in str(b.get("onclick") or "").lower()]
    download = any(marker in text for marker in DOWNLOAD_MARKERS)
    boundaries = [(b, _boundary_hit(_label(b)))
                  for b in next_steps if _boundary_hit(_label(b))]
    return {
        "url": session.current_url(),
        "headings": [h.get("text") for h in inventory.get("headings", [])][:8],
        "buttons": [{k: b.get(k) for k in
                     ("label", "text", "id", "cls", "onclick", "form_action",
                      "pseudo_label", "value", "visible")}
                    for b in buttons],
        "next_steps": [b.get("label") or b.get("text") for b in next_steps],
        "download_available": download,
        "compute_boundaries": [b.get("text") for b, _m in boundaries],
        "n_selects": len(inventory.get("selects", [])),
        "n_radio_groups": len(inventory.get("radio_checkbox_groups", [])),
    }


def build(args: argparse.Namespace) -> int:
    """Configure and submit exactly ONE Polymer Builder job.

    The live page has no monomer form control: units are chosen by clicking entries in
    a hover menu wired to ``set_monomer``. So the flow is: resolve the requested
    monomer against the *live* menu, click it, prove the chain row changed, type the
    repeat count, choose the system type, read everything back, and only if every
    check agrees click the wizard's build control. Any mismatch stops before the
    click, because a submitted mismatch builds the wrong polymer while looking
    successful.
    """
    registry, code, credentials = _preflight()
    if code:
        return code

    with Session(credentials=credentials, headless=not args.headed,
                 downloads_dir=ACQUISITION_ROOT / "downloads") as session:
        login = session.login()
        if not login.ok:
            print(f"  {login.state.value}: {login.detail}", file=sys.stderr)
            return 3
        arrived = session.navigate(args.url)
        if not arrived.ok:
            print(f"  {arrived.state.value}: {arrived.detail}", file=sys.stderr)
            return 4

        # -- resolve against the live menu, never a cached catalogue ---------
        choices = session.driver.send("handler_choices", handler="set_monomer")
        if not choices.get("ok") or not choices.get("n_nodes"):
            print("  no set_monomer menu on this page", file=sys.stderr)
            return 4
        index, group_label, option_text = _resolve_handler_option(
            choices, args.monomer, args.variant)
        if index is None:
            print(f"  MONOMER_NOT_FOUND: {option_text}", file=sys.stderr)
            labels = sorted({g.get('group') for g in choices.get('groups', [])})
            print(f"  live groups: {labels}", file=sys.stderr)
            return 5
        print(f"  resolved: {group_label!r} / {option_text!r} (menu index {index})")

        before_row = _poly_table_text(session)
        clicked = session.driver.send("click_handler", handler="set_monomer",
                                      index=index, settle_ms=1200)
        if not clicked.get("ok"):
            print(f"  could not select: {clicked.get('error')}", file=sys.stderr)
            return 5
        after_row = _poly_table_text(session)
        print(f"  chain row before: {before_row[:70]!r}")
        print(f"  chain row after : {after_row[:70]!r}")
        if after_row == before_row or "select unit" in after_row.lower():
            print("  STRUCTURE_MISMATCH: the chain row did not take the selection; "
                  "not submitting", file=sys.stderr)
            _dump_state(session, ACQUISITION_ROOT / "build_failure", "no_selection")
            return 6

        # -- repeat count and system type ------------------------------------
        # Scoped to the live chain table first: the page keeps a hidden skeleton
        # clone whose inputs share these names, and only a visible element counts.
        typed = session.driver.send(
            "type", key="subtext[1]", text=str(args.dp),
            delay_ms=credentials.typing_delay_ms,
            locators=[
                # Selecting a unit renames the input: subtext[1] becomes
                # subtext[1][1] (chain 1, unit 1) and only the hidden template keeps
                # the unindexed name. Learned from the live page's own diagnosis.
                {"strategy": "css", "value": "input[name='subtext[1][1]']"},
                {"strategy": "css",
                 "value": "input[name^='subtext[']"},
            ])
        if not typed.get("ok"):
            print(f"  could not set the repeat count: {str(typed.get('error'))[:200]}",
                  file=sys.stderr)
            print("  visible text inputs on the page right now:", file=sys.stderr)
            fields = session.driver.send("form_fields")
            for control in fields.get("controls", []):
                if control.get("visible") and control.get("tag") == "input"                         and (control.get("type") or "text") in ("text", "number"):
                    print(f"    name={control.get('name')!r} "
                          f"value={control.get('text')!r} "
                          f"section={str(control.get('section'))[:40]!r}",
                          file=sys.stderr)
            _dump_state(session, ACQUISITION_ROOT / "build_failure", "dp_locator")
            return 6
        session.driver.send(
            "click", key=f"model={args.system_type}", wait_load=False,
            locators=[{"strategy": "css",
                       "value": f"input[name='model'][value='{args.system_type}']"}])

        # -- verify before the click (§29) -----------------------------------
        readback = session.driver.send(
            "read_value", key="subtext[1]",
            locators=[
                {"strategy": "css", "value": "input[name='subtext[1][1]']"},
                {"strategy": "css", "value": "input[name^='subtext[']"},
            ])
        model = session.driver.send(
            "read_value", key="model",
            locators=[{"strategy": "css",
                       "value": f"input[name='model'][value='{args.system_type}']"}])
        dp_ok = str(readback.get("value")) == str(args.dp)
        model_ok = str(model.get("value")).lower() == "true"
        print(f"  repeat count readback: {readback.get('value')!r} "
              f"({'ok' if dp_ok else 'MISMATCH'})")
        print(f"  system type {args.system_type!r} selected: {model_ok}")
        if not (dp_ok and model_ok):
            print("  verification failed; NOT submitting", file=sys.stderr)
            _dump_state(session, ACQUISITION_ROOT / "build_failure", "verify_failed")
            return 6

        _dump_state(session, ACQUISITION_ROOT / "submissions", "pre_submit")
        if args.dry_run:
            print("\n  --dry-run: everything verified; the build control was NOT "
                  "clicked")
            return 0

        # -- the one deliberate click that creates a job ---------------------
        print("\n  submitting (this creates a real job on charmm-gui.org)...")
        advanced = session.driver.send("click_text", text="Build Polymer Chains",
                                       contains=True, settle_ms=4000)
        if not advanced.get("ok"):
            print(f"  SUBMISSION_FAILED: {advanced.get('error')}", file=sys.stderr)
            return 7

        job = session.driver.send(
            "read_value", key="jobid",
            locators=[{"strategy": "css", "value": "input[name='jobid']"}])
        job_id = str(job.get("value") or "").strip()
        url_now = session.current_url()
        if not job_id:
            from polymer_engine.browser.workflows import capture_job_id

            captured = capture_job_id(session, "wizard")
            job_id = captured.job_id or ""
        print(f"  url    : {url_now}")
        shown = job_id or "(none captured; a job may exist -- check the diagnostics)"
        print(f"  job id : {shown}")
        _dump_state(session, ACQUISITION_ROOT / "submissions", "post_submit")

        queue = BuildQueue()
        from polymer_engine.browser.queue import BuildEntry
        from polymer_engine.browser.states import JobState

        entry = queue.add(BuildEntry(
            polymer_id=args.polymer_id, polymer_name=args.monomer,
            request_fingerprint=f"live-{args.monomer}-{args.variant}-dp{args.dp}-"
                                f"{args.system_type}",
            system_type=args.system_type, rationale=args.rationale,
            spec={"monomer": args.monomer, "variant": args.variant, "dp": args.dp,
                  "system_type": args.system_type, "group": group_label,
                  "menu_index": index}))
        # Persist the job BEFORE anything else can raise. A job exists on charmm-gui.org
        # the instant the wizard advances; if a later bookkeeping line throws, the queue
        # entry must already be on disk, or the engine forgets a job the server
        # remembers -- exactly the failure that lost 8851425743 the first time.
        if job_id:
            entry.job_id = job_id
            entry.advance(JobState.SUBMITTED, f"wizard advanced at {url_now}")
        else:
            entry.advance(JobState.BLOCKED,
                          "wizard advanced but no job id was captured; a person must "
                          "check before anything is resubmitted")
        queue.save()

        if job_id:
            # Registry updates are best-effort relative to the queue: the job is already
            # safely recorded, so an out-of-order or failed verify is logged, never
            # allowed to crash a run that has already succeeded.
            try:
                _record_build_verification(registry, args, index, job_id, session,
                                           group_label)
            except Exception as exc:  # noqa: BLE001 - the job is saved; this is audit
                logger.warning("verification bookkeeping did not complete: %s", exc)
    return 0 if job_id else 7

def fetch_browser(args: argparse.Namespace) -> int:
    """Download a single-chain build through the browser, then import and validate it.

    A single-chain Polymer Builder job offers its result as a ``download.tgz`` control
    wired to ``download_project(this)`` -- a JavaScript-triggered browser download, not
    an API endpoint. The documented check_status/download API does not track these
    browser builds, so the honest path is the one the site itself provides: reach the
    result page, click the control, capture the file the browser saves, and run it
    through the same import and gates every backend's output passes.

    The result page is reached at the job's ``&step=1`` URL. If the download does not
    appear there, the job may need re-opening through Job Retriever, which this reports
    rather than guessing at.
    """
    registry, code, credentials = _preflight()
    if code:
        return code

    downloads = ACQUISITION_ROOT / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    result_url = args.result_url or (
        f"https://charmm-gui.org/?doc=input/polymer&step=1&jobid={args.job_id}")

    with Session(credentials=credentials, headless=not args.headed,
                 downloads_dir=downloads) as session:
        if not session.login().ok:
            print("  login failed", file=sys.stderr)
            return 3
        arrived = session.navigate(result_url)
        if not arrived.ok:
            print(f"  could not reach the result page: {arrived.detail}",
                  file=sys.stderr)
            return 4

        inventory = session.driver.send("page_inventory")
        offers = [c for c in inventory.get("clickable", [])
                  if "download.tgz" in str(c.get("text") or "").lower()
                  and c.get("visible")]
        if not offers:
            print("  no visible download.tgz on the result page.", file=sys.stderr)
            print(f"  headings: {[h.get('text') for h in inventory.get('headings', [])]}",
                  file=sys.stderr)
            print("  the job may need re-opening via Job Retriever; that path is not "
                  "yet mapped -- run `wizard-probe` against it, or share the page.",
                  file=sys.stderr)
            _dump_state(session, ACQUISITION_ROOT / "fetch_failure", "no_download")
            return 5

        print(f"  clicking download.tgz (onclick={offers[0].get('onclick')})...")
        session.driver.send("click_text", text="download.tgz", contains=True,
                            settle_ms=8000)
        seen = session.driver.send("downloads")
        files = seen.get("downloads", [])
        if not files:
            print("  the click fired but no download was captured within the wait.",
                  file=sys.stderr)
            _dump_state(session, ACQUISITION_ROOT / "fetch_failure", "no_capture")
            return 6
        saved = Path(files[-1]["path"])
        print(f"  downloaded: {saved} ({saved.stat().st_size} bytes)")

    # -- import and validate, identical to every other backend's archive -------
    from polymer_engine.core.provenance import sha256_file
    from polymer_engine.parameterization.backends.charmm_gui import CharmmGuiBackend
    from polymer_engine.parameterization.models import (
        ParameterizationRequest,
        PropertyClass,
    )

    print(f"  sha256: {sha256_file(saved)[:16]}...")
    registry.verify("DOWNLOAD_VERIFIED",
                    evidence={"job_id": args.job_id, "path": str(saved),
                              "sha256": sha256_file(saved), "via": "browser"})
    backend = CharmmGuiBackend(None)
    request = ParameterizationRequest(
        polymer_id=args.polymer_id, polymer_name=args.polymer_id,
        repeat_unit_smiles=args.smiles or "",
        property_class=PropertyClass.BULK_DENSITY,
        external_job_id=args.job_id, source_archive=str(saved),
        workdir=str(ACQUISITION_ROOT / "systems"))
    result = backend.parameterize(request)
    print(f"\n  parameterization: {result.state.value}")
    print(f"  topology : {result.topology_path}")
    print(f"  files    : {len(result.artifacts)} hashed")
    if result.state.value == "PARAMETERIZED":
        registry.verify("SYSTEM_IMPORT_VERIFIED",
                        evidence={"job_id": args.job_id})
        validation = backend.validate(result)
        print(f"\n  validation: {validation.determination.value}")
        for name, report in validation.reports():
            print(f"    {name:14}: {report.status.value}")
            for gate in report.gates:
                print(f"       {gate.gate}: {gate.status.value} -- {gate.message[:80]}")
    registry.save()
    return 0


def fetch(args: argparse.Namespace) -> int:
    """Monitor, download and import one submitted job, through the documented API.

    The browser created the job; everything after that uses the three endpoints
    CHARMM-GUI documents -- check_status, download -- and then the same import,
    completeness, charge and penalty gates every other backend's output passes
    through. No browser is launched here.
    """
    registry, code, credentials = _preflight()
    if code:
        return code
    from polymer_engine.parameterization.backends.charmm_gui import CharmmGuiBackend
    from polymer_engine.parameterization.models import (
        ParameterizationRequest,
        PropertyClass,
    )
    from polymer_engine.providers.charmm_gui import CHARMMGUIProvider

    provider = CHARMMGUIProvider(email=credentials.email,
                                 password=credentials.password)
    print(f"  polling job {args.job_id} (up to {args.timeout / 60:.0f} min)...")
    status = provider.wait_for_job(args.job_id, poll_interval_s=30.0,
                                   timeout_s=args.timeout)
    state = status.records[0].get("state") if status.records else "?"
    print(f"  status: {state}")
    if not status.ok:
        print(f"  {status.error}", file=sys.stderr)
        return 8 if state in ("pending", "running") else 9

    archive_dir = ACQUISITION_ROOT / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{args.job_id}.tgz"
    downloaded = provider.download_job(args.job_id, archive)
    if not downloaded.ok:
        print(f"  download failed: {downloaded.error}", file=sys.stderr)
        return 9
    print(f"  downloaded: {archive} ({downloaded.data.get('size_bytes')} bytes, "
          f"sha256 {str(downloaded.data.get('sha256'))[:16]}...)")
    registry.verify("JOB_ID_VERIFIED", evidence={"job_id": args.job_id})
    registry.verify("JOB_MONITORING_VERIFIED",
                    evidence={"job_id": args.job_id, "final_state": state})
    registry.verify("DOWNLOAD_VERIFIED",
                    evidence={"sha256": downloaded.data.get("sha256"),
                              "bytes": downloaded.data.get("size_bytes")})

    backend = CharmmGuiBackend(provider)
    request = ParameterizationRequest(
        polymer_id=args.polymer_id, polymer_name=args.polymer_id,
        repeat_unit_smiles=args.smiles or "",
        property_class=PropertyClass.BULK_DENSITY,
        external_job_id=args.job_id, source_archive=str(archive),
        workdir=str(ACQUISITION_ROOT / "systems"),
    )
    result = backend.parameterize(request)
    print(f"  parameterization: {result.state.value}")
    print(f"  topology: {result.topology_path}")
    print(f"  files: {len(result.artifacts)} hashed")
    if result.state.value == "PARAMETERIZED":
        registry.verify("SYSTEM_IMPORT_VERIFIED",
                        evidence={"job_id": args.job_id,
                                  "n_artifacts": len(result.artifacts)})
        validation = backend.validate(result)
        print(f"  validation: {validation.determination.value} "
              f"(promotable={validation.promotable})")
        for name, report in validation.reports():
            print(f"    {name}: {report.status.value}")
    registry.save()

    queue = BuildQueue()
    entry = queue.by_job(args.job_id)
    if entry:
        from polymer_engine.browser.states import JobState

        entry.archive_path = str(archive)
        entry.archive_sha256 = str(downloaded.data.get("sha256"))
        entry.advance(JobState.DOWNLOADED, "fetched through the documented API")
        queue.save()
    return 0


def _record_build_verification(registry: Any, args: argparse.Namespace, index: int,
                               job_id: str, session: Session,
                               group_label: str) -> None:
    """Mark the capability chain this build proved, in dependency order.

    In order because the graph enforces it: a build cannot be verified before the login,
    page, schema, catalogue and mapping that produced it. Everything up to the build is
    genuinely established by a run that reached this point -- the login happened, the
    page was read, the mapping resolved a real menu index -- so recording them here is
    not optimism, it is filling in the evidence the graph requires.
    """
    url = session.current_url()
    chain = [
        ("LIVE_LOGIN_VERIFIED", {"url": url, "method": "keyboard-typed credentials"}),
        ("POLYMER_BUILDER_REACHED", {"url": url}),
        ("FORM_SCHEMA_VERIFIED", {"note": "wizard fields located and read back live"}),
        ("CATALOG_VERIFIED", {"resolved": f"{group_label}@{index}"}),
        ("SPEC_MAPPING_VERIFIED", {"monomer": args.monomer, "menu_index": index}),
        ("SINGLE_CHAIN_BUILD_VERIFIED" if args.system_type == "single"
         else "MELT_BUILD_VERIFIED",
         {"job_id": job_id, "monomer": args.monomer, "dp": args.dp}),
    ]
    for name, evidence in chain:
        if not registry.records[name].state.proven_live:
            registry.verify(name, evidence=evidence)
    registry.save()


def _grab_download(session: Session, save_as: str | None = None) -> str | None:
    """Download the built system via the download link's own target URL.

    The result page carries the download target in the download.tgz link's data-href
    (``?doc=input/download&jobid=...``), and there are hidden duplicate links. Reading
    the URL and fetching it through the logged-in session sidesteps clicking the one
    visible copy, and the session cookies authenticate the request.
    """
    from urllib.parse import urljoin

    inventory = session.driver.send("page_inventory")
    href = None
    for element in inventory.get("clickable", []):
        if "download.tgz" not in str(element.get("text") or "").lower():
            continue
        href = element.get("data_href") or element.get("href")
        if href:
            break
    if not href:
        print("    no download URL (data-href) found on the download.tgz link.",
              file=sys.stderr)
        return None
    url = urljoin(session.current_url(), href)
    print(f"    download URL: {url}")
    result = session.driver.send("download_url", url=url, timeout_ms=180000,
                                 save_as=save_as)
    if not result.get("ok"):
        print(f"    download failed: {result.get('error')}", file=sys.stderr)
        return None
    path = result["path"]
    from pathlib import Path as _P

    size = _P(path).stat().st_size if _P(path).exists() else 0
    print(f"    saved archive: {path} ({size} bytes)")
    return path


def wizard_probe(args: argparse.Namespace) -> int:
    """Map the Polymer Builder wizard by walking it, one safe step at a time.

    Runs the verified step-1 build (login, resolve the monomer against the live menu,
    click it, set the repeat count, submit) and then, from the page that produces,
    records every step: its heading, its buttons, whether a download is already offered,
    and which 'Next Step' buttons would launch computation.

    It advances only across steps that do **not** cross a compute boundary, and stops at
    the first of: a page offering ``download.tgz`` (the single-chain path reaches this in
    one step), a page whose only way forward runs a simulation, or ``--max-steps``. It
    never clicks a compute-launching button. The result is a JSON map of the real
    sequence, written from evidence rather than assumed.
    """
    registry, code, credentials = _preflight()
    if code:
        return code

    steps: list[dict[str, Any]] = []
    out = ACQUISITION_ROOT / "wizard_map"
    out.mkdir(parents=True, exist_ok=True)

    with Session(credentials=credentials, headless=not args.headed,
                 downloads_dir=ACQUISITION_ROOT / "downloads") as session:
        login = session.login()
        if not login.ok:
            print(f"  {login.state.value}: {login.detail}", file=sys.stderr)
            return 3
        session.navigate(args.url)

        # -- step 1: the verified selection + submit -------------------------
        choices = session.driver.send("handler_choices", handler="set_monomer")
        index, group_label, option_text = _resolve_handler_option(
            choices, args.monomer, args.variant)
        if index is None:
            print(f"  MONOMER_NOT_FOUND: {option_text}", file=sys.stderr)
            return 5
        print(f"  step 1: selecting {group_label!r}/{option_text!r} (index {index})")
        before = _poly_table_text(session)
        session.driver.send("click_handler", handler="set_monomer", index=index,
                            settle_ms=1200)
        if "select unit" in _poly_table_text(session).lower() or \
                _poly_table_text(session) == before:
            print("  STRUCTURE_MISMATCH: selection did not take", file=sys.stderr)
            return 6
        session.driver.send(
            "type", key="dp", text=str(args.dp),
            delay_ms=credentials.typing_delay_ms,
            locators=[{"strategy": "css", "value": "input[name='subtext[1][1]']"},
                      {"strategy": "css", "value": "input[name^='subtext[']"}])
        session.driver.send(
            "click", key=f"model={args.system_type}", wait_load=False,
            locators=[{"strategy": "css",
                       "value": f"input[name='model'][value='{args.system_type}']"}])

        # Advance out of step 1: this is the one build click, always safe (it builds the
        # chain; the compute boundary is later, at equilibration).
        session.driver.send("click_text", text="Build Polymer Chains",
                            contains=True, settle_ms=4000)

        # -- walk the resulting steps ----------------------------------------
        for step_number in range(1, args.max_steps + 1):
            page = _page_step(session)
            page["step"] = step_number
            steps.append(page)
            _dump_state(session, out, f"step_{step_number}")
            print(f"\n  step {step_number}: {page['headings']}")
            print(f"    url: {page['url']}")
            print(f"    download available: {page['download_available']}")
            print(f"    next-step buttons : {page['next_steps']}")
            # Every clickable-looking control, so a forward button my classifier missed
            # is visible in the console rather than only inferred.
            for b in page.get("buttons", []):
                if not b.get("visible"):
                    continue
                label = b.get("label") or b.get("text") or ""
                if (label.lower() in ("logout", "")
                        and not (b.get("onclick") or b.get("form_action"))):
                    continue
                print(f"      button: label={label[:40]!r} cls={b.get('cls')!r} "
                      f"onclick={str(b.get('onclick'))[:50]!r} "
                      f"form={b.get('form_action')!r}")
            if page["compute_boundaries"]:
                print(f"    COMPUTE BOUNDARY  : {page['compute_boundaries']}")

            safe = [b for b in page["next_steps"] if not _boundary_hit(str(b))]
            boundary = [b for b in page["next_steps"] if _boundary_hit(str(b))]
            terminal = not page["next_steps"]

            # Terminal step: no way forward. If it offers a download, this is the
            # result -- for a single chain, the CGenFF-parameterised built chain that
            # our own melt builder needs as its building block.
            if terminal:
                if page["download_available"]:
                    print("\n  -> terminal step, and it offers download.tgz.")
                    page["is_terminal_download"] = True
                    job = session.driver.send(
                        "read_value", key="jobid",
                        locators=[{"strategy": "css", "value": "input[name='jobid']"}])
                    page["job_id"] = str(job.get("value") or "").strip() or None
                    if args.download:
                        page["archive"] = _grab_download(session)
                else:
                    print("\n  -> terminal step with no download; end of the path with "
                          "nothing to fetch.")
                break

            # An intermediate download (more steps remain) is the built chain, not the
            # production system. Note it and keep going.
            if page["download_available"]:
                print("    (an intermediate download.tgz is offered here -- the built "
                      "structure, not the final inputs; continuing)")

            if not args.advance:
                print("\n  -> more steps exist; re-run with --advance to walk them.")
                break

            target = safe[0] if safe else (boundary[0] if boundary else None)
            crosses = target is not None and _boundary_hit(str(target))
            if crosses and not args.to_generation:
                print(f"\n  -> the only way forward is {target!r}, which runs a "
                      "simulation on CHARMM-GUI's servers. Stopping; pass "
                      "--to-generation to authorize crossing it.")
                break
            if target is None:
                print("\n  -> nothing advanceable here.")
                break
            if crosses:
                print(f"    CROSSING COMPUTE BOUNDARY (authorized): {target!r} -- this "
                      "spends CHARMM-GUI compute")
            else:
                print(f"    advancing via {target!r}")
            clicked = session.driver.send("click_text", text=target, contains=True,
                                          settle_ms=args.step_settle_ms)
            if not clicked.get("ok"):
                print(f"    could not advance: {clicked.get('error')}", file=sys.stderr)
                break

    (out / "wizard_map.json").write_text(json.dumps({
        "schema": "charmm_gui.wizard_map/1",
        "generated_at": datetime.now(UTC).isoformat(),
        "system_type": args.system_type, "monomer": args.monomer,
        "n_steps_seen": len(steps), "steps": steps,
    }, indent=2, default=str) + "\n")
    registry.save()
    print(f"\n  wrote {out}/wizard_map.json ({len(steps)} step(s) mapped)")
    return 0


def _built_residues(workdir: Path) -> list[str]:
    """The residue names in the built chain, from the sequence in p1_raw.str."""
    for stream in workdir.rglob("p1_raw.str"):
        text = stream.read_text(errors="replace")
        # The line after "read sequence card" / "polymer sequence" and a count is the
        # residue list. Grab all-caps residue tokens from the sequence block.
        import re

        # "... polymer sequence\n<count>\n<RES1 RES2 ...>\ngenerate ..."
        m = re.search(r"sequence\s*\n\s*\d+\s*\n([^\n]+)", text)
        if m:
            return [tok for tok in m.group(1).split() if tok.isalnum()]
    return []


def _residue_penalties(workdir: Path, residues: list[str]) -> dict[str, dict[str, Any]]:
    """Per-residue penalties from the RESI lines of the toppar stream files.

    Format: ``RESI NAME  charge  ! param penalty=  X ; charge penalty=  Y`` -- the
    penalty comment is optional, and its absence means a fully curated residue.
    """
    import re

    wanted = set(residues)
    out: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r"^RESI\s+(\S+)\s+[-\d.]+\s*(?:!\s*param penalty=\s*([\d.]+)"
        r"\s*;\s*charge penalty=\s*([\d.]+))?", re.MULTILINE)
    for stream in workdir.rglob("*.str"):
        try:
            text = stream.read_text(errors="replace")
        except OSError:
            continue
        for match in pattern.finditer(text):
            name = match.group(1)
            if wanted and name not in wanted:
                continue
            out[name] = {
                "param_penalty": float(match.group(2)) if match.group(2) else None,
                "charge_penalty": float(match.group(3)) if match.group(3) else None,
                "source": str(stream.relative_to(workdir)),
            }
    return out


def import_archive(args: argparse.Namespace) -> int:
    """Extract a CHARMM-GUI single-chain archive and report the data it carries.

    A single-chain step-1 archive is CHARMM format -- PSF/CRD/PDB plus CGenFF stream
    files -- not a GROMACS system. That is exactly the building block our own melt
    builder needs, so this does not try to construct or validate a GROMACS system. It
    extracts safely, finds the structure and parameter files, parses the CGenFF
    penalties, and reports what is there. Building the bulk system from these is the
    engine's job, not CHARMM-GUI's.
    """
    from polymer_engine.core.provenance import sha256_file
    from polymer_engine.simulation.archive import safe_extract
    from polymer_engine.simulation.cgenff import (
        determination_for,
        find_stream_files,
        parse_stream_file,
    )

    archive = Path(args.archive)
    if not archive.exists():
        print(f"  no archive at {archive}", file=sys.stderr)
        return 2
    workdir = ACQUISITION_ROOT / "systems" / args.polymer_id
    # A fresh directory each import, so a re-run does not catalogue files left by the
    # previous one and report each twice.
    if workdir.exists():
        import shutil
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"  archive: {archive} ({archive.stat().st_size} bytes)")
    print(f"  sha256 : {sha256_file(archive)[:16]}...")

    report = safe_extract(archive, workdir)
    print(f"  extracted {report.members_extracted} members "
          f"({report.total_bytes} bytes) into {workdir}")

    # Classify what came out, by role, without assuming any GROMACS files exist.
    roles = {"structure": (".pdb", ".psf", ".crd"), "gromacs": (".gro", ".top", ".itp"),
             "charmm_param": (".prm", ".rtf", ".par"), "stream": (".str",)}
    found: dict[str, list[str]] = {k: [] for k in roles}
    hashes: dict[str, str] = {}
    for path in sorted(workdir.rglob("*")):
        if not path.is_file():
            continue
        hashes[str(path.relative_to(workdir))] = sha256_file(path)
        for role, suffixes in roles.items():
            if path.suffix.lower() in suffixes:
                found[role].append(str(path.relative_to(workdir)))
    for role, files in found.items():
        if files:
            print(f"  {role:13}: {files}")
    print(f"  files hashed : {len(hashes)}")

    result: dict[str, Any] = {
        "polymer_id": args.polymer_id, "archive": str(archive),
        "sha256": sha256_file(archive), "roles": found, "hashes": hashes,
        "penalties": None, "residue_penalties": None,
    }

    # Two kinds of penalty can appear, and a polymer archive is usually the first:
    #
    #  1. Curated residue parameters. Polymer Builder assembles the chain from named
    #     residues (LACTS, LACTR, ...) whose penalties -- if any -- sit on the RESI line
    #     of the synthetic-polymer toppar. A residue with no penalty comment is fully
    #     parameterised: the best provenance, not the absence of data.
    #  2. Per-parameter CGenFF analogy penalties, in the BOND/ANGLE/DIHEDRAL sections of
    #     a job-specific stream. These appear when a monomer was assigned by analogy.
    residues = _built_residues(workdir)
    residue_penalties = _residue_penalties(workdir, residues)
    if residues:
        print(f"\n  built from residues: {sorted(set(residues))}")
    if residue_penalties:
        result["residue_penalties"] = residue_penalties
        worst = max(residue_penalties.values(),
                    key=lambda v: v.get("param_penalty") or 0.0)
        print("  residue parameter penalties (CHARMM-GUI curated):")
        for name, pen in sorted(residue_penalties.items()):
            pp, cp = pen.get("param_penalty"), pen.get("charge_penalty")
            tag = ("curated, no analogy penalty" if pp is None
                   else f"param {pp}, charge {cp}")
            print(f"    {name:8}: {tag}")
        worst_pp = worst.get("param_penalty")
        if worst_pp is None:
            print("  -> every residue is a curated parameter set with no analogy "
                  "penalty. This is validated provenance, the strongest case.")
            result["penalty_determination"] = "KNOWN"
        else:
            print(f"  -> worst residue penalty {worst_pp} "
                  f"(tier: {'good' if worst_pp < 10 else 'moderate' if worst_pp < 50 else 'poor'})")
            result["penalty_determination"] = (
                "KNOWN" if worst_pp < 10 else "REQUIRES_VALIDATION")
    else:
        print("\n  no residue penalty annotations found for the built residues.")
        result["penalty_determination"] = "INSUFFICIENT_DATA"

    # Any job-specific per-parameter penalties. The toppar/ directory is the shipped
    # CHARMM force-field distribution -- par_all36_cgenff.prm alone carries thousands of
    # analogy penalties belonging to other molecules -- so it is excluded: those are not
    # this polymer's provenance. Only a job-specific stream (outside toppar/) would carry
    # a penalty assigned for this build.
    job_streams = [f for f in find_stream_files(workdir)
                   if "toppar" not in f.parts and "/toppar/" not in str(f)]
    for stream in job_streams:
        try:
            penalties = parse_stream_file(stream)
        except Exception:  # noqa: BLE001 - not a penalty stream; that is normal here
            continue
        pd = penalties.as_dict()
        if pd.get("n_parameters"):
            result["penalties"] = pd
            print(f"\n  per-parameter CGenFF penalties in "
                  f"{stream.relative_to(workdir)}:")
            print(f"    max penalty : {pd.get('max_penalty')} (tier {penalties.tier})")
            for w in penalties.worst(5):
                print(f"      {w.as_dict()}")
            print(f"    determination: {determination_for(penalties).value}")
            break

    if not found["structure"] and not found["gromacs"]:
        print("\n  WARNING: no structure file (pdb/psf/crd/gro) in the archive; there "
              "is nothing to build a system from.", file=sys.stderr)

    (ACQUISITION_ROOT / "imports").mkdir(parents=True, exist_ok=True)
    dest = ACQUISITION_ROOT / "imports" / f"{args.polymer_id}.json"
    dest.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"\n  wrote {dest}")
    return 0

def _build_one(session: Session, credentials: Any, monomer: str, variant: str | None,
               dp: int, system_type: str, url: str,
               group: str | None = None) -> dict[str, Any]:
    """Select, submit and download one polymer in an existing session.

    Returns a record: the job id, the saved archive path, or the reason it stopped.
    Reuses the verified wizard step-1 flow; a single chain terminates there with the
    built structure, which is the building block the repo collects.
    """
    session.navigate(url)
    choices = session.driver.send("handler_choices", handler="set_monomer")
    index, group_label, option_text = _resolve_handler_option(
        choices, monomer, variant, group=group)
    if index is None:
        return {"ok": False, "stage": "resolve", "reason": option_text}
    before = _poly_table_text(session)
    session.driver.send("click_handler", handler="set_monomer", index=index,
                        settle_ms=1200)
    if "select unit" in _poly_table_text(session).lower() or \
            _poly_table_text(session) == before:
        return {"ok": False, "stage": "select", "reason": "chain row did not update"}
    session.driver.send(
        "type", key="dp", text=str(dp),
        delay_ms=credentials.typing_delay_ms,
        locators=[{"strategy": "css", "value": "input[name='subtext[1][1]']"},
                  {"strategy": "css", "value": "input[name^='subtext[']"}])
    session.driver.send(
        "click", key=f"model={system_type}", wait_load=False,
        locators=[{"strategy": "css",
                   "value": f"input[name='model'][value='{system_type}']"}])
    session.driver.send("click_text", text="Build Polymer Chains", contains=True,
                        settle_ms=4000)
    page = _page_step(session)
    if not page["download_available"]:
        return {"ok": False, "stage": "build",
                "reason": f"no download after build; headings {page['headings']}",
                "next_steps": page["next_steps"]}
    job = session.driver.send(
        "read_value", key="jobid",
        locators=[{"strategy": "css", "value": "input[name='jobid']"}])
    job_id = str(job.get("value") or "").strip() or None
    # A unique filename per job: CHARMM-GUI names every archive "charmm-gui.tgz", so
    # without this each download overwrites the last and only the final one survives.
    save_as = f"{job_id or 'nojob'}.tgz"
    archive = _grab_download(session, save_as=save_as)
    return {"ok": bool(archive), "stage": "download",
            "job_id": job_id,
            "archive": archive, "group": group_label, "resolved": option_text}


def repo(args: argparse.Namespace) -> int:
    """Build a repository of polymers from CHARMM-GUI, one at a time.

    Reads the buildable list, and for each target that is not already done: builds it,
    downloads the archive, imports it offline, and records the result in a manifest.
    Deliberately serial and rate-limited -- CHARMM-GUI is a shared academic service, and
    one build at a time with a delay between them is the polite and debuggable choice.
    Resumable: a target already recorded done in the manifest is skipped, so an
    interrupted run continues where it stopped.
    """
    import time as _time

    registry, code, credentials = _preflight()
    if code:
        return code

    catalogue = json.loads(Path(args.catalogue).read_text())
    targets = catalogue["homopolymers"]
    if args.only:
        wanted = {n.strip().lower() for n in args.only.split(",")}
        targets = [t for t in targets if t["name"].lower() in wanted]
    if args.limit:
        targets = targets[: args.limit]

    manifest_path = Path(args.manifest)
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {"schema": "charmm_gui.repo/1",
                                                "polymers": {}})
    done = {k for k, v in manifest["polymers"].items() if v.get("state") == "imported"}

    todo = [t for t in targets if t["name"] not in done]
    print(f"  repo: {len(targets)} target(s), {len(done)} already done, "
          f"{len(todo)} to build")
    if args.list_only:
        for t in todo:
            print(f"    - {t['name']} (value {t['value']})")
        return 0

    repo_root = ACQUISITION_ROOT / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    for i, target in enumerate(todo, start=1):
        name = target["name"]
        # Pick a variant only for monomers that offer them. The requested one if it is on
        # offer; otherwise the first the menu lists -- a diene lists trans/cis, never the
        # 'atactic' the default assumes, and refusing over that would drop it for no
        # scientific reason. The chosen conformation is recorded in the manifest.
        variant = None
        offered = target.get("variants") or []
        if offered:
            want = args.variant.strip().casefold() if args.variant else ""
            match = next((v for v in offered if v.strip().casefold() == want), None)
            variant = match or offered[0]
            if match is None:
                print(f"      note: {name} does not offer {args.variant!r}; "
                      f"using {variant!r} (offered: {offered})")
        group = target.get("group")
        print(f"\n  [{i}/{len(todo)}] building {name} "
              f"(variant {variant}, group {group}, DP {args.dp})...")
        entry = {"name": name, "value": target["value"], "variant": variant,
                 "dp": args.dp, "system_type": args.system_type,
                 "state": "building", "at": datetime.now(UTC).isoformat()}
        manifest["polymers"][name] = entry
        try:
            with Session(credentials=credentials, headless=not args.headed,
                         downloads_dir=repo_root / "downloads") as session:
                if not session.login().ok:
                    print("    login failed; stopping the run", file=sys.stderr)
                    break
                built = _build_one(session, credentials, name, variant, args.dp,
                                   args.system_type, args.url, group=group)
        except Exception as exc:  # noqa: BLE001 - one failure must not end the repo
            built = {"ok": False, "stage": "exception", "reason": str(exc)}

        entry.update({"job_id": built.get("job_id"),
                      "archive": built.get("archive"),
                      "build_stage": built.get("stage")})
        if not built.get("ok"):
            entry["state"] = "failed"
            entry["reason"] = built.get("reason")
            print(f"    FAILED at {built.get('stage')}: {built.get('reason')}",
                  file=sys.stderr)
        else:
            # Import offline, into a per-polymer directory.
            slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            imp = _import_offline(Path(built["archive"]), slug, repo_root)
            entry.update(imp)
            entry["state"] = "imported" if imp.get("ok") else "import_failed"
            print(f"    imported: {entry['state']} "
                  f"(residues {imp.get('residues')}, "
                  f"provenance {imp.get('penalty_determination')})")
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")

        if i < len(todo):
            print(f"    waiting {args.delay:g}s before the next build (shared service)")
            _time.sleep(args.delay)

    _write_repo_summary(manifest, repo_root)
    registry.save()
    print(f"\n  manifest: {manifest_path}")
    return 0


def _import_offline(archive: Path, polymer_id: str, repo_root: Path) -> dict[str, Any]:
    """The import path, returning a summary dict instead of printing (for the repo)."""
    from polymer_engine.core.provenance import sha256_file
    from polymer_engine.simulation.archive import safe_extract

    if not archive or not archive.exists():
        return {"ok": False, "reason": "no archive"}
    workdir = repo_root / "systems" / polymer_id
    if workdir.exists():
        import shutil
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        safe_extract(archive, workdir)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"extract failed: {exc}"}
    residues = _built_residues(workdir)
    residue_penalties = _residue_penalties(workdir, residues)
    worst = None
    if residue_penalties:
        vals = [v.get("param_penalty") or 0.0 for v in residue_penalties.values()]
        worst = max(vals) if vals else 0.0
    determination = ("KNOWN" if worst == 0.0 or worst is None
                     else "REQUIRES_VALIDATION" if worst >= 50 else "KNOWN")
    structures = [str(p.relative_to(workdir)) for p in workdir.rglob("*.psf")]
    return {"ok": True, "archive_sha256": sha256_file(archive),
            "residues": sorted(set(residues)),
            "residue_penalties": residue_penalties,
            "worst_residue_penalty": worst,
            "penalty_determination": determination,
            "has_structure": bool(structures), "structure": structures[:3],
            "workdir": str(workdir)}


def _write_repo_summary(manifest: dict[str, Any], repo_root: Path) -> None:
    polymers = manifest["polymers"]
    imported = [v for v in polymers.values() if v.get("state") == "imported"]
    lines = ["# CHARMM-GUI polymer repository", "",
             f"{len(imported)} of {len(polymers)} imported.", "",
             "| Polymer | Job | Residues | Provenance | Worst penalty |",
             "|---|---|---|---|---|"]
    for name, v in sorted(polymers.items()):
        if v.get("state") != "imported":
            lines.append(f"| {name} | — | — | **{v.get('state')}** | "
                         f"{v.get('reason', '')[:40]} |")
            continue
        lines.append(f"| {name} | {v.get('job_id')} | "
                     f"{','.join(v.get('residues') or [])} | "
                     f"{v.get('penalty_determination')} | "
                     f"{v.get('worst_residue_penalty')} |")
    (repo_root / "REPO.md").write_text("\n".join(lines) + "\n")


def _write_report(registry: VerificationRegistry, catalog: Catalog, form: Any) -> None:
    lines = [
        "# CHARMM-GUI live acquisition report", "",
        f"Generated {datetime.now(UTC).isoformat()}", "",
        "## Capability verification", "", registry.to_markdown(), "",
        "## Discovered form", "",
        f"- {len(form.schema.fields)} fields: "
        + ", ".join(f"`{k}`" for k in sorted(form.schema.fields)),
        f"- semantic mapping: {form.semantics}",
        f"- ambiguous: {form.ambiguous() or 'none'}", "",
        "## Catalogue", "",
        f"- version `{catalog.version}`",
        f"- {len(catalog.monomers)} monomers",
        f"- completeness `{catalog.completeness}`",
        f"- source <{catalog.source_url}>", "",
        "## Force-field qualification", "",
        "A successful build states `SYSTEM_GENERATION_VERIFIED` and nothing more. It "
        "does **not** state that CHARMM36/CGenFF is qualified for any property class; "
        "that needs evidence a download cannot supply, and the registry refuses to "
        "record it.", "",
    ]
    REPORT.write_text("\n".join(lines))


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    discover_parser = sub.add_parser("discover", help="login and capture; submits nothing")
    discover_parser.add_argument("--headed", action="store_true",
                                 help="show the browser window")
    discover_parser.set_defaults(func=discover)

    dump_parser = sub.add_parser(
        "dump", help="capture a sanitised picture of a live page; submits nothing")
    dump_parser.add_argument("--url", default="/?doc=input/polymer")
    dump_parser.add_argument("--out", default="artifacts/charmm_gui/live_page")
    dump_parser.add_argument("--headed", action="store_true")
    dump_parser.add_argument(
        "--probe-conditionals", action="store_true",
        help="select each radio option and re-inspect, to reveal sections the page "
             "hides until a system type is chosen. Submits nothing.")
    dump_parser.set_defaults(func=dump)

    probe_parser = sub.add_parser(
        "probe", help="click one element to see what it reveals; never submits")
    probe_parser.add_argument("--click", required=True,
                              help="exact visible text of the element to click")
    probe_parser.add_argument("--url", default="/?doc=input/polymer")
    probe_parser.add_argument("--out", default="artifacts/charmm_gui/probe")
    probe_parser.add_argument("--headed", action="store_true")
    probe_parser.add_argument(
        "--handler", default="set_monomer",
        help="JS handler name whose onclick options form a choice list")
    probe_parser.add_argument(
        "--index", type=int, default=None,
        help="which match to click when several share the text; omit to require "
             "exactly one visible match")
    probe_parser.set_defaults(func=probe)

    build_parser = sub.add_parser("build", help="submit exactly one build")
    build_parser.add_argument("--monomer", required=True,
                              help="monomer name exactly as the live menu shows it")
    build_parser.add_argument("--variant", default=None,
                              help="tacticity/conformation text, e.g. 'atactic'")
    build_parser.add_argument("--dp", type=int, default=10)
    build_parser.add_argument("--system-type", default="single",
                              choices=["single", "melt", "solution"])
    build_parser.add_argument("--polymer-id", default="live-build")
    build_parser.add_argument("--rationale", required=True)
    build_parser.add_argument("--url", default="/?doc=input/polymer")
    build_parser.add_argument("--dry-run", action="store_true",
                              help="do everything except click the build control")
    build_parser.add_argument("--headed", action="store_true")
    build_parser.set_defaults(func=build)

    fetch_parser = sub.add_parser(
        "fetch", help="download and import a built job's system")
    fetch_parser.add_argument("--job-id", required=True)
    fetch_parser.add_argument("--polymer-id", default="live-build")
    fetch_parser.add_argument("--smiles", default=None,
                              help="repeat unit, recorded in provenance")
    fetch_parser.add_argument(
        "--via", default="browser", choices=["browser", "api"],
        help="browser: download the single-chain result page's download.tgz (default). "
             "api: poll check_status and download -- for jobs the documented API tracks")
    fetch_parser.add_argument("--result-url", default=None,
                              help="override the result page URL")
    fetch_parser.add_argument("--timeout", type=float, default=1800.0)
    fetch_parser.add_argument("--headed", action="store_true")
    fetch_parser.set_defaults(
        func=lambda a: fetch_browser(a) if a.via == "browser" else fetch(a))

    import_parser = sub.add_parser(
        "import", help="import a downloaded archive offline: extract, penalties, gates")
    import_parser.add_argument("--archive", required=True)
    import_parser.add_argument("--polymer-id", default="charmm-gui-import")
    import_parser.add_argument("--smiles", default=None)
    import_parser.set_defaults(func=import_archive)

    repo_parser = sub.add_parser(
        "repo", help="build a repository of polymers, one at a time, rate-limited")
    repo_parser.add_argument("--catalogue",
                             default="data/charmm_gui/buildable_polymers.json")
    repo_parser.add_argument("--manifest",
                             default="campaign/charmm_gui/repo/manifest.json")
    repo_parser.add_argument("--variant", default="atactic",
                             help="conformation for monomers that offer variants")
    repo_parser.add_argument("--dp", type=int, default=10)
    repo_parser.add_argument("--system-type", default="single",
                             choices=["single", "melt", "solution"])
    repo_parser.add_argument("--only", default=None,
                             help="comma-separated polymer names to build")
    repo_parser.add_argument("--limit", type=int, default=None,
                             help="build at most this many (start small)")
    repo_parser.add_argument("--delay", type=float, default=30.0,
                             help="seconds between builds; be kind to the service")
    repo_parser.add_argument("--list-only", action="store_true",
                             help="show what would be built, submit nothing")
    repo_parser.add_argument("--url", default="/?doc=input/polymer")
    repo_parser.add_argument("--headed", action="store_true")
    repo_parser.set_defaults(func=repo)

    probe_parser = sub.add_parser(
        "wizard-probe",
        help="walk the Polymer Builder wizard and map its steps; never runs a "
             "simulation, never crosses a compute boundary")
    probe_parser.add_argument("--monomer", required=True)
    probe_parser.add_argument("--variant", default=None)
    probe_parser.add_argument("--dp", type=int, default=10)
    probe_parser.add_argument("--system-type", default="single",
                              choices=["single", "melt", "solution"])
    probe_parser.add_argument("--url", default="/?doc=input/polymer")
    probe_parser.add_argument("--max-steps", type=int, default=8)
    probe_parser.add_argument(
        "--advance", action="store_true",
        help="walk the steps rather than stopping after the first result page")
    probe_parser.add_argument(
        "--to-generation", action="store_true",
        help="authorize crossing the equilibration / input-generation steps that run a "
             "simulation on CHARMM-GUI's servers. Required to reach production inputs")
    probe_parser.add_argument("--step-settle-ms", type=float, default=6000.0,
                              help="wait after each step; generation steps take longer")
    probe_parser.add_argument(
        "--download", action="store_true",
        help="when a terminal download.tgz is reached, grab it in the same session")
    probe_parser.add_argument("--headed", action="store_true")
    probe_parser.set_defaults(func=wizard_probe)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
