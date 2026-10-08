"""Reading the Polymer Builder page and turning it into a catalogue and a form schema.

Everything here derives from the live DOM. Nothing is transcribed from a tutorial, a
screenshot or a previous release, because a stale selector that still matches *some*
control is more dangerous than one that matches none: it fills the wrong field and
submits a different polymer than the one requested.

Navigation follows the same rule. :func:`find_polymer_builder` walks the visible links
of the authenticated site looking for the Input Generator's Polymer Builder entry rather
than jumping to a hard-coded deep URL, so a moved page is *reported* as not found
instead of turning into a 404 that gets mistaken for a build failure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from polymer_engine.browser.catalog import Catalog, CatalogState, MonomerEntry
from polymer_engine.browser.selectors import ControlKind, FieldSpec, FormSchema, Locator
from polymer_engine.browser.session import ActionResult, Session
from polymer_engine.browser.states import BrowserState
from polymer_engine.core.logging import get_logger

logger = get_logger("browser.discovery")

#: Link text that identifies the Polymer Builder entry point. Matched against what the
#: page actually offers; if none of these appear, discovery reports that rather than
#: navigating somewhere hopeful.
BUILDER_LINK_HINTS: tuple[str, ...] = ("polymer builder", "polymer")
GENERATOR_LINK_HINTS: tuple[str, ...] = ("input generator", "input")

#: Words in a control's label or name that suggest what it configures. Used only to
#: *annotate* a discovered field with a guess at its meaning -- never to select it. The
#: semantic mapping a workflow relies on comes from :func:`map_semantics` and is
#: reviewable, not from a substring match at submission time.
SEMANTIC_HINTS: dict[str, tuple[str, ...]] = {
    "monomer": ("monomer", "repeat unit", "residue", "unit type"),
    "degree_of_polymerization": ("degree of polymer", "dp", "number of repeat",
                                 "chain length", "n_repeat", "nrepeat", "polymer length"),
    "n_chains": ("number of chain", "nchain", "n_chain", "chains", "number of molecule"),
    "tacticity": ("tacticity", "tactic", "stereochem"),
    "system_type": ("system type", "build type", "single chain", "melt", "solution"),
    "temperature": ("temperature", "temp"),
    "density": ("density",),
    "box": ("box", "edge length", "cell size"),
    "solvent": ("solvent", "water model", "solvation"),
    "composition": ("composition", "fraction", "ratio", "mole percent"),
    "sequence": ("sequence", "pattern", "blockiness"),
    "terminal": ("terminal", "end group", "capping", "patch"),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def find_polymer_builder(session: Session) -> ActionResult:
    """Navigate to Polymer Builder through the visible interface.

    Follows links the page actually offers. A page that no longer exposes the module
    yields ``NAVIGATION_FAILED`` with the links that *were* found, which is a diagnosis;
    guessing a URL would only produce an error further downstream.
    """
    landing = session.navigate("/")
    if not landing.ok:
        return landing

    for hints in (BUILDER_LINK_HINTS, GENERATOR_LINK_HINTS):
        response = session.driver.send("links")
        links = response.get("links", []) if response.get("ok") else []
        match = _best_link(links, hints)
        if match is None:
            continue
        moved = session.navigate(match["href"])
        if not moved.ok:
            return moved
        if hints is BUILDER_LINK_HINTS:
            return ActionResult(BrowserState.OK, f"reached {match['href']}",
                                {"via": match["text"], "url": session.current_url()})
        # Landed on the Input Generator index; look for Polymer Builder from here.
        response = session.driver.send("links")
        inner = _best_link(response.get("links", []), BUILDER_LINK_HINTS)
        if inner is not None:
            moved = session.navigate(inner["href"])
            if moved.ok:
                return ActionResult(BrowserState.OK, f"reached {inner['href']}",
                                    {"via": inner["text"], "url": session.current_url()})

    response = session.driver.send("links")
    seen = [link.get("text", "")[:60] for link in response.get("links", [])][:40]
    return ActionResult(
        BrowserState.NAVIGATION_FAILED,
        "no visible link to Polymer Builder was found from the authenticated landing "
        "page; the module may have moved or may require a different entry point",
        {"links_seen": seen, "url": session.current_url()},
    )


def _best_link(links: list[dict[str, Any]], hints: tuple[str, ...]) -> dict[str, Any] | None:
    for hint in hints:
        for link in links:
            text = (link.get("text") or "").strip().lower()
            if hint in text:
                return link
    for hint in hints:
        for link in links:
            if hint.replace(" ", "") in (link.get("href") or "").lower():
                return link
    return None


def discover_form_schema(
    session: Session, *, name: str = "polymer_builder",
) -> tuple[FormSchema, list[dict[str, Any]]]:
    """Read every form control on the current page into a schema (§20).

    Each control becomes a :class:`FieldSpec` whose locators are built from what the DOM
    actually provides -- test id, ``id``, ``name``, accessible label -- in that order of
    preference. No coordinates are recorded, because a pixel position encodes a window
    size and not a meaning.
    """
    response = session.driver.send("form_fields")
    if not response.get("ok"):
        return FormSchema(name=name, discovered=False), []
    controls: list[dict[str, Any]] = response.get("controls", [])
    schema = FormSchema(name=name, discovered=True,
                        source_url=response.get("url"), discovered_at=_now())
    skipped: dict[str, int] = {}

    for control in controls:
        if control.get("type") == "hidden" or not control.get("visible"):
            skipped["hidden_or_invisible"] = skipped.get("hidden_or_invisible", 0) + 1
            continue
        key = _field_key(control)
        if key is None:
            # No name, id, test id or label. Dropping it would make the control
            # invisible to everything downstream -- which is how a monomer selector
            # with no name becomes "zero monomers". Keyed positionally instead, and
            # marked so nothing treats a positional match as solid.
            key = _positional_key(control)
        if key in schema.fields:
            # A repeated structure -- one row per building block, say. Both matter, so
            # the later one is disambiguated rather than discarded.
            key = f"{key}#{control.get('index')}"
        locators = _locators_for(control)
        if not locators:
            skipped["no_usable_locator"] = skipped.get("no_usable_locator", 0) + 1
            continue
        schema.add(FieldSpec(
            key=key,
            description=(control.get("label") or control.get("name")
                         or control.get("id") or "").strip()[:200],
            locators=tuple(locators),
            required=bool(control.get("required")),
            control=_control_kind(control),
        ))

    logger.info("discovered %d of %d controls on %s (skipped: %s)",
                len(schema.fields), len(controls), response.get("url"),
                skipped or "none")
    if skipped.get("no_usable_locator"):
        logger.warning("%d control(s) had no usable locator and are absent from the "
                       "schema", skipped["no_usable_locator"])
    return schema, controls


def _positional_key(control: dict[str, Any]) -> str:
    """A stable-within-this-capture key for a control with no identifier of its own."""
    return f"{control.get('tag', 'control')}@{control.get('index')}"


def _field_key(control: dict[str, Any]) -> str | None:
    for candidate in (control.get("test_id"), control.get("name"), control.get("id")):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    label = (control.get("label") or "").strip()
    if label:
        return "label:" + label[:60]
    return None


def _locators_for(control: dict[str, Any]) -> list[Locator]:
    locators: list[Locator] = []
    if control.get("test_id"):
        locators.append(Locator("test_id", str(control["test_id"])))
    tag = control.get("tag", "input")
    if control.get("id"):
        locators.append(Locator("css", f"{tag}#{_css_escape(str(control['id']))}"))
    if control.get("name"):
        locators.append(Locator("css", f"{tag}[name='{control['name']}']"))
    label = (control.get("label") or "").strip()
    if label and len(label) < 80:
        locators.append(Locator("label", label))
    # Last resort, and only when nothing else identifies the control: a computed CSS
    # path. Positional, so it breaks when the page's structure shifts -- which is why
    # it goes last and why a workflow re-verifies every field it fills.
    if not locators and control.get("css_path"):
        locators.append(Locator("css", str(control["css_path"])))
    return locators


def _css_escape(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else f"\\{ch}" for ch in value)


def _control_kind(control: dict[str, Any]) -> ControlKind:
    tag = str(control.get("tag") or "")
    kind = str(control.get("type") or "").lower()
    if tag == "select":
        return "select"
    if tag == "button" or kind in {"submit", "button"}:
        return "button"
    if kind == "checkbox":
        return "checkbox"
    if kind == "radio":
        return "radio"
    if kind == "number":
        return "number"
    if tag == "textarea" or kind in {"text", "email", ""}:
        return "text"
    return "any"


def map_semantics(controls: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Suggest which discovered controls correspond to which scientific quantity.

    This is a *suggestion for a person to confirm*, not a binding. It is returned
    separately from the schema and written into the schema file as ``semantic_hints`` so
    that a human can check the mapping once, rather than having every submission depend
    on a substring match made at the last moment.
    """
    suggestions: dict[str, list[str]] = {}
    for control in controls:
        # A button's label is an action, not a quantity. "Next Step: Build Polymer
        # Chains" contains the word "chains" and is not the chain count; matching it
        # would map a scientific field onto a submit control.
        if _control_kind(control) == "button":
            continue
        haystack = " ".join(str(control.get(k) or "").lower()
                            for k in ("label", "name", "id", "placeholder"))
        if not haystack.strip():
            continue
        key = _field_key(control)
        if key is None:
            continue
        for semantic, needles in SEMANTIC_HINTS.items():
            if any(needle in haystack for needle in needles):
                suggestions.setdefault(semantic, []).append(key)
    return {k: sorted(set(v)) for k, v in sorted(suggestions.items())}


def choice_sets(
    controls: list[dict[str, Any]], inventory: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Every way this page offers a list of things to pick from.

    A ``<select>`` is one way. A radio group is another, a checkbox group a third, a
    ``<datalist>`` a fourth. The original version of this function looked only at
    ``<select>`` and therefore reported "0 monomers" for any page that used a different
    control -- a statement about our code that reads as a statement about the site's
    chemistry.

    Returns ``{key: {kind, label, options}}`` where each option is
    ``{value, text}``. Nothing here decides which set is the monomer list.
    """
    sets: dict[str, dict[str, Any]] = {}

    for control in controls:
        if control.get("tag") != "select" or not control.get("options"):
            continue
        key = _field_key(control)
        if key is None:
            continue
        options = [{"value": str(o.get("value", "")), "text": str(o.get("text", "")).strip()}
                   for o in control["options"] if _real_option(o)]
        if options:
            sets[key] = {"kind": "select", "label": control.get("label"),
                         "options": options}

    if inventory:
        for group in inventory.get("radio_checkbox_groups", []):
            options = [
                {"value": str(o.get("value", "")),
                 "text": str(o.get("label") or o.get("value") or "").strip()}
                for o in group.get("options", [])
                if _real_option({"value": o.get("value"),
                                 "text": o.get("label") or o.get("value") or ""})
            ]
            # A group of one is a toggle, not a catalogue.
            if len(options) > 1:
                sets.setdefault(str(group.get("name") or "?"),
                                {"kind": group.get("kind", "radio"),
                                 "label": None, "options": options})
        for datalist in inventory.get("datalists", []):
            options = [{"value": str(o.get("value", "")), "text": str(o.get("text", "")).strip()}
                       for o in datalist.get("options", []) if _real_option(o)]
            if options:
                sets.setdefault(f"datalist:{datalist.get('id')}",
                                {"kind": "datalist", "label": None, "options": options})
    return sets


def monomers_from_handler_choices(choices: dict[str, Any]) -> list[MonomerEntry]:
    """Turn an onclick-driven choice list into catalogue entries.

    The live Polymer Builder presents two shapes under one mechanism, and telling them
    apart is the whole job:

    ``Polystyrene`` -> isotactic (R), isotactic (S), syndio (R), atactic
        One polymer offered in four tacticities. Every option carries the same form
        value, ``STYR``.

    ``Olefins`` -> Polyethylene, Polyisobutane
        A chemical *class* holding two different polymers, each with its own value.

    Reading the second shape as variants would drop Nylon 6, poly(ethylene
    terephthalate), PTFE, polyketone and poly(ethylene oxide) from the catalogue
    entirely, while leaving a plausible-looking result behind.

    The rule comes from the page's own data rather than a list of words we think mean
    tacticity: **options sharing one value are variants of one monomer; options with
    distinct values are distinct monomers.** A wordlist would need updating whenever
    CHARMM-GUI adds a conformation; the values cannot drift out of step with themselves.

    Every menu also appears twice -- once live, once inside the hidden skeleton that is
    cloned per chain row -- so options are deduplicated by ``(value, text)``.
    """
    # How many groups each option text appears in. A conformation label -- "atactic",
    # "cis" -- recurs across dozens of groups; a polymer name occurs in exactly one.
    # This is the tie-breaker for a group holding a single option, where the value rule
    # alone cannot say whether that option is a conformation or the only member of a
    # chemical class. Without it, a class with one member collapses into its class name
    # and the polymer inside it disappears -- which is how "Halides" swallowed
    # polytetrafluoroethylene.
    text_groups: dict[str, set[str]] = {}
    for group in choices.get("groups", []):
        gname = str(group.get("group") or "")
        for option in group.get("options", []):
            text = str(option.get("text") or "").strip()
            if text:
                text_groups.setdefault(text, set()).add(gname)

    def is_conformation(text: str) -> bool:
        return len(text_groups.get(text, set())) > 1

    entries: list[MonomerEntry] = []
    for group in choices.get("groups", []):
        name = str(group.get("group") or "").strip()
        if not name or name == "(ungrouped)" or name.startswith("#"):
            # No name above the group means we cannot say what these belong to, and a
            # variant list with no monomer is not a catalogue entry.
            continue

        seen: set[tuple[str, str]] = set()
        options: list[tuple[str, str]] = []
        for option in group.get("options", []):
            value = str(option.get("value") or "").strip()
            text = str(option.get("text") or "").strip()
            if not text:
                continue
            key = (value, text)
            if key in seen:
                continue          # the skeleton clone of a menu already counted
            seen.add(key)
            options.append(key)
        if not options:
            continue

        values = {value for value, _text in options if value}
        # A single option whose text names something else entirely is a class of one,
        # not a conformation of the group.
        lone_member = (len(options) == 1
                       and not is_conformation(options[0][1])
                       and options[0][1] != name)
        if len(values) <= 1 and not lone_member:
            # One monomer, offered in several conformations.
            entries.append(MonomerEntry(
                value=next(iter(values), name), label=name, control="set_monomer",
                kind="handler", variants=tuple(text for _v, text in options),
                notes="selected by clicking; conformations share one form value",
            ))
        else:
            # A class of distinct polymers, or a class holding exactly one. Either
            # way the option text names the polymer.
            for value, text in options:
                entries.append(MonomerEntry(
                    value=value or text, label=text, control="set_monomer",
                    kind="handler", variants=(),
                    notes=f"listed under the {name!r} group; distinct form value",
                ))
    return entries


def catalog_from_controls(
    controls: list[dict[str, Any]], *, source_url: str = "",
    semantics: dict[str, list[str]] | None = None,
    inventory: dict[str, Any] | None = None,
    handler_choices: dict[str, Any] | None = None,
) -> Catalog:
    """Build a catalogue from whatever control the page uses to list monomers.

    Placeholder options -- an empty value, or a label like "-- choose --" -- are dropped,
    because offering one as a buildable monomer would be inventing a capability.

    The distinction the result carries is the important part. ``CATALOG_CAPTURED`` means
    a monomer control was found and read. ``CATALOG_EMPTY`` means one was found and
    genuinely offers nothing. ``CATALOG_NOT_IDENTIFIED`` means none was found -- which is
    a defect in this function, never evidence about the site's chemistry.
    """
    semantics = semantics or {}
    monomer_controls = set(semantics.get("monomer", []))
    catalog = Catalog(source_url=source_url, completeness="unknown")
    sets = choice_sets(controls, inventory)

    catalog.inspected = {
        "n_controls": len(controls),
        "n_choice_sets": len(sets),
        "choice_sets": {k: {"kind": v["kind"], "n_options": len(v["options"])}
                        for k, v in sorted(sets.items())},
        "semantic_monomer_candidates": sorted(monomer_controls),
        "inventory_available": inventory is not None,
    }

    other: dict[str, list[str]] = {}
    identified = False

    # A handler-driven list is a monomer list in its own right, and on this page it is
    # the only one. It needs no semantic mapping, because the handler name says what it
    # selects.
    if handler_choices and handler_choices.get("n_nodes"):
        handler_entries = monomers_from_handler_choices(handler_choices)
        if handler_entries:
            catalog.monomers += handler_entries
            identified = True
            catalog.inspected["handler"] = {
                "name": handler_choices.get("handler"),
                "n_nodes": handler_choices.get("n_nodes"),
                "n_groups": len(handler_choices.get("groups", [])),
                "n_named_groups": len(handler_entries),
            }

    for key, entry in sets.items():
        if key in monomer_controls:
            identified = True
            catalog.monomers += [
                MonomerEntry(value=o["value"], label=o["text"], control=key,
                             kind=entry["kind"])
                for o in entry["options"]
            ]
        else:
            other[key] = [o["text"] for o in entry["options"]]

    for semantic, target in (("tacticity", "tacticity_options"),
                             ("system_type", "system_modes")):
        for key in semantics.get(semantic, []):
            if key in other:
                setattr(catalog, target, other.pop(key))
    catalog.other_options = other

    if identified and catalog.monomers:
        catalog.state = CatalogState.CAPTURED
    elif identified:
        catalog.state = CatalogState.EMPTY
        catalog.notes = (
            "A monomer control was identified but offers no selectable options."
        )
    else:
        catalog.state = CatalogState.NOT_IDENTIFIED
        catalog.notes = (
            "No control on this page could be identified as the monomer selector, so "
            "no monomers were recorded. This describes our discovery, not CHARMM-GUI: "
            "it does not mean the Polymer Builder offers no monomers, only that we did "
            f"not find where it lists them. {len(sets)} choice set(s) were seen "
            f"({', '.join(sorted(sets)) or 'none'}); none was mapped to 'monomer'."
        )
        logger.warning("monomer control not identified; %d choice set(s) seen: %s",
                       len(sets), sorted(sets))
    return catalog


def _real_option(option: dict[str, Any]) -> bool:
    value = str(option.get("value", "")).strip()
    text = str(option.get("text", "")).strip().lower()
    if not value:
        return False
    return not (text.startswith("--") or text in {"", "select", "choose", "none",
                                                  "select one", "please select"})


__all__ = [
    "BUILDER_LINK_HINTS", "GENERATOR_LINK_HINTS", "SEMANTIC_HINTS",
    "catalog_from_controls", "choice_sets", "discover_form_schema",
    "find_polymer_builder", "map_semantics", "monomers_from_handler_choices",
]
