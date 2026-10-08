# CHARMM-GUI browser automation

CHARMM-GUI publishes three API endpoints — `login`, `check_status`, `download` — and no
way to create a job. Polymer Builder is not mentioned in the API documentation at all.
So the build itself happens where CHARMM-GUI intends it to happen: in the web interface,
through a normal authenticated browser session.

This subsystem automates that session. It is **automation around CHARMM-GUI, not
automation against it**: it fills the visible form, it reads the visible page, and it
uses the documented API for the two things the API documents.

## The two interfaces, kept apart

| Task | Interface | Why |
|---|---|---|
| Login, catalogue discovery, form filling, submission | Browser | No endpoint exists |
| Job status | `GET /api/check_status?jobid=…` | Documented |
| Job download | `GET /api/download?jobid=…` | Documented |

No Polymer Builder endpoint is invented, no private request is reverse-engineered, and
the browser is never described as an API. `CHARMMGUIProvider.submit_module` still raises
`UnsupportedCapability`, because upstream still publishes no submission endpoint.

## Install

Playwright and its Chromium live in `.browserenv`, a third environment alongside `.venv`
(the engine) and `.paramenv` (OpenFF). A running campaign's environment never changes.

```bash
python3 -m venv .browserenv && .browserenv/bin/pip install playwright && PLAYWRIGHT_BROWSERS_PATH=.browserenv/browsers .browserenv/bin/playwright install chromium
```

```bash
polymer-engine browser check
```

## Credentials

Read from the environment, and from nowhere else:

```bash
export CHARMM_GUI_EMAIL='you@example.edu'
export CHARMM_GUI_PASSWORD='…'
```

There is no `--password` flag, no config key and no prompt. `CHARMM_GUI_TYPING_DELAY_MS`
tunes the keystroke interval (default 40 ms, clamped to 0–1000).

The password is wrapped in `Secret`, so interpolating it into a log line or an f-string
yields a mask. It is registered with the log redactor the moment it is read. And it is
**never serialised**: the browser worker reads it from its own inherited environment, so
it never appears in a protocol message at all. The only thing that crosses the process
boundary is the *name* of the variable.

### Why the password is typed, not filled

CHARMM-GUI's login form may not accept clipboard-pasted input, and `fill()` sets a
value the way a paste does rather than the way a person does. `type_secret` clicks the
field, clears it with Select-All and Delete, and calls `keyboard.type(value, delay=…)`,
which dispatches real keydown/keypress/keyup events for every character.

No clipboard is touched anywhere in the flow: there is no `navigator.clipboard` call, no
`Control+V`, and `fill()` appears on no path. The worker reports back how many
characters the field ended up holding — enough to detect a field that truncated the
input, and not the input itself.

The regression test for this runs a real browser against a fixture that **rejects paste
events and counts keydowns**, and only signs in after more than five real keystrokes. A
`fill()`-based implementation cannot pass it.

## What is discovered rather than assumed

Only *universal HTML structure* is hard-coded: an `input[type=password]` is a password
field on every site on the web, guaranteed by the platform. Nothing specific to Polymer
Builder is written into source.

`selectors.builder_schema_placeholder()` returns an **empty** schema. Every Polymer
Builder field, option and control type comes from `data/charmm_gui/builder_form_schema.json`,
captured from the live page. Asking an undiscovered schema for a field raises rather than
guessing, and a workflow that needs a field the schema lacks returns `UI_SCHEMA_MISMATCH`.

That strictness is the point. A selector transcribed from a tutorial and gone stale is
worse than one that matches nothing: it still matches *something*, fills the wrong box,
and submits a job that builds a different polymer while looking entirely successful.

```bash
polymer-engine browser discover        # login → Polymer Builder → capture
```

writes `monomer_catalog.json`, `monomer_catalog.md` and `builder_form_schema.json`, and
diffs against the previous capture — reporting added, removed and **relabelled**
monomers separately, because a form value whose label changed would silently resolve
differently for anything matching by name.

A catalogue never reports `completeness: complete`. A page shows what it shows; whether
that is everything CHARMM-GUI supports is not observable from the page.

### The submit control must be inside the form

The live CHARMM-GUI sign-in page carries a navigation control *outside* the login form:

```html
<button onclick="window.location.href='./?doc=sign'">Login</button>
```

while the form's own submit is `<input type="submit" value="Submit">`. The first
version of this schema matched a button by the word "login" and clicked the navigation
control, which reloaded the page and discarded everything typed. The result was a login
that failed with **correct credentials**, leaving the form present — from the outside,
indistinguishable from a rejected password.

Submit locators are now scoped to `form:has(input[type='password'])` and ordered
structurally: `input[type=submit]` and `button[type=submit]` inside the login form
first, accessible names last. `FieldSpec.strict_order` exists so that ordering is
honoured rather than re-sorted by strategy preference — which is right for a generated
schema and wrong for a hand-written one, where the author knows a structural match beats
a match on a word.

`tests/browser/fixtures/login_navbutton.html` reproduces the page structure, and the
regression test fails without the fix.

### Pages that hide most of themselves

The live Polymer Builder reports 27 form controls, 16 of them hidden, alongside a
"System Type:" radio group. A page built that way shows a fraction of what it offers
until something is chosen, so a single snapshot of the initial state can conclude there
is no monomer control when there is one behind a radio button.

```bash
.venv/bin/python scripts/charmm_gui_live.py dump --probe-conditionals
```

selects each option of each radio group in turn and re-inspects, recording what appeared
into `conditional_sections.json`. Selecting a radio is ordinary form interaction: it
reveals sections and creates nothing on the server. A regression test asserts that every
click the probe issues targets a radio input by name and value, and that none can reach
a submit control.

### What the live page turned out to look like

Recorded because it took three dumps to establish, and because none of it was
guessable from documentation.

The authenticated Polymer Builder page carries 27 form controls, 11 visible. Its four
`<select>` elements are **all end-capping groups**, not monomers:

| Control | Options |
|---|---|
| `capf[1]` (label "Polymer chain 1:") | `H-`, `H₃C-`, `HO-` |
| `capl[1]` | `-H`, `-CH₃`, `-OH` |
| two unnamed selects | hidden duplicates of the same |

The radio group `model` (`single`, `solution`, `melt`) is the system type. Selecting
each in turn changes nothing: 11 visible controls before and after, so the building-block
section is not gated on it.

What remains, all hidden: four unnamed text inputs, `subtext[1]`, and
`button#chainBtn`. That pattern suggests monomers are **entered or picked through a
control that is not a `<select>` at all** — which is why discovery needs a control's
text, its section, and the reason it is hidden, not just its name.

**The monomer list is `<li onclick="set_monomer(this)">`.** Each monomer is a group of
those, and the option's own text is the *tacticity variant* -- `atactic`,
`isotactic (R)`, `isotactic (S)`, `syndio (R)` -- while the monomer name sits above the
group. Reading option text as the monomer would produce a catalogue of four tacticities
repeated many times: a catalogue of the wrong thing.

The live menu holds **312 options in 39 groups**, and presents *two shapes* through one
mechanism:

| Group | Options | What it is |
|---|---|---|
| `Polystyrene` (`STYR`) | isotactic (R/S), syndio (R), atactic | one polymer, four conformations |
| `Amides` | Polyamide, Polyamide (inv), Nylon 3, Nylon 6 | a class holding four polymers |

Reading the second as conformations would drop Nylon 6, poly(ethylene terephthalate),
PTFE, polyketone and poly(ethylene oxide) from the catalogue while leaving a
plausible-looking result behind.

The rule comes from the page's own data, not a list of words we think mean tacticity:
**options sharing one form value are conformations of one polymer; options with distinct
values are distinct polymers.** A wordlist would go stale the moment CHARMM-GUI adds a
conformation; values cannot drift out of step with themselves.

One edge case needed a second signal. A class holding exactly *one* polymer --
`Halides -> Polytetrafluoroethylene` -- has a single value either way, so the value rule
collapsed it into "Halides" and lost PTFE. The tie-breaker is also data-driven: a
conformation label recurs across dozens of groups, a polymer name appears in exactly
one.

`handler_choices` reads them with their ancestor group name, and
`monomers_from_handler_choices` turns each group into one `MonomerEntry` whose
`variants` are its tacticities. A group with no name above it is **dropped**, not
invented into a monomer.

**No form control carries this choice at all.** The fourth dump, once buttons
carried their text and tables their contents, showed `poly_table` holding one row:

```
Polymer chain 1:   [capf]   [   select unit   ]   [capl]
```

`select unit` is placeholder text between literal brackets, and the surrounding controls
are all JavaScript-driven:

| Element | Text | Handler |
|---|---|---|
| `button#chainBtn` | "Add polymer chain" | — |
| `button#butt[1]` | "Add monomer unit" | `insert_row_block(this)` |
| `button` | `×` | `remove_monomer(this)` |
| `button` | `-` | `remove_row_chain(this)` |

The sixteen hidden controls are also explained, and not as a collapsed section: fourteen
of them sit inside `div#skel`, and `table#poly_table_skel` is a **template row** cloned
when a chain or monomer unit is added. They are not meant to be visible, and probing the
system-type radio was never going to reveal them. `subtext[1]` carries a default of `10`
-- plausibly the repeat count for a unit -- and `input[name=project]` carries `polymer`.

So a monomer is chosen by clicking `select unit`, which presumably opens a picker. The
catalogue is not enumerable from this page's DOM, which means `CATALOG_NOT_IDENTIFIED`
was the correct verdict at every step: there was no monomer control to find here.

The 68 "card" elements are the Input Generator's own navigation -- Job Retriever, PDB
Reader, Membrane Builder and so on -- not monomers.

### Enumerating what a build needs

```bash
.venv/bin/python scripts/charmm_gui_live.py probe --click 'select unit'
```

`probe` clicks one named element and reports what appeared: newly visible controls,
their options, and a summary of every field a build would need with `REQUIRED` marked.
It compares **visibility**, not presence — a page that hides its controls already has
them in the DOM, so "newly present" finds nothing and "newly visible" finds everything
the click revealed.

It also lists clickable non-form elements: anything with an `onclick` or a pointer
cursor. That is how `select unit` is found at all, being a `<span>` with no name, id or
role.

`probe` **refuses** to click anything whose text matches a build control — "next step",
"build polymer", "submit", "generate", "continue", "proceed" — before it launches a
browser or reads a credential. Discovery must never be one typo away from creating a job
on a shared academic service. A test asserts the refusal covers the live page's control
verbatim, including its embedded newline.

### Three catalogue states

`0 monomers` is not one fact. It is three, and conflating them turns a bug in our
selector into a claim about somebody else's chemistry:

| State | Meaning |
|---|---|
| `CATALOG_CAPTURED` | a monomer control was found and its options read |
| `CATALOG_EMPTY` | a monomer control was found and genuinely offers nothing |
| `CATALOG_NOT_IDENTIFIED` | no control could be identified as the monomer selector |

The first live discovery run returned 8 form fields and 0 monomers. That was
`CATALOG_NOT_IDENTIFIED`: discovery scanned only `<select>` elements, so any page
listing monomers as radio buttons, checkboxes or a datalist reported zero. It says
nothing about what Polymer Builder offers.

`choice_sets()` now scans every representation a page can use for a list, and a
`NOT_IDENTIFIED` catalogue records what it *did* see, so the failure is actionable
rather than just negative. `CATALOG_NOT_IDENTIFIED` maps to `REQUIRES_HUMAN_REVIEW`,
never to "unsupported".

## Exact matching, never substitution

`resolve_monomer` matches on the catalogue's form value or its normalised label —
lower-cased, punctuation stripped. That is the only latitude. It will not pick the
closest name, the most similar structure, or the first of several matches.

"polylactide" *is* PLA to a chemist. The engine still refuses it, lists it as a
candidate for a person to confirm, and stops. Confirming it is what `aliases` is for:
an alias is a human's recorded decision, which is exactly the judgement the matcher
declines to make alone.

## Verify, then submit

Between filling the form and clicking build sits `verify_before_submit`, which reads
every control back off the page and compares it to the normalised request. Any
disagreement, any field that cannot be read, any field that could not be located —
**nothing is submitted**.

Filling a form is open-loop. A select can reject a value, a numeric input can clamp, a
handler can rewrite what was typed, a changed layout can put the right value in the wrong
box. Each produces a job for a *different polymer*, and the resulting system looks
perfectly valid downstream: correct topology, sensible density, clean simulation,
answering a question nobody asked. Reading the page back is the only thing that tells
those cases apart from success.

## Job identifiers are read, never chosen

After submission the page is searched for a job id. One unambiguous candidate is the job
id. Several distinct candidates → `REQUIRES_HUMAN_REVIEW`; none → `SUBMISSION_FAILED`
with the note that the job may nonetheless exist. No id is ever fabricated or picked out
of a set: monitoring the wrong one downloads somebody else's system and validates it as
ours.

## Never submitting the same build twice

The queue is indexed by request fingerprint — chemistry, composition, system type, box,
temperature; **not** job id, timestamp or credentials. Two identical requests fingerprint
identically.

The rule is asymmetric on purpose:

| Situation | Action |
|---|---|
| No prior entry, or queued and never sent | submit |
| In flight (`SUBMITTED`/`PENDING`/`RUNNING`) | monitor |
| `DOWNLOADED`/`VALIDATED` | reuse |
| Failed **before** the click | stays queued, retryable |
| Failed **after** the click, outcome unknown | `BLOCKED`; a person checks first |

"We did not see a job id" is not evidence that no job was created. Resubmitting on that
basis is how one request becomes three on a shared academic service.

One build is in flight at a time. `next_ready()` returns nothing while any job is
running.

## When the site changes

`UI_SCHEMA_MISMATCH`, and a sanitised diagnostic under
`campaign/charmm_gui/diagnostics/`: the current URL, the discovered schema, the form
controls, the error. Values are stripped in the worker before the HTML leaves it;
diagnostics keep an **allow-list** of keys, so a field nobody anticipated is dropped
rather than leaked. Cookies, storage and authorisation headers are not filtered out —
nothing in this subsystem ever reads them.

A screenshot is taken only when the page holds no password control, and the manifest
records why one is missing when it is.

## CAPTCHA and MFA

Detected, reported as `HUMAN_INTERVENTION_REQUIRED`, and never solved, retried or worked
around. They are access controls. Detection exists so a person can be asked, and the
password is never typed on a page that presents one.

A rejected password is likewise never retried: it will not have changed by itself, and
repeating it is how an account gets locked.

## Running the live sequence

The whole of §3–§19 is one operator command, so the password stays between the
operator's shell and the browser process:

```bash
.venv/bin/python scripts/charmm_gui_live.py discover
```

It prompts for whatever is not already in the environment. The password is read with
`getpass`, so terminal echo is off and the characters reach the process without passing
through the shell, a history file, or `ps`.

If there is no terminal, the prompt **raises** rather than falling back to `input()`.
A silent fallback would echo the password to the screen and into whatever is capturing
the session — which is exactly what the prompt exists to prevent. A non-interactive
caller must set the environment variables instead:

```bash
export CHARMM_GUI_EMAIL='you@example.edu'
read -rs CHARMM_GUI_PASSWORD && export CHARMM_GUI_PASSWORD   # keeps it out of history
```

`discover` is read-only: login, navigate to Polymer Builder through the visible
interface, capture the catalogue and form schema, and compare both against
`PolymerBuilderSpec`. It submits nothing. It writes `monomer_catalog.{json,md}`,
`builder_form_schema.json`, `spec_mapping_*.md` and `CHARMM_GUI_LIVE_REPORT.md`.

A build is a separate command, and needs the monomer and the submit control **named
from that capture** — neither is guessed:

```bash
.venv/bin/python scripts/charmm_gui_live.py build --monomer '<from the catalogue>' --submit-field '<from the schema>' --dp 10 --rationale 'first live build: smallest system'
```

`read -rs` is the reason there is no `--password` flag: a password in a command
argument is visible in `ps` to every user on the machine and lands in shell history.

## Capability verification

`campaign/charmm_gui/capability_verification.json` records what has actually been
proven. A capability reaches `LIVE_VERIFIED` only when handed evidence — a job id, an
archive hash, a `grompp` exit — and the registry refuses to record one out of order: a
download cannot be proven without the job it came from.

`FIXTURE_VERIFIED` is deliberately a *lower* state that can never be promoted without
live evidence. Passing tests against a page we wrote ourselves proves our decisions, not
the site's behaviour.

And `SYSTEM_GENERATION_VERIFIED` is not `FORCE_FIELD_QUALIFIED`. The registry cannot
record qualification at all — it raises if asked — because qualification needs property
evidence from a campaign that no successful build can supply.

## Live tests

Off unless switched on. Ordinary CI needs no credentials and touches nothing.

```bash
CHARMM_GUI_LIVE_TEST=1 CHARMM_GUI_EMAIL=… CHARMM_GUI_PASSWORD=… pytest tests/browser/test_live_charmm_gui.py
```

A real submission needs a **second** opt-in (`CHARMM_GUI_LIVE_SUBMIT=1`) plus an
explicitly named monomer from the captured catalogue and an explicitly named submit
control. Neither is guessed, and there is no bulk path. Everything written lands under
`campaign/charmm_gui/live_test/`, away from production research data.

## Human-in-the-loop boundary

**Automated**: catalogue discovery, specification, navigation, form filling,
verification, submission, monitoring, download, import, validation.

**Human**: CAPTCHA and MFA; confirming the semantic mapping from discovered controls to
scientific quantities; resolving an ambiguous or absent monomer; any scientific decision
that cannot be safely inferred.

The semantic mapping deserves emphasis. Discovery *suggests* that a control labelled
"Number of chains" is the chain count, by matching label text. That suggestion is written
into the schema file as `semantic_hints` for a person to confirm once — not consulted
afresh at submission time. A control whose label merely contains "chains" is not
necessarily the chain count.

## What has and has not been tested

| | Status |
|---|---|
| Playwright 1.62.0 + Chromium 151.0.7922.34 in `.browserenv` | **REAL SOFTWARE TESTED** |
| Keyboard typing against a paste-rejecting, keydown-counting form | **REAL SOFTWARE TESTED** |
| Form discovery, catalogue extraction from a real DOM | **REAL SOFTWARE TESTED** |
| Verification blocking a real clamping form | **REAL SOFTWARE TESTED** |
| CAPTCHA/MFA stop, ambiguity refusal, duplicate protection | **FIXTURE TESTED** |
| Login to charmm-gui.org | **NOT TESTED** — requires credentials |
| Catalogue contents, form schema, job submission, download | **NOT TESTED** — requires credentials |

No claim is made about CHARMM-GUI's actual page structure. Nothing in this repository
has ever loaded it.
