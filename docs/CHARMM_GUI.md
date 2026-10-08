# CHARMM-GUI

Status labels: [STATUS_LEGEND.md](STATUS_LEGEND.md)

CHARMM-GUI is the route that could unblock 15 of the 18 dataset polymers. It is also the
one place in this engine where a human is a required part of the loop, and that is
stated everywhere rather than hidden.

## The constraint

CHARMM-GUI publishes exactly three endpoints:

```
/api/login   /api/check_status   /api/download
```

There is **no job-submission endpoint**. Automating the build would mean
reverse-engineering an undocumented web form — the same class of mistake as inventing an
API, and forbidden by this project's charter.

`CHARMMGUIProvider.submit_module` therefore raises `UnsupportedCapability`, and a test
pins that behaviour so it cannot quietly change.

## What is automated, and what is not

```
polymer record
    │  AUTOMATED — derive the specification, hash it, write a brief
    ▼
CHARMM-GUI Polymer Builder
    │  HUMAN — a person runs the build and notes the job id
    ▼
job id or downloaded archive
    │  AUTOMATED — download, safe-extract, catalogue, hash
    ▼
system validation + CGenFF penalty parsing
    │  AUTOMATED — completeness, charges, penalties, quality gate
    ▼
PARAMETERIZED  (not validated, not qualified)
```

| Step | Status |
|---|---|
| Specification generation | REAL |
| Job submission | **NOT IMPLEMENTED — no documented endpoint** |
| Download by job id | REAL, REQUIRES-CREDENTIALS |
| Local archive import | REAL |
| Safe extraction | REAL — 30 archive-security tests |
| Artifact cataloguing and hashing | REAL |
| CGenFF penalty parsing | REAL |
| System validation | REAL |

## The specification

`parameterization charmm-gui spec` derives the repeat unit, degree of polymerisation,
chain count and box edge from the polymer record and the campaign settings, and writes
both a JSON spec and a human brief.

The spec carries a **fingerprint that excludes the job id**, so two people building the
same specification produce the same fingerprint. That is what makes a manual step
auditable.

The brief deliberately does **not** transcribe web-form field labels. Those change
between CHARMM-GUI releases, and a stale walkthrough is its own kind of fabrication. It
states the information the module asks for, in its own terms.

## CGenFF penalties

CGenFF assigns parameters **by analogy** and reports how far it had to reach. That number
lives in a trailing comment:

```
CG321 CG321 OG302 CG2O2  0.1500 3 0.00 ! PLA , from analogy, penalty= 64.0
```

Any parser treating `!` as "ignore the rest of the line" discards the one field that says
whether the parameters are trustworthy. `simulation/cgenff.py` reads comments *for* the
penalties, and a stream file with the penalties stripped comes back **INCONCLUSIVE**,
because "no penalties found" otherwise reads exactly like "no penalties incurred".

Tiers (<10 good, 10–50 moderate, >50 poor) are the CGenFF program's published guidance,
treated as the convention they are. See [PARAMETERIZATION.md](PARAMETERIZATION.md) for
how a penalty interacts with the property-class quality gate — in short, a high penalty
means *validate this*, not *reject this*.

## Ready for a documented API

If CHARMM-GUI ever publishes a submission endpoint, only
`CharmmGuiBackend.parameterize` changes: the specification generator, the import path,
the validation and the provenance all stay as they are. Nothing else in the engine
assumes the step is manual.

## The browser route

Since the browser subsystem landed, the manual step described above is no longer the
only option. `polymer-engine browser discover` logs in through the visible interface,
captures the monomer catalogue and the form schema, and `CharmmGuiAcquisition` fills,
verifies and submits a build, then monitors and downloads it through the documented API.

The API boundary has not moved: CHARMM-GUI still publishes no submission endpoint, and
none is invented. What changed is that the web interaction is now automated rather than
described in a brief for a person to follow.

See `docs/CHARMM_GUI_BROWSER.md` for credentials, keyboard password entry, selector
discovery and the duplicate-submission rules, and `docs/CHARMM_GUI_AUTOMATION.md` for
the route from a build to a qualification state.

**Not yet run live.** No credentials have been supplied, so nothing in this repository
has loaded charmm-gui.org. The catalogue is empty and the coverage matrix reads
`unknown` for every candidate.
