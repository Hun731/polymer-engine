# The CHARMM-GUI acquisition route, end to end

`docs/CHARMM_GUI_BROWSER.md` covers the browser. This document covers what happens on
either side of it — how a polymer candidate becomes a validated system, and where that
route stops short of a scientific claim.

## The route

```
polymer candidate
  → capability discovery          which backends exist on this machine
  → parameterization router       chemistry, capability, evidence, cost
  → CHARMM-GUI browser build      login, discover, resolve, fill, verify, submit
  → documented status API         check_status, bounded polling with backoff
  → documented download API       download, checksum, tar verification
  → safe extraction               traversal, symlinks, bombs, truncation
  → artifact discovery            PDB/PSF/CRD/GRO/TOP/ITP/PRM/STR/RTF, hashed
  → force-field identification    read from the files, not inferred from the source
  → CGenFF penalty parsing        max penalty, worst terms, distribution
  → parameter completeness        types, masses, charges, bonded terms, LJ
  → charge validation             repeat unit, chain, system
  → QM validation where required  ORCA torsion scans against MM
  → system validation             real grompp, no -maxwarn
  → provenance registration       every hash, every parent, back to the request
  → qualification state           per property class, or not at all
```

Everything from *safe extraction* down already existed and is shared with OpenFF and
OPLS-AA. There is no separate, easier path for a CHARMM-GUI system: it goes through the
same gates, and it can fail them.

## Three states that are never collapsed

**PARAMETERIZED** — a topology exists. Nothing has been checked.
**SYSTEM_VALIDATED** — complete, neutral, and accepted by a real `grompp` with no
warnings suppressed.
**QUALIFIED** — validated *for a stated property class*, with provenance, on a stated
family.

A downloaded CHARMM-GUI archive is evidence of the first and of nothing else.

## What CHARMM-GUI is good for, and what it is not evidence of

A CHARMM-GUI system is a high-quality reference route: it is built by the tool the
CHARMM community maintains, it carries CGenFF penalties that state how much of the
parameter set was assigned by analogy, and it is an independent construction path.

When a local backend produces a system for the same polymer, the two are worth comparing
— topology, charges, equilibrated density, RDF, Rg, packing, energy behaviour. But:

> **Agreement between two force fields is not evidence that either is right.**

Two force fields can share a fitting set, an analogy, or an error. Independent QM or
experimental evidence stays necessary, and a comparison is recorded as a comparison, not
promoted to a validation.

## CGenFF penalties

Parsed by the existing reader, never suppressed. Published guidance:

| Penalty | Reading |
|---|---|
| < 10 | good analogy |
| 10–50 | moderate; QM checking is worthwhile |
| > 50 | poor analogy; QM is required before use |

Two rules go with that table:

* a **high** penalty is not automatic failure — it says the parameters were assigned by
  weak analogy, which is a reason to check them, not a verdict;
* a **low** penalty is not proof of correctness — it says a good analogy existed, not
  that the analogy holds for the property being measured.

Missing penalty information is `INCONCLUSIVE`, never `PASS`. Absence of evidence is not
evidence of adequacy, and a stream file with the penalties stripped is exactly the case
where a permissive default would be most wrong.

An OpenFF route reports no penalty at all, which is a different thing again: silence
from a model that does not compute penalties is not a low penalty.

## Capability dependencies are a graph, not a list

The first version of `verification.py` used position in a flat tuple as the dependency
relation: everything listed earlier was treated as a prerequisite. That forces a total
order onto what is really a graph with siblings, and it produced a genuinely nonsensical
refusal — proving a *form schema* was blocked on proving a *catalogue*, with an error
message about downloads and job ids.

Dependencies are now an explicit DAG in `PREREQUISITES`, with `ANY_OF` for alternatives:

```
LIVE_LOGIN_VERIFIED
  └─ POLYMER_BUILDER_REACHED
       ├─ FORM_SCHEMA_VERIFIED  ─┐
       └─ CATALOG_VERIFIED      ─┴─ SPEC_MAPPING_VERIFIED
                                     ├─ SINGLE_CHAIN_BUILD_VERIFIED ─┐
                                     └─ MELT_BUILD_VERIFIED         ─┴─ JOB_ID_VERIFIED
                                          └─ JOB_MONITORING_VERIFIED
                                               └─ DOWNLOAD_VERIFIED
                                                    └─ SYSTEM_IMPORT_VERIFIED
                                                         └─ GROMACS_VALIDATION_VERIFIED
                                                              └─ SYSTEM_GENERATION_VERIFIED
```

Form schema and catalogue are **siblings**: both are read off the same page, neither
depends on the other, so a page whose monomer control cannot be identified still proves
its form schema. Nothing in the discovery half depends on anything in the build half — a
catalogue is evidence about a page, not about a job.

`dependency_problems()` audits the graph for cycles and unknown names, and a regression
test asserts that no discovery state can reach a build artifact.

## Duplicate protection and the shared service

CHARMM-GUI is a shared academic resource. One session, one build at a time, bounded
polling with backoff, and a fingerprint-indexed queue that reuses rather than
resubmits — see `docs/CHARMM_GUI_BROWSER.md` for the state table.

## Capabilities exposed to the autonomous engine

When `.browserenv` is genuinely present and Chromium actually launches, capability
discovery reports:

```
CHARMM_GUI_CATALOG  CHARMM_GUI_BUILD_SPEC  CHARMM_GUI_BROWSER_BUILD
CHARMM_GUI_JOB_STATUS  CHARMM_GUI_DOWNLOAD  CHARMM_GUI_IMPORT
```

The probe launches a browser rather than importing a module, because the Python package
installs happily without the browser binaries — and a session that discovers this at
login time has already spent a credential on it.

`CHARMM_GUI_BROWSER_BUILD` then becomes an action the research loop can choose, weighed
against OpenFF parameterization, ORCA validation, more sampling, or a new candidate, on
information gain against compute cost.

## Where this route stands today

The browser subsystem is implemented and tested against a real Chromium driving local
fixtures. It has never been run against charmm-gui.org, because no credentials have been
supplied. Consequently:

* the monomer catalogue is **empty**, and `charmm_gui_coverage_matrix.csv` reads
  `unknown` for every candidate — the honest state before discovery, not "no";
* no job has been submitted, downloaded, imported or validated through this route;
* no CHARMM-GUI system is `QUALIFIED` for anything.

Meanwhile all 18 candidates have an automatic local route (15 OpenFF, 3 OPLS-AA), so
CHARMM-GUI is currently the route for *CGenFF penalties* and for chemistry OpenFF cannot
type — not the only way forward.

## The wizard has different shapes for different system types

From the CHARMM-GUI Polymer Builder tutorial
(charmm-gui.org/download/polymer_builder_tutorial.pdf), confirmed by probing the live
page. The number of steps is not fixed -- it depends entirely on the system type.

**Single chain** — one step to a downloadable system:

```
select monomer(s) + repeat count
  → "Next Step: Build Polymer Chains"
  → results page: "view structure" and download.tgz are offered directly
```

**Solution / melt** — several steps, one of which runs a simulation on CHARMM-GUI's
servers:

```
select monomer(s) + repeat count
  → "Next Step: Build Polymer Chains"
  → System Size Determination + Solvation options
  → "Next Step: Generate Equilibrium"   ← runs a coarse-grained OpenMM equilibration
  → "Next Step: Replace into All-atom"   ← back-maps to atomistic
  → "Next Step: Input Generation"        ← writes the GROMACS/CHARMM inputs
  → download.tgz
```

The `wizard-probe` subcommand maps this from evidence and writes
`campaign/charmm_gui/wizard_map/wizard_map.json`:

```bash
.venv/bin/python scripts/charmm_gui_live.py wizard-probe --monomer 'Polylactic acid' --variant atactic --system-type single --advance
```

It records each step's heading, buttons, and download availability, and it **stops at
the first compute boundary** — any "Next Step" whose text launches equilibration,
all-atom replacement or input generation. It never clicks such a button: crossing one
spends the shared service's compute, which is a person's decision, not a mapping tool's.
Consequently a single-chain probe completes automatically (the download is one step
away), while a melt probe maps up to the equilibration step and hands off.

### What this corrected

An earlier `build` command treated the first "Next Step: Build Polymer Chains" click as
a completed build for every system type. That is right for a single chain and wrong for
a melt, where the click only reaches step 2 of a longer wizard. The probe establishes
which case applies before anything downstream assumes a finished system exists.
