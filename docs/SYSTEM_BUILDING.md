# Building a simulation system

Three routes produce a system the engine will simulate. They differ in what they can
cover and in how much of the work is automated, not in the standard they are held to:
every one ends at the same gates.

| Route | Force field | Coverage | Automation | State today |
|---|---|---|---|---|
| OpenFF | Sage 2.2.0 + NAGL charges | broad organic chemistry | full | **REAL** — 15 of 18 candidates |
| OPLS-AA | tabulated OPLS-AA/L | alkanes only | full | **REAL** — 3 of 18 candidates |
| CHARMM-GUI | CHARMM36/CGenFF | what Polymer Builder offers | browser-driven, credentials needed | **IMPLEMENTED, never run live** |
| GAFF | GAFF2 | broad organic chemistry | n/a | **UNAVAILABLE** — AmberTools not installed |

## Local construction (OpenFF, OPLS-AA)

```
repeat unit SMILES
  → grow chain to the requested DP
  → embed and relax
  → assign types and charges
  → pack N chains into a box sized from the target density
  → write topology
  → grompp
```

Two details that were wrong once and are now guarded:

**Packing is verified, not assumed.** `gmx insert-molecules` will silently place fewer
chains than asked for. The builder grows the box and retries, then reads the achieved
count back out of the structure — a system with 15 of 20 chains at an unrecorded density
is not the system that was requested.

**Atom names come from the file the packer reads**, not from the writer's own numbering.
Two independently-correct numbering schemes produced 2440 name mismatches that `grompp`
reported as missing atoms.

### OPLS-AA is tabulated, not derived

`gmx x2top` is **not** a source of OPLS bonded parameters. For n-butane it emits 3
dihedrals instead of 27, with placeholder Ryckaert-Bellemans coefficients. Two systems
built that way agree with each other perfectly and are both wrong; only comparison
against QM caught it. `simulation/opls_typing.py` emits interactions without inline
parameters so GROMACS resolves them from `ffbonded.itp`, and the regression test for
this stays.

### The charge model is part of the force field

Sage was fitted against AM1-BCC. AM1-BCC needs AmberTools' `sqm` or an OpenEye licence,
neither installed here, so the OpenFF worker uses **NAGL** — OpenFF's published
graph-network surrogate — and names the exact model in provenance. It refuses Gasteiger
rather than falling back: pairing Sage with a different charge model is a silent change
of force field, not a degradation of one.

## External construction (CHARMM-GUI)

See `docs/CHARMM_GUI_AUTOMATION.md`. The build happens in the web interface because
CHARMM-GUI publishes no submission endpoint; everything on either side of it is
automated, and the result enters the same pipeline.

## System validation

Whatever built it, a system must pass:

| Check | Failure means |
|---|---|
| coordinates | present, finite, atom count matching the topology |
| topology | parseable, `moleculetype` non-empty, sections consistent |
| parameter completeness | no missing types, masses, bonded terms or LJ parameters; no placeholders, unresolved wildcards or unexpected fallbacks |
| charges | repeat unit, chain and system charge as expected |
| box | positive, periodic, large enough for the cut-off |
| provenance | every artifact hashed, every parent reachable |
| `grompp` | exit 0 **with no warnings suppressed** |

`-maxwarn` is never used to conceal a missing or incorrect scientific parameter. A
warning that has been argued away in writing is a decision; one silenced by a flag is a
missing parameter that will surface later as a physical result nobody can explain.

An empty `moleculetype` used to pass the charge gate as "neutral" — the sum of no
charges is zero. It is now `INCONCLUSIVE`.

## The manifest

Every built or imported system writes `polymer_system_manifest.json`: backend, job id,
request fingerprint, force field and version, chain count, DP, tacticity, composition,
box, target and achieved starting density, topology hash, coordinate hash, parameter
hashes, CGenFF penalties, QM validation, system validation, and qualification scope.

That is what makes a final PMF or density traceable back through simulation, system,
parameter source and build request to the polymer.
