# Force fields

Status labels: [STATUS_LEGEND.md](STATUS_LEGEND.md)

A force field is a claim about a chemistry, and this engine will not make one it cannot
support. The rule throughout is **assign, never derive**: every parameter comes out of an
installed force-field file, and a repeat unit whose parameters are not all tabulated is
refused rather than approximated.

---

## Capability table

| Capability | Module | Status |
|---|---|---|
| Read an installed OPLS-AA and its atom types | `simulation/opls_typing.py` | REAL |
| Type a saturated-hydrocarbon repeat unit | `simulation/opls_typing.py` | REAL |
| Generate a GROMACS topology from a typing | `simulation/opls_typing.py` | REAL |
| Build an amorphous melt from a repeat unit | `simulation/melt_builder.py` | REAL |
| Validate a force field against QM | `qm/validation.py` | REAL, REQUIRES-LOCAL-SOFTWARE |
| Type any polar / aromatic / halogenated polymer | — | NOT IMPLEMENTED |
| CGenFF, OpenFF/Sage, GAFF routes | — | REQUIRES-LOCAL-SOFTWARE (none installed) |
| Charge derivation (RESP/ESP) | — | NOT IMPLEMENTED |
| Choosing which force field to trust | — | REQUIRES-EXPERT-DECISION |

---

## Why hydrocarbons and nothing else

OPLS-AA charges are fitted per *reference molecule*, so a per-type default charge is only
correct in the environment it came from. For saturated hydrocarbons that distinction
collapses, because every alkane group is neutral on its own:

```
CH3  -0.180 + 3(+0.060) = 0     CH  -0.060 + 1(+0.060) = 0
CH2  -0.120 + 2(+0.060) = 0     C    0.000             = 0
```

An alkane assembled from these is neutral by construction — no charge derivation, no
fragment template, no judgement call. All four share the bonded type `CT`, so bonds,
angles and dihedrals come from the same published set.

For polar repeat units it does not collapse, and the engine measures the shortfall rather
than absorbing it:

* **Poly(ethylene oxide)** — two `C(H2OR)` groups (`opls_182`, +0.140, each with two H at
  +0.060) and one dialkyl ether oxygen (`opls_180`, −0.400) sum to **+0.120 e per repeat
  unit**. The balance lives in diethyl ether's terminal groups.
* **Polystyrene** — OPLS tabulates the benzylic series `opls_148` (toluene CH₃) and
  `opls_149` (ethylbenzene CH₂), each set so the benzylic group carries +0.115 against the
  ipso carbon's −0.115. Polystyrene needs the next member, a benzylic **CH**, and it is
  not in the table. The convention makes its value obvious, which is exactly why writing
  it in would be fabrication rather than lookup.

Both numbers are asserted in `tests/forcefield/test_opls_typing.py` against the installed
force field, so the justification cannot drift away from the parameters it describes.

`TypingStatus` reports which case applies: `SUPPORTED`, `REQUIRES_CALIBRATION` (typed but
not neutral), `UNSUPPORTED` (an atom has no tabulated type), or `UNCERTAIN`.

---

## `gmx x2top` does not produce a valid OPLS topology

This is worth stating plainly, because `x2top` is the obvious route and it is wrong.

For **n-butane** it emits **3 dihedrals** where the molecule has 27, and fills them with
placeholder Ryckaert-Bellemans coefficients `60, 5, 3, 60, 5, 3` — not OPLS values.
`grompp` accepts the topology. `mdrun` runs it. The torsional energy comes out near
**300 kJ/mol** against a true profile spanning 21 kJ/mol, with the MM minimum on the
*eclipsed* conformer.

It also assigns `opls_157`/`opls_158` ("alcohols") to alkane carbons. That part is
harmless — those types share the `CT` bonded type and identical LJ parameters with
`opls_135`/`opls_136`, and x2top writes the correct alkane charges explicitly, so the
energies are identical to the digit. Checking *that* equivalence is what created the false
confidence: it compared two x2top topologies, and both were equally broken. Only the QM
comparison exposed the dihedral problem.

`write_opls_topology` emits every bonded interaction **without inline parameters**, so
GROMACS resolves the published values from `ffbonded.itp` and there is no second copy to
drift.

---

## Qualification: MM against QM

`qm/validation.py` compares a force field to a QM reference under criteria a person
chose. The result for the campaign's force field:

| | |
|---|---|
| Reference | n-butane C–C–C–C relaxed torsion scan, 13 points |
| Level of theory | B3LYP-D3BJ / def2-SVP (ORCA 6.1.1, `COMPLETED`) |
| QM profile | anti 0.00, gauche 2.31, barrier 14.52, syn 21.50 kJ/mol |
| Torsion RMSE | **1.130 kJ/mol** (threshold 2.0) |
| Barrier error | **−2.128 kJ/mol** (threshold 4.0) |
| Relative barrier error | **9.9 %** (threshold 25 %) |
| Verdict | **PASS** |

The thresholds are `AcceptanceCriteria.general_organic_forcefield()` — conventions from
the parameterisation literature, not derivations, and named as such in the campaign
config so the manifest records which convention was adopted.

The MM side uses `gmx mdrun -rerun` over the QM-optimised geometries, which compares the
two surfaces at identical structures rather than at each method's own minima.

---

## What is not installed here

| Route | Why unavailable |
|---|---|
| CGenFF / CHARMM36 | no `cgenff` program and no `charmm36.ff`; only `charmm27.ff`, a protein/nucleic-acid set |
| OpenFF / Sage | `openff-toolkit` not installed |
| GAFF / GAFF2 | no AmberTools (`antechamber`, `parmchk2`) |

No CGenFF penalty scores appear anywhere in this repository because CGenFF is not
installed. There is nothing being suppressed.

Installing any of these would widen the qualified set well beyond polyolefins. Until then,
15 of the 18 dataset polymers are **BLOCKED — no qualified force field**, which is a
different statement from "not yet run".

## CHARMM36 / CGenFF

Reached through CHARMM-GUI Polymer Builder, now browser-driven rather than
hand-operated (`docs/CHARMM_GUI_BROWSER.md`). The route is implemented and tested
against a real browser on local fixtures; it has never been run against the live site,
because no credentials have been supplied.

CGenFF penalties are the reason this route matters even though OpenFF now covers all 18
candidates: they state how much of a parameter set was assigned by analogy, and no local
backend reports an equivalent. A high penalty raises QM priority; it is not a failure. A
low penalty is not proof. Missing penalties are `INCONCLUSIVE`.

Nothing from this route is qualified for any property class, because nothing has been
built through it yet.
