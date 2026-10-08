# Parameterization

Status labels: [STATUS_LEGEND.md](STATUS_LEGEND.md)

Turning a polymer candidate into a simulation you can believe means answering several
questions that are easy to run together and must be kept apart:

| State | What it means | What it does *not* mean |
|---|---|---|
| **PARAMETERIZED** | a topology file exists | that any number in it is right |
| **VALIDATED** | it was checked against independent evidence | that it is fit for your purpose |
| **QUALIFIED** | validated well enough for a **stated property class** on a **stated polymer** | that the force field is qualified generally |

A topology file is evidence of the first and of nothing else. The whole subsystem is
built to stop the first from being mistaken for the third.

---

## Capability table

| Capability | Module | Status |
|---|---|---|
| Tool discovery (17 tools, measured) | `parameterization/capability.py` | REAL |
| Backend interface and registry | `parameterization/backend.py` | REAL |
| OPLS-AA backend | `parameterization/backends/opls.py` | REAL (saturated hydrocarbons) |
| CHARMM-GUI backend | `parameterization/backends/charmm_gui.py` | REAL, REQUIRES-CREDENTIALS, human-in-the-loop |
| OpenFF backend | `parameterization/backends/openff.py` | **REAL** — Sage 2.2.0 in an isolated `.paramenv`, verified through real `grompp` |
| GAFF / AmberTools backend | `parameterization/backends/gaff.py` | **UNAVAILABLE** — not installed |
| Parameter completeness | `parameterization/completeness.py` | REAL |
| Charge consistency | `parameterization/charges.py` | REAL |
| Property-class quality gate | `parameterization/quality.py` | REAL |
| QM-sensitivity detection | `parameterization/quality.py` | REAL |
| Routing | `parameterization/router.py` | REAL |
| State machine | `parameterization/state.py` | REAL |
| Registry and family coverage | `parameterization/registry.py` | REAL |
| Representative selection | `parameterization/representatives.py` | REAL |

---

## Capability is a ladder, not a boolean

```
UNAVAILABLE → STRUCTURE_SUPPORTED → PARAMETERIZATION_AVAILABLE
            → SYSTEM_BUILD_AVAILABLE → PARAMETERS_VALIDATED → QUALIFIED
```

`BLOCKED` and `REQUIRES_EXPERT_REVIEW` sit deliberately **off** the ladder. They are
verdicts, not progress, and `at_least(BLOCKED, anything)` is false — otherwise "we
cannot type this chemistry" would sort above "we have not installed the tool".

The distinction that does the most work is `STRUCTURE_SUPPORTED` versus
`PARAMETERIZATION_AVAILABLE`. CHARMM-GUI can represent almost any organic polymer, and
parameterizes none of them until a person has actually run the build, so it never
reports higher than `STRUCTURE_SUPPORTED` from an assessment.

---

## Routing

`SystemBuildRouter` prefers automatic routes over human-in-the-loop ones, and higher
capability over lower. Cost only ever breaks ties between routes with **equal**
evidence; it never promotes a weaker route because it is cheaper.

When two backends are equally capable and the evidence does not separate them, the
router returns `ambiguous` with both attached and selects **neither**. Choosing between
two force fields is a scientific judgement, and resolving it by sort order would
manufacture a decision nobody made.

---

## Parameter completeness: why `grompp` is not the check

`grompp` exiting 0 means "I found a number for everything I was asked to look up". Three
things satisfy that while the physics is wrong:

* a wildcard type matched where a specific one should have;
* parameters written inline, so nothing was ever looked up;
* the interaction is simply **absent**, so there was nothing to miss.

The third is the dangerous one. A topology with no dihedral section preprocesses
perfectly and simulates a molecule with free internal rotation. `gmx x2top` produces
exactly this: **three dihedrals for n-butane, which needs 27**, filled with placeholder
Ryckaert-Bellemans coefficients. The torsional energy comes out near 300 kJ/mol against
a true profile spanning 21, with the MM minimum on the eclipsed conformer.

So completeness is measured against the **connectivity**: angles and dihedrals implied
by the bond list, compared with what the topology declares. A short section shows up as
a shortfall rather than as a smaller-but-consistent file.

---

## Charges are measured, never repaired

A neutral polymer whose topology does not sum to zero has a real defect upstream — a
mis-assigned type, a truncated charge group, a fragment charge carried from the reference
molecule it was fitted to.

Renormalising hides all of them. Spreading +0.12 e over 182 atoms yields a topology that
passes every later check and is wrong in a way nothing downstream can detect.

| Residual | Verdict |
|---|---|
| ≤ 1e-6 e | PASS |
| between that and 0.5 e | **INCONCLUSIVE** — real, small, and not repaired here |
| ≥ 0.5 e | **FAIL** — structural, e.g. a whole missing group |
| no atoms declared | **INCONCLUSIVE** — unknown, not zero |

Charges *may* be adjusted deliberately. `ChargeAdjustment` cannot be constructed without
a method, a reason, software and an author, and it keeps the original value.

---

## Quality is relative to the question

There is no universal threshold anywhere in `quality.py`. Each `PropertyClass` carries a
`QualityStandard` with a written justification:

| Property class | Torsional QM required? | Penalty ceiling without QM |
|---|---|---|
| `bulk_density` | no | moderate |
| `thermodynamic`, `transport`, `mechanical`, `structural` | yes | moderate |
| `conformational_free_energy` | yes, RMSE ≤ 1.0 kJ/mol | zero |
| `interfacial_free_energy` | yes, RMSE ≤ 1.5 kJ/mol | zero |

The reason is physical. A conformer population depends exponentially on relative
energies: at 300 K, 4 kJ/mol is about a factor of five in a population ratio. A density
averages over conformers and barely notices.

### A penalty is evidence, not a verdict

A high CGenFF penalty says the analogy was weak. It does not say the parameter is wrong.
So the gate returns:

* **PASS** when the penalty is within the ceiling for this property class;
* **INCONCLUSIVE** when it is over and QM has not been run — "check this";
* **PASS** when it is over and QM **confirmed** the parameters anyway;
* **FAIL** only when QM was run and **refuted** them.

The third case matters: leaving a weak analogy INCONCLUSIVE after a successful QM
validation would make the QM run pointless, because nothing could ever clear it.

---

## The state machine

```
DISCOVERED → BACKEND_SELECTED → PARAMETERIZED → TOPOLOGY_VALIDATED
  → PARAMETER_COMPLETENESS_VALIDATED → QM_VALIDATION_REQUIRED → QM_VALIDATED
  → SYSTEM_VALIDATED → QUALIFIED
```

Forward progress is strictly sequential — `QUALIFIED` cannot be asserted from
`DISCOVERED`. Verdicts (`BLOCKED`, `FAILED`, `INCONCLUSIVE`, `REQUIRES_EXPERT_REVIEW`)
are reachable from anywhere, because any step can discover the answer is no. Every
transition records its reason, so a QUALIFIED record reads backwards to its evidence.

---

## Family qualification needs more than one example

One qualified polyester does not qualify polyesters. PLA and PET share a functional group
and very little else. `FamilyCoverage.qualified_for_family` requires at least
`MIN_FAMILY_REPRESENTATIVES` (2) distinct qualified members, and reports `confidence` of
`none` / `single-representative` / `moderate` rather than a boolean.

`representatives.py` chooses which polymers to spend validation on by rewarding the
*awkward* members — chemically unusual relative to the family, high analogy penalty,
descriptor distance from what is already validated. Validating polyethylene and declaring
polyolefins covered is the failure it exists to prevent.

---

## CLI

```bash
polymer-engine parameterization inventory -o campaign/parameterization
polymer-engine parameterization backends
polymer-engine parameterization assess "poly(lactic acid)"
polymer-engine parameterization compare "poly(lactic acid)"
polymer-engine parameterization qualify --property-class bulk_density
polymer-engine parameterization validate path/to/topol.top
```

`assess` exits 3 when no route is decided. `validate` exits 3 when a topology fails
completeness or charge consistency.

---

## Coverage today

All 18 dataset candidates have an **automatic** route: 15 to OpenFF Sage, 3 to OPLS-AA.
None requires a human step any more, though CHARMM-GUI remains available as an
alternative route and as the only one that produces CGenFF penalties.

Routed is still not qualified. One polymer — poly(lactic acid) — has been carried
through the full path and reaches `SYSTEM_VALIDATED` for `bulk_density`; nothing is
`QUALIFIED`, because qualification also needs the property-class evidence the campaign
produces.

### The isolated environment

OpenFF lives in `.paramenv`, not the engine's `.venv`, so a running campaign's
environment never acquires a large new dependency tree mid-run.
`scripts/openff_worker.py` is the only code that imports OpenFF; the backend drives it as
a subprocess and treats a worker failure as a chemistry limit, not an outage.

### The charge model is part of the force field

Sage was fitted against AM1-BCC. AM1-BCC needs AmberTools' `sqm` or an OpenEye licence,
neither installed here, so the worker uses **NAGL** — OpenFF's published graph-network
surrogate — and names the exact model in provenance. It **refuses** Gasteiger rather than
falling back, because pairing Sage with a different charge model is a silent change of
force field, not a degradation of one.

## What this subsystem does not do

* It does not produce CGenFF penalties from the OpenFF route. OpenFF reports no analogy
  penalty at all, and silence is not evidence of quality — so chemistry alone sets QM
  priority there.
* It does not submit CHARMM-GUI jobs. See [CHARMM_GUI.md](CHARMM_GUI.md).
* It does not qualify anything on its own. Routing says a backend *could* try;
  qualification requires validation evidence that does not yet exist for any polymer
  outside the three polyolefins.
