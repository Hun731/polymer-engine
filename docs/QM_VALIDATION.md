# QM validation

Status labels: [STATUS_LEGEND.md](STATUS_LEGEND.md)

A force field is a claim about a chemistry. QM validation is how that claim earns the
right to produce a number. This document covers what the engine validates, how it decides
what is worth validating, and what it refuses to conclude.

Detail on the ORCA layer itself is in [ORCA.md](ORCA.md); this is about how it is used
to qualify parameters.

## What gets validated

| Comparison | Status | Notes |
|---|---|---|
| Torsion profile (MM vs QM) | REAL | The primary check; validated on n-butane |
| Optimised geometry (Kabsch RMSD) | REAL | Reflection-guarded |
| Relative conformer energies | REAL | Same machinery as the torsion profile |
| Energy barriers | REAL | Extracted from the scan |
| Vibrational frequencies | FIXTURE-BASED | Imaginary-mode check |

The MM side uses `gmx mdrun -rerun` over the **QM-optimised geometries**, so the two
surfaces are compared at identical structures rather than at each method's own minima.

## Not everything deserves a QM job

Running a torsion scan for every parameter would cost more than the science is worth.
`detect_sensitive_terms` picks targets cheaply and structurally, before any calculation
is scheduled:

| Signal | Priority | Why |
|---|---|---|
| Torsional penalty > 25, or any penalty ≥ 50 | HIGH | Torsions set conformer populations |
| Charge penalty > 10 | HIGH | Charges propagate into every electrostatic observable |
| Ester / amide / carbonyl backbone linkage | HIGH | The torsion about it sets chain conformation |
| Other penalties > 10 | MEDIUM | Weak analogy, less consequential term |
| Nitrile, hydroxyl, ether | MEDIUM | Polar; local electrostatics |
| F, Cl, Br, I, S, P, Si | MEDIUM | Sparsely represented in general force fields |

It errs toward flagging. A needless torsion scan costs an hour; a missed one costs a
wrong answer.

Chemistry is flagged even when no penalty was reported, because a backend without a
penalty scheme — OPLS-AA, OpenFF — reports nothing at all, and silence is not evidence of
quality.

## Acceptance criteria are a decision, not a default

`compare_torsion_profiles` without an `AcceptanceCriteria` returns `passed=None` and
`REQUIRES_VALIDATION`. The engine computes the deviation; whether it is acceptable
depends on what you are measuring.

The property-class standards in `parameterization/quality.py` supply that decision
explicitly, and record which convention was adopted:

| Property class | Torsion RMSE tolerance |
|---|---|
| `conformational_free_energy` | 1.0 kJ/mol |
| `interfacial_free_energy` | 1.5 kJ/mol |
| `thermodynamic`, `transport`, `mechanical`, `structural` | 2.0 kJ/mol |
| `bulk_density` | not required |

## The worked example

OPLS-AA against n-butane, B3LYP-D3BJ/def2-SVP, 13-point relaxed C–C–C–C scan:

| | QM | Literature |
|---|---|---|
| anti (180°) | 0.00 kJ/mol | 0 by definition |
| gauche (60°) | 2.31 | ~2.8–3.3 |
| anti→gauche barrier | 14.52 | ~14–16 |
| syn (0°) | 21.50 | ~19–25 |

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| Torsion RMSE | 1.130 kJ/mol | 2.0 | PASS |
| Barrier error | −2.128 kJ/mol | 4.0 | PASS |
| Relative barrier error | 9.9 % | 25 % | PASS |

This regression test is permanent. It is the only end-to-end proof that the topology
generator, the force field and the QM layer agree.

## What a passing QM validation does and does not license

It licenses: the parameters reproduce the QM reference for the terms that were checked,
to the stated tolerance, for the stated property class.

It does **not** license: agreement with experiment. QM is an independent computational
reference, not a measurement. A force field that matches B3LYP perfectly can still
disagree with a calorimeter, and nothing in this engine converts one into the other.

## Failure is recorded, not retried away

A QM job that exits 0 without converging is `FAILED_SCIENTIFICALLY`, not a success. The
first n-butane scan attempt did exactly this — it hit its cycle limit because the scan
started at the eclipsed conformer. The fix was to scan outward from the relaxed geometry,
**not** to relax the convergence criteria.
