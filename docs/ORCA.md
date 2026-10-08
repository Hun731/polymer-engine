# ORCA / quantum-chemistry layer

Status labels: [STATUS_LEGEND.md](STATUS_LEGEND.md)

The QM layer exists for one reason: **a force field is a claim, and a claim needs
evidence.** It generates ORCA inputs, runs them, parses the output, and compares
molecular-mechanics behaviour against the QM reference under tolerances that a human
chose and recorded.

It is not a general-purpose QM front end. It supports the calculation types the
force-field-validation workflow needs, and refuses anything else explicitly.

---

## Capability table

| Capability | Module | Status |
|---|---|---|
| Input generation (6 job kinds) | `qm/orca_input.py` | REAL |
| Job execution | `qm/orca_runner.py` | REAL, REQUIRES-LOCAL-SOFTWARE |
| Output parsing and status classification | `qm/orca_parser.py` | REAL + FIXTURE-BASED |
| Relaxed torsion scan → barrier | `qm/orca_parser.py`, `qm/validation.py` | REAL |
| Frequency analysis, imaginary-mode check | `qm/orca_parser.py` | FIXTURE-BASED |
| Mulliken / Löwdin population charges | `qm/orca_parser.py` | FIXTURE-BASED |
| MM-vs-QM geometry and torsion comparison | `qm/validation.py` | REAL |
| Choice of method and basis set | — | REQUIRES-EXPERT-DECISION |
| Acceptance tolerances | `qm/validation.py` | REQUIRES-EXPERT-DECISION |
| Implicit solvation (CPCM / SMD) | `qm/orca_input.py` | REAL (input generation); never run here |
| RESP / ESP charge derivation | — | NOT IMPLEMENTED |
| Automated basis-set convergence study | — | NOT IMPLEMENTED |
| Explicit-solvent QM/MM | — | NOT IMPLEMENTED |

Verified against **ORCA 6.1.1** on the development machine. Five tests execute ORCA
for real and are skipped with an explicit reason where it is absent.

---

## The level of theory is yours to choose

`QMJobSpec` has **no default method and no default basis set**. Both are required
fields. This is the single most consequential decision in a QM calculation and the
engine will not make it silently:

```python
QMJobSpec(structure=..., kind=JobKind.OPTIMIZATION)          # TypeError
QMJobSpec(structure=..., kind=JobKind.OPTIMIZATION,
          method="B3LYP", basis="def2-TZVP")                  # explicit, recorded
```

The same applies at the CLI:

```bash
polymer-engine qm run molecule.xyz \
    --kind optimization --method B3LYP --basis def2-TZVP --execute
```

Omitting either is a usage error, not a fallback. Note `--execute`: without it the input
file is written and validated but **nothing runs**, and the command exits non-zero
because nothing ran is not success.

Solvation follows the same rule. `--solvation-model` and `--solvent` must be given
together; either alone is rejected, because "CPCM" without a solvent and a solvent
without a model are both incomplete specifications rather than things to guess at.

`electron_consistency()` rejects charge/multiplicity combinations that are impossible
for the electron count — an even-electron doublet is a typo, not a calculation.

---

## Exit code 0 is not success

This is the rule the whole module is built around, and it is not hypothetical. The
fixture `tests/fixtures/orca/scan_nonconverged.out` is a **real ORCA run captured on
this machine** that:

* exited with status **0**,
* printed `ORCA finished by error termination`,
* never printed the normal-termination banner,
* produced a partially converged geometry.

A pipeline keyed on the return code would have promoted it.

`parse_orca_output()` classifies a log into one of five states:

| `QMStatus` | Meaning |
|---|---|
| `COMPLETED` | terminated normally, SCF converged, every expected quantity present |
| `FAILED_SCIENTIFICALLY` | ORCA finished, but the science did not: SCF did not converge, the geometry did not converge, requested frequencies are missing, or no final energy was produced |
| `FAILED_TERMINATION` | ORCA reported an error termination |
| `INCOMPLETE` | no normal-termination banner: killed, truncated, out of time |
| `UNPARSEABLE` | the file is not an ORCA log |

Classification is *expectation-aware*. Asking for an optimisation and getting no
geometry-convergence banner is `FAILED_SCIENTIFICALLY`, because "the file never said
whether it converged" is not evidence that it did:

```python
if expect_geometry:
    if result.geometry_converged is False:
        return QMStatus.FAILED_SCIENTIFICALLY
    if result.geometry_converged is None:      # silence is not consent
        return QMStatus.FAILED_SCIENTIFICALLY
```

```bash
polymer-engine qm parse run.out --expect-geometry   # exit 3 when the science failed
```

---

## Parsed quantities

Every regular expression matches a documented ORCA output marker. Nothing is inferred
from position in the file.

| Quantity | Marker |
|---|---|
| Final energy | `FINAL SINGLE POINT ENERGY` |
| SCF convergence | `SCF CONVERGED AFTER n CYCLES` |
| Geometry convergence | `THE OPTIMIZATION HAS CONVERGED` |
| Normal termination | `****ORCA TERMINATED NORMALLY****` |
| Error termination | `ORCA finished by error termination` |
| Frequencies | `VIBRATIONAL FREQUENCIES` block, `n: x cm**-1` |
| Thermochemistry | `Zero point energy`, `Total Enthalpy`, `Final Gibbs free energy` |
| Scan surface | `The Calculated Surface using the 'Actual Energy'` |
| Program version | `Program Version` |

Energies are parsed in hartree and converted with a single named constant,
`HARTREE_TO_KJ_MOL = 2625.4996394799`. There is no second conversion path.

Compressed logs (`.gz`) are read transparently, which is how the full 856 KB scan and
frequency fixtures stay in the repository.

---

## A truncation bug worth remembering

`LocalRunner` originally kept only the **last** 200 KB of stdout, to bound memory on a
runaway process. ORCA prints its banner and version at the **front** of the log, so
every ORCA job long enough to exceed the cap arrived at the parser with its header
removed and was classified `UNPARSEABLE`. The 856 KB torsion scan hit this exactly.

The fix made the cap a parameter, and `ORCARunner` sets it to `None`:

```python
def _capture(self, stream: str | None) -> str:
    text = stream or ""
    if self.capture_limit is None:
        return text
    return text[-self.capture_limit :]
```

Three regression tests in `TestOutputCapture` cover it. The lesson generalises: a
resource guard that silently changes the *content* of scientific input is a scientific
bug, not an operational one.

---

## Fixtures

`tests/fixtures/orca/` contains real ORCA 6.1.1 output, not hand-written text.

| File | What it is |
|---|---|
| `h2.out.gz` | HF/STO-3G single point on H₂ |
| `opt.out.gz` | converged geometry optimisation |
| `freq.out.gz` | frequency calculation with thermochemistry |
| `scan_ethane.out` | relaxed torsion scan, 5 points, barrier 12.01 kJ/mol |
| `scan_nonconverged.out` | the exit-0-but-failed case described above |

`scan_nonconverged.out` is an excerpt of a much longer log. The elided region is marked
in the file itself:

```
[... N lines of repeated SCF/geometry cycles elided from the verbatim ORCA log
     to keep this fixture small ...]
```

Fixtures are never edited to make a test pass. If a fixture disagrees with the parser,
the parser is wrong.

---

## The ethane check

The ethane rotational barrier is the module's end-to-end sanity check: a relaxed
torsion scan about the C–C bond, run for real, gives **12.01 kJ/mol** against an
experimental value of about 12.1 kJ/mol.

This validates the toolchain — input generation, execution, parsing, unit conversion,
barrier extraction — on a system whose answer is known. It is *not* a validation of
any force field, and the code does not present it as one.

An earlier attempt used the torsion indices `D 0 1 2 3`, which on this atom ordering is
not the C–C dihedral at all. ORCA ran to its cycle limit and error-terminated. That run
became `scan_nonconverged.out`.

---

## Acceptance criteria are a decision, not a default

`compare_torsion_profiles()` and `compare_geometries()` accept an optional
`AcceptanceCriteria`. **Without one, they return `passed=None`, the determination `REQUIRES_VALIDATION`,
and a note reading `REQUIRES_EXPERT_DECISION: no ... tolerance was configured`.** The engine computes the deviation; it does not decide
whether the deviation is acceptable for your application.

Two named presets exist so that a manifest can record *which convention was adopted*:

| Preset | RMSD (Å) | Torsion RMSE (kJ/mol) | Barrier error (kJ/mol) | Relative |
|---|---|---|---|---|
| `general_organic_forcefield()` | 0.20 | 2.0 | 4.0 | 25% |
| `conformational_free_energy()` | 0.15 | 1.0 | 2.0 | 15% |

Each carries a `justification` string that travels into the manifest. The second is
tighter because conformer populations depend exponentially on relative energies: at
300 K, 4 kJ/mol is roughly a factor of five in a population ratio.

Neither preset is a derivation. They are conventions from the parameterisation
literature, and the documentation says so rather than presenting them as physics.

---

## Geometry comparison

`compare_geometries()` superposes with the Kabsch algorithm, including the reflection
guard: when the determinant of the rotation is negative the smallest singular direction
is flipped, so an improper rotation can never masquerade as a good fit. Without it, a
mirror image of a chiral molecule superposes perfectly.

---

## What this layer will not do

* Choose a functional, a basis set, or a solvation model.
* Decide that an MM-vs-QM deviation is acceptable.
* Report a barrier from a scan that did not converge.
* Derive force-field parameters. Comparing MM to QM is validation; generating
  parameters from QM is a separate problem this engine does not attempt.
* Claim that agreement with QM implies agreement with experiment.
