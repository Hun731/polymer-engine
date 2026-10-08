# Benchmark campaign

The whole research loop on a system small enough to run in seconds.

```bash
./run.sh              # no external software needed
./run.sh --execute    # runs real ORCA and GROMACS where installed
```

**This demonstrates the machinery, not a scientific conclusion.** The simulation
parameters are deliberately far too short to support a result, and the convergence
gates refuse the output. That refusal is the most important thing the benchmark shows.

## What it contains

| File | Purpose |
|---|---|
| `config.yaml` | every scientific assumption, explicit and recorded in the manifest |
| `candidates.csv` | 18 common polymers with literature Tg, density and modulus |
| `analysis_spec.yaml` | what is measured and what would have to be true for it to count |
| `run.sh` | the twelve-step loop |

`analysis_spec.yaml` deliberately encodes **structural** checks (does the output carry
units, an uncertainty method, a seed record?) rather than expected values. The one
numerical reference it does contain — the ethane rotational barrier — is there to test
the toolchain, not to pre-decide a result.

## Output from a real run

Recorded on a machine with GROMACS 2026.3 and ORCA 6.1.1. Your numbers will differ in
the last digits; the *verdicts* should not.

### Step 2 — curation reconciles

```text
rows_read: 18   accepted: 18   rejected: 0   duplicates: 0   reconciles: true
```

Units are converted on the way in: °C → K, g/cm³ → kg/m³, GPa → MPa. Every row is
accounted for, so "the dataset has 18 polymers" and "the file had 18 rows" are the same
claim.

### Step 4 — QM reference (real ORCA)

```text
ethane rotational barrier: 12.01 kJ/mol      (experimental ~12.1 kJ/mol)
scan points: 5   QM validation passed: true
```

This is a relaxed HF/STO-3G surface scan run by ORCA. Input generation, execution and
parsing all have to be correct for this number to come out. It checks the toolchain.

### Steps 7–8 — real MD, and the gates refusing it

Both replicas ran EM → NVT → NPT → production. **Every GROMACS stage exited 0.** Then:

```text
Only 2.9 effective samples after accounting for autocorrelation (1001 raw frames); need 20
Observable is still drifting: half-to-half change 4.335% exceeds both the 2.000%
    threshold and the statistical noise
2 replica(s) available but 3 are required; reproducibility is not demonstrated
```

A thousand frames were written. After correcting for autocorrelation there are **2.9
independent measurements** of the density. The engine refuses to call that a result.

Note the contrast in the same report:

```text
temperature: 1001.0 effective samples from 1001 frames (statistical inefficiency 1.0)
potential:     87.4 effective samples from 1001 frames (statistical inefficiency 11.4)
density:        2.9 effective samples from 1001 frames
```

Different observables decorrelate at very different rates from the *same* trajectory.
A single frame count would hide that completely.

### Step 9 — correlation screening

```text
n_tests: 14
significant_after_correction: []
```

Fourteen descriptor–density tests on 18 polymers, and **nothing survives** the
multiple-comparison correction. Reporting the best of fourteen correlations at p < 0.05
without that correction is how spurious structure–property "laws" get published.

### Step 10 — surrogate model

```text
r2 = 0.026
```

Essentially zero, which is the honest answer for 18 polymers and 15 descriptors under
grouped cross-validation. Getting a flattering number instead would take only fitting
the scaler before splitting, or splitting by row rather than by canonical identity. The
engine closes both routes structurally.

## What would make this a real campaign

| Setting | Benchmark | Realistic |
|---|---|---|
| production length | 0.02 ns | 50–500 ns |
| replicas | 2 | ≥ 3 |
| system | 68-atom fixture | thousands of atoms, real chains |
| force field | `test-UA` (the fixture's own parameters) | a real, QM-validated force field |
| dataset | 18 polymers | hundreds |

Change those in `config.yaml` and `candidates.csv`. Nothing in the code changes.

## Reading the exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | engine or scientific failure |
| 2 | usage error |
| 3 | **a validation gate did not pass** |

Step 8 exits 3 on a real run. That is the benchmark working, not the benchmark failing.
