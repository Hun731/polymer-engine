# Scientific execution layer — completion report

Version 0.7.0, built on the 0.6.0 consolidation documented in
[AUDIT_REPORT.md](AUDIT_REPORT.md). Status labels: [STATUS_LEGEND.md](STATUS_LEGEND.md).

Every number in this report was produced by running the command stated, on this
machine, and is reproducible with it.

---

## Verified totals

```
$ pytest -rs
1016 passed, 1 skipped, 0 failed          (57 s)

$ pytest --cov --cov-report=term
86 % line coverage over 10,784 statements

$ ruff check src tests
All checks passed!

$ mypy
Success: no issues found in 90 source files
```

| Metric | Value |
|---|---|
| Total tests | **1016** |
| Passed | **1016** |
| Failed | **0** |
| Skipped | **1** |
| Coverage | **86 %** |
| ruff | clean |
| mypy | clean, 90 source files |
| Real GROMACS runs (tests) | **4** — GROMACS 2026.3 |
| Real ORCA runs (tests) | **5** — ORCA 6.1.1 |
| Real PLUMED runs (tests) | **0 of 1** — PLUMED not installed here |

The single skip is `TestRealPlumedExecution`, and its reason is printed on every run:

```
SKIPPED [1] tests/e2e/test_research_pipeline.py:382:
            PLUMED is not on PATH; real-execution tests cannot run in this environment
```

Test distribution: `unit` 238, `providers` 126, `analysis` 124, `simulation` 112,
`orchestration` 107, `qm` 89, `properties` 69, `security` 52, `science` 32,
`property` (Hypothesis) 26, `e2e` 22, `integration` 19.

---

## Capability status

### IMPLEMENTED and TESTED LOCALLY against real external software

| Capability | Evidence |
|---|---|
| ORCA execution and output parsing | 5 tests run ORCA 6.1.1 |
| ORCA relaxed torsion scan → barrier | ethane, **12.01 kJ/mol** vs experimental ~12.1 |
| GROMACS campaign execution | full EM/NVT/NPT/production, both replicas, exit 0 throughout |
| `.mdp` generation | real `gmx grompp`, **zero warnings** |
| Local tool discovery | real versions, GPU/MPI/SIMD/precision capabilities |
| End-to-end benchmark campaign | `examples/benchmark_campaign/run.sh --execute`, 12 steps, exit 0 |

### IMPLEMENTED and TESTED WITH FIXTURES

| Capability | Fixture |
|---|---|
| ORCA five-state classification | real ORCA logs, incl. an exit-0-but-failed run |
| ORCA frequency + thermochemistry parsing | real `freq.out.gz` |
| WHAM / PMF / uncertainty | analytic harmonic PMF, recovered to 0.45 kJ/mol |
| Umbrella execution loop and refinement | analytically sampled windows |
| Correlation-aware statistics | AR(1) theory: φ=0.9 ⇒ τ=9.0, g=19.0 |
| Trajectory observables | analytic trajectories with exact answers |
| 7 data providers | recorded HTTP responses, full failure matrix |
| Property calculators | series with known means, drifts and regimes |

For numerics, a fixture with a known right answer is the *stronger* test. For
integration, it proves nothing — the distinction is kept throughout.

### REQUIRES LOCAL SOFTWARE

| Capability | Tool | Verified here? |
|---|---|---|
| MD execution | GROMACS ≥ 2021 | **yes** — 2026.3 |
| QM execution | ORCA ≥ 5 | **yes** — 6.1.1 |
| Umbrella window execution | PLUMED + PLUMED-enabled GROMACS | **no** — not installed |

### REQUIRES CREDENTIALS

| Capability | Credential | Verified here? |
|---|---|---|
| CHARMM-GUI download / import | account | no — MOCKED only |
| Materials Project queries | `MP_API_KEY` | no — FIXTURE-BASED only |
| Crossref polite pool | `CROSSREF_MAILTO` | no — FIXTURE-BASED only |

No credential exists in the repository, in any log, or in any manifest. Enforced by
`tests/security/test_secrets.py`, which includes a scan of `src/`.

### REQUIRES DOMAIN EXPERT

The engine stops rather than defaulting:

| Decision | What happens without it |
|---|---|
| Force field, when several profiles cover the chemistry | `Confidence.REQUIRES_EXPERT_DECISION`; `require_ready_force_field()` raises |
| QM method and basis set | `TypeError` — required fields, no default |
| MM-vs-QM acceptance tolerance | `passed=None`, `REQUIRES_VALIDATION` |
| Reaction coordinate and its justification | `UmbrellaStatus.REQUIRES_EXPERT_DECISION`; **zero windows run** |
| Whether an MD proxy answers an experimental question | `comparable_to_experiment=False` above 1e4 s⁻¹ |
| De-novo system construction | `BuildStatus.UNSUPPORTED` or `REQUIRES_INPUT` |

### NOT IMPLEMENTED

De-novo system building · MBAR · deformation protocols (anisotropic pressure coupling) ·
Green-Kubo viscosity · thermal and ion conductivity · Yeh–Hummer finite-size correction ·
RESP/ESP charge derivation · glass-transition detection · copolymer sequence handling ·
crosslinked-network handling · coarse-grained models · HPC/SLURM submission ·
LLM hypothesis proposal · automated literature comparison.

CHARMM-GUI job submission is not on this list because it is not a gap: the service
publishes no such endpoint, and the backend returns `UNSUPPORTED` permanently.

Detail and rationale: [ROADMAP.md](ROADMAP.md).

---

## Per-capability breakdown

For each scientific capability, four independent questions. A tick in the first column
and a blank in the third is exactly the situation this table exists to make visible.

| Capability | Workflow exists | Actual execution exists | Numerical validation exists | Acceptance criteria exist |
|---|:--:|:--:|:--:|:--:|
| QM single point | ✓ | ✓ real ORCA | ✓ energy parsed, hartree→kJ/mol | ✓ convergence gates |
| QM optimisation | ✓ | ✓ real ORCA | ✓ convergence table parsed | ✓ geometry gate; silence fails |
| QM frequencies | ✓ | ✓ real ORCA | ✓ fixture with thermochemistry | ✓ imaginary-mode gate |
| QM torsion scan | ✓ | ✓ real ORCA | ✓ ethane 12.01 vs ~12.1 kJ/mol | ✓ scan-completeness gate |
| MM-vs-QM validation | ✓ | ✓ | ✓ Kabsch with reflection guard | **expert-supplied only** |
| Force-field selection | ✓ | ✓ | — *(no numeric output)* | ✓ `Confidence` ladder |
| System import + validation | ✓ | ✓ | ✓ 1×POL(8)+20×SOL(3)=68 | ✓ 12 named gates |
| De-novo system building | ✗ | ✗ | ✗ | ✗ |
| MD equilibration + production | ✓ | ✓ real GROMACS | ✓ `grompp`, zero warnings | ✓ parameter validation |
| Convergence assessment | ✓ | ✓ | ✓ AR(1) theory | ✓ N_eff, drift, replica gates |
| Replica agreement | ✓ | ✓ | ✓ reduced χ² | ✓ threshold 4.0, ≥3 replicas |
| Thermodynamic properties | ✓ | ✓ real MD | ✓ known-mean series | ✓ per-property sampling floors |
| Mechanical properties | ✓ | **analysis only** — no deformation protocol | ✓ synthetic stress–strain | ✓ strain-rate comparability |
| Transport properties | ✓ | ✓ | ✓ analytic MSD, α recovered | ✓ diffusive-regime refusal |
| Structural properties | ✓ | ✓ | ✓ Rg = √2 Å exactly | ✓ sampling + fit-quality gates |
| Umbrella planning | ✓ | ✓ | ✓ spacing from σ=√(kT/k) | ✓ overlap criterion |
| Umbrella execution | ✓ | **✗ never run — PLUMED absent** | ✓ analytic sampling | ✓ justification + execution gates |
| WHAM / PMF | ✓ | ✓ | ✓ analytic PMF, 0.45 kJ/mol | ✓ overlap, convergence, uncertainty |
| Correlation screening | ✓ | ✓ | ✓ known-correlation fixtures | ✓ Bonferroni, min observations |
| Surrogate modelling | ✓ | ✓ | ✓ noise control, grouped CV | ✓ applicability domain |
| Candidate generation | ✓ | ✓ | ✓ validity checks | ✓ constraint + novelty gates |
| Scheduling | ✓ | ✓ | ✓ concurrency counted in a lock | ✓ capacity refusals with reasons |
| Research loop | ✓ | ✓ | — *(no numeric output)* | ✓ usable-only knowledge update |

Two rows deserve to be read twice.

**Umbrella execution** has a workflow, a numerical validation, and acceptance criteria —
and no verified execution. The estimator is right; whether the engine drives GROMACS and
PLUMED correctly is unknown.

**Mechanical properties** has execution in the sense that the analysis runs on real data,
but the *deformation protocol that produces stress–strain data does not exist*. The
engine can analyse a curve you bring it; it cannot yet generate one.

---

## Defects found and fixed in this phase

Each has a regression test. None was found by a test that already existed — which is the
point of writing tests that assert values rather than absence of exceptions.

| # | Defect | Why it mattered |
|---|---|---|
| 1 | `LocalRunner` kept the **last** 200 KB of stdout; ORCA's banner is at the **front** | Every ORCA job over the cap was `UNPARSEABLE`. The 856 KB torsion scan hit it. A resource guard was silently altering scientific input. |
| 2 | Four property calculators declared `units="1"` for volume, K⁻¹, nm², nm²/ps | A "dimensionless" volume converts silently against any other dimensionless number — the exact hidden-unit failure the engine exists to prevent. |
| 3 | `result.report = pmf_gates(...)` overwrote the umbrella execution gates | The justification and execution verdicts vanished from the record of runs that had them. |
| 4 | `classify_failure("SCF did not converge")` → generic `NOT_CONVERGED` | The generic pattern is a substring of the specific one. Wrong classification means wrong recovery advice. |
| 5 | `starting_structures` param declared but never used (caught by ruff ARG002) | A documented feature that did nothing. Now implemented with `assign_starting_structures()` and its own gate. |
| 6 | A test asserted disjoint WHAM windows never converge | They converge in one iteration — the bins decouple. The test encoded a false belief; rewritten to assert what actually distinguishes a good PMF. |
| 7 | Benchmark `run.sh` aborted at step 7 under `set -e` | The gates correctly refused a result and returned 3; the script treated a correct scientific refusal as a script failure. |

---

## The hard safety rules, and where each is enforced

| Rule | Enforcement |
|---|---|
| No fake scientific observations | `Measurement` cannot hold a value unless `determination is KNOWN` |
| No silent fallback from real execution to dry-run | `CommandResult.succeeded` is false unless the process ran; dry runs are `SKIPPED`/`CANCELLED` |
| No promotion of failed simulations | `GateReport.promotable`; `INCONCLUSIVE` blocks exactly as `FAIL` does |
| No treating exit code 0 as scientific success | `QMStatus.FAILED_SCIENTIFICALLY`; the reference fixture is a real exit-0 failure |
| No fabricated force-field parameters | No parameter generation exists; `Confidence.KNOWN` requires a passed QM validation |
| No undocumented API endpoint assumptions | `CharmmGuiImportBackend` returns `UNSUPPORTED` for submission |
| No unsupported PMF interpretation | `pmf_gates` refuses a barrier without overlap; the module reports the surface, not its meaning |
| No hidden unit conversions | Every unit is registered with a dimension; cross-dimension conversion raises; a regression test checks all 18 properties |
| No silent parameter mutation | `StrategyRegistry.adapt_parameters` rejects out-of-bounds proposals rather than clamping |
| No claiming experimental equivalence from an MD proxy | `EXPERIMENTAL_STRAIN_RATE_CEILING`; metrics named `*_proxy` |
| No causal claim from correlation alone | `EvidenceStrength.language` contains no causal verb; Bonferroni by default |
| No replica pseudoreplication | Uncertainty is the standard error of **replica means**; frame counts are corrected by `g = 1 + 2τ_int` |
| No self-modification of scientific validation rules | The reasoning layer has no API that can alter a gate; the validation layer never consults the planner |

---

## What "autonomous" means, precisely

The engine chooses what to investigate, executes it, judges the result, classifies its
own failures, adjusts future rankings, and resumes from durable state — recording, for
every choice, why this candidate, why this simulation, why now, what uncertainty it was
reducing, and what design decision was at stake.

It does not choose a force field, a level of theory, a reaction coordinate, or an
acceptance tolerance. It cannot promote a result the gates refused, and it cannot alter
a gate.

The clearest demonstration is the benchmark campaign. Real GROMACS ran to completion on
both replicas, every stage exited 0 — and the engine refused the result:

```
Only 2.9 effective samples after accounting for autocorrelation (1001 raw frames); need 20
Observable is still drifting: half-to-half change 4.335% exceeds both the 2.000%
    threshold and the statistical noise
2 replica(s) available but 3 are required; reproducibility is not demonstrated
```

In the same run, temperature reached 1001.0 effective samples and potential energy 87.4.
The refusal is property-specific and correct. The correlation screen over 18 polymers
reported `n_tests: 14, significant_after_correction: []`, and the surrogate model
reported `r2 = 0.026`.

A week of compute that teaches the engine nothing is the right outcome when a week of
compute produced nothing that survived validation. An engine that reported a density
from that run would be faster, and wrong.
