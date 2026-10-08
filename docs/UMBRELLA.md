# Umbrella sampling and free energy

Status labels: [STATUS_LEGEND.md](STATUS_LEGEND.md)

Enhanced sampling is where a pipeline is most likely to produce a confident, precise,
wrong number. A PMF always looks like a curve. Nothing about its appearance tells you
whether the windows overlapped, whether the sampling was converged, or whether the
reaction coordinate meant anything.

This module treats all three as gates, and refuses a PMF that fails them.

---

## Capability table

| Capability | Module | Status |
|---|---|---|
| Reaction-coordinate definition (5 CV kinds) | `simulation/umbrella.py` | REAL |
| Window planning from `σ = √(kT/k)` | `simulation/umbrella.py` | REAL |
| PLUMED input generation | `simulation/umbrella.py` | REAL |
| Window execution loop and per-window state | `simulation/umbrella_execution.py` | REAL |
| Adaptive gap refinement | `simulation/umbrella_execution.py` | REAL |
| Starting-structure assignment | `simulation/umbrella_execution.py` | REAL |
| COLVAR reading | `simulation/umbrella_execution.py` | REAL |
| WHAM, PMF, uncertainty | `analysis/free_energy.py` | FIXTURE-BASED (analytic PMF) |
| Overlap and convergence gates | `analysis/free_energy.py` | REAL |
| `gmx mdrun -plumed` execution backend | `simulation/umbrella_execution.py` | REQUIRES-LOCAL-SOFTWARE — **never executed here** |
| Choice of reaction coordinate | — | REQUIRES-EXPERT-DECISION |
| MBAR | — | NOT IMPLEMENTED |
| Replica exchange / metadynamics | — | NOT IMPLEMENTED |

**Be clear about the last real-execution row.** Everything above it is verified. The
one thing that is not verified on this machine is the runner that drives
`gmx grompp` + `gmx mdrun -plumed`, because PLUMED is not installed here.
`TestRealPlumedExecution` in `tests/e2e/test_research_pipeline.py` exercises exactly
that path and reports, on every run:

```
SKIPPED [1] PLUMED is not on PATH; real-execution tests cannot run in this environment
```

The estimator is validated analytically; the integration is not validated at all.
Those are different claims and this document does not merge them.

---

## Nothing runs without a justification

`UmbrellaCampaign.run()` requires an `UmbrellaJustification` with seven fields
answered by a person:

| Field | The question it answers |
|---|---|
| `question` | what is actually being asked |
| `reaction_coordinate` | what the CV is, in words |
| `physical_interpretation` | what the resulting free energy *means* |
| `expected_observable` | what feature of the PMF answers the question |
| `reason_for_method` | why equilibrium MD is insufficient |
| `starting_state` | what the windows start from |
| `endpoint_definition` | where the coordinate ends and why |

Without a complete justification the campaign returns
`UmbrellaStatus.REQUIRES_EXPERT_DECISION` and **runs zero windows**. It does not run
them and mark the result provisional; it does not run them and warn. A reaction
coordinate nobody can justify produces a free energy nobody can interpret, and the
cheapest moment to discover that is before the compute is spent.

The justification travels into the campaign manifest, so a PMF in the record can always
be traced back to the question it was meant to answer.

---

## Window spacing is derived, not chosen

For a harmonic bias of force constant `k` at temperature `T`, the width of the sampled
distribution in each window is

```
σ = √(kT / k)
```

Windows spaced much further apart than `σ` do not overlap, and WHAM has no information
with which to stitch them. The planner therefore derives spacing from `k` and `T`:

```
MAX_SPACING_SIGMA = 2.0
```

This is the one geometry parameter in the module that is physics rather than
convention, and the code says so at the definition site. Requesting a spacing wider
than `2σ` produces a planning warning before anything runs.

---

## The gates

The first four are applied by `UmbrellaCampaign` during execution; the rest by
`pmf_gates()` in `analysis/free_energy.py` after WHAM. The campaign report is built
so that **execution-time gates survive** rather than being overwritten by the PMF
report — an earlier version assigned `result.report = pmf_gates(...)` and silently
discarded the justification and execution verdicts.

| Gate | Checks | Fails when |
|---|---|---|
| `umbrella:justification` | a person justified the method | no justification supplied |
| `umbrella:windows_ran` | every planned window actually executed | a window errored or never ran |
| `umbrella:executed` | execution was enabled at all | inputs written, nothing run (`NOT_EXECUTED`) |
| `umbrella:starting_structures` | windows started from appropriate structures | falls to WARN, not FAIL — see below |
| `umbrella:windows_sampled` | each window produced usable samples | a window is empty or below the sampling floor |
| `umbrella:window_overlap` | adjacent-window histogram overlap | any adjacent pair below `min_pair_overlap` (default 0.10) |
| `umbrella:wham_converged` | the self-consistent iteration converged | the iteration cap was reached |
| `umbrella:sampling_converged` | first half vs second half of the sampling | half-split PMF difference exceeds 2.0 kJ/mol (≈ 0.8 kT at 300 K) |
| `umbrella:uncertainty_estimated` | an uncertainty exists at all | bootstrap did not run |
| `umbrella:refinement` | gaps were closed within the refinement bound | `REFINEMENT_EXHAUSTED` |

`starting_structures` is a **WARN**, deliberately. Starting every window from the same
structure biases the early frames, but per-window equilibration trimming already
removes them, and the evidence that actually matters — overlap and half-split
convergence — is measured directly. Escalating this to FAIL would block campaigns that
the real diagnostics say are fine.

---

## WHAM

Implemented in log space and gauge-fixed to the first window, so the free-energy
offsets have a defined zero rather than drifting. `PMF = −kT ln p`. Uncertainty comes
from a **block** bootstrap, because umbrella samples within a window are strongly
autocorrelated and a naive bootstrap would understate the error by the square root of
the statistical inefficiency.

Bins that no window sampled adequately are masked (`well_sampled`, threshold
`WELL_SAMPLED_FRACTION = 0.02`) rather than extrapolated into. An unsampled region of a
PMF is not a flat region.

### The iteration cap is a measurement

```python
#: Measured on this implementation: well-overlapped windows (0.08 nm spacing,
#: ~1.6 sigma) converge in ~250 iterations and sparse-but-overlapping windows
#: (0.3 nm) in ~4,500. Windows that do not overlap have no solution to converge
#: to and will run forever, so 20,000 is comfortably above any well-posed problem
#: while failing fast on an ill-posed one.
DEFAULT_WHAM_MAX_ITERATIONS = 20_000
```

Those numbers were measured, not guessed. An earlier version carried a stall detector
alongside the cap; profiling showed it never fired on any well-posed or ill-posed input
tested, so it was removed. Dead safety machinery is not safety.

### Convergence is not correctness

`test_wham_converging_is_not_evidence_the_pmf_is_right` exists because the intuitive
assumption is false. Give WHAM a set of **disjoint** windows and it converges in a
single iteration — the bins decouple entirely, so there is nothing to iterate. The
resulting PMF is meaningless.

The test asserts what actually matters: WHAM converged, `pmf.trustworthy is False`, and
the overlap gate FAILs. Convergence of the estimator is a numerical fact. Whether the
answer means anything is a separate question, decided by the gates.

---

## Adaptive refinement

When a coverage gap is detected between windows, the campaign inserts a new window at the **midpoint
of the gap** and reruns **only the new window**. Existing windows keep their samples.
Refinement is bounded; exhausting the bound yields `REFINEMENT_EXHAUSTED`, which is a
recorded outcome, not a silent stop.

---

## Validation against a known answer

The estimator is checked against a problem with an exact solution: windows sampled
analytically from a harmonic potential of known force constant and minimum. The
recovered PMF must match `½k(x − x₀)²` to within 1.5 kJ/mol across every well-sampled
bin. Earlier validation against the same analytic surface recovered it to 0.45 kJ/mol.

This is `FIXTURE-BASED` in the legend's sense and, for the numerics, that is the
*stronger* test: a PMF from a real trajectory has no known right answer to check
against.

---

## CLI

```bash
polymer-engine umbrella plan windows/ \
    --min 0.6 --max 1.4 \
    --group-a 1-100 --group-b 101-200 \
    --justification "Interchain separation at the interface" \
    --strict

polymer-engine umbrella analyze windows/ -o pmf.json
```

`--justification` is a **required** option on `plan`, mirroring the API. `--strict`
refuses a window spacing that cannot overlap rather than warning about it. Temperature
and force constant come from the configuration, so they are recorded in the manifest
rather than retyped per invocation.

`analyze` operates on COLVAR files you already have, so the analysis half of the module
is usable without PLUMED installed.

---

## What this module will not do

* Run without a justified reaction coordinate.
* Report a PMF whose windows do not overlap.
* Extrapolate into unsampled coordinate regions.
* Report a free energy without an uncertainty and the method that produced it.
* Interpret the PMF for you. It reports the surface, the uncertainty and the gate
  verdicts; what the well depth *means* for your system is your claim to make.
