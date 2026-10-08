# Autonomy

Status labels: [STATUS_LEGEND.md](STATUS_LEGEND.md)

**An orchestrator is not autonomy.** A loop that picks the next item off a list and runs
it is a queue with ambition. This document states what the engine actually decides, what
it records about deciding, and — at the end — what it does not do.

---

## Capability table

| Capability | Module | Status |
|---|---|---|
| Deterministic action planning and scoring | `orchestrator/planner.py` | REAL |
| Strategy registry and ranking | `orchestrator/strategy.py` | REAL |
| Iterative research loop with checkpointing | `orchestrator/autonomy.py` | REAL |
| Auditable decision records | `orchestrator/autonomy.py` | REAL |
| Failure classification and learning | `orchestrator/failure_learning.py` | REAL |
| Active learning / acquisition | `discovery/active_learning.py` | REAL, OPTIONAL (scikit-learn) |
| Candidate generation by graph mutation | `discovery/candidates.py` | REAL, OPTIONAL (RDKit) |
| Correlation screening with multiplicity control | `science/correlation.py` | REAL |
| Surrogate models with leakage protection | `science/models.py`, `discovery/qspr.py` | REAL, OPTIONAL (scikit-learn) |
| LLM hypothesis proposal | — | NOT IMPLEMENTED |
| Automated literature comparison | — | NOT IMPLEMENTED |
| Self-modification of validation rules | — | **NEVER** — see the last section |

---

## The five questions

Every iteration of `ResearchLoop` emits a `ResearchDecision` that answers five questions
by name. They are separate fields, not a free-text blob, so a decision can be audited by
someone who was not present when it was made:

| Field | Answers |
|---|---|
| `why_this_candidate` | why this polymer rather than another |
| `why_this_simulation` | why this calculation rather than another |
| `why_now` | why at this point in the campaign |
| `uncertainty_to_reduce` | what is currently unknown that this should narrow |
| `design_decision_at_stake` | what downstream choice depends on the answer |

Alongside them the record carries the estimated cost, the estimated information gain,
the number of candidates considered, the resource snapshot at the moment of choosing,
the strategy and its score, and the failure penalty applied.

```bash
polymer-engine research decisions          # the decision log
polymer-engine research state loop.json    # where a loop got to
polymer-engine research failures           # what has gone wrong, and how often
```

A campaign whose decisions cannot be reconstructed afterwards is not reproducible, no
matter how deterministic the code was.

---

## Only real results change what the engine believes

`_update_knowledge()` acts **only** on outcomes where `scientifically_usable` is true.

This is the join between the autonomy layer and the validation layer, and it runs one
way. A simulation that ran, exited 0, and produced a number the gates refused does not
move the knowledge state. It is not recorded as a weak result, a provisional value, or a
prior. It updates the *failure* ledger instead.

The consequence is worth stating plainly: **the engine can spend a week of compute and
learn nothing.** That is the correct behaviour when a week of compute produced nothing
that survived validation.

---

## Learning from failure without locking itself out

`classify_failure()` maps an error to one of twelve `FailureType`s, each carrying
`recoverable` and `suggested_action`:

| Type | Recoverable? |
|---|---|
| `SYSTEM_INVALID`, `PARAMETER_INVALID`, `SIMULATION_DIVERGED` | no — it will fail again identically |
| `INSUFFICIENT_SAMPLING`, `NOT_CONVERGED`, `TIMEOUT`, `RESOURCE_EXHAUSTED` | yes — more, or longer, would plausibly work |
| `REPLICA_DISAGREEMENT`, `POOR_OVERLAP`, `QM_NOT_CONVERGED` | depends on the diagnosis |

Pattern matching is ordered specific-before-generic. `"SCF did not converge"` must
classify as `QM_NOT_CONVERGED`, not the generic `NOT_CONVERGED`, so the SCF patterns are
tested first — a regression test pins this, because the generic pattern is a substring
of the specific message.

`FailureLedger.penalty_for()` deprioritises a strategy that keeps failing on a polymer
family, and **saturates at 0.6**:

```python
return min(0.6, 0.15 * len(matching) + 0.05 * unrecoverable)
```

It never reaches 1.0, so a strategy can always be chosen again. A strategy that failed on
three polyesters may be right for the fourth, and an engine that locks itself out of an
approach because of a small sample has learned the wrong lesson. A pattern also requires
`MIN_OCCURRENCES_FOR_PATTERN = 3` before any penalty applies: twice is a coincidence.

---

## Correlation is not causation, and the code will not say otherwise

`science/correlation.py` screens for structure–property relationships. Two rules are
enforced in the code rather than left to the reader:

**Multiplicity is corrected.** Screening 14 descriptors against a target is 14 tests, and
at α = 0.05 you expect roughly one spurious hit. `CorrelationMatrix.significant()` applies
a Bonferroni correction by default and reports `n_tests` alongside the result, so the
multiplicity is visible even when the correction is declined.

**The language is bounded.** `EvidenceStrength.language` maps every strength level onto
a phrase that stops short of causation:

| Strength | Phrase |
|---|---|
| `NONE` | "shows no detectable relationship with" |
| `INSUFFICIENT` | "cannot be related to (insufficient data)" |
| `ASSOCIATION` | "is associated with" |
| `ROBUST_ASSOCIATION` | "is robustly associated with" |
| `PREDICTIVE` | "predicts" |

"Causes", "drives", "determines" and "leads to" appear nowhere. Even `PREDICTIVE` claims
only predictive value, which is a statement about the model, not about mechanism. This is
not stylistic: a report saying a descriptor *drives* a property has made a causal claim
from observational data, and once that phrasing is in a document nobody reconstructs
where it came from.

A correlation may direct the next experiment. It may not become a conclusion.

The benchmark campaign in `examples/benchmark_campaign/` illustrates the point on real
output: 14 tests, `significant_after_correction: []`. Eighteen polymers is not enough
data, and the engine says so rather than reporting the largest raw coefficient.

---

## Surrogate models and leakage

Structure–property models leak in ways that are easy to miss and produce impressive
cross-validation scores from nothing. `science/models.py` defends on three axes at once:

* **Grouped k-fold by canonical polymer id** — the same polymer under two names cannot sit
  on both sides of a split.
* **Structural similarity clustering** — near-duplicate repeat units are grouped, because
  a trivially modified analogue is not an independent test case.
* **Simulation source id** — results from the same simulation batch are grouped, so a
  shared systematic error cannot be scored as predictive skill.

Preprocessing is fitted **per fold**, never on the full dataset. A scaler fitted on all
the data before splitting has already leaked the test distribution.

`UNCERTAINTY_SOURCE` states, per model kind, where an uncertainty came from — or that
there isn't one:

| Model | Uncertainty source |
|---|---|
| linear, ridge, gradient boosting | `none` |
| random forest | `ensemble spread across trees (relative, not calibrated)` |
| Gaussian process | `posterior standard deviation (calibrated under its own prior)` |

Ensemble spread ranks candidates usefully; it is not a calibrated confidence interval, and
presenting it as one would be a fabricated uncertainty. A model with `none` reports a
prediction without an error bar rather than inventing one.

The benchmark's model reports `r2 = 0.026` on 18 polymers. That is the honest number.

---

## What "autonomous" means here, precisely

The engine can, without a human in the loop:

* choose which candidate to investigate next, from a scored ranking with a recorded
  rationale;
* choose which calculation to run, subject to resource availability;
* execute it, detect that it failed, classify the failure, and adjust future rankings;
* decide that a result is not usable and decline to learn from it;
* checkpoint, stop, and resume from durable state.

It **cannot**, by design:

* promote a result the gates refused;
* choose a force field, a level of theory, a reaction coordinate, or an acceptance
  tolerance;
* weaken, disable, or rewrite a validation rule;
* claim experimental equivalence for an MD proxy;
* turn a correlation into a causal statement;
* modify its own source code.

The last two lines of defence are structural, not procedural. The reasoning layer has no
API by which it can alter a gate, and the validation layer never consults the planner.
A planner may choose what to investigate; it cannot decide what is true.
