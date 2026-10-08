# Architecture

## Design principle

The engine separates four concerns that must never be allowed to blur:

| Layer | Responsibility | May it decide what is true? |
|---|---|---|
| **Reasoning** | propose hypotheses, rank candidate actions | No |
| **Execution** | run deterministic tools, produce artifacts | No |
| **Validation** | apply scientific gates to artifacts | **Yes** |
| **Knowledge** | record provenance, evidence, claims | No |

Only the validation layer decides whether an observation may be believed. A planner
may choose what to investigate; it cannot promote a result. An executor may run
GROMACS; a zero exit code is an input to validation, not a verdict.

## Module map

```text
polymer_engine/
│
├── core/                      no dependencies on any other layer
│   ├── config.py              EngineConfig; env + file + override precedence; Secret
│   ├── errors.py              typed hierarchy separating infrastructure from science
│   ├── logging.py             structured events + credential redaction
│   ├── models.py              Action, Measurement, GateResult, ExecutionRecord, ...
│   ├── provenance.py          Artifact, ProvenanceGraph (lineage DAG)
│   └── units.py               explicit unit conversion; refuses cross-dimension
│
├── providers/                 external data; depends on core
│   ├── base.py                Provider ABC, Capability, ProviderResult
│   ├── http.py                retry/backoff, rate limit, cache, error classification
│   ├── registry.py            construction from config; capability reporting
│   ├── testing.py             FixtureTransport, offline fixtures for tests
│   └── {pubchem, crossref, europe_pmc, openalex, rcsb_pdb,
│         materials_project, charmm_gui}/
│
├── local/                     the machine; depends on core
│   ├── discovery.py           locate gmx/orca/plumed/python; versions, capabilities
│   ├── runner.py              subprocess execution; dry-run is never success
│   └── resources.py           CPU, memory, GPU, disk
│
├── polymer/                   chemistry; depends on core
│   ├── identity.py            canonical repeat units, oligomers, deduplication
│   ├── normalization.py       property names and units from free text
│   ├── descriptors.py         repeat-unit descriptors with declared units
│   ├── taxonomy.py            family classification from an oligomer
│   ├── records.py             the canonical PolymerRecord
│   └── ingestion.py           CSV/JSONL ingestion that reconciles every row
│
├── simulation/                inputs; depends on core
│   ├── archive.py             safe extraction (traversal, links, bombs)
│   ├── formats.py             .gro, .top/.itp, .xvg parsers
│   ├── system.py              artifact discovery, manifests, validation gates
│   ├── mdp.py                 validated GROMACS input generation, per-replica seeds
│   ├── replicas.py            replica materialisation
│   └── umbrella.py            reaction coordinates, window planning, PLUMED input
│
├── analysis/                  results; depends on core + simulation
│   ├── statistics.py          autocorrelation-aware estimators, bootstrap, correlation
│   ├── convergence.py         equilibration detection, drift, replica agreement
│   ├── md.py                  trajectory observables with explicit units
│   └── free_energy.py         overlap diagnostics, WHAM, PMF with uncertainty
│
├── discovery/                 models; depends on core + polymer + analysis
│   ├── qspr.py                surrogates, grouped CV, applicability domain
│   ├── active_learning.py     screening, acquisition, Pareto, diversity
│   └── candidates.py          graph-based mutation with validity checks
│
├── evidence/                  claims; depends on core
│   └── claims.py              Evidence, Claim, ClaimLedger, promotion rules
│
├── executors/                 depends on core + local + simulation + analysis
├── orchestrator/              depends on everything above
│   ├── planner.py             structured assessment and selection
│   ├── campaign.py            spec, build, manifest, fingerprint
│   ├── runner.py              the campaign loop
│   ├── scheduler.py           resource pool, job states, non-oversubscription
│   ├── failure_learning.py    failure classification, ledger, saturating penalties
│   ├── autonomy.py            the research loop and its auditable decisions
│   └── strategy.py            registry and bounded performance learning
│
├── db/store.py                durable SQLite state
└── cli/                       command-line surface
```

Dependencies point downward only. `core` imports nothing from the engine;
`orchestrator` may import anything.

## Data flow

```text
force-field selection       ForceFieldAdvisor -> Confidence
   │                        KNOWN only via a QM validation that passed;
   │                        several covering profiles -> REQUIRES_EXPERT_DECISION
   ▼
provider archive / local directory
   │  safe_extract          rejects traversal, links, device files, bombs
   ▼
discover_artifacts          classify + SHA-256 every file
   │
   ▼
validate_system             GateReport: coordinates, topology, includes,
   │                        finiteness, box, contents-fit-box,
   │                        topology↔coordinate atom-count agreement
   ▼  (blocks if not promotable)
materialize_replicas        per-replica dirs, distinct reproducible seeds
   │
   ▼
generate_stages             EM → NVT → NPT → production .mdp, parameter-validated
   │
   ▼
Scheduler                   reserve -> run -> release; never oversubscribes
   │
   ▼
CampaignRunner              planner selects → executor runs → gates judge
   │                        every decision written to the decision log
   ▼
analyse_series              equilibration detection, effective sample size
   │
   ▼
combine_replicas            replica means; standard error of *replica* means
   │
   ▼
PropertyCalculator          per-property sampling floors, declared units,
   │                        regime and comparability refusals
   ▼
GateReport                  PASS / WARN / FAIL / INCONCLUSIVE
   │
   ├──── not promotable ──→ FailureLedger   classify, count, penalise a strategy
   │                                        (saturating at 0.6, never forbidding)
   ▼
Claim                       promoted only on validated, replicated,
                            uncertainty-bearing evidence
```

Two side paths run off the same trunk:

```text
QMJobSpec ─→ ORCA ─→ parse ─→ five-state classification ─→ MM-vs-QM comparison
                                  (exit 0 is not one of the success conditions)

ReactionCoordinate + justification ─→ windows ─→ PLUMED ─→ COLVAR ─→ WHAM ─→ PMF
                                  (no justification ⇒ zero windows run)
```

## Control loops

There are two, and they are not the same thing.

`orchestrator/runner.py` executes **one campaign**:

1. Load the campaign from the store (resumable).
2. Ask the planner to assess every pending action: feasibility gates first, then
   information gain, cost, uncertainty reduction, design relevance, risk.
3. Record the decision — every candidate, every score, the reason, the tie-break.
4. Transition the execution record `VALIDATED → QUEUED → RUNNING`.
5. Run the deterministic executor.
6. Settle the record from what actually happened: `COMPLETED → PROMOTED` only when
   the result is real, succeeded, and gate-passing; `CANCELLED` for a dry run.
7. Persist actions, observations, gate reports and the manifest.
8. Repeat until no action is feasible.

`orchestrator/autonomy.py` runs the **research loop** above that: it chooses what to
investigate next, records a `ResearchDecision` answering five named questions
(`why_this_candidate`, `why_this_simulation`, `why_now`, `uncertainty_to_reduce`,
`design_decision_at_stake`), executes, and updates its knowledge state **only** from
outcomes the gates found scientifically usable. Everything else updates the failure
ledger instead. See [AUTONOMY.md](AUTONOMY.md).

## The four-valued gate

`GateStatus` is `PASS`, `WARN`, `FAIL`, `INCONCLUSIVE`. A `GateReport` aggregates
worst-case, and an **empty report is `INCONCLUSIVE`** — running no checks is not a
pass. `INCONCLUSIVE` blocks promotion exactly as `FAIL` does, because absence of
evidence is not evidence of adequacy.

This is why the engine can distinguish:

- *the density did not converge* (`FAIL`)
- *we could not tell whether the density converged* (`INCONCLUSIVE`)

Collapsing those two is how an autonomous pipeline starts believing things.

## Execution lifecycle

```text
CREATED ─→ VALIDATED ─→ QUEUED ─→ RUNNING ─→ COMPLETED ─→ PROMOTED
   │           │           │         │           │       └→ REJECTED
   │           │           │         ├→ RETRYING ─→ QUEUED
   ├───────────┴───────────┴─────────┴→ FAILED ─→ RETRYING / REJECTED
   └→ CANCELLED (terminal)
```

`COMPLETED → RUNNING` is deliberately absent. Rerunning finished work requires
`ExecutionRecord.restart()`, which creates a **new** record pointing back at the old
one, so the original outcome is never overwritten. Every transition records
timestamp, actor, reason, inputs and outputs.

## Autonomy boundary

The engine may autonomously:

- generate and retire hypotheses
- rank and select actions
- retrain surrogates and select candidates
- adapt strategy parameters **within declared bounds**
- schedule work

The engine may **not**:

- modify its own source code
- weaken or bypass a validation gate
- promote a claim without qualifying evidence
- invent a force-field parameter, a PMF, an experimental value, or a citation
- select a force field, a level of theory, a reaction coordinate, or an acceptance
  tolerance — those are `REQUIRES-EXPERT-DECISION` and stop the workflow rather than
  defaulting

`StrategyRegistry.adapt_parameters` rejects an out-of-bounds proposal rather than
clamping it, so an attempt to exceed the boundary is visible rather than absorbed.

## Extension points

**A new provider** — subclass `Provider`, declare `capabilities` honestly, register a
factory:

```python
registry.register("in_house", lambda config, http: InHouseProvider(http))
```

The orchestrator does not change. Capabilities are read from the class, so the
registry cannot claim something the implementation does not do.

**A new executor** — subclass `Executor`, set `kind`, register it. There is no
fallback executor: an unknown action kind raises rather than quietly running a stub
and reporting success.

**A new analysis** — return a `Measurement` with units and an uncertainty, or an
explicit non-`KNOWN` determination. Never a bare float.

**A new gate** — return a `GateResult` with a value, threshold, four-valued status
and a diagnostic message. Add it to the relevant `GateReport`. When a report is
assembled from several sources, *concatenate* the gate lists; assigning over
`result.report` discards earlier verdicts, which is how the umbrella justification gate
was once silently lost.

**A new property** — subclass `PropertyCalculator` and declare a `PropertyDefinition`
with real units, the observable, the estimator, the uncertainty method and a
`SamplingRequirement`. The unit must be one `core/units.py` recognises; a regression
test enforces this, because a real quantity declared dimensionless converts silently
against anything.
