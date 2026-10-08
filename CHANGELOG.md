# Changelog

## 0.7.0 — the scientific execution layer

Built on top of the 0.6.0 consolidation, without restarting it. Where 0.6.0 made the
orchestration framework trustworthy, 0.7.0 gives it real computational work to
orchestrate: quantum chemistry, force-field validation, system construction boundaries,
umbrella execution, scheduling, and eighteen property calculators.

### New subsystems

- **`qm/`** — ORCA input generation (6 job kinds), execution, output parsing, and a
  five-state scientific classification. `COMPLETED`, `FAILED_SCIENTIFICALLY`,
  `FAILED_TERMINATION`, `INCOMPLETE`, `UNPARSEABLE`. Method and basis set are **required
  fields with no default**; the engine does not choose a level of theory. MM-vs-QM
  comparison requires an explicit `AcceptanceCriteria` or returns
  `REQUIRES_VALIDATION`. Verified against real ORCA 6.1.1: the ethane rotational barrier
  comes out at 12.01 kJ/mol against an experimental ~12.1.
- **`properties/`** — 18 calculators across thermodynamic, mechanical, structural,
  transport and intermolecular classes. Each declares its units, observable, estimator,
  uncertainty method and sampling requirement as data. Sampling floors are per-property:
  `bulk_modulus` needs 200 effective samples where `density` needs 20, because a variance
  converges far more slowly than a mean.
- **`simulation/builder.py`, `simulation/forcefield.py`** — an explicit system-construction
  boundary and a force-field advisor. `Confidence.KNOWN` is reachable only through a QM
  validation that passed; multiple covering profiles yield `REQUIRES_EXPERT_DECISION`
  rather than a ranked pick.
- **`simulation/umbrella_execution.py`** — the umbrella window execution loop, adaptive
  gap refinement, and starting-structure assignment. A campaign without a seven-field
  `UmbrellaJustification` runs **zero windows**.
- **`orchestrator/scheduler.py`** — a resource-aware scheduler over CPUs, GPUs, memory and
  disk. `exceeds_capacity()` returns a reason string, and states that VRAM cannot be
  pooled across cards.
- **`orchestrator/failure_learning.py`** — twelve failure types with recoverability and
  suggested actions; penalties saturate at 0.6 so a strategy is deprioritised, never
  forbidden.
- **`orchestrator/autonomy.py`** — the research loop. Every iteration records a
  `ResearchDecision` answering five named questions, and the knowledge state is updated
  **only** from outcomes the gates found scientifically usable.
- **`science/`** — correlation screening with Bonferroni multiplicity control and a
  reporting vocabulary containing no causal verb; surrogate models with three-axis leakage
  protection (polymer id, structural similarity, simulation source) and per-fold
  preprocessing.

### Scientific correctness

- **A resource guard was silently corrupting scientific input.** `LocalRunner` kept only
  the *last* 200 KB of stdout. ORCA prints its banner and version at the *front* of the
  log, so every ORCA job long enough to exceed the cap — including the 856 KB torsion
  scan — arrived at the parser with its header removed and was classified `UNPARSEABLE`.
  `capture_limit` is now a parameter and `ORCARunner` sets it to `None`. Three regression
  tests in `TestOutputCapture`.
- **Four property calculators declared `units="1"` while carrying real quantities** — a
  volume in nm³, a thermal expansion coefficient in K⁻¹, an MSD in nm², a diffusivity in
  nm²/ps — with the true unit stated only in prose. A "dimensionless" volume converts
  silently against any other dimensionless number. `core/units.py` now registers `volume`,
  `area`, `diffusivity` and `inverse_temperature`, and a regression test asserts that every
  registered property declares a unit the registry recognises.
- **The umbrella report overwrote its own execution gates.** `result.report = pmf_gates(...)`
  discarded the justification and execution verdicts recorded earlier in the run. The report
  is now built by concatenation.
- **`classify_failure("SCF did not converge")` returned the generic `NOT_CONVERGED`**,
  because the generic pattern is a substring of the specific message. SCF patterns are now
  matched first, with a regression test.
- **A test asserted that disjoint umbrella windows never converge.** They converge in a
  single iteration — the bins decouple entirely, so there is nothing to iterate. Rewritten
  as `test_wham_converging_is_not_evidence_the_pmf_is_right`, which asserts that WHAM
  converged *and* the PMF is untrustworthy *and* the overlap gate FAILs.
- **`starting_structures` was an `INCONCLUSIVE` gate that blocked promotion.** Per-window
  equilibration trimming already removes the biased frames, and overlap plus half-split
  convergence measure the thing that actually matters. Downgraded to WARN — the one gate
  relaxation in this release, and it is justified by what the other gates measure directly.

### Performance

- **WHAM: 49 s → 7.9 s.** The inner loop over windows was vectorised. The iteration cap was
  then set from measurement rather than guesswork: well-overlapped windows converge in ~250
  iterations, sparse-but-overlapping in ~4,500, so `DEFAULT_WHAM_MAX_ITERATIONS = 20_000`
  sits comfortably above any well-posed problem while failing fast on an ill-posed one. A
  stall detector added alongside it was **removed** after profiling showed it never fired on
  any input tested — dead safety machinery is not safety.

### CLI

Three new command groups: `qm` (`run`, `parse`), `property` (`list`, `compute`), and
`research` (`failures`, `decisions`, `state`, `correlate`). `qm parse` exits 3 for a job
that exited 0 without converging.

### Testing

1016 tests (up from 653), 86 % coverage, ruff and mypy clean over 90 source files.

Ten tests execute an external scientific tool for real: 4 GROMACS, 5 ORCA, and 1 PLUMED.
The PLUMED test — the `gmx mdrun -plumed` umbrella backend — **has never run**, because
PLUMED is not installed on the development machine, and it reports that reason on every
invocation rather than being quietly absent. The umbrella estimator is validated against an
analytic PMF; the umbrella execution backend is not validated at all.

`tests/fixtures/orca/` holds real ORCA 6.1.1 output including `scan_nonconverged.out`: a
run that exited **0**, printed `ORCA finished by error termination`, and never printed the
normal-termination banner.

### Documentation

`docs/STATUS_LEGEND.md` defines REAL / FIXTURE-BASED / MOCKED / OPTIONAL /
REQUIRES-LOCAL-SOFTWARE / REQUIRES-CREDENTIALS / REQUIRES-EXPERT-DECISION / NOT IMPLEMENTED,
and every capability table in the documentation set uses them. New: `ORCA.md`,
`UMBRELLA.md`, `SIMULATION.md`, `AUTONOMY.md`. `SCIENTIFIC_VALIDATION.md` became
`SCIENCE.md`. `ROADMAP.md` was rewritten — it listed six now-implemented subsystems as
absent.

`examples/benchmark_campaign/` is a runnable 12-step campaign with its real recorded output.

## 0.6.0 — architectural consolidation and scientific hardening

A takeover audit of the 0.5 tree. The engine was reorganised into a single canonical
implementation per responsibility, and a number of defects that would have produced
wrong scientific answers were fixed.

### Architecture

- **Removed duplicate implementations.** The legacy `science/`, `runtime/`, `domain/`,
  `planning/`, `knowledge/`, `validation/` and `agents/` trees, a second `cli.py`, a
  second engine in `core/engine.py`, and four superseded executor modules were deleted.
  There had been three campaign builders wrapping one another, two gate systems with
  incompatible types, two `sha256_file` implementations and two utility-scoring
  functions.
- Restructured into `core / providers / local / polymer / simulation / analysis /
  discovery / orchestrator / evidence / executors / db / cli`, with dependencies
  pointing downward only.
- `pyproject.toml` could not build: the package is `polymer_engine` but the project is
  `polymer-autonomous-engine`, and no wheel target was declared. Fixed, with the
  scientific dependencies moved to optional extras.

### Scientific correctness

- **`target_t` is not a GROMACS keyword.** Generated NVT/NPT/production inputs used
  `target_t` instead of `ref_t`. `grompp` warns about the unknown option and silently
  uses its default temperature, so runs completed at the wrong temperature.
- **Replicas shared a random seed.** No `gen_seed` or `ld_seed` was emitted, so
  "independent replicas" were byte-identical trajectories, replica agreement was a
  tautology and the reported uncertainty was fiction. Seeds are now derived per
  (replica, stage) and are distinct and reproducible.
- **Berendsen barostat in production.** Berendsen does not sample the NPT ensemble and
  was removed in GROMACS 2025+. Defaults are now C-rescale (equilibration) and
  Parrinello-Rahman (production), and a Berendsen production barostat is rejected.
- **`temperature_lambdas = 1`** — a free-energy-perturbation keyword — was emitted in
  NVT inputs. Removed.
- **Missing continuation semantics.** `gen_vel`/`continuation` were absent, so
  equilibration was either discarded or the start double-constrained. Now explicit per
  stage.
- **Uncertainties ignored autocorrelation.** `stderr = sd/√n` over MD frames
  understates the error by √(2τ) — for a τ = 200 ps observable sampled every 10 ps,
  about 6×. All uncertainties now use the effective sample size via the Chodera
  statistical inefficiency. Validated against AR(1) theory.
- **Replica agreement had no statistics.** It compared `(max − min)/mean` against a
  fixed tolerance. Now a reduced chi-square against the per-replica uncertainties.
- **A single replica reported `pass`.** One run cannot demonstrate reproducibility; it
  is now `INCONCLUSIVE`.
- **PMF sign convention was inverted.** `integrate_pmf` computed `+∫F dx`; the PMF is
  `−kT ln p`. WHAM now solves the self-consistent equations directly and is validated
  against an analytic harmonic PMF to 0.45 kJ/mol.
- **`pmf_barrier` returned the global range**, not a barrier. It now measures outward
  from the free-energy minimum.
- **Under-overlapped windows were mis-detected.** `deficient_windows` used
  `max(neighbours)`, so a window with one bad neighbour was not flagged even though the
  *pair* had a gap. Overlap is now diagnosed per adjacent pair.
- **PMF convergence compared the gauge, not the shape.** A PMF is defined up to an
  additive constant; the half-split check now aligns the curves first and compares only
  well-sampled bins.
- **Family classification ran on a hydrogen-capped monomer**, which destroys the
  in-chain linkage that defines the family: capped nylon-6 has no amide at all, and
  capped poly(ethylene oxide) gains a hydroxyl the polymer does not have. It now runs
  on a four-unit oligomer with backbone/pendant discrimination, so poly(ethylene
  terephthalate) and poly(methyl methacrylate) are no longer both "polyacrylate".
- **Ballistic motion was given a diffusion coefficient.** An R²-based linearity test
  accepts a quadratic over a narrow window; the check is now the MSD scaling exponent,
  which must be ≈ 1.
- **`.gro` files were parsed by whitespace splitting.** The format is fixed-column and
  fields routinely run together. Replaced with a column-accurate parser.
- **Topology and coordinates were never compared.** The validator now resolves
  `#include` directives, counts atoms per `[ moleculetype ]`, expands `[ molecules ]`,
  and checks the total against the coordinate file — reporting `INCONCLUSIVE` when a
  molecule type is defined in an unresolvable library include.

### Safety and honesty

- **Dry runs reported success.** A disabled runner returned exit code 0 and
  `ok = True`, and the replica executor returned `SUCCEEDED` with "validated in dry-run
  mode". A dry run is now `SKIPPED` with `execution_mode="dry_run"`, its execution
  record is `CANCELLED`, and it does not satisfy a downstream dependency.
- **The engine silently substituted a fake executor.** Any unregistered action kind
  fell through to `DryRunExecutor`, which emitted a synthetic observation
  (`dry_run = 1.0`) that passed every gate and was recorded as a succeeded experiment.
  Both bootstrap actions did exactly this. There is now no fallback executor.
- **A validation status was overwritten.** `build_campaign` set
  `manifest["system"]["status"] = "pass"` unconditionally after validation had failed.
  Removed; a failed system blocks the campaign.
- **Provider failures were indistinguishable from empty results.** Every provider
  wrapped its work in `except Exception` and returned `ok=False`. Errors are now
  classified, and `ok=True, records=[]` means the provider genuinely has nothing.
- **The provider registry could not be constructed.** Four providers did not implement
  the abstract `health()`, so instantiating the registry raised `TypeError`. It was
  never called anywhere, so this was never noticed.
- **The registry advertised capabilities that did not exist**, including CHARMM-GUI
  `module_submission`. Capabilities are now read from the implementations, and
  submission is declared `UNSUPPORTED` — CHARMM-GUI publishes no such endpoint.
- Crossref's base URL was `https://api.crossref.org/v1`; there is no `/v1` path.
- Archive extraction rejected traversal but not decompression bombs, zip symlink
  attributes, device files, or member-count limits, and a truncated `.tar.gz` raised an
  unhandled `EOFError`. All now handled, with nothing written until every member passes.
- Credentials are held in a masked `Secret`, scrubbed from logs by pattern as well as
  by value, excluded from cache keys, and a group-readable credential file is refused.

### Bugs found while building the new tree

Each has a regression test.

- `replica_seed` used Python's `hash()`, which is per-process randomised — so a
  "reproducible" seed differed on every run. Now CRC32.
- Reloading a campaign did not restore its replica set, so saving it again wrote a
  manifest with no seeds and destroyed the reproducibility record.
- Replica artifact ids were derived from the literal directory name `replicas`, giving
  every campaign a `replica_01_replicas` and silently sharing provenance across
  campaigns. Now namespaced by campaign id.
- Manifest seed keys were integers, which JSON turns into strings, so a stored manifest
  never compared equal to a freshly generated one.
- The topology parser flushed the pending `[ moleculetype ]` on every section header,
  discarding its atom count before `[ atoms ]` was read.
- `ProviderResult.as_dict()` omitted the `data` field.
- Candidate generation excluded backbone carbons from substitution, so polyethylene
  could never become polypropylene.
- A `.gro` missing its box line was reported as "truncated" rather than as a box error.
- The shipped `configs/default.yaml` used the pre-0.6 schema and could not be loaded.

### Added

- Four-valued gates (`PASS`/`WARN`/`FAIL`/`INCONCLUSIVE`) with values, thresholds,
  units and diagnostics.
- A validated execution-state machine; `COMPLETED → RUNNING` requires an explicit
  restart that creates a new record.
- `Measurement`, which cannot hold a value unless its determination is `KNOWN`.
- Explicit unit handling; cross-dimension conversion raises.
- A provenance DAG with digest verification and bidirectional lineage.
- A structured decision engine with feasibility vetoes, deterministic tie-breaking, and
  pessimistic treatment of unknown cost.
- A strategy registry that learns from outcomes and refuses out-of-bounds parameter
  adaptation.
- Claims with evidence-based promotion; a surrogate prediction is never evidence.
- Campaign manifests and fingerprints; resumable campaigns.
- A CLI with meaningful exit codes (3 = a gate did not pass).
- 647 tests across unit, provider, simulation, analysis, orchestration, integration,
  security, property-based and end-to-end suites; the core suite needs no network.
- Ruff and mypy configuration, both clean; GitHub Actions CI.

## 0.5.0 and earlier

See git history. The 0.5 tree is superseded in full.
