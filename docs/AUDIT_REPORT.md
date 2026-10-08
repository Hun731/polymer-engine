# Audit report — v0.5 → v0.6

Takeover audit of the Polymer Autonomous Research Engine. Starting point: 2,687 lines
across 47 modules, 10 tests (2 of which failed on a clean checkout).

Result: 17,215 lines across 70 modules, 6,028 lines of tests across 20 files,
653 tests passing, ruff clean, mypy clean.

---

## IMPLEMENTED

### Consolidated to a single source of truth

The 0.5 tree contained three campaign builders wrapping one another
(`CampaignPlanner` → `AutonomousCampaignBuilder` → `FullCampaign`), two gate systems
with incompatible types (pydantic `GateResult` vs plain dicts), two `sha256_file`
implementations, two `now()` helpers, two utility-scoring functions with the same
hard-coded weights, and two CLI entry points. Removed: `science/`, `runtime/`,
`domain/`, `planning/`, `knowledge/`, `validation/`, `agents/`, `cli.py`,
`core/engine.py`, and four superseded executor modules.

The `providers/registry.py` in 0.5 could not be constructed at all — four providers
did not implement the abstract `health()`, so instantiating it raised `TypeError`. It
was never called from anywhere, so this was never noticed.

### New subsystems

| Module | What it provides |
|---|---|
| `core/` | config (env + file + override precedence), typed errors, unit system, four-valued gates, `Measurement`, execution state machine, provenance DAG |
| `providers/` | classified HTTP transport (retry, backoff, rate limit, cache), 7 providers, offline test transports |
| `local/` | tool discovery with version parsing and capability detection; runners where a dry run is never a success |
| `polymer/` | oligomer-based identity and taxonomy, unit-safe property normalisation, descriptors, reconciling ingestion |
| `simulation/` | safe archive extraction, `.gro`/`.top`/`.xvg` parsers, system validation gates, validated `.mdp` generation, replica materialisation, umbrella planning |
| `analysis/` | correlation-aware statistics, convergence and replica gates, trajectory observables, WHAM/PMF with overlap and convergence checks |
| `discovery/` | leakage-protected QSPR, screened active learning, graph-based candidate generation |
| `orchestrator/` | structured decision engine, campaigns with manifests and fingerprints, autonomous runner, bounded strategy learning |
| `evidence/` | claims with evidence-based promotion |
| `db/` | durable SQLite state with resumption |
| `cli/` | 9 command groups with meaningful exit codes |

---

## FIXED

### Defects that would have produced wrong scientific answers

1. **`target_t` instead of `ref_t`** in generated NVT/NPT/production inputs.
   `target_t` is not a GROMACS keyword; `grompp` warns about the unknown option and
   silently uses its default temperature. Every simulation would have run at the wrong
   temperature while appearing to succeed.

2. **All replicas shared a random seed.** No `gen_seed` or `ld_seed` was emitted, so
   "independent replicas" were byte-identical trajectories. Replica agreement was a
   tautology and every reported uncertainty was fiction.

3. **Berendsen barostat in production.** Berendsen does not sample the isothermal-
   isobaric ensemble — its volume fluctuations are wrong — and it was removed outright
   in GROMACS 2025+, so the inputs would not even run on a current build.

4. **Uncertainties ignored autocorrelation.** `stderr = sd/√n` over MD frames
   understates the error by √(2τ+1). For a τ = 200 ps observable sampled every 10 ps
   that is a factor of ~6, which turns a disagreement into a "significant" result.

5. **PMF sign convention inverted.** `integrate_pmf` computed `+∫F dx`; the PMF is
   `−kT ln p`. Every free-energy landscape would have been upside down.

6. **`pmf_barrier` returned the global range**, not a barrier — max minus min
   regardless of position along the coordinate.

7. **Under-overlapped umbrella windows were mis-detected.** `deficient_windows` used
   `max(neighbours)`, so a window with one bad neighbour was not flagged even though
   that adjacent *pair* had a gap and the free-energy difference across it was
   undetermined.

8. **Family classification ran on a hydrogen-capped monomer**, which destroys the
   in-chain linkage that defines the family. Capped nylon-6 (`*NCCCCCC(=O)*`) contains
   no amide bond at all; capped poly(ethylene oxide) gains a hydroxyl the polymer does
   not have. Verified in the 0.5 code: PEO classified as poly(vinyl alcohol), nylon-6
   as unclassified, PET and polycarbonate both as polyacrylate.

9. **`.gro` files parsed by whitespace splitting.** The format is fixed-column and
   fields routinely run together for large coordinates or long atom names.

10. **Topology and coordinates were never compared.** Nothing checked that the
    topology described the same system as the coordinate file.

11. **`temperature_lambdas = 1`** — a free-energy-perturbation keyword — emitted in
    NVT inputs where it is meaningless.

12. **Missing continuation semantics.** No `gen_vel`/`continuation`, so equilibration
    was either discarded or the starting structure double-constrained.

### Defects that let the engine claim results it did not have

13. **The engine silently substituted a fake executor.** `ResearchEngine.step()` fell
    back to `DryRunExecutor` for any unregistered action kind. That executor emitted a
    synthetic observation (`dry_run = 1.0`) with an artifact path and provenance, which
    passed all three gates and was recorded as a succeeded experiment. Both bootstrap
    actions (`dataset_audit`, `baseline_qspr`) took this path — the one passing test in
    the original suite was asserting that fabricated data was accepted.

14. **A validation status was overwritten.** `AutonomousCampaignBuilder.build_campaign`
    executed `manifest["system"]["status"] = "pass"` unconditionally, after validation
    had already returned a failure.

15. **Dry runs reported success.** A disabled `LocalRunner` returned exit code 0 with
    `ok = True`; `GROMACSReplicaEquilibrateExecutor` returned `SUCCEEDED` with
    "Replica validated in dry-run mode".

16. **One replica passed the agreement gate.** `replica_agreement` returned
    `{"pass": True}` for a single value.

17. **Unknown was conflated with failure.** `evaluate_replicas` returned `"unknown"`
    for no data, and the caller mapped anything that was not `"pass"` to `FAILED` —
    losing the distinction between "it failed" and "we could not tell".

18. **Replicas with no data were silently skipped**, so a gate could pass on one of
    three replicas.

19. **Provider failures were indistinguishable from empty results.** Every provider
    wrapped its work in `except Exception` and returned `ok=False` — a rate limit, a
    DNS failure and "no papers match" were the same outcome.

20. **The registry advertised capabilities that did not exist**, including CHARMM-GUI
    `module_submission`, for which no endpoint is published.

### Security

21. Archive extraction rejected path traversal but not: decompression bombs, zip
    symlink attributes, device/FIFO members, member-count limits, or `.tar.bz2`/`.tar.xz`
    at all. A truncated `.tar.gz` raised an unhandled `EOFError` mid-campaign.
22. Nothing prevented a partial extraction — a mixed archive wrote its safe members
    before hitting the unsafe one.
23. Credentials had no masking, no log redaction, and no file-permission check.

### Other

24. **The package could not be built.** `pyproject.toml` declared
    `polymer-autonomous-engine` with no wheel target for `src/polymer_engine`.
25. Crossref's base URL was `https://api.crossref.org/v1`; there is no `/v1` path.
26. `GROMACSRunner.energy` invoked an interactive command with no stdin — a comment in
    the source acknowledged it would not work.
27. `cli.py build-campaign` never staged a system, so `materialize_replicas` always
    raised.

### Bugs found while building the replacement

Each has a regression test.

28. `replica_seed` used Python's `hash()`, which is per-process randomised — a
    "reproducible" seed differed on every run. (Caught by running it twice.)
29. Reloading a campaign did not restore its replica set, so saving it again wrote a
    manifest with no seeds, destroying the reproducibility record.
30. Replica artifact ids derived from the literal directory name `replicas`, so every
    campaign had a `replica_01_replicas` and provenance was silently shared.
31. Manifest seed keys were integers, which JSON turns into strings, so a stored
    manifest never compared equal to a freshly generated one.
32. The topology parser flushed the pending `[ moleculetype ]` on every section header,
    discarding its atom count before `[ atoms ]` was read.
33. An R²-based MSD linearity test accepted ballistic motion as diffusion (a quadratic
    fits a line with R² = 0.986 over a narrow window). Replaced with the scaling
    exponent, which reports t^2.00 for the ballistic fixture.
34. PMF half-split convergence compared raw values, measuring the arbitrary additive
    gauge rather than the shape.
35. `ProviderResult.as_dict()` omitted the `data` field.
36. Candidate generation excluded backbone carbons, so polyethylene could never become
    polypropylene.
37. `design generate` accepted a discrete molecule as a parent.
38. The CLI dataset builder iterated a dict as a list, corrupting feature order.
39. A `.gro` missing its box line reported "truncated" rather than a box error.
40. The shipped `configs/default.yaml` used the pre-0.6 schema and could not be loaded.

---

## NEW TESTS

653 tests. Highlights, by what they prove rather than what they cover:

**Numerical correctness against known answers** — autocorrelation time vs AR(1) theory
(τ = 9.52 vs 9.0 analytic); WHAM vs an analytic harmonic PMF (0.45 kJ/mol over a
21 kJ/mol range); Rg of a 5-atom chain (√2 Å exactly); MSD of a tracer moving 1 Å/frame;
D recovered from a random walk to 10 %; PS repeat unit = ethylbenzene, 106.17 g/mol;
topology 1×POL(8) + 20×SOL(3) = 68 atoms.

**Security** — 30 archive tests (traversal both formats, absolute members, backslash
traversal, symlinks, hardlinks, zip symlink attributes, device files, size and count
limits, compression bombs, truncated archives, HTML masquerading as an archive, and
that *nothing is written* when any member is unsafe); 52 secret tests (masked repr,
redacted dumps, log scrubbing by value and by pattern, tokens absent from results and
error bodies, cache keys excluding `Authorization`, refusal to read a group-readable
credential file, and a scan of `src/` for committed credentials).

**Property-based (Hypothesis)** — extraction paths can never escape their root for any
member name; the state machine never enters an illegal state for any command sequence;
`COMPLETED → RUNNING` is never permitted; unit conversion round-trips and refuses
cross-dimension; temperature *intervals* survive K↔°C; effective samples never exceed
raw samples; replica seeds are always distinct and always reproducible; the recommended
umbrella spacing always satisfies its own overlap criterion.

**Refusal tests** — the ones that matter most: a dry run is `SKIPPED` and does not
satisfy a dependency; 2 replicas where 3 are required is `INCONCLUSIVE`; a replica with
no data fails rather than being skipped; disagreeing replicas fail on reduced
chi-square; a poorly overlapped PMF reports no barrier; ballistic motion is refused a
diffusion coefficient; a training-set member is rejected as a candidate; a claim cannot
reach `SUPPORTED` from dry-run evidence; an out-of-bounds strategy parameter is
rejected rather than clamped; an unknown action kind raises rather than running a stub.

---

## Verification status

### Verified locally on this machine

- **Real GROMACS 2026.3**: all four generated `.mdp` stages pass `gmx grompp` with
  **zero warnings**; a full two-replica campaign ran EM → NVT → NPT → production and
  produced trajectories, energies and extracted observables.
- **The central claim, on real data**: that run exited 0 at every stage and wrote 766
  frames, and the engine still refused it — 2.9 effective samples after autocorrelation
  correction, density still drifting, 2 replicas where 3 are required.
- **Real ORCA 6.1.1**: discovered, version parsed from its banner despite a non-zero
  exit code.
- Tool discovery against real hardware: GROMACS 2026.3 (CUDA, mixed precision, AVX-512,
  thread-MPI), RTX 5090, 24 cores.
- Wheel builds and installs into a clean venv; `polymer-engine --help` works.
- **The core suite passes with no scientific extras**: 527 passed, 126 skipped, 0
  failed, with RDKit, MDAnalysis, scikit-learn and SciPy all absent.
- Full suite: 653 passed. Ruff clean. Mypy clean on 70 modules. 78 % line coverage.
- The example campaign runs end to end.

### Tested with mocks or fixtures only

- All 7 providers, against recorded offline responses. The record shapes come from each
  service's published contract, but no live call was made from this environment.
- CHARMM-GUI login, status polling, download and archive verification.
- The complete campaign lifecycle in dry-run mode.

### Implemented but requires local software

- Umbrella sampling execution needs PLUMED (absent here). Window planning, PLUMED input
  generation and PMF analysis are implemented and tested; running the windows is not
  orchestrated — see KNOWN LIMITATIONS.
- Production-scale MD needs a real system and hours of GPU time. Verified only on a
  68-atom fixture over picoseconds.
- `hydrogen_bonds` needs MDAnalysis's hydrogen-bond module; implemented, not exercised
  on a system with hydrogen bonds.

### Requires credentials

- CHARMM-GUI login, job status and download (`CHARMM_GUI_EMAIL`/`PASSWORD` or `TOKEN`).
- Materials Project search (`MP_API_KEY`).
- Crossref and OpenAlex polite pools (`CROSSREF_MAILTO`, `OPENALEX_MAILTO`) — optional.

### Planned / not implemented

Documented in [ROADMAP.md](ROADMAP.md). The significant absences: ORCA/QM workflows
(the runner is scaffolding), umbrella *execution* orchestration, MBAR, mechanical and
transport properties beyond MSD, system construction, and LLM hypothesis generation.

---

## KNOWN LIMITATIONS

1. **The engine validates systems; it does not build them.** No chain builder, packing,
   solvation or ionisation. Systems come from CHARMM-GUI or another external tool. This
   is a deliberate boundary — a half-correct polymer builder is worse than none.

2. **Umbrella sampling is planning-only.** Windows and PLUMED inputs are generated and
   validated; there is no umbrella executor and no adaptive-insertion loop driving new
   simulations. `umbrella analyze` works on COLVAR files you produce yourself.

3. **ORCA is not wired into any workflow.** No input generation, no output parsing, no
   MM-vs-QM validation gates.

4. **Actions run sequentially.** `resources.max_concurrent_jobs` is recorded in the
   manifest but not enforced; there is no scheduler, no parallel dispatch, no HPC
   submission.

5. **Ensemble-spread uncertainty is not calibrated.** The QSPR surrogate's uncertainty
   is useful for *ranking* what to run next. It is deliberately not described as a
   confidence interval.

6. **Trajectories are not bit-reproducible across machines.** Expected: different SIMD
   widths, GPU kernels and reduction orders change rounding, and MD is chaotic. What
   must reproduce is the ensemble average within its stated uncertainty — which is
   exactly what the replica-agreement gate tests.

7. **Copolymers and crosslinked networks are recorded but not handled.**
   Sequence-aware identity, descriptors and generation are not implemented.

8. **Some thresholds are conventions, not derivations.** `min_effective_samples = 20`,
   `max_relative_stderr = 2 %`, `max_drift_fraction = 2 %` are defensible defaults, not
   results. `SCIENCE.md` (then named `SCIENTIFIC_VALIDATION.md`) marks which is which. The umbrella spacing limit
   *is* derived, from σ = √(kT/k).

---

## REQUIRES SCIENTIFIC DECISION

Choices a domain expert should make deliberately rather than inherit from a default:

1. **Force field and water model.** `simulation.force_field` ships as `UNSPECIFIED`,
   which is honest but is not a choice. Nothing validates that the force field is
   appropriate for the polymer, or that the system was built with the one you name.

2. **Convergence thresholds per property.** 2 % relative uncertainty is reasonable for
   density and far too tight for pressure (which is genuinely noisy in small systems —
   the real run reported 52 %). These should be set per observable.

3. **Reaction coordinate justification.** `ReactionCoordinate.justification` is required
   text and is carried into the manifest, but no code can judge whether a coordinate is
   physically meaningful. A PMF along an unjustified coordinate is a number without a
   meaning.

4. **Required replica count.** The default of 3 is a floor, not a recommendation.

5. **Production length.** No heuristic sets it. The gates will tell you when a run was
   too short; they cannot tell you in advance how long it needs to be.

6. **Candidate constraints.** The element whitelist, mass and ring limits encode one
   view of what is a reasonable polymer. They should be reviewed per project.

7. **Whether `REQUIRES_REVIEW` candidates are acceptable.** The engine flags strained
   rings, formal charges and Si/P chemistry; it does not decide whether to pursue them.

8. **Synthetic accessibility.** Not assessed at all. Candidates may be chemically valid
   and entirely unmakeable.
