# Science

Every gate in the engine, what it checks, the threshold it uses, and why that
threshold. Where a number is a convention rather than a derivation, this document says
so — an unjustified threshold is a hidden assumption.

Subsystem detail lives alongside this document:
[SIMULATION.md](SIMULATION.md) · [ORCA.md](ORCA.md) · [UMBRELLA.md](UMBRELLA.md) ·
[AUTONOMY.md](AUTONOMY.md). Status labels used throughout:
[STATUS_LEGEND.md](STATUS_LEGEND.md).

## The four states

| Status | Meaning | Blocks promotion? |
|---|---|---|
| `PASS` | the check ran and succeeded | no |
| `WARN` | the check ran; the result is marginal | no |
| `FAIL` | the check ran and failed | **yes** |
| `INCONCLUSIVE` | the check could not be evaluated | **yes** |

`INCONCLUSIVE` blocking is the single most important design decision here. A pipeline
that treats "we could not tell" as "fine" will eventually promote something false. An
empty `GateReport` is `INCONCLUSIVE`, so running no checks never reads as a pass.

---

## 1. System validation

`simulation/system.py::validate_system`

| Gate | Check | Failure means |
|---|---|---|
| `coordinates_present` | a `.gro` exists | nothing to simulate |
| `topology_present` | a `.top` exists | no force-field assignment |
| `coordinates_parse` | fixed-column `.gro` parses | corrupt or truncated file |
| `atom_count` | declared atom count > 0 | empty system |
| `coordinates_finite` | all coordinates finite, `< 10^4 nm` | corrupt file or a diverged run |
| `box_defined` | box vectors present and positive | no periodic box |
| `contents_fit_box` | atom extent ≤ box along every axis | the molecule interacts with its own periodic image, silently corrupting every energetic quantity |
| `topology_includes` | every `#include "..."` resolves | missing parameters |
| `molecules_section` | `[ molecules ]` present and non-empty | topology defines no composition |
| `molecule_types_resolved` | every molecule type has an atom count | **`INCONCLUSIVE`** — cannot verify locally |
| `topology_matches_coordinates` | Σ(count × atoms) equals the `.gro` atom count | topology and coordinates describe different systems |

**Notes.** `#include <...>` (angle brackets) resolves against `GMXLIB` at run time and
is *not* treated as missing — flagging it would block every valid CHARMM-GUI system.
When a molecule type lives in an unresolved include, the atom-count comparison is
`INCONCLUSIVE`, not `PASS`: we could not check, which is different from having checked.

---

## 2. Simulation parameters

`simulation/mdp.py::validate_parameters`, run before any input is written.

| Check | Bound | Rationale |
|---|---|---|
| temperature | (0, 2000] K | outside this, condensed-phase force fields are not parameterised |
| pressure | > 0 bar | negative reference pressure is unphysical here |
| timestep | (0, 5] fs | above ~5 fs, even constrained integration is unstable |
| timestep vs constraints | > 2.5 fs requires constraints | unconstrained X–H stretches need ~1 fs |
| replicas | [1, 64] | below 1 is meaningless; above 64 is almost certainly an error |
| `tau_t` | > 10 × timestep | a thermostat faster than this perturbs the dynamics |
| `tau_p` | > `tau_t` | a barostat outrunning the thermostat produces artefacts |
| `base_seed` | > 0 | `-1` (random) would make the run irreproducible |
| production barostat | never Berendsen | Berendsen does not sample the NPT ensemble; its volume fluctuations are wrong, and it was **removed in GROMACS 2025+** |

Three errors that the generator fixes and that are easy to miss by eye:

1. **`ref_t`, not `target_t`.** `target_t` is not a GROMACS keyword. `grompp` warns
   about the unknown option and silently uses its default temperature — so the run
   completes, at the wrong temperature.
2. **Distinct seeds per replica.** With a shared `gen_seed`, "independent replicas"
   are byte-identical trajectories, replica agreement is a tautology, and the reported
   uncertainty is fiction. Seeds are derived as
   `f(base_seed, replica_index, stage)` via CRC32 — distinct, and reproducible across
   processes (Python's `hash()` is per-process randomised and would not be).
3. **Explicit continuation.** `gen_vel = yes, continuation = no` for NVT (coming from
   minimisation); `gen_vel = no, continuation = yes` for NPT and production. Getting
   this wrong either discards equilibration or double-constrains the start.

**Verified:** all four generated stages pass real `gmx grompp` (GROMACS 2026.3) with
zero warnings — see `tests/e2e/test_full_campaign.py::test_generated_mdp_files_are_accepted_by_grompp`.

---

## 3. Convergence

`analysis/convergence.py`

### The pseudoreplication problem

A 100 ns trajectory written every 10 ps gives 10 000 frames. If the density
autocorrelation time is 200 ps, it contains roughly 250 independent measurements.
Reporting `sd/√10000` understates the uncertainty by about 6× and turns a
disagreement into a "significant" result.

Every uncertainty in this engine is therefore computed from the **effective sample
size**:

```text
g   = 1 + 2·τ_int          (statistical inefficiency, Chodera et al. 2007)
N_eff = N / g
SE  = sd / √N_eff
```

`τ_int` uses an automatic truncation window (Sokal, c = 6), which avoids the variance
blow-up from summing the noisy tail of the autocorrelation function.

*Validated against theory:* for AR(1) with φ = 0.9, the analytic τ_int is 9.0 and
g is 19.0; the implementation recovers 9.5 and 20.1
(`tests/analysis/test_statistics.py`).

### Gates

| Gate | Threshold | Rationale |
|---|---|---|
| `<metric>:effective_samples` | ≥ 20 (configurable) | below ~20 independent samples the standard error is itself unreliable |
| `<metric>:relative_uncertainty` | ≤ 2 % → `PASS`, else `WARN` | a convention, not a derivation; tune per property |
| `<metric>:drift` | half-to-half change ≤ 2 % | drift above the threshold **and** above the statistical noise is `FAIL`; above the threshold but within noise is `WARN` |
| `<metric>:equilibration` | > 60 % discarded → `WARN` | the production window is short relative to the run |

Drift is only a failure when it exceeds *both* the threshold and 2σ of the two
half-means. A large but statistically insignificant change means "sample more", not
"this is drifting".

### Equilibration detection

Default is reverse-cumulative: for each candidate start `t0`, compute
`(N − t0)/g(t0)` and keep the `t0` that maximises it (Chodera 2016). This trades bias
(early unequilibrated frames) against variance (throwing data away), rather than
applying an arbitrary "discard the first 20 %". The fixed-fraction alternative remains
available via `analysis.equilibration_detection`.

---

## 4. Replica agreement

`analysis/convergence.py::replica_agreement_gate`

Replicas are the independent experimental units. The combined uncertainty is the
**standard error of the replica means**, not the within-replica frame-level error —
using the latter is textbook pseudoreplication.

Agreement is tested with a reduced chi-square:

```text
χ²_red = Σ((x_i − x̄_w)² / σ_i²) / (n − 1)
```

where `x̄_w` is the inverse-variance weighted mean. `χ²_red > 4` is `FAIL`: the spread
between replicas is far larger than their individual uncertainties, so at least one
error bar is wrong or the replicas are not sampling the same ensemble.

| Situation | Verdict |
|---|---|
| no usable replica | `FAIL` |
| fewer replicas than required (default 3) | **`INCONCLUSIVE`** |
| replicas present, no per-replica uncertainties | `INCONCLUSIVE` |
| `χ²_red > 4` | `FAIL` |
| otherwise | `PASS` |

**One replica is always `INCONCLUSIVE`, however precise it is.** A single run cannot
demonstrate reproducibility; reporting it as a pass would be the most consequential
lie the engine could tell.

A replica that produced no data is reported as a failure, never silently excluded —
otherwise a gate could "pass" on one of three replicas.

---

## 5. Umbrella sampling and PMF

`simulation/umbrella.py`, `analysis/free_energy.py`

### Window spacing is physics, not preference

A harmonic restraint of stiffness `k` at temperature `T` samples a distribution of
width `σ = √(kT/k)`. Windows separated by more than ~2σ do not overlap, and a PMF
built across a gap is not merely imprecise — the free-energy difference across the gap
is **not determined by the data at all**. WHAM will still return numbers.

The planner derives the recommended spacing from `k` and `T`, and refuses (or warns
about) a configuration that cannot overlap. Shipped default: 0.08 nm at
k = 1000 kJ/mol/nm², which is ~1.6σ at 300 K.

### Gates

| Gate | Threshold | Meaning |
|---|---|---|
| `umbrella:windows_sampled` | every window has samples | an empty window is a hole in the coordinate |
| `umbrella:window_overlap` | every adjacent pair ≥ 0.10 overlap coefficient | below this, ΔG across the gap is undetermined |
| `umbrella:wham_converged` | self-consistent iteration reached tolerance | otherwise the free energies are not a solution |
| `umbrella:sampling_converged` | first vs second half agree within 2 kJ/mol | ≈ 0.8 kT at 300 K; still-changing PMF |
| `umbrella:uncertainty_estimated` | peak bootstrap σ ≤ 2 kJ/mol → `PASS` | else `WARN` |

Two refinements that matter:

- **The gauge is removed before comparing.** A PMF is defined only up to an additive
  constant, so the half-split comparison shifts the two curves onto a common mean
  first. Comparing raw values measures the gauge choice, not convergence.
- **Only well-sampled bins are compared.** A bin holding under 2 % of the busiest
  bin's samples (and fewer than 30 counts) is statistically empty; comparing PMF
  values there compares noise and would reject converged results.

Uncertainty uses a **block** bootstrap within each window. Resampling frames
independently ignores correlation and produces bands several times too narrow.

`PmfResult.barrier()` and `.well_depth()` return `REQUIRES_VALIDATION` — not a
number — whenever the PMF is untrustworthy. `well_depth()` additionally refuses when
the PMF has not plateaued at large separation, because without a plateau there is no
reference state and the depth would be arbitrary.

*Validated:* WHAM recovers an analytic harmonic PMF to within 0.45 kJ/mol over a
21 kJ/mol range (`tests/analysis/test_convergence_and_free_energy.py`).

---

## 6. Trajectory analysis

`analysis/md.py`

MDAnalysis works in ångström; the engine works in nm. Conversion happens once,
explicitly, at the module boundary, and every returned `Measurement` carries its unit.

| Observable | Unit | Note |
|---|---|---|
| radius of gyration | nm | mass-weighted |
| end-to-end distance | nm | first/last atom indices recorded |
| COM distance | nm | refuses a zero-mass group (would give a NaN centre) |
| density | kg/m³ | flags guessed masses (see below) |
| RDF | nm | ideal-gas normalisation, minimum-image convention |
| MSD | nm² | all time origins; requires an unwrapped trajectory |
| contacts | count | minimum-image convention |
| hydrogen bonds | count | MDAnalysis geometric criterion |

Two honesty guards:

- **Guessed masses are flagged.** `.pdb` and `.gro` carry no masses, so MDAnalysis
  guesses them from atom names. Any density derived from guessed masses is marked
  approximate rather than presented as quantitative.
- **Ballistic motion is refused a diffusion coefficient.** `D` is reported only when
  the MSD actually scales as `t^α` with α ≈ 1. A particle moving at constant velocity
  gives α = 2 and fits a straight line with R² = 0.99 over a narrow window — an R²
  test alone would happily quote a diffusion coefficient for it.

---

## 7. Structure–property modelling

`discovery/qspr.py`

Three leakage routes are closed structurally rather than by convention:

1. **Duplicate polymers across a split.** Cross-validation folds are assigned by
   `polymer_id`, derived from the canonical repeat unit, so the same material
   described two ways cannot appear on both sides.
2. **Preprocessing fitted on all the data.** Imputation and scaling are fitted inside
   each fold, on the training part only. Fitting the scaler on the whole dataset first
   is the most common silent leak in QSPR work and inflates every reported score.
3. **Scoring a training member as a discovery.** Predictions for polymers already in
   the training set are marked `REQUIRES_VALIDATION`; in candidate selection they are
   **rejected outright**.

Predictions outside the applicability domain (95th-percentile training distance in
standardised feature space) are labelled extrapolations. Ensemble spread is offered as
a *relative* signal for choosing what to run next, and is deliberately **not**
described as a calibrated confidence interval.

---

## 8. Candidate generation

`discovery/candidates.py`

Mutations operate on the RDKit molecular graph, never on the SMILES string. String
mutation produces "candidates" like `*CC((*)` that look plausible in a list and are not
molecules. Every candidate must pass:

- chemical validity (parses, sanitises, valences satisfied)
- exactly two attachment points (still a linear repeat unit)
- differs from its parent
- not a duplicate of anything already known
- configured constraints (mass, rings, rotatable bonds, element whitelist)
- descriptors can actually be computed

Structures RDKit accepts but that warrant human judgement — strained rings, formal
charges, Si/P where force-field coverage is patchy — are returned as
`REQUIRES_REVIEW` rather than accepted or dropped.

**Synthetic accessibility is not claimed.** There is no retrosynthesis model here, and
asserting synthesisability would be fabrication.

---

## 9. Claims

`evidence/claims.py`

| Status | Requires |
|---|---|
| `SUPPORTED` | ≥ 1 validated, replicated, uncertainty-bearing supporting piece **and no refuting evidence** |
| `PARTIALLY_SUPPORTED` | validated support alongside refuting evidence |
| `CONTRADICTED` | validated refutation with no validated support |
| `INSUFFICIENT_EVIDENCE` | default and resting state |

For simulation evidence, "usable" requires all three of: gates returned `PASS`, an
uncertainty was estimated, and ≥ 3 independent replicates. Missing gate results count
as *not* validated — unrun checks are not passed checks.

**A surrogate model prediction is never usable evidence.** It is a reason to run
something, not a finding about the world.

`Claim.evaluate()` recomputes the status from the evidence every time and is the only
path to a status, so a status cannot be assigned by hand and then quietly retained.

---

## 10. Quantum-chemistry validation

`qm/orca_parser.py`, `qm/validation.py` — detail in [ORCA.md](ORCA.md).

A QM run is judged on five states, not two. The decisive one is
`FAILED_SCIENTIFICALLY`: ORCA terminated normally and returned 0, but the SCF did not
converge, or the geometry did not converge, or a requested quantity is absent.

| Gate | Check | Failure means |
|---|---|---|
| `qm:output_available` | an ORCA log exists and parses | the job never produced output |
| `qm:termination` | the normal-termination banner is present | killed, truncated, or error-terminated |
| `qm:scf_converged` | `SCF CONVERGED AFTER n CYCLES` | the wavefunction is not a solution |
| `qm:geometry_converged` | `THE OPTIMIZATION HAS CONVERGED`, when a geometry was requested | the structure is not a minimum; **silence also fails** |
| `qm:frequencies_present` | frequencies present when requested | no vibrational analysis to check |
| `qm:no_imaginary_modes` | no imaginary frequencies | a saddle point, not a minimum |
| `qm:energy_present` | `FINAL SINGLE POINT ENERGY` | nothing was actually computed |

Comparison against molecular mechanics needs an `AcceptanceCriteria`. **Without one the
comparison returns `passed=None` and `REQUIRES_VALIDATION`** — the deviation is
computed and reported, and whether it is acceptable is left to a person. Two named
presets exist (`general_organic_forcefield`, `conformational_free_energy`) so that a
manifest records *which convention was adopted* rather than inheriting an invisible
default. Both are conventions from the parameterisation literature, not derivations.

---

## 11. Property extraction

`properties/` — full catalogue in [SIMULATION.md](SIMULATION.md).

Each of the 18 calculators declares its sampling requirement as data, and each is
different because the statistics are different. `density` needs 20 effective samples;
`bulk_modulus` needs **200**, because a variance estimator converges far more slowly
than a mean. Every calculator applies the same gates:

| Gate | Check |
|---|---|
| `<name>:effective_samples` | `N_eff` after autocorrelation ≥ the property's own floor |
| `<name>:equilibration` | the observable settled before the averaging window |
| `<name>:replicas` | ≥ the property's own replica minimum |
| `<name>:duration` | the production window meets the property's minimum length |

Individual calculators add their own: `diffusion_coefficient:diffusive_regime`,
`thermal_expansion_coefficient:slope_resolved`, `relaxation_time:decayed`,
`mechanical:strain_rate`, `mechanical:experimental_comparability`, and so on. The
MM-vs-QM comparison emits `ff_vs_qm:<label>:rmsd`, `:rmse`, `:barrier` and
`:barrier_relative`.

A `PropertyResult` is `usable` only when the determination is `KNOWN` **and** the report
is promotable. Both, not either.

### Units are typed, and the typing is enforced

Four calculators once declared `units="1"` while carrying a volume in nm³, a thermal
expansion coefficient in K⁻¹, an MSD in nm², and a diffusivity in nm²/ps — with the
true unit stated only in prose. A "dimensionless" volume converts silently against any
other dimensionless quantity, which is the hidden-unit failure this engine exists to
prevent. `core/units.py` now carries `volume`, `area`, `diffusivity` and
`inverse_temperature` as real dimensions, and a regression test asserts that every
registered property declares a unit the registry recognises.

### Two refusals that are physics, not policy

**Strain rate.** MD tensile deformation runs at 10⁷–10⁹ s⁻¹; laboratory testing runs
near 10⁻³ s⁻¹ and split-Hopkinson bar impact reaches only ~10⁴ s⁻¹. Above
`EXPERIMENTAL_STRAIN_RATE_CEILING = 1e4`, `interpret_strain_rate()` sets
`comparable_to_experiment=False`, and the metrics are named `yield_stress_proxy` and
`peak_stress` rather than "tensile strength". A polymer pulled a million times faster
than any experiment is exhibiting a different process, not a noisy version of the same
one.

**Diffusive regime.** `DiffusionCoefficient.compute()` fits the MSD scaling exponent α
and refuses the Einstein relation unless `|α − 1| ≤ 0.15`. A polymer melt in the Rouse
or reptation regime gives a slope; that slope is not a diffusion coefficient. Finite-size
effects on D in a periodic box are substantial and are **not** corrected, and that caveat
travels with the result.

---

## 12. Correlation and surrogate models

`science/correlation.py`, `science/models.py` — detail in [AUTONOMY.md](AUTONOMY.md).

These are enforced in the estimators themselves rather than as named `GateResult`s:

| Rule | Where | Why |
|---|---|---|
| ≥ `MIN_OBSERVATIONS` (8) paired values, else `INSUFFICIENT` | `CorrelationMatrix` | a correlation on five points is noise |
| `n_tests` recorded; Bonferroni applied by default in `significant()` | `CorrelationMatrix` | screening 14 descriptors at α = 0.05 expects ~1 spurious hit |
| folds grouped by polymer id, structural cluster, and simulation source | `build_groups()` | otherwise the same polymer sits on both sides of a split |
| preprocessing fitted per fold | `science/models.py` | fitting on the full dataset leaks the test distribution |
| applicability domain, 0.95 quantile | `discovery/qspr.py` | extrapolation is not prediction |

`EvidenceStrength.language` never produces a causal verb. The strongest phrase available
is "predicts", which is a claim about the model, not about mechanism.

---

## 13. Force-field selection

`simulation/forcefield.py` — detail in [SIMULATION.md](SIMULATION.md).

`Confidence.KNOWN` — the only level usable without human review — is reachable **only**
by supplying a QM validation that itself passed. The advisor's own opinion cannot
produce it, because "this force field is commonly used for this chemistry" is a
literature fact, not evidence about your system.

When several profiles cover a chemistry the result is `REQUIRES_EXPERT_DECISION`, not a
ranked pick. Choosing between CHARMM36 and OPLS-AA is a judgement about which published
parameterisation suits the property being measured; taking the first alphabetically
would be a fabricated decision.

---

## Thresholds summary

Every number below is configurable, and every one is recorded in the campaign
manifest.

| Setting | Default | Kind |
|---|---|---|
| `analysis.min_effective_samples` | 20 | convention |
| `analysis.max_relative_stderr` | 0.02 | convention |
| `analysis.max_drift_fraction` | 0.02 | convention |
| `analysis.confidence_level` | 0.95 | convention |
| replica agreement `χ²_red` | 4.0 | ~2σ for small n |
| required replicas | 3 | minimum for a spread estimate |
| `umbrella.min_pair_overlap` | 0.10 | conservative; below this WHAM is ill-conditioned |
| `MAX_SPACING_SIGMA` | 2.0 | **derived** from `σ = √(kT/k)` |
| PMF half-split tolerance | 2.0 kJ/mol | ≈ 0.8 kT at 300 K |
| `WELL_SAMPLED_FRACTION` | 0.02 | statistical |
| `DEFAULT_WHAM_MAX_ITERATIONS` | 20,000 | **measured** — see [UMBRELLA.md](UMBRELLA.md) |
| applicability domain quantile | 0.95 | convention |
| `MIN_OBSERVATIONS` (correlation) | 8 | convention |
| correlation multiplicity correction | Bonferroni | conservative by choice |
| `DIFFUSIVE_TOLERANCE` (\|α − 1\|) | 0.15 | convention |
| `BALLISTIC_THRESHOLD` (α) | 1.7 | convention |
| `EXPERIMENTAL_STRAIN_RATE_CEILING` | 1e4 s⁻¹ | **physical** — the fastest experimental regime that exists |
| `MIN_OCCURRENCES_FOR_PATTERN` | 3 | twice is a coincidence |
| failure-penalty ceiling | 0.6 | deliberately < 1: deprioritise, never forbid |
| QM acceptance tolerances | *none* | **REQUIRES-EXPERT-DECISION** — no default exists |
| per-property `min_effective_samples` | 5–200 | per-property; a variance needs more than a mean |

## What the engine will not do

- report a number without units, or under a unit the registry does not recognise
- report an uncertainty without saying how it was computed
- treat a dry run, a zero exit code, or a single replica as a result
- treat an ORCA job that exited 0 without converging as a success
- fabricate a force-field parameter, a PMF, an experimental value, or a citation
- invent an API endpoint a provider does not document
- promote a claim whose evidence does not qualify
- choose a force field, a level of theory, a reaction coordinate, or an acceptance
  tolerance on your behalf
- call an MD proxy an experimental measurement
- turn a correlation into a causal statement
- treat trajectory frames, or replicas of the same starting structure, as independent
  replicates beyond what the autocorrelation supports
- weaken, disable, or rewrite one of its own validation rules

Where it cannot determine something, it returns `UNKNOWN`, `UNSUPPORTED`,
`INSUFFICIENT_DATA` or `REQUIRES_VALIDATION`, and says why.

## Two ways a density refuses to settle

A drift check reports one thing when a melt has not finished relaxing and the same thing
when it is crystallising. They need opposite responses, and run3 contained both.

Across three polyolefins at 300 K, 100 ns production, three replicas each:

| Candidate | Δdensity | Δpotential energy | What it is |
|---|---|---|---|
| polypropylene | −0.6 … +0.2% | −0.05 … −0.19 sd | equilibrated |
| polyisobutylene | +1.7 … +8.6% | −0.11 … −0.15 sd | relaxation unfinished |
| polyethylene | +4.5 … +9.1% | **−0.48 … −0.99 sd** | **ordering** |

Packing chains into register releases energy. A density that climbs *while potential
energy falls* is a system leaving the amorphous state; a density that climbs with energy
flat is one still finding it. `analysis/ordering.py` makes that distinction, and
`scripts/diagnose_ordering.py` applies it to a campaign.

### Why polyethylene cannot be fixed by running longer

The chains here are 182 atoms — C₆₀H₁₂₂, n-hexacontane, whose melting point is about
372 K. At 300 K the system sits some 70 K **below** its melting point, so an amorphous
melt is not its equilibrium state and no protocol makes it one. Atactic polypropylene
has no crystalline phase at all, which is exactly why it equilibrates without trouble.

The engine now reports this as `ORDERING` with determination `REQUIRES_VALIDATION`
rather than `INSUFFICIENT_DATA`. More data is not what is missing.

Deciding what to do about it is a scientific choice, not an engineering one. Running the
melt above its melting point answers a different question than the one the campaign
asks; keeping 300 K means accepting that polyethylene's amorphous state there is
supercooled and metastable. Neither is chosen automatically.

### Two thresholds, and why neither works alone

R² is the wrong criterion by itself: a real density rise buried in 11 kg/m³ of
fluctuation explains 13% of the variance, and any R² cut strict enough to exclude noise
excludes that too. Statistical significance alone fails the other way: with 26,000
effective samples a 0.4% drift is significant at 24 σ and physically irrelevant.

So a trend must be **both** at least 0.5% of the mean across the window **and** at least
3 standard errors, with the error computed from an autocorrelation-corrected sample
count. The energy threshold of 0.25 sd was placed in a measured gap — equilibrated
systems moved at most 0.19 sd, ordering ones at least 0.48 — which makes it a choice
from data rather than from theory, and worth re-checking on unlike chemistry.

## Annealed equilibration

`gmx insert-molecules` can only pack rigid pre-built chains at roughly a third of melt
density, so NPT has to compress the box threefold, and some replicas jam. The evidence
was bimodal effective sample counts — 4 to 19 for stuck replicas against 600 to 44,000
for the rest — with the stuck ones sitting 30–50 kg/m³ from their siblings in *both*
directions.

`anneal_ns` inserts a heat–hold–cool stage between NVT and NPT: 300 → 500 K over 3 ns,
hold 7 ns, cool back over 10 ns, under pressure coupling so the box finds its own density
at each temperature. 500 K is above the 372 K melting point of these chains, which is the
point — an anneal that stays below it leaves the crystallites it exists to erase.

Setting `anneal_ns: 0` reproduces the original four-stage protocol exactly.

## Studies beyond density, from the trajectories already on disk

`scripts/run_studies.py` runs every study the finished production trajectories support,
per replica, through the engine's existing calculators and gates:

| Source | Studies |
|---|---|
| `prod.edr` | volume, potential energy, enthalpy, temperature, pressure, **bulk modulus** from NPT volume fluctuations |
| `prod.xtc` | **radius of gyration** and **end-to-end distance** per chain, carbon–carbon **RDF**, chain-COM **MSD**, and a **diffusion coefficient** issued only for genuinely diffusive motion |

```bash
.venv/bin/python scripts/run_studies.py campaign/run4
```

Read-only with respect to the campaign, CPU-only, and it skips any replica whose
production has not finished — safe to run while a campaign is live. Output lands in
`<root>/studies/` as one JSON per replica plus `STUDIES.md`.

### Correctness details that matter

* **Two PBC corrections, for two different questions.** `trjconv -pbc mol` makes each
  chain whole again — a chain wrapped across the boundary has a nonsense Rg. `-pbc
  nojump` removes wrapping entirely, which is what MSD needs: a wrapped centre of mass
  teleports by a box length. Both are produced by GROMACS from the tpr (which GROMACS
  can always read even when MDAnalysis cannot — the 2026 tpr format is newer than
  MDAnalysis 2.10 supports; the topology used is `prod.gro`, whose residues are the
  chains).
* **Per chain, never pooled into one selection.** The Rg of twenty chains selected
  together is the size of the box contents, not of a chain.
* **The gates are the same four-state gates the density passes through.** First real
  results, polypropylene at 300 K: bulk modulus 582/545 MPa across two replicas;
  Rg 1.01/1.06 nm and end-to-end 2.46/2.51 nm — a ratio of ≈2.43 against the ideal-chain
  √6 ≈ 2.45; and the diffusion coefficient **refused** with "subdiffusive: MSD grows
  more slowly than linearly, as expected for a polymer melt in the Rouse or reptation
  regime". That refusal is a physics finding, not a failure.
* **Single-replica verdicts are mostly INCONCLUSIVE by design** — the `min_replicas: 3`
  sampling gates fire. Rg's effective-sample shortfall is *time*-limited (chain
  conformations decorrelate over nanoseconds), so denser frames do not help; longer
  runs or replica pooling do.
