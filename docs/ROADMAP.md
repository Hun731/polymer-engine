# Roadmap

This document is deliberately blunt about what does **not** exist. A roadmap that reads
as a feature list is how documentation drifts away from code.

Status labels: [STATUS_LEGEND.md](STATUS_LEGEND.md).

## Implemented and verified

| Capability | Verified how | Status |
|---|---|---|
| Configuration, units, typed errors, provenance | unit + property tests | REAL |
| HTTP transport: retry, backoff, rate limit, cache, error classification | 26 tests, offline | REAL |
| 7 providers with normalised records | recorded fixtures, full failure matrix | FIXTURE-BASED |
| CHARMM-GUI login / status / polling / download | mocked | MOCKED, REQUIRES-CREDENTIALS |
| Local tool discovery with versions and capabilities | real GROMACS 2026.3, ORCA 6.1.1 | REAL |
| Polymer identity, deduplication, family classification | 18 real polymers | REAL |
| Property/unit normalisation | two-route temperature check | REAL |
| Dataset ingestion that reconciles every row | rows = accepted + rejected + duplicates | REAL |
| Safe archive extraction | 30 security tests | REAL |
| GROMACS format parsers (`.gro`, `.top`, `.xvg`) | malformed-input tests | REAL |
| System validation incl. topology↔coordinate agreement | valid + 12 broken systems | REAL |
| `.mdp` generation | **real `grompp`, zero warnings** | REAL |
| Replica materialisation with distinct seeds | property test over seed space | REAL |
| Correlation-aware statistics | AR(1) theory | REAL |
| Convergence and replica-agreement gates | synthetic + real MD | REAL |
| Trajectory analysis with explicit units | analytic fixtures | REAL, OPTIONAL (MDAnalysis) |
| Umbrella planning, WHAM, PMF with uncertainty | analytic PMF, 0.45 kJ/mol | FIXTURE-BASED |
| QSPR with leakage protection | grouped CV, noise control | REAL, OPTIONAL (sklearn) |
| Active learning with screening and diversity | rejection tests | REAL |
| Candidate generation on molecular graphs | validity + constraint tests | REAL, OPTIONAL (RDKit) |
| Planner, campaign, runner, strategy registry | deterministic-choice tests | REAL |
| Claims with evidence-based promotion | promotion-refusal tests | REAL |
| Durable state and resumption | reload tests | REAL |
| CLI with meaningful exit codes | 42 tests across 12 command groups | REAL |

### Added in the scientific-execution layer

Everything below was listed under "Not implemented" in the previous revision.

| Capability | Verified how | Status |
|---|---|---|
| ORCA input generation, 6 job kinds | real ORCA accepts them | REAL |
| ORCA execution | 5 tests run ORCA 6.1.1 | REAL, REQUIRES-LOCAL-SOFTWARE |
| ORCA output parsing, five-state classification | real logs incl. an exit-0-but-failed run | REAL + FIXTURE-BASED |
| MM-vs-QM geometry and torsion validation | ethane barrier 12.01 vs ~12.1 kJ/mol | REAL |
| Force-field advisor with confidence levels | coverage and ambiguity tests | REAL |
| System builder abstraction (import backends) | valid + rejected systems | REAL |
| Umbrella execution loop, adaptive refinement | analytic sampling; gap-midpoint insertion | REAL |
| `gmx mdrun -plumed` backend | **never executed** — PLUMED absent here | REQUIRES-LOCAL-SOFTWARE |
| Resource-aware scheduler | concurrency counted inside a lock | REAL |
| 18 property calculators | per-property sampling floors, declared units | REAL + FIXTURE-BASED |
| Mechanical properties + strain-rate honesty | proxy naming, comparability flag | REAL |
| Transport properties + regime classification | refusal outside the diffusive regime | REAL |
| Correlation screening with multiplicity control | Bonferroni, no causal vocabulary | REAL |
| Surrogate models with three-axis leakage protection | grouped CV, per-fold preprocessing | REAL, OPTIONAL (sklearn) |
| Failure classification and saturating penalties | ordered-pattern regression test | REAL |
| Research loop with auditable five-field decisions | checkpoint/resume tests | REAL |

## Not implemented

Honest gaps, in rough order of how much they limit the engine today.

### Real PLUMED execution — **the largest unverified integration**

The umbrella *estimator* is validated against an analytic PMF. The umbrella *execution
backend* — `make_gromacs_plumed_runner`, which drives `gmx grompp` + `gmx mdrun -plumed`
— has **never been run**, because PLUMED is not installed on the development machine.
`TestRealPlumedExecution` exercises exactly that path and skips with a stated reason on
every run.

*Needed:* a machine with PLUMED and a PLUMED-enabled GROMACS build. The test is written
and waiting; nothing else has to change.

### System construction

The engine **validates** systems; it does not **build** them. There is no polymer chain
builder, no packing, no solvation, no ionisation. Systems come from CHARMM-GUI or another
external builder, and `SystemBuilder` exists to make that boundary explicit rather than
to hide it.

This is a deliberate boundary. A half-correct builder is worse than none: it produces
systems that run, look plausible, and are wrong.

### CHARMM-GUI job submission — permanently UNSUPPORTED

Not a gap to be filled. CHARMM-GUI publishes no job-submission endpoint, so automating
submission would mean reverse-engineering an undocumented form. `CharmmGuiImportBackend`
returns `UNSUPPORTED` with instructions pointing at the web interface. Downloading and
importing a job **you** submitted is supported.

### MBAR

WHAM is implemented and validated. MBAR is not. For most umbrella work MBAR is the better
estimator — binless, lower variance. The intended route is an optional `pymbar`
integration reporting `UNSUPPORTED` when absent, not a hand-rolled reimplementation.

### Deformation protocols

`properties/mechanical.py` analyses a stress–strain curve, including the strain-rate
comparability gate. What is missing is the **protocol** that produces one: `simulation/mdp.py`
generates isotropic NPT only, and anisotropic and semi-isotropic pressure coupling are not
exposed. Bulk modulus from volume fluctuations works today; a tensile modulus needs a
deformation workflow that does not yet exist.

### Viscosity, thermal conductivity, ion conductivity

MSD, regime classification, a guarded diffusion coefficient and relaxation times exist.
Green-Kubo viscosity with proper plateau detection, thermal conductivity and ion
conductivity do not. The Yeh–Hummer finite-size correction to D is consequently also
absent, since it needs the shear viscosity.

### QM charge derivation

Population charges are parsed. RESP/ESP fitting — the route to actual force-field
parameters — is not implemented. Comparing MM to QM is validation; *generating* parameters
from QM is a separate problem this engine does not attempt.

### LLM-assisted hypothesis generation

Removed during consolidation, and not restored. *Intended design when added:* an LLM may
**propose** hypotheses and candidate actions; the deterministic planner still scores and
selects them, and the validation gates are untouched. The reasoning layer must never be
able to promote a result.

### Literature-grounded claim comparison

`Claim.literature_comparison` is a free-text field. Automatically matching a computed
value against published values — resolving which paper measured the same quantity under
comparable conditions — is not implemented. Doing it badly would manufacture false
agreement, which is worse than leaving it manual.

### Other gaps

- **HPC submission:** the scheduler is local (a thread pool). No SLURM, no queue
  submission, no campaign state surviving a job restart on a cluster.
- **Copolymers:** `PolymerIdentity` has a `copolymer_of` field. Sequence-aware identity,
  descriptors and generation are not implemented.
- **Crosslinked networks:** architecture is recorded; no network-specific handling.
- **Coarse-grained models:** all-atom assumptions throughout (mass, cutoffs, timestep).
- **Glass transition:** no `Tg` protocol. `ThermalExpansion` explicitly warns that a
  single linear fit across a transition is invalid, but does not detect the transition.
- **Chain orientation and segmental dynamics:** not implemented.

## Planned

### Near term

1. Run the real-PLUMED test on a machine that has PLUMED, and record the result.
2. Anisotropic pressure coupling and a tensile-deformation protocol, closing the gap
   between the mechanical analysis that exists and the simulation that produces its input.
3. Optional MBAR via `pymbar`.
4. RESP charge derivation from the ESP already parseable from ORCA output.

### Medium term

5. Copolymer sequence handling across identity, descriptors and generation.
6. Green-Kubo viscosity with plateau detection, then the Yeh–Hummer correction to D.
7. HPC submission (SLURM), with campaign state surviving job restarts.
8. LLM hypothesis proposal behind the deterministic planner.

### Long term

9. A polymer system builder, or a documented integration with an existing one.
10. Automated literature comparison with explicit confidence in the match.
11. Multi-objective Bayesian optimisation over the candidate space.

## Non-goals

Things deliberately excluded:

- **Self-modifying source code.** Adaptation happens through strategy ranking and bounded
  parameter adjustment. Rewriting Python at run time is not on the roadmap.
- **Automatic gate relaxation.** No mechanism will ever weaken a validation gate to let a
  workflow proceed.
- **Automatic selection of a force field, level of theory, reaction coordinate or
  acceptance tolerance.** These are `REQUIRES-EXPERT-DECISION` permanently. Adding a
  default would not be a feature; it would be a silently fabricated scientific decision.
- **Synthetic accessibility scoring** without a real retrosynthesis model.
- **Force-field parameter generation** without QM validation.
- **A general-purpose MD front end.** This engine orchestrates a specific, gate-checked
  polymer-discovery workflow.
