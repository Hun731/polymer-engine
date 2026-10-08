# Simulation layer

Status labels: [STATUS_LEGEND.md](STATUS_LEGEND.md)

Everything between "here is a polymer" and "here is a number with an uncertainty":
force-field selection, system construction, GROMACS input generation, replica
materialisation, scheduling, execution, and property extraction.

---

## Capability table

| Capability | Module | Status |
|---|---|---|
| Force-field candidate selection | `simulation/forcefield.py` | REAL |
| Force-field *approval* | — | REQUIRES-EXPERT-DECISION |
| System import from an existing directory | `simulation/builder.py` | REAL |
| System validation (12 failure modes) | `simulation/system.py` | REAL |
| Safe archive extraction | `simulation/archive.py` | REAL (30 security tests) |
| `.gro` / `.top` / `.itp` / `.xvg` parsing | `simulation/formats.py` | REAL |
| `.mdp` generation for EM/NVT/NPT/production | `simulation/mdp.py` | REAL — accepted by real `grompp`, zero warnings |
| Replica materialisation with distinct seeds | `simulation/replicas.py` | REAL |
| Local execution of GROMACS stages | `local/runner.py`, `executors/gromacs.py` | REAL, REQUIRES-LOCAL-SOFTWARE |
| Resource-aware scheduling | `orchestrator/scheduler.py` | REAL |
| 18 property calculators | `properties/` | REAL + FIXTURE-BASED |
| Trajectory observables | `analysis/md.py` | OPTIONAL (MDAnalysis) |
| CHARMM-GUI job submission | `providers/charmm_gui/` | **UNSUPPORTED — no endpoint exists** |
| De-novo chain building and melt packing | `simulation/melt_builder.py` | REAL (saturated hydrocarbons only) |
| OPLS-AA atom typing and topology generation | `simulation/opls_typing.py` | REAL (saturated hydrocarbons only) |
| Solvation, ionisation | — | NOT IMPLEMENTED |
| Anisotropic / semi-isotropic pressure coupling | — | NOT IMPLEMENTED |
| SLURM / HPC submission | — | NOT IMPLEMENTED |
| Coarse-grained models | — | NOT IMPLEMENTED |

---

## Force fields: the engine narrows, a person decides

`ForceFieldAdvisor` matches a polymer against profiles for CHARMM36, OPLS-AA, GAFF2,
OpenFF-2.x and PCFF, and returns a `ForceFieldStrategy` carrying a `Confidence`:

| `Confidence` | Meaning | Usable without review? |
|---|---|---|
| `KNOWN` | a QM validation was supplied **and passed** | **yes** |
| `SUPPORTED` | the profile covers this chemistry, unvalidated here | no |
| `UNCERTAIN` | partial coverage | no |
| `REQUIRES_EXPERT_DECISION` | no profile covers it, **or several do** | no |

Two things about this table matter.

`KNOWN` is reachable **only** by passing `qm_validation` that itself passed. There is no
path in which the advisor's own opinion produces `KNOWN`. "This force field is commonly
used for this chemistry" is a literature fact, not evidence about your system.

Several covering profiles produce `REQUIRES_EXPERT_DECISION`, not a ranked pick. When
CHARMM36 and OPLS-AA both cover a chemistry, choosing between them is a scientific
judgement about which published parameterisation suits the property being measured.
Silently taking the first alphabetically would be a fabricated decision.

`require_ready_force_field()` raises unless the strategy is usable without review, so a
workflow cannot proceed on an unapproved field by forgetting to check.

---

## System construction: the honest boundary

The engine builds melts for the chemistries whose force-field parameters are actually
tabulated, and refuses the rest. See [FORCE_FIELDS.md](FORCE_FIELDS.md) for where that
line falls and why.

`SystemBuilder` dispatches to backends, each of which reports one of six statuses:

| `BuildStatus` | When |
|---|---|
| `BUILT` | the backend constructed the system |
| `IMPORTED` | an existing, validated system was taken in |
| `FAILED` | the attempt ran and failed |
| `UNSUPPORTED` | the backend cannot do this at all |
| `REQUIRES_EXPERT_DECISION` | a human must supply something the engine will not guess |
| `REQUIRES_INPUT` | credentials or an external job id are missing |

`LocalDirectoryBackend` — REAL. Imports a directory, runs the full validation suite,
records provenance hashes for every file.

`CharmmGuiImportBackend` — returns `UNSUPPORTED` for submission, with the message:

> CHARMM-GUI publishes no job-submission endpoint …

and `required_actions` pointing at the web interface. This is deliberate and permanent.
Automating submission would mean reverse-engineering an undocumented form and guessing
at parameters — which is exactly the "undocumented API endpoint assumption" the charter
forbids. Downloading and importing a job **you** submitted is supported, and is
REQUIRES-CREDENTIALS.

A de-novo builder is a large problem in its own right, and a half-correct builder is
worse than none: it produces systems that run, look plausible, and are wrong.

---

## GROMACS input generation

`simulation/mdp.py` generates four stages — minimisation, NVT, NPT, production — from
the configured `SimulationDefaults`. Every parameter written is also recorded in the
stage's `parameters` dict, so the manifest and the `.mdp` cannot disagree.

Two defaults are worth naming:

* **`grompp -maxwarn` defaults to 0.** `-maxwarn` suppresses exactly the diagnostics
  that catch a broken system. Raising it must be a deliberate, recorded choice.
* **Minimisation runs unconstrained** (`constraints = none`), so a bad starting geometry
  can actually relax rather than being frozen into place.

The strongest check on this module is not a unit test: generated stages are fed to a
real `gmx grompp` and must be accepted **with zero warnings**
(`test_generated_mdp_files_are_accepted_by_grompp`).

GPU offload is opt-in and driven by discovered capability. Passing `-nb gpu` to a
CPU-only build makes GROMACS abort, so `mdrun` checks the discovered capability first
and returns an `unavailable` result rather than a failed run.

---

## Replicas

Replicas get distinct, recorded velocity seeds; a property-based test covers the seed
space. Three replicas is the configured minimum, because you cannot estimate a spread
from fewer, and the replica-agreement machinery treats **replica means** as the
independent experimental units — not frames. See [SCIENCE.md](SCIENCE.md) for why that
distinction decides whether a result is real.

---

## Scheduling

`orchestrator/scheduler.py` provides a resource-aware job scheduler with a
`ResourcePool` covering CPUs, GPUs, memory and disk.

`exceeds_capacity()` returns a **reason string** rather than a boolean, so a job that
can never run is distinguishable from one that is merely waiting:

```
"needs 4 GPUs but the pool has 2; VRAM cannot be pooled across cards"
```

That last clause is a physical fact the scheduler must not paper over: a 40 GB model
does not fit on two 24 GB cards. `release()` clamps at zero, so a double release cannot
inflate the pool into oversubscription.

Non-oversubscription is verified by counting *concurrently executing handlers inside a
lock*, not by reading the scheduler's own report — a report that says "2 running" proves
nothing about what actually ran.

Scheduling is local (`LocalExecutor`, a thread pool). HPC submission is NOT IMPLEMENTED.

---

## Properties

18 calculators across five classes. Every one declares — as data, not prose — its
units, the observable it reads, the estimator, the uncertainty method, and its sampling
requirements.

```bash
polymer-engine property list
polymer-engine property compute density production.xvg --replicas 3
```

| Property | Class | Units | Uncertainty | Min N_eff / replicas / ns |
|---|---|---|---|---|
| `density` | thermodynamic | kg/m^3 | correlation-aware SE | 20 / 3 / 5 |
| `potential_energy` | thermodynamic | kJ/mol | correlation-aware SE | 20 / 3 / — |
| `enthalpy` | thermodynamic | kJ/mol | correlation-aware SE | 20 / 3 / — |
| `temperature` | thermodynamic | K | correlation-aware SE | 20 / 1 / — |
| `pressure` | thermodynamic | bar | correlation-aware SE | 50 / 3 / — |
| `volume` | thermodynamic | nm^3 | correlation-aware SE | 20 / 3 / — |
| `thermal_expansion_coefficient` | thermodynamic | 1/K | fit covariance | 20 / 3 / — |
| `bulk_modulus` | mechanical | MPa | block bootstrap | 200 / 3 / 50 |
| `tensile_response` | mechanical | MPa | fit covariance | 5 / 3 / — |
| `radius_of_gyration` | structural | nm | correlation-aware SE | 20 / 3 / 20 |
| `end_to_end_distance` | structural | nm | correlation-aware SE | 20 / 3 / 20 |
| `persistence_length` | structural | nm | fit covariance | 20 / 3 / 20 |
| `free_volume_fraction` | structural | 1 | block bootstrap | 20 / 3 / — |
| `mean_squared_displacement` | transport | nm^2 | none | 10 / 3 / 10 |
| `diffusion_coefficient` | transport | nm^2/ps | fit covariance | 10 / 3 / 50 |
| `relaxation_time` | dynamical | ps | block bootstrap | 10 / 3 / — |
| `hydrogen_bond_count` | intermolecular | 1 | correlation-aware SE | 20 / 3 / — |
| `contact_number` | intermolecular | 1 | correlation-aware SE | 20 / 3 / — |

`bulk_modulus` needs **200** effective samples where `density` needs 20, because it is a
*variance* estimator and a variance converges far more slowly than a mean. The number
is per-property because the statistics are per-property.

A `PropertyResult` is `usable` only when the determination is `KNOWN` **and** the gate
report is promotable. Both, not either.

### Units are declared, and the declaration is checked

Four calculators once declared `units="1"` while carrying a volume in nm³, a
coefficient in K⁻¹, an MSD in nm², and a diffusivity in nm²/ps — with the true unit
mentioned only in prose. That is precisely the hidden-unit failure the engine exists to
prevent: a "dimensionless" volume converts without complaint against any other
dimensionless number.

The fix registered `volume`, `area`, `diffusivity` and `inverse_temperature` as real
dimensions in `core/units.py`. A regression test now asserts that **every** registered
property declares a unit the registry knows, and that `convert(1.0, "nm^2/ps", "nm^3")`
raises.

---

## Mechanical properties: the strain-rate honesty gate

MD tensile deformation runs at 10⁷–10⁹ s⁻¹. Laboratory tensile testing runs near
10⁻³ s⁻¹, and even split-Hopkinson bar impact testing reaches only about 10⁴ s⁻¹.

```python
EXPERIMENTAL_STRAIN_RATE_CEILING = 1.0e4
```

Above that, `interpret_strain_rate()` returns `comparable_to_experiment=False`. The
metrics are named accordingly — `yield_stress_proxy`, `peak_stress` — and carry the note
that they are **not** an experimental tensile strength. A polymer pulled a million times
faster than any experiment is exhibiting a different physical process, not a noisy
version of the same one.

The engine will report the proxy. It will not call it a tensile strength.

---

## Transport properties: the regime must be diffusive

`classify_regime()` fits the MSD scaling exponent `α` on a log-log plot:

The classification is ordered, so the diffusive band is claimed first and only then
the more exotic regimes:

| Test, in order | Regime |
|---|---|
| `abs(alpha - 1) <= 0.15` | diffusive — the Einstein relation applies |
| `alpha >= 1.7` | ballistic — directed motion |
| `alpha > 1.15` | superdiffusive — the diffusive regime has not been reached |
| otherwise | subdiffusive — Rouse/reptation; a slope here is not a diffusion coefficient |

`UNDETERMINED` is returned separately when the log-log fit itself is not usable (too
few lag times, non-positive MSD), rather than being folded into a regime.

`DiffusionCoefficient.compute()` **refuses** to apply the Einstein relation outside the
diffusive regime, returning `INSUFFICIENT_DATA` with the measured α in the message. A
polymer melt can take hundreds of nanoseconds to reach diffusion; fitting a slope to
subdiffusive motion produces a number that is not a diffusion coefficient.

Finite-size effects on D in a periodic box are substantial and are **not** corrected —
the Yeh–Hummer correction needs the shear viscosity, which this engine does not compute.
That caveat travels with the result.
