# Polymer Autonomous Research Engine

A reproducible orchestration framework for computational polymer discovery: data
curation, system construction, molecular dynamics, enhanced sampling, statistical
validation, surrogate modelling, and rational design — with deterministic scientific
gates between every stage.

The organising principle is that **a pipeline that cannot refuse a result is not a
scientific instrument.** GROMACS exiting 0 is not evidence. A dry run is not a
success. One replica is not reproducibility. Ten thousand correlated frames are not
ten thousand measurements. Each of those is enforced in code and covered by tests.

---

## Status at a glance

Labels are defined in [STATUS_LEGEND.md](docs/STATUS_LEGEND.md) and used consistently
across this documentation set. They describe what has actually been *executed*, not what
the code intends to do.

| Area | State |
|---|---|
| Core (config, units, errors, models, provenance) | REAL |
| Providers (7, with offline fixtures) | FIXTURE-BASED; CHARMM-GUI is REQUIRES-CREDENTIALS |
| Local tool discovery + runners | REAL — GROMACS 2026.3, ORCA 6.1.1 |
| Polymer identity, taxonomy, descriptors, ingestion | REAL (descriptors OPTIONAL: RDKit) |
| System import + validation gates | REAL, incl. 30 archive-security tests |
| GROMACS input generation | REAL — warning-free through real `grompp` |
| Convergence / replica gates | REAL — validated against AR(1) theory |
| **ORCA / QM workflow** | REAL — 5 tests execute ORCA; ethane barrier 12.01 kJ/mol |
| **Umbrella execution + WHAM/PMF** | Estimator FIXTURE-BASED (analytic PMF); `mdrun -plumed` backend REQUIRES-LOCAL-SOFTWARE and is **not verified here** |
| **18 property calculators** | REAL + FIXTURE-BASED |
| **Resource-aware scheduler** | REAL |
| **Research loop, failure learning** | REAL |
| QSPR + active learning | REAL, OPTIONAL (scikit-learn), leakage-protected |
| Candidate generation | REAL, OPTIONAL (RDKit) |
| Orchestrator, strategies, claims, CLI | REAL |
| Force field / level of theory / reaction coordinate | REQUIRES-EXPERT-DECISION — the engine refuses to choose |
| System building, MBAR, HPC submission, LLM proposal | NOT IMPLEMENTED — see [ROADMAP](docs/ROADMAP.md) |

**1016 tests, 1 skipped, 86 % coverage, ruff clean, mypy clean.** The one skip is the
real-PLUMED umbrella test, and it reports its reason on every run rather than being
quietly absent. See [TESTING.md](docs/TESTING.md) for what each suite actually proves.

---

## Installation

```bash
git clone <repository-url> && cd Polymer
python -m venv .venv && source .venv/bin/activate

pip install -e .              # core engine + CLI
pip install -e ".[science]"   # RDKit, MDAnalysis, scikit-learn, SciPy, pandas
pip install -e ".[dev]"       # pytest, hypothesis, ruff, mypy
```

Python 3.11+. The scientific extras are genuinely optional: every module that needs
one degrades to an explicit `UNSUPPORTED` result rather than crashing or guessing.

**External software** is *not* installed by pip and must be on `PATH` (or configured):

| Tool | Needed for | Without it |
|---|---|---|
| GROMACS ≥ 2021 | running MD | inputs still generate and validate; execution is blocked |
| PLUMED ≥ 2.7 | umbrella biasing | window planning and PMF analysis still work on existing COLVAR files |
| ORCA ≥ 5 | QM reference calculations and force-field validation | inputs still generate; execution is blocked |

```bash
polymer-engine engine discover-tools    # what is actually available here
```

---

## Quickstart

```bash
polymer-engine engine init
polymer-engine engine resources

# Curate polymer data (units normalised, identities canonicalised, rows reconciled)
polymer-engine polymer descriptors '*CC(*)c1ccccc1'
polymer-engine polymer ingest data/polymers.csv --output data/records.jsonl

# Import and validate a simulation system
polymer-engine system import ~/Downloads/charmm-gui-1234.tgz
polymer-engine system validate data/charmm-gui-1234

# Build a campaign (nothing runs yet)
polymer-engine campaign create pe-density \
    --polymer pol_7cf1ea13aa5bbc1b \
    --system data/charmm-gui-1234 \
    --question "What is the equilibrium density of polyethylene at 300 K?"
polymer-engine campaign plan pe-density

# Dry run: validates every input, executes nothing
polymer-engine campaign run pe-density

# Actually run MD (requires GROMACS)
polymer-engine campaign run pe-density --execute
polymer-engine campaign analyze pe-density

# Audit
polymer-engine evidence manifest pe-density
polymer-engine evidence report

# QM: validate the force field against a reference calculation
# (method and basis are REQUIRED -- the engine will not choose a level of theory)
polymer-engine qm run ethane.xyz --kind torsion_scan \
    --method B3LYP --basis def2-TZVP \
    --torsion 2,0,1,6 --scan-range 0,360,13 --execute

# Properties: what can be computed, and what sampling each one needs
polymer-engine property list
polymer-engine property compute density production.xvg --replicas 3

# The research loop's own record of what it decided and what went wrong
polymer-engine research decisions
polymer-engine research failures
polymer-engine research correlate data/records.csv --target density
```

A runnable 12-step campaign that exercises QM, MD, gating, correlation and modelling
end to end — with its real recorded output on a machine with GROMACS 2026.3 and ORCA
6.1.1 — is in [examples/benchmark_campaign/](examples/benchmark_campaign/).

A complete worked example, with the real commands and their real output, is in
[examples/README.md](examples/README.md).

### Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | engine or scientific failure |
| 2 | usage error (unknown campaign, bad argument, missing file) |
| 3 | a validation gate did not pass |

Code 3 exists so CI can distinguish "the tool broke" from "the science did not
pass". A failed gate is a *result*, not a crash.

---

## Configuration

Precedence, highest first: CLI arguments → environment variables → config file →
declared defaults. See [`configs/default.yaml`](configs/default.yaml), which
documents every setting.

```bash
export POLYMER_ROOT=/scratch/polymer
export POLYMER_GMX=/opt/gromacs-2026.3/bin/gmx     # pins an exact binary
export POLYMER_EXECUTION_ENABLED=false             # local software off by default
```

Every scientific assumption — temperature, pressure, timestep, thermostat, barostat,
cutoff, replica count, seeds, window spacing, convergence tolerances — is
configuration, is copied verbatim into each campaign manifest, and contributes to the
campaign fingerprint. None of it is hard-coded in the simulation layer.

### Credentials

Never in the repository, never in a log, never in a manifest.

```bash
export CHARMM_GUI_EMAIL=you@lab.org
export CHARMM_GUI_PASSWORD=...      # or CHARMM_GUI_TOKEN
export MP_API_KEY=...               # Materials Project
export CROSSREF_MAILTO=you@lab.org  # polite-pool identification
```

Or point `credentials.charmm_gui_token_file` at a `chmod 600` file; the engine
refuses to read a credential file that is group- or world-readable. Secrets are held
in a `Secret` wrapper whose `repr` is masked, and a log filter scrubs both known
secrets and credential-shaped patterns. This is enforced by
[`tests/security/test_secrets.py`](tests/security/test_secrets.py).

---

## Real-machine setup

```bash
# 1. Confirm the toolchain, versions and capabilities the engine will actually use
polymer-engine engine discover-tools
# Reports path, version, GPU/MPI/precision, version compatibility, and warns when
# several installations of the same tool are on PATH.

# 2. Confirm resources
polymer-engine engine resources     # CPUs, memory, GPUs, free scratch space

# 3. Pin what matters for reproducibility, in polymer.yaml
#    local_tools.gromacs.path, resources.gpu_available, simulation.force_field

# 4. Rehearse with execution disabled — every input is validated, nothing runs
polymer-engine campaign run my-campaign

# 5. Enable execution only when the dry run is clean
polymer-engine campaign run my-campaign --execute
```

---

## Architecture

```text
polymer_engine/
├── core/          config, logging, errors, units, models, provenance
├── providers/     PubChem, Crossref, Europe PMC, OpenAlex, RCSB PDB,
│                  Materials Project, CHARMM-GUI  (+ offline test transports)
├── local/         tool discovery, runners, resource inspection
├── polymer/       identity, normalization, descriptors, taxonomy, ingestion
├── simulation/    archives, formats, validation, builder, force fields, mdp,
│                  replicas, umbrella planning and execution
├── qm/            ORCA input, execution, parsing, MM-vs-QM validation
├── properties/    18 calculators with declared units and sampling floors
├── analysis/      statistics, convergence, trajectories, free energy
├── science/       correlation screening, leakage-protected surrogate models
├── discovery/     QSPR, active learning, candidate generation
├── orchestrator/  planner, campaign, runner, scheduler, failure learning,
│                  autonomy, strategy
├── evidence/      claims and their supporting evidence
├── executors/     deterministic action executors
├── db/            durable state (SQLite)
└── cli/           command-line interface
```

Full detail in [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## The rules the code actually enforces

These are not aspirations; each links to the test that holds it.

1. **A dry run is never a success.** `CommandResult.succeeded` is false unless the
   process really ran; a dry-run action is `SKIPPED`, and its execution record is
   `CANCELLED`, not `COMPLETED`.
2. **Exit code 0 is not a scientific result.** A real GROMACS run producing 1001
   frames is still rejected when it yields 2.9 *effective* samples. A real ORCA run that
   exits 0 while printing `ORCA finished by error termination` is classified
   `FAILED_SCIENTIFICALLY` — the fixture proving it is a genuine ORCA log, not a
   hand-written one.
3. **Inconclusive blocks promotion.** Gates are four-valued; a check that could not
   be evaluated never reads as a pass.
4. **One replica is not reproducibility.** Replica agreement with fewer than the
   required replicas is `INCONCLUSIVE`, however precise the single run.
5. **Frames are not independent samples.** Every uncertainty is computed from the
   effective sample size after correcting for autocorrelation.
6. **Units are never silently changed.** Cross-dimension conversion raises; an
   unrecognised unit yields `REQUIRES_VALIDATION` rather than an assumption.
7. **Replicas get distinct, reproducible seeds.** Identical seeds would make replica
   agreement a tautology.
8. **A PMF without overlap is refused.** No barrier or ΔG is reported when adjacent
   windows do not overlap, WHAM did not converge, or the two halves disagree.
9. **Undocumented endpoints are not invented.** CHARMM-GUI job submission is
   declared `UNSUPPORTED` because the service publishes no such endpoint.
10. **Untrusted archives cannot escape their root.** Path traversal, absolute paths,
    links, device files and decompression bombs are all rejected before any write.
11. **Self-modification is bounded.** Strategy parameters may adapt only inside their
    declared bounds; source code is never rewritten.
12. **Claims need validated, replicated, uncertainty-bearing evidence.** A surrogate
    model's prediction is never evidence about the world.
13. **The engine refuses to make scientific choices for you.** Force field, level of
    theory, reaction coordinate and acceptance tolerance are all
    `REQUIRES-EXPERT-DECISION`; the workflow stops rather than defaulting. Umbrella
    sampling without a seven-field justification runs *zero* windows.
14. **An MD proxy is not an experimental measurement.** Above 10⁴ s⁻¹ — faster than any
    laboratory tensile test — mechanical metrics are named `yield_stress_proxy` and
    flagged `comparable_to_experiment=False`.
15. **A correlation never becomes a cause.** Screening corrects for multiplicity, and the
    reporting vocabulary contains no causal verb.
16. **Only usable results teach the engine anything.** A campaign whose gates refused
    everything updates the failure ledger and leaves the knowledge state untouched. The
    engine can spend a week of compute and learn nothing — correctly.

---

## Documentation

| Document | Contents |
|---|---|
| [STATUS_LEGEND.md](docs/STATUS_LEGEND.md) | what REAL / FIXTURE-BASED / MOCKED / … mean |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | module map, data flow, extension points |
| [SCIENCE.md](docs/SCIENCE.md) | every gate, its threshold, and its justification |
| [SIMULATION.md](docs/SIMULATION.md) | system building, MD, scheduling, properties |
| [FORCE_FIELDS.md](docs/FORCE_FIELDS.md) | what can be typed, what is refused, and the QM qualification |
| [PARAMETERIZATION.md](docs/PARAMETERIZATION.md) | the backend system: routing, completeness, charges, qualification |
| [CHARMM_GUI.md](docs/CHARMM_GUI.md) | the human-in-the-loop acquisition route |
| [QM_VALIDATION.md](docs/QM_VALIDATION.md) | what QM validates, and what it does not license |
| [ORCA.md](docs/ORCA.md) | the QM layer and why exit code 0 means nothing |
| [UMBRELLA.md](docs/UMBRELLA.md) | enhanced sampling, WHAM, and what is *not* verified |
| [AUTONOMY.md](docs/AUTONOMY.md) | what the engine decides, and what it refuses to decide |
| [PROVIDERS.md](docs/PROVIDERS.md) | each provider's contract, capabilities and limits |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | manifests, fingerprints, seeds, provenance |
| [TESTING.md](docs/TESTING.md) | test architecture and what each suite proves |
| [ROADMAP.md](docs/ROADMAP.md) | what exists, what does not, and what is planned |
| [EXECUTION_LAYER_REPORT.md](docs/EXECUTION_LAYER_REPORT.md) | what is verified, what is not, and how each was checked |
| [AUDIT_REPORT.md](docs/AUDIT_REPORT.md) | the consolidation audit that preceded this layer |
| [CHANGELOG.md](CHANGELOG.md) | release history |

## Development

```bash
pytest                        # 1016 tests, no network required
pytest -rs                    # ... and list every skip with its reason
pytest -m "not slow"          # skip the real-MD and real-QM tests
ruff check src tests
mypy
pytest --cov --cov-report=term
```

The core suite never touches the network: an autouse fixture makes real socket
connections raise. Tests needing GROMACS, ORCA, PLUMED, RDKit, MDAnalysis or
scikit-learn are skipped with an explicit reason, never silently — run `pytest -rs` and
read them, because the reasons are the honest inventory of what this machine could not
verify.
