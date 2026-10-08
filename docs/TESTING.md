# Testing

Status labels: [STATUS_LEGEND.md](STATUS_LEGEND.md)

```bash
pytest                          # 1016 tests, no network required
pytest -rs                      # ... and list every skip with its reason
pytest -m "not slow"            # skip the real-MD and real-QM tests
pytest tests/security           # security suite only
pytest --cov --cov-report=term  # coverage
```

On a machine with GROMACS 2026.3 and ORCA 6.1.1 installed and PLUMED absent:
**1016 passed, 1 skipped, 0 failed**, 86 % line coverage, ruff clean, mypy clean over
90 source files.

| Suite | Tests |
|---|---|
| `unit` | 238 |
| `providers` | 126 |
| `analysis` | 124 |
| `simulation` | 112 |
| `orchestration` | 107 |
| `qm` | 89 |
| `properties` | 69 |
| `security` | 52 |
| `science` | 32 |
| `property` (Hypothesis) | 26 |
| `e2e` | 22 |
| `integration` | 19 |

## Layout

```text
tests/
├── conftest.py         shared fixtures; autouse network block
├── markers.py          skip markers that state *why* they skipped
├── unit/               single modules in isolation
├── providers/          transport, each provider, the registry
├── simulation/         formats, system validation, manifests, umbrella execution
├── analysis/           statistics, convergence, free energy, trajectories
├── qm/                 ORCA input, parsing, classification, MM-vs-QM validation
├── properties/         the 18 property calculators and their refusals
├── science/            correlation screening, leakage-protected models
├── orchestration/      planner, strategy learning, scheduler, autonomy
├── integration/        provider→store, system→campaign, campaign→analysis, lineage
├── security/           archive safety, secret leakage
├── property/           Hypothesis invariants
├── e2e/                whole pipelines; real GROMACS, ORCA, PLUMED when available
└── fixtures/
    ├── orca/           real ORCA 6.1.1 output, including an exit-0-but-failed run
    ├── providers/      recorded HTTP responses
    ├── systems/        a valid system and its broken variants
    └── trajectories/   analytic trajectories with known answers
```

## No network, structurally

`conftest.py` installs an autouse fixture that makes `socket.connect` and
`socket.create_connection` raise. A test that tries to reach the internet fails
loudly rather than passing on a developer's machine and failing in CI.

Provider behaviour is exercised through `FixtureTransport`, which serves canned
responses and **raises on an unmatched request** — so a forgotten fixture is a visible
error, not a silent live call.

## Skips are never silent

```python
requires_gromacs = pytest.mark.skipif(
    shutil.which("gmx") is None,
    reason="GROMACS 'gmx' is not on PATH; real-execution tests cannot run in this environment",
)
```

`pytest -rs` lists every skip with its reason. Optional dependencies (RDKit,
MDAnalysis, scikit-learn, SciPy) and external tools (gmx, orca, plumed) each have a
marker.

This matters more than it looks. On the development machine exactly **one** test skips,
and its reason is the honest statement of the engine's largest unverified integration:

```
SKIPPED [1] tests/e2e/test_research_pipeline.py:
            PLUMED is not on PATH; real-execution tests cannot run in this environment
```

A suite that silently omitted that test would report a clean run and imply that
`gmx mdrun -plumed` had been exercised. It has not been.

## What each suite proves

### Numerical correctness, against known answers

Tests assert on *numbers*, not on "it ran".

| Test | Checks against |
|---|---|
| autocorrelation time | AR(1) theory: φ=0.9 ⇒ τ=9.0, g=19.0 |
| WHAM | an analytic harmonic PMF, to 0.45 kJ/mol over 21 kJ/mol |
| radius of gyration | 5 unit masses at x=0..4 ⇒ Rg = √2 Å exactly |
| end-to-end distance | 0.4 nm exactly |
| MSD | tracer moving 1 Å/frame ⇒ MSD(lag) = (0.1·lag)² |
| diffusion coefficient | 3-D random walk ⇒ D = 3σ²/6, within 10 % |
| descriptors | PS repeat unit ⇒ ethylbenzene, 106.17 g/mol |
| unit conversion | 105 °C ⇒ 378.15 K, via two independent routes |
| topology↔coordinates | 1 POL (8 atoms) + 20 SOL (3 atoms) = 68 |

### Security

`tests/security/test_archive_safety.py` — path traversal (tar and zip), absolute
members, Windows backslash traversal, symlinks, hardlinks, zip symlink attributes,
device/FIFO members, size and member-count limits, compression bombs, truncated
archives, HTML masquerading as an archive, and **that nothing is written when any
member is unsafe** (a partial extraction leaks attacker content).

`tests/security/test_secrets.py` — masked `repr`, redacted config dumps, log scrubbing
of both known secrets and credential-shaped patterns, tokens absent from provider
results and error bodies, cache keys that exclude `Authorization`, refusal to read a
group-readable credential file, and a scan of `src/` for committed credentials.

### Property-based invariants

`tests/property/` uses Hypothesis for rules that must hold for *all* inputs:

- an extracted path can never escape its root, for any member name
- the state machine never enters an illegal state, for any command sequence
- `COMPLETED → RUNNING` is never permitted
- unit conversion round-trips losslessly; cross-dimension conversion always raises
- a temperature *interval* is preserved across K↔°C
- effective samples never exceed raw samples; `g ≥ 1`; uncertainty ≥ 0
- uncertainty is translation-invariant
- replica seeds are always distinct and always reproducible
- the recommended umbrella spacing always satisfies the overlap criterion
- deduplication is order-independent

### Failure-mode coverage

Every provider is tested for: valid response, empty result, malformed result, HTTP
429, HTTP 5xx, timeout, connection failure, duplicates, cache hit, cache miss.

CHARMM-GUI adds: successful login, invalid credentials, expired token, 401, 403, 429,
5xx, malformed JSON, all four job states, unrecognised status, download failure,
corrupt archive, empty download, digest mismatch, and that submission raises rather
than guessing an endpoint.

System validation adds: valid system, missing topology, missing coordinates, missing
include, unresolvable molecule type (⇒ `INCONCLUSIVE`), atom-count mismatch,
non-finite coordinates, missing box, zero box dimension, contents larger than the box,
missing `[ molecules ]`, empty directory, broken symlink.

### The gates actually refuse things

The most important tests are the ones that assert a *refusal*:

- a dry run is `SKIPPED`, never `SUCCEEDED`, and its record is `CANCELLED`
- a dry run does not satisfy a downstream dependency
- 2 replicas where 3 are required is `INCONCLUSIVE`, not `PASS`
- a replica with no data fails the gate rather than being skipped
- disagreeing replicas fail on reduced chi-square
- a PMF with poor overlap reports no barrier
- **WHAM converging is not evidence the PMF is right** — disjoint windows converge in a
  single iteration because the bins decouple; the test asserts that WHAM converged *and*
  `pmf.trustworthy is False` *and* the overlap gate FAILs
- an umbrella campaign without a justification runs **zero windows**
- ballistic motion is refused a diffusion coefficient
- an ORCA job that exited 0 without converging is `FAILED_SCIENTIFICALLY`, and
  `qm parse` exits 3
- an optimisation whose log never mentions convergence fails — silence is not consent
- an MM-vs-QM comparison without acceptance criteria returns `passed=None`
- a chemistry covered by two force fields yields `REQUIRES_EXPERT_DECISION`, not a pick
- CHARMM-GUI submission returns `UNSUPPORTED`, not a guessed endpoint
- a strain rate above 1e4 s⁻¹ is not comparable to experiment
- the scheduler never oversubscribes — verified by counting concurrent handlers inside a
  lock, not by reading its own report
- a property whose declared unit the registry does not recognise fails the suite
- a training-set member is rejected as a candidate
- a claim cannot reach `SUPPORTED` from dry-run evidence
- an out-of-bounds strategy parameter is rejected, not clamped
- an unknown action kind raises rather than running a stub

### Real-software tests

Ten tests execute an external scientific tool for real. Nine run here; one cannot.

| Tool | Tests | Status on this machine |
|---|---|---|
| GROMACS 2026.3 | 4 | run |
| ORCA 6.1.1 | 5 | run |
| PLUMED | 1 | **skipped — not installed** |

**GROMACS** (`test_full_campaign.py::TestRealExecution`,
`test_research_pipeline.py::TestFullPipeline`, `test_local_runner.py`):

1. A short two-replica campaign runs to completion — every GROMACS stage exits 0 — and
   the analysis gate **fails**, because 10 ps across 2 replicas is not a result. That is
   the whole thesis of the engine in one assertion.

   The recorded benchmark run in `examples/benchmark_campaign/README.md` shows what the
   refusal looks like in numbers, from the same code path: 1001 frames written, and

   ```
   temperature: 1001.0 effective samples from 1001 frames (statistical inefficiency 1.0)
   potential:     87.4 effective samples from 1001 frames (statistical inefficiency 11.4)
   density:        2.9 effective samples from 1001 frames
   ```

   The refusal is property-specific, not a blanket "too short".
2. Every generated `.mdp` passes real `gmx grompp` with **zero warnings**.
3. Tool discovery reports the real version and capabilities (GPU, MPI, SIMD, precision).

**ORCA** (`test_qm_workflow.py::TestRealOrcaExecution`,
`test_research_pipeline.py`, `test_local_runner.py`): a real single point, a real
optimisation, and a real relaxed torsion scan of ethane whose barrier comes out at
**12.01 kJ/mol** against an experimental ~12.1. Parsing, classification and unit
conversion are exercised on output ORCA actually produced.

**PLUMED** (`test_research_pipeline.py::TestRealPlumedExecution`): drives
`gmx grompp` + `gmx mdrun -plumed` across real umbrella windows. It has **never been
executed** — PLUMED is not installed on the development machine, and the test says so
on every run rather than being quietly absent. The umbrella *estimator* is validated
against an analytic PMF; the umbrella *execution backend* is not validated at all.
Those are different claims and this document does not merge them.

### QM fixtures: real output, never hand-written

`tests/fixtures/orca/` holds output ORCA 6.1.1 produced, including
`scan_nonconverged.out` — a run that **exited 0**, printed
`ORCA finished by error termination`, and never printed the normal-termination banner.
It is the reference case for "a zero exit code is not scientific success".

Where a fixture is an excerpt, the elision is marked in the file itself. Fixtures are
never edited to make a test pass; if a fixture disagrees with the parser, the parser is
wrong.

## Coverage

**86 % line coverage** over 10,784 statements. Uncovered code is mostly error paths for
conditions that need a specific broken environment (unreadable `/proc/meminfo`, a tool
that disappears mid-probe) or the real-PLUMED execution path that cannot run here.

Coverage is not a target to game. A module at 100 % that only asserts "it ran" is
worth less than one at 70 % whose tests check numbers against theory.

## Adding tests

Two rules:

1. **Assert on a value, not on the absence of an exception.** If there is no known
   answer, construct a fixture that has one.
2. **Every bug gets a regression test.** Every fix in this repository has one; they are
   listed in [CHANGELOG.md](../CHANGELOG.md). Two worth reading, because both were
   silent:
   * `TestOutputCapture` — the runner kept the *last* 200 KB of stdout, and ORCA prints
     its banner at the *front*, so every long ORCA log arrived `UNPARSEABLE`.
   * `TestDeclaredUnitsAreReal` — four calculators declared `units="1"` while carrying a
     volume in nm³, a coefficient in K⁻¹, an MSD in nm² and a diffusivity in nm²/ps.
3. **A test that cannot run must say so.** Never delete a real-software test because the
   software is missing locally, and never let it pass vacuously. Mark it, give the marker
   a reason, and let `-rs` report it.

```python
def test_density_gate_fails_when_still_drifting():
    values = 1000.0 + 0.02 * np.arange(5000)          # a known, deliberate drift
    analysis = analyse_series(values, name="density", units="kg/m^3")
    gate = next(g for g in convergence_gates(analysis) if g.gate == "density:drift")
    assert gate.status is GateStatus.FAIL
    assert gate.value > gate.threshold                 # and for the right reason
```

## Browser tests

`tests/browser/` is split three ways, because the three kinds of evidence are worth
different amounts.

**Fake driver** (`conftest.py`) — a scripted page implementing the same JSON protocol
the real worker speaks. This is how the situations that matter most and are hardest to
trigger on demand get tested deterministically: a CAPTCHA appearing mid-login, a select
silently rejecting a value, a form that changed since discovery, two job ids on one page.
These prove the engine's *decisions*. They prove nothing about CHARMM-GUI, and no test
there claims otherwise.

**Real browser** (`test_real_browser.py`) — Playwright and Chromium from `.browserenv`
driving local fixture pages. This covers what a fake cannot vouch for: that the browser
environment really exists, that the DOM-reading JavaScript is valid, and that
`keyboard.type` dispatches genuine key events.

The login fixture **rejects paste events and counts keydowns**, signing in only after
more than five real keystrokes. A `fill()`-based implementation cannot pass it, which
makes that test the closest thing to a proof of the keyboard-typing requirement
obtainable without the live site. Skipped when `.browserenv` is absent.

**Live** (`test_live_charmm_gui.py`) — off unless `CHARMM_GUI_LIVE_TEST=1` *and*
credentials are present. A real submission needs a second opt-in
(`CHARMM_GUI_LIVE_SUBMIT=1`) plus an explicitly named monomer and submit control;
neither is guessed and there is no bulk path. Ordinary CI never needs credentials and
never touches the live service.

### Security tests that check structure, not intent

`TestBrowserCredentials` in `tests/security/test_secrets.py` parses the source rather
than reading it: no `--password` flag exists anywhere, the worker's `type_secret` reads
only from its own environment, `read_value` refuses password-like keys, and no clipboard
API appears in executable code. The clipboard scan uses `ast` and skips docstrings —
the docstrings deliberately name those APIs to record that they are not used, and a
naive text scan would flag its own documentation.

Each of these was verified to fail when the violation it describes is introduced.
