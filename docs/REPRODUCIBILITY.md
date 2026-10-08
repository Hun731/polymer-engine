# Reproducibility

A campaign is reproducible when someone else, on another machine, can reconstruct what
you ran and say precisely where their result should and should not match yours.

## The manifest

Every campaign writes `campaign_manifest.json` and stores the same document in the
database. It contains:

| Section | Contents |
|---|---|
| `specification` | the complete `CampaignSpec`: simulation, analysis and umbrella settings |
| `fingerprint` | digest of the deterministic subset (below) |
| `force_field`, `water_model` | recorded verbatim, including `UNSPECIFIED` |
| `software` | discovered tool versions (e.g. `{"gromacs": "2026.3"}`) |
| `random_seeds` | every per-replica, per-stage seed |
| `replicas` | requested vs materialised, directories, `seeds_distinct` |
| `system` | root, coordinate and topology paths, content digest, artifact id, full validation report |
| `analysis_settings` | equilibration rule, thresholds, bootstrap count, RNG seed |
| `actions` | the action graph with cost and information-gain estimates |
| `execution_records` | every state transition with timestamp, actor and reason |
| `gates` | every gate result: value, threshold, units, verdict, message |

```bash
polymer-engine evidence manifest my-campaign --output manifest.json
```

## Fingerprints

`campaign_fingerprint()` hashes only what determines the inputs:

- polymer identity
- the full simulation, analysis and umbrella parameter sets
- the system content digest (SHA-256 of every artifact's digest)

It deliberately excludes campaign ids, paths and timestamps, so two campaigns that
will produce identical inputs fingerprint identically — and changing any scientific
parameter changes the fingerprint.

```python
assert campaign_fingerprint(spec_a) == campaign_fingerprint(spec_b)
```

## What is deterministic

| Artifact | Deterministic? | Why |
|---|---|---|
| campaign fingerprint | **yes** | pure function of the specification |
| generated `.mdp` files | **yes** | byte-identical for the same spec |
| replica seeds | **yes** | CRC32-derived from `(base_seed, replica, stage)` |
| polymer ids | **yes** | SHA-256 of canonical repeat unit + architecture + tacticity |
| artifact digests | **yes** | content-addressed |
| bootstrap intervals | **yes** | seeded RNG |
| cross-validation folds | **yes** | seeded assignment |
| **MD trajectories** | **no** | see below |
| wall-clock timings | no | machine-dependent |

`replica_seed` uses `zlib.crc32`, not Python's `hash()`. String hashing is randomised
per process, so a `hash()`-derived seed would differ on every run while claiming to be
reproducible. This is verified by a property test.

### Why trajectories are not bit-reproducible

Even with identical inputs and seeds, GROMACS trajectories diverge across runs on
different hardware — different SIMD widths, GPU vs CPU kernels, thread counts and
non-deterministic reduction order all change floating-point rounding, and MD is
chaotic, so tiny differences grow exponentially.

**This is expected and is not a bug.** What must reproduce is not the trajectory but
the *ensemble average within its stated uncertainty*. That is exactly what replica
agreement tests: if two people's densities differ by more than their combined error
bars, something is genuinely wrong.

To make a comparison as tight as it can be, pin: the GROMACS build and version, the
thread and GPU configuration, and `base_seed`.

## Provenance

Every artifact carries:

```text
artifact_id, parents, source, source_version, created_at,
software, command, parameters, input_hash, output_hash,
size_bytes, random_seed, environment, units, validation_state
```

Lineage is a DAG walkable in both directions:

```bash
polymer-engine evidence lineage sys_6e17a1a15bf5
```

```python
graph.ancestors(artifact_id)     # everything upstream
graph.roots(artifact_id)         # the original external inputs
graph.descendants(artifact_id)   # everything that depends on it
graph.verify_all()               # re-hash every file-backed artifact
graph.dangling_parents()         # referenced parents with no record — should be empty
```

A verified chain runs archive → extracted system → replica inputs → analysis, and is
asserted end-to-end in `tests/integration/test_pipeline.py::TestLineage`.

`Artifact.verify()` **raises** on a digest mismatch rather than warning. An input that
changed under you invalidates everything downstream, and that should be loud.

Artifact ids are namespaced by campaign (`my-campaign:replica_01`). Deriving them from
a directory name alone gave every campaign a `replica_01_replicas`, so two campaigns
silently shared provenance records — a bug this repository had and now has a
regression test for.

## The environment

`environment_fingerprint()` records the Python version, implementation, OS and
machine architecture. It deliberately omits hostnames and usernames: provenance must
be shareable.

Tool versions come from real probes, not assumptions:

```bash
polymer-engine engine discover-tools
```

This also warns when several installations of the same tool are on `PATH`, which is a
reproducibility hazard. Pin one with `local_tools.gromacs.path`.

## Reconstructing a campaign

```python
manifest = json.loads(Path("campaign_manifest.json").read_text())
spec = CampaignSpec.from_dict(manifest["specification"])

builder = CampaignBuilder(config, software=manifest["software"])
campaign = builder.create(spec)
builder.attach_system(campaign, import_directory(manifest["system"]["root"]))
builder.plan(campaign)

assert campaign.manifest()["fingerprint"] == manifest["fingerprint"]
assert campaign.manifest()["random_seeds"] == manifest["random_seeds"]
```

Verified by `tests/e2e/test_full_campaign.py::test_manifest_is_reproducible_and_complete`
and `tests/integration/test_pipeline.py::TestReproducibility`.

## Resumption

Campaign state lives in SQLite and is written as it changes, so an interrupted run can
be continued rather than restarted:

```bash
polymer-engine campaign run my-campaign     # interrupt at any point
polymer-engine campaign run my-campaign     # continues from where it stopped
```

Reloading restores the specification, actions, execution records **and the replica set
with its seeds**. That last one was a real bug: without it, saving a reloaded campaign
wrote a manifest with no seeds and destroyed the reproducibility record written at
plan time. `tests/integration/test_pipeline.py::test_reloading_preserves_the_random_seeds`
now guards it.

## The decision log

Every planning round is appended to `decisions`, never updated in place:

```bash
polymer-engine campaign status my-campaign
```

Each entry holds the timestamp, every candidate action with its full assessment, the
selected action, the reason, the tie-break rule, and the cost and information-gain
estimates. **"Why did the engine run this?" is always answerable.**

## Checklist for a publishable campaign

- [ ] `simulation.force_field` and `water_model` set to real values, not `UNSPECIFIED`
- [ ] `local_tools.gromacs.path` pinned to an exact binary
- [ ] `resources.gpu_available` set explicitly rather than detected
- [ ] `base_seed` recorded (it is, in the manifest)
- [ ] at least 3 replicas, and the replica-agreement gate passing
- [ ] convergence gates passing for every reported observable
- [ ] `graph.verify_all()` reports no mismatch
- [ ] manifest archived alongside the results
- [ ] every claim `SUPPORTED` by evidence that meets the bar in
      [SCIENCE.md](SCIENCE.md)
