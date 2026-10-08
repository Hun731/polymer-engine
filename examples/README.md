# Examples

## `polyethylene_density/` — a complete campaign

Fifteen common polymers, from raw CSV through curation, modelling, design and a
gate-checked simulation campaign.

```bash
cd examples/polyethylene_density
./run.sh
```

Steps 1–10 need no external software and no network. Step 11 needs GROMACS and is
skipped automatically when `gmx` is absent.

---

### What each step demonstrates

#### 3. Curation with unit normalisation

The CSV writes Tg in °C, density in g/cm³, modulus in GPa and crystallinity in %. All
four are converted on the way in, and the row counts must reconcile:

```text
rows_read = 15   accepted = 15   rejected = 0   duplicates = 0   reconciles = true
```

`reconciles` is the point. A curation step that silently drops rows makes "the dataset
has 15 polymers" and "the file had 15 rows" two different claims. Here they are the
same claim, and the engine checks it.

The source file's SHA-256 is recorded on every record, so a result can be traced back
to the exact bytes it came from.

#### 4. Family classification

```text
polyacrylate      3    polyester         3    polyolefin        2
polyvinyl_halide  2    polyamide         1    polyether         1
polynitrile       1    polystyrenic      1    polyvinyl_alcohol 1
```

Worth noticing: **poly(ethylene terephthalate) is a polyester and poly(methyl
methacrylate) is a polyacrylate**, though both contain `C(=O)O`. The classifier builds
a four-unit oligomer and distinguishes an ester *in* the chain from one hanging *off*
it. Classifying a hydrogen-capped monomer instead — the obvious implementation — gets
this wrong, and gets nylon-6 wrong too, because capping destroys the amide bond
entirely.

#### 6. A model that honestly reports it cannot learn

```text
n = 15   folds = 5   r2 = -0.195   rmse = 82.3 K   spearman = 0.002
```

**A negative R² is the correct answer here**, and it is the most instructive line in
this example. Fifteen polymers, fifteen descriptors and grouped 5-fold
cross-validation cannot predict Tg — the model is worse than predicting the mean.

Getting a flattering number instead would take only: fitting the scaler on all the
data before splitting, or splitting by row rather than by canonical polymer identity,
or reporting training-set performance. The engine closes all three routes
structurally, so the score it reports is the score the data supports.

Feed it a few hundred polymers and the number becomes useful. That is a data problem,
and the engine says so rather than papering over it.

#### 7. Rational candidate generation

From polyethylene (`*CC*`):

```text
valid    CC(*)C*         add methyl          -> polypropylene
valid    OC(*)C*         add hydroxyl        -> poly(vinyl alcohol)
valid    N#CC(*)C*       add nitrile         -> polyacrylonitrile
valid    *CCC*           insert one methylene
valid    *CO*            backbone carbon -> ether oxygen
```

Mutations operate on the molecular graph, never on the SMILES string, so every
candidate is a real molecule with exactly two attachment points. Candidates whose
chemistry warrants human judgement — strained rings, formal charges, Si or P where
force-field coverage is patchy — come back as `REQUIRES_REVIEW` rather than being
accepted or dropped.

#### 9. A dry run that cannot pretend to be a result

```json
{
  "kind": "gromacs_equilibrate",
  "status": "skipped",
  "execution_mode": "dry_run",
  "scientifically_usable": false,
  "state": "CANCELLED"
}
```

Every input was generated and validated; nothing ran. The status is `skipped`, not
`succeeded`, the execution record is `CANCELLED`, not `COMPLETED`, and the downstream
analysis action does **not** run, because a dry run does not satisfy a dependency.

#### 10. The audit trail

`evidence manifest` emits everything needed to reconstruct the campaign: the full
parameter set, the force field, tool versions, the system content digest, per-replica
seeds, and every gate result with its value and threshold.

`campaign status` shows the decision log — for each planning round, every candidate
action with its assessment, the one selected, and why:

```text
highest-scoring of 2 feasible action(s) out of 3 considered;
expected information gain 0.60; estimated cost 28.00; score 1.775
```

"Why did the engine run this?" is always answerable.

---

### Running it for real

With GROMACS installed:

```bash
polymer-engine campaign run pe-density --execute
polymer-engine campaign analyze pe-density
```

The fixture system is a 68-atom toy, so **the gates will fail** — and that is the
demonstration. A real run of this system produces output like:

```text
fail          density:effective_samples   Only 2.9 effective samples after accounting
                                          for autocorrelation (766 raw frames); need 20
fail          density:drift               Observable is still drifting: half-to-half
                                          change 2.894% exceeds both the 2.000%
                                          threshold and the statistical noise
inconclusive  density:replica_agreement   2 replica(s) available but 3 are required;
                                          reproducibility is not demonstrated
```

GROMACS exited 0 on every stage. 766 frames were written. The engine still refuses to
call it a result, because 766 correlated frames are 2.9 independent measurements and
two replicas cannot demonstrate reproducibility.

To make those gates pass you need a real system, nanoseconds rather than picoseconds,
and at least three replicas. The gates are not obstacles to work around; they are the
difference between a number and a measurement.

---

### Using your own data

Ingestion accepts CSV or JSONL. Columns are matched case-insensitively.

| Column | Accepted spellings |
|---|---|
| structure | `repeat_unit_smiles`, `repeat_unit`, `smiles`, `psmiles`, `monomer_smiles` |
| name | `name`, `polymer_name`, `polymer`, `label` |
| properties | `Tg (degC)`, `density [g/cm3]`, `Young's Modulus (GPa)`, … |

Write the unit in the header, in brackets or parentheses. An unrecognised unit is
**not** assumed: the value comes back as `REQUIRES_VALIDATION` with the reason
attached, because guessing between K and °C is a 273-degree error that nothing
downstream would catch.

SMILES must be repeat units with attachment points:

```text
*CC*                  polyethylene
*CC(*)c1ccccc1        polystyrene
*NCCCCCC(=O)*         nylon-6
```

A discrete molecule (`CCO`) is rejected with an explanation rather than silently
treated as a polymer.
