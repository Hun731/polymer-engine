#!/usr/bin/env bash
# A complete campaign, from raw data to an auditable manifest.
#
# Steps 1-6 need no external software and no network.
# Step 7 needs GROMACS; it is skipped automatically when gmx is absent.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POLYMER_ROOT="${POLYMER_ROOT:-$HERE/workspace}"
PE="${PE:-polymer-engine}"

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

step "1. Initialise the workspace"
$PE engine init --title "Polyolefin density study"

step "2. Inspect the environment"
$PE engine resources | head -20
$PE engine discover-tools || echo "(GROMACS is unavailable; execution steps will be skipped)"

step "3. Curate the dataset"
# Units are normalised on the way in: degC -> K, g/cm3 -> kg/m^3, GPa -> MPa, % -> fraction.
# The row counts must reconcile: read == accepted + rejected + duplicates.
$PE polymer ingest "$HERE/polymers.csv" --output "$POLYMER_ROOT/records.jsonl"

step "4. Group by polymer family"
$PE polymer cluster "$HERE/polymers.csv"

step "5. Look at one polymer in detail"
$PE polymer descriptors '*CC(*)c1ccccc1'

step "6. Fit a structure-property model"
# Grouped cross-validation: folds are assigned by canonical polymer identity, and
# imputation and scaling are fitted inside each fold.
$PE model evaluate "$HERE/polymers.csv" --target "glass transition temperature" || \
    echo "(scikit-learn is unavailable; modelling skipped)"

step "7. Generate and rank design candidates"
$PE design generate '*CC*' --max 8

step "8. Build a campaign against a validated system"
SYSTEM="$HERE/../../tests/fixtures/systems/valid_system"
$PE system validate "$SYSTEM"
$PE campaign create pe-density \
    --polymer pol_7cf1ea13aa5bbc1b \
    --system "$SYSTEM" \
    --question "What is the equilibrium density of polyethylene at 300 K?"
$PE campaign plan pe-density

step "9. Dry run: validate every input, execute nothing"
$PE campaign run pe-density

step "10. Audit"
$PE campaign status pe-density
$PE evidence manifest pe-density --output "$POLYMER_ROOT/manifest.json"
$PE evidence strategies

if command -v gmx >/dev/null 2>&1; then
    step "11. Real execution (GROMACS found)"
    echo "To run for real:  $PE campaign run pe-density --execute"
    echo "Then:             $PE campaign analyze pe-density"
    echo
    echo "Note: the fixture system is a 68-atom toy. Its gates will correctly FAIL"
    echo "      on sampling grounds -- that is the point of the demonstration."
else
    step "11. Real execution skipped"
    echo "GROMACS is not on PATH, so nothing was executed. Every input above was"
    echo "still generated and validated."
fi

printf '\n\033[1mDone.\033[0m Workspace: %s\n' "$POLYMER_ROOT"
