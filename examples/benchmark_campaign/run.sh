#!/usr/bin/env bash
# The benchmark campaign: the whole loop, end to end, on a system small enough to run
# in seconds.
#
# This demonstrates the MACHINERY. It does not establish a scientific result, and the
# convergence gates will correctly refuse the MD output because 20 ps across 2 replicas
# cannot support a density. That refusal is the most important line of output here.
#
#   ./run.sh            # no external software required
#   ./run.sh --execute  # runs real ORCA and GROMACS where installed
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POLYMER_ROOT="${POLYMER_ROOT:-$HERE/workspace}"
export POLYMER_CONFIG="${POLYMER_CONFIG:-$HERE/config.yaml}"
PE="${PE:-polymer-engine}"
EXECUTE=""
[[ "${1:-}" == "--execute" ]] && EXECUTE="--execute"

SYSTEM="$HERE/../../tests/fixtures/systems/valid_system"

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }
note() { printf '   \033[2m%s\033[0m\n' "$1"; }

step "1. Environment"
$PE engine init --title "Benchmark: descriptors and bulk density" >/dev/null
$PE engine discover-tools || note "some tools are unavailable; those steps will be skipped"

step "2. Curate candidates"
note "Units normalised on ingest: degC -> K, g/cm3 -> kg/m^3, GPa -> MPa."
note "Row counts must reconcile: read == accepted + rejected + duplicates."
$PE polymer ingest "$HERE/candidates.csv" --output "$POLYMER_ROOT/records.jsonl"
$PE polymer cluster "$HERE/candidates.csv"

step "3. Force-field decision"
note "The engine records the choice and how much confidence it deserves."
note "It will NOT pick between force fields that all claim coverage."
$PE polymer descriptors '*CC*'

step "4. QM reference: the ethane torsional barrier"
if command -v orca >/dev/null 2>&1 && [[ -n "$EXECUTE" ]]; then
    cat > "$POLYMER_ROOT/ethane.xyz" <<'XYZ'
8
ethane
C  -0.765000   0.000000   0.000000
C   0.765000   0.000000   0.000000
H  -1.140000   1.018000   0.000000
H  -1.140000  -0.509000   0.881500
H  -1.140000  -0.509000  -0.881500
H   1.140000  -1.018000   0.000000
H   1.140000   0.509000   0.881500
H   1.140000   0.509000  -0.881500
XYZ
    $PE qm run "$POLYMER_ROOT/ethane.xyz" --kind torsion_scan --method HF --basis STO-3G \
        --torsion '2,0,1,6' --scan-range '0,120,5' --workdir "$POLYMER_ROOT/qm" --execute
    note "The experimental ethane rotational barrier is about 12.1 kJ/mol."
    note "This checks input generation, execution and parsing -- it is not a result."
else
    note "ORCA unavailable or --execute not given; QM step skipped."
fi

step "5. Build and validate the system"
$PE system validate "$SYSTEM" || note "the system failed validation (exit 3)"

step "6. Plan the campaign"
$PE campaign create benchmark --polymer pol_7cf1ea13aa5bbc1b --system "$SYSTEM" \
    --question "What is the equilibrium density at 300 K?"
$PE campaign plan benchmark

step "7. Run"
if [[ -n "$EXECUTE" ]] && command -v gmx >/dev/null 2>&1; then
    note "Running real GROMACS. Every stage will exit 0."
    # Exit 3 means a validation gate refused the result. That is the expected outcome
    # here and is the point of the benchmark, so it must not abort the script.
    $PE campaign run benchmark --execute || note "gates refused the result (exit 3), as expected"
    step "8. Analyse -- expect the gates to REFUSE this"
    note "20 ps across 2 replicas is not a density. The gates should say so."
    $PE campaign analyze benchmark || note "gates refused the result (exit 3), as expected"
else
    note "Dry run: inputs are generated and validated, nothing is executed."
    $PE campaign run benchmark || note "nothing was executed, so no result was promoted"
    note "A dry run is SKIPPED, never SUCCEEDED, and does not satisfy a dependency."
fi

step "9. Structure-property screening"
note "Multiple comparisons are corrected: 15 descriptors x 1 property is 15 tests."
$PE research correlate "$HERE/candidates.csv" --target density || \
    note "screening unavailable (scikit-learn or RDKit missing)"

step "10. Surrogate model"
note "Grouped cross-validation by polymer identity AND structural similarity."
$PE model evaluate "$HERE/candidates.csv" --target density || \
    note "modelling unavailable, or the data cannot support a model -- both are honest outcomes"

step "11. Design candidates"
$PE design generate '*CC*' --max 8

step "12. Audit"
$PE evidence manifest benchmark --output "$POLYMER_ROOT/manifest.json"
$PE research decisions --campaign benchmark --limit 5
$PE research failures
$PE evidence strategies

printf '\n\033[1mDone.\033[0m Workspace: %s\n' "$POLYMER_ROOT"
printf 'The manifest records every parameter, seed and software version used.\n'
