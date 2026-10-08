#!/usr/bin/env python
"""Re-score a campaign's stored trajectories with the current analysis code.

Exists because the analysis can be wrong while the simulations are fine. The barostat-
ringing defect counted 50,001 correlated frames as independent, understated every
standard error forty-fold, and failed replicas that agreed to a third of a percent on a
chi-square inflated three orders of magnitude. The trajectories on disk were always
good; the verdicts derived from them were not.

This reruns exactly the per-replica and combined analysis the campaign driver runs --
same functions, same thresholds, same gates -- against the density series already
extracted, and writes the corrected scoring beside the original:

    campaign/<root>/reanalysis.json
    campaign/<root>/reanalysis.md

``campaign_state.json`` is left untouched. It is the record of what the campaign
concluded with the code it ran; the reanalysis is the record of what the same data
supports under the fixed code, and keeping both is what makes the correction auditable.

    .venv/bin/python scripts/reanalyse_campaign.py campaign/run4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymer_engine.core.logging import configure_logging, get_logger
from polymer_engine.orchestrator.density_campaign import (
    ReplicaResult,
    analyse_replica_density,
    combine_replica_densities,
    read_density_series,
)

logger = get_logger("reanalyse")


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--min-effective-samples", type=float, default=20.0)
    parser.add_argument("--max-drift-fraction", type=float, default=0.02)
    parser.add_argument("--chi-square-max", type=float, default=4.0)
    parser.add_argument("--required-replicas", type=int, default=3)
    args = parser.parse_args()

    state_file = args.root / "campaign_state.json"
    if not state_file.exists():
        print(f"no campaign state under {args.root}", file=sys.stderr)
        return 2
    original = json.loads(state_file.read_text())

    reanalysis: dict[str, dict] = {}
    for candidate in sorted((args.root / "experiments").iterdir()):
        if not candidate.is_dir():
            continue
        replicas: list[ReplicaResult] = []
        for index, replica_dir in enumerate(sorted(candidate.glob("replica_*")), start=1):
            xvg = replica_dir / "density.xvg"
            if not xvg.exists():
                continue
            _times, densities = read_density_series(xvg)
            stats = analyse_replica_density(
                densities,
                min_effective_samples=args.min_effective_samples,
                max_drift_fraction=args.max_drift_fraction,
            )
            record = ReplicaResult(replica=index, seed=0, directory=str(replica_dir))
            record.n_frames = int(stats.get("n_frames", 0))
            record.effective_samples = stats.get("effective_samples")
            record.statistical_inefficiency = stats.get("statistical_inefficiency")
            record.density_kg_m3 = stats.get("mean")
            record.density_stderr = stats.get("stderr")
            record.succeeded = bool(stats.get("usable"))
            if not record.succeeded:
                record.error = stats.get("reason")
            replicas.append(record)
        if not replicas:
            continue
        combined = combine_replica_densities(
            replicas, chi_square_max=args.chi_square_max,
            required=args.required_replicas,
        )
        before = original.get("results", {}).get(candidate.name, {})
        reanalysis[candidate.name] = {
            "combined": combined,
            "replicas": [r.as_dict() for r in replicas],
            "original": {
                "gate_status": before.get("gate_status"),
                "density_kg_m3": before.get("density_kg_m3"),
                "diagnostics": (before.get("diagnostics") or [])[:2],
            },
        }

    payload = {
        "schema": "campaign.reanalysis/1",
        "generated_at": datetime.now(UTC).isoformat(),
        "root": str(args.root),
        "note": ("re-scored with the current analysis code; campaign_state.json is the "
                 "original verdict and is deliberately untouched"),
        "thresholds": {
            "min_effective_samples": args.min_effective_samples,
            "max_drift_fraction": args.max_drift_fraction,
            "chi_square_max": args.chi_square_max,
            "required_replicas": args.required_replicas,
        },
        "candidates": reanalysis,
    }
    (args.root / "reanalysis.json").write_text(json.dumps(payload, indent=2,
                                                          default=str) + "\n")

    lines = ["# Campaign re-analysis", "",
             f"Generated {payload['generated_at']} · same gates, current estimator", "",
             "| Candidate | Was | Now | Density | chi2_red | Replica n_eff |",
             "|---|---|---|---|---|---|"]
    for name, entry in reanalysis.items():
        combined = entry["combined"]
        neffs = "/".join(f"{r['effective_samples']:.0f}"
                         for r in entry["replicas"] if r["effective_samples"])
        rho = combined.get("density_kg_m3")
        unc = combined.get("uncertainty_kg_m3") or combined.get("stderr_kg_m3")
        chi = combined.get("reduced_chi_square")
        lines.append(
            f"| {name} | {entry['original']['gate_status']} | "
            f"{combined.get('status')} | "
            + (f"{rho:.1f} ± {unc:.1f}" if rho and unc else
               f"{rho:.1f}" if rho else "—")
            + f" | {chi:.2f}" if chi is not None else " | —")
        lines[-1] += f" | {neffs} |"
        print(f"  {name:16} {entry['original']['gate_status']:12} -> "
              f"{combined.get('status'):12} "
              + (f"rho={rho:.1f}" if rho else "") 
              + (f" chi2={chi:.2f}" if chi is not None else ""))
    (args.root / "reanalysis.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.root}/reanalysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
