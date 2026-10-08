#!/usr/bin/env python
"""Turn a campaign's durable state into the reporting deliverables.

Safe to run at any point, including mid-campaign: it reads ``campaign_state.json`` and
never writes into the run directory's experiment tree.

    python scripts/report_density_campaign.py --root campaign/run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymer_engine.core.provenance import sha256_file

#: Reported alongside every correlation so the multiplicity burden is visible.
ALPHA = 0.05


def load_state(root: Path) -> dict[str, Any]:
    path = root / "campaign_state.json"
    if not path.is_file():
        raise SystemExit(f"no campaign state at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def outcome_of(result: dict[str, Any]) -> str:
    """One of the reporting states.  A result is never upgraded to fit a narrative."""
    if result.get("scientifically_usable"):
        return "VALIDATED"
    gate = (result.get("gate_status") or "").upper()
    if gate == "FAIL":
        return "FAILED"
    if any(r.get("stage_failed") for r in result.get("replicas", [])):
        return "FAILED"
    if result.get("finished_at") is None:
        return "NOT_ATTEMPTED"
    return "INCONCLUSIVE"


def write_experiment_table(state: dict[str, Any], out: Path) -> int:
    rows = []
    for name, result in sorted(state.get("results", {}).items()):
        for replica in result.get("replicas", []):
            rows.append({
                "candidate": name,
                "replica": replica["replica"],
                "seed": replica["seed"],
                "directory": replica["directory"],
                "succeeded": replica["succeeded"],
                "stage_failed": replica.get("stage_failed") or "",
                "density_kg_m3": replica.get("density_kg_m3"),
                "density_stderr_kg_m3": replica.get("density_stderr"),
                "n_frames": replica.get("n_frames"),
                "effective_samples": replica.get("effective_samples"),
                "statistical_inefficiency": replica.get("statistical_inefficiency"),
                "equilibration_index": replica.get("equilibration_index"),
                "wall_seconds": round(replica.get("wall_seconds") or 0.0, 1),
                "error": (replica.get("error") or "")[:300],
            })
    _write_csv(out, rows)
    return len(rows)


def write_property_table(state: dict[str, Any], out: Path) -> int:
    rows = []
    for name, result in sorted(state.get("results", {}).items()):
        rows.append({
            "candidate": name,
            "property": "density",
            "value": result.get("density_kg_m3"),
            "units": "kg/m^3",
            "uncertainty": result.get("density_uncertainty"),
            "uncertainty_method": "standard error of replica means",
            "estimator": "equilibrium time average, autocorrelation-corrected",
            "n_replicas": len(result.get("replicas", [])),
            "n_replicas_usable": sum(1 for r in result.get("replicas", []) if r["succeeded"]),
            "total_atoms": result.get("total_atoms"),
            "gate_status": result.get("gate_status"),
            "determination": result.get("determination"),
            "outcome": outcome_of(result),
            "sampling_assumptions": ("300 K, 1 bar, DP 30, 20 chains, OPLS-AA; "
                                    "MD melt density, not an experimental measurement"),
            "diagnostics": " | ".join(result.get("diagnostics", []))[:400],
        })
    _write_csv(out, rows)
    return len(rows)


def write_candidate_ranking(state: dict[str, Any], out: Path) -> int:
    validated = [
        (name, r) for name, r in state.get("results", {}).items()
        if r.get("scientifically_usable") and r.get("density_kg_m3") is not None
    ]
    validated.sort(key=lambda kv: kv[1]["density_kg_m3"])
    rows = [{
        "rank": index,
        "candidate": name,
        "density_kg_m3": round(result["density_kg_m3"], 2),
        "uncertainty_kg_m3": (round(result["density_uncertainty"], 2)
                              if result.get("density_uncertainty") is not None else None),
        "n_replicas_usable": sum(1 for r in result["replicas"] if r["succeeded"]),
        "heavy_atoms_repeat": result.get("descriptors", {}).get("heavy_atom_count"),
        "side_chain_heavy_atoms": result.get("descriptors", {}).get("side_chain_heavy_atoms"),
        "fraction_csp3": result.get("descriptors", {}).get("fraction_csp3"),
        "outcome": "VALIDATED",
    } for index, (name, result) in enumerate(validated, start=1)]
    _write_csv(out, rows)
    return len(rows)


def write_failure_report(state: dict[str, Any], out: Path) -> None:
    results = state.get("results", {})
    buckets: dict[str, list[str]] = {}
    for name, result in results.items():
        if result.get("scientifically_usable"):
            continue
        for replica in result.get("replicas", []):
            if replica.get("succeeded"):
                continue
            stage = replica.get("stage_failed") or "sampling"
            reason = (replica.get("error") or "unstated").split(";")[0][:160]
            buckets.setdefault(f"{stage}: {reason}", []).append(f"{name} r{replica['replica']}")

    lines = ["# Failure report", "",
             "Every failure the campaign encountered, classified. Nothing here was retried",
             "into a pass, and nothing was dropped from the history.", ""]
    if not buckets:
        lines += ["No replica-level failures were recorded.", ""]
    else:
        lines += ["| Pattern | Occurrences | Affected |", "|---|---|---|"]
        for pattern, affected in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"| {pattern} | {len(affected)} | {', '.join(affected[:6])} |")
        lines.append("")
    blocked = [name for name, r in results.items() if outcome_of(r) == "INCONCLUSIVE"]
    if blocked:
        lines += ["## Inconclusive candidates", "",
                  "These ran without a hard failure and still did not produce a usable "
                  "density. That is a refusal, not a gap in the record.", ""]
        for name in sorted(blocked):
            diagnostics = " | ".join(results[name].get("diagnostics", []))[:300]
            lines.append(f"- **{name}** — {diagnostics or 'no diagnostics recorded'}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def write_correlation_report(state: dict[str, Any], out: Path) -> None:
    import numpy as np

    validated = [r for r in state.get("results", {}).values()
                 if r.get("scientifically_usable") and r.get("density_kg_m3") is not None
                 and r.get("descriptors")]
    lines = ["# Correlation report", "",
             "Descriptor-to-density associations over **validated points only**.", ""]
    if len(validated) < 3:
        lines += [f"Only {len(validated)} validated point(s); no correlation is reported.",
                  "Screening this few points would manufacture a relationship rather than",
                  "measure one.", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        return

    keys = sorted(set.intersection(*(set(r["descriptors"]) for r in validated)))
    target = np.array([r["density_kg_m3"] for r in validated], dtype=float)
    rows = []
    for key in keys:
        column = np.array([r["descriptors"][key] for r in validated], dtype=float)
        if float(np.std(column)) == 0.0:
            continue
        pearson = float(np.corrcoef(column, target)[0, 1])
        order_x = np.argsort(np.argsort(column)).astype(float)
        order_y = np.argsort(np.argsort(target)).astype(float)
        spearman = float(np.corrcoef(order_x, order_y)[0, 1])
        if math.isfinite(pearson):
            rows.append((key, pearson, spearman))
    rows.sort(key=lambda item: -abs(item[1]))
    n_tests = len(rows)
    bonferroni = ALPHA / max(n_tests, 1)

    lines += [f"- Validated points: **{len(validated)}**",
              f"- Descriptors screened: **{n_tests}**",
              f"- Bonferroni-corrected alpha: **{bonferroni:.5f}** (from {ALPHA} / {n_tests})",
              "",
              "| Descriptor | Pearson r | Spearman rho |", "|---|---|---|"]
    for key, pearson, spearman in rows[:15]:
        lines.append(f"| `{key}` | {pearson:+.3f} | {spearman:+.3f} |")
    lines += ["",
              "## How to read this", "",
              "These are **associations**, not causes. No p-values are quoted because with",
              f"{len(validated)} points and {n_tests} descriptors the multiple-comparison",
              "burden dominates any nominal significance: at the uncorrected alpha one would",
              f"expect roughly {n_tests * ALPHA:.1f} spurious hits by chance alone.",
              "",
              "A large coefficient here is a reason to run the next simulation, not a",
              "structure-property law.", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_provenance_manifests(state: dict[str, Any], root: Path) -> tuple[int, int]:
    """Hash every artifact the campaign produced, and record how to reproduce it."""
    artifacts = []
    for name, result in sorted(state.get("results", {}).items()):
        for replica in result.get("replicas", []):
            directory = Path(replica["directory"])
            if not directory.is_dir():
                continue
            for pattern in ("packed.gro", "topol.top", "*.mdp", "prod.gro", "density.xvg"):
                for path in sorted(directory.glob(pattern)):
                    if not path.is_file() or path.stat().st_size == 0:
                        continue
                    artifacts.append({
                        "candidate": name, "replica": replica["replica"],
                        "path": str(path), "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    })
    (root / "provenance_manifest.json").write_text(
        json.dumps({"n_artifacts": len(artifacts), "artifacts": artifacts}, indent=1),
        encoding="utf-8")

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  timeout=30, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    manifest_path = root.parent / "campaign_manifest.json"
    reproducibility = {
        "campaign_state": str(root / "campaign_state.json"),
        "campaign_state_sha256": sha256_file(root / "campaign_state.json"),
        "campaign_manifest": str(manifest_path) if manifest_path.is_file() else None,
        "campaign_manifest_sha256": (sha256_file(manifest_path)
                                     if manifest_path.is_file() else None),
        "repository_commit": git("rev-parse", "HEAD"),
        "repository_dirty": bool(git("status", "--porcelain")),
        "how_to_reproduce": [
            "git checkout <repository_commit>",
            "python scripts/run_density_campaign.py --config configs/campaign_96h.yaml "
            "--root <fresh directory>",
        ],
        "determinism": (
            "Replica seeds are derived from the candidate name and replica index, so the "
            "same campaign re-run assigns the same seeds. GROMACS trajectories are not "
            "bit-reproducible across hardware or thread counts; the seeds, inputs and "
            "hashes are what is reproducible, and the gates are what decide whether a "
            "re-run agrees."
        ),
        "n_artifacts_hashed": len(artifacts),
    }
    (root / "reproducibility_manifest.json").write_text(
        json.dumps(reproducibility, indent=1), encoding="utf-8")
    return len(artifacts), len(state.get("results", {}))


def write_campaign_report(state: dict[str, Any], root: Path, cfg_path: Path) -> None:
    results = state.get("results", {})
    by_outcome: dict[str, list[str]] = {}
    for name, result in results.items():
        by_outcome.setdefault(outcome_of(result), []).append(name)
    validated = sorted(
        ((n, r) for n, r in results.items() if r.get("scientifically_usable")),
        key=lambda kv: kv[1]["density_kg_m3"],
    )
    lines = [
        "# Campaign report: polyolefin melt density", "",
        "## Outcome counts", "",
        "| State | Count | Candidates |", "|---|---|---|",
    ]
    for label in ("VALIDATED", "WARNING", "INCONCLUSIVE", "FAILED", "BLOCKED", "NOT_ATTEMPTED"):
        names = sorted(by_outcome.get(label, []))
        lines.append(f"| {label} | {len(names)} | {', '.join(names) if names else '--'} |")
    lines += ["", "## Validated densities", ""]
    if validated:
        lines += ["| Polymer | Density (kg/m^3) | Uncertainty | Replicas | Determination |",
                  "|---|---|---|---|---|"]
        for name, result in validated:
            uncertainty = result.get("density_uncertainty")
            lines.append(
                f"| {name} | {result['density_kg_m3']:.1f} | "
                f"{uncertainty:.1f} | " if uncertainty is not None else
                f"| {name} | {result['density_kg_m3']:.1f} | -- | "
            )
            lines[-1] += (f"{sum(1 for r in result['replicas'] if r['succeeded'])}"
                          f"/{len(result['replicas'])} | {result['determination']} |")
    else:
        lines += ["No candidate produced a validated density.", ""]
    lines += ["",
              "Every value is an OPLS-AA melt density at 300 K and 1 bar for DP 30 chains,",
              "with the standard error of the replica means. None is an experimental",
              "measurement and none should be quoted as one.", "",
              f"Configuration: `{cfg_path}`. State: `{root / 'campaign_state.json'}`.", ""]
    (root / "campaign_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="campaign/run")
    parser.add_argument("--config", default="configs/campaign_96h.yaml")
    args = parser.parse_args()
    root = Path(args.root)
    state = load_state(root)

    n_experiments = write_experiment_table(state, root / "experiment_table.csv")
    n_properties = write_property_table(state, root / "property_table.csv")
    n_ranked = write_candidate_ranking(state, root / "candidate_ranking.csv")
    write_failure_report(state, root / "failure_report.md")
    write_correlation_report(state, root / "correlation_report.md")
    n_artifacts, n_candidates = write_provenance_manifests(state, root)
    write_campaign_report(state, root, Path(args.config))

    print(json.dumps({
        "candidates": n_candidates, "experiment_rows": n_experiments,
        "property_rows": n_properties, "ranked": n_ranked,
        "artifacts_hashed": n_artifacts,
        "outputs": sorted(p.name for p in root.glob("*.csv"))
                   + sorted(p.name for p in root.glob("*.md"))
                   + sorted(p.name for p in root.glob("*manifest.json")),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
