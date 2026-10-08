#!/usr/bin/env python
"""Classify every replica of a campaign as equilibrated, relaxing, or ordering.

Reads the density and potential-energy series each replica already wrote, so it runs
against a finished or a still-running campaign without touching either.

The question it answers is the one a drift check cannot: a melt that has not finished
relaxing and a melt that is crystallising both show a density that will not settle, and
they need opposite responses. Ordering releases potential energy; relaxation does not.

    .venv/bin/python scripts/diagnose_ordering.py campaign/run4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymer_engine.analysis.ordering import (
    OrderingVerdict,
    analyse,
    determination_for,
    ordering_gates,
)

GMX = "/mnt/data/biodesign/opt/gromacs-2026.3/bin/gmx"


def read_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for line in path.read_text().splitlines():
        if line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
    return np.asarray(xs), np.asarray(ys)


def potential_energy(directory: Path, cache: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract potential energy, caching so a rerun costs nothing."""
    edr = directory / "prod.edr"
    if not edr.exists():
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        try:
            subprocess.run(
                [GMX, "energy", "-f", str(edr), "-o", str(cache)],
                input="Potential\n", text=True, capture_output=True,
                timeout=180, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
    return read_xvg(cache) if cache.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".ordering_cache"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    experiments = args.root / "experiments"
    if not experiments.is_dir():
        print(f"no experiments under {args.root}", file=sys.stderr)
        return 2

    findings: list[dict[str, object]] = []
    print(f"{'candidate':18} {'rep':>4}  {'verdict':14} {'d(rho)':>8} {'d(E)':>8}  "
          f"{'gate':12} more time?")
    for candidate in sorted(p for p in experiments.iterdir() if p.is_dir()):
        for replica in sorted(candidate.glob("replica_*")):
            density_file = replica / "density.xvg"
            if not density_file.exists():
                continue
            times, density = read_xvg(density_file)
            energy = potential_energy(
                replica, args.cache / f"{candidate.name}_{replica.name}.xvg")
            if energy is None:
                series = None
            else:
                n = min(len(density), len(energy[1]))
                times, density, series = times[:n], density[:n], energy[1][:n]

            # The whole production window: a system that ordered early looks stationary
            # if the early part is discarded, and that is the run most needing this.
            result = analyse(times, density, series)
            gate = ordering_gates(result)
            findings.append({
                "candidate": candidate.name, "replica": replica.name,
                "verdict": result.verdict.value, "gate": gate.status.value,
                "determination": determination_for(result).value,
                "more_time_would_help": result.verdict.more_time_would_help,
                "reason": result.reason, **result.as_dict(),
            })
            d = result.density
            e = result.energy
            print(f"{candidate.name[:18]:18} {replica.name[-2:]:>4}  "
                  f"{result.verdict.value:14} "
                  f"{d.relative_change * 100:+7.2f}% " if d else "",
                  end="")
            print(f"{e.relative_change:+7.2f}s " if e else "    n/a ", end="")
            print(f" {gate.status.value:12} {result.verdict.more_time_would_help}")

    counts: dict[str, int] = {}
    for f in findings:
        key = str(f["verdict"])
        counts[key] = counts.get(key, 0) + 1
    print(f"\n{len(findings)} replica(s): " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if counts.get(OrderingVerdict.ORDERING.value):
        print("\nORDERING means the system is leaving the amorphous state. A longer run "
              "does not fix it; the requested state is not the equilibrium one at this "
              "temperature.")
    if args.json:
        args.json.write_text(json.dumps(findings, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
