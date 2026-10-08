#!/usr/bin/env python
"""Keep ``campaign_status.json`` / ``.md`` current while the campaign runs.

**This is an observer.** It reads the campaign's checkpoint, the GROMACS logs of the
stage currently in flight, and the machine's resources. It writes only the two status
files and never touches state, experiments, configuration or the engine.

It exists because the driver writes status at candidate boundaries, and a candidate now
takes about two and a quarter hours. Between boundaries the status file would say
"running polyethylene" and nothing else, which is not enough to tell a healthy campaign
from a stalled one.

Both this and the driver write the status files through a temp-file replace, so a
concurrent write can never leave a truncated file; the loser of a race is simply
overwritten a few seconds later.

    python scripts/monitor_campaign.py --root campaign/run --interval 60
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Imported, never written out. A hard-coded copy here is why an in-flight anneal was
# invisible: the monitor skipped the stage it did not know about and reported the newest
# log it did recognise -- an interrupted npt from a different candidate -- as current.
# That is worse than reporting nothing, because it looks like an answer.
from polymer_engine.simulation.mdp import STAGE_ORDER as STAGES

_STEP = re.compile(r"^\s+(\d+)\s+([\d.]+)\s*$", re.MULTILINE)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def gpu() -> dict[str, Any]:
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout.strip().splitlines()
        if not raw:
            return {}
        used, total = (float(v) for v in raw[0].split(",")[1:3])
        return {"utilisation_percent": float(raw[0].split(",")[0]),
                "memory_used_mb": used, "memory_total_mb": total,
                "memory_percent": round(100.0 * used / total, 1) if total else None}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def disk(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {"free_gb": round(usage.free / 1024**3, 1),
            "used_percent": round(100.0 * usage.used / usage.total, 1)}


def driver_alive() -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", "scripts/run_density_campaign.py"],
                             capture_output=True, text=True, timeout=10, check=False)
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


#: A stage log untouched for longer than this is treated as not running. GROMACS writes
#: progress every few seconds at these settings, so a minute is generous.
STALE_LOG_SECONDS = 90.0


def in_flight(root: Path) -> dict[str, Any]:
    """Which candidate, replica and stage are running, and how far through.

    Determined from file modification times and the GROMACS log the stage is writing,
    so it reflects what is actually happening rather than what was last checkpointed.
    """
    experiments = root / "experiments"
    if not experiments.is_dir():
        return {}
    logs = []
    for candidate in experiments.iterdir():
        if not candidate.is_dir():
            continue
        for replica in candidate.iterdir():
            if not replica.is_dir():
                continue
            for stage in STAGES:
                log = replica / f"{stage}.log"
                # The most recently written stage log wins, and whether that stage has
                # *finished* is decided by comparing its output to its log rather than
                # by the output merely existing.
                #
                # Presence alone was the bug. On a rerun a .gro survives from the
                # previous attempt, so the stage actually running was excluded while a
                # log abandoned fourteen hours earlier stayed on screen as current. A
                # .gro older than its own log is a leftover; one newer than its log
                # means the stage really did just finish.
                if log.is_file():
                    logs.append((log.stat().st_mtime, candidate.name, replica.name, stage, log))
    if not logs:
        return {}
    mtime, candidate, replica, stage, log = max(logs)
    age = time.time() - mtime
    output = log.with_suffix(".gro")
    finished = output.is_file() and output.stat().st_mtime >= mtime
    info: dict[str, Any] = {
        "candidate": candidate, "replica": replica, "stage": stage,
        "log_age_seconds": round(age, 1),
        # A log nobody has written to for minutes is not a running stage, and neither
        # is one whose output has already been written. Reporting either as running is
        # how a stopped campaign reads as busy.
        "live": age <= STALE_LOG_SECONDS and not finished,
        "finished": finished,
    }
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return info
    steps = _STEP.findall(text[-20000:])
    if steps:
        info["current_step"] = int(steps[-1][0])
        info["current_time_ps"] = float(steps[-1][1])
    total = re.search(r"nsteps\s*=\s*(\d+)", text)
    if total and steps:
        n = int(total.group(1))
        info["total_steps"] = n
        info["percent_complete"] = round(100.0 * int(steps[-1][0]) / n, 1) if n else None
    performance = re.search(r"Performance:\s+([\d.]+)\s+([\d.]+)", text)
    if performance:
        info["ns_per_day"] = float(performance.group(1))
    return info


def build_status(root: Path, cfg: dict[str, Any], budget: float | None) -> dict[str, Any]:
    state = json.loads((root / "campaign_state.json").read_text(encoding="utf-8"))
    elapsed = (time.time() - state["started_epoch"]) / 3600.0
    results = state.get("results", {})
    validated = [r for r in results.values() if r.get("scientifically_usable")]

    ranking = []
    for name, result in results.items():
        usable = [r for r in result.get("replicas", []) if r.get("succeeded")]
        ranking.append({
            "candidate": name,
            "density_kg_m3": result.get("density_kg_m3"),
            "uncertainty_kg_m3": result.get("density_uncertainty"),
            "gate": result.get("gate_status"),
            "usable": result.get("scientifically_usable"),
            "replicas_usable": len(usable),
            "replicas_run": len(result.get("replicas", [])),
            "replica_densities": [r.get("density_kg_m3") for r in result.get("replicas", [])],
            "replica_n_eff": [r.get("effective_samples") for r in result.get("replicas", [])],
            "replica_g": [r.get("statistical_inefficiency") for r in result.get("replicas", [])],
            "diagnostics": result.get("diagnostics", []),
        })
    ranking.sort(key=lambda row: (row["density_kg_m3"] is None, row["density_kg_m3"] or 0.0))

    model_path = root / "model_report.json"
    model = (json.loads(model_path.read_text(encoding="utf-8"))
             if model_path.is_file() else {"status": "not yet trained"})

    return {
        "campaign_id": cfg["campaign"]["id"],
        "question": cfg["campaign"]["question"],
        "protocol": {
            "production_ns": cfg["simulation"]["production_ns"],
            "replicas": cfg["simulation"]["replicas"],
            "temperature_k": cfg["simulation"]["temperature_k"],
            "pressure_bar": cfg["simulation"]["pressure_bar"],
            "degree_of_polymerization": cfg["simulation"]["degree_of_polymerization"],
            "chains_per_system": cfg["simulation"]["chains_per_system"],
            "energy_output_ps": cfg["simulation"]["energy_output_ps"],
            "force_field": cfg["force_field"]["name"],
            "min_effective_samples": cfg["validation"]["min_effective_samples"],
            "analysis": "Geyer initial positive sequence (post BUG-002)",
        },
        "start_time": state["started_at"],
        "current_time": utc_now(),
        "elapsed_hours": round(elapsed, 3),
        # An open-ended campaign has no remaining time, which is different from having
        # none left. Both are reported as null rather than as zero.
        "remaining_hours": (None if budget is None
                            else round(max(0.0, budget - elapsed), 3)),
        "budget_hours": budget,
        "open_ended": budget is None,
        "driver_running": driver_alive(),
        "iteration": state.get("iteration", 0),
        "decisions_recorded": state.get("decisions", 0),
        "attempts": state.get("attempts", {}),
        "queued": [c["name"] for c in state.get("queue", [])],
        "in_flight": in_flight(root),
        "completed_jobs": len(state.get("completed", [])),
        "failed_jobs": len(state.get("failed", [])),
        "validated_jobs": len(validated),
        "candidates_evaluated": len(results),
        "candidate_ranking": ranking,
        "model": model,
        "umbrella_enabled": cfg["umbrella"]["enabled"],
        "umbrella_studies_launched": 0,
        "mechanics_enabled": cfg["mechanics"]["enabled"],
        "gpu": gpu(),
        "disk": disk(root),
        "disk_reserve_gb": cfg["resources"]["disk_reserve_gb"],
        "campaign_bytes": sum(f.stat().st_size for f in root.rglob("*") if f.is_file()),
        "stopped_reason": state.get("stopped_reason"),
        "monitor": "scripts/monitor_campaign.py (read-only observer)",
    }


def render_markdown(status: dict[str, Any]) -> str:
    flight = status["in_flight"]
    if flight:
        progress = (f" — step {flight.get('current_step', '?')}"
                    f"/{flight.get('total_steps', '?')}"
                    f" ({flight.get('percent_complete', '?')}%)")
        running = (f"`{flight['candidate']}` / {flight['replica']} / **{flight['stage']}**"
                   f"{progress}")
    else:
        running = "nothing in flight"

    lines = [
        f"# {status['campaign_id']}", "",
        f"**Question.** {status['question']}", "",
        f"**Now running.** {running}", "", "| | |", "|---|---|",
        f"| Driver | {'running' if status['driver_running'] else '**not running**'} |",
        f"| Started | {status['start_time']} |",
        f"| Updated | {status['current_time']} |",
        (f"| Elapsed | {status['elapsed_hours']:.2f} h (open-ended) |"
         if status["budget_hours"] is None
         else f"| Elapsed | {status['elapsed_hours']:.2f} h of "
              f"{status['budget_hours']:g} |"),
        (f"| Remaining | no deadline; stops on {status.get('stop_policy', 'science')} |"
         if status["remaining_hours"] is None
         else f"| Remaining | {status['remaining_hours']:.2f} h |"),
        f"| Iteration | {status['iteration']} |",
        f"| Candidates evaluated | {status['candidates_evaluated']} |",
        f"| Validated | {status['validated_jobs']} |",
        f"| Failed | {status['failed_jobs']} |",
        f"| Queued | {len(status['queued'])} |",
        f"| GPU | {status['gpu'].get('utilisation_percent', '?')}% util, "
        f"{status['gpu'].get('memory_percent', '?')}% VRAM |",
        f"| Disk free | {status['disk']['free_gb']} GB "
        f"(reserve {status['disk_reserve_gb']} GB) |",
        f"| Campaign size | {status['campaign_bytes'] / 1024**2:.0f} MB |", "",
        "## Protocol", "",
        f"OPLS-AA (QM-qualified) · {status['protocol']['production_ns']:g} ns production · "
        f"{status['protocol']['replicas']} replicas · "
        f"{status['protocol']['temperature_k']:g} K / {status['protocol']['pressure_bar']:g} bar · "
        f"DP {status['protocol']['degree_of_polymerization']} x "
        f"{status['protocol']['chains_per_system']} chains · "
        f"N_eff floor {status['protocol']['min_effective_samples']} · "
        f"{status['protocol']['analysis']}", "",
        "## Candidates", "",
        "| Polymer | Density (kg/m³) | Uncertainty | Replicas | N_eff | g | Gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in status["candidate_ranking"]:
        density = (f"{row['density_kg_m3']:.1f}" if row["density_kg_m3"] is not None else "—")
        uncertainty = (f"±{row['uncertainty_kg_m3']:.1f}"
                       if row["uncertainty_kg_m3"] is not None else "—")
        n_eff = ", ".join(f"{v:.0f}" if v else "—" for v in row["replica_n_eff"]) or "—"
        g = ", ".join(f"{v:.0f}" if v else "—" for v in row["replica_g"]) or "—"
        lines.append(
            f"| {row['candidate']} | {density} | {uncertainty} | "
            f"{row['replicas_usable']}/{row['replicas_run']} | {n_eff} | {g} | {row['gate']} |"
        )
    lines += ["", "Densities are **simulation-derived** OPLS-AA melt values, not "
              "experimental measurements.", ""]

    diagnosed = [r for r in status["candidate_ranking"] if r["diagnostics"]]
    if diagnosed:
        lines += ["## Diagnostics", ""]
        for row in diagnosed:
            for note in row["diagnostics"]:
                lines.append(f"- **{row['candidate']}** — {note}")
        lines.append("")
    if status["queued"]:
        lines += [f"**Queued.** {', '.join(status['queued'])}", ""]
    if status["stopped_reason"]:
        lines += [f"**Stopped.** {status['stopped_reason']}", ""]
    return "\n".join(lines)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".monitor.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="campaign/run")
    parser.add_argument("--config", default="configs/campaign_96h.yaml")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    # None means open-ended, which is now the default. The monitor used to call float()
    # on this unconditionally and die at startup against an open-ended config -- leaving
    # a running campaign unmonitored while looking, from the outside, like the campaign
    # itself had failed.
    raw_budget = cfg["campaign"].get("duration_hours")
    budget = None if raw_budget is None else float(raw_budget)
    reserve = float(cfg["resources"]["disk_reserve_gb"])

    while True:
        if not (root / "campaign_state.json").is_file():
            print(f"waiting for {root}/campaign_state.json", flush=True)
        else:
            status = build_status(root, cfg, budget)
            write_atomic(root / "campaign_status.json", json.dumps(status, indent=1))
            write_atomic(root / "campaign_status.md", render_markdown(status))
            if status["disk"]["free_gb"] < reserve:
                print(f"DISK ALERT: {status['disk']['free_gb']} GB free is below the "
                      f"{reserve} GB reserve", flush=True)
            if not status["driver_running"] and not status["stopped_reason"]:
                print("DRIVER ALERT: the campaign driver is not running and recorded no "
                      "stop reason", flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
