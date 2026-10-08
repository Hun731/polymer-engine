#!/usr/bin/env bash
# One-glance status of the CHARMM density campaign. Run: bash scripts/campaign_status.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="$ROOT/campaign/charmm_gui/repo/runs"

# --- is it alive? -------------------------------------------------------------
PID="$(cat "$RUNS/campaign.pid" 2>/dev/null)"
if [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; then
  ET="$(ps -p "$PID" -o etime= | tr -d ' ')"
  echo "● RUNNING   pid $PID   up $ET"
else
  echo "○ NOT RUNNING (restart: cd $ROOT && setsid nohup .venv/bin/python \\"
  echo "                 scripts/run_charmm_system.py --all-ready \\"
  echo "                 > $RUNS/campaign_resume.log 2>&1 < /dev/null & )"
fi

# --- how far along ------------------------------------------------------------
DONE="$(python3 -c "import json;p=json.load(open('$RUNS/campaign_progress.json'));print(p['n_done'],p['n_selected'])" 2>/dev/null)"
REPS="$(find "$RUNS" -name prod.gro 2>/dev/null | wc -l)"
echo "  systems: ${DONE:-? ?} done   |   replicas finished: $REPS / 129"

# --- what it's doing now ------------------------------------------------------
CUR="$(ls -t "$RUNS"/*/replica_*/*.log 2>/dev/null | head -1)"
if [ -n "$CUR" ]; then
  REL="${CUR#$RUNS/}"
  STAGE="$(basename "${CUR%.log}")"
  MDP="${CUR%/*}/$STAGE.mdp"
  NSTEPS="$(grep -aE '^nsteps' "$MDP" 2>/dev/null | grep -oE '[0-9]+' | head -1)"
  STEP="$(grep -aE '^ +[0-9]+ +[0-9]' "$CUR" 2>/dev/null | tail -1 | awk '{print $1}')"
  AGE=$(( $(date +%s) - $(stat -c %Y "$CUR") ))
  PCT=""; [ -n "$STEP" ] && [ -n "$NSTEPS" ] && PCT="$(( STEP * 100 / NSTEPS ))% of $STAGE"
  echo "  now:     $REL   ${PCT}   (log ${AGE}s ago)"
fi

# --- gpu ----------------------------------------------------------------------
echo -n "  gpu:     "
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "n/a"
