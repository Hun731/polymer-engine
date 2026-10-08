#!/usr/bin/env bash
# One read-only supervision sample. No side effects on the campaign.
set -uo pipefail
ROOT="${ROOT:-campaign/run}"
# `kill -0 0` signals the caller's own process group and always succeeds, so the old
# `|| echo 0` fallback made a *missing* pid file read as a running campaign -- the one
# case where the answer matters most.
alive() {
    local pid="$1"
    [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -0 "$pid" 2>/dev/null
}
D=$(cat campaign/pids/driver.pid 2>/dev/null || true)
M=$(cat campaign/pids/monitor.pid 2>/dev/null || true)
alive "$D" && DS=up || DS=DOWN
alive "$M" && MS=up || MS=DOWN
RAM=$(free -g | awk 'NR==2{print $3"/"$2}')
LOAD=$(awk '{print $1}' /proc/loadavg)
AGE=$(( $(date +%s) - $(stat -c %Y "$ROOT/campaign_state.json" 2>/dev/null || date +%s) ))
SAGE=$(( $(date +%s) - $(stat -c %Y "$ROOT/campaign_status.json" 2>/dev/null || date +%s) ))
.venv/bin/python - "$ROOT" "$DS" "$MS" "$RAM" "$LOAD" "$AGE" "$SAGE" <<'PY'
import json, sys
root, ds, ms, ram, load, age, status_age = sys.argv[1:8]
s = json.load(open(f"{root}/campaign_status.json"))
f = s["in_flight"]
budget = s['budget_hours']
clock = (f"{s['elapsed_hours']:6.2f}h/{budget:g}" if budget is not None
         else f"{s['elapsed_hours']:6.2f}h/open")
# Everything below comes from a file the monitor writes. With the monitor stopped it
# describes whenever it last ran, so say so rather than presenting it as now.
stale = int(status_age) > 180
print(("[STALE %ss] " % status_age) if stale else "", end="")
print(f"{s['current_time'][11:19]}Z {clock} "
      f"driver={ds} mon={ms} | "
      f"{f.get('candidate','-')[:14]}/{f.get('replica','-')[-2:]}/{f.get('stage','-')} "
      f"{f.get('percent_complete','-')}%{'' if f.get('live', True) else ' STALE'} "
      f"| eval={s['candidates_evaluated']} "
      f"val={s['validated_jobs']} fail={s['failed_jobs']} q={len(s['queued'])} | "
      f"gpu={s['gpu'].get('utilisation_percent')}%/{s['gpu'].get('memory_percent')}vram "
      f"ram={ram}GB load={load} disk={s['disk']['free_gb']:.0f}GB "
      f"size={s['campaign_bytes']/1024**2:.0f}MB ckpt={int(age)}s")
PY
