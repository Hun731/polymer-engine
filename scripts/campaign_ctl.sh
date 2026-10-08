#!/usr/bin/env bash
# Start / stop / status for the density campaign and its monitor.
#
# Exists because `pkill -f run_density_campaign` also matches the shell running the
# pkill, so an interactive stop kills the caller before it kills the campaign. PIDs are
# written to files and signalled directly, which cannot self-match.
set -uo pipefail

ROOT="${ROOT:-campaign/run}"
CONFIG="${CONFIG:-configs/campaign_96h.yaml}"
PYTHON="${PYTHON:-.venv/bin/python}"
PIDDIR="campaign/pids"
mkdir -p "$PIDDIR"

start() {
    if [ -f "$PIDDIR/driver.pid" ] && kill -0 "$(cat "$PIDDIR/driver.pid")" 2>/dev/null; then
        echo "driver already running (pid $(cat "$PIDDIR/driver.pid"))"; return 1
    fi
    mkdir -p "$ROOT"
    nohup "$PYTHON" scripts/run_density_campaign.py --config "$CONFIG" --root "$ROOT" \
        >> campaign/run_driver.log 2>&1 &
    echo $! > "$PIDDIR/driver.pid"
    sleep 5
    nohup "$PYTHON" scripts/monitor_campaign.py --root "$ROOT" --config "$CONFIG" --interval 60 \
        >> campaign/monitor.log 2>&1 &
    echo $! > "$PIDDIR/monitor.pid"
    echo "driver  pid $(cat "$PIDDIR/driver.pid")"
    echo "monitor pid $(cat "$PIDDIR/monitor.pid")"
}

# Every gmx process descended from a PID, found before that PID is signalled.
# Recorded first because killing the driver orphans its children, after which there is
# no longer any link back to say which gmx belonged to this campaign.
descendant_gmx() {
    local root="$1" out=""
    local queue="$root" next pid
    while [ -n "$queue" ]; do
        next=""
        for pid in $queue; do
            for child in $(pgrep -P "$pid" 2>/dev/null); do
                next="$next $child"
                if [ "$(cat /proc/$child/comm 2>/dev/null)" = "gmx" ]; then out="$out $child"; fi
            done
        done
        queue="$next"
    done
    echo $out
}

stop() {
    # `pgrep -x gmx` used to stand here. It matches every GROMACS process on the
    # machine, so stopping this campaign also killed an unrelated job belonging to
    # someone else's project that happened to share the GPU. Only processes descended
    # from this campaign's own driver are signalled now.
    local driver_pid="" gmx_pids=""
    if [ -f "$PIDDIR/driver.pid" ]; then
        driver_pid=$(cat "$PIDDIR/driver.pid")
        if kill -0 "$driver_pid" 2>/dev/null; then
            gmx_pids=$(descendant_gmx "$driver_pid")
        fi
    fi

    # Stop the driver first so it cannot launch another stage while gmx is being killed.
    for name in driver monitor; do
        f="$PIDDIR/$name.pid"
        if [ -f "$f" ]; then
            pid=$(cat "$f")
            if kill -0 "$pid" 2>/dev/null; then kill "$pid" && echo "stopped $name ($pid)"; fi
            rm -f "$f"
        fi
    done

    if [ -z "$gmx_pids" ]; then
        echo "no gmx processes belonged to this campaign"
        local others
        others=$(pgrep -xc gmx 2>/dev/null || echo 0)
        [ "$others" -gt 0 ] && echo "  ($others gmx process(es) belong to something else and were left alone)"
        return 0
    fi

    sleep 2
    for pid in $gmx_pids; do
        kill "$pid" 2>/dev/null && echo "stopped gmx ($pid, this campaign's)"
    done
    sleep 3
    for pid in $gmx_pids; do
        kill -9 "$pid" 2>/dev/null && echo "force-stopped gmx ($pid)"
    done
}

status() {
    for name in driver monitor; do
        f="$PIDDIR/$name.pid"
        if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
            echo "$name: running (pid $(cat "$f"))"
        else
            echo "$name: not running"
        fi
    done
    echo "gmx processes: $(pgrep -xc gmx || echo 0)"
    [ -f "$ROOT/campaign_status.md" ] && sed -n '5p' "$ROOT/campaign_status.md" | sed 's/\*\*//g'
}

case "${1:-status}" in
    start)  start ;;
    stop)   stop ;;
    status) status ;;
    *) echo "usage: $0 {start|stop|status}"; exit 2 ;;
esac
