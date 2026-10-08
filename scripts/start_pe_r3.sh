#!/usr/bin/env bash
# Scheduled start of the campaign to run polyethylene replica 3.
#
# The campaign requeues polyethylene on start (2 of 3 replicas usable, attempts under
# the raised budget), reuses the finished replicas, runs only the missing one, and
# stops itself when the queue drains. Refuses to start over someone else's GPU job:
# that contention is what consumed polyethylene's first attempt.
set -uo pipefail
cd /mnt/data/Polymer
exec >> campaign/scheduled_start.log 2>&1
echo "=== scheduled start $(date -Is) ==="
if pgrep -x gmx >/dev/null; then
    echo "REFUSED: a GROMACS job is already on the GPU:"
    pgrep -ax gmx
    echo "not starting; reschedule manually when the GPU is free"
    exit 1
fi
if [ -f campaign/pids/driver.pid ] && kill -0 "$(cat campaign/pids/driver.pid)" 2>/dev/null; then
    echo "driver already running (pid $(cat campaign/pids/driver.pid)); nothing to do"
    exit 0
fi
ROOT=campaign/run4 CONFIG=configs/campaign_anneal.yaml ./scripts/campaign_ctl.sh start
