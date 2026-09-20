#!/bin/bash
# Restarts a run if its records file goes stale (no writes) for >900s.
# Usage: nohup bash scripts/watchdog.sh >> results/watchdog.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
source scripts/env.sh >/dev/null 2>&1

stale() {  # $1=file  -> 0 if older than 900s
  local m
  m=$(stat -f %m "$1" 2>/dev/null) || return 1
  [ $(( $(date +%s) - m )) -gt 1800 ]
}

while true; do
  sleep 300
  # Run A: freshest of runs.jsonl / checkpoints.jsonl
  if stale results/runs.jsonl && stale results/checkpoints.jsonl; then
    echo "$(date '+%H:%M') watchdog: run A stale — restarting"
    pkill -f "experiment.py --config config/experiment.json" 2>/dev/null
    sleep 3
    nohup python3 experiment.py --config config/experiment.json run \
      >> results/fullrun.log 2>&1 &
  fi
  # Run B
  if stale results_b/checkpoints.jsonl && stale results_b/runs.jsonl; then
    echo "$(date '+%H:%M') watchdog: run B stale — restarting"
    pkill -f "scripts/run_b.py" 2>/dev/null
    sleep 3
    nohup python3 scripts/run_b.py >> results_b_run.log 2>&1 &
  fi
done
