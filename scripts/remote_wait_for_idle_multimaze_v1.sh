#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/z00819216/vla-research/areal-pacman
LOG_DIR="$ROOT/run_artifacts"
LOCK_FILE="$LOG_DIR/multi_maze_v1_20260715_waiter.lock"

mkdir -p "$LOG_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "ABORT: another multi-maze waiter already holds $LOCK_FILE" >&2
    exit 73
fi

cd "$ROOT"
empty_checks=0
while (( empty_checks < 2 )); do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
    if [[ -z "$pids" ]]; then
        empty_checks=$((empty_checks + 1))
        echo "$(date -Is) GPU empty check $empty_checks/2"
    else
        empty_checks=0
        echo "$(date -Is) waiting for GPU processes: $(echo "$pids" | tr '\n' ',')"
    fi
    if (( empty_checks < 2 )); then
        sleep 60
    fi
done

echo "$(date -Is) GPU idle confirmed; starting multi-maze supervisor"
exec bash scripts/remote_supervise_multimaze_v1.sh
