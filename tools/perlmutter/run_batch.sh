#!/usr/bin/env bash
# Run every staged task of a batch, in tasks.txt order. Called by launch.sh or job.sbatch.
#
# Usage: run_batch.sh BATCH_DIR
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
batch="${1:?usage: run_batch.sh BATCH_DIR}"
# shellcheck source=/dev/null
source "$batch/batch.env"

mapfile -t runs <"$batch/tasks.txt"
log "batch $batch: ${#runs[@]} runs, layout $LAYOUT, action $ACTION"
{
    hostname
    date -u
    env | grep -E '^SLURM_' | sort
} >"$batch/meta/node.txt" 2>&1
nvidia-smi -q >"$batch/meta/nvidia-smi.txt" 2>&1 || true

failures=0
if [[ "$LAYOUT" == concurrent && "$ACTION" != local ]]; then
    # One GPU per run; at most four runs at a time on a node.
    running=0
    for run in "${runs[@]}"; do
        if [[ $running -ge 4 ]]; then
            wait -n || failures=$((failures + 1))
            running=$((running - 1))
        fi
        FBPIC_CONCURRENT=1 bash "$batch/tools/run_task.sh" "$run" &
        running=$((running + 1))
    done
    while [[ $running -gt 0 ]]; do
        wait -n || failures=$((failures + 1))
        running=$((running - 1))
    done
else
    for run in "${runs[@]}"; do
        bash "$batch/tools/run_task.sh" "$run" || failures=$((failures + 1))
    done
fi

log "batch finished: $((${#runs[@]} - failures)) succeeded, $failures failed"
[[ $failures -eq 0 ]]
