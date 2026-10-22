#!/usr/bin/env bash
# Copy batch results from scratch to the long-term archive (FBPIC_ARCHIVE).
# Scratch is purged after 8 weeks without access.
#
# Usage: archive.sh [--no-nsys] BATCH_DIR [BATCH_DIR...]
# Copies results, metadata, logs, kernel summaries, and reports.
# Skips source snapshots, compiler caches, and simulation output.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config

include_nsys=1
if [[ "${1:-}" == --no-nsys ]]; then
    include_nsys=0
    shift
fi
[[ $# -gt 0 ]] || die "usage: archive.sh [--no-nsys] BATCH_DIR..."

filters=(
    --include='*/'
    --include='result.json' --include='resources.jsonl' --include='run.log'
    --include='meta/***' --include='report/***' --include='profile/*.csv'
    --include='profile/*.log' --include='tasks.txt' --include='batch.env'
    --include='job.sbatch' --include='slurm-*.out'
)
[[ $include_nsys -eq 1 ]] && filters+=(--include='profile/*.nsys-rep')
filters+=(--exclude='src/' --exclude='cache/' --exclude='out/' --exclude='tools/' --exclude='*')

mkdir -p "$FBPIC_ARCHIVE"
for batch in "$@"; do
    batch="$(cd "$batch" && pwd)"
    rsync -a --prune-empty-dirs "${filters[@]}" "$batch/" "$FBPIC_ARCHIVE/$(basename "$batch")/"
    log "archived $batch -> $FBPIC_ARCHIVE/$(basename "$batch")"
done
