#!/usr/bin/env bash
# Run a command under Nsight Systems. The run's StepClock starts and stops the capture
# with cudaProfilerStart/Stop, so only the --profile-window iterations are recorded.
#
# Usage: NSYS=/path/to/nsys nsys_wrap.sh RUN_DIR COMMAND [ARGS...]
set -euo pipefail
run="$1"
shift
rank="${SLURM_PROCID:-0}"
exec "${NSYS:?NSYS is not set}" profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output="$run/profile/rank$rank" \
    "$@"
