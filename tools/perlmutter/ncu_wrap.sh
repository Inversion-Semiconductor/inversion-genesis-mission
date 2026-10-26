#!/usr/bin/env bash
# Run a command under Nsight Compute. The run's StepClock starts and stops the capture
# with cudaProfilerStart/Stop, so only the kernels of the --profile-window steps are
# profiled. Each kernel launch is replayed to collect the metrics below.
#
# Usage: NCU=/path/to/ncu ncu_wrap.sh RUN_DIR COMMAND [ARGS...]
set -euo pipefail
run="$1"
shift
rank="${SLURM_PROCID:-0}"

# Throughput (percent of peak), traffic, FP64/FP32 operation counts, and warp stalls.
# bench/ncu_table.py reads these names; check new names with
# `ncu --query-metrics --chip ga100 --query-metrics-mode suffix --metrics <base>`.
metrics=(
    gpu__time_duration.sum
    gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
    sm__throughput.avg.pct_of_peak_sustained_elapsed
    dram__throughput.avg.pct_of_peak_sustained_elapsed
    dram__bytes_read.sum
    dram__bytes_write.sum
    lts__throughput.avg.pct_of_peak_sustained_elapsed
    lts__t_sector_hit_rate.pct
    l1tex__throughput.avg.pct_of_peak_sustained_active
    l1tex__t_sector_hit_rate.pct
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
    sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active
    smsp__sass_thread_inst_executed_op_dadd_pred_on.sum
    smsp__sass_thread_inst_executed_op_dmul_pred_on.sum
    smsp__sass_thread_inst_executed_op_dfma_pred_on.sum
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
    smsp__sass_thread_inst_executed_op_fmul_pred_on.sum
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
    sm__warps_active.avg.pct_of_peak_sustained_active
    smsp__issue_active.avg.pct_of_peak_sustained_active
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
    smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
    smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
)
metric_list="$(IFS=,; echo "${metrics[*]}")"

exec "${NCU:?NCU is not set}" \
    --profile-from-start off \
    --target-processes all \
    --replay-mode kernel \
    --section LaunchStats \
    --section Occupancy \
    --metrics "$metric_list" \
    --force-overwrite \
    --export "$run/profile/rank$rank" \
    "$@"
