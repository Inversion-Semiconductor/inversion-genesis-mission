#!/usr/bin/env bash
# Run one staged benchmark task on a GPU node (or locally with ACTION=local).
#
# Usage: run_task.sh RUN_DIR
# The batch directory holds config.sh, batch.env, and tools/; RUN_DIR holds meta/task.env.
set -euo pipefail
tools="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$tools/lib.sh"
run="${1:?usage: run_task.sh RUN_DIR}"
batch="$(dirname "$run")"
load_config
# shellcheck source=/dev/null
source "$batch/batch.env"
# shellcheck source=/dev/null
source "$run/meta/task.env"
log "run $run ($VARIANT, $MODE, steps=$STEPS)"

# The env records the MPI flavor it was built with; it overrides the config.
if [[ -f "$FBPIC_ENV.mpi" ]]; then
    FBPIC_MPI="$(<"$FBPIC_ENV.mpi")"
fi
if [[ "$ACTION" != local ]]; then
    module load conda
    if [[ "${FBPIC_MPI:-external}" == external ]]; then
        module load cray-mpich-abi
    fi
    for module_name in ${FBPIC_MODULES:-}; do
        module load "$module_name"
    done
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activation hooks (cuda-nvcc) read unset variables; relax nounset for them.
set +u
conda activate "$FBPIC_ENV"
set -u

# CUDA comes from the conda env; do not load the cudatoolkit module.
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export MPICH_GPU_SUPPORT_ENABLED=0
export FBPIC_ENABLE_GPUDIRECT=0
threads=$((${SLURM_CPUS_PER_TASK:-8} / 2))
export NUMBA_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export MPLBACKEND=Agg PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1
export OPENPMD_VERIFY_HOMOGENEOUS_EXTENTS=0
# Cold, run-private compiler caches: every run pays the same JIT cost before the timed window.
export WARP_CACHE_PATH="$run/cache/warp" NUMBA_CACHE_DIR="$run/cache/numba"
export CUPY_CACHE_DIR="$run/cache/cupy" CUDA_CACHE_PATH="$run/cache/cuda"
producer_case="$PRODUCER_DIR/runs/initial_sample"
export PYTHONPATH="$SOLVER_DIR:$PRODUCER_DIR:$producer_case:$batch/tools${PYTHONPATH:+:$PYTHONPATH}"

launcher=()
if [[ "$ACTION" == local ]]; then
    export CUDA_VISIBLE_DEVICES="$LOCAL_GPU"
else
    launcher=(srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" --gpus-per-task=1 --cpu-bind=cores)
    [[ -n "${FBPIC_CONCURRENT:-}" ]] && launcher+=(--exact)
    [[ "${FBPIC_MPI:-external}" == conda ]] && launcher+=(--mpi=none)
fi

{
    env | grep -E '^(CUDA|NUMBA|WARP|CUPY|OMP|MPICH|FBPIC|PYTHON|LD_LIBRARY_PATH|CONDA_PREFIX|SLURM_JOB_ID)' | sort
    module list 2>&1 || true
} >"$run/meta/environment.txt"

if ! "${launcher[@]}" python -m bench.check_node --out "$run/meta/node_check.json" \
    >"$run/meta/node_check.log" 2>&1; then
    cat "$run/meta/node_check.log" >&2
    die "node check failed for $run"
fi

command=(python -u -m bench.run_case --run-dir "$run" --variant "$VARIANT" --steps "$STEPS")
if [[ -n "$PROFILE_WINDOW" ]]; then
    command+=(--profile-window "$PROFILE_WINDOW")
    if [[ -z "${NSYS:-}" ]]; then
        # nsys comes from the cudatoolkit module; keep that module out of the run env.
        NSYS="$(command -v nsys || { module load cudatoolkit >/dev/null 2>&1 && command -v nsys; } || true)"
    fi
    [[ -x "$NSYS" ]] || die "nsys not found; set NSYS=/path/to/nsys"
    export NSYS
    echo "$NSYS" >"$run/meta/nsys_path.txt"
    "$NSYS" --version >>"$run/meta/nsys_path.txt" 2>&1 || true
    command=(bash "$batch/tools/nsys_wrap.sh" "$run" "${command[@]}")
fi

set +e
"${launcher[@]}" "${command[@]}" 2>&1 | tee "$run/run.log"
status=${PIPESTATUS[0]}
set -e

if [[ $status -eq 0 && -n "$PROFILE_WINDOW" ]]; then
    python -m bench.kernels "$run" --nsys "$NSYS" >"$run/profile/kernels.log" 2>&1 ||
        log "kernel summary failed for $run (see profile/kernels.log); rerun bench.kernels later"
fi
log "run $run exited with $status"
exit "$status"
