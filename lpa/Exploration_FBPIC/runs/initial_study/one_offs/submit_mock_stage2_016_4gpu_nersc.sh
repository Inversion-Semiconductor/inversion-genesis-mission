#!/bin/bash
#SBATCH -J fbpic-oneoff
#SBATCH -A gen0007
#SBATCH -C gpu
#SBATCH --qos regular
#SBATCH --time=02:00:00
#SBATCH --nodes 1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cedoss@inversionsemi.com

set -euo pipefail

module load PrgEnv-gnu cpe-cuda cudatoolkit python
source activate fbpic

export MPICH_GPU_SUPPORT_ENABLED=0
export FBPIC_ENABLE_GPUDIRECT=0

SCRIPT_DIRECTORY="${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the one_offs directory.}"
if [[ ! -f "${SCRIPT_DIRECTORY}/run_one_off.py" ]]; then
  echo "Expected run_one_off.py in submit directory: $SCRIPT_DIRECTORY" >&2
  exit 2
fi
CANDIDATE="${ONE_OFF_CANDIDATE:-${SCRIPT_DIRECTORY}/candidate_mock_stage2_016.json}"
RUN_ROOT="${PSCRATCH:?PSCRATCH must be set}/ionization_injection_exploration_evaluations"
RUN_ID="${ONE_OFF_RUN_ID:-mock_stage2_016}"
RUN_DIRECTORY="${RUN_ROOT}/${RUN_ID}"

if [[ ! -f "$CANDIDATE" ]]; then
  echo "Candidate JSON not found: $CANDIDATE" >&2
  exit 2
fi

python "${SCRIPT_DIRECTORY}/run_one_off.py" prepare \
  --candidate "$CANDIDATE" \
  --run-root "$RUN_ROOT" \
  --run-id "$RUN_ID"

exec >"${RUN_DIRECTORY}/slurm-${SLURM_JOB_ID}.out" \
  2>"${RUN_DIRECTORY}/slurm-${SLURM_JOB_ID}.err"

pushd "$RUN_DIRECTORY" >/dev/null
# Slurm directives above already reserve four MPI ranks and one GPU per rank.
srun -n 4 python run_fbpic.py
popd >/dev/null

python "${SCRIPT_DIRECTORY}/run_one_off.py" finalize \
  --candidate "$CANDIDATE" \
  --run-root "$RUN_ROOT" \
  --run-id "$RUN_ID"