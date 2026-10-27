#!/bin/bash
#SBATCH -J fbpic-memory-stress-4g
#SBATCH -A gen0007
#SBATCH -C gpu
#SBATCH --qos regular
#SBATCH --time 02:30:00
#SBATCH --nodes 1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --gpus-per-task=1

set -euo pipefail

export MEMORY_STRESS_RUN_LABEL="memory_stress_4gpu"
RUN_DIRECTORY="${PSCRATCH}/ionization_injection_exploration_stage1_4gpu/${MEMORY_STRESS_RUN_LABEL}"
mkdir -p "$RUN_DIRECTORY"
exec >"$RUN_DIRECTORY/slurm-${SLURM_JOB_ID}.out" \
  2>"$RUN_DIRECTORY/slurm-${SLURM_JOB_ID}.err"

module load PrgEnv-gnu cpe-cuda cudatoolkit python
source activate fbpic

export MPICH_GPU_SUPPORT_ENABLED=0
export FBPIC_ENABLE_GPUDIRECT=0

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from its directory.}"
srun --cpu-bind=none python run_memory_stress.py