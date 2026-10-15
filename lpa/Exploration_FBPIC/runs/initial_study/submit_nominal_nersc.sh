#!/bin/bash
#SBATCH -J fbpic-nominal
#SBATCH -A gen0007
#SBATCH -C gpu
#SBATCH --qos debug
#SBATCH --time 00:30:00
#SBATCH --nodes 1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --gpus-per-task=1

set -euo pipefail

RUN_DIRECTORY="${PSCRATCH}/ionization_injection_exploration/nominal_test"
mkdir -p "$RUN_DIRECTORY"
exec >"$RUN_DIRECTORY/slurm-${SLURM_JOB_ID}.out" \
	2>"$RUN_DIRECTORY/slurm-${SLURM_JOB_ID}.err"

module load PrgEnv-gnu cpe-cuda cudatoolkit python
source activate fbpic

export MPICH_GPU_SUPPORT_ENABLED=0
export FBPIC_ENABLE_GPUDIRECT=0

srun --cpu-bind=none python run_nominal.py
