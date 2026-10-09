#!/bin/bash
#SBATCH -J fbpic-smoke
#SBATCH -A gen0007
#SBATCH -C gpu
#SBATCH --qos debug
#SBATCH --time 00:10:00
# Eight two-GPU cases fit on four Perlmutter GPU nodes.
#SBATCH --nodes 4

set -euo pipefail

module load PrgEnv-gnu cpe-cuda cudatoolkit python
source activate fbpic

# libEnsemble detects Perlmutter resource sets and gives each worker a unique
# two-GPU allocation. Avoid Slurm task/GPU directives here: the MPI executor
# supplies GPU-per-rank settings to the FBPIC sub-tasks.
export LIBE_PLATFORM=perlmutter_g
export SLURM_EXACT=1
export SLURM_MEM_PER_NODE=0
export MPICH_GPU_SUPPORT_ENABLED=0
export FBPIC_ENABLE_GPUDIRECT=0

CONFIG=campaign.yaml

# One manager plus eight simulation workers. Do not bind the outer libEnsemble
# ranks to cores: each worker starts a separate two-rank FBPIC job step.
srun --ntasks=9 --cpu-bind=none python -m exploration_fbpic.libensemble_driver "$CONFIG" --stage smoke