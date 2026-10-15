#!/bin/bash
#SBATCH -J fbpic-explore
#SBATCH -A REPLACE_WITH_PROJECT
#SBATCH -C gpu
#SBATCH --qos regular
#SBATCH --time 04:00:00
# Four simultaneous eight-GPU FBPIC cases: 32 GPUs total on Perlmutter.
#SBATCH --nodes 8

set -euo pipefail

module load PrgEnv-gnu cpe-cuda cudatoolkit python
source activate fbpic

# Install this once per environment or replace with a managed package path.
# python -m pip install -e ../../../Simulation_FBPIC
# python -m pip install -e ../..'[nersc,dev]'

export MPICH_GPU_SUPPORT_ENABLED=0
export FBPIC_ENABLE_GPUDIRECT=0

# One manager-hosted Xopt generator and four eight-GPU libEnsemble workers.
srun --ntasks=5 --cpu-bind=none python -m exploration_fbpic.libensemble_driver \
  campaign.yaml --stage stage_1
