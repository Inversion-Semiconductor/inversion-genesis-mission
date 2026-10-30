#!/bin/bash
#SBATCH -J fbpic-stage1
#SBATCH -A gen0007
#SBATCH -C gpu
#SBATCH --qos regular
#SBATCH --time 18:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=cedoss@inversionsemi.com
# Thirty-two simultaneous four-GPU FBPIC cases: one 4-GPU node per case.
#SBATCH --nodes 32

set -euo pipefail

module load PrgEnv-gnu cpe-cuda cudatoolkit python
source activate fbpic

# Install this once per environment or replace with a managed package path.
# python -m pip install -e ../../../Simulation_FBPIC
# python -m pip install -e ../..'[nersc,dev]'

export MPICH_GPU_SUPPORT_ENABLED=0
export FBPIC_ENABLE_GPUDIRECT=0
export LIBE_PLATFORM=perlmutter_g
export SLURM_EXACT=1
export SLURM_MEM_PER_NODE=0

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from its directory.}"

# One manager-hosted Xopt generator and 32 four-GPU libEnsemble workers.
srun --ntasks=33 --cpu-bind=none python -m exploration_fbpic.libensemble_driver \
  campaign.yaml --stage stage_1
