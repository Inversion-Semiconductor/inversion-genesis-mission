#!/bin/bash
#SBATCH -J fbpic-surrogate-smoke
#SBATCH -A gen0007
#SBATCH -C gpu
#SBATCH --qos regular
#SBATCH --time 03:00:00
# Six two-GPU full-FBPIC cases fit on three Perlmutter GPU nodes.
#SBATCH --nodes 3

set -euo pipefail

module load PrgEnv-gnu cpe-cuda cudatoolkit python
source activate fbpic

export LIBE_PLATFORM=perlmutter_g
export SLURM_EXACT=1
export SLURM_MEM_PER_NODE=0
export MPICH_GPU_SUPPORT_ENABLED=0
export FBPIC_ENABLE_GPUDIRECT=0

# One manager-hosted Xopt generator and six libEnsemble simulation workers.
srun --ntasks=7 --cpu-bind=none python -m exploration_fbpic.libensemble_driver \
  campaign.yaml --stage surrogate_smoke