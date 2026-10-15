#!/bin/bash
#SBATCH -J xopt-sine-libe
#SBATCH -A gen0007
#SBATCH -C gpu
#SBATCH --qos debug
#SBATCH --time 00:10:00
# Eight two-GPU resource sets fit on four Perlmutter GPU nodes.
#SBATCH --nodes 4

set -euo pipefail

module load PrgEnv-gnu cpe-cuda cudatoolkit python
source activate fbpic

export LIBE_PLATFORM=perlmutter_g
export SLURM_EXACT=1
export SLURM_MEM_PER_NODE=0
export MPICH_GPU_SUPPORT_ENABLED=0
export FBPIC_ENABLE_GPUDIRECT=0

# One Xopt manager plus eight two-GPU toy simulation workers.
srun --ntasks=9 --cpu-bind=none python ../../toy_xopt_libe.py campaign.yaml

# One transferable artifact containing campaign metadata, libEnsemble logs,
# inputs, results, GPU mappings, timings, and posterior uncertainty snapshots.
tar -czf xopt_sine_libe_bundle.tar.gz campaign.yaml submit_nersc.sh ensemble.log \
	libE_stats.txt -C "${PSCRATCH}/exploration_fbpic_xopt_sine_libe" .