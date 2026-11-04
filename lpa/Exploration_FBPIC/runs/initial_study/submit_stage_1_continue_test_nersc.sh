#!/bin/bash
#SBATCH -J fbpic-stage1-continue-test
#SBATCH -A gen0007
#SBATCH -C gpu
#SBATCH --qos debug
#SBATCH --time 00:30:00
#SBATCH --nodes 4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cedoss@inversionsemi.com

set -euo pipefail

module load PrgEnv-gnu cpe-cuda cudatoolkit python
source activate fbpic

export MPICH_GPU_SUPPORT_ENABLED=0
export FBPIC_ENABLE_GPUDIRECT=0
export LIBE_PLATFORM=perlmutter_g
export SLURM_EXACT=1
export SLURM_MEM_PER_NODE=0

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from its directory.}"

# Copy only compact production history; all generated files remain isolated.
TEST_ROOT="${PSCRATCH:?PSCRATCH must be set}/ionization_injection_stage_1_continue_test"
PRODUCTION_HISTORY="${PSCRATCH}/ionization_injection_exploration/campaign_runs.parquet"
mkdir -p "${TEST_ROOT}/history" "${TEST_ROOT}/runs"
cp -f "${PRODUCTION_HISTORY}" "${TEST_ROOT}/history/campaign_runs.parquet"

# One manager and four four-GPU workers: feasibility-gated Stage 1 proposals.
srun --ntasks=5 --cpu-bind=none python -m exploration_fbpic.libensemble_driver \
  campaign.yaml --stage stage_1_continue_test