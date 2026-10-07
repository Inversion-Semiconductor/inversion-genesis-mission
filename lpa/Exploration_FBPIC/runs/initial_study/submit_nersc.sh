#!/bin/bash
#SBATCH -J fbpic-explore
#SBATCH -A REPLACE_WITH_PROJECT
#SBATCH -C gpu
#SBATCH --qos regular
#SBATCH --time 04:00:00
# Four simultaneous eight-GPU FBPIC cases: 32 GPUs total on Perlmutter.
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4

set -euo pipefail

module load python cudatoolkit
source activate fbpic

# Install this once per environment or replace with a managed package path.
# python -m pip install -e ../../Simulation_FBPIC
# python -m pip install -e ../..'[nersc,dev]'

export MPICH_GPU_SUPPORT_ENABLED=0
export FBPIC_ENABLE_GPUDIRECT=0

# campaign.yaml is the source of truth for the stage GPU contract: 8 GPUs per
# simulation, 32 GPUs total, and 4 concurrent simulations. Keep these Slurm
# directives synchronized with that contract.
# A production driver should call libEnsemble with one persistent manager and
# GPU-bound workers, using exploration_fbpic.libensemble_adapter.run_fbpic_simulation.
# This command intentionally generates and audits the Stage 1 initial design
# before consuming allocation time with FBPIC.
explore-fbpic campaign.yaml --stage stage_1 --features \
  total_beam_charge_pc mean_uz cov_x_x cov_y_y cov_uz_uz \
  initial-design --count 24 --seed 0 --output initial_design.json
