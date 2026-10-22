# Perlmutter FBPIC benchmark configuration.
#
# Copy this file to ~/.config/fbpic-perlmutter.sh and edit the values.
# Set FBPIC_PERLMUTTER_CONFIG to use a different file.
# Every script in tools/perlmutter sources the selected file.

# NERSC project directory name and Slurm account for GPU jobs.
# The GPU account can need a "_g" suffix. Check with: sacctmgr show assoc user=$USER
export NERSC_PROJECT="gen0007"
export NERSC_ACCOUNT="gen0007"

# Bare clones of this repository and upstream FBPIC (managed by sync.sh).
export FBPIC_SRC="$HOME/fbpic-src"
export FBPIC_REPO_URL="git@github.com:Inversion-Semiconductor/inversion-genesis-mission.git"
export UPSTREAM_FBPIC_URL="https://github.com/fbpic/fbpic.git"
# The upstream commit that FBPIC-Warp forked from (FBPIC 0.27.1).
export UPSTREAM_FBPIC_SHA="dc1cc32ca36ab1ee813d82ea5e9f6b180be0e2a9"

# Shared dependency environment (created by setup_env.sh). It holds no fbpic package.
export FBPIC_ENV="/global/common/software/$NERSC_PROJECT/$USER/fbpic-deps"
# MPI flavor: "external" uses cray-mpich through cray-mpich-abi (required for multi-node).
# "conda" uses conda-forge mpich with srun --mpi=none (one rank per job step only).
export FBPIC_MPI="external"
# Extra modules for every job step. numba-cuda-580-patch fixes numba<0.63 on the 580 driver.
export FBPIC_MODULES="numba-cuda-580-patch"

# Run directories (purged after 8 weeks without access) and long-term archive.
export FBPIC_RUNS="$SCRATCH/fbpic-runs"
export FBPIC_ARCHIVE="/global/cfs/cdirs/$NERSC_PROJECT/$USER/fbpic-bench"

# Default Slurm settings for launch.sh --submit.
export FBPIC_QOS="regular"
export FBPIC_TIME="01:00:00"
