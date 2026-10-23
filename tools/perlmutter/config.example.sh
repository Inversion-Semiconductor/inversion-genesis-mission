# Perlmutter FBPIC benchmark configuration.
#
# Copy this file to ~/.config/fbpic-perlmutter.sh and edit the values.
# Set FBPIC_PERLMUTTER_CONFIG to use a different file.
# Every script in tools/perlmutter sources the selected file.

# NERSC project directory name and Slurm account for GPU jobs.
# The GPU account can need a "_g" suffix. Check with: sacctmgr show assoc user=$USER
export NERSC_PROJECT="gen0007"
export NERSC_ACCOUNT="gen0007"

# Worktree root for this repository (managed by sync.sh), laid out like the workstation:
#   $FBPIC_WORKTREES/.bare     bare clone; origin branches are refs/remotes/origin/*
#   $FBPIC_WORKTREES/.git      the line "gitdir: ./.bare"
#   $FBPIC_WORKTREES/<name>/   one worktree per branch; <name> is the part after the last "/"
export FBPIC_WORKTREES="$HOME/fbpic/inversion-genesis-mission-worktrees"
export FBPIC_REPO_URL="git@github.com:Inversion-Semiconductor/inversion-genesis-mission.git"
# Branches that sync.sh keeps as worktrees.
export FBPIC_WORKTREE_BRANCHES="snidhan/perl-main-run snidhan/perl-warp-run main snidhan/dev-main"
# Bare mirror of upstream FBPIC. launch.sh only archives UPSTREAM_FBPIC_SHA from it.
export UPSTREAM_FBPIC_GIT="$HOME/fbpic/fbpic.git"
export UPSTREAM_FBPIC_URL="https://github.com/fbpic/fbpic.git"
# The upstream commit that FBPIC-Warp forked from (FBPIC 0.27.1).
export UPSTREAM_FBPIC_SHA="dc1cc32ca36ab1ee813d82ea5e9f6b180be0e2a9"

# Shared dependency environment (created by setup_env.sh). It holds no fbpic package.
export FBPIC_ENV="/global/common/software/$NERSC_PROJECT/$USER/fbpic-deps"
# MPI flavor for setup_env.sh: "conda" uses conda-forge mpich with srun --mpi=none
# (one rank per job step, enough for these single-GPU runs). "external" tries cray-mpich
# through cray-mpich-abi (needed for multi-node) and falls back to "conda" when conda-forge
# has no fitting build. Jobs use the flavor that the env records in $FBPIC_ENV.mpi.
export FBPIC_MPI="conda"
# Extra modules for every job step. numba-cuda-580-patch fixes numba<0.63 on the 580 driver.
export FBPIC_MODULES="numba-cuda-580-patch"

# Run directories (purged after 8 weeks without access) and long-term archive.
export FBPIC_RUNS="$SCRATCH/fbpic-runs"
export FBPIC_ARCHIVE="/global/cfs/cdirs/$NERSC_PROJECT/$USER/fbpic-bench"

# Default Slurm settings for launch.sh --submit.
export FBPIC_QOS="regular"
export FBPIC_TIME="01:00:00"
