"""
Boilerplate code to import MPI and cupy so that it is configured for running on AWS.

This script may be required to be uploaded to AWS if the launch script imports this.
"""

from mpi4py import MPI
import os

# --------------------------------------------------------------------------
# MPI / GPU selection
# --------------------------------------------------------------------------
# Each MPI rank must lock itself to a single GPU.  We first ask the MPI
# launcher (Open MPI, MPICH, etc.) if it exported a LOCAL_RANK environment
# variable; if not we fall back to a simple modulo on the global rank.
# This guarantees we launch *one* cooperating FBPIC domain per GPU instead
# of N independent runs.
# --------------------------------------------------------------------------

try:
    import cupy as cp
except ImportError:
    print("Module 'cupy' not available, will not run mpi")
    cp = None

if cp is not None:
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()

    # Detect the per-node rank if provided by the MPI runtime
    local_rank_env = (
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("MPI_LOCALRANKID")  # Intel MPI / MPICH
        or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK")
    )

    if local_rank_env is not None:
        gpu_id = int(local_rank_env)
    else:
        # Fallback that also works in single-node tests
        n_gpu = cp.cuda.runtime.getDeviceCount()
        gpu_id = world_rank % n_gpu

    # Bind the rank to its GPU
    cp.cuda.Device(gpu_id).use()

    # Optional: only the master rank prints banner information
    if world_rank == 0:
        from fbpic import __version__ as fbpic_version

        print(
            f"FBPIC {fbpic_version} running with {comm.Get_size()} MPI ranks on {n_gpu if 'n_gpu' in locals() else 'unknown'} GPUs per node."
        )