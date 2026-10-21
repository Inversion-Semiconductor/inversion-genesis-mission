# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It creates the NCCL communicator that carries the domain exchanges on the GPU
when FBPIC_COMM_BACKEND=nccl (see fbpic/utils/mpi.py).
"""
import atexit
import sys
import numpy as np

try:
    import nccl.core as nccl

    nccl_installed = True
except ImportError:
    nccl = None
    nccl_installed = False

# One NCCL communicator per process, over the ranks of MPI COMM_WORLD
# (like `comm` in fbpic/utils/mpi.py). Successive Simulation objects reuse it.
_nccl_comm = None
# Set by an unhandled exception: the collective teardown is then skipped
_unhandled_exception = False

# NCCL has no complex type: send complex buffers as views of their real parts.
_real_view_dtype = {
    np.dtype(np.complex128): np.float64,
    np.dtype(np.complex64): np.float32,
}


def get_nccl_comm(mpi_comm):
    """Return the NCCL communicator over `mpi_comm`, created at the first call.

    The first call is collective: every rank of `mpi_comm` must make it, with
    its own GPU current.

    Args:
        mpi_comm: The mpi4py communicator of all ranks (COMM_WORLD).

    Returns:
        A `nccl.core.Communicator` with one rank per MPI rank.
    """
    global _nccl_comm
    if _nccl_comm is None:
        unique_id = nccl.get_unique_id() if mpi_comm.rank == 0 else None
        unique_id = mpi_comm.bcast(unique_id, root=0)
        _nccl_comm = nccl.Communicator.init(mpi_comm.size, mpi_comm.rank, unique_id)
        # Registered after mpi4py, so it runs before MPI finalization.
        atexit.register(_destroy_nccl_comm)
        sys.excepthook = _flag_unhandled_exception(sys.excepthook)
    return _nccl_comm


def _flag_unhandled_exception(excepthook):
    def flagged_excepthook(*args):
        global _unhandled_exception
        _unhandled_exception = True
        excepthook(*args)

    return flagged_excepthook


def _destroy_nccl_comm():
    global _nccl_comm
    # `destroy` is collective over the ranks of a node. After an unhandled
    # exception, the other ranks may never reach it, so let the process exit
    # (the launcher then stops the other ranks).
    if _nccl_comm is not None and not _unhandled_exception:
        _nccl_comm.destroy()
    _nccl_comm = None


def as_nccl_buffer(array):
    """Return a view of a CuPy array that NCCL can send or receive.

    nccl4py reads only the pointer and the size of the array, and ignores the
    strides. A complex array becomes a view of its real and imaginary parts,
    which is exact.

    Args:
        array: C-contiguous CuPy array.

    Returns:
        The same memory, with a real or integer dtype.

    Raises:
        ValueError: The array is not C-contiguous.
    """
    if not array.flags.c_contiguous:
        raise ValueError("NCCL buffers must be C-contiguous.")
    real_dtype = _real_view_dtype.get(array.dtype)
    if real_dtype is not None:
        return array.view(real_dtype)
    return array
