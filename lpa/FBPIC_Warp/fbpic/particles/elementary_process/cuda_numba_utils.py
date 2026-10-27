# Copyright 2017, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a number of methods that are useful for elementary processes
(e.g. ionization, Compton scattering) on CPU and GPU
"""
import numpy as np
from fbpic.utils.threading import njit_parallel, prange

# Check if CUDA is available, then import CUDA functions
from fbpic.utils.cuda import cuda_installed

if cuda_installed:
    import cupy
    import warp as wp
    from typing import Any
    from fbpic.utils.warp import warp_kernel
    from fbpic.particles.utilities.capacity import grow_particle_arrays


def allocate_empty(shape, use_cuda, dtype):
    """
    Allocate and return an empty array, of size `N` and type `dtype`,
    either on GPU or CPU, depending on whether `use_cuda` is True or False
    """
    if type(shape) is not tuple:
        # Convert single scalar to tuple
        shape = (shape,)
    if use_cuda:
        return cupy.empty(shape, dtype=dtype)
    else:
        return np.empty(shape, dtype=dtype)


def perform_cumsum(input_array, use_cuda):
    """
    Return an array containing the cumulative sum of the 1darray `input_array`

    (The returned array has one more element than `input_array`; its first
    element is 0 and its last element is the total sum of `input_array`)
    """
    if use_cuda:
        cumulative_array = cupy.zeros(len(input_array) + 1, dtype=cupy.int64)
        cupy.cumsum(input_array, out=cumulative_array[1:])
    else:
        cumulative_array = np.zeros(len(input_array) + 1, dtype=np.int64)
        np.cumsum(input_array, out=cumulative_array[1:])
    return cumulative_array


def perform_cumsum_2d(input_array, use_cuda, out=None):
    """
    Return an array containing the cumulative sum of the 2darray `input_array`,
    where the cumsum is taken along the last axis.

    (The returned array has one more element than `input_array` along the
    last axis; its first element is 0 and its last element is the
    total sum of `input_array`)

    If needed, the calculation is performed on the GPU. On the GPU, `out`
    can provide the int64 result array (C-contiguous, of the result shape):
    then nothing is allocated. Each row is scanned separately, because a
    batched CuPy scan of several rows into the strided `out[:, 1:]`
    allocates a contiguous temporary.
    """
    new_shape = (input_array.shape[0], input_array.shape[1] + 1)
    if use_cuda and out is not None:
        assert out.shape == new_shape and out.dtype == cupy.int64
        out[:, 0] = 0
        for i in range(new_shape[0]):
            cupy.cumsum(input_array[i : i + 1], out=out[i : i + 1, 1:], axis=-1)
        cumulative_array = out
    elif use_cuda:
        cumulative_array = cupy.zeros(new_shape, dtype=cupy.int64)
        cupy.cumsum(input_array, out=cumulative_array[:, 1:], axis=-1)
    else:
        cumulative_array = np.zeros(new_shape, dtype=np.int64)
        np.cumsum(input_array, out=cumulative_array[:, 1:], axis=-1)
    return cumulative_array


def reallocate_and_copy_old(species, use_cuda, old_Ntot, new_Ntot):
    """
    Copy the particle quantities of `species` from arrays of size `old_Ntot`
    into arrays of size `new_Ntot`. Set these arrays as attributes of `species.

    (The first `old_Ntot` elements of the new arrays are copied from the old
    arrays ; the last elements are left empty and expected to be filled later.)

    When `use_cuda` is True, this function also reallocates
    the sorting buffers for GPU, with a size `new_Ntot`

    When the data is on the GPU, the arrays are capacity views (see
    `fbpic.particles.utilities.capacity`): if `new_Ntot` fits in the capacity
    of the species, the arrays become longer views of the same memory,
    without allocation or copy; otherwise all arrays move to new memory with
    more capacity.

    Parameters
    ----------
    species: an fbpic Particles object
    use_cuda: bool
        If True, the new arrays are device arrays, and copying is done on GPU.
        If False, the arrays are on CPU, and copying is done on CPU.
    old_Ntot, new_Ntot: int
        Size of the old and new arrays (with old_Ntot < new_Ntot)
    """
    # Check if the data is on the GPU
    data_on_gpu = type(species.w) is not np.ndarray
    if data_on_gpu:
        grow_particle_arrays(species, old_Ntot, new_Ntot)
        return

    # The data is on the host: copy on the CPU
    # (With use_cuda, the ids and the sorting buffers are device arrays: the
    # data is on the host only before the first `send_particles_to_gpu`.)
    for attr in [
        "x",
        "y",
        "z",
        "ux",
        "uy",
        "uz",
        "w",
        "inv_gamma",
        "Ex",
        "Ey",
        "Ez",
        "Bx",
        "By",
        "Bz",
    ]:
        old_array = getattr(species, attr)
        new_array = allocate_empty(new_Ntot, False, dtype=np.float64)
        copy_particle_data_numba(old_Ntot, old_array, new_array)
        setattr(species, attr, new_array)
    # Copy the tracking id, if needed
    if species.tracker is not None:
        old_array = species.tracker.id
        new_array = allocate_empty(new_Ntot, use_cuda, dtype=np.uint64)
        copy_particle_data_numba(old_Ntot, old_array, new_array)
        species.tracker.id = new_array
    # And the spin too, if needed
    if species.spin_tracker is not None:
        for attr in ["sx", "sy", "sz"]:
            old_array = getattr(species.spin_tracker, attr)
            new_array = allocate_empty(new_Ntot, False, dtype=np.float64)
            copy_particle_data_numba(old_Ntot, old_array, new_array)
            setattr(species.spin_tracker, attr, new_array)

    # Allocate the auxiliary arrays for GPU
    # (cell_idx and sorted_idx exist only during a sort)
    if use_cuda:
        species.sorting_buffer = cupy.empty((new_Ntot,), dtype=np.float64)
        if species.n_integer_quantities > 0:
            species.int_sorting_buffer = cupy.empty((new_Ntot,), dtype=np.uint64)

    # Modify the total number of particles
    species.Ntot = new_Ntot


def generate_new_ids(species, old_Ntot, new_Ntot):
    """
    If `species` is tracked, then generate new ids, between the
    indices `old_Ntot` and `new_Ntot` of the particle ID arrays

    (This is performed either on GPU or CPU.)
    """
    if species.tracker is not None:
        if species.use_cuda:
            species.tracker.generate_new_ids_gpu(old_Ntot, new_Ntot)
        else:
            species.tracker.id[old_Ntot:new_Ntot] = species.tracker.generate_new_ids(
                new_Ntot - old_Ntot
            )


@njit_parallel
def copy_particle_data_numba(Ntot, old_array, new_array):
    """
    Copy the `Ntot` elements of `old_array` into `new_array`, on CPU
    """
    # Loop over single particles (in parallel if threading is enabled)
    for ip in prange(Ntot):
        new_array[ip] = old_array[ip]
    return new_array


if cuda_installed:

    @warp_kernel
    def copy_particle_data_cuda(
        Ntot: int,
        old_array: wp.array(dtype=Any),
        new_array: wp.array(dtype=Any),
    ):
        """
        Copy the `Ntot` elements of `old_array` into `new_array`, on GPU
        """
        # Loop over single particles
        ip = wp.tid()
        if ip < Ntot:
            new_array[ip] = old_array[ip]
