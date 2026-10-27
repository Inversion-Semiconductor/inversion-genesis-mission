# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""Warp kernels retain the baseline particle formulas and array layout."""
import warp as wp
from fbpic.utils.warp import as_warp_array, get_runtime, warp_kernel

from typing import Any
import numpy as np
from fbpic.utils.printing import catch_gpu_memory_error
from fbpic.particles.utilities.capacity import reserve_workspace

wp.set_module_options({"enable_backward": False})


@warp_kernel
def get_cell_idx_per_particle(
    cell_idx: wp.array(dtype=wp.int32),
    sorted_idx: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    z: wp.array(dtype=wp.float64),
    invdz: wp.float64,
    zmin: wp.float64,
    Nz: wp.int32,
    invdr: wp.float64,
    rmin: wp.float64,
    Nr: wp.int32,
):
    """
    Get the cell index of each particle.
    The cell index is 1d and calculated by:
    cell index in z + cell index in r * number of cells in z.
    The cell_idx of a particle is defined by
    the lower cell in r and z, that it deposits its field to.

    Parameters
    ----------
    cell_idx : 1darray of integers
        The cell index of the particle

    sorted_idx : 1darray of integers
        The sorted index array needs to be reset
        before doing the sort

    x, y, z : 1darray of floats (in meters)
        The position of the particles
        (is modified by this function)

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box, in each direction

    Nz, Nr : int
        Number of gridpoints along the considered direction
    """
    i = wp.tid()
    if i < cell_idx.shape[0]:
        # Preliminary arrays for the cylindrical conversion
        xj = x[i]
        yj = y[i]
        zj = z[i]
        rj = wp.sqrt(xj * xj + yj * yj)

        # Positions of the particles, in the cell unit
        r_cell = invdr * (rj - rmin) - wp.float64(0.5)
        z_cell = invdz * (zj - zmin) - wp.float64(0.5)

        # Original index of the uppper grid point in z and r
        ir_upper = wp.int32(wp.ceil(r_cell))
        iz_upper = wp.int32(wp.ceil(z_cell))

        # Treat the boundary conditions
        # absorbing in upper r
        if ir_upper > Nr:
            ir_upper = Nr
        # periodic boundaries in z
        if iz_upper < 0:
            iz_upper += Nz
        elif iz_upper > Nz - 1:
            iz_upper -= Nz
        # iz_upper has values between 0 and Nz-1.
        # ir_upper has values between 0 and Nr (included).
        # This corresponds to the Nz*(Nr+1) different inter-gridpoint
        # areas in a box that is periodic in z but aperiodic in r.

        # Reset sorted_idx array
        sorted_idx[i] = i
        # Calculate the 1D cell_idx
        cell_idx[i] = ir_upper + iz_upper * (Nr + 1)


_SORT_WORKSPACE = "particle_sort"


def reserve_sort_scratch(count):
    """Reserve the shared int32 sort block for `count` particles.

    The block holds the keys in ``[0, 2 * cap)`` and the values in
    ``[2 * cap, 4 * cap)``, where ``cap`` is a quarter of its size: the radix
    sort needs ``2 * n`` elements for each, as a double buffer. All species
    share the block (see `fbpic.particles.utilities.capacity`), because the
    sorts run one after the other on the runtime stream and the sort indices
    live only inside `Particles.sort_particles`.

    When the block grows, one sort of ``cap`` uninitialised elements also
    grows the cached CUB temporary storage of Warp (``acquire_temp_buffer``
    in ``warp/native/sort.cu``, one buffer per stream that only grows). The
    later sorts of at most ``cap`` particles then allocate nothing. This
    relies on the Warp 1.18 cache; `tests/test_gpu_steady_state_allocations.py`
    detects a change.

    Args:
        count: Number of particles; the species capacity, so that a growth
            within the capacity does not grow the block.

    Returns:
        The 1D int32 CuPy block.

    Raises:
        ValueError: `count` does not fit in an int32 sort index.
    """
    count = int(count)
    if count >= 2**31:
        raise ValueError("The int32 particle sort needs Ntot < 2**31.")
    block, grown = reserve_workspace(_SORT_WORKSPACE, 4 * count, np.int32)
    runtime = get_runtime()
    if grown and runtime.recorded_launches is None:
        cap = block.size // 4
        with wp.ScopedStream(runtime.stream, sync_enter=False):
            wp.utils.radix_sort_pairs(
                as_warp_array(block[: 2 * cap], wp.int32),
                as_warp_array(block[2 * cap :], wp.int32),
                min(cap, 2**31 - 1),
            )
    return block


def allocate_sort_scratch(species):
    """Point ``cell_idx`` and ``sorted_idx`` at the shared sort block.

    The block (`reserve_sort_scratch`) has room for the capacity of the
    species, ``species.particle_capacity``, so that a growth of the species
    within its capacity does not allocate. `release_sort_scratch` drops the
    views after the sort; the block stays until the end of
    ``Simulation.step``.

    Args:
        species: Particles object on the GPU. The function sets
            ``species.cell_idx`` and ``species.sorted_idx`` to int32 CuPy views
            of length ``species.Ntot``, and ``species._sort_keys`` and
            ``species._sort_values`` to the radix-sort storage.

    Raises:
        ValueError: ``species.Ntot`` does not fit in an int32 sort index.
    """
    n = int(species.Ntot)
    if n >= 2**31:
        raise ValueError("The int32 particle sort needs Ntot < 2**31.")
    count = min(max(n, int(getattr(species, "particle_capacity", 0))), 2**31 - 1)
    block = reserve_sort_scratch(count)
    cap = block.size // 4
    species._sort_keys = block[: 2 * n]
    species._sort_values = block[2 * cap : 2 * cap + 2 * n]
    species.cell_idx = block[:n]
    species.sorted_idx = block[2 * cap : 2 * cap + n]


def release_sort_scratch(species):
    """Drop the views of `allocate_sort_scratch` (the shared block stays)."""
    species._sort_keys = species._sort_values = None
    species.cell_idx = species.sorted_idx = None


@catch_gpu_memory_error
def sort_particles_per_cell(species):
    """
    Sort the cell index of the particles and
    modify the sorted index array accordingly.

    The stable radix sort of Warp (CUB Onesweep) sorts the int32 keys
    ``species.cell_idx`` and the int32 values ``species.sorted_idx`` in place.
    The sort runs the full 32 key bits: an even number of passes leaves the
    result in the first half of the storage, so no copy-back is needed.

    Parameters
    ----------
    species : a Particles object
        Its ``cell_idx`` and ``sorted_idx`` arrays are views of the block
        that `allocate_sort_scratch` binds.
    """
    Ntot = int(species.Ntot)
    if Ntot > 0:
        runtime = get_runtime()
        if runtime.recorded_launches is not None:
            # The sort runs immediately, so it cannot be ordered after
            # kernels that a CUDA graph section only records.
            raise RuntimeError("Particle sorting cannot run in a CUDA graph section.")
        # The sort runs on the current Warp stream of the current context:
        # select the shared runtime stream (and its device) explicitly.
        with wp.ScopedStream(runtime.stream, sync_enter=False):
            wp.utils.radix_sort_pairs(
                as_warp_array(species._sort_keys, wp.int32),
                as_warp_array(species._sort_values, wp.int32),
                Ntot,
            )


@warp_kernel
def incl_prefix_sum(
    cell_idx: wp.array(dtype=wp.int32),
    prefix_sum: wp.array(dtype=wp.int32),
):
    """
    Perform an inclusive parallel prefix sum on the sorted
    cell index array. The prefix sum array represents the
    cumulative sum of the number of particles per cell
    for each cell index.

    Parameters
    ----------
    cell_idx : 1darray of integers
        The cell index of the particle

    prefix_sum : 1darray of integers
        Represents the cumulative sum of
        the particles per cell
    """
    # i is the index of the macroparticle
    i = wp.tid()
    if i < cell_idx.shape[0] - 1:
        # ci: index of the cell of the present macroparticle
        ci = cell_idx[i]
        # ci_next: index of the cell of the next macroparticle
        ci_next = cell_idx[i + 1]
        # Fill all the cells between ci and ci_next with the
        # inclusive cumulative sum of the number particles until ci
        while ci < ci_next:
            # The cumulative sum of the number of particle per cell
            # until ci is i+1 (since i obeys python index, starting at 0)
            prefix_sum[ci] = i + 1
            ci += 1


@warp_kernel
def prefill_prefix_sum(
    cell_idx: wp.array(dtype=wp.int32),
    prefix_sum: wp.array(dtype=wp.int32),
    Ntot: wp.int32,
):
    """
    Prefill the prefix sum array so that:
        - the cells that have a lower index than the cell that contains
        the first particle are set to 0
        - the cells that have a higher index than the cell that contains
        the last particle are set to the total number of particles (Ntot)

    All the cells in between will have their value set by `incl_prefix_sum`

    Parameters
    ----------
    cell_idx : 1darray of integers
        The cell index of the particles
    prefix_sum : 1darray of integers
        Represents the cumulative sum of
        the particles per cell
    Ntot: int
        The total number of particles in the current species
    """
    # One thread per cell
    i = wp.tid()
    if i < prefix_sum.shape[0]:
        if Ntot > 0:
            # Fill the first cells with 0
            if i < cell_idx[0]:
                prefix_sum[i] = 0
            # Fill the last cells with Ntot
            elif i >= cell_idx[Ntot - 1]:
                prefix_sum[i] = Ntot
        else:
            # If this species has no particles, fill all cells with 0
            prefix_sum[i] = 0


@warp_kernel
def write_sorting_buffer(
    sorted_idx: wp.array(dtype=wp.int32),
    val: wp.array(dtype=Any),
    buf: wp.array(dtype=Any),
):
    """
    Writes the values of a particle array to a buffer,
    while rearranging them to match the sorted cell index array.

    Parameters
    ----------
    sorted_idx : 1darray of integers
        Represents the original index of the
        particle before the sorting

    val : 1d array of floats
        A particle data array

    buf : 1d array of floats
        A buffer array to temporarily store the
        sorted particle data array
    """
    i = wp.tid()
    if i < val.shape[0]:
        buf[i] = val[sorted_idx[i]]


@warp_kernel
def write_sorting_buffer2(
    sorted_idx: wp.array(dtype=wp.int32),
    val0: wp.array(dtype=Any),
    val1: wp.array(dtype=Any),
    buf0: wp.array(dtype=Any),
    buf1: wp.array(dtype=Any),
):
    """
    Writes the values of two particle arrays to two buffers,
    while rearranging them to match the sorted cell index array.
    Each thread reads the sorted index once for both arrays.

    Parameters
    ----------
    sorted_idx : 1darray of integers
        Represents the original index of the
        particle before the sorting

    val0, val1 : 1d arrays of floats or integers
        Particle data arrays

    buf0, buf1 : 1d arrays of floats or integers
        Buffer arrays (with the dtype of val0 and val1)
        to temporarily store the sorted particle data arrays
    """
    i = wp.tid()
    if i < sorted_idx.shape[0]:
        j = sorted_idx[i]
        buf0[i] = val0[j]
        buf1[i] = val1[j]


@warp_kernel
def write_sorting_buffer4(
    sorted_idx: wp.array(dtype=wp.int32),
    val0: wp.array(dtype=wp.float64),
    val1: wp.array(dtype=wp.float64),
    val2: wp.array(dtype=wp.float64),
    val3: wp.array(dtype=wp.float64),
    buf0: wp.array(dtype=wp.float64),
    buf1: wp.array(dtype=wp.float64),
    buf2: wp.array(dtype=wp.float64),
    buf3: wp.array(dtype=wp.float64),
):
    """
    Writes the values of four particle arrays to four buffers,
    while rearranging them to match the sorted cell index array.
    Each thread reads the sorted index once for the four arrays,
    and has four independent gathers in flight.

    Parameters
    ----------
    sorted_idx : 1darray of integers
        Represents the original index of the
        particle before the sorting

    val0, val1, val2, val3 : 1d arrays of floats
        Particle data arrays

    buf0, buf1, buf2, buf3 : 1d arrays of floats
        Buffer arrays to temporarily store the sorted particle data arrays
    """
    i = wp.tid()
    if i < sorted_idx.shape[0]:
        j = sorted_idx[i]
        buf0[i] = val0[j]
        buf1[i] = val1[j]
        buf2[i] = val2[j]
        buf3[i] = val3[j]
