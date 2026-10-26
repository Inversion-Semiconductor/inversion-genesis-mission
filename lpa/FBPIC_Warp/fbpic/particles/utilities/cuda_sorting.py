# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""Warp kernels retain the baseline particle formulas and array layout."""
import warp as wp
from fbpic.utils.warp import warp_kernel

from typing import Any
import numpy as np
from cupy.cuda import thrust
from fbpic.utils.printing import catch_gpu_memory_error

wp.set_module_options({"enable_backward": False})


@warp_kernel
def get_cell_idx_per_particle(
    cell_idx: wp.array(dtype=wp.int32),
    sorted_idx: wp.array(dtype=wp.int64),
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
        sorted_idx[i] = wp.int64(i)
        # Calculate the 1D cell_idx
        cell_idx[i] = ir_upper + iz_upper * (Nr + 1)


@catch_gpu_memory_error
def sort_particles_per_cell(cell_idx, sorted_idx):
    """
    Sort the cell index of the particles and
    modify the sorted index array accordingly.

    Parameters
    ----------
    cell_idx : 1darray of integers
        The cell index of the particle

    sorted_idx : 1darray of integers
        Represents the original index of the
        particle before the sorting.
    """
    Ntot = cell_idx.shape[0]
    if Ntot > 0:
        if type(cell_idx) is np.ndarray or type(sorted_idx) is np.ndarray:
            raise ValueError("Unexpected CPU array")
        d_cell_idx = cell_idx
        d_sorted_idx = sorted_idx
        # `thrust.argsort` will simultaneously:
        # - find the indices `sorted_idx` that sort the initial array cell_idx
        # - sort `cell_idx` in place
        thrust.argsort(
            dtype=d_cell_idx.dtype,
            idx_start=d_sorted_idx.data.ptr,
            data_start=d_cell_idx.data.ptr,
            keys_start=0,
            shape=d_cell_idx.shape,
        )
        # `thrust.argsort` allocates its temporary storage through the CuPy
        # allocator. The allocator keeps this memory cached for later sorts
        # (see `fbpic.utils.warp.release_unused_device_memory`).


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
    sorted_idx: wp.array(dtype=wp.int64),
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
