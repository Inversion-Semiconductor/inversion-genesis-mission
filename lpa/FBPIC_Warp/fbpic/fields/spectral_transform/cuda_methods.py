# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of functions that are useful when converting the
fields from interpolation grid to the spectral grid and vice-versa
"""
import warp as wp
from fbpic.utils.warp import warp_kernel, mul_i

# ------------------
# Copying functions
# ------------------

# Micro-tile of the z <-> r transposing copies: one warp of 32 threads
# copies TILE_Z consecutive z by TILE_R consecutive r. With complex128,
# the 1d side then moves TILE_Z*16 = 128 B per r column and the 2d side
# moves TILE_R*16 = 64 B per z row, so both sides use full 32 B sectors.
# (This is the warp shape of the upstream numba kernels on V100/A100.)
TILE_Z = wp.constant(8)
TILE_R = wp.constant(4)
COPY_TPB = 256


def cuda_copy_1d_2d_launch(Nz, Nr):
    """Return the launch shape of `cuda_copy_2d_to_1d` and `cuda_copy_1d_to_2d`.

    Args:
        Nz: Number of grid points along z (axis 0 of the 2d array).
        Nr: Number of grid points along r (axis 1 of the 2d array).

    Returns:
        A ``(blocks, threads_per_block)`` pair for a 1D launch with one
        warp per micro-tile. The thread count is a multiple of the
        warp size; the kernels mask the threads beyond the last tile.
    """
    n_tiles = ((Nz + TILE_Z - 1) // TILE_Z) * ((Nr + TILE_R - 1) // TILE_R)
    return (32 * n_tiles + COPY_TPB - 1) // COPY_TPB, COPY_TPB


@wp.func
def _tile_indices(t: int, Nr: int):
    """Map the linear thread index `t` to (iz, ir) in its warp micro-tile."""
    n_tile_r = (Nr + TILE_R - 1) // TILE_R
    warp = t // 32
    lane = t - warp * 32
    iz = (warp // n_tile_r) * TILE_Z + lane % TILE_Z
    ir = (warp % n_tile_r) * TILE_R + lane // TILE_Z
    return iz, ir


@warp_kernel
def cuda_copy_2dC_to_2dR(
    array_in: wp.array2d(dtype=wp.vec2d),
    array_out: wp.array2d(dtype=wp.float64),
):
    """
    Store the complex Nz x Nr array `array_in`
    into the real 2Nz x Nr array `array_out`,
    by storing the real part in the first Nz elements of `array_out` along z,
    and the imaginary part in the next Nz elements.

    Parameters :
    ------------
    array_in: 2darray of complexs
        Array of shape (Nz, Nr)
    array_out: 2darray of reals
        Array of shape (2*Nz, Nr)
    """
    # Set up cuda grid
    iz, ir = wp.tid()
    Nz = array_in.shape[0]
    Nr = array_in.shape[1]

    # Copy from array_in to array_out
    if (iz < Nz) and (ir < Nr):
        array_out[iz, ir] = array_in[iz, ir][0]
        array_out[iz + Nz, ir] = array_in[iz, ir][1]


@warp_kernel
def cuda_copy_2dR_to_2dC(
    array_in: wp.array2d(dtype=wp.float64),
    array_out: wp.array2d(dtype=wp.vec2d),
):
    """
    Reconstruct the complex Nz x Nr array `array_out`,
    from the real 2Nz x Nr array `array_in`,
    by interpreting the first Nz elements of `array_in` along z as
    the real part, and the next Nz elements as the imaginary part.

    Parameters :
    ------------
    array_in: 2darray of reals
        Array of shape (2*Nz, Nr)
    array_out: 2darray of complexs
        Array of shape (Nz, Nr)
    """
    # Set up cuda grid
    iz, ir = wp.tid()
    Nz = array_out.shape[0]
    Nr = array_out.shape[1]

    # Copy from array_in to array_out
    if (iz < Nz) and (ir < Nr):
        array_out[iz, ir] = wp.vec2d(array_in[iz, ir], array_in[iz + Nz, ir])


@warp_kernel
def cuda_copy_2d_to_1d(
    array_2d: wp.array2d(dtype=wp.vec2d),
    array_1d: wp.array(dtype=wp.vec2d),
):
    """
    Copy array_2d to array_1d, so that the first axis of array_2d
    is contiguous in array_1d

    This is typically done before using the cuFFT API,
    which requires 1d arrays to do FFT in only one dimension.

    Parameters :
    ------------
    array_2d : 2darray of complexs
        Array of shape (Nz, Nr)

    array_1d : 1d array of complexs
        Array of shape (Nz*Nr,)

    Launch with the 1D shape from `cuda_copy_1d_2d_launch`.
    """

    # Set up cuda grid: one warp per TILE_Z x TILE_R micro-tile
    iz, ir = _tile_indices(wp.tid(), array_2d.shape[1])

    # Copy from array_2d to array_1d
    if (iz < array_2d.shape[0]) and (ir < array_2d.shape[1]):
        i = iz + array_2d.shape[0] * ir
        array_1d[i] = array_2d[iz, ir]


@warp_kernel
def cuda_copy_1d_to_2d(
    array_1d: wp.array(dtype=wp.vec2d),
    array_2d: wp.array2d(dtype=wp.vec2d),
):
    """
    Copy array_1d to array_2d, so that the first axis of array_2d
    is contiguous in array_1d

    This is typically done after using the cuFFT API,
    since the the output of cuFFT is a 1d array

    Parameters :
    ------------
    array_2d : 2darray of complexs
        Array of shape (Nz, Nr)

    array_1d : 1d array of complexs
        Array of shape (Nz*Nr,)

    Launch with the 1D shape from `cuda_copy_1d_2d_launch`.
    """

    # Set up cuda grid: one warp per TILE_Z x TILE_R micro-tile
    iz, ir = _tile_indices(wp.tid(), array_2d.shape[1])

    # Copy from array_1d to array_2d
    if (iz < array_2d.shape[0]) and (ir < array_2d.shape[1]):
        i = iz + array_2d.shape[0] * ir
        array_2d[iz, ir] = array_1d[i]


# ----------------------------------------------------
# Functions that combine components in spectral space
# ----------------------------------------------------


@warp_kernel
def cuda_rt_to_pm(
    buffer_r: wp.array2d(dtype=wp.vec2d),
    buffer_t: wp.array2d(dtype=wp.vec2d),
    buffer_p: wp.array2d(dtype=wp.vec2d),
    buffer_m: wp.array2d(dtype=wp.vec2d),
):
    """
    Combine the arrays buffer_r and buffer_t to produce the
    arrays buffer_p and buffer_m, according to the rules of
    the Fourier-Hankel decomposition (see associated paper)
    """
    # Set up cuda grid
    iz, ir = wp.tid()

    if (iz < buffer_r.shape[0]) and (ir < buffer_r.shape[1]):
        # Use intermediate variables, as the arrays
        # buffer_r and buffer_t may actually point to the same
        # object as buffer_p and buffer_m, for economy of memory
        value_r = buffer_r[iz, ir]
        value_t = buffer_t[iz, ir]
        # Combine the values
        buffer_p[iz, ir] = wp.float64(0.5) * (value_r - mul_i(value_t))
        buffer_m[iz, ir] = wp.float64(0.5) * (value_r + mul_i(value_t))


@warp_kernel
def cuda_pm_to_rt(
    buffer_p: wp.array2d(dtype=wp.vec2d),
    buffer_m: wp.array2d(dtype=wp.vec2d),
    buffer_r: wp.array2d(dtype=wp.vec2d),
    buffer_t: wp.array2d(dtype=wp.vec2d),
):
    """
    Combine the arrays buffer_p and buffer_m to produce the
    arrays buffer_r and buffer_t, according to the rules of
    the Fourier-Hankel decomposition (see associated paper)
    """
    # Set up cuda grid
    iz, ir = wp.tid()

    if (iz < buffer_p.shape[0]) and (ir < buffer_p.shape[1]):
        # Use intermediate variables, as the arrays
        # buffer_r and buffer_t may actually point to the same
        # object as buffer_p and buffer_m, for economy of memory
        value_p = buffer_p[iz, ir]
        value_m = buffer_m[iz, ir]
        # Combine the values
        buffer_r[iz, ir] = value_p + value_m
        buffer_t[iz, ir] = mul_i(value_p - value_m)
