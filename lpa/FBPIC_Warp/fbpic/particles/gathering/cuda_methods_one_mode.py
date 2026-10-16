# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""Warp kernels retain the baseline particle formulas and array layout."""
import warp as wp
from fbpic.utils.warp import warp_kernel, cmul, square, cube

from .cuda_inline_functions import add_linear_gather_for_mode, add_cubic_gather_for_mode

wp.set_module_options({"enable_backward": False})


@warp_kernel
def erase_eb_cuda(
    Ex: wp.array(dtype=wp.float64),
    Ey: wp.array(dtype=wp.float64),
    Ez: wp.array(dtype=wp.float64),
    Bx: wp.array(dtype=wp.float64),
    By: wp.array(dtype=wp.float64),
    Bz: wp.array(dtype=wp.float64),
    Ntot: wp.int32,
):
    """
    Reset the arrays of fields (i.e. set them to 0)

    Parameters
    ----------
    Ex, Ey, Ez, Bx, By, Bz: 1d arrays of floats
        (One element per macroparticle)
        Represents the fields on the macroparticles
    """
    i = wp.tid()
    if i < Ntot:
        Ex[i] = wp.float64(0)
        Ey[i] = wp.float64(0)
        Ez[i] = wp.float64(0)
        Bx[i] = wp.float64(0)
        By[i] = wp.float64(0)
        Bz[i] = wp.float64(0)


# -----------------------
# Field gathering linear
# -----------------------


@warp_kernel
def gather_field_gpu_linear_one_mode(
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    z: wp.array(dtype=wp.float64),
    rmax_gather: wp.float64,
    invdz: wp.float64,
    zmin: wp.float64,
    Nz: wp.int32,
    invdr: wp.float64,
    rmin: wp.float64,
    Nr: wp.int32,
    Er_m: wp.array2d(dtype=wp.vec2d),
    Et_m: wp.array2d(dtype=wp.vec2d),
    Ez_m: wp.array2d(dtype=wp.vec2d),
    Br_m: wp.array2d(dtype=wp.vec2d),
    Bt_m: wp.array2d(dtype=wp.vec2d),
    Bz_m: wp.array2d(dtype=wp.vec2d),
    m: wp.int32,
    Ex: wp.array(dtype=wp.float64),
    Ey: wp.array(dtype=wp.float64),
    Ez: wp.array(dtype=wp.float64),
    Bx: wp.array(dtype=wp.float64),
    By: wp.array(dtype=wp.float64),
    Bz: wp.array(dtype=wp.float64),
):
    """
    Gathering of the fields (E and B) using Warp on the GPU.
    Iterates over the particles, calculates the weighted amount
    of fields acting on each particle based on its shape (linear).
    Fields are gathered in cylindrical coordinates and then
    transformed to cartesian coordinates.
    Supports only mode 0 and 1.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    rmax_gather: float (in meters)
        The radius above which particle do not gather anymore

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box along the
        direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    Er_m, Et_m, Ez_m : 2darray of complexs
        The electric fields on the interpolation grid for the mode m

    Br_m, Bt_m, Bz_m : 2darray of complexs
        The magnetic fields on the interpolation grid for the mode m

    m: int
        Index of the azimuthal mode

    Ex, Ey, Ez : 1darray of floats
        The electric fields acting on the particles
        (is modified by this function)

    Bx, By, Bz : 1darray of floats
        The magnetic fields acting on the particles
        (is modified by this function)
    """
    # Get the 1D CUDA grid
    i = wp.tid()
    # Deposit the field per cell in parallel
    # (for threads < number of particles)
    if i < x.shape[0]:
        # Preliminary arrays for the cylindrical conversion
        # --------------------------------------------
        # Position
        xj = x[i]
        yj = y[i]
        zj = z[i]

        # Cylindrical conversion
        rj = wp.sqrt(xj * xj + yj * yj)
        if rj != wp.float64(0.0):
            invr = wp.float64(1.0) / rj
            cos = xj * invr  # Cosine
            sin = yj * invr  # Sine
        else:
            cos = wp.float64(1.0)
            sin = wp.float64(0.0)
        # Calculate azimuthal complex factor
        exptheta_m = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
        for _ in range(m):
            exptheta_m = cmul(
                exptheta_m,
                wp.vec2d(cos, wp.float64(0.0))
                - wp.vec2d(wp.float64(0.0), wp.float64(1.0)) * sin,
            )

        # Get linear weights for the deposition
        # --------------------------------------------
        # Positions of the particles, in the cell unit
        r_cell = invdr * (rj - rmin) - wp.float64(0.5)
        z_cell = invdz * (zj - zmin) - wp.float64(0.5)

        # Only perform gathering for particles that are below rmax_gather
        if rj < rmax_gather:

            # Original index of the uppper and lower cell
            ir_lower = wp.int32(wp.floor(r_cell))
            ir_upper = ir_lower + 1
            iz_lower = wp.int32(wp.floor(z_cell))
            iz_upper = iz_lower + 1
            # Linear weight
            Sr_lower = wp.float64(ir_upper) - r_cell
            Sr_upper = r_cell - wp.float64(ir_lower)
            Sz_lower = wp.float64(iz_upper) - z_cell
            Sz_upper = z_cell - wp.float64(iz_lower)
            # Set guard weights to zero
            Sr_guard = wp.float64(0.0)

            # Treat the boundary conditions
            # --------------------------------------------
            # guard cells in lower r
            if ir_lower < 0:
                Sr_guard = Sr_lower
                Sr_lower = wp.float64(0.0)
                ir_lower = wp.int32(0)
            # absorbing in upper r
            if ir_lower > Nr - 1:
                ir_lower = Nr - 1
            if ir_upper > Nr - 1:
                ir_upper = Nr - 1
            # periodic boundaries in z
            # lower z boundaries
            if iz_lower < 0:
                iz_lower += Nz
            if iz_upper < 0:
                iz_upper += Nz
            # upper z boundaries
            if iz_lower > Nz - 1:
                iz_lower -= Nz
            if iz_upper > Nz - 1:
                iz_upper -= Nz

            # Precalculate Shapes
            S_ll = Sz_lower * Sr_lower
            S_lu = Sz_lower * Sr_upper
            S_ul = Sz_upper * Sr_lower
            S_uu = Sz_upper * Sr_upper
            S_lg = Sz_lower * Sr_guard
            S_ug = Sz_upper * Sr_guard

            # E-Field
            # -------
            Fr = wp.float64(0.0)
            Ft = wp.float64(0.0)
            Fz = wp.float64(0.0)
            # Add contribution from mode m
            Fr, Ft, Fz = add_linear_gather_for_mode(
                m,
                Fr,
                Ft,
                Fz,
                exptheta_m,
                Er_m,
                Et_m,
                Ez_m,
                iz_lower,
                iz_upper,
                ir_lower,
                ir_upper,
                S_ll,
                S_lu,
                S_lg,
                S_ul,
                S_uu,
                S_ug,
            )
            # Convert to Cartesian coordinates
            # and write to particle field arrays
            Ex[i] += cos * Fr - sin * Ft
            Ey[i] += sin * Fr + cos * Ft
            Ez[i] += Fz

            # B-Field
            # -------
            # Clear the placeholders for the
            # gathered field for each coordinate
            Fr = wp.float64(0.0)
            Ft = wp.float64(0.0)
            Fz = wp.float64(0.0)
            # Add contribution from mode m
            Fr, Ft, Fz = add_linear_gather_for_mode(
                m,
                Fr,
                Ft,
                Fz,
                exptheta_m,
                Br_m,
                Bt_m,
                Bz_m,
                iz_lower,
                iz_upper,
                ir_lower,
                ir_upper,
                S_ll,
                S_lu,
                S_lg,
                S_ul,
                S_uu,
                S_ug,
            )
            # Convert to Cartesian coordinates
            # and write to particle field arrays
            Bx[i] += cos * Fr - sin * Ft
            By[i] += sin * Fr + cos * Ft
            Bz[i] += Fz


# -----------------------
# Field gathering cubic
# -----------------------


@warp_kernel
def gather_field_gpu_cubic_one_mode(
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    z: wp.array(dtype=wp.float64),
    rmax_gather: wp.float64,
    invdz: wp.float64,
    zmin: wp.float64,
    Nz: wp.int32,
    invdr: wp.float64,
    rmin: wp.float64,
    Nr: wp.int32,
    Er_m: wp.array2d(dtype=wp.vec2d),
    Et_m: wp.array2d(dtype=wp.vec2d),
    Ez_m: wp.array2d(dtype=wp.vec2d),
    Br_m: wp.array2d(dtype=wp.vec2d),
    Bt_m: wp.array2d(dtype=wp.vec2d),
    Bz_m: wp.array2d(dtype=wp.vec2d),
    m: wp.int32,
    Ex: wp.array(dtype=wp.float64),
    Ey: wp.array(dtype=wp.float64),
    Ez: wp.array(dtype=wp.float64),
    Bx: wp.array(dtype=wp.float64),
    By: wp.array(dtype=wp.float64),
    Bz: wp.array(dtype=wp.float64),
):
    """
    Gathering of the fields (E and B) using Warp on the GPU.
    Iterates over the particles, calculates the weighted amount
    of fields acting on each particle based on its shape (cubic).
    Fields are gathered in cylindrical coordinates and then
    transformed to cartesian coordinates.
    Supports only mode 0 and 1.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    rmax_gather: float (in meters)
        The radius above which particle do not gather anymore

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box along the
        direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    Er_m, Et_m, Ez_m : 2darray of complexs
        The electric fields on the interpolation grid for the mode m

    Br_m, Bt_m, Bz_m : 2darray of complexs
        The magnetic fields on the interpolation grid for the mode m

    m: int
        Index of the azimuthal mode

    Ex, Ey, Ez : 1darray of floats
        The electric fields acting on the particles
        (is modified by this function)

    Bx, By, Bz : 1darray of floats
        The magnetic fields acting on the particles
        (is modified by this function)
    """

    # Get the 1D CUDA grid
    i = wp.tid()
    # Deposit the field per cell in parallel
    # (for threads < number of particles)
    if i < x.shape[0]:
        # Preliminary arrays for the cylindrical conversion
        # --------------------------------------------
        # Position
        xj = x[i]
        yj = y[i]
        zj = z[i]

        # Cylindrical conversion
        rj = wp.sqrt(xj * xj + yj * yj)
        if rj != wp.float64(0.0):
            invr = wp.float64(1.0) / rj
            cos = xj * invr  # Cosine
            sin = yj * invr  # Sine
        else:
            cos = wp.float64(1.0)
            sin = wp.float64(0.0)
        # Calculate azimuthal complex factor
        exptheta_m = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
        for _ in range(m):
            exptheta_m = cmul(
                exptheta_m,
                wp.vec2d(cos, wp.float64(0.0))
                - wp.vec2d(wp.float64(0.0), wp.float64(1.0)) * sin,
            )

        # Get weights for the deposition
        # --------------------------------------------
        # Positions of the particle, in the cell unit
        r_cell = invdr * (rj - rmin) - wp.float64(0.5)
        z_cell = invdz * (zj - zmin) - wp.float64(0.5)

        # Only perform gathering for particles that are below rmax_gather
        if rj < rmax_gather:

            # Calculate the shape factors
            Sr = wp.vec4d()
            ir_lowest = wp.int32(wp.floor(r_cell)) - 1
            r_local = r_cell - wp.float64(ir_lowest)
            Sr[0] = -wp.float64(1.0) / wp.float64(6.0) * cube(r_local - wp.float64(2.0))
            Sr[1] = (
                wp.float64(1.0)
                / wp.float64(6.0)
                * (
                    wp.float64(3.0) * cube(r_local - wp.float64(1.0))
                    - wp.float64(6.0) * square(r_local - wp.float64(1.0))
                    + wp.float64(4.0)
                )
            )
            Sr[2] = (
                wp.float64(1.0)
                / wp.float64(6.0)
                * (
                    wp.float64(3.0) * cube(wp.float64(2.0) - r_local)
                    - wp.float64(6.0) * square(wp.float64(2.0) - r_local)
                    + wp.float64(4.0)
                )
            )
            Sr[3] = -wp.float64(1.0) / wp.float64(6.0) * cube(wp.float64(1.0) - r_local)
            Sz = wp.vec4d()
            iz_lowest = wp.int32(wp.floor(z_cell)) - 1
            z_local = z_cell - wp.float64(iz_lowest)
            Sz[0] = -wp.float64(1.0) / wp.float64(6.0) * cube(z_local - wp.float64(2.0))
            Sz[1] = (
                wp.float64(1.0)
                / wp.float64(6.0)
                * (
                    wp.float64(3.0) * cube(z_local - wp.float64(1.0))
                    - wp.float64(6.0) * square(z_local - wp.float64(1.0))
                    + wp.float64(4.0)
                )
            )
            Sz[2] = (
                wp.float64(1.0)
                / wp.float64(6.0)
                * (
                    wp.float64(3.0) * cube(wp.float64(2.0) - z_local)
                    - wp.float64(6.0) * square(wp.float64(2.0) - z_local)
                    + wp.float64(4.0)
                )
            )
            Sz[3] = -wp.float64(1.0) / wp.float64(6.0) * cube(wp.float64(1.0) - z_local)

            # E-Field
            # -------
            Fr = wp.float64(0.0)
            Ft = wp.float64(0.0)
            Fz = wp.float64(0.0)
            # Add contribution from mode m
            Fr, Ft, Fz = add_cubic_gather_for_mode(
                m,
                Fr,
                Ft,
                Fz,
                exptheta_m,
                Er_m,
                Et_m,
                Ez_m,
                ir_lowest,
                iz_lowest,
                Sr,
                Sz,
                Nr,
                Nz,
            )
            # Convert to Cartesian coordinates
            # and write to particle field arrays
            Ex[i] += cos * Fr - sin * Ft
            Ey[i] += sin * Fr + cos * Ft
            Ez[i] += Fz

            # B-Field
            # -------
            # Clear the placeholders for the
            # gathered field for each coordinate
            Fr = wp.float64(0.0)
            Ft = wp.float64(0.0)
            Fz = wp.float64(0.0)
            # Add contribution from mode m
            Fr, Ft, Fz = add_cubic_gather_for_mode(
                m,
                Fr,
                Ft,
                Fz,
                exptheta_m,
                Br_m,
                Bt_m,
                Bz_m,
                ir_lowest,
                iz_lowest,
                Sr,
                Sz,
                Nr,
                Nz,
            )
            # Convert to Cartesian coordinates
            # and write to particle field arrays
            Bx[i] += cos * Fr - sin * Ft
            By[i] += sin * Fr + cos * Ft
            Bz[i] += Fz
