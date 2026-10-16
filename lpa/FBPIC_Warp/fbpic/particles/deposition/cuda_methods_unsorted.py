# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""Warp kernels retain the baseline particle formulas and array layout."""
import warp as wp
from fbpic.utils.warp import warp_kernel, cmul

from .cuda_particle_shapes import Sz_linear, Sr_linear

wp.set_module_options({"enable_backward": False})


@warp_kernel
def deposit_rho_gpu_unsorted(
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    z: wp.array(dtype=wp.float64),
    w: wp.array(dtype=wp.float64),
    q: wp.float64,
    invdz: wp.float64,
    zmin: wp.float64,
    Nz: wp.int32,
    invdr: wp.float64,
    rmin: wp.float64,
    Nr: wp.int32,
    rho_m: wp.array2d(dtype=wp.vec2d),
    m: wp.int32,
    beta_n: wp.array(dtype=wp.float64),
):
    """
    Deposition of the charge density rho using Warp on the GPU.
    Iterates over all particles.
    Calculates the weighted amount of rho that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The particles do not have to be sorted. Thus this kernel is
    less parallel then the standard deposition kernels.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    w : 1d array of floats
        The weights of the particles
        (For ionizable atoms: weight times the ionization level)

    q : float
        Charge of the species
        (For ionizable atoms: this is always the elementary charge e)

    rho_m: 2darray of complexs
        The charge density on the interpolation grid for
        mode m. (is modified by this function)

    m: int
        The index of the azimuthal mode

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the considered direction

    Nz, Nr : int
        Number of gridpoints along the considered direction

    beta_n : 1darray of floats
        Ruyten-corrected particle shape factor coefficients
    """
    # Get the 1D CUDA grid
    i = wp.tid()

    if i < w.shape[0]:

        # Sign for the shape factors
        f = 1 - 2 * (m % 2)
        # Preliminary arrays for the cylindrical conversion
        # --------------------------------------------
        # Position
        xj = x[i]
        yj = y[i]
        zj = z[i]
        # Weights
        wj = q * w[i]

        # Cylindrical conversion
        rj = wp.sqrt(xj * xj + yj * yj)
        # Avoid division by 0.
        if rj != wp.float64(0.0):
            invr = wp.float64(1.0) / rj
            cos = xj * invr  # Cosine
            sin = yj * invr  # Sine
        else:
            cos = wp.float64(1.0)
            sin = wp.float64(0.0)
        # Calculate azimuthal factor
        exptheta_m = wp.vec2d(wp.float64(1.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        for _ in range(m):
            exptheta_m = cmul(
                exptheta_m,
                wp.vec2d(cos, wp.float64(0.0))
                + wp.vec2d(wp.float64(0.0), wp.float64(1.0)) * sin,
            )

        # Positions of the particles, in the cell unit
        r_cell = invdr * (rj - rmin) - wp.float64(0.5)
        z_cell = invdz * (zj - zmin) - wp.float64(0.5)

        # Cell indices of the upper cell bounds
        ir = min(wp.int32(wp.ceil(r_cell)), Nr)
        iz = wp.int32(wp.ceil(z_cell))
        # Handle periodic boundaries in z
        if iz < 0:
            iz += Nz
        elif iz >= Nz:
            iz -= Nz

        # Calculate longitudinal indices at which to add charge
        iz0 = iz - 1
        iz1 = iz
        if iz0 < 0:
            iz0 += Nz
        # Calculate radial indices at which to add charge
        ir0 = ir - 1
        ir1 = min(ir, Nr - 1)
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)

        # Ruyten-corrected shape factor coefficient
        bn = beta_n[ir]

        # Calculate rho
        # --------------------------------------------
        R_m_scal = wj * exptheta_m
        R_m_00 = (
            Sr_linear(r_cell, 0, wp.float64(f), bn) * Sz_linear(z_cell, 0) * R_m_scal
        )
        R_m_01 = (
            Sr_linear(r_cell, 0, wp.float64(f), bn) * Sz_linear(z_cell, 1) * R_m_scal
        )
        R_m_10 = (
            Sr_linear(r_cell, 1, wp.float64(f), bn) * Sz_linear(z_cell, 0) * R_m_scal
        )
        R_m_11 = (
            Sr_linear(r_cell, 1, wp.float64(f), bn) * Sz_linear(z_cell, 1) * R_m_scal
        )

        # Add the calculated fields to the global grid
        wp.atomic_add(rho_m, iz0, ir0, wp.vec2d(R_m_00[0], wp.float64(0.0)))
        wp.atomic_add(rho_m, iz0, ir1, wp.vec2d(R_m_10[0], wp.float64(0.0)))
        wp.atomic_add(rho_m, iz1, ir0, wp.vec2d(R_m_01[0], wp.float64(0.0)))
        wp.atomic_add(rho_m, iz1, ir1, wp.vec2d(R_m_11[0], wp.float64(0.0)))
        if m > 0:
            # For azimuthal modes beyond m=0: add imaginary part
            wp.atomic_add(rho_m, iz0, ir0, wp.vec2d(wp.float64(0.0), R_m_00[1]))
            wp.atomic_add(rho_m, iz0, ir1, wp.vec2d(wp.float64(0.0), R_m_10[1]))
            wp.atomic_add(rho_m, iz1, ir0, wp.vec2d(wp.float64(0.0), R_m_01[1]))
            wp.atomic_add(rho_m, iz1, ir1, wp.vec2d(wp.float64(0.0), R_m_11[1]))


# -------------------------------
# Field deposition - linear - J
# -------------------------------


@warp_kernel
def deposit_J_gpu_unsorted(
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    z: wp.array(dtype=wp.float64),
    w: wp.array(dtype=wp.float64),
    q: wp.float64,
    vx: wp.array(dtype=wp.float64),
    vy: wp.array(dtype=wp.float64),
    vz: wp.array(dtype=wp.float64),
    invdz: wp.float64,
    zmin: wp.float64,
    Nz: wp.int32,
    invdr: wp.float64,
    rmin: wp.float64,
    Nr: wp.int32,
    j_r_m: wp.array2d(dtype=wp.vec2d),
    j_t_m: wp.array2d(dtype=wp.vec2d),
    j_z_m: wp.array2d(dtype=wp.vec2d),
    m: wp.int32,
    beta_n: wp.array(dtype=wp.float64),
):
    """
    Deposition of the current J using Warp on the GPU.
    Iterates over all particles.
    Calculates the weighted amount of J that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The particles do not have to be sorted. Thus this kernel is
    less parallel then the standard deposition kernels.

    Parameters
    ----------
    x, y, z : 1darray of floats (in meters)
        The position of the particles

    w : 1d array of floats
        The weights of the particles
        (For ionizable atoms: weight times the ionization level)

    q : float
        Charge of the species
        (For ionizable atoms: this is always the elementary charge e)

    vx, vy, vz : 1darray of floats (in meters * second^-1)
        The non-relativistic velocity of the particles

    j_r_m, j_t_m, j_z_m,: 2darrays of complexs
        The current component in each direction (r, t, z)
        on the interpolation grid for mode m.
        (is modified by this function)

    m: int
        The index of the azimuthal mode considered

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    beta_n : 1darray of floats
        Ruyten-corrected particle shape factor coefficients
    """
    # Get the 1D CUDA grid
    i = wp.tid()

    if i < w.shape[0]:

        # Sign for the shape factors
        f = 1 - 2 * (m % 2)

        # Preliminary arrays for the cylindrical conversion
        # --------------------------------------------
        # Position
        xj = x[i]
        yj = y[i]
        zj = z[i]
        # Velocity
        vxj = vx[i]
        vyj = vy[i]
        vzj = vz[i]
        # Weights
        wj = q * w[i]

        # Cylindrical conversion
        rj = wp.sqrt(xj * xj + yj * yj)
        # Avoid division by 0.
        if rj != wp.float64(0.0):
            invr = wp.float64(1.0) / rj
            cos = xj * invr  # Cosine
            sin = yj * invr  # Sine
        else:
            cos = wp.float64(1.0)
            sin = wp.float64(0.0)
        # Calculate azimuthal factor
        exptheta_m = wp.vec2d(wp.float64(1.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        for _ in range(m):
            exptheta_m = cmul(
                exptheta_m,
                wp.vec2d(cos, wp.float64(0.0))
                + wp.vec2d(wp.float64(0.0), wp.float64(1.0)) * sin,
            )

        # Positions of the particles, in the cell unit
        r_cell = invdr * (rj - rmin) - wp.float64(0.5)
        z_cell = invdz * (zj - zmin) - wp.float64(0.5)

        # Cell indices of the upper cell bounds
        ir = min(wp.int32(wp.ceil(r_cell)), Nr)
        iz = wp.int32(wp.ceil(z_cell))
        # Handle periodic boundaries in z
        if iz < 0:
            iz += Nz
        elif iz >= Nz:
            iz -= Nz

        # Calculate longitudinal indices at which to add charge
        iz0 = iz - 1
        iz1 = iz
        if iz0 < 0:
            iz0 += Nz
        # Calculate radial indices at which to add charge
        ir0 = ir - 1
        ir1 = min(ir, Nr - 1)
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)

        # Ruyten-corrected shape factor coefficient
        bn = beta_n[ir]

        # Calculate the currents
        # ----------------------
        J_r_m_scal = wj * (cos * vxj + sin * vyj) * exptheta_m
        J_t_m_scal = wj * (cos * vyj - sin * vxj) * exptheta_m
        J_z_m_scal = wj * vzj * exptheta_m

        J_t_m_00 = (
            Sr_linear(r_cell, 0, wp.float64(-f), bn) * Sz_linear(z_cell, 0) * J_t_m_scal
        )
        J_r_m_00 = (
            Sr_linear(r_cell, 0, wp.float64(-f), bn) * Sz_linear(z_cell, 0) * J_r_m_scal
        )
        J_z_m_00 = (
            Sr_linear(r_cell, 0, wp.float64(f), bn) * Sz_linear(z_cell, 0) * J_z_m_scal
        )
        J_r_m_01 = (
            Sr_linear(r_cell, 0, wp.float64(-f), bn) * Sz_linear(z_cell, 1) * J_r_m_scal
        )
        J_t_m_01 = (
            Sr_linear(r_cell, 0, wp.float64(-f), bn) * Sz_linear(z_cell, 1) * J_t_m_scal
        )
        J_z_m_01 = (
            Sr_linear(r_cell, 0, wp.float64(f), bn) * Sz_linear(z_cell, 1) * J_z_m_scal
        )

        J_r_m_10 = (
            Sr_linear(r_cell, 1, wp.float64(-f), bn) * Sz_linear(z_cell, 0) * J_r_m_scal
        )
        J_t_m_10 = (
            Sr_linear(r_cell, 1, wp.float64(-f), bn) * Sz_linear(z_cell, 0) * J_t_m_scal
        )
        J_z_m_10 = (
            Sr_linear(r_cell, 1, wp.float64(f), bn) * Sz_linear(z_cell, 0) * J_z_m_scal
        )
        J_r_m_11 = (
            Sr_linear(r_cell, 1, wp.float64(-f), bn) * Sz_linear(z_cell, 1) * J_r_m_scal
        )
        J_t_m_11 = (
            Sr_linear(r_cell, 1, wp.float64(-f), bn) * Sz_linear(z_cell, 1) * J_t_m_scal
        )
        J_z_m_11 = (
            Sr_linear(r_cell, 1, wp.float64(f), bn) * Sz_linear(z_cell, 1) * J_z_m_scal
        )

        # Atomically add the registers to global memory
        # jr
        wp.atomic_add(j_r_m, iz0, ir0, wp.vec2d(J_r_m_00[0], wp.float64(0.0)))
        wp.atomic_add(j_r_m, iz0, ir1, wp.vec2d(J_r_m_10[0], wp.float64(0.0)))
        wp.atomic_add(j_r_m, iz1, ir0, wp.vec2d(J_r_m_01[0], wp.float64(0.0)))
        wp.atomic_add(j_r_m, iz1, ir1, wp.vec2d(J_r_m_11[0], wp.float64(0.0)))
        if m > 0:
            wp.atomic_add(j_r_m, iz0, ir0, wp.vec2d(wp.float64(0.0), J_r_m_00[1]))
            wp.atomic_add(j_r_m, iz0, ir1, wp.vec2d(wp.float64(0.0), J_r_m_10[1]))
            wp.atomic_add(j_r_m, iz1, ir0, wp.vec2d(wp.float64(0.0), J_r_m_01[1]))
            wp.atomic_add(j_r_m, iz1, ir1, wp.vec2d(wp.float64(0.0), J_r_m_11[1]))
        # jt
        wp.atomic_add(j_t_m, iz0, ir0, wp.vec2d(J_t_m_00[0], wp.float64(0.0)))
        wp.atomic_add(j_t_m, iz0, ir1, wp.vec2d(J_t_m_10[0], wp.float64(0.0)))
        wp.atomic_add(j_t_m, iz1, ir0, wp.vec2d(J_t_m_01[0], wp.float64(0.0)))
        wp.atomic_add(j_t_m, iz1, ir1, wp.vec2d(J_t_m_11[0], wp.float64(0.0)))
        if m > 0:
            wp.atomic_add(j_t_m, iz0, ir0, wp.vec2d(wp.float64(0.0), J_t_m_00[1]))
            wp.atomic_add(j_t_m, iz0, ir1, wp.vec2d(wp.float64(0.0), J_t_m_10[1]))
            wp.atomic_add(j_t_m, iz1, ir0, wp.vec2d(wp.float64(0.0), J_t_m_01[1]))
            wp.atomic_add(j_t_m, iz1, ir1, wp.vec2d(wp.float64(0.0), J_t_m_11[1]))
        # jz
        wp.atomic_add(j_z_m, iz0, ir0, wp.vec2d(J_z_m_00[0], wp.float64(0.0)))
        wp.atomic_add(j_z_m, iz0, ir1, wp.vec2d(J_z_m_10[0], wp.float64(0.0)))
        wp.atomic_add(j_z_m, iz1, ir0, wp.vec2d(J_z_m_01[0], wp.float64(0.0)))
        wp.atomic_add(j_z_m, iz1, ir1, wp.vec2d(J_z_m_11[0], wp.float64(0.0)))
        if m > 0:
            wp.atomic_add(j_z_m, iz0, ir0, wp.vec2d(wp.float64(0.0), J_z_m_00[1]))
            wp.atomic_add(j_z_m, iz0, ir1, wp.vec2d(wp.float64(0.0), J_z_m_10[1]))
            wp.atomic_add(j_z_m, iz1, ir0, wp.vec2d(wp.float64(0.0), J_z_m_01[1]))
            wp.atomic_add(j_z_m, iz1, ir1, wp.vec2d(wp.float64(0.0), J_z_m_11[1]))
