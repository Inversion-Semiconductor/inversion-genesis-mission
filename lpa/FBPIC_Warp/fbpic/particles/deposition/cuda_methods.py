# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen, Kevin Peters
# License: 3-Clause-BSD-LBNL
"""Warp kernels retain the baseline particle formulas and array layout."""
import warp as wp
from fbpic.utils.warp import warp_kernel

from scipy.constants import c
from .cuda_particle_shapes import Sz_linear, Sr_linear, Sz_cubic, Sr_cubic

wp.set_module_options({"enable_backward": False})


c = wp.constant(wp.float64(c))


@warp_kernel
def deposit_rho_gpu_linear(
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
    rho_m0: wp.array2d(dtype=wp.vec2d),
    rho_m1: wp.array2d(dtype=wp.vec2d),
    prefix_sum: wp.array(dtype=wp.int32),
    beta_n_m0: wp.array(dtype=wp.float64),
    beta_n_m1: wp.array(dtype=wp.float64),
):
    """
    Deposition of the charge density rho using Warp on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of rho that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 4 variables (one for each possible direction,
    e.g. upper in z, lower in r) to maintain parallelism while
    avoiding any race conditions.

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

    rho_m0, rho_m1 : 2darrays of complexs
        The charge density on the interpolation grid for
        mode 0 and 1. (is modified by this function)

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the considered direction

    Nz, Nr : int
        Number of gridpoints along the considered direction

    prefix_sum : 1darray of integers
        Represents the cumulative sum of
        the particles per cell

    beta_n_m0, beta_n_m1 : 1darrays of floats
        Ruyten-corrected particle shape factor coefficients for mode 0 and 1
    """
    # Get the 1D CUDA grid
    i = wp.tid()
    # Deposit the field per cell in parallel (for threads < number of cells)
    if i < prefix_sum.shape[0]:
        # Retrieve index of upper grid point (in z and r) from prefix-sum index
        # (See calculation of prefix-sum index in `get_cell_idx_per_particle`)
        iz_upper = wp.int32(i // (Nr + 1))
        ir_upper = wp.int32(i - iz_upper * (Nr + 1))
        # Calculate the inclusive offset for the current cell
        # It represents the number of particles contained in all other cells
        # with an index smaller than i + the total number of particles in the
        # current cell (inclusive).
        incl_offset = wp.int32(prefix_sum[i])
        # Calculate the frequency per cell from the offset and the previous
        # offset (prefix_sum[i-1]).
        if i > 0:
            frequency_per_cell = wp.int32(incl_offset - prefix_sum[i - 1])
        if i == 0:
            frequency_per_cell = wp.int32(incl_offset)

        # Declare local field arrays
        R_m0_00 = wp.float64(0.0)
        R_m0_01 = wp.float64(0.0)
        R_m0_10 = wp.float64(0.0)
        R_m0_11 = wp.float64(0.0)
        R_m1_00 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        R_m1_01 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        R_m1_10 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        R_m1_11 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        for j in range(frequency_per_cell):
            # Get the particle index before the sorting
            # --------------------------------------------
            # (Since incl_offset is a cumulative sum of particle number,
            # and since python index starts at 0, one has to add -1)
            ptcl_idx = incl_offset - 1 - j

            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
            # Position
            xj = x[ptcl_idx]
            yj = y[ptcl_idx]
            zj = z[ptcl_idx]
            # Weights
            wj = q * w[ptcl_idx]

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
            exptheta_m0 = wp.float64(1.0)
            exptheta_m1 = (
                wp.vec2d(cos, wp.float64(0.0))
                + wp.vec2d(wp.float64(0.0), wp.float64(1.0)) * sin
            )

            # Positions of the particles, in the cell unit
            r_cell = invdr * (rj - rmin) - wp.float64(0.5)
            z_cell = invdz * (zj - zmin) - wp.float64(0.5)

            # Ruyten-corrected shape factor coefficients for both modes
            ir = min(wp.int32(wp.ceil(r_cell)), Nr)
            bn_m0 = beta_n_m0[ir]
            bn_m1 = beta_n_m1[ir]

            # Calculate rho
            # --------------------------------------------
            # Mode 0
            R_m0_scal = wj * exptheta_m0
            # Mode 1
            R_m1_scal = wj * exptheta_m1

            R_m0_00 += (
                Sr_linear(r_cell, 0, wp.float64(1), bn_m0)
                * Sz_linear(z_cell, 0)
                * R_m0_scal
            )
            R_m0_01 += (
                Sr_linear(r_cell, 0, wp.float64(1), bn_m0)
                * Sz_linear(z_cell, 1)
                * R_m0_scal
            )
            R_m1_00 += (
                Sr_linear(r_cell, 0, wp.float64(-1), bn_m1)
                * Sz_linear(z_cell, 0)
                * R_m1_scal
            )
            R_m1_01 += (
                Sr_linear(r_cell, 0, wp.float64(-1), bn_m1)
                * Sz_linear(z_cell, 1)
                * R_m1_scal
            )
            R_m0_10 += (
                Sr_linear(r_cell, 1, wp.float64(1), bn_m0)
                * Sz_linear(z_cell, 0)
                * R_m0_scal
            )
            R_m0_11 += (
                Sr_linear(r_cell, 1, wp.float64(1), bn_m0)
                * Sz_linear(z_cell, 1)
                * R_m0_scal
            )
            R_m1_10 += (
                Sr_linear(r_cell, 1, wp.float64(-1), bn_m1)
                * Sz_linear(z_cell, 0)
                * R_m1_scal
            )
            R_m1_11 += (
                Sr_linear(r_cell, 1, wp.float64(-1), bn_m1)
                * Sz_linear(z_cell, 1)
                * R_m1_scal
            )

        # Calculate longitudinal indices at which to add charge
        iz0 = iz_upper - 1
        iz1 = iz_upper
        if iz0 < 0:
            iz0 += Nz
        # Calculate radial indices at which to add charge
        ir0 = ir_upper - 1
        ir1 = min(ir_upper, Nr - 1)
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)

        # Atomically add the registers to global memory
        if frequency_per_cell > 0:
            # Mode 0
            wp.atomic_add(rho_m0, iz0, ir0, wp.vec2d(R_m0_00, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz0, ir1, wp.vec2d(R_m0_10, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz1, ir0, wp.vec2d(R_m0_01, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz1, ir1, wp.vec2d(R_m0_11, wp.float64(0.0)))
            # Mode 1
            wp.atomic_add(rho_m1, iz0, ir0, wp.vec2d(R_m1_00[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz0, ir0, wp.vec2d(wp.float64(0.0), R_m1_00[1]))
            wp.atomic_add(rho_m1, iz0, ir1, wp.vec2d(R_m1_10[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz0, ir1, wp.vec2d(wp.float64(0.0), R_m1_10[1]))
            wp.atomic_add(rho_m1, iz1, ir0, wp.vec2d(R_m1_01[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz1, ir0, wp.vec2d(wp.float64(0.0), R_m1_01[1]))
            wp.atomic_add(rho_m1, iz1, ir1, wp.vec2d(R_m1_11[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz1, ir1, wp.vec2d(wp.float64(0.0), R_m1_11[1]))


# -------------------------------
# Field deposition - linear - J
# -------------------------------


@warp_kernel
def deposit_J_gpu_linear(
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    z: wp.array(dtype=wp.float64),
    w: wp.array(dtype=wp.float64),
    q: wp.float64,
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    inv_gamma: wp.array(dtype=wp.float64),
    invdz: wp.float64,
    zmin: wp.float64,
    Nz: wp.int32,
    invdr: wp.float64,
    rmin: wp.float64,
    Nr: wp.int32,
    j_r_m0: wp.array2d(dtype=wp.vec2d),
    j_r_m1: wp.array2d(dtype=wp.vec2d),
    j_t_m0: wp.array2d(dtype=wp.vec2d),
    j_t_m1: wp.array2d(dtype=wp.vec2d),
    j_z_m0: wp.array2d(dtype=wp.vec2d),
    j_z_m1: wp.array2d(dtype=wp.vec2d),
    prefix_sum: wp.array(dtype=wp.int32),
    beta_n_m0: wp.array(dtype=wp.float64),
    beta_n_m1: wp.array(dtype=wp.float64),
):
    """
    Deposition of the current J using Warp on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of J that is deposited to the
    4 cells surounding the particle based on its shape (linear).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 4 variables (one for each possible direction,
    e.g. upper in z, lower in r) to maintain parallelism while
    avoiding any race conditions.

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

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    j_r_m0, j_r_m1, j_t_m0, j_t_m1, j_z_m0, j_z_m1,: 2darrays of complexs
        The current component in each direction (r, t, z)
        on the interpolation grid for mode 0 and 1.
        (is modified by this function)

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    prefix_sum : 1darray of integers
        Represents the cumulative sum of
        the particles per cell

    beta_n_m0, beta_n_m1 : 1darrays of floats
        Ruyten-corrected particle shape factor coefficients for mode 0 and 1
    """
    # Get the 1D CUDA grid
    i = wp.tid()
    # Deposit the field per cell in parallel (for threads < number of cells)
    if i < prefix_sum.shape[0]:
        # Retrieve index of upper grid point (in z and r) from prefix-sum index
        # (See calculation of prefix-sum index in `get_cell_idx_per_particle`)
        iz_upper = wp.int32(i // (Nr + 1))
        ir_upper = wp.int32(i - iz_upper * (Nr + 1))
        # Calculate the inclusive offset for the current cell
        # It represents the number of particles contained in all other cells
        # with an index smaller than i + the total number of particles in the
        # current cell (inclusive).
        incl_offset = wp.int32(prefix_sum[i])
        # Calculate the frequency per cell from the offset and the previous
        # offset (prefix_sum[i-1]).
        if i > 0:
            frequency_per_cell = wp.int32(incl_offset - prefix_sum[i - 1])
        if i == 0:
            frequency_per_cell = wp.int32(incl_offset)

        # Declare the local field value for
        # all possible deposition directions,
        # depending on the shape order and per mode for r,t and z.

        J_r_m0_00 = wp.float64(0.0)
        J_r_m1_00 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m0_00 = wp.float64(0.0)
        J_t_m1_00 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m0_00 = wp.float64(0.0)
        J_z_m1_00 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_01 = wp.float64(0.0)
        J_r_m1_01 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m0_01 = wp.float64(0.0)
        J_t_m1_01 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m0_01 = wp.float64(0.0)
        J_z_m1_01 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_10 = wp.float64(0.0)
        J_r_m1_10 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m0_10 = wp.float64(0.0)
        J_t_m1_10 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m0_10 = wp.float64(0.0)
        J_z_m1_10 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_11 = wp.float64(0.0)
        J_r_m1_11 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m0_11 = wp.float64(0.0)
        J_t_m1_11 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m0_11 = wp.float64(0.0)
        J_z_m1_11 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        # Loop over the number of particles per cell
        for j in range(frequency_per_cell):
            # Get the particle index
            # ----------------------
            # (Since incl_offset is a cumulative sum of particle number,
            # and since python index starts at 0, one has to add -1)
            ptcl_idx = incl_offset - 1 - j

            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
            # Position
            xj = x[ptcl_idx]
            yj = y[ptcl_idx]
            zj = z[ptcl_idx]
            # Velocity
            uxj = ux[ptcl_idx]
            uyj = uy[ptcl_idx]
            uzj = uz[ptcl_idx]
            # Inverse gamma
            inv_gammaj = inv_gamma[ptcl_idx]
            # Weights
            wj = q * w[ptcl_idx]

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
            exptheta_m0 = wp.float64(1.0)
            exptheta_m1 = (
                wp.vec2d(cos, wp.float64(0.0))
                + wp.vec2d(wp.float64(0.0), wp.float64(1.0)) * sin
            )

            # Get weights for the deposition
            # --------------------------------------------
            # Positions of the particles, in the cell unit
            r_cell = invdr * (rj - rmin) - wp.float64(0.5)
            z_cell = invdz * (zj - zmin) - wp.float64(0.5)

            # Ruyten-corrected shape factor coefficients for both modes
            ir = min(wp.int32(wp.ceil(r_cell)), Nr)
            bn_m0 = beta_n_m0[ir]
            bn_m1 = beta_n_m1[ir]

            # Calculate the currents
            # ----------------------
            # Mode 0
            J_r_m0_scal = wj * c * inv_gammaj * (cos * uxj + sin * uyj) * exptheta_m0
            J_t_m0_scal = wj * c * inv_gammaj * (cos * uyj - sin * uxj) * exptheta_m0
            J_z_m0_scal = wj * c * inv_gammaj * uzj * exptheta_m0
            # Mode 1
            J_r_m1_scal = wj * c * inv_gammaj * (cos * uxj + sin * uyj) * exptheta_m1
            J_t_m1_scal = wj * c * inv_gammaj * (cos * uyj - sin * uxj) * exptheta_m1
            J_z_m1_scal = wj * c * inv_gammaj * uzj * exptheta_m1

            J_r_m0_00 += (
                Sr_linear(r_cell, 0, wp.float64(-1), bn_m0)
                * Sz_linear(z_cell, 0)
                * J_r_m0_scal
            )
            J_t_m0_00 += (
                Sr_linear(r_cell, 0, wp.float64(-1), bn_m0)
                * Sz_linear(z_cell, 0)
                * J_t_m0_scal
            )
            J_z_m0_00 += (
                Sr_linear(r_cell, 0, wp.float64(1), bn_m0)
                * Sz_linear(z_cell, 0)
                * J_z_m0_scal
            )
            J_r_m0_01 += (
                Sr_linear(r_cell, 0, wp.float64(-1), bn_m0)
                * Sz_linear(z_cell, 1)
                * J_r_m0_scal
            )
            J_t_m0_01 += (
                Sr_linear(r_cell, 0, wp.float64(-1), bn_m0)
                * Sz_linear(z_cell, 1)
                * J_t_m0_scal
            )
            J_z_m0_01 += (
                Sr_linear(r_cell, 0, wp.float64(1), bn_m0)
                * Sz_linear(z_cell, 1)
                * J_z_m0_scal
            )
            J_r_m1_00 += (
                Sr_linear(r_cell, 0, wp.float64(1), bn_m1)
                * Sz_linear(z_cell, 0)
                * J_r_m1_scal
            )
            J_t_m1_00 += (
                Sr_linear(r_cell, 0, wp.float64(1), bn_m1)
                * Sz_linear(z_cell, 0)
                * J_t_m1_scal
            )
            J_z_m1_00 += (
                Sr_linear(r_cell, 0, wp.float64(-1), bn_m1)
                * Sz_linear(z_cell, 0)
                * J_z_m1_scal
            )
            J_r_m1_01 += (
                Sr_linear(r_cell, 0, wp.float64(1), bn_m1)
                * Sz_linear(z_cell, 1)
                * J_r_m1_scal
            )
            J_t_m1_01 += (
                Sr_linear(r_cell, 0, wp.float64(1), bn_m1)
                * Sz_linear(z_cell, 1)
                * J_t_m1_scal
            )
            J_z_m1_01 += (
                Sr_linear(r_cell, 0, wp.float64(-1), bn_m1)
                * Sz_linear(z_cell, 1)
                * J_z_m1_scal
            )

            J_r_m0_10 += (
                Sr_linear(r_cell, 1, wp.float64(-1), bn_m0)
                * Sz_linear(z_cell, 0)
                * J_r_m0_scal
            )
            J_t_m0_10 += (
                Sr_linear(r_cell, 1, wp.float64(-1), bn_m0)
                * Sz_linear(z_cell, 0)
                * J_t_m0_scal
            )
            J_z_m0_10 += (
                Sr_linear(r_cell, 1, wp.float64(1), bn_m0)
                * Sz_linear(z_cell, 0)
                * J_z_m0_scal
            )
            J_r_m0_11 += (
                Sr_linear(r_cell, 1, wp.float64(-1), bn_m0)
                * Sz_linear(z_cell, 1)
                * J_r_m0_scal
            )
            J_t_m0_11 += (
                Sr_linear(r_cell, 1, wp.float64(-1), bn_m0)
                * Sz_linear(z_cell, 1)
                * J_t_m0_scal
            )
            J_z_m0_11 += (
                Sr_linear(r_cell, 1, wp.float64(1), bn_m0)
                * Sz_linear(z_cell, 1)
                * J_z_m0_scal
            )
            J_r_m1_10 += (
                Sr_linear(r_cell, 1, wp.float64(1), bn_m1)
                * Sz_linear(z_cell, 0)
                * J_r_m1_scal
            )
            J_t_m1_10 += (
                Sr_linear(r_cell, 1, wp.float64(1), bn_m1)
                * Sz_linear(z_cell, 0)
                * J_t_m1_scal
            )
            J_z_m1_10 += (
                Sr_linear(r_cell, 1, wp.float64(-1), bn_m1)
                * Sz_linear(z_cell, 0)
                * J_z_m1_scal
            )
            J_r_m1_11 += (
                Sr_linear(r_cell, 1, wp.float64(1), bn_m1)
                * Sz_linear(z_cell, 1)
                * J_r_m1_scal
            )
            J_t_m1_11 += (
                Sr_linear(r_cell, 1, wp.float64(1), bn_m1)
                * Sz_linear(z_cell, 1)
                * J_t_m1_scal
            )
            J_z_m1_11 += (
                Sr_linear(r_cell, 1, wp.float64(-1), bn_m1)
                * Sz_linear(z_cell, 1)
                * J_z_m1_scal
            )

        # Calculate longitudinal indices at which to add charge
        iz0 = iz_upper - 1
        iz1 = iz_upper
        if iz0 < 0:
            iz0 += Nz
        # Calculate radial indices at which to add charge
        ir0 = ir_upper - 1
        ir1 = min(ir_upper, Nr - 1)
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)

        # Atomically add the registers to global memory
        if frequency_per_cell > 0:
            # jr: Mode 0
            wp.atomic_add(j_r_m0, iz0, ir0, wp.vec2d(J_r_m0_00, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz0, ir1, wp.vec2d(J_r_m0_10, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz1, ir0, wp.vec2d(J_r_m0_01, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz1, ir1, wp.vec2d(J_r_m0_11, wp.float64(0.0)))
            # jr: Mode 1
            wp.atomic_add(j_r_m1, iz0, ir0, wp.vec2d(J_r_m1_00[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz0, ir0, wp.vec2d(wp.float64(0.0), J_r_m1_00[1]))
            wp.atomic_add(j_r_m1, iz0, ir1, wp.vec2d(J_r_m1_10[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz0, ir1, wp.vec2d(wp.float64(0.0), J_r_m1_10[1]))
            wp.atomic_add(j_r_m1, iz1, ir0, wp.vec2d(J_r_m1_01[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz1, ir0, wp.vec2d(wp.float64(0.0), J_r_m1_01[1]))
            wp.atomic_add(j_r_m1, iz1, ir1, wp.vec2d(J_r_m1_11[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz1, ir1, wp.vec2d(wp.float64(0.0), J_r_m1_11[1]))
            # jt: Mode 0
            wp.atomic_add(j_t_m0, iz0, ir0, wp.vec2d(J_t_m0_00, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz0, ir1, wp.vec2d(J_t_m0_10, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz1, ir0, wp.vec2d(J_t_m0_01, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz1, ir1, wp.vec2d(J_t_m0_11, wp.float64(0.0)))
            # jt: Mode 1
            wp.atomic_add(j_t_m1, iz0, ir0, wp.vec2d(J_t_m1_00[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz0, ir0, wp.vec2d(wp.float64(0.0), J_t_m1_00[1]))
            wp.atomic_add(j_t_m1, iz0, ir1, wp.vec2d(J_t_m1_10[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz0, ir1, wp.vec2d(wp.float64(0.0), J_t_m1_10[1]))
            wp.atomic_add(j_t_m1, iz1, ir0, wp.vec2d(J_t_m1_01[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz1, ir0, wp.vec2d(wp.float64(0.0), J_t_m1_01[1]))
            wp.atomic_add(j_t_m1, iz1, ir1, wp.vec2d(J_t_m1_11[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz1, ir1, wp.vec2d(wp.float64(0.0), J_t_m1_11[1]))
            # jz: Mode 0
            wp.atomic_add(j_z_m0, iz0, ir0, wp.vec2d(J_z_m0_00, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz0, ir1, wp.vec2d(J_z_m0_10, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz1, ir0, wp.vec2d(J_z_m0_01, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz1, ir1, wp.vec2d(J_z_m0_11, wp.float64(0.0)))
            # jz: Mode 1
            wp.atomic_add(j_z_m1, iz0, ir0, wp.vec2d(J_z_m1_00[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz0, ir0, wp.vec2d(wp.float64(0.0), J_z_m1_00[1]))
            wp.atomic_add(j_z_m1, iz0, ir1, wp.vec2d(J_z_m1_10[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz0, ir1, wp.vec2d(wp.float64(0.0), J_z_m1_10[1]))
            wp.atomic_add(j_z_m1, iz1, ir0, wp.vec2d(J_z_m1_01[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz1, ir0, wp.vec2d(wp.float64(0.0), J_z_m1_01[1]))
            wp.atomic_add(j_z_m1, iz1, ir1, wp.vec2d(J_z_m1_11[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz1, ir1, wp.vec2d(wp.float64(0.0), J_z_m1_11[1]))


# -------------------------------
# Field deposition - cubic - rho
# -------------------------------


@warp_kernel
def deposit_rho_gpu_cubic(
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
    rho_m0: wp.array2d(dtype=wp.vec2d),
    rho_m1: wp.array2d(dtype=wp.vec2d),
    prefix_sum: wp.array(dtype=wp.int32),
    beta_n_m0: wp.array(dtype=wp.float64),
    beta_n_m1: wp.array(dtype=wp.float64),
):
    """
    Deposition of the charge density rho using Warp on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of rho that is deposited to the
    16 cells surounding the particle based on its shape (cubic).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 16 variables (one for each surrounding cell) to
    maintain parallelism while avoiding any race conditions.

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

    rho_m0, rho_m1 : 2darrays of complexs
        The charge density on the interpolation grid for
        mode 0 and 1. (is modified by this function)

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the considered direction

    Nz, Nr : int
        Number of gridpoints along the considered direction

    prefix_sum : 1darray of integers
        Represents the cumulative sum of
        the particles per cell

    beta_n_m0, beta_n_m1 : 1darrays of floats
        Ruyten-corrected particle shape factor coefficients for mode 0 and 1
    """
    # Get the 1D CUDA grid
    i = wp.tid()
    # Deposit the field per cell in parallel (for threads < number of cells)
    if i < prefix_sum.shape[0]:
        # Retrieve index of upper grid point (in z and r) from prefix-sum index
        # (See calculation of prefix-sum index in `get_cell_idx_per_particle`)
        iz_upper = wp.int32(i // (Nr + 1))
        ir_upper = wp.int32(i - iz_upper * (Nr + 1))
        # Calculate the inclusive offset for the current cell
        # It represents the number of particles contained in all other cells
        # with an index smaller than i + the total number of particles in the
        # current cell (inclusive).
        incl_offset = wp.int32(prefix_sum[i])
        # Calculate the frequency per cell from the offset and the previous
        # offset (prefix_sum[i-1]).
        if i > 0:
            frequency_per_cell = wp.int32(incl_offset - prefix_sum[i - 1])
        if i == 0:
            frequency_per_cell = wp.int32(incl_offset)

        # Declare local field arrays
        R_m0_00 = wp.float64(0.0)
        R_m1_00 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_01 = wp.float64(0.0)
        R_m1_01 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_02 = wp.float64(0.0)
        R_m1_02 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_03 = wp.float64(0.0)
        R_m1_03 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_10 = wp.float64(0.0)
        R_m1_10 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_11 = wp.float64(0.0)
        R_m1_11 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_12 = wp.float64(0.0)
        R_m1_12 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_13 = wp.float64(0.0)
        R_m1_13 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_20 = wp.float64(0.0)
        R_m1_20 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_21 = wp.float64(0.0)
        R_m1_21 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_22 = wp.float64(0.0)
        R_m1_22 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_23 = wp.float64(0.0)
        R_m1_23 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_30 = wp.float64(0.0)
        R_m1_30 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_31 = wp.float64(0.0)
        R_m1_31 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_32 = wp.float64(0.0)
        R_m1_32 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        R_m0_33 = wp.float64(0.0)
        R_m1_33 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        for j in range(frequency_per_cell):
            # Get the particle index before the sorting
            # --------------------------------------------
            # (Since incl_offset is a cumulative sum of particle number,
            # and since python index starts at 0, one has to add -1)
            ptcl_idx = incl_offset - 1 - j

            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
            # Position
            xj = x[ptcl_idx]
            yj = y[ptcl_idx]
            zj = z[ptcl_idx]
            # Weights
            wj = q * w[ptcl_idx]

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
            exptheta_m0 = wp.float64(1.0)
            exptheta_m1 = (
                wp.vec2d(cos, wp.float64(0.0))
                + wp.vec2d(wp.float64(0.0), wp.float64(1.0)) * sin
            )

            # Positions of the particles, in the cell unit
            r_cell = invdr * (rj - rmin) - wp.float64(0.5)
            z_cell = invdz * (zj - zmin) - wp.float64(0.5)

            # Ruyten-corrected shape factor coefficients for both modes
            ir = min(wp.int32(wp.ceil(r_cell)), Nr)
            bn_m0 = beta_n_m0[ir]
            bn_m1 = beta_n_m1[ir]

            # Calculate rho
            # -------------
            # Mode 0
            R_m0_scal = wj * exptheta_m0
            # Mode 1
            R_m1_scal = wj * exptheta_m1

            R_m0_00 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * R_m0_scal
            )
            R_m1_00 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * R_m1_scal
            )
            R_m0_01 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * R_m0_scal
            )
            R_m1_01 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * R_m1_scal
            )
            R_m0_02 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * R_m0_scal
            )
            R_m1_02 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * R_m1_scal
            )
            R_m0_03 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * R_m0_scal
            )
            R_m1_03 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * R_m1_scal
            )

            R_m0_10 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * R_m0_scal
            )
            R_m1_10 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * R_m1_scal
            )
            R_m0_11 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * R_m0_scal
            )
            R_m1_11 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * R_m1_scal
            )
            R_m0_12 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * R_m0_scal
            )
            R_m1_12 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * R_m1_scal
            )
            R_m0_13 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * R_m0_scal
            )
            R_m1_13 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * R_m1_scal
            )

            R_m0_20 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * R_m0_scal
            )
            R_m1_20 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * R_m1_scal
            )
            R_m0_21 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * R_m0_scal
            )
            R_m1_21 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * R_m1_scal
            )
            R_m0_22 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * R_m0_scal
            )
            R_m1_22 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * R_m1_scal
            )
            R_m0_23 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * R_m0_scal
            )
            R_m1_23 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * R_m1_scal
            )

            R_m0_30 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * R_m0_scal
            )
            R_m1_30 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * R_m1_scal
            )
            R_m0_31 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * R_m0_scal
            )
            R_m1_31 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * R_m1_scal
            )
            R_m0_32 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * R_m0_scal
            )
            R_m1_32 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * R_m1_scal
            )
            R_m0_33 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * R_m0_scal
            )
            R_m1_33 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * R_m1_scal
            )

        # Calculate longitudinal indices at which to add charge
        iz0 = iz_upper - 2
        iz1 = iz_upper - 1
        iz2 = iz_upper
        iz3 = iz_upper + 1
        if iz0 < 0:
            iz0 += Nz
        if iz1 < 0:
            iz1 += Nz
        if iz3 > Nz - 1:
            iz3 -= Nz
        # Calculate radial indices at which to add charge
        ir0 = ir_upper - 2
        ir1 = min(ir_upper - 1, Nr - 1)
        ir2 = min(ir_upper, Nr - 1)
        ir3 = min(ir_upper + 1, Nr - 1)
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)
        if ir1 < 0:
            # Deposition below the axis: fold index into physical region
            ir1 = -(1 + ir1)

        # Atomically add the registers to global memory
        if frequency_per_cell > 0:
            # Mode 0
            wp.atomic_add(rho_m0, iz0, ir0, wp.vec2d(R_m0_00, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz0, ir1, wp.vec2d(R_m0_10, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz0, ir2, wp.vec2d(R_m0_20, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz0, ir3, wp.vec2d(R_m0_30, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz1, ir0, wp.vec2d(R_m0_01, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz1, ir1, wp.vec2d(R_m0_11, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz1, ir2, wp.vec2d(R_m0_21, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz1, ir3, wp.vec2d(R_m0_31, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz2, ir0, wp.vec2d(R_m0_02, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz2, ir1, wp.vec2d(R_m0_12, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz2, ir2, wp.vec2d(R_m0_22, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz2, ir3, wp.vec2d(R_m0_32, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz3, ir0, wp.vec2d(R_m0_03, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz3, ir1, wp.vec2d(R_m0_13, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz3, ir2, wp.vec2d(R_m0_23, wp.float64(0.0)))
            wp.atomic_add(rho_m0, iz3, ir3, wp.vec2d(R_m0_33, wp.float64(0.0)))
            # Mode 1
            wp.atomic_add(rho_m1, iz0, ir0, wp.vec2d(R_m1_00[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz0, ir0, wp.vec2d(wp.float64(0.0), R_m1_00[1]))
            wp.atomic_add(rho_m1, iz0, ir1, wp.vec2d(R_m1_10[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz0, ir1, wp.vec2d(wp.float64(0.0), R_m1_10[1]))
            wp.atomic_add(rho_m1, iz0, ir2, wp.vec2d(R_m1_20[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz0, ir2, wp.vec2d(wp.float64(0.0), R_m1_20[1]))
            wp.atomic_add(rho_m1, iz0, ir3, wp.vec2d(R_m1_30[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz0, ir3, wp.vec2d(wp.float64(0.0), R_m1_30[1]))
            wp.atomic_add(rho_m1, iz1, ir0, wp.vec2d(R_m1_01[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz1, ir0, wp.vec2d(wp.float64(0.0), R_m1_01[1]))
            wp.atomic_add(rho_m1, iz1, ir1, wp.vec2d(R_m1_11[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz1, ir1, wp.vec2d(wp.float64(0.0), R_m1_11[1]))
            wp.atomic_add(rho_m1, iz1, ir2, wp.vec2d(R_m1_21[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz1, ir2, wp.vec2d(wp.float64(0.0), R_m1_21[1]))
            wp.atomic_add(rho_m1, iz1, ir3, wp.vec2d(R_m1_31[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz1, ir3, wp.vec2d(wp.float64(0.0), R_m1_31[1]))
            wp.atomic_add(rho_m1, iz2, ir0, wp.vec2d(R_m1_02[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz2, ir0, wp.vec2d(wp.float64(0.0), R_m1_02[1]))
            wp.atomic_add(rho_m1, iz2, ir1, wp.vec2d(R_m1_12[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz2, ir1, wp.vec2d(wp.float64(0.0), R_m1_12[1]))
            wp.atomic_add(rho_m1, iz2, ir2, wp.vec2d(R_m1_22[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz2, ir2, wp.vec2d(wp.float64(0.0), R_m1_22[1]))
            wp.atomic_add(rho_m1, iz2, ir3, wp.vec2d(R_m1_32[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz2, ir3, wp.vec2d(wp.float64(0.0), R_m1_32[1]))
            wp.atomic_add(rho_m1, iz3, ir0, wp.vec2d(R_m1_03[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz3, ir0, wp.vec2d(wp.float64(0.0), R_m1_03[1]))
            wp.atomic_add(rho_m1, iz3, ir1, wp.vec2d(R_m1_13[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz3, ir1, wp.vec2d(wp.float64(0.0), R_m1_13[1]))
            wp.atomic_add(rho_m1, iz3, ir2, wp.vec2d(R_m1_23[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz3, ir2, wp.vec2d(wp.float64(0.0), R_m1_23[1]))
            wp.atomic_add(rho_m1, iz3, ir3, wp.vec2d(R_m1_33[0], wp.float64(0.0)))
            wp.atomic_add(rho_m1, iz3, ir3, wp.vec2d(wp.float64(0.0), R_m1_33[1]))


# -------------------------------
# Field deposition - cubic - J
# -------------------------------


@warp_kernel
def deposit_J_gpu_cubic(
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    z: wp.array(dtype=wp.float64),
    w: wp.array(dtype=wp.float64),
    q: wp.float64,
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    inv_gamma: wp.array(dtype=wp.float64),
    invdz: wp.float64,
    zmin: wp.float64,
    Nz: wp.int32,
    invdr: wp.float64,
    rmin: wp.float64,
    Nr: wp.int32,
    j_r_m0: wp.array2d(dtype=wp.vec2d),
    j_r_m1: wp.array2d(dtype=wp.vec2d),
    j_t_m0: wp.array2d(dtype=wp.vec2d),
    j_t_m1: wp.array2d(dtype=wp.vec2d),
    j_z_m0: wp.array2d(dtype=wp.vec2d),
    j_z_m1: wp.array2d(dtype=wp.vec2d),
    prefix_sum: wp.array(dtype=wp.int32),
    beta_n_m0: wp.array(dtype=wp.float64),
    beta_n_m1: wp.array(dtype=wp.float64),
):
    """
    Deposition of the current J using Warp on the GPU.
    Iterates over the cells and over the particles per cell.
    Calculates the weighted amount of J that is deposited to the
    16 cells surounding the particle based on its shape (cubic).

    The particles are sorted by their cell index (the lower cell
    in r and z that they deposit to) and the deposited field
    is split into 16 variables (one for each cell) to maintain
    parallelism while avoiding any race conditions.

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

    ux, uy, uz : 1darray of floats (in meters * second^-1)
        The velocity of the particles

    inv_gamma : 1darray of floats
        The inverse of the relativistic gamma factor

    j_r_m0, j_r_m1, j_t_m0, j_t_m1, j_z_m0, j_z_m1,: 2darrays of complexs
        The current component in each direction (r, t, z)
        on the interpolation grid for mode 0 and 1.
        (is modified by this function)

    invdz, invdr : float (in meters^-1)
        Inverse of the grid step along the considered direction

    zmin, rmin : float (in meters)
        Position of the edge of the simulation box,
        along the direction considered

    Nz, Nr : int
        Number of gridpoints along the considered direction

    prefix_sum : 1darray of integers
        Represents the cumulative sum of
        the particles per cell

    beta_n_m0, beta_n_m1 : 1darrays of floats
        Ruyten-corrected particle shape factor coefficients for mode 0 and 1
    """
    # Get the 1D CUDA grid
    i = wp.tid()
    # Deposit the field per cell in parallel (for threads < number of cells)
    if i < prefix_sum.shape[0]:
        # Retrieve index of upper grid point (in z and r) from prefix-sum index
        # (See calculation of prefix-sum index in `get_cell_idx_per_particle`)
        iz_upper = wp.int32(i // (Nr + 1))
        ir_upper = wp.int32(i - iz_upper * (Nr + 1))
        # Calculate the inclusive offset for the current cell
        # It represents the number of particles contained in all other cells
        # with an index smaller than i + the total number of particles in the
        # current cell (inclusive).
        incl_offset = wp.int32(prefix_sum[i])
        # Calculate the frequency per cell from the offset and the previous
        # offset (prefix_sum[i-1]).
        if i > 0:
            frequency_per_cell = wp.int32(incl_offset - prefix_sum[i - 1])
        if i == 0:
            frequency_per_cell = wp.int32(incl_offset)

        # Declare the local field value for
        # all possible deposition directions,
        # depending on the shape order and per mode for r,t and z.
        J_r_m0_00 = wp.float64(0.0)
        J_t_m0_00 = wp.float64(0.0)
        J_z_m0_00 = wp.float64(0.0)
        J_r_m1_00 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_00 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_00 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_01 = wp.float64(0.0)
        J_t_m0_01 = wp.float64(0.0)
        J_z_m0_01 = wp.float64(0.0)
        J_r_m1_01 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_01 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_01 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_02 = wp.float64(0.0)
        J_t_m0_02 = wp.float64(0.0)
        J_z_m0_02 = wp.float64(0.0)
        J_r_m1_02 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_02 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_02 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_03 = wp.float64(0.0)
        J_t_m0_03 = wp.float64(0.0)
        J_z_m0_03 = wp.float64(0.0)
        J_r_m1_03 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_03 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_03 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_10 = wp.float64(0.0)
        J_t_m0_10 = wp.float64(0.0)
        J_z_m0_10 = wp.float64(0.0)
        J_r_m1_10 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_10 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_10 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_11 = wp.float64(0.0)
        J_t_m0_11 = wp.float64(0.0)
        J_z_m0_11 = wp.float64(0.0)
        J_r_m1_11 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_11 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_11 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_12 = wp.float64(0.0)
        J_t_m0_12 = wp.float64(0.0)
        J_z_m0_12 = wp.float64(0.0)
        J_r_m1_12 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_12 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_12 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_13 = wp.float64(0.0)
        J_t_m0_13 = wp.float64(0.0)
        J_z_m0_13 = wp.float64(0.0)
        J_r_m1_13 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_13 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_13 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_20 = wp.float64(0.0)
        J_t_m0_20 = wp.float64(0.0)
        J_z_m0_20 = wp.float64(0.0)
        J_r_m1_20 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_20 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_20 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_21 = wp.float64(0.0)
        J_t_m0_21 = wp.float64(0.0)
        J_z_m0_21 = wp.float64(0.0)
        J_r_m1_21 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_21 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_21 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_22 = wp.float64(0.0)
        J_t_m0_22 = wp.float64(0.0)
        J_z_m0_22 = wp.float64(0.0)
        J_r_m1_22 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_22 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_22 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_23 = wp.float64(0.0)
        J_t_m0_23 = wp.float64(0.0)
        J_z_m0_23 = wp.float64(0.0)
        J_r_m1_23 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_23 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_23 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_30 = wp.float64(0.0)
        J_t_m0_30 = wp.float64(0.0)
        J_z_m0_30 = wp.float64(0.0)
        J_r_m1_30 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_30 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_30 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_31 = wp.float64(0.0)
        J_t_m0_31 = wp.float64(0.0)
        J_z_m0_31 = wp.float64(0.0)
        J_r_m1_31 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_31 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_31 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_32 = wp.float64(0.0)
        J_t_m0_32 = wp.float64(0.0)
        J_z_m0_32 = wp.float64(0.0)
        J_r_m1_32 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_32 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_32 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        J_r_m0_33 = wp.float64(0.0)
        J_t_m0_33 = wp.float64(0.0)
        J_z_m0_33 = wp.float64(0.0)
        J_r_m1_33 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_t_m1_33 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )
        J_z_m1_33 = wp.vec2d(wp.float64(0.0), wp.float64(0.0)) + wp.vec2d(
            wp.float64(0.0), wp.float64(0.0)
        )

        # Loop over the number of particles per cell
        for j in range(frequency_per_cell):
            # Get the particle index
            # ----------------------
            # (Since incl_offset is a cumulative sum of particle number,
            # and since python index starts at 0, one has to add -1)
            ptcl_idx = incl_offset - 1 - j

            # Preliminary arrays for the cylindrical conversion
            # --------------------------------------------
            # Position
            xj = x[ptcl_idx]
            yj = y[ptcl_idx]
            zj = z[ptcl_idx]
            # Velocity
            uxj = ux[ptcl_idx]
            uyj = uy[ptcl_idx]
            uzj = uz[ptcl_idx]
            # Inverse gamma
            inv_gammaj = inv_gamma[ptcl_idx]
            # Weights
            wj = q * w[ptcl_idx]

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
            exptheta_m0 = wp.float64(1.0)
            exptheta_m1 = (
                wp.vec2d(cos, wp.float64(0.0))
                + wp.vec2d(wp.float64(0.0), wp.float64(1.0)) * sin
            )

            # Get weights for the deposition
            # --------------------------------------------
            # Positions of the particles, in the cell unit
            r_cell = invdr * (rj - rmin) - wp.float64(0.5)
            z_cell = invdz * (zj - zmin) - wp.float64(0.5)

            # Ruyten-corrected shape factor coefficients for both modes
            ir = min(wp.int32(wp.ceil(r_cell)), Nr)
            bn_m0 = beta_n_m0[ir]
            bn_m1 = beta_n_m1[ir]

            # Calculate the currents
            # --------------------------------------------
            # Mode 0
            J_r_m0_scal = wj * c * inv_gammaj * (cos * uxj + sin * uyj) * exptheta_m0
            J_t_m0_scal = wj * c * inv_gammaj * (cos * uyj - sin * uxj) * exptheta_m0
            J_z_m0_scal = wj * c * inv_gammaj * uzj * exptheta_m0
            # Mode 1
            J_r_m1_scal = wj * c * inv_gammaj * (cos * uxj + sin * uyj) * exptheta_m1
            J_t_m1_scal = wj * c * inv_gammaj * (cos * uyj - sin * uxj) * exptheta_m1
            J_z_m1_scal = wj * c * inv_gammaj * uzj * exptheta_m1

            J_r_m0_00 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * J_r_m0_scal
            )
            J_r_m1_00 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * J_r_m1_scal
            )
            J_r_m0_01 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * J_r_m0_scal
            )
            J_r_m1_01 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * J_r_m1_scal
            )
            J_r_m0_02 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * J_r_m0_scal
            )
            J_r_m1_02 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * J_r_m1_scal
            )
            J_r_m0_03 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * J_r_m0_scal
            )
            J_r_m1_03 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * J_r_m1_scal
            )

            J_r_m0_10 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * J_r_m0_scal
            )
            J_r_m1_10 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * J_r_m1_scal
            )
            J_r_m0_11 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * J_r_m0_scal
            )
            J_r_m1_11 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * J_r_m1_scal
            )
            J_r_m0_12 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * J_r_m0_scal
            )
            J_r_m1_12 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * J_r_m1_scal
            )
            J_r_m0_13 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * J_r_m0_scal
            )
            J_r_m1_13 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * J_r_m1_scal
            )

            J_r_m0_20 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * J_r_m0_scal
            )
            J_r_m1_20 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * J_r_m1_scal
            )
            J_r_m0_21 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * J_r_m0_scal
            )
            J_r_m1_21 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * J_r_m1_scal
            )
            J_r_m0_22 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * J_r_m0_scal
            )
            J_r_m1_22 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * J_r_m1_scal
            )
            J_r_m0_23 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * J_r_m0_scal
            )
            J_r_m1_23 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * J_r_m1_scal
            )

            J_r_m0_30 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * J_r_m0_scal
            )
            J_r_m1_30 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * J_r_m1_scal
            )
            J_r_m0_31 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * J_r_m0_scal
            )
            J_r_m1_31 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * J_r_m1_scal
            )
            J_r_m0_32 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * J_r_m0_scal
            )
            J_r_m1_32 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * J_r_m1_scal
            )
            J_r_m0_33 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * J_r_m0_scal
            )
            J_r_m1_33 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * J_r_m1_scal
            )

            J_t_m0_00 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * J_t_m0_scal
            )
            J_t_m1_00 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * J_t_m1_scal
            )
            J_t_m0_01 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * J_t_m0_scal
            )
            J_t_m1_01 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * J_t_m1_scal
            )
            J_t_m0_02 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * J_t_m0_scal
            )
            J_t_m1_02 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * J_t_m1_scal
            )
            J_t_m0_03 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * J_t_m0_scal
            )
            J_t_m1_03 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * J_t_m1_scal
            )

            J_t_m0_10 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * J_t_m0_scal
            )
            J_t_m1_10 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * J_t_m1_scal
            )
            J_t_m0_11 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * J_t_m0_scal
            )
            J_t_m1_11 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * J_t_m1_scal
            )
            J_t_m0_12 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * J_t_m0_scal
            )
            J_t_m1_12 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * J_t_m1_scal
            )
            J_t_m0_13 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * J_t_m0_scal
            )
            J_t_m1_13 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * J_t_m1_scal
            )

            J_t_m0_20 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * J_t_m0_scal
            )
            J_t_m1_20 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * J_t_m1_scal
            )
            J_t_m0_21 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * J_t_m0_scal
            )
            J_t_m1_21 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * J_t_m1_scal
            )
            J_t_m0_22 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * J_t_m0_scal
            )
            J_t_m1_22 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * J_t_m1_scal
            )
            J_t_m0_23 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * J_t_m0_scal
            )
            J_t_m1_23 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * J_t_m1_scal
            )

            J_t_m0_30 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * J_t_m0_scal
            )
            J_t_m1_30 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * J_t_m1_scal
            )
            J_t_m0_31 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * J_t_m0_scal
            )
            J_t_m1_31 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * J_t_m1_scal
            )
            J_t_m0_32 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * J_t_m0_scal
            )
            J_t_m1_32 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * J_t_m1_scal
            )
            J_t_m0_33 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * J_t_m0_scal
            )
            J_t_m1_33 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * J_t_m1_scal
            )

            J_z_m0_00 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * J_z_m0_scal
            )
            J_z_m1_00 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * J_z_m1_scal
            )
            J_z_m0_01 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * J_z_m0_scal
            )
            J_z_m1_01 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * J_z_m1_scal
            )
            J_z_m0_02 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * J_z_m0_scal
            )
            J_z_m1_02 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * J_z_m1_scal
            )
            J_z_m0_03 += (
                Sr_cubic(r_cell, 0, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * J_z_m0_scal
            )
            J_z_m1_03 += (
                Sr_cubic(r_cell, 0, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * J_z_m1_scal
            )

            J_z_m0_10 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * J_z_m0_scal
            )
            J_z_m1_10 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * J_z_m1_scal
            )
            J_z_m0_11 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * J_z_m0_scal
            )
            J_z_m1_11 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * J_z_m1_scal
            )
            J_z_m0_12 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * J_z_m0_scal
            )
            J_z_m1_12 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * J_z_m1_scal
            )
            J_z_m0_13 += (
                Sr_cubic(r_cell, 1, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * J_z_m0_scal
            )
            J_z_m1_13 += (
                Sr_cubic(r_cell, 1, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * J_z_m1_scal
            )

            J_z_m0_20 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * J_z_m0_scal
            )
            J_z_m1_20 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * J_z_m1_scal
            )
            J_z_m0_21 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * J_z_m0_scal
            )
            J_z_m1_21 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * J_z_m1_scal
            )
            J_z_m0_22 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * J_z_m0_scal
            )
            J_z_m1_22 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * J_z_m1_scal
            )
            J_z_m0_23 += (
                Sr_cubic(r_cell, 2, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * J_z_m0_scal
            )
            J_z_m1_23 += (
                Sr_cubic(r_cell, 2, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * J_z_m1_scal
            )

            J_z_m0_30 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 0)
                * J_z_m0_scal
            )
            J_z_m1_30 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 0)
                * J_z_m1_scal
            )
            J_z_m0_31 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 1)
                * J_z_m0_scal
            )
            J_z_m1_31 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 1)
                * J_z_m1_scal
            )
            J_z_m0_32 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 2)
                * J_z_m0_scal
            )
            J_z_m1_32 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 2)
                * J_z_m1_scal
            )
            J_z_m0_33 += (
                Sr_cubic(r_cell, 3, wp.float64(1), bn_m0)
                * Sz_cubic(z_cell, 3)
                * J_z_m0_scal
            )
            J_z_m1_33 += (
                Sr_cubic(r_cell, 3, wp.float64(-1), bn_m1)
                * Sz_cubic(z_cell, 3)
                * J_z_m1_scal
            )

        # Calculate longitudinal indices at which to add charge
        iz0 = iz_upper - 2
        iz1 = iz_upper - 1
        iz2 = iz_upper
        iz3 = iz_upper + 1
        if iz0 < 0:
            iz0 += Nz
        if iz1 < 0:
            iz1 += Nz
        if iz3 > Nz - 1:
            iz3 -= Nz
        # Calculate radial indices at which to add charge
        ir0 = ir_upper - 2
        ir1 = min(ir_upper - 1, Nr - 1)
        ir2 = min(ir_upper, Nr - 1)
        ir3 = min(ir_upper + 1, Nr - 1)
        if ir0 < 0:
            # Deposition below the axis: fold index into physical region
            ir0 = -(1 + ir0)
        if ir1 < 0:
            # Deposition below the axis: fold index into physical region
            ir1 = -(1 + ir1)

        # Atomically add the registers to global memory
        if frequency_per_cell > 0:
            # jr: Mode 0
            wp.atomic_add(j_r_m0, iz0, ir0, wp.vec2d(J_r_m0_00, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz0, ir1, wp.vec2d(J_r_m0_10, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz0, ir2, wp.vec2d(J_r_m0_20, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz0, ir3, wp.vec2d(J_r_m0_30, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz1, ir0, wp.vec2d(J_r_m0_01, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz1, ir1, wp.vec2d(J_r_m0_11, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz1, ir2, wp.vec2d(J_r_m0_21, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz1, ir3, wp.vec2d(J_r_m0_31, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz2, ir0, wp.vec2d(J_r_m0_02, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz2, ir1, wp.vec2d(J_r_m0_12, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz2, ir2, wp.vec2d(J_r_m0_22, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz2, ir3, wp.vec2d(J_r_m0_32, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz3, ir0, wp.vec2d(J_r_m0_03, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz3, ir1, wp.vec2d(J_r_m0_13, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz3, ir2, wp.vec2d(J_r_m0_23, wp.float64(0.0)))
            wp.atomic_add(j_r_m0, iz3, ir3, wp.vec2d(J_r_m0_33, wp.float64(0.0)))
            # jr: Mode 1
            wp.atomic_add(j_r_m1, iz0, ir0, wp.vec2d(J_r_m1_00[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz0, ir0, wp.vec2d(wp.float64(0.0), J_r_m1_00[1]))
            wp.atomic_add(j_r_m1, iz0, ir1, wp.vec2d(J_r_m1_10[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz0, ir1, wp.vec2d(wp.float64(0.0), J_r_m1_10[1]))
            wp.atomic_add(j_r_m1, iz0, ir2, wp.vec2d(J_r_m1_20[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz0, ir2, wp.vec2d(wp.float64(0.0), J_r_m1_20[1]))
            wp.atomic_add(j_r_m1, iz0, ir3, wp.vec2d(J_r_m1_30[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz0, ir3, wp.vec2d(wp.float64(0.0), J_r_m1_30[1]))
            wp.atomic_add(j_r_m1, iz1, ir0, wp.vec2d(J_r_m1_01[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz1, ir0, wp.vec2d(wp.float64(0.0), J_r_m1_01[1]))
            wp.atomic_add(j_r_m1, iz1, ir1, wp.vec2d(J_r_m1_11[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz1, ir1, wp.vec2d(wp.float64(0.0), J_r_m1_11[1]))
            wp.atomic_add(j_r_m1, iz1, ir2, wp.vec2d(J_r_m1_21[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz1, ir2, wp.vec2d(wp.float64(0.0), J_r_m1_21[1]))
            wp.atomic_add(j_r_m1, iz1, ir3, wp.vec2d(J_r_m1_31[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz1, ir3, wp.vec2d(wp.float64(0.0), J_r_m1_31[1]))
            wp.atomic_add(j_r_m1, iz2, ir0, wp.vec2d(J_r_m1_02[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz2, ir0, wp.vec2d(wp.float64(0.0), J_r_m1_02[1]))
            wp.atomic_add(j_r_m1, iz2, ir1, wp.vec2d(J_r_m1_12[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz2, ir1, wp.vec2d(wp.float64(0.0), J_r_m1_12[1]))
            wp.atomic_add(j_r_m1, iz2, ir2, wp.vec2d(J_r_m1_22[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz2, ir2, wp.vec2d(wp.float64(0.0), J_r_m1_22[1]))
            wp.atomic_add(j_r_m1, iz2, ir3, wp.vec2d(J_r_m1_32[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz2, ir3, wp.vec2d(wp.float64(0.0), J_r_m1_32[1]))
            wp.atomic_add(j_r_m1, iz3, ir0, wp.vec2d(J_r_m1_03[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz3, ir0, wp.vec2d(wp.float64(0.0), J_r_m1_03[1]))
            wp.atomic_add(j_r_m1, iz3, ir1, wp.vec2d(J_r_m1_13[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz3, ir1, wp.vec2d(wp.float64(0.0), J_r_m1_13[1]))
            wp.atomic_add(j_r_m1, iz3, ir2, wp.vec2d(J_r_m1_23[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz3, ir2, wp.vec2d(wp.float64(0.0), J_r_m1_23[1]))
            wp.atomic_add(j_r_m1, iz3, ir3, wp.vec2d(J_r_m1_33[0], wp.float64(0.0)))
            wp.atomic_add(j_r_m1, iz3, ir3, wp.vec2d(wp.float64(0.0), J_r_m1_33[1]))
            # jt: Mode 0
            wp.atomic_add(j_t_m0, iz0, ir0, wp.vec2d(J_t_m0_00, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz0, ir1, wp.vec2d(J_t_m0_10, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz0, ir2, wp.vec2d(J_t_m0_20, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz0, ir3, wp.vec2d(J_t_m0_30, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz1, ir0, wp.vec2d(J_t_m0_01, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz1, ir1, wp.vec2d(J_t_m0_11, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz1, ir2, wp.vec2d(J_t_m0_21, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz1, ir3, wp.vec2d(J_t_m0_31, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz2, ir0, wp.vec2d(J_t_m0_02, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz2, ir1, wp.vec2d(J_t_m0_12, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz2, ir2, wp.vec2d(J_t_m0_22, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz2, ir3, wp.vec2d(J_t_m0_32, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz3, ir0, wp.vec2d(J_t_m0_03, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz3, ir1, wp.vec2d(J_t_m0_13, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz3, ir2, wp.vec2d(J_t_m0_23, wp.float64(0.0)))
            wp.atomic_add(j_t_m0, iz3, ir3, wp.vec2d(J_t_m0_33, wp.float64(0.0)))
            # jt: Mode 1
            wp.atomic_add(j_t_m1, iz0, ir0, wp.vec2d(J_t_m1_00[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz0, ir0, wp.vec2d(wp.float64(0.0), J_t_m1_00[1]))
            wp.atomic_add(j_t_m1, iz0, ir1, wp.vec2d(J_t_m1_10[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz0, ir1, wp.vec2d(wp.float64(0.0), J_t_m1_10[1]))
            wp.atomic_add(j_t_m1, iz0, ir2, wp.vec2d(J_t_m1_20[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz0, ir2, wp.vec2d(wp.float64(0.0), J_t_m1_20[1]))
            wp.atomic_add(j_t_m1, iz0, ir3, wp.vec2d(J_t_m1_30[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz0, ir3, wp.vec2d(wp.float64(0.0), J_t_m1_30[1]))
            wp.atomic_add(j_t_m1, iz1, ir0, wp.vec2d(J_t_m1_01[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz1, ir0, wp.vec2d(wp.float64(0.0), J_t_m1_01[1]))
            wp.atomic_add(j_t_m1, iz1, ir1, wp.vec2d(J_t_m1_11[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz1, ir1, wp.vec2d(wp.float64(0.0), J_t_m1_11[1]))
            wp.atomic_add(j_t_m1, iz1, ir2, wp.vec2d(J_t_m1_21[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz1, ir2, wp.vec2d(wp.float64(0.0), J_t_m1_21[1]))
            wp.atomic_add(j_t_m1, iz1, ir3, wp.vec2d(J_t_m1_31[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz1, ir3, wp.vec2d(wp.float64(0.0), J_t_m1_31[1]))
            wp.atomic_add(j_t_m1, iz2, ir0, wp.vec2d(J_t_m1_02[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz2, ir0, wp.vec2d(wp.float64(0.0), J_t_m1_02[1]))
            wp.atomic_add(j_t_m1, iz2, ir1, wp.vec2d(J_t_m1_12[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz2, ir1, wp.vec2d(wp.float64(0.0), J_t_m1_12[1]))
            wp.atomic_add(j_t_m1, iz2, ir2, wp.vec2d(J_t_m1_22[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz2, ir2, wp.vec2d(wp.float64(0.0), J_t_m1_22[1]))
            wp.atomic_add(j_t_m1, iz2, ir3, wp.vec2d(J_t_m1_32[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz2, ir3, wp.vec2d(wp.float64(0.0), J_t_m1_32[1]))
            wp.atomic_add(j_t_m1, iz3, ir0, wp.vec2d(J_t_m1_03[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz3, ir0, wp.vec2d(wp.float64(0.0), J_t_m1_03[1]))
            wp.atomic_add(j_t_m1, iz3, ir1, wp.vec2d(J_t_m1_13[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz3, ir1, wp.vec2d(wp.float64(0.0), J_t_m1_13[1]))
            wp.atomic_add(j_t_m1, iz3, ir2, wp.vec2d(J_t_m1_23[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz3, ir2, wp.vec2d(wp.float64(0.0), J_t_m1_23[1]))
            wp.atomic_add(j_t_m1, iz3, ir3, wp.vec2d(J_t_m1_33[0], wp.float64(0.0)))
            wp.atomic_add(j_t_m1, iz3, ir3, wp.vec2d(wp.float64(0.0), J_t_m1_33[1]))
            # jz: Mode 0
            wp.atomic_add(j_z_m0, iz0, ir0, wp.vec2d(J_z_m0_00, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz0, ir1, wp.vec2d(J_z_m0_10, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz0, ir2, wp.vec2d(J_z_m0_20, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz0, ir3, wp.vec2d(J_z_m0_30, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz1, ir0, wp.vec2d(J_z_m0_01, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz1, ir1, wp.vec2d(J_z_m0_11, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz1, ir2, wp.vec2d(J_z_m0_21, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz1, ir3, wp.vec2d(J_z_m0_31, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz2, ir0, wp.vec2d(J_z_m0_02, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz2, ir1, wp.vec2d(J_z_m0_12, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz2, ir2, wp.vec2d(J_z_m0_22, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz2, ir3, wp.vec2d(J_z_m0_32, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz3, ir0, wp.vec2d(J_z_m0_03, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz3, ir1, wp.vec2d(J_z_m0_13, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz3, ir2, wp.vec2d(J_z_m0_23, wp.float64(0.0)))
            wp.atomic_add(j_z_m0, iz3, ir3, wp.vec2d(J_z_m0_33, wp.float64(0.0)))
            # jz: Mode 1
            wp.atomic_add(j_z_m1, iz0, ir0, wp.vec2d(J_z_m1_00[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz0, ir0, wp.vec2d(wp.float64(0.0), J_z_m1_00[1]))
            wp.atomic_add(j_z_m1, iz0, ir1, wp.vec2d(J_z_m1_10[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz0, ir1, wp.vec2d(wp.float64(0.0), J_z_m1_10[1]))
            wp.atomic_add(j_z_m1, iz0, ir2, wp.vec2d(J_z_m1_20[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz0, ir2, wp.vec2d(wp.float64(0.0), J_z_m1_20[1]))
            wp.atomic_add(j_z_m1, iz0, ir3, wp.vec2d(J_z_m1_30[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz0, ir3, wp.vec2d(wp.float64(0.0), J_z_m1_30[1]))
            wp.atomic_add(j_z_m1, iz1, ir0, wp.vec2d(J_z_m1_01[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz1, ir0, wp.vec2d(wp.float64(0.0), J_z_m1_01[1]))
            wp.atomic_add(j_z_m1, iz1, ir1, wp.vec2d(J_z_m1_11[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz1, ir1, wp.vec2d(wp.float64(0.0), J_z_m1_11[1]))
            wp.atomic_add(j_z_m1, iz1, ir2, wp.vec2d(J_z_m1_21[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz1, ir2, wp.vec2d(wp.float64(0.0), J_z_m1_21[1]))
            wp.atomic_add(j_z_m1, iz1, ir3, wp.vec2d(J_z_m1_31[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz1, ir3, wp.vec2d(wp.float64(0.0), J_z_m1_31[1]))
            wp.atomic_add(j_z_m1, iz2, ir0, wp.vec2d(J_z_m1_02[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz2, ir0, wp.vec2d(wp.float64(0.0), J_z_m1_02[1]))
            wp.atomic_add(j_z_m1, iz2, ir1, wp.vec2d(J_z_m1_12[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz2, ir1, wp.vec2d(wp.float64(0.0), J_z_m1_12[1]))
            wp.atomic_add(j_z_m1, iz2, ir2, wp.vec2d(J_z_m1_22[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz2, ir2, wp.vec2d(wp.float64(0.0), J_z_m1_22[1]))
            wp.atomic_add(j_z_m1, iz2, ir3, wp.vec2d(J_z_m1_32[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz2, ir3, wp.vec2d(wp.float64(0.0), J_z_m1_32[1]))
            wp.atomic_add(j_z_m1, iz3, ir0, wp.vec2d(J_z_m1_03[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz3, ir0, wp.vec2d(wp.float64(0.0), J_z_m1_03[1]))
            wp.atomic_add(j_z_m1, iz3, ir1, wp.vec2d(J_z_m1_13[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz3, ir1, wp.vec2d(wp.float64(0.0), J_z_m1_13[1]))
            wp.atomic_add(j_z_m1, iz3, ir2, wp.vec2d(J_z_m1_23[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz3, ir2, wp.vec2d(wp.float64(0.0), J_z_m1_23[1]))
            wp.atomic_add(j_z_m1, iz3, ir3, wp.vec2d(J_z_m1_33[0], wp.float64(0.0)))
            wp.atomic_add(j_z_m1, iz3, ir3, wp.vec2d(wp.float64(0.0), J_z_m1_33[1]))
