# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the optimized fields methods that use cuda on a GPU
"""
import warp as wp
from fbpic.utils.warp import warp_kernel, cmul, mul_i
from scipy.constants import c, epsilon_0, mu_0

c2 = wp.constant(wp.float64(c**2))
epsilon_0 = wp.constant(wp.float64(epsilon_0))
mu_0 = wp.constant(wp.float64(mu_0))

# ------------------
# Erasing functions
# ------------------


@warp_kernel
def cuda_erase_scalar(array: wp.array2d(dtype=wp.vec2d)):
    """
    Set input array to 0

    The array is typically an interpolation grid array, and it
    is set to zero before depositing the currents

    Parameters:
    ------------
    array: 2darrays of complexs
       Array that represent the fields on the grid
       (The first axis corresponds to z and the second axis to r) d
    """

    # Cuda 2D grid
    iz, ir = wp.tid()

    # Set the elements of the array to 0
    if (iz < array.shape[0]) and (ir < array.shape[1]):
        array[iz, ir] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))


@warp_kernel
def cuda_erase_vector(
    array_r: wp.array2d(dtype=wp.vec2d),
    array_t: wp.array2d(dtype=wp.vec2d),
    array_z: wp.array2d(dtype=wp.vec2d),
):
    """
    Set the input arrays to 0

    These arrays are typically interpolation grid arrays, and they
    are set to zero before depositing the currents

    Parameters:
    ------------
    array_r, array_t, array_z: 2darrays of complexs
       Arrays that represent the fields on the grid
       (The first axis corresponds to z and the second axis to r)
    """

    # Cuda 2D grid
    iz, ir = wp.tid()

    # Set the elements of the array to 0
    if (iz < array_r.shape[0]) and (ir < array_r.shape[1]):
        array_r[iz, ir] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        array_t[iz, ir] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        array_z[iz, ir] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))


# ---------------------------
# Divide by volume functions
# ---------------------------


@warp_kernel
def cuda_divide_scalar_by_volume(
    array: wp.array2d(dtype=wp.vec2d),
    invvol: wp.array(dtype=wp.float64),
):
    """
    Multiply the input array by the corresponding invvol

    Parameters:
    ------------
    array: 2darray of complexs
       Array that represent the fields on the grid
       (The first axis corresponds to z and the second axis to r)

    invvol: 1darray of floats
       Array that contain the inverse of the volume of the cell
       The axis corresponds to r
    """

    # Cuda 2D grid
    iz, ir = wp.tid()

    # Multiply by inverse volume
    if (iz < array.shape[0]) and (ir < array.shape[1]):
        array[iz, ir] = array[iz, ir] * invvol[ir]


@warp_kernel
def cuda_divide_vector_by_volume(
    array_r: wp.array2d(dtype=wp.vec2d),
    array_t: wp.array2d(dtype=wp.vec2d),
    array_z: wp.array2d(dtype=wp.vec2d),
    invvol: wp.array(dtype=wp.float64),
):
    """
    Multiply the input arrays by the corresponding invvol

    Parameters:
    ------------
    array_r, array_t, array_z: 2darrays of complexs
       Arrays that represent the fields on the grid
       (The first axis corresponds to z and the second axis to r)

    invvol: 1darray of floats
       Arrays that contain the inverse of the volume of the cell
       The axis corresponds to r
    """

    # Cuda 2D grid
    iz, ir = wp.tid()

    # Multiply by inverse volume
    if (iz < array_r.shape[0]) and (ir < array_r.shape[1]):
        array_r[iz, ir] = array_r[iz, ir] * invvol[ir]
        array_t[iz, ir] = array_t[iz, ir] * invvol[ir]
        array_z[iz, ir] = array_z[iz, ir] * invvol[ir]


# -----------------------------------
# Methods of the SpectralGrid object
# -----------------------------------


@warp_kernel
def cuda_correct_currents_curlfree_standard(
    rho_prev: wp.array2d(dtype=wp.vec2d),
    rho_next: wp.array2d(dtype=wp.vec2d),
    Jp: wp.array2d(dtype=wp.vec2d),
    Jm: wp.array2d(dtype=wp.vec2d),
    Jz: wp.array2d(dtype=wp.vec2d),
    kz: wp.array2d(dtype=wp.float64),
    kr: wp.array2d(dtype=wp.float64),
    inv_k2: wp.array2d(dtype=wp.float64),
    inv_dt: wp.float64,
    Nz: int,
    Nr: int,
):
    """
    Correct the currents in spectral space, using the standard pstad
    """
    # Cuda 2D grid
    iz, ir = wp.tid()

    # Perform the current correction
    if (iz < Nz) and (ir < Nr):

        # Calculate the intermediate variable F
        F = -inv_k2[iz, ir] * (
            (rho_next[iz, ir] - rho_prev[iz, ir]) * inv_dt
            + kz[iz, ir] * mul_i(Jz[iz, ir])
            + kr[iz, ir] * (Jp[iz, ir] - Jm[iz, ir])
        )

        # Correct the currents accordingly
        Jp[iz, ir] += wp.float64(0.5) * kr[iz, ir] * F
        Jm[iz, ir] += -wp.float64(0.5) * kr[iz, ir] * F
        Jz[iz, ir] += -kz[iz, ir] * mul_i(F)


@warp_kernel
def cuda_correct_currents_crossdeposition_standard(
    rho_prev: wp.array2d(dtype=wp.vec2d),
    rho_next: wp.array2d(dtype=wp.vec2d),
    rho_next_z: wp.array2d(dtype=wp.vec2d),
    rho_next_xy: wp.array2d(dtype=wp.vec2d),
    Jp: wp.array2d(dtype=wp.vec2d),
    Jm: wp.array2d(dtype=wp.vec2d),
    Jz: wp.array2d(dtype=wp.vec2d),
    kz: wp.array2d(dtype=wp.float64),
    kr: wp.array2d(dtype=wp.float64),
    inv_dt: wp.float64,
    Nz: int,
    Nr: int,
):
    """
    Correct the currents in spectral space, using the cross-deposition
    algorithm adapted to the standard psatd.
    """
    # Cuda 2D grid
    iz, ir = wp.tid()

    # Perform the current correction
    if (iz < Nz) and (ir < Nr):
        # Calculate the intermediate variable Dz and Dxy
        # (Such that Dz + Dxy is the error in the continuity equation)
        Dz = kz[iz, ir] * mul_i(Jz[iz, ir]) + wp.float64(0.5) * inv_dt * (
            rho_next[iz, ir]
            - rho_next_xy[iz, ir]
            + rho_next_z[iz, ir]
            - rho_prev[iz, ir]
        )
        Dxy = kr[iz, ir] * (Jp[iz, ir] - Jm[iz, ir]) + wp.float64(0.5) * inv_dt * (
            rho_next[iz, ir]
            - rho_next_z[iz, ir]
            + rho_next_xy[iz, ir]
            - rho_prev[iz, ir]
        )

        # Correct the currents accordingly
        if kr[iz, ir] != 0:
            inv_kr = wp.float64(1.0) / kr[iz, ir]
            Jp[iz, ir] += -wp.float64(0.5) * Dxy * inv_kr
            Jm[iz, ir] += wp.float64(0.5) * Dxy * inv_kr
        if kz[iz, ir] != 0:
            inv_kz = wp.float64(1.0) / kz[iz, ir]
            Jz[iz, ir] += mul_i(Dz) * inv_kz


@warp_kernel
def cuda_correct_currents_curlfree_comoving(
    rho_prev: wp.array2d(dtype=wp.vec2d),
    rho_next: wp.array2d(dtype=wp.vec2d),
    Jp: wp.array2d(dtype=wp.vec2d),
    Jm: wp.array2d(dtype=wp.vec2d),
    Jz: wp.array2d(dtype=wp.vec2d),
    kz: wp.array2d(dtype=wp.float64),
    kr: wp.array2d(dtype=wp.float64),
    inv_k2: wp.array2d(dtype=wp.float64),
    j_corr_coef: wp.array2d(dtype=wp.vec2d),
    T_eb: wp.array2d(dtype=wp.vec2d),
    T_cc: wp.array2d(dtype=wp.vec2d),
    inv_dt: wp.float64,
    Nz: int,
    Nr: int,
):
    """
    Correct the currents in spectral space, using the curl-free correction
    which is adapted to the galilean/comoving-currents assumption
    """
    # Cuda 2D grid
    iz, ir = wp.tid()

    # Perform the current correction
    if (iz < Nz) and (ir < Nr):

        # Calculate the intermediate variable F
        F = -inv_k2[iz, ir] * (
            cmul(
                cmul(T_cc[iz, ir], j_corr_coef[iz, ir]),
                rho_next[iz, ir] - cmul(rho_prev[iz, ir], T_eb[iz, ir]),
            )
            + kz[iz, ir] * mul_i(Jz[iz, ir])
            + kr[iz, ir] * (Jp[iz, ir] - Jm[iz, ir])
        )

        # Correct the currents accordingly
        Jp[iz, ir] += wp.float64(0.5) * kr[iz, ir] * F
        Jm[iz, ir] += -wp.float64(0.5) * kr[iz, ir] * F
        Jz[iz, ir] += -kz[iz, ir] * mul_i(F)


@warp_kernel
def cuda_correct_currents_crossdeposition_comoving(
    rho_prev: wp.array2d(dtype=wp.vec2d),
    rho_next: wp.array2d(dtype=wp.vec2d),
    rho_next_z: wp.array2d(dtype=wp.vec2d),
    rho_next_xy: wp.array2d(dtype=wp.vec2d),
    Jp: wp.array2d(dtype=wp.vec2d),
    Jm: wp.array2d(dtype=wp.vec2d),
    Jz: wp.array2d(dtype=wp.vec2d),
    kz: wp.array2d(dtype=wp.float64),
    kr: wp.array2d(dtype=wp.float64),
    j_corr_coef: wp.array2d(dtype=wp.vec2d),
    T_eb: wp.array2d(dtype=wp.vec2d),
    T_cc: wp.array2d(dtype=wp.vec2d),
    inv_dt: wp.float64,
    Nz: int,
    Nr: int,
):
    """
    Correct the currents in spectral space, using the cross-deposition
    algorithm adapted to the galilean/comoving-currents assumption.
    """
    # Cuda 2D grid
    iz, ir = wp.tid()

    # Perform the current correction
    if (iz < Nz) and (ir < Nr):

        # Calculate the intermediate variable Dz and Dxy
        # (Such that Dz + Dxy is the error in the continuity equation)
        Dz = kz[iz, ir] * mul_i(Jz[iz, ir]) + cmul(
            cmul(wp.float64(0.5) * T_cc[iz, ir], j_corr_coef[iz, ir]),
            rho_next[iz, ir]
            - cmul(T_eb[iz, ir], rho_next_xy[iz, ir])
            + rho_next_z[iz, ir]
            - cmul(T_eb[iz, ir], rho_prev[iz, ir]),
        )
        Dxy = kr[iz, ir] * (Jp[iz, ir] - Jm[iz, ir]) + cmul(
            cmul(wp.float64(0.5) * T_cc[iz, ir], j_corr_coef[iz, ir]),
            rho_next[iz, ir]
            + cmul(T_eb[iz, ir], rho_next_xy[iz, ir])
            - rho_next_z[iz, ir]
            - cmul(T_eb[iz, ir], rho_prev[iz, ir]),
        )

        # Correct the currents accordingly
        if kr[iz, ir] != 0:
            inv_kr = wp.float64(1.0) / kr[iz, ir]
            Jp[iz, ir] += -wp.float64(0.5) * Dxy * inv_kr
            Jm[iz, ir] += wp.float64(0.5) * Dxy * inv_kr
        if kz[iz, ir] != 0:
            inv_kz = wp.float64(1.0) / kz[iz, ir]
            Jz[iz, ir] += mul_i(Dz) * inv_kz


@warp_kernel
def cuda_push_eb_standard(
    Ep: wp.array2d(dtype=wp.vec2d),
    Em: wp.array2d(dtype=wp.vec2d),
    Ez: wp.array2d(dtype=wp.vec2d),
    Bp: wp.array2d(dtype=wp.vec2d),
    Bm: wp.array2d(dtype=wp.vec2d),
    Bz: wp.array2d(dtype=wp.vec2d),
    Jp: wp.array2d(dtype=wp.vec2d),
    Jm: wp.array2d(dtype=wp.vec2d),
    Jz: wp.array2d(dtype=wp.vec2d),
    rho_prev: wp.array2d(dtype=wp.vec2d),
    rho_next: wp.array2d(dtype=wp.vec2d),
    rho_prev_coef: wp.array2d(dtype=wp.float64),
    rho_next_coef: wp.array2d(dtype=wp.float64),
    j_coef: wp.array2d(dtype=wp.float64),
    C: wp.array2d(dtype=wp.float64),
    S_w: wp.array2d(dtype=wp.float64),
    kr: wp.array2d(dtype=wp.float64),
    kz: wp.array2d(dtype=wp.float64),
    dt: wp.float64,
    use_true_rho: bool,
    Nz: int,
    Nr: int,
):
    """
    Push the fields over one timestep, using the standard psatd algorithm

    See the documentation of SpectralGrid.push_eb_with
    """
    # Cuda 2D grid
    iz, ir = wp.tid()

    # Push the fields
    if (iz < Nz) and (ir < Nr):

        # Save the electric fields, since it is needed for the B push
        Ep_old = Ep[iz, ir]
        Em_old = Em[iz, ir]
        Ez_old = Ez[iz, ir]

        # Calculate useful auxiliary arrays
        if use_true_rho:
            # Evaluation using the rho projected on the grid
            rho_diff = (
                rho_next_coef[iz, ir] * rho_next[iz, ir]
                - rho_prev_coef[iz, ir] * rho_prev[iz, ir]
            )
        else:
            # Evaluation using div(E) and div(J)
            divE = kr[iz, ir] * (Ep[iz, ir] - Em[iz, ir]) + kz[iz, ir] * mul_i(
                Ez[iz, ir]
            )
            divJ = kr[iz, ir] * (Jp[iz, ir] - Jm[iz, ir]) + kz[iz, ir] * mul_i(
                Jz[iz, ir]
            )

            rho_diff = (
                rho_next_coef[iz, ir] - rho_prev_coef[iz, ir]
            ) * epsilon_0 * divE - rho_next_coef[iz, ir] * dt * divJ

        # Push the E field
        Ep[iz, ir] = (
            C[iz, ir] * Ep[iz, ir]
            + wp.float64(0.5) * kr[iz, ir] * rho_diff
            + c2
            * S_w[iz, ir]
            * (
                -wp.float64(0.5) * kr[iz, ir] * mul_i(Bz[iz, ir])
                + kz[iz, ir] * Bp[iz, ir]
                - mu_0 * Jp[iz, ir]
            )
        )

        Em[iz, ir] = (
            C[iz, ir] * Em[iz, ir]
            - wp.float64(0.5) * kr[iz, ir] * rho_diff
            + c2
            * S_w[iz, ir]
            * (
                -wp.float64(0.5) * kr[iz, ir] * mul_i(Bz[iz, ir])
                - kz[iz, ir] * Bm[iz, ir]
                - mu_0 * Jm[iz, ir]
            )
        )

        Ez[iz, ir] = (
            C[iz, ir] * Ez[iz, ir]
            - kz[iz, ir] * mul_i(rho_diff)
            + c2
            * S_w[iz, ir]
            * (
                kr[iz, ir] * mul_i(Bp[iz, ir])
                + kr[iz, ir] * mul_i(Bm[iz, ir])
                - mu_0 * Jz[iz, ir]
            )
        )

        # Push the B field
        Bp[iz, ir] = (
            C[iz, ir] * Bp[iz, ir]
            - S_w[iz, ir]
            * (-wp.float64(0.5) * kr[iz, ir] * mul_i(Ez_old) + kz[iz, ir] * Ep_old)
            + j_coef[iz, ir]
            * (
                -wp.float64(0.5) * kr[iz, ir] * mul_i(Jz[iz, ir])
                + kz[iz, ir] * Jp[iz, ir]
            )
        )

        Bm[iz, ir] = (
            C[iz, ir] * Bm[iz, ir]
            - S_w[iz, ir]
            * (-wp.float64(0.5) * kr[iz, ir] * mul_i(Ez_old) - kz[iz, ir] * Em_old)
            + j_coef[iz, ir]
            * (
                -wp.float64(0.5) * kr[iz, ir] * mul_i(Jz[iz, ir])
                - kz[iz, ir] * Jm[iz, ir]
            )
        )

        Bz[iz, ir] = (
            C[iz, ir] * Bz[iz, ir]
            - S_w[iz, ir] * (kr[iz, ir] * mul_i(Ep_old) + kr[iz, ir] * mul_i(Em_old))
            + j_coef[iz, ir]
            * (kr[iz, ir] * mul_i(Jp[iz, ir]) + kr[iz, ir] * mul_i(Jm[iz, ir]))
        )


@warp_kernel
def cuda_push_eb_pml_standard(
    Ep_pml: wp.array2d(dtype=wp.vec2d),
    Em_pml: wp.array2d(dtype=wp.vec2d),
    Bp_pml: wp.array2d(dtype=wp.vec2d),
    Bm_pml: wp.array2d(dtype=wp.vec2d),
    Ez: wp.array2d(dtype=wp.vec2d),
    Bz: wp.array2d(dtype=wp.vec2d),
    C: wp.array2d(dtype=wp.float64),
    S_w: wp.array2d(dtype=wp.float64),
    kr: wp.array2d(dtype=wp.float64),
    kz: wp.array2d(dtype=wp.float64),
    Nz: int,
    Nr: int,
):
    """
    Push the PML split fields over one timestep, using the standard psatd algorithm

    See the documentation of SpectralGrid.push_eb_with
    """
    # Cuda 2D grid
    iz, ir = wp.tid()

    # Push the fields
    if (iz < Nz) and (ir < Nr):

        # Push the PML E field
        Ep_pml[iz, ir] = C[iz, ir] * Ep_pml[iz, ir] + c2 * S_w[iz, ir] * (
            -wp.float64(0.5) * kr[iz, ir] * mul_i(Bz[iz, ir])
        )

        Em_pml[iz, ir] = C[iz, ir] * Em_pml[iz, ir] + c2 * S_w[iz, ir] * (
            -wp.float64(0.5) * kr[iz, ir] * mul_i(Bz[iz, ir])
        )

        # Push the PML B field
        Bp_pml[iz, ir] = C[iz, ir] * Bp_pml[iz, ir] - S_w[iz, ir] * (
            -wp.float64(0.5) * kr[iz, ir] * mul_i(Ez[iz, ir])
        )

        Bm_pml[iz, ir] = C[iz, ir] * Bm_pml[iz, ir] - S_w[iz, ir] * (
            -wp.float64(0.5) * kr[iz, ir] * mul_i(Ez[iz, ir])
        )


@warp_kernel
def cuda_push_eb_comoving(
    Ep: wp.array2d(dtype=wp.vec2d),
    Em: wp.array2d(dtype=wp.vec2d),
    Ez: wp.array2d(dtype=wp.vec2d),
    Bp: wp.array2d(dtype=wp.vec2d),
    Bm: wp.array2d(dtype=wp.vec2d),
    Bz: wp.array2d(dtype=wp.vec2d),
    Jp: wp.array2d(dtype=wp.vec2d),
    Jm: wp.array2d(dtype=wp.vec2d),
    Jz: wp.array2d(dtype=wp.vec2d),
    rho_prev: wp.array2d(dtype=wp.vec2d),
    rho_next: wp.array2d(dtype=wp.vec2d),
    rho_prev_coef: wp.array2d(dtype=wp.vec2d),
    rho_next_coef: wp.array2d(dtype=wp.vec2d),
    j_coef: wp.array2d(dtype=wp.vec2d),
    C: wp.array2d(dtype=wp.float64),
    S_w: wp.array2d(dtype=wp.float64),
    T_eb: wp.array2d(dtype=wp.vec2d),
    T_cc: wp.array2d(dtype=wp.vec2d),
    T_rho: wp.array2d(dtype=wp.vec2d),
    kr: wp.array2d(dtype=wp.float64),
    kz: wp.array2d(dtype=wp.float64),
    dt: wp.float64,
    V: wp.float64,
    use_true_rho: bool,
    Nz: int,
    Nr: int,
):
    """
    Push the fields over one timestep, using the psatd algorithm,
    with the assumptions of comoving currents
    (either with the galilean scheme or comoving scheme, depending on
    the values of the coefficients that are passed)

    See the documentation of SpectralGrid.push_eb_with
    """
    # Cuda 2D grid
    iz, ir = wp.tid()

    # Push the fields
    if (iz < Nz) and (ir < Nr):

        # Save the electric fields, since it is needed for the B push
        Ep_old = Ep[iz, ir]
        Em_old = Em[iz, ir]
        Ez_old = Ez[iz, ir]

        # Calculate useful auxiliary arrays
        if use_true_rho:
            # Evaluation using the rho projected on the grid
            rho_diff = cmul(rho_next_coef[iz, ir], rho_next[iz, ir]) - cmul(
                rho_prev_coef[iz, ir], rho_prev[iz, ir]
            )
        else:
            # Evaluation using div(E) and div(J)
            divE = kr[iz, ir] * (Ep[iz, ir] - Em[iz, ir]) + kz[iz, ir] * mul_i(
                Ez[iz, ir]
            )
            divJ = kr[iz, ir] * (Jp[iz, ir] - Jm[iz, ir]) + kz[iz, ir] * mul_i(
                Jz[iz, ir]
            )

            rho_diff = cmul(
                (cmul(T_eb[iz, ir], rho_next_coef[iz, ir]) - rho_prev_coef[iz, ir])
                * epsilon_0,
                divE,
            ) + cmul(cmul(T_rho[iz, ir], rho_next_coef[iz, ir]), divJ)

        # Push the E field
        Ep[iz, ir] = (
            cmul(T_eb[iz, ir] * C[iz, ir], Ep[iz, ir])
            + wp.float64(0.5) * kr[iz, ir] * rho_diff
            + cmul(mul_i(j_coef[iz, ir]) * kz[iz, ir] * V, Jp[iz, ir])
            + cmul(
                c2 * T_eb[iz, ir] * S_w[iz, ir],
                -wp.float64(0.5) * kr[iz, ir] * mul_i(Bz[iz, ir])
                + kz[iz, ir] * Bp[iz, ir]
                - cmul(mu_0 * T_cc[iz, ir], Jp[iz, ir]),
            )
        )

        Em[iz, ir] = (
            cmul(T_eb[iz, ir] * C[iz, ir], Em[iz, ir])
            - wp.float64(0.5) * kr[iz, ir] * rho_diff
            + cmul(mul_i(j_coef[iz, ir]) * kz[iz, ir] * V, Jm[iz, ir])
            + cmul(
                c2 * T_eb[iz, ir] * S_w[iz, ir],
                -wp.float64(0.5) * kr[iz, ir] * mul_i(Bz[iz, ir])
                - kz[iz, ir] * Bm[iz, ir]
                - cmul(mu_0 * T_cc[iz, ir], Jm[iz, ir]),
            )
        )

        Ez[iz, ir] = (
            cmul(T_eb[iz, ir] * C[iz, ir], Ez[iz, ir])
            - kz[iz, ir] * mul_i(rho_diff)
            + cmul(mul_i(j_coef[iz, ir]) * kz[iz, ir] * V, Jz[iz, ir])
            + cmul(
                c2 * T_eb[iz, ir] * S_w[iz, ir],
                kr[iz, ir] * mul_i(Bp[iz, ir])
                + kr[iz, ir] * mul_i(Bm[iz, ir])
                - cmul(mu_0 * T_cc[iz, ir], Jz[iz, ir]),
            )
        )

        # Push the B field
        Bp[iz, ir] = (
            cmul(T_eb[iz, ir] * C[iz, ir], Bp[iz, ir])
            - cmul(
                T_eb[iz, ir] * S_w[iz, ir],
                -wp.float64(0.5) * kr[iz, ir] * mul_i(Ez_old) + kz[iz, ir] * Ep_old,
            )
            + cmul(
                j_coef[iz, ir],
                -wp.float64(0.5) * kr[iz, ir] * mul_i(Jz[iz, ir])
                + kz[iz, ir] * Jp[iz, ir],
            )
        )

        Bm[iz, ir] = (
            cmul(T_eb[iz, ir] * C[iz, ir], Bm[iz, ir])
            - cmul(
                T_eb[iz, ir] * S_w[iz, ir],
                -wp.float64(0.5) * kr[iz, ir] * mul_i(Ez_old) - kz[iz, ir] * Em_old,
            )
            + cmul(
                j_coef[iz, ir],
                -wp.float64(0.5) * kr[iz, ir] * mul_i(Jz[iz, ir])
                - kz[iz, ir] * Jm[iz, ir],
            )
        )

        Bz[iz, ir] = (
            cmul(T_eb[iz, ir] * C[iz, ir], Bz[iz, ir])
            - cmul(
                T_eb[iz, ir] * S_w[iz, ir],
                kr[iz, ir] * mul_i(Ep_old) + kr[iz, ir] * mul_i(Em_old),
            )
            + cmul(
                j_coef[iz, ir],
                kr[iz, ir] * mul_i(Jp[iz, ir]) + kr[iz, ir] * mul_i(Jm[iz, ir]),
            )
        )


@warp_kernel
def cuda_push_eb_pml_comoving(
    Ep_pml: wp.array2d(dtype=wp.vec2d),
    Em_pml: wp.array2d(dtype=wp.vec2d),
    Bp_pml: wp.array2d(dtype=wp.vec2d),
    Bm_pml: wp.array2d(dtype=wp.vec2d),
    Ez: wp.array2d(dtype=wp.vec2d),
    Bz: wp.array2d(dtype=wp.vec2d),
    C: wp.array2d(dtype=wp.float64),
    S_w: wp.array2d(dtype=wp.float64),
    T_eb: wp.array2d(dtype=wp.vec2d),
    kr: wp.array2d(dtype=wp.float64),
    kz: wp.array2d(dtype=wp.float64),
    Nz: int,
    Nr: int,
):
    """
    Push the PML split fields over one timestep,
    using the galilean/comoving psatd algorithm

    See the documentation of SpectralGrid.push_eb_with
    """
    # Cuda 2D grid
    iz, ir = wp.tid()

    # Push the fields
    if (iz < Nz) and (ir < Nr):

        # Push the E field
        Ep_pml[iz, ir] = cmul(T_eb[iz, ir] * C[iz, ir], Ep_pml[iz, ir]) + cmul(
            c2 * T_eb[iz, ir] * S_w[iz, ir],
            -wp.float64(0.5) * kr[iz, ir] * mul_i(Bz[iz, ir]),
        )
        Em_pml[iz, ir] = cmul(T_eb[iz, ir] * C[iz, ir], Em_pml[iz, ir]) + cmul(
            c2 * T_eb[iz, ir] * S_w[iz, ir],
            -wp.float64(0.5) * kr[iz, ir] * mul_i(Bz[iz, ir]),
        )

        # Push the B field
        Bp_pml[iz, ir] = cmul(T_eb[iz, ir] * C[iz, ir], Bp_pml[iz, ir]) - cmul(
            T_eb[iz, ir] * S_w[iz, ir],
            -wp.float64(0.5) * kr[iz, ir] * mul_i(Ez[iz, ir]),
        )
        Bm_pml[iz, ir] = cmul(T_eb[iz, ir] * C[iz, ir], Bm_pml[iz, ir]) - cmul(
            T_eb[iz, ir] * S_w[iz, ir],
            -wp.float64(0.5) * kr[iz, ir] * mul_i(Ez[iz, ir]),
        )


@warp_kernel
def cuda_push_rho(
    rho_prev: wp.array2d(dtype=wp.vec2d),
    rho_next: wp.array2d(dtype=wp.vec2d),
    Nz: int,
    Nr: int,
):
    """
    Transfer the values of rho_next to rho_prev,
    and set rho_next to zero

    Parameters :
    ------------
    rho_prev, rho_next : 2darrays
        Arrays that represent rho in spectral space

    Nz, Nr : ints
        Dimensions of the arrays
    """

    # Cuda 2D grid
    iz, ir = wp.tid()

    # Push the fields
    if (iz < Nz) and (ir < Nr):

        rho_prev[iz, ir] = rho_next[iz, ir]
        rho_next[iz, ir] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))


@warp_kernel
def cuda_filter_scalar(
    field: wp.array2d(dtype=wp.vec2d),
    Nz: int,
    Nr: int,
    filter_array_z: wp.array(dtype=wp.float64),
    filter_array_r: wp.array(dtype=wp.float64),
):
    """
    Multiply the input field by the filter_array

    Parameters :
    ------------
    field : 2darray of complexs
        An array that represent the fields in spectral space

    filter_array_z, filter_array_r : 1darray of reals
        An array that damps the fields at high k, in z and r respectively

    Nz, Nr : ints
        Dimensions of the arrays
    """

    # Cuda 2D grid
    iz, ir = wp.tid()

    # Filter the field
    if (iz < Nz) and (ir < Nr):

        field[iz, ir] = filter_array_z[iz] * filter_array_r[ir] * field[iz, ir]


@warp_kernel
def cuda_filter_vector(
    fieldr: wp.array2d(dtype=wp.vec2d),
    fieldt: wp.array2d(dtype=wp.vec2d),
    fieldz: wp.array2d(dtype=wp.vec2d),
    Nz: int,
    Nr: int,
    filter_array_z: wp.array(dtype=wp.float64),
    filter_array_r: wp.array(dtype=wp.float64),
):
    """
    Multiply the input field by the filter_array

    Parameters :
    ------------
    field : 2darray of complexs
        An array that represent the fields in spectral space

    filter_array_z, filter_array_r : 1darray of reals
        An array that damps the fields at high k, in z and r respectively

    Nz, Nr : ints
        Dimensions of the arrays
    """

    # Cuda 2D grid
    iz, ir = wp.tid()

    # Filter the field
    if (iz < Nz) and (ir < Nr):

        fieldr[iz, ir] = filter_array_z[iz] * filter_array_r[ir] * fieldr[iz, ir]
        fieldt[iz, ir] = filter_array_z[iz] * filter_array_r[ir] * fieldt[iz, ir]
        fieldz[iz, ir] = filter_array_z[iz] * filter_array_r[ir] * fieldz[iz, ir]
