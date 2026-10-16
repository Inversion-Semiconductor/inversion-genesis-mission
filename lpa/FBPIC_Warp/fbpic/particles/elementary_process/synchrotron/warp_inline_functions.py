# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines inline functions that are compiled for both GPU and CPU, and
used in the syncrotron radiation code.
"""
import warp as wp
from scipy.constants import c, e, m_e

e_mc = wp.constant(wp.float64(e / (m_e * c)))


@wp.func
def get_angles(
    ux: wp.float64,
    uy: wp.float64,
    uz: wp.float64,
):
    """
    Calculate angular projections of particle momentum vector on
    `(theta_x, theta_y)` plane.

    Parameters
    ----------
    ux, uy, uz: floats
        Components of particles normalized momentum.

    Returns
    -------
    theta_x, theta_y: floats
        angular projections of particle momentum vector in radians
    """

    theta_x = wp.atan2(ux, uz)
    theta_y = wp.atan2(uy, uz)

    return (theta_x, theta_y)


@wp.func
def get_linear_coefficients(
    x: wp.float64,
    xmin: wp.float64,
    dx: wp.float64,
):
    """
    Calculate shape coefficients and index for the 1D linear
    interpolation on a uniform grid (used for angle projections)

    Parameters
    ----------
    x: float
        Coordinate to be projected.

    xmin: float
         Grid origin

    dx: float
        Grid step

    Returns
    -------
    ix: integer
        Index of the cell that contains the point

    s0, s1: floats
        Weights projected to the left and right nodes of the cell
    """
    s_ix = (x - xmin) / dx
    ix = wp.floor(s_ix)
    s1 = s_ix - ix
    s0 = wp.float64(1.0) - s1
    return int(ix), s0, s1


@wp.func
def get_particle_radiation(
    ux: wp.float64,
    uy: wp.float64,
    uz: wp.float64,
    w: wp.float64,
    Ex: wp.float64,
    Ey: wp.float64,
    Ez: wp.float64,
    cBx: wp.float64,
    cBy: wp.float64,
    cBz: wp.float64,
    gamma_inv: wp.float64,
    Larmore_factor_density: wp.float64,
    Larmore_factor_momentum: wp.float64,
    SR_dxi: wp.float64,
    SR_xi_data: wp.array(dtype=wp.float64),
    omega_ax: wp.array(dtype=wp.float64),
    spect_loc: wp.array(dtype=wp.float64),
):
    """
    Calculate spectal energy distribution emitted by the particle.

    Parameters
    ----------
    ux, uy, uz, w: floats
        Components momentum and weight of the particle

    Ex, Ey, Ez: float
         Components of electric field on the particle (V/m)

    cBx, cBy, cBz: float
         Components of magnetic field on the particle multiplied by
         the speed of light (V/m)

    gamma_inv: float
        Reciprocal of particle Lorentz factor

    Larmore_factor_density: float
        Normalization factor for spectral-angular density,
        `e**2 * dt / (6 * np.pi * epsilon_0 * c * hbar * d_theta_x * d_theta_y)`

    Larmore_factor_momentum: float
        Normalization factor for the photon momentum,
        `e**2 * dt / ( 6 * np.pi * epsilon_0 * c**2 )`

    omega_ax: 1D vector of floats
        frequencies on which spectrum is calculated

    spect_loc: 1D vector of floats
        calculated spectral density of the radiation

    Returns
    -------
    spect_loc: 1D vector of floats
        calculated spectral density of the radiation
    """
    d_omega = omega_ax[1] - omega_ax[0]
    N_omega_src = SR_xi_data.shape[0]
    gamma = wp.float64(1.0) / gamma_inv

    # get normalized velocity
    beta_x = ux * gamma_inv
    beta_y = uy * gamma_inv
    beta_z = uz * gamma_inv

    # get momentum time derivative
    dt_ux = -e_mc * (Ex + beta_y * cBz - beta_z * cBy)
    dt_uy = -e_mc * (Ey + beta_z * cBx - beta_x * cBz)
    dt_uz = -e_mc * (Ez + beta_x * cBy - beta_y * cBx)

    # get Lorentz factor derivative
    dt_gamma = -e_mc * (Ex * beta_x + Ey * beta_y + Ez * beta_z)

    # get acceleration
    dt_beta_x = (dt_ux - beta_x * dt_gamma) * gamma_inv
    dt_beta_y = (dt_uy - beta_y * dt_gamma) * gamma_inv
    dt_beta_z = (dt_uz - beta_z * dt_gamma) * gamma_inv

    # calculate Larmore energy
    Energy_norm = (
        w
        * gamma ** wp.float64(6)
        * (
            dt_beta_x ** wp.float64(2)
            + dt_beta_y ** wp.float64(2)
            + dt_beta_z ** wp.float64(2)
            - (beta_y * dt_beta_z - beta_z * dt_beta_y) ** wp.float64(2)
            - (beta_z * dt_beta_x - beta_x * dt_beta_z) ** wp.float64(2)
            - (beta_x * dt_beta_y - beta_y * dt_beta_x) ** wp.float64(2)
        )
    )

    # calculate emitted radiation momentum
    Momentum_Larmor = Larmore_factor_momentum * Energy_norm / w
    u_abs_inv = wp.float64(1.0) / wp.sqrt(ux * ux + uy * uy + uz * uz)
    ux_ph = Momentum_Larmor * ux * u_abs_inv
    uy_ph = Momentum_Larmor * uy * u_abs_inv
    uz_ph = Momentum_Larmor * uz * u_abs_inv

    # calculate critical frequency
    dt_beta_abs2 = (
        dt_beta_x ** wp.float64(2)
        + dt_beta_y ** wp.float64(2)
        + dt_beta_z ** wp.float64(2)
    )
    beta_abs2_inv = wp.float64(1.0) / (
        beta_x ** wp.float64(2) + beta_y ** wp.float64(2) + beta_z ** wp.float64(2)
    )
    beta_dot_dt_beta = beta_x * dt_beta_x + beta_y * dt_beta_y + beta_z * dt_beta_z

    omega_c = (
        wp.float64(1.5)
        * gamma ** wp.float64(3)
        * beta_abs2_inv
        * wp.sqrt(dt_beta_abs2 - beta_dot_dt_beta ** wp.float64(2) * beta_abs2_inv)
    )

    # discard too low critical frequencies as not resolved
    if omega_c < wp.float64(4.0) * d_omega:
        for i_omega in range(omega_ax.shape[0]):
            spect_loc[i_omega] = wp.float64(0.0)
        return (spect_loc, wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))

    omega_c_inv = wp.float64(1.0) / omega_c

    Density_Larmore = Larmore_factor_density * Energy_norm * omega_c_inv

    # Loop over the frequencies to project the spectrum
    for i_omega in range(omega_ax.shape[0]):
        xi_loc = omega_ax[i_omega] * omega_c_inv
        s_ix_src = xi_loc / SR_dxi
        ix_src = wp.floor(s_ix_src)

        if ix_src >= wp.float64(N_omega_src - 1):
            spect_loc[i_omega] = wp.float64(0.0)
        else:
            s1 = s_ix_src - ix_src
            s0 = wp.float64(1.0) - s1
            ix_src_int = int(ix_src)
            S_xi_loc = SR_xi_data[ix_src_int] * s0 + SR_xi_data[ix_src_int + 1] * s1
            spect_loc[i_omega] = Density_Larmore * S_xi_loc

    return (spect_loc, ux_ph, uy_ph, uz_ph)
