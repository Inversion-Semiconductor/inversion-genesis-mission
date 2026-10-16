# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines inline functions that are used both on GPU and CPU, and
used in the Compton scattering code.

These functions are compiled for GPU or CPU respectively, when imported
into the files numba_methods.py and cuda_methods.py respectively.
"""
import warp as wp
from scipy.constants import c, m_e, physical_constants

# Get additional useful constants
r_e = physical_constants["classical electron radius"][0]
PI_RE_2 = wp.constant(wp.float64(wp.pi * r_e**2))
INV_MC = wp.constant(wp.float64(1.0 / (m_e * c)))
c = wp.constant(wp.float64(c))


@wp.func
def lorentz_transform(
    p_in: wp.float64,
    px_in: wp.float64,
    py_in: wp.float64,
    pz_in: wp.float64,
    gamma: wp.float64,
    beta: wp.float64,
    nx: wp.float64,
    ny: wp.float64,
    nz: wp.float64,
):
    """
    Perform a Lorentz transform of the 4-momentum (p_in, px_in, py_in, pz_in)
    and return the results

    Parameters
    ----------
    p_in, px_in, py_in, pz_in: floats
        The coordinates of the 4-momentum
    gamma, beta: floats
        Lorentz factor and corresponding beta of the Lorentz transform
    nx, ny, nz: floats
        Coordinates of *normalized* vector that indicates
        the direction of the transform
    """
    p_parallel_in = nx * px_in + ny * py_in + nz * pz_in
    p_out = gamma * (p_in - beta * p_parallel_in)
    p_parallel_out = gamma * (p_parallel_in - beta * p_in)

    px_out = px_in + nx * (p_parallel_out - p_parallel_in)
    py_out = py_in + ny * (p_parallel_out - p_parallel_in)
    pz_out = pz_in + nz * (p_parallel_out - p_parallel_in)

    return (p_out, px_out, py_out, pz_out)


@wp.func
def get_scattering_probability(
    dt: wp.float64,
    elec_ux: wp.float64,
    elec_uy: wp.float64,
    elec_uz: wp.float64,
    elec_inv_gamma: wp.float64,
    photon_n: wp.float64,
    photon_p: wp.float64,
    photon_beta_x: wp.float64,
    photon_beta_y: wp.float64,
    photon_beta_z: wp.float64,
):
    """
    Return the probability of Comton scattering, for a given electron,
    during `dt` (taken in the frame of the simulation).

    The actual calculation is done in the rest frame of the electron,
    in order to apply the Klein-Nishina formula ; therefore `dt` is converted
    to the corresponding proper time, and the properties of the incoming photon
    flux are Lorentz-transformed.

    Parameters:
    -----------
    dt: float (in seconds)
        Time interval considered, in the frame of the simulation.
    elec_ux, elec_uy, elec_uz, elec_inv_gamma: floats (dimensionless)
        The momenta and inverse gamma factor of the emitting electron
        (in the frame of the simulation)
    photon_n, photon_p, photon_beta_x, photon_beta_y, photon_beta_z
        Properties of the photon flux (in the frame of the simulation)
    """
    # Get electron intermediate variable
    elec_gamma = wp.float64(1.0) / elec_inv_gamma

    # Get photon density and momentum in the rest frame of the electron
    transform_factor = (
        elec_gamma
        - elec_ux * photon_beta_x
        - elec_uy * photon_beta_y
        - elec_uz * photon_beta_z
    )
    photon_n_rest = photon_n * transform_factor
    photon_p_rest = photon_p * transform_factor

    # Calculate the Klein-Nishina cross-section
    k = photon_p_rest * INV_MC
    f1 = (
        wp.float64(2.0)
        * (wp.float64(2.0) + k * (wp.float64(1.0) + k) * (wp.float64(8.0) + k))
        / (
            k ** wp.float64(2)
            * (wp.float64(1.0) + wp.float64(2.0) * k) ** wp.float64(2)
        )
    )
    f2 = (
        (wp.float64(2.0) + k * (wp.float64(2.0) - k))
        * wp.log(wp.float64(1.0) + wp.float64(2.0) * k)
        / k ** wp.float64(3)
    )
    sigma = PI_RE_2 * (f1 - f2)
    # Get the electron proper time
    proper_dt_rest = dt * elec_inv_gamma
    # Calculate the probability of scattering
    p = wp.float64(1.0) - wp.exp(-sigma * photon_n_rest * c * proper_dt_rest)

    return p


@wp.func
def get_photon_density_gaussian(
    elec_x: wp.float64,
    elec_y: wp.float64,
    elec_z: wp.float64,
    ct: wp.float64,
    photon_n_lab_max: wp.float64,
    inv_laser_waist2: wp.float64,
    inv_laser_ctau2: wp.float64,
    laser_initial_z0: wp.float64,
    gamma_boost: wp.float64,
    beta_boost: wp.float64,
):
    """
    Get the photon density in the scattering Gaussian laser pulse,
    at the position of a given electron, and at the current time.

    Parameters
    ----------
    elec_x, elec_y, elec_z: floats
        The position of the given electron (in the frame of the simulation)
    ct: float
        Current time in the simulation frame (multiplied by c)
    photon_n_lab_max: float
        Peak photon density (in the lab frame)
        (i.e. at the peak of the Gaussian pulse)
    inv_laser_waist2, inv_laser_ctau2, laser_initial_z0: floats
        Properties of the Gaussian laser pulse (in the lab frame)
    gamma_boost, beta_boost: floats
        Properties of the Lorentz boost between the lab and simulation frame.

    Returns
    -------
    photon_n_sim: float
        The photon density in the frame of the simulation
    """
    # Transform electrons coordinates from simulation frame to lab frame
    elec_zlab = gamma_boost * (elec_z + beta_boost * ct)
    elec_ctlab = gamma_boost * (ct + beta_boost * elec_z)

    # Get photon density *in the lab frame*
    photon_n_lab = photon_n_lab_max * wp.exp(
        -wp.float64(2.0)
        * inv_laser_waist2
        * (elec_x ** wp.float64(2) + elec_y ** wp.float64(2))
        - wp.float64(2.0)
        * inv_laser_ctau2
        * (elec_zlab - laser_initial_z0 + elec_ctlab) ** wp.float64(2)
    )

    # Get photon density *in the simulation frame*
    photon_n_sim = gamma_boost * photon_n_lab * (wp.float64(1.0) + beta_boost)

    return photon_n_sim
