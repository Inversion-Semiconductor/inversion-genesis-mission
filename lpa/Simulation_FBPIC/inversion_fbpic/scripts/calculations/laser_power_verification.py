# TODO: make into a proper unit test script

from inversion_fbpic.utils.simulation_setup_tools import (
    calculate_laser_energy_from_a0,
    calculate_fwhm_intensity_from_laser_tau,
)
from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.utils.laser_utils import compute_laser_energy
from scipy import constants as const

if __name__ == "__main__":
    # set up laser parameters
    a0 = 2.0
    wavelength = 800e-9
    w0 = 28e-6
    tau = 38e-15

    # calculate the energy from a0
    laser_energy = calculate_laser_energy_from_a0(
        a0, wavelength * 1e6, w0, calculate_fwhm_intensity_from_laser_tau(tau)
    )

    # lasy hyperparameters
    num_points = 100
    dim = "rt"
    if dim == "rt":
        r_range = (0, 5 * w0)
        t_range = (-5 * tau, 5 * tau)
    elif dim == "xyt":
        x_range = (-5 * w0, 5 * w0)
        y_range = (-5 * w0, 5 * w0)
        t_range = (-5 * tau, 5 * tau)
    else:
        raise ValueError(f"Invalid dimension: {dim}")

    # calculate laser energy via lasy
    # create a laser object
    laser = Laser(
        dim=dim,
        lo=(
            (r_range[0], t_range[0])
            if dim == "rt"
            else (x_range[0], y_range[0], t_range[0])
        ),
        hi=(
            (r_range[1], t_range[1])
            if dim == "rt"
            else (x_range[1], y_range[1], t_range[1])
        ),
        npoints=(
            (num_points, num_points)
            if dim == "rt"
            else (num_points, num_points, num_points)
        ),
        profile=GaussianProfile(
            wavelength=wavelength,
            pol=(1, 0),
            laser_energy=1.0,
            w0=w0,
            tau=tau,
            t_peak=0.0,
        ),
    )
    # set the peak amplitude
    laser.normalize(
        a0 * const.m_e * const.c / const.e * (2.0 * const.pi * const.c / wavelength),
        "field",
    )
    # calculate the energy
    lasy_energy = compute_laser_energy(dim, laser.grid)

    # print the results
    print(f"Laser energy from lasy: {lasy_energy}")
    print(f"Laser energy from our calculation: {laser_energy}")
