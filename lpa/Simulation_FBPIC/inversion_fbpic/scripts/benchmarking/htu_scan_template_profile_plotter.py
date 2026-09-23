"""
Mimics the behavior of scanning a plasma/laser parameter for an FBPIC simulation, and plots the
 gas density profile that would be used for that simulation. Change CASE for different parameter scans.
 This script was primarily used to debug the density profiles created in FBPIC parameter scans.  This
 script is using the generalized Gaussian function, but can be updated to use other functions if needed.

Usage:
    Edit the CASE variable to select which parameter scan to plot.  If needed can update the scanned parameters
      for that CASE
    Run in any python interpreter

"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from inversion_fbpic.density_profiles.downramp_injection import (
    build_generalized_gaussian_with_downramp,
)

# Global configuration variables
CASE: int = 3  # See "main()" for list of valid parameter scans


def main() -> None:
    """
    Script entry point for scanning and plotting density profiles for different parameter scans.

    Returns:
        None
    """
    case_selection = CASE

    if case_selection == 0:
        print("Scanning peak plasma density")
        cases: list[float] = [2.9e18, 3.1e18, 3.3e18, 3.5e18, 3.7e18, 3.9e18, 4.1e18]
        directory = "htu_den_scan"
        setup_key = "peak_plasma_density"

    elif case_selection == 1:
        print("Scanning downramp height")
        cases: list[float] = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35]
        directory = "htu_ramp_height_scan"
        setup_key = "downramp_height"

    elif case_selection == 2:
        print("Scanning downramp length")
        cases: list[float] = [
            0.055e-3,
            0.060e-3,
            0.065e-3,
            0.070e-3,
            0.075e-3,
            0.080e-3,
            0.085e-3,
        ]
        directory = "htu_ramp_len_scan"
        setup_key = "downramp_length"

    elif case_selection == 3:
        print("Scanning downramp position")
        cases: list[float] = [
            1.935e-3,
            1.985e-3,
            2.035e-3,
            2.085e-3,
            2.135e-3,
            2.185e-3,
            2.235e-3,
        ]
        directory = "htu_ramp_pos_scan"
        setup_key = "downramp_position"

    elif case_selection == 4:
        print("Scanning laser focus position")
        cases: list[float] = [2.1e-3, 2.4e-3, 2.7e-3, 3.0e-3, 3.3e-3, 3.6e-3, 3.9e-3]
        directory = "htu_foc_pos_scan"
        setup_key = "laser_focal_position"

    else:
        print("no valid case")
        sys.exit(1)

    for case in cases:
        print(f"{directory}: Case {setup_key} = {case}")

        # Plot each density profile
        kwargs = {setup_key: case}
        plot_case_density_profile(**kwargs)

    plt.show()


def plot_case_density_profile(
    peak_plasma_density: float = 3.5e18,
    downramp_height: float = 1.2,
    downramp_length: float = 0.07e-3,
    downramp_position: float = 2.035e-3,
    laser_focal_position: float = 3.3e-3,
) -> None:
    """
    Plot the gas density profile for a given set of scan parameters.

    Args:
        peak_plasma_density (float): Peak plasma density in m^-3.
        downramp_height (float): Height of the downramp.
        downramp_length (float): Length of the downramp in meters.
        downramp_position (float): Position of the downramp tip in meters.
        laser_focal_position (float): Laser focal position in meters.

    Returns:
        None
    """
    # The density profile
    density_peak = 1.1
    density_alpha = 1.1e-3
    density_beta = 1.1
    density_center_location = 2.3e-3
    density_left_width = 0.50e-3

    density = build_generalized_gaussian_with_downramp(
        gauss_peak=density_peak,
        gauss_alpha=density_alpha,
        gauss_beta=density_beta,
        gauss_z0=density_center_location,
        ramp_z0=downramp_position,
        ramp_left_tau=density_left_width,
        ramp_right_width=downramp_length,
        ramp_height=downramp_height,
    )

    # plot density curve:
    z_start = 0
    z_end = 2.8 * density_alpha + density_center_location
    num = 1000

    z_arr = np.linspace(z_start, z_end, num)
    den_on_axis = density(z_arr, r=0) * peak_plasma_density
    plt.plot(z_arr, den_on_axis, ls="--")
    plt.scatter([laser_focal_position], [0.5 * peak_plasma_density])

    return


if __name__ == "__main__":
    main()
