"""
Test script that mimics the simulation setup for FBPIC run scripts and plots
the density profile used in the simulation.
"""

import sys
import numpy as np
from fbpic.main import Simulation
import matplotlib.pyplot as plt
from inversion_fbpic.density_profiles.downramp_injection import build_generalized_gaussian_with_downramp


# Default parameters for this scan
DEFAULT_PARAMS = {
    "peak_plasma_density": 3.6e18,
    "downramp_height": 1.471463,
    "downramp_length": 0.130e-3,
    "downramp_position": 3.0e-3,
    "simulation_upramp_extension": 1e-3,
}


def setup_simulation(
    case_name: str,
    peak_plasma_density: float,
    downramp_height: float,
    downramp_length: float,
    downramp_position: float,
    simulation_upramp_extension: float,
):
    """
    Sets up and configures the FBPIC simulation.

    Args:
        simulation_set_name: Name of the simulation set (used for output directory)
        case_name: Name of the specific case (used for output directory)
        peak_plasma_density: Peak plasma density in cm^-3
        downramp_height: Height of the downramp
        downramp_length: Length of the downramp in meters
        downramp_position: Position of the downramp in meters
        laser_focal_position: Laser focal position in meters
        laser_energy_scale: Scaling factor for laser energy
        laser_focus_size: Laser focus size in meters
        laser_temporal_width: Laser temporal width in seconds
        simulation_upramp_extension: Extension for simulation upramp in meters

    Returns:
        tuple[Simulation, float]: A tuple containing a configured Simulation object ready to run and the interaction time.
    """

    # The density profile
    gauss_alpha = 0.001701
    gauss_z0 = 0.003062 + simulation_upramp_extension
    ramp_z0 = downramp_position + simulation_upramp_extension

    density = build_generalized_gaussian_with_downramp(
        gauss_peak=1.183455,
        gauss_alpha=gauss_alpha,
        gauss_beta = 1.277460,
        gauss_z0=gauss_z0,
        ramp_z0=ramp_z0,
        ramp_left_tau=0.001278,
        ramp_right_width=downramp_length,
        ramp_height=downramp_height,
    )

    # The interaction length of the simulation (meters)
    L_interact: float = 2.8 * gauss_alpha + gauss_z0

    z_arr = np.linspace(start=0, stop=L_interact, num=1000)
    plt.plot(z_arr*1e3, density(z=z_arr, r=0)*peak_plasma_density, label=case_name)

def main() -> None:
    """
    Given a command-line input, runs a parameter scan using FBPIC simulations of the given case
    """

    case_selection = 0

    scan_cases = [
        (
            "Scanning downramp position",
            [2.90e-3, 2.95e-3, 3.00e-3, 3.05e-3, 3.10e-3],
            "htu_ramp_pos_scan",
            "downramp_position",
        ),
        (
            "Scanning peak plasma density",
            [3.2e18, 3.4e18, 3.6e18, 3.8e18, 4.0e18],
            "htu_den_scan",
            "peak_plasma_density",
        ),
        (
            "Scanning laser energy",
            [0.7, 0.8, 0.9, 1.0, 1.1],
            "htu_energy_scan",
            "laser_energy_scale",
        ),
        (
            "Scanning laser focal position",
            [1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3, 3.0e-3],
            "htu_foc_pos_scan",
            "laser_focal_position",
        ),
    ]

    if 0 <= case_selection < len(scan_cases):
        description, cases, directory, setup_key = scan_cases[case_selection]
        print(description)
    else:
        print("no valid case")
        sys.exit(1)

    for case in cases:
        print(f"{directory}: Case {setup_key} = {case}")

        # Set up the simulation
        kwargs = DEFAULT_PARAMS.copy()
        kwargs.update({setup_key: case})

        setup_simulation(case_name=str(case), **kwargs)

    plt.legend()
    plt.ylabel("Density (cm^-3)")
    plt.xlabel("Longitudinal Axis (mm)")
    plt.title("Simulated Longitudinal Density Profiles")
    plt.show()

if __name__ == "__main__":
    main()
