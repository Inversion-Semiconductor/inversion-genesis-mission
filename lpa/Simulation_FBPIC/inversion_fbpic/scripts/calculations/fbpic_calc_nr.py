"""
Short calculation script for setting up FBPIC simulations.  This calculates a recommended radial grid
resolution based on the plasma density, and a recommended radial window size based on the laser spot
size at the start of the simulation.

All lengths are in meters, densities are normalized to 10^18 cm^-3.

Usage:
    Command-line usage:
        python fbpic_calc_nr.py [OPTIONS]
        fbpic-calc-nr [OPTIONS]

    Required Arguments:
        -lam, --wavelength FLOAT
            Laser wavelength in meters.

        -np, --plasma-density FLOAT
            Plasma density in units of 10^18 cm^-3.

        -w0, --w0 FLOAT
            Laser focus spot size (w_0) in meters.

        -zf, --z-foc FLOAT
            Distance from start of simulation to focus in meters.

    Examples:
        fbpic-calc-nr -lam 800e-9 -np 1.4 -w0 21e-6 -zf 4.68e-3
"""

import argparse

from scipy.constants import pi, e, epsilon_0, c, m_e
import numpy as np


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Calculate recommended radial grid resolution and window size for FBPIC simulations."
    )

    parser.add_argument(
        "-lam",
        "--wavelength",
        type=float,
        required=True,
        help="Laser wavelength in meters (e.g. 800e-9)",
    )

    parser.add_argument(
        "-np",
        "--plasma-density",
        type=float,
        required=True,
        help="Plasma density in units of 10^18 cm^-3 (e.g. 1.4)",
    )

    parser.add_argument(
        "-w0",
        "--w0",
        type=float,
        required=True,
        help="Laser spot size (w_0) in meters (e.g. 21e-6)",
    )

    parser.add_argument(
        "-zf",
        "--z-foc",
        type=float,
        required=True,
        help="Distance to focus in meters (e.g. 4.68e-3)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    wavelength = args.wavelength
    plasma_density = args.plasma_density
    w_0 = args.w0
    z_foc = args.z_foc

    if wavelength <= 0:
        raise ValueError(f"Wavelength must be positive: {wavelength}")
    if plasma_density <= 0:
        raise ValueError(f"Plasma density must be positive: {plasma_density}")
    if w_0 <= 0:
        raise ValueError(f"w_0 must be positive: {w_0}")

    # Calculations
    n_p = plasma_density * 1e18 * 100**3  # n_p is in meters
    k_p = np.sqrt((n_p * e**2) / (m_e * epsilon_0 * c**2))
    dr = 0.15 / k_p  # Guideline according to PIP4 CDR
    print(f"Plasma wavenumber at peak: {k_p:.3f} m^-1")

    zr = pi * np.square(w_0) / wavelength
    w_start = w_0 * np.sqrt(1 + (z_foc / zr) ** 2)
    print(f"Spot size at focus: {w_0*1e6:.2f} um")
    print(f"Distance to sim start: {z_foc*1e3:.2f} mm")
    print(f"Starting spot size: {w_start*1e6:.2f} um")
    minimum_r_range = w_start * 4.5  # Recommended to avoid sim boundary
    minimum_r_cells = minimum_r_range / dr

    print()
    print(f"Recommended Maximum dr: {dr*1e6} um")
    print(f"Recommended Minimum R_range: {minimum_r_range*1e6} um")
    print(f"Corresponding R Grid Size: {round(minimum_r_cells)}")


if __name__ == "__main__":
    main()
