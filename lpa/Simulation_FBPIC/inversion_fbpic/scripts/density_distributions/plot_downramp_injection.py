"""
Generates and plots density profiles for various cases that use downramp injection or Gaussian profiles.

This script demonstrates usage of the density profile functions in downramp_injection.py by plotting several example profiles.

Usage:
    python plot_downramp_injection.py [case]
"""

import sys
from typing import Optional
import numpy as np
from scipy.constants import c

from visualize import plot_2d_density_profile
import inversion_fbpic.density_profiles.downramp_injection as dj

CASE_DESCRIPTIONS: dict[str, str] = {
    "gtrihtu": "Gaussian + Triangle Close to HTU",
    "ggshock": "Generalized Gaussian + Exponential/Linear Shock",
    "gtrian": "Gaussian + Triangle",
    "pwlin": "Piecewise Downramp",
    "gshort": "Short Gaussian for Eksma Laser",
    "moddrp": "Modulated Downramp",
    "xumod": "Xu's XFEL Modulated Downramp",
    "pwconst": "Piecewise Constant Downramp",
    "ftblade": "Smoothed Flattop with Asymmetric Blade Downramp",
}


def main(case: Optional[str] = None):
    """
    Script entry point for generating and plotting density profiles using downramp injection and Gaussian profiles.

    Select a profile by passing a case string (either directly to this function or from CLI).
    If ``case`` is missing or not a known key, raises ``ValueError`` with the list of valid cases.

    Returns:
        None
    """
    plot_peak_density: Optional[float] = None

    if case is None or case not in CASE_DESCRIPTIONS:
        valid_cases = "\n".join(
            [f"{c}\t|\t{CASE_DESCRIPTIONS[c]}" for c in CASE_DESCRIPTIONS]
        )
        raise ValueError(f"Unknown case '{case}'. Expected one of:\n{valid_cases}")

    print(f"Plasma profile: {CASE_DESCRIPTIONS[case]}")

    # Example using a Gaussian plus triangle close to HTU
    if case == "gtrihtu":  # TODO: convert if elif ... to match case in a future PR
        sigma = 1.4e-3
        center_location = 3.7e-3
        z_tip = 3.2e-3
        left_width = 0.90e-3
        right_width = 0.11e-3
        triangle_height = 1.5
        density = dj.build_gaussian_plus_triangle_z_density_function(
            sigma=sigma,
            center_location=center_location,
            z_tip=z_tip,
            left_width=left_width,
            right_width=right_width,
            triangle_height=triangle_height,
        )
        plot_end = 2.8 * sigma + center_location
        plot_peak_density = 4e18

    # Example using the generalized Gaussian and exponential upramp
    elif case == "ggshock":
        upramp_extension = 2e-3

        gauss_peak = 1.183455
        gauss_alpha = 0.001701
        gauss_beta = 1.277460
        gauss_z0 = 0.003062 + upramp_extension
        ramp_z0 = 0.002559 + upramp_extension
        ramp_left_tau = 0.001278
        ramp_right_width = 0.000130
        ramp_height = 1.471463

        density = dj.build_generalized_gaussian_with_downramp(
            gauss_peak=gauss_peak,
            gauss_alpha=gauss_alpha,
            gauss_beta=gauss_beta,
            gauss_z0=gauss_z0,
            ramp_z0=ramp_z0,
            ramp_left_tau=ramp_left_tau,
            ramp_right_width=ramp_right_width,
            ramp_height=ramp_height,
        )
        plot_end = 2.8 * gauss_alpha + gauss_z0

        plot_peak_density = 4e18

    # Example using the Gaussian plus triangle
    elif case == "gtrian":
        density = dj.build_gaussian_plus_triangle_z_density_function(
            sigma=0.35e-3,
            center_location=1.0e-3,
            z_tip=0.8e-3,
            left_width=0.3e-3,
            right_width=0.05e-3,
            triangle_height=1.4,
        )
        plot_end = 2.0e-3

    # Example using the piecewise linear downramp function
    elif case == "pwlin":
        upramp_len = 40e-6
        downramp_pos = 50e-6
        downramp_len = 5e-6
        downramp_height = 0.35
        plateau_end_pos = 170e-6
        plateau_downramp_len = 150e-6
        density = dj.build_piecewise_linear_downramp(
            upramp_length=upramp_len,
            downramp_start_position=downramp_pos,
            downramp_length=downramp_len,
            downramp_height_ratio=downramp_height,
            plateau_end_position=plateau_end_pos,
            plateau_downramp_length=plateau_downramp_len,
        )
        plot_end = plateau_end_pos + plateau_downramp_len + 10e-6

    elif case == "gshort":
        sigma = 0.3e-3
        center_location = 3 * sigma
        density = dj.build_gaussian_profile(
            sigma=sigma, center_location=center_location
        )
        plot_end = center_location + 3 * sigma

    # Example using the piecewise linear downramp function with a modulation
    #  relevant for inversion purposes
    elif case == "moddrp":
        upramp_len = 200e-6
        downramp_pos = 250e-6
        downramp_len = 60e-6
        downramp_height = 0.95
        plateau_end_pos = 400e-6
        plateau_downramp_len = 50e-6

        modulation_amplitude = 2e-3
        modulation_wavelength = 1.2e-6

        plot_peak_density = 1.0e18  # cm-3
        wp0 = 5.64e4 * np.sqrt(plot_peak_density)
        kp0 = wp0 / (c * 100)
        kp0_m = kp0 * 100

        print(f"Ramp Len: {(downramp_len) * kp0_m:.02f} kp0")
        print(f" Plasma Density np0: {plot_peak_density} cm^-3")
        print(f" Plasma frequency kp0: {kp0} cm^-1")

        density = dj.build_piecewise_linear_modulated_downramp(
            upramp_length=upramp_len,
            downramp_start_position=downramp_pos,
            downramp_length=downramp_len,
            downramp_height_ratio=downramp_height,
            plateau_end_position=plateau_end_pos,
            plateau_downramp_length=plateau_downramp_len,
            modulation_amplitude=modulation_amplitude,
            modulation_wavelength=modulation_wavelength,
        )
        plot_end = plateau_end_pos + plateau_downramp_len + 10e-6

    # Example using the piecewise linear downramp function with modulation
    #  inspired by Xinlu Xu's 2022 Nat.Comm. Paper
    elif case == "xumod":
        upramp_len = 200e-6
        downramp_pos = 250e-6
        downramp_len = 30e-6
        downramp_height = 0.9
        plateau_end_pos = 400e-6
        plateau_downramp_len = 50e-6

        modulation_amplitude = 2e-3
        modulation_wavelength = 0.4e-6

        plot_peak_density = 1.97e19  # cm-3
        wp0 = 5.64e4 * np.sqrt(plot_peak_density)
        kp0 = wp0 / (c * 100)
        kp0_m = kp0 * 100
        print(f"Ramp Start: {(downramp_pos)*kp0_m:.02f} kp0")
        print(f"Ramp End: {(downramp_pos+downramp_len)*kp0_m:.02f} kp0")
        print(f"Ramp Len: {(downramp_len)*kp0_m:.02f} kp0")

        print(f" Plasma Density np0: {plot_peak_density} cm^-3")
        print(f" Plasma frequency kp0: {kp0} cm^-1")

        grid_size = 1 / 512
        print(f"Grid Size {grid_size}*k_p0^-1 = {(grid_size/kp0_m)*1e9} nm")
        grid_extent = 12
        print(f"Grid Extent {grid_extent}*k_p0^-1 = {(grid_extent / kp0_m) * 1e6} um")

        density = dj.build_piecewise_linear_modulated_downramp(
            upramp_length=upramp_len,
            downramp_start_position=downramp_pos,
            downramp_length=downramp_len,
            downramp_height_ratio=downramp_height,
            plateau_end_position=plateau_end_pos,
            plateau_downramp_length=plateau_downramp_len,
            modulation_amplitude=modulation_amplitude,
            modulation_wavelength=modulation_wavelength,
        )
        plot_end = plateau_end_pos + plateau_downramp_len + 10e-6

    elif case == "pwconst":
        downramp_pos = 50e-6
        downramp_height = 0.35
        plateau_end_pos = 170e-6
        density = dj.build_piecewise_constant_downramp(
            downramp_position=downramp_pos,
            downramp_height_ratio=downramp_height,
            plateau_end_position=plateau_end_pos,
            plateau_cutoff_smoothness=10e-6,
        )
        plot_end = plateau_end_pos + 10e-6

    elif case == "ftblade":
        density = dj.build_flattop_with_asymmetric_blade_downramp_density(
            flattop_width=1e-3,
            flattop_upramp_length=100e-6,
            flattop_downramp_length=100e-6,
            blade_relative_height=0.5,
            blade_upramp_length=200e-6,
            blade_downramp_length=50e-6,
            downramp_position=250e-6,
        )
        plot_end = 1e-3 + 200e-6

    if plot_peak_density is None:
        plot_peak_density = 1

    plot_2d_density_profile(
        z_start=-20e-6,
        z_end=plot_end,
        r_width=100e-6,
        peak_density_value=plot_peak_density,
        density_function=density,
        units="mm",
        num=10000,
    )


if __name__ == "__main__":
    selected_case = sys.argv[1] if len(sys.argv) > 1 else None
    main(selected_case)
