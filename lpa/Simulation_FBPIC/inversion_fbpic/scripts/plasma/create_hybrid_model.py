"""
Script to take the standard, full-model of the HASO lineout and instead use a model for just the downramp.

Takes a similar approach to lineout_vs_model_comparison.py, then
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from inversion_fbpic.density_profiles.downramp_injection import (
    build_generalized_gaussian_profile,
    build_generalized_gaussian_with_downramp,
)

LINEOUT_FILE: str = (
    "./plasma_lineouts_data.npy"  # Path to lineout data saved from plasma_lineout_analysis.py
)
LINEOUT_SAMPLE: Optional[int] = 5

CASE: int = 1  # 0 for Gaussian, 1 for Generalized Gaussian
DATA_RANGE: float = (
    660 * 10.1e-3
)  # Set to match the HASO's 660 wide pixel image at 10.1 um/pixel


def generate_approximate_profile(
    z_range: tuple[float, float] = (0, DATA_RANGE / 1e3),
    num_points: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an approximate density profile for comparison.

    Args:
        z_range (tuple[float, float], optional): Range of z values (start, end) in meters. Defaults to (0, DATA_RANGE / 1e3).
        num_points (int, optional): Number of points to generate. Defaults to 1000.

    Returns:
        tuple[np.ndarray, np.ndarray]: (z_positions, density_values)
    """
    # Create z positions
    z_positions = np.linspace(z_range[0], z_range[1], num_points)

    # Generate the density function
    density_func = build_generalized_gaussian_with_downramp(
        gauss_peak=1.183455,
        gauss_alpha=1.701e-3,
        gauss_beta=1.27746,
        gauss_z0=3.062e-3,
        ramp_z0=2.559e-3,
        ramp_left_tau=1.278e-3,
        ramp_right_width=0.13e-3,
        ramp_height=1.471463,
    )

    generalized_gaussian = build_generalized_gaussian_profile(
        gauss_peak=1.183455,
        gauss_z0=3.062e-3,
        gauss_alpha=1.701e-3,
        gauss_beta=1.27746,
    )
    # Calculate density at r=0 (along the axis)
    r_positions = np.zeros_like(z_positions)  # r=0 for all points
    density_values = np.array(density_func(z_positions, r_positions))
    gaussian_values = np.array(generalized_gaussian(z_positions, r_positions))

    peak_gaussian_index: int = int(np.argmax(gaussian_values))
    ratio: float = float(density_values[peak_gaussian_index]) / float(gaussian_values[peak_gaussian_index])

    return z_positions, density_values, gaussian_values*ratio


def load_lineout_data(
    file_path: str = LINEOUT_FILE,
) -> np.ndarray | None:
    """
    Load lineout data from numpy file.

    Args:
        file_path (str, optional): Path to the numpy file containing lineout data. Defaults to LINEOUT_FILE global var

    Returns:
        np.ndarray | None: Array of lineouts where each row is a lineout, or None if file not found.
    """
    if not Path(file_path).exists():
        print(f"Error: File {file_path} not found!")
        return None

    lineouts = np.load(file_path)
    print(f"Loaded lineout data: {lineouts.shape}")
    print(f"Number of lineouts: {lineouts.shape[0]}")
    print(f"Points per lineout: {lineouts.shape[1]}")

    return lineouts


def plot_comparison(
    lineouts: np.ndarray,
    z_approx: np.ndarray,
    density_approx: np.ndarray,
) -> None:
    """
    Plot comparison between experimental lineouts and approximate profile.

    Args:
        lineouts (np.ndarray): Array of experimental lineouts.
        z_approx (np.ndarray): Z positions for approximate profile.
        density_approx (np.ndarray): Approximate density values.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    # Convert approximate z positions to mm for plotting
    z_approx_mm = z_approx * 1000  # Convert m to mm

    # Plot 1: Individual lineouts with approx
    colors = plt.cm.viridis(np.linspace(0, 1, lineouts.shape[0]))
    # Create x-axis for lineouts (0 to 5 mm)
    x_lineouts = np.linspace(0, DATA_RANGE, lineouts.shape[1])

    if LINEOUT_SAMPLE is not None:
        plt.plot(
            x_lineouts,
            lineouts[LINEOUT_SAMPLE, :],
            color=colors[LINEOUT_SAMPLE],
            alpha=0.8,
            linewidth=1,
            label=f"Lineout {LINEOUT_SAMPLE + 1}",
        )
    else:
        for i in range(lineouts.shape[0]):
            plt.plot(
                x_lineouts,
                lineouts[i, :],
                color=colors[i],
                alpha=0.8,
                linewidth=1,
                label=f"Lineout {i + 1}",
            )

    # Plot approximate profile (scaled to match)
    # Scale the approximate profile to match the experimental range
    plt.plot(
        z_approx_mm,
        density_approx*135,
        "r-",
        linewidth=2,
        label="Approximate Profile",
        alpha=0.8,
    )

    plt.xlabel("Position (mm)")
    plt.ylabel("Intensity / Density")
    plt.title("Experimental Lineouts vs Approximate Density Profile")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, DATA_RANGE)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Script entry point to run the comparison between experimental and approximate profiles.

    Returns:
        None
    """
    print("Loading experimental lineout data...")
    lineouts = load_lineout_data()

    if lineouts is None:
        print("Failed to load lineout data. Exiting.")
        return

    print("\nGenerating approximate density profile...")
    # Generate approximate profile with default parameters
    z_approx, density_approx, gaussian_approx = generate_approximate_profile()

    isolated_ramp = density_approx - gaussian_approx

    x_lineouts = np.linspace(z_approx[0], z_approx[-1], lineouts.shape[1])
    haso_data = np.interp(z_approx, x_lineouts, lineouts[LINEOUT_SAMPLE, :])/135
    approximate_profile = haso_data - isolated_ramp

    print("\nCreating comparison plot...")
    plot_comparison(lineouts, z_approx, approximate_profile)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
