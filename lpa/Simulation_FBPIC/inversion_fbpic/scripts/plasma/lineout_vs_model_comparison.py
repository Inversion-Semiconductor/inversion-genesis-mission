"""
Script to compare experimental lineouts with approximate density profiles.

This script loads lineout data from numpy files and compares it with
approximate density profiles generated using the density profile functions.

Usage:
    Run `plasma_lineout_analysis.py` first to generate the output .npy file
    Pick a CASE for the "generate_approximate_profile()" function to use.  Within that CASE adjust the parameters as
      needed to get a match between the data.  There is no fitting here, just manually parameter tweaking.
    Can change the LINEOUT_SAMPLE parameter to use a different lineout stored within LINEOUT_FILE.  Set to None to plot
      all lineouts contained in LINEOUT_FILE
    Run with any python interpreter.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from inversion_fbpic.density_profiles.downramp_injection import (
    build_gaussian_plus_triangle_z_density_function,
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
    num_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
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
    if CASE == 0:
        density_func = build_gaussian_plus_triangle_z_density_function(
            sigma=1.4e-3,
            center_location=3.0e-3,
            z_tip=2.55e-3,
            left_width=0.9e-3,
            right_width=0.15e-3,
            triangle_height=1.5,
        )

    elif CASE == 1:
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

    else:
        print("Specify a valid 'CASE'")
        sys.exit()

    # Calculate density at r=0 (along the axis)
    r_positions = np.zeros_like(z_positions)  # r=0 for all points
    density_values = density_func(z_positions, r_positions)

    return z_positions, density_values


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
    if LINEOUT_SAMPLE is not None:
        approx_scaled = density_approx * (
            lineouts[LINEOUT_SAMPLE, :].max() / density_approx.max()
        )
    else:
        approx_scaled = density_approx * (lineouts.max() / density_approx.max())
    plt.plot(
        z_approx_mm,
        approx_scaled,
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


def calculate_correlation(
    lineouts: np.ndarray,
    z_approx: np.ndarray,
    density_approx: np.ndarray,
) -> dict:
    """
    Calculate correlation between experimental and approximate profiles.

    Args:
        lineouts (np.ndarray): Array of experimental lineouts.
        z_approx (np.ndarray): Z positions for approximate profile.
        density_approx (np.ndarray): Approximate density values.

    Returns:
        dict: Dictionary containing correlation metrics.
    """
    # Interpolate approximate profile to match experimental x-axis
    x_lineouts = np.linspace(0, 5, lineouts.shape[1])
    z_approx_mm = z_approx * 1000  # Convert to mm

    # Scale approximate profile
    approx_scaled = density_approx * (lineouts.max() / density_approx.max())

    # Interpolate to match experimental grid
    from scipy.interpolate import interp1d

    approx_interp = interp1d(
        z_approx_mm, approx_scaled, bounds_error=False, fill_value=0
    )
    approx_on_exp_grid = approx_interp(x_lineouts)

    # Calculate correlations for each lineout
    correlations = []
    for i in range(lineouts.shape[0]):
        corr = np.corrcoef(lineouts[i, :], approx_on_exp_grid)[0, 1]
        correlations.append(corr)

    # Calculate average correlation
    avg_correlation = np.mean(correlations)
    std_correlation = np.std(correlations)

    # Calculate R-squared for average lineout
    avg_lineout = np.mean(lineouts, axis=0)
    ss_res = np.sum((avg_lineout - approx_on_exp_grid) ** 2)
    ss_tot = np.sum((avg_lineout - np.mean(avg_lineout)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        "individual_correlations": correlations,
        "average_correlation": avg_correlation,
        "correlation_std": std_correlation,
        "r_squared": r_squared,
    }


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
    z_approx, density_approx = generate_approximate_profile()

    print(f"Approximate profile generated with {len(z_approx)} points")
    print(f"Z range: {z_approx[0] * 1000:.2f} to {z_approx[-1] * 1000:.2f} mm")
    print(f"Density range: {density_approx.min():.3f} to {density_approx.max():.3f}")

    print("\nCalculating correlation metrics...")
    correlation_metrics = calculate_correlation(lineouts, z_approx, density_approx)

    print(
        f"Average correlation: {correlation_metrics['average_correlation']:.3f}  b1 {correlation_metrics['correlation_std']:.3f}"
    )
    print(f"R-squared (average lineout): {correlation_metrics['r_squared']:.3f}")
    print(
        f"Individual correlations: {[f'{c:.3f}' for c in correlation_metrics['individual_correlations']]}"
    )

    print("\nCreating comparison plot...")
    plot_comparison(lineouts, z_approx, density_approx)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
