"""Electron beam dispersion analysis and visualization.

This module provides functionality to analyze electron beam dispersion by plotting
transverse position vs energy distributions to identify potential dispersion effects.

Usage:
    First, extract the ebeam from the full .h5 output file using either
    "extract_ebeam_from_hdf5.py" or "extract_ebeam_from_set.py". The extracted ebeam
    will be located in trimmed-down h5 files in an "ebeam" directory.

    Command-line usage:
        python plot_dispersion.py [OPTIONS]

    Optional Arguments:
        -d, --diag-folder PATH
            Path to the ebeam .h5 files directory. If not provided, uses the default
            value defined in the script.

        -i, --iteration-number INT
            Which dump to load from DIAG_FOLDER. Set to "-1" for final dump.
            Default: -1

        -s, --species STR
            Which species to load from DIAG_FOLDER. Typically "electrons".
            Default: "electrons"

        --min-z FLOAT|None
            Minimum z position when selecting electrons. Use 'None' or 'null' to
            explicitly set to None.
            Default: None

        --max-z FLOAT|None
            Maximum z position when selecting electrons. Use 'None' or 'null' to
            explicitly set to None.
            Default: None

    Examples:
        # Use all defaults
        python plot_dispersion.py

        # Specify a different diagnostics folder
        python plot_dispersion.py -d /path/to/hdf5

        # Load a specific iteration and set z window
        python plot_dispersion.py -i 100 --min-z 8000e-6 --max-z 10000e-6
"""

from pathlib import Path
import argparse
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as const

import inversion_fbpic.utils.analysis as an
from inversion_fbpic.utils.argparse_utils import float_or_none

# Default configuration variables
DEFAULT_DIAG_FOLDER: Optional[Path] = None
DEFAULT_MIN_Z: Optional[float] = None  # Minimum z when selecting electrons
DEFAULT_MAX_Z: Optional[float] = None  # Maximum z when selecting electrons
# Minimum normalized longitudinal momentum uz [mc] when loading beam data; restricts
# to the main beam and excludes low-energy / trailing electrons. Analysis assumes
# this cutoff; change only if the physical regime differs.
DEFAULT_MIN_UZ: float = 220.0
DEFAULT_ITERATION_NUMBER: int = -1  # Which dump to load from DIAG_FOLDER
DEFAULT_SPECIES: str = "n_elec"  # Which species to load from DIAG_FOLDER


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze electron beam dispersion by plotting transverse position vs energy."
    )

    parser.add_argument(
        "-d",
        "--diag-folder",
        type=Path,
        default=DEFAULT_DIAG_FOLDER,
        help="Path to the ebeam .h5 files directory",
    )

    parser.add_argument(
        "-i",
        "--iteration-number",
        type=int,
        default=DEFAULT_ITERATION_NUMBER,
        help="Which dump to load from DIAG_FOLDER. Set to '-1' for final dump",
    )

    parser.add_argument(
        "-s",
        "--species",
        type=str,
        default=DEFAULT_SPECIES,
        help="Which species to load from DIAG_FOLDER. Typically 'electrons'",
    )

    parser.add_argument(
        "--min-z",
        type=float_or_none,
        default=DEFAULT_MIN_Z,
        help="Minimum z position when selecting electrons. Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "--max-z",
        type=float_or_none,
        default=DEFAULT_MAX_Z,
        help="Maximum z position when selecting electrons. Use 'None' or 'null' to explicitly set to None",
    )

    return parser.parse_args()


def load_and_filter_beam_data(
    diag_folder: Optional[Path],
    species: str,
    iteration_number: int,
    min_z: Optional[float],
    max_z: Optional[float],
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Load and filter electron beam data.

    Loads particles with uz >= DEFAULT_MIN_UZ (normalized longitudinal momentum
    in mc), then applies optional min_z / max_z cuts. The uz cutoff restricts
    to the main beam; see DEFAULT_MIN_UZ for the physical assumption.

    Args:
        diag_folder: Path to the ebeam .h5 files directory.
        species: Which species to load.
        iteration_number: Which dump to load.
        min_z: Minimum z position when selecting electrons.
        max_z: Maximum z position when selecting electrons.

    Returns:
        Tuple containing filtered beam arrays (x, y, z, ux, uy, uz, w).

    Raises:
        FileNotFoundError: If beam data cannot be found.
        Exception: If beam data cannot be loaded.
    """
    if diag_folder is None:
        raise RuntimeError(
            "Beam data path not set. Use --diag-folder or set DEFAULT_DIAG_FOLDER in the script."
        )
    try:
        x, y, z, ux, uy, uz, w, q, ts = an.load_beam_data(
            diag_folder,
            species,
            iteration=iteration_number,
            select={"uz": (DEFAULT_MIN_UZ, None)},
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data in {diag_folder}")
    except Exception as e:
        raise Exception(f"Error loading beam data: {e}")

    arrs = {
        "x": x,
        "y": y,
        "z": z,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "w": w,
    }

    if min_z is not None:
        arrs = an.apply_cut(arrs, "z", min_z, op="gt")
    if max_z is not None:
        arrs = an.apply_cut(arrs, "z", max_z, op="lt")

    return (
        arrs["x"],
        arrs["y"],
        arrs["z"],
        arrs["ux"],
        arrs["uy"],
        arrs["uz"],
        arrs["w"],
    )


def calculate_beam_ranges(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    w: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float], np.ndarray]:
    """Calculate beam ranges and energy distribution.

    Args:
        x: Particle x positions in meters.
        y: Particle y positions in meters.
        ux: Normalized x-momenta (dimensionless).
        uy: Normalized y-momenta (dimensionless).
        uz: Normalized z-momenta (dimensionless).
        w: Particle weights.

    Returns:
        Tuple containing (x_range, y_range, energy_mev).
    """
    moments = an._calculate_moments(x, y, ux, uy, uz, w)
    x_std = np.sqrt(moments["x2"]) * 1e6
    x_range = (-2.5 * x_std, 2.5 * x_std)
    y_std = np.sqrt(moments["y2"]) * 1e6
    y_range = (-2.5 * y_std, 2.5 * y_std)

    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    energy_mev = (gamma - 1) * const.m_e * const.c**2 / const.e / 1e6

    return x_range, y_range, energy_mev


def create_dispersion_plot(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    energy_mev: np.ndarray,
    w: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> None:
    """Create and display the dispersion analysis plot.

    Args:
        x: Particle x positions in meters.
        y: Particle y positions in meters.
        ux: Normalized x-momenta (dimensionless).
        uy: Normalized y-momenta (dimensionless).
        uz: Normalized z-momenta (dimensionless).
        energy_mev: Particle energies in MeV.
        w: Particle weights.
        x_range: Range for x-axis plotting in micrometers.
        y_range: Range for y-axis plotting in micrometers.
    """
    # Normalize transverse momenta by gamma_z = sqrt(1 + uz**2)
    gamma_z = np.sqrt(1 + uz**2)
    ux = ux / gamma_z
    uy = uy / gamma_z

    energy_range = [np.min(energy_mev), np.max(energy_mev)]
    hist_kw = dict(bins=200, weights=w, cmap="plasma", vmax=5e5)

    # --- Plot 1: x and y vs energy (transverse position dispersion) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist2d(
        x * 1e6,
        energy_mev,
        range=[x_range, energy_range],
        **hist_kw,
    )
    axes[0].set_ylabel("Energy (MeV)")
    axes[0].set_xlabel("Horizontal Position (μm)")

    axes[1].hist2d(
        y * 1e6,
        energy_mev,
        range=[y_range, energy_range],
        **hist_kw,
    )
    axes[1].set_ylabel("Energy (MeV)")
    axes[1].set_xlabel("Vertical Position (μm)")

    plt.suptitle("Transverse Dispersion")
    plt.tight_layout()
    plt.show()

    # --- Plot 2: ux and uy vs energy (momentum dispersion) ---
    ux_std = np.sqrt(np.average((ux - np.average(ux, weights=w)) ** 2, weights=w))
    uy_std = np.sqrt(np.average((uy - np.average(uy, weights=w)) ** 2, weights=w))
    ux_range = (-2.5 * ux_std, 2.5 * ux_std)
    uy_range = (-2.5 * uy_std, 2.5 * uy_std)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

    axes2[0].hist2d(
        ux,
        energy_mev,
        range=[ux_range, energy_range],
        **hist_kw,
    )
    axes2[0].set_ylabel("Energy (MeV)")
    axes2[0].set_xlabel(r"$u_x/\gamma_z$")

    axes2[1].hist2d(
        uy,
        energy_mev,
        range=[uy_range, energy_range],
        **hist_kw,
    )
    axes2[1].set_ylabel("Energy (MeV)")
    axes2[1].set_xlabel(r"$u_y/\gamma_z$")

    plt.suptitle("Momentum vs Energy")
    plt.tight_layout()
    plt.show()

    # --- Plot 3: Energy-slice weighted means <x>, <ux>, <y>, <uy> vs energy ---
    n_bins = 50
    edges = np.linspace(energy_mev.min(), energy_mev.max(), n_bins + 1)
    bin_idx = np.digitize(energy_mev, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2

    mean_x = np.full(n_bins, np.nan)
    mean_ux = np.full(n_bins, np.nan)
    mean_y = np.full(n_bins, np.nan)
    mean_uy = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = bin_idx == i
        wi = w[mask]
        if wi.sum() > 0:
            mean_x[i] = np.average(x[mask], weights=wi)
            mean_ux[i] = np.average(ux[mask], weights=wi)
            mean_y[i] = np.average(y[mask], weights=wi)
            mean_uy[i] = np.average(uy[mask], weights=wi)

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax3a.plot(centers, mean_x * 1e6, "-", color="red", label=r"$\langle x \rangle$")
    ax3a.plot(centers, mean_y * 1e6, "-", color="blue", label=r"$\langle y \rangle$")
    ax3a.set_ylabel(r"Position (μm)")
    ax3a.legend(loc="best")
    ax3a.set_title("Position")
    ax3a.grid(visible=True)

    ax3b.plot(
        centers, mean_ux, "--", color="red", label=r"$\langle u_x/\gamma_z \rangle$"
    )
    ax3b.plot(
        centers, mean_uy, "--", color="blue", label=r"$\langle u_y/\gamma_z \rangle$"
    )
    ax3b.set_xlabel("Energy (MeV)")
    ax3b.set_ylabel(r"$u/\gamma_z$")
    ax3b.legend(loc="best")
    ax3b.set_title("Momentum")
    ax3b.grid(visible=True)

    plt.suptitle("Energy-slice weighted averages")
    plt.tight_layout()
    plt.show()


def process(args: argparse.Namespace) -> None:
    """
    Main function to perform electron beam dispersion analysis.

    Args:
        args: Parsed command-line arguments. See module docstring for argument details.
    """
    try:
        x, y, z, ux, uy, uz, w = load_and_filter_beam_data(
            args.diag_folder,
            args.species,
            args.iteration_number,
            args.min_z,
            args.max_z,
        )
    except (FileNotFoundError, Exception) as e:
        print(f"Error: {e}")
        return

    # Calculate beam ranges and energy distribution
    x_range, y_range, energy_mev = calculate_beam_ranges(x, y, ux, uy, uz, w)

    # Create and display the dispersion plot
    create_dispersion_plot(x, y, ux, uy, uz, energy_mev, w, x_range, y_range)


def main() -> None:
    """Main entry point for script execution: parses command-line arguments and calls process()"""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
