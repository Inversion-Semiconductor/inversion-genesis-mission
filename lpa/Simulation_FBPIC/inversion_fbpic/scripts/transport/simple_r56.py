"""
Loads a given FBPIC output dump specified by the DIAG_FOLDER path, the ITERATION_NUMBER within this
folder, and the SELECTION cropping region; and applies a basic R56 transformation on the beam.  The
goal is to calculate the maximum peak current and/or minimum bunch length achievable by a compression
chicane.

Recommended to use the script "ebeam/plot_ebeam_analysis.py" to find the correct values for particle selection.

Usage:
    Command-line usage:
        python simple_r56.py [OPTIONS]

    Optional Arguments:
        -d, --diag-folder PATH
            Path to the ebeam .h5 files directory. If not provided, uses the default
            value defined in the script.

        --min-uz FLOAT|None
            Minimum uz value for particle selection. Use 'None' or 'null' to explicitly
            set to None.
            Default: 200.0

        --max-uz FLOAT|None
            Maximum uz value for particle selection. Use 'None' or 'null' to explicitly
            set to None.
            Default: None

        --min-z FLOAT|None
            Minimum z value for particle selection. Use 'None' or 'null' to explicitly
            set to None.
            Default: 45018e-6

        --max-z FLOAT|None
            Maximum z value for particle selection. Use 'None' or 'null' to explicitly
            set to None.
            Default: None

        -i, --iteration-number INT
            Which dump to load from DIAG_FOLDER. Set to "-1" for final dump.
            Default: -1

        -s, --species STR
            Which species to load from DIAG_FOLDER. Typically "electrons" or "n_elec".
            Default: "n_elec"

        --r56-min FLOAT
            Minimum R56 value for scan (in meters).
            Default: 0.0

        --r56-max FLOAT
            Maximum R56 value for scan (in meters).
            Default: 50e-6

        --r56-steps INT
            Number of steps in R56 scan.
            Default: 100

        -b, --bins INT
            Number of bins for current profile calculation.
            Default: 1000

    Examples:
        # Use all defaults
        python simple_r56.py

        # Specify a different diagnostics folder and particle selection
        python simple_r56.py -d /path/to/hdf5 --min-uz 300 --min-z 50000e-6

        # Scan a different R56 range with more steps
        python simple_r56.py --r56-min 0 --r56-max 100e-6 --r56-steps 200
"""

import argparse
from pathlib import Path
import inversion_fbpic.utils.analysis as an
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from inversion_fbpic.utils.argparse_utils import float_or_none, build_selection_dict

# Default configuration constants
DEFAULT_DIAG_FOLDER: Path = Path(
    "../../../../../../inversion-fbpic-runscripts/optimas/metrology/ionization_injection/internal/version_2/downramp_1/sim0077/lab_diags/hdf5"
)  # Path to ebeam .h5 files

DEFAULT_MIN_UZ: Optional[float] = 200.0  # Minimum uz value for particle selection
DEFAULT_MAX_UZ: Optional[float] = None  # Maximum uz value for particle selection
DEFAULT_MIN_Z: Optional[float] = 45018e-6  # Minimum z value for particle selection
DEFAULT_MAX_Z: Optional[float] = None  # Maximum z value for particle selection

DEFAULT_ITERATION_NUMBER: int = (
    -1
)  # Which dump to load from DIAG_FOLDER. Set to "-1" for final dump

DEFAULT_SPECIES: str = (
    "n_elec"  # Which species to load from DIAG_FOLDER. Typically "electrons"
)

DEFAULT_R56_MIN: float = 0.0  # Minimum R56 value for scan (m)
DEFAULT_R56_MAX: float = 50e-6  # Maximum R56 value for scan (m)
DEFAULT_R56_STEPS: int = 100  # Number of steps in R56 scan

DEFAULT_BINS: int = 1000  # Number of bins for current profile calculation


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Apply R56 transformation to electron beam and analyze compression chicane performance."
    )

    parser.add_argument(
        "-d",
        "--diag-folder",
        type=Path,
        default=DEFAULT_DIAG_FOLDER,
        help="Path to the ebeam .h5 files directory",
    )

    parser.add_argument(
        "--min-uz",
        type=float_or_none,
        default=DEFAULT_MIN_UZ,
        help="Minimum uz value for particle selection. Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "--max-uz",
        type=float_or_none,
        default=DEFAULT_MAX_UZ,
        help="Maximum uz value for particle selection. Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "--min-z",
        type=float_or_none,
        default=DEFAULT_MIN_Z,
        help="Minimum z value for particle selection. Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "--max-z",
        type=float_or_none,
        default=DEFAULT_MAX_Z,
        help="Maximum z value for particle selection. Use 'None' or 'null' to explicitly set to None",
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
        help="Which species to load from DIAG_FOLDER. Typically 'electrons' or 'n_elec'",
    )

    parser.add_argument(
        "--r56-min",
        type=float,
        default=DEFAULT_R56_MIN,
        help="Minimum R56 value for scan (in meters)",
    )

    parser.add_argument(
        "--r56-max",
        type=float,
        default=DEFAULT_R56_MAX,
        help="Maximum R56 value for scan (in meters)",
    )

    parser.add_argument(
        "--r56-steps",
        type=int,
        default=DEFAULT_R56_STEPS,
        help="Number of steps in R56 scan",
    )

    parser.add_argument(
        "-b",
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of bins for current profile calculation",
    )

    return parser.parse_args()


def process(args: argparse.Namespace) -> None:
    """
    Apply R56 transformation and analyze beam properties.

    Args:
        args: Parsed command-line arguments. See module docstring for argument details.
    """
    # Build selection dictionary from individual parameters
    selection = build_selection_dict(
        min_uz=args.min_uz,
        max_uz=args.max_uz,
        min_z=args.min_z,
        max_z=args.max_z,
    )

    try:
        x, y, z, ux, uy, uz, w, q, ts = an.load_beam_data(
            args.diag_folder,
            args.species,
            iteration=args.iteration_number,
            select=selection,
        )
    except FileNotFoundError:
        print(f"Error: Could not find data in {args.diag_folder}")
        return
    except Exception as e:
        print(f"Error loading beam data: {e}")
        return

    # Specify a range of R56 values to scan
    r56_values = np.linspace(args.r56_min, args.r56_max, args.r56_steps)

    # Calculate the average uz value
    uz_mean = np.average(uz, weights=w)
    print(f"Average uz value: {uz_mean:.3f}")

    # Arrays to store results
    peak_currents = []
    bunch_sizes = []

    print("Scanning R56 values...")
    for i, r56 in enumerate(r56_values):
        print(f"  Processing R56 = {r56*1e6:.1f} μm ({i+1}/{len(r56_values)})")

        # For each R56 setting:
        # Take the z and uz arrays, modify them as such
        #  z_final = z_initial + r56 * ((uz_initial-uz_mean)/uz_mean)
        z_modified = z + r56 * ((uz - uz_mean) / uz_mean)

        # With this new z_final distribution and uz, use the analysis functions
        #  to calculate longitudinal bunch size and peak current
        try:
            # Calculate current profile with modified z positions
            current, z_axis = an.calculate_current(
                z_modified, ux, uy, uz, w, q, bins=args.bins
            )
            peak_current = np.max(current)
            peak_currents.append(peak_current)

            # Calculate longitudinal bunch size (RMS)
            z_mean_modified = np.average(z_modified, weights=w)
            sigma_z = np.sqrt(
                np.average((z_modified - z_mean_modified) ** 2, weights=w)
            )
            bunch_sizes.append(sigma_z)

        except Exception as e:
            print(f"    Warning: Error processing R56 = {r56*1e6:.1f} μm: {e}")
            peak_currents.append(np.nan)
            bunch_sizes.append(np.nan)

    # Convert to numpy arrays for easier handling
    peak_currents = np.array(peak_currents)
    bunch_sizes = np.array(bunch_sizes)

    # Plot peak current and longitudinal bunch size vs R56, in the same plot but using different y axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot peak current on left y-axis
    color1 = "tab:blue"
    ax1.set_xlabel("R56 (μm)")
    ax1.set_ylabel("Peak Current (kA)", color=color1)
    line1 = ax1.plot(
        r56_values * 1e6,
        peak_currents * 1e-3,
        "o-",
        color=color1,
        linewidth=2,
        markersize=6,
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for bunch size
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Longitudinal Bunch Size (μm)", color=color2)
    line2 = ax2.plot(
        r56_values * 1e6,
        bunch_sizes * 1e6,
        "s-",
        color=color2,
        linewidth=2,
        markersize=6,
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add title and legend
    plt.title("Beam Properties vs R56 Dispersion")

    # Create combined legend
    lines = line1 + line2
    labels = ["Peak Current (kA)", "Bunch Size (μm)"]
    ax1.legend(lines, labels, loc="right")

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n=== R56 SCAN SUMMARY ===")
    print(f"R56 range: {r56_values[0]*1e6:.1f} to {r56_values[-1]*1e6:.1f} μm")
    print(
        f"Peak current range: {np.nanmin(peak_currents)*1e-3:.2f} to {np.nanmax(peak_currents)*1e-3:.2f} kA"
    )
    print(
        f"Bunch size range: {np.nanmin(bunch_sizes)*1e6:.2f} to {np.nanmax(bunch_sizes)*1e6:.2f} μm"
    )

    # Find optimal R56 for minimum bunch size
    valid_indices = ~np.isnan(bunch_sizes)
    if np.any(valid_indices):
        min_bunch_idx = np.nanargmin(bunch_sizes)
        optimal_r56 = r56_values[min_bunch_idx]
        print(
            f"Minimum bunch size: {bunch_sizes[min_bunch_idx]*1e6:.2f} μm at R56 = {optimal_r56*1e6:.1f} μm"
        )


def main() -> None:
    """
    Main entry point for script execution: parses command-line arguments and calls process()
    """
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
