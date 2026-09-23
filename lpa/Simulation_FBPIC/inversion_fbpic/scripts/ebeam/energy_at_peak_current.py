"""
Iterates through HDF5 diagnostics and calculates the energy at the z location of peak current.

This script loads electron beam data from diagnostic files, calculates the current profile
to find the location of peak current, then determines the mean energy of particles at that location.

Usage:
    Command-line usage:
        python energy_at_peak_current.py [OPTIONS]

    Optional Arguments:
        -d, --diag-folder PATH
            Path to the ebeam .h5 files directory. If not provided, uses the default
            value defined in the script.

        --start-iteration INT
            First iteration to process.
            Default: 1

        -i, --end-iteration INT
            Last iteration to process.
            Default: 100

        -s, --species STR
            Which species to load from DIAG_FOLDER.
            Default: "n_elec"

        --min-uz FLOAT|None
            Minimum uz value for particle selection. Use 'None' or 'null' to explicitly
            set to None.
            Default: 100.0

        --max-uz FLOAT|None
            Maximum uz value for particle selection. Use 'None' or 'null' to explicitly
            set to None.
            Default: None

        --min-z FLOAT|None
            Minimum z value for particle selection. Use 'None' or 'null' to explicitly
            set to None.
            Default: None

        --max-z FLOAT|None
            Maximum z value for particle selection. Use 'None' or 'null' to explicitly
            set to None.
            Default: None

        -b, --bins INT
            Number of bins for current profile calculation.
            Default: 200

        --z-tolerance FLOAT
            Tolerance for finding particles near peak current location (in meters).
            Default: 10e-6

        --results-file PATH
            Path to save/load results JSON file.
            Default: "energy_at_peak_current_results.json"

        --overwrite-results
            Flag to force re-analysis even if results file exists. Default value is set by
            DEFAULT_OVERWRITE_RESULTS in the script.

        --no-overwrite-results
            Flag to skip analysis if results file exists. This overrides the default value
            and sets it to False.

    Examples:
        # Use all defaults
        python energy_at_peak_current.py

        # Specify a different diagnostics folder and iteration range
        python energy_at_peak_current.py -d /path/to/hdf5 --start-iteration 10 -i 50

        # Set particle selection criteria
        python energy_at_peak_current.py --min-uz 200 --min-z 5000e-6
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import inversion_fbpic.utils.analysis as an
from typing import Optional

from inversion_fbpic.utils.argparse_utils import float_or_none, build_selection_dict

# Default configuration variables
DEFAULT_DIAG_FOLDER: Path = Path(
    "../../../../../../inversion-fbpic-runscripts/simulations/metrology/ionization_injection/infinite/case_107/lab_diags/hdf5"
)  # Path to ebeam .h5 files

DEFAULT_START_ITERATION: int = 1  # First iteration to process
DEFAULT_END_ITERATION: int = 100  # Last iteration to process

DEFAULT_SPECIES: str = "n_elec"  # Species to analyze

DEFAULT_MIN_UZ: Optional[float] = 100.0  # Minimum uz value for particle selection
DEFAULT_MAX_UZ: Optional[float] = None  # Maximum uz value for particle selection
DEFAULT_MIN_Z: Optional[float] = None  # Minimum z value for particle selection
DEFAULT_MAX_Z: Optional[float] = None  # Maximum z value for particle selection

DEFAULT_BINS: int = 200  # Number of bins for current profile calculation

# Tolerance for finding particles near peak current location (in meters)
DEFAULT_Z_TOLERANCE: float = 10e-6  # 10 microns

# Results file settings
DEFAULT_RESULTS_FILE: Path = Path("energy_at_peak_current_results.json")
DEFAULT_OVERWRITE_RESULTS: bool = (
    True  # Set to True to force re-analysis even if results exist
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Calculate energy at the z location of peak current for electron beam diagnostics."
    )

    parser.add_argument(
        "-d",
        "--diag-folder",
        type=Path,
        default=DEFAULT_DIAG_FOLDER,
        help="Path to the ebeam .h5 files directory",
    )

    parser.add_argument(
        "--start-iteration",
        type=int,
        default=DEFAULT_START_ITERATION,
        help="First iteration to process",
    )

    parser.add_argument(
        "--end-iteration",
        "-i",
        type=int,
        default=DEFAULT_END_ITERATION,
        help="Last iteration to process",
    )

    parser.add_argument(
        "--species",
        "-s",
        type=str,
        default=DEFAULT_SPECIES,
        help="Which species to load from DIAG_FOLDER",
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
        "-b",
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of bins for current profile calculation",
    )

    parser.add_argument(
        "--z-tolerance",
        type=float,
        default=DEFAULT_Z_TOLERANCE,
        help="Tolerance for finding particles near peak current location (in meters)",
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="Path to save/load results JSON file",
    )

    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        default=DEFAULT_OVERWRITE_RESULTS,
        help="Flag to force re-analysis even if results file exists",
    )

    parser.add_argument(
        "--no-overwrite-results",
        action="store_false",
        dest="overwrite_results",
        help="Flag to skip analysis if results file exists (overrides default)",
    )

    args = parser.parse_args()

    # Validate z_tolerance is positive (required for particle selection logic)
    if args.z_tolerance <= 0:
        parser.error("--z-tolerance must be positive (meters)")

    # Validate bins is positive (required for histogram binning)
    if args.bins <= 0:
        parser.error("--bins must be a positive integer")

    return args


def find_energy_at_peak_current(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    w: np.ndarray,
    q: np.ndarray,
    bins: int = 200,
    z_tol: float = 10e-6,
) -> tuple[float, float, float]:
    """
    Finds the mean energy of particles at the z location of peak current.

    Args:
        x, y, z: Particle positions (m)
        ux, uy, uz: Normalized momenta
        w: Particle weights
        q: Particle charges (C)
        bins: Number of bins for current profile
        z_tol: Tolerance for selecting particles near peak current location (m)

    Returns:
        Tuple of (z_peak_current, mean_energy_mev, std_energy_mev):
            - z_peak_current: z location of peak current (m)
            - mean_energy_mev: weighted mean energy at that location (MeV)
            - std_energy_mev: weighted std of energy at that location (MeV)
    """
    # Calculate current profile
    current, z_axis = an.calculate_current(z, ux, uy, uz, w, q, bins=bins)

    # Find z location of peak current
    peak_idx = np.argmax(current)
    z_peak = z_axis[peak_idx]

    # Select particles near the peak current location
    selection = np.abs(z - z_peak) < z_tol

    if not np.any(selection):
        # If no particles found, return NaN
        return z_peak, np.nan, np.nan

    # Get energy of selected particles
    ux_sel = ux[selection]
    uy_sel = uy[selection]
    uz_sel = uz[selection]
    w_sel = w[selection]

    # Convert to MeV
    energy_mev = an.convert_to_mev(ux_sel, uy_sel, uz_sel)

    # Calculate weighted mean and std
    mean_energy = np.average(energy_mev, weights=w_sel)
    std_energy = np.sqrt(np.average((energy_mev - mean_energy) ** 2, weights=w_sel))

    return z_peak, mean_energy, std_energy


def plot_energy_vs_z_peak(results: list[dict]) -> None:
    """
    Creates a plot of energy vs z_peak with shaded error region.

    Args:
        results: List of dictionaries containing iteration results
    """
    # Filter out NaN values
    valid_results = [r for r in results if not np.isnan(r["mean_energy_mev"])]

    if not valid_results:
        print("No valid results to plot")
        return

    # Extract data
    z_peak_mm = np.array([r["z_peak_mm"] for r in valid_results])
    mean_energy = np.array([r["mean_energy_mev"] for r in valid_results])
    std_energy = np.array([r["std_energy_mev"] for r in valid_results])
    iterations = np.array([r["iteration"] for r in valid_results])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot mean energy line
    ax.plot(
        z_peak_mm,
        mean_energy,
        "o-",
        linewidth=2,
        markersize=6,
        color="#2E86AB",
        label="Mean Energy",
        zorder=3,
    )

    # Add shaded error region (±1 std)
    ax.fill_between(
        z_peak_mm,
        mean_energy - std_energy,
        mean_energy + std_energy,
        alpha=0.3,
        color="#2E86AB",
        label="±1σ",
        zorder=2,
    )

    # Styling
    ax.set_xlabel("Z Position of Peak Current (mm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Energy (MeV)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Electron Energy at Peak Current Location",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(fontsize=12, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add iteration numbers as annotations (every 5th point to avoid clutter)
    for i in range(0, len(iterations), 5):
        ax.annotate(
            f"{iterations[i]}",
            (z_peak_mm[i], mean_energy[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            alpha=0.6,
        )

    plt.tight_layout()

    # Save figure
    output_file = Path("energy_at_peak_current_plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    plt.show()


def process(args: argparse.Namespace) -> None:
    """
    Main function to iterate through diagnostics and calculate energy at peak current.

    Args:
        args: Parsed command-line arguments. See module docstring for argument details.
    """
    import json

    # Build selection dictionary from individual parameters
    selection = build_selection_dict(
        min_uz=args.min_uz,
        max_uz=args.max_uz,
        min_z=args.min_z,
        max_z=args.max_z,
    )

    # Check if results already exist
    if args.results_file.exists() and not args.overwrite_results:
        print(f"Found existing results file: {args.results_file}")
        print("Loading results and skipping analysis...")
        print("(Use --overwrite-results to force re-analysis)")
        print("-" * 60)

        # Load existing results
        with open(args.results_file, "r") as f:
            results = json.load(f)

        print(f"Loaded {len(results)} iterations from file")

        # Print summary statistics
        energies = [
            r["mean_energy_mev"] for r in results if not np.isnan(r["mean_energy_mev"])
        ]
        if energies:
            print("\nSummary:")
            print(f"  Mean energy across all iterations: {np.mean(energies):.1f} MeV")
            print(
                f"  Energy range: {np.min(energies):.1f} - {np.max(energies):.1f} MeV"
            )

        # Create visualization
        print("\nGenerating plot...")
        plot_energy_vs_z_peak(results)
        return

    # Perform analysis
    if args.results_file.exists() and args.overwrite_results:
        print(f"Overwriting existing results file: {args.results_file}")

    print(f"Processing diagnostics from: {args.diag_folder}")
    print(f"Iterations: {args.start_iteration} to {args.end_iteration}")
    print(f"Species: {args.species}")
    print(f"Z tolerance for peak current: {args.z_tolerance*1e6:.1f} μm")
    print("-" * 60)

    results = []

    for iteration in range(args.start_iteration, args.end_iteration + 1):
        try:
            # Load beam data for this iteration
            x, y, z, ux, uy, uz, w, q, ts = an.load_beam_data(
                args.diag_folder, args.species, iteration=iteration, select=selection
            )

            # Find energy at peak current location
            z_peak, mean_energy, std_energy = find_energy_at_peak_current(
                x, y, z, ux, uy, uz, w, q, bins=args.bins, z_tol=args.z_tolerance
            )

            # Store results
            results.append(
                {
                    "iteration": iteration,
                    "z_peak_mm": z_peak * 1e3,
                    "mean_energy_mev": mean_energy,
                    "std_energy_mev": std_energy,
                }
            )

            # Print progress
            print(
                f"Iteration {iteration:3d}: z_peak = {z_peak*1e3:8.3f} mm, "
                f"Energy = {mean_energy:6.1f} ± {std_energy:5.1f} MeV"
            )

        except FileNotFoundError:
            print(f"Iteration {iteration:3d}: File not found, skipping...")
            continue
        except Exception as e:
            print(f"Iteration {iteration:3d}: Error - {e}")
            continue

    print("-" * 60)
    print(f"Processed {len(results)} iterations successfully")

    # Save results to file
    if results:
        with open(args.results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.results_file}")

        # Print summary statistics
        energies = [
            r["mean_energy_mev"] for r in results if not np.isnan(r["mean_energy_mev"])
        ]
        if energies:
            print("\nSummary:")
            print(f"  Mean energy across all iterations: {np.mean(energies):.1f} MeV")
            print(
                f"  Energy range: {np.min(energies):.1f} - {np.max(energies):.1f} MeV"
            )

        # Create visualization
        print("\nGenerating plot...")
        plot_energy_vs_z_peak(results)


def main() -> None:
    """Main entry point for script execution: parses command-line arguments and calls process()"""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
