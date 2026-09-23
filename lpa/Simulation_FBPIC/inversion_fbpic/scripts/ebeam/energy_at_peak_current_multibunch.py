"""
Iterates through HDF5 diagnostics and calculates the energy at the z location of peak current for multiple bunches.

This script loads electron beam data from diagnostic files, calculates the current profile
to find multiple peaks (bunches), then determines the mean energy of particles at each peak location.
Bunches are detected as separate peaks in the current profile that are separated by at least 10e-6 m.

Usage:
    Command-line usage:
        python energy_at_peak_current_multibunch.py [OPTIONS]

    Optional Arguments:
        -d, --diag-folder PATH
            Path to the ebeam .h5 files directory. If not provided, uses the default
            value defined in the script.

        --start-iteration INT
            First iteration to process.
            Default: 1

        -i, --end-iteration INT
            Last iteration to process.
            Default: 80

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

        --min-bunch-separation FLOAT
            Minimum separation between bunches (in meters).
            Default: 10e-6

        --peak-height-threshold FLOAT
            Minimum peak height as fraction of max current.
            Default: 0.1

        --z-tolerance FLOAT
            Tolerance for finding particles near peak current location (in meters).
            Default: 10e-6

        --results-file PATH
            Path to save/load results JSON file.
            Default: "energy_at_peak_current_multibunch_results.json"

        --overwrite-results
            Flag to force re-analysis even if results file exists. Default value is set by
            DEFAULT_OVERWRITE_RESULTS in the script.

        --no-overwrite-results
            Flag to skip analysis if results file exists. This overrides the default value
            and sets it to False.

    Examples:
        # Use all defaults
        python energy_at_peak_current_multibunch.py

        # Specify a different diagnostics folder and iteration range
        python energy_at_peak_current_multibunch.py -d /path/to/hdf5 --start-iteration 10 -i 50

        # Set particle selection criteria
        python energy_at_peak_current_multibunch.py --min-uz 200 --min-z 5000e-6
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import inversion_fbpic.utils.analysis as an
from typing import Optional, List, Tuple
from scipy.signal import find_peaks

from inversion_fbpic.utils.argparse_utils import float_or_none, build_selection_dict

# Default configuration variables
DEFAULT_DIAG_FOLDER: Path = Path(
    "../../../../../../inversion-fbpic-runscripts/simulations/metrology/ionization_injection/infinite/case_107/lab_diags/hdf5"
)  # Path to ebeam .h5 files

DEFAULT_START_ITERATION: int = 1  # First iteration to process
DEFAULT_END_ITERATION: int = 80  # Last iteration to process

DEFAULT_SPECIES: str = "n_elec"  # Species to analyze

DEFAULT_MIN_UZ: Optional[float] = 100.0  # Minimum uz value for particle selection
DEFAULT_MAX_UZ: Optional[float] = None  # Maximum uz value for particle selection
DEFAULT_MIN_Z: Optional[float] = None  # Minimum z value for particle selection
DEFAULT_MAX_Z: Optional[float] = None  # Maximum z value for particle selection

DEFAULT_BINS: int = 200  # Number of bins for current profile calculation

# Bunch detection parameters
DEFAULT_MIN_BUNCH_SEPARATION: float = 10e-6  # Minimum separation between bunches (m)
DEFAULT_PEAK_HEIGHT_THRESHOLD: float = (
    0.1  # Minimum peak height as fraction of max current
)
DEFAULT_Z_TOLERANCE: float = (
    10e-6  # Tolerance for finding particles near peak current location (m)
)

# Results file settings
DEFAULT_RESULTS_FILE: Path = Path("energy_at_peak_current_multibunch_results.json")
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
        description="Calculate energy at peak current locations for multiple bunches in electron beam diagnostics."
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
        "-i",
        "--end-iteration",
        type=int,
        default=DEFAULT_END_ITERATION,
        help="Last iteration to process",
    )

    parser.add_argument(
        "-s",
        "--species",
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
        "--min-bunch-separation",
        type=float,
        default=DEFAULT_MIN_BUNCH_SEPARATION,
        help="Minimum separation between bunches (in meters)",
    )

    parser.add_argument(
        "--peak-height-threshold",
        type=float,
        default=DEFAULT_PEAK_HEIGHT_THRESHOLD,
        help="Minimum peak height as fraction of max current",
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

    return parser.parse_args()


def detect_bunches(
    current: np.ndarray,
    z_axis: np.ndarray,
    min_separation: float = 10e-6,
    height_threshold: float = 0.1,
) -> List[Tuple[float, int]]:
    """
    Detects multiple bunches in the current profile.

    Args:
        current: Current profile array
        z_axis: z-axis bin centers
        min_separation: Minimum separation between bunches (m)
        height_threshold: Minimum peak height as fraction of max current

    Returns:
        List of tuples (z_peak, peak_index) for each detected bunch
    """
    # Convert minimum separation to bin units
    bin_width = z_axis[1] - z_axis[0]
    min_separation_bins = int(min_separation / bin_width)

    # Set height threshold as fraction of maximum current
    min_height = height_threshold * np.max(current)

    # Find peaks using scipy
    peaks, properties = find_peaks(
        current, height=min_height, distance=min_separation_bins
    )

    # Convert peak indices to z positions and sort by z (descending - largest z first)
    bunch_peaks = [(z_axis[peak_idx], peak_idx) for peak_idx in peaks]
    bunch_peaks.sort(
        key=lambda x: x[0], reverse=True
    )  # Sort by z position (descending)

    return bunch_peaks


def find_energy_at_peak_current_multibunch(
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
    min_separation: float = 10e-6,
    height_threshold: float = 0.1,
) -> List[Tuple[float, float, float]]:
    """
    Finds the mean energy of particles at each detected bunch location.

    Args:
        x, y, z: Particle positions (m)
        ux, uy, uz: Normalized momenta
        w: Particle weights
        q: Particle charges (C)
        bins: Number of bins for current profile
        z_tol: Tolerance for selecting particles near peak current location (m)
        min_separation: Minimum separation between bunches (m)
        height_threshold: Minimum peak height as fraction of max current

    Returns:
        List of tuples (z_peak, mean_energy_mev, std_energy_mev) for each bunch
    """
    # Calculate current profile
    current, z_axis = an.calculate_current(z, ux, uy, uz, w, q, bins=bins)

    # Detect bunches
    bunch_peaks = detect_bunches(current, z_axis, min_separation, height_threshold)

    results = []

    for z_peak, peak_idx in bunch_peaks:
        # Select particles near this peak current location
        selection = np.abs(z - z_peak) < z_tol

        if not np.any(selection):
            # If no particles found, append NaN values
            results.append((z_peak, np.nan, np.nan))
            continue

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

        results.append((z_peak, mean_energy, std_energy))

    return results


def plot_energy_vs_z_peak_multibunch(results: List[dict]) -> None:
    """
    Creates a plot of energy vs z_peak for multiple bunches with different colors.

    Args:
        results: List of dictionaries containing iteration results with multiple bunches
    """
    # Define colors for different bunches
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#7209B7"]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Group results by bunch number (sorted by z position)
    bunch_data = {}

    for result in results:
        iteration = result["iteration"]
        for bunch_idx, bunch_info in enumerate(result["bunches"]):
            if bunch_idx not in bunch_data:
                bunch_data[bunch_idx] = {
                    "iterations": [],
                    "z_peaks": [],
                    "mean_energies": [],
                    "std_energies": [],
                }

            if not np.isnan(bunch_info["mean_energy_mev"]):
                bunch_data[bunch_idx]["iterations"].append(iteration)
                bunch_data[bunch_idx]["z_peaks"].append(bunch_info["z_peak_mm"])
                bunch_data[bunch_idx]["mean_energies"].append(
                    bunch_info["mean_energy_mev"]
                )
                bunch_data[bunch_idx]["std_energies"].append(
                    bunch_info["std_energy_mev"]
                )

    # Plot each bunch with different colors
    for bunch_idx in sorted(bunch_data.keys()):
        data = bunch_data[bunch_idx]
        if not data["iterations"]:
            continue

        color = colors[bunch_idx % len(colors)]
        z_peaks = np.array(data["z_peaks"])
        mean_energies = np.array(data["mean_energies"])
        std_energies = np.array(data["std_energies"])
        iterations = np.array(data["iterations"])

        # Sort by z position for plotting (descending - largest z first)
        sort_idx = np.argsort(z_peaks)[::-1]  # Reverse to get descending order
        z_peaks = z_peaks[sort_idx]
        mean_energies = mean_energies[sort_idx]
        std_energies = std_energies[sort_idx]
        iterations = iterations[sort_idx]

        # Plot mean energy line
        ax.plot(
            z_peaks,
            mean_energies,
            "o-",
            linewidth=2,
            markersize=6,
            color=color,
            label=f"Bunch {bunch_idx + 1}",
            zorder=3,
        )

        # Add shaded error region (±1 std)
        ax.fill_between(
            z_peaks,
            mean_energies - std_energies,
            mean_energies + std_energies,
            alpha=0.3,
            color=color,
            zorder=2,
        )

        # Add iteration numbers as annotations (every 3rd point to avoid clutter)
        for i in range(0, len(iterations), 3):
            ax.annotate(
                f"{iterations[i]}",
                (z_peaks[i], mean_energies[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                alpha=0.6,
            )

    # Styling
    ax.set_xlabel("Z Position of Peak Current (mm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Energy (MeV)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Electron Energy at Peak Current Locations (Multi-Bunch)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(fontsize=12, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Save figure
    output_file = Path("energy_at_peak_current_multibunch_plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    plt.show()


def process(args: argparse.Namespace) -> None:
    """
    Main function to iterate through diagnostics and calculate energy at peak current for multiple bunches.

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
        total_bunches = 0
        for result in results:
            total_bunches += len(
                [b for b in result["bunches"] if not np.isnan(b["mean_energy_mev"])]
            )

        print("\nSummary:")
        print(f"  Total bunches detected: {total_bunches}")
        print(f"  Average bunches per iteration: {total_bunches/len(results):.1f}")

        # Create visualization
        print("\nGenerating plot...")
        plot_energy_vs_z_peak_multibunch(results)
        return

    # Perform analysis
    if args.results_file.exists() and args.overwrite_results:
        print(f"Overwriting existing results file: {args.results_file}")

    print(f"Processing diagnostics from: {args.diag_folder}")
    print(f"Iterations: {args.start_iteration} to {args.end_iteration}")
    print(f"Species: {args.species}")
    print(f"Z tolerance for peak current: {args.z_tolerance*1e6:.1f} μm")
    print(f"Minimum bunch separation: {args.min_bunch_separation*1e6:.1f} μm")
    print(
        f"Peak height threshold: {args.peak_height_threshold*100:.1f}% of max current"
    )
    print("-" * 60)

    results = []

    for iteration in range(args.start_iteration, args.end_iteration + 1):
        try:
            # Load beam data for this iteration
            x, y, z, ux, uy, uz, w, q, ts = an.load_beam_data(
                args.diag_folder, args.species, iteration=iteration, select=selection
            )

            # Find energy at peak current locations for all bunches
            bunch_results = find_energy_at_peak_current_multibunch(
                x,
                y,
                z,
                ux,
                uy,
                uz,
                w,
                q,
                bins=args.bins,
                z_tol=args.z_tolerance,
                min_separation=args.min_bunch_separation,
                height_threshold=args.peak_height_threshold,
            )

            # Store results for this iteration
            iteration_data = {"iteration": iteration, "bunches": []}

            for i, (z_peak, mean_energy, std_energy) in enumerate(bunch_results):
                iteration_data["bunches"].append(
                    {
                        "bunch_index": i,
                        "z_peak_mm": z_peak * 1e3,
                        "mean_energy_mev": mean_energy,
                        "std_energy_mev": std_energy,
                    }
                )

            results.append(iteration_data)

            # Print progress
            if bunch_results:
                print(f"Iteration {iteration:3d}: Found {len(bunch_results)} bunch(es)")
                for i, (z_peak, mean_energy, std_energy) in enumerate(bunch_results):
                    if not np.isnan(mean_energy):
                        print(
                            f"  Bunch {i+1}: z_peak = {z_peak*1e3:8.3f} mm, "
                            f"Energy = {mean_energy:6.1f} ± {std_energy:5.1f} MeV"
                        )
                    else:
                        print(
                            f"  Bunch {i+1}: z_peak = {z_peak*1e3:8.3f} mm, No particles found"
                        )
            else:
                print(f"Iteration {iteration:3d}: No bunches detected")

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
        total_bunches = 0
        valid_bunches = 0
        for result in results:
            for bunch in result["bunches"]:
                total_bunches += 1
                if not np.isnan(bunch["mean_energy_mev"]):
                    valid_bunches += 1

        print("\nSummary:")
        print(f"  Total bunches detected: {total_bunches}")
        print(f"  Valid bunches (with particles): {valid_bunches}")
        print(f"  Average bunches per iteration: {total_bunches/len(results):.1f}")

        # Create visualization
        print("\nGenerating plot...")
        plot_energy_vs_z_peak_multibunch(results)


def main() -> None:
    """Main entry point for script execution: parses command-line arguments and calls process()"""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
