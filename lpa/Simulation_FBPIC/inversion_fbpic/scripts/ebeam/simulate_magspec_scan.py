"""Simulate magnetic spectrometer data from 1D parameter scan simulations.

This module processes simulation data from a 1D parameter scan and creates
energy spectra waterfall plots similar to what HTU would record with their
magnetic spectrometer data.

Usage:
    Command-line usage:
        python simulate_magspec_scan.py [OPTIONS]

    Optional Arguments:
        --directory-folder PATH
            Path to the directory containing all scans. If not provided, uses the default
            value defined in the script.

        --analysis-folder STR
            Directory name containing the specific scan set of interest. If not provided,
            uses the default value defined in the script.

        --set-name STR
            Particular 1D scan to analyze within ANALYSIS_FOLDER. If not provided, uses
            the default value defined in the script.

        --factor FLOAT
            Factor to multiply the scanned parameter in plots.
            Default: -1e3

        --axis-offset FLOAT|None
            Factor to offset case values to match HTU. Use 'None' or 'null' to explicitly
            set to None.
            Default: -15.0

        --units STR
            Label to use for the scanned parameter in plots.
            Default: "JetBlade (mm)"

        --energy-bins INT
            Number of bins for energy histogram.
            Default: 130

        --energy-range-min FLOAT
            Minimum energy for histogram range in MeV.
            Default: 40.0

        --energy-range-max FLOAT
            Maximum energy for histogram range in MeV.
            Default: 180.0

        --beam-analysis-bins INT
            Number of bins for beam analysis.
            Default: 200

        --min-z FLOAT|None
            Minimum z position of ebeam particles to consider. Use 'None' or 'null' to
            explicitly set to None.
            Default: 4860e-6

        --flip-y-axis
            Flag to flip the y-axis in plots. Default value is set by
            DEFAULT_FLIP_Y_AXIS in the script.

        --no-flip-y-axis
            Flag to disable flipping the y-axis. This overrides the default value and
            sets it to False.

        --show-title
            Flag to show plot titles. Default value is set by DEFAULT_SHOW_TITLE in the script.

        --no-show-title
            Flag to disable showing plot titles. This overrides the default value and
            sets it to False.

        --top-right-annotation STR
            Text to display in top-right corner of waterfall plot. Set to empty string
            or omit to disable.
            Default: None

    Examples:
        # Use all defaults
        python simulate_magspec_scan.py

        # Specify a different directory and analysis folder
        python simulate_magspec_scan.py --directory-folder /path/to/scans --analysis-folder analysis_1

        # Customize plotting parameters
        python simulate_magspec_scan.py --factor -2e3 --axis-offset -20 --units "Custom (mm)"
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import inversion_fbpic.utils.analysis as an
from inversion_fbpic.utils.argparse_utils import float_or_none

# Default configuration variables
DEFAULT_DIRECTORY_FOLDER: Path = Path(
    "../../../../../../inversion-fbpic-runscripts/simulations/htu/version_1/htu_updated_scans"
)
DEFAULT_ANALYSIS_FOLDER: str = "analysis2"  # Set of scans of interest
DEFAULT_SET_NAME: str = "htu_ramp_pos_scan"  # Particular 1D scan to analyze

# Default plotting constants
DEFAULT_FACTOR: float = -1e3  # Factor to multiply the scanned parameter in plots
DEFAULT_AXIS_OFFSET: Optional[float] = (
    -15.0
)  # Factor to offset to match HTU. -15.0 for JetBlade
DEFAULT_UNITS: str = "JetBlade (mm)"  # Label to use for the scanned parameter in plots

# Default analysis constants
DEFAULT_ENERGY_BINS: int = 130
DEFAULT_ENERGY_RANGE_MIN: float = 40.0  # MeV
DEFAULT_ENERGY_RANGE_MAX: float = 180.0  # MeV
DEFAULT_BEAM_ANALYSIS_BINS: int = 200

DEFAULT_FLIP_Y_AXIS: bool = True
DEFAULT_SHOW_TITLE: bool = True
DEFAULT_TOP_RIGHT_ANNOTATION: Optional[str] = (
    None  # Set to a string to display in top-right corner, or None to disable
)

# Default minimum z position (in meters)
DEFAULT_MIN_Z: Optional[float] = 4860e-6


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Simulate magnetic spectrometer data from 1D parameter scan simulations."
    )

    parser.add_argument(
        "--directory-folder",
        type=Path,
        default=DEFAULT_DIRECTORY_FOLDER,
        help="Path to the directory containing all scans",
    )

    parser.add_argument(
        "--analysis-folder",
        type=str,
        default=DEFAULT_ANALYSIS_FOLDER,
        help="Directory name containing the specific scan set of interest",
    )

    parser.add_argument(
        "--set-name",
        type=str,
        default=DEFAULT_SET_NAME,
        help="Particular 1D scan to analyze within ANALYSIS_FOLDER",
    )

    parser.add_argument(
        "--factor",
        type=float,
        default=DEFAULT_FACTOR,
        help="Factor to multiply the scanned parameter in plots",
    )

    parser.add_argument(
        "--axis-offset",
        type=float_or_none,
        default=DEFAULT_AXIS_OFFSET,
        help="Factor to offset case values to match HTU. Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "--units",
        type=str,
        default=DEFAULT_UNITS,
        help="Label to use for the scanned parameter in plots",
    )

    parser.add_argument(
        "--energy-bins",
        type=int,
        default=DEFAULT_ENERGY_BINS,
        help="Number of bins for energy histogram",
    )

    parser.add_argument(
        "--energy-range-min",
        type=float,
        default=DEFAULT_ENERGY_RANGE_MIN,
        help="Minimum energy for histogram range in MeV",
    )

    parser.add_argument(
        "--energy-range-max",
        type=float,
        default=DEFAULT_ENERGY_RANGE_MAX,
        help="Maximum energy for histogram range in MeV",
    )

    parser.add_argument(
        "--beam-analysis-bins",
        type=int,
        default=DEFAULT_BEAM_ANALYSIS_BINS,
        help="Number of bins for beam analysis",
    )

    parser.add_argument(
        "--min-z",
        type=float_or_none,
        default=DEFAULT_MIN_Z,
        help="Minimum z position of ebeam particles to consider. Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "--flip-y-axis",
        action="store_true",
        default=DEFAULT_FLIP_Y_AXIS,
        help="Flag to flip the y-axis in plots",
    )

    parser.add_argument(
        "--no-flip-y-axis",
        action="store_false",
        dest="flip_y_axis",
        help="Flag to disable flipping the y-axis (overrides default)",
    )

    parser.add_argument(
        "--show-title",
        action="store_true",
        default=DEFAULT_SHOW_TITLE,
        help="Flag to show plot titles",
    )

    parser.add_argument(
        "--no-show-title",
        action="store_false",
        dest="show_title",
        help="Flag to disable showing plot titles (overrides default)",
    )

    parser.add_argument(
        "--top-right-annotation",
        type=str,
        default=DEFAULT_TOP_RIGHT_ANNOTATION,
        help="Text to display in top-right corner of waterfall plot. Set to empty string to disable",
    )

    return parser.parse_args()


def load_and_process_beam_data(
    data_folder: Path, min_z: Optional[float]
) -> Tuple[np.ndarray, ...]:
    """Load and process beam data from a simulation folder.

    Args:
        data_folder: Path to the beam data folder.
        min_z: Minimum z position of ebeam particles to consider.

    Returns:
        Tuple of processed beam arrays (x, y, z, ux, uy, uz, w, q, ts).
    """
    x, y, z, ux, uy, uz, w, q, ts = an.load_beam_data(
        data_folder, "electrons", iteration=-1
    )
    arrs = {"x": x, "y": y, "z": z, "ux": ux, "uy": uy, "uz": uz, "w": w}

    # Apply cropping
    arrs = an.apply_cut(arrs, "z", min_z, op="gt")

    return (
        arrs["x"],
        arrs["y"],
        arrs["z"],
        arrs["ux"],
        arrs["uy"],
        arrs["uz"],
        arrs["w"],
        q,
        ts,
    )


def extract_case_value(case_directory: Path) -> float:
    """Extract the case value from the directory name.

    Args:
        case_directory: Path to the case directory.

    Returns:
        The case value as a float.
    """
    return float(case_directory.name.split("_")[0])


def calculate_energy_histogram(
    energy_mev: np.ndarray,
    weights: np.ndarray,
    energy_bins: int,
    energy_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate energy histogram from beam data.

    Args:
        energy_mev: Energy values in MeV.
        weights: Particle weights.
        energy_bins: Number of bins for histogram.
        energy_range: Tuple of (min, max) energy range in MeV.

    Returns:
        Tuple of (histogram, bin_edges).
    """
    return np.histogram(
        energy_mev, weights=weights, bins=energy_bins, range=energy_range
    )


def calculate_peak_charge_density(
    hist: np.ndarray, bins: np.ndarray, total_charge_c: float
) -> float:
    """Calculate peak charge density in pC/MeV.

    Args:
        hist: Energy histogram.
        bins: Histogram bin edges.
        total_charge_c: Total charge in Coulombs.

    Returns:
        Peak charge density in pC/MeV.
    """
    ratio = np.sum(hist) / (-total_charge_c * 1e12)  # C to pC conversion
    cell_size = bins[1] - bins[0]
    return np.max(hist) / cell_size / ratio


def process_simulation_data(
    args: argparse.Namespace,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Process all simulation data and extract relevant parameters.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of processed arrays: (case_values, energy_histograms, charge_values,
        peak_charge_den_values, central_energy_values, energy_at_peak_values,
        energy_fwhm_values).
    """
    case_values: List[float] = []
    energy_histograms: List[np.ndarray] = []
    charge_values: List[float] = []
    peak_charge_den_values: List[float] = []
    central_energy_values: List[float] = []
    energy_at_peak_values: List[float] = []
    energy_fwhm_values: List[float] = []
    energy_bins: Optional[np.ndarray] = None

    set_directory = args.directory_folder / args.analysis_folder / args.set_name
    energy_range = (args.energy_range_min, args.energy_range_max)

    for case_directory in set_directory.iterdir():
        if not case_directory.is_dir():
            continue

        data_folder = case_directory / "ebeam"
        if not data_folder.is_dir():
            continue

        try:
            x, y, z, ux, uy, uz, w, q, ts = load_and_process_beam_data(
                data_folder, args.min_z
            )

            if len(z) == 0:
                energy_histograms.append(np.zeros(100))
                continue

            results = an.analyze_beam(
                x, y, z, ux, uy, uz, w, q, bins=args.beam_analysis_bins
            )
            energy_mev = an.convert_to_mev(ux, uy, uz)

            # Extract case value from folder name
            case = extract_case_value(case_directory)
            case_values.append(case)

            # Create energy histogram
            hist, bins = calculate_energy_histogram(
                energy_mev, w, args.energy_bins, energy_range
            )
            energy_histograms.append(hist)

            # Extract analysis from results dictionary
            charge_values.append(results["total_charge_c"] * 1e12)
            central_energy_values.append(
                results["energy_parameters"]["central_energy_mev"]
            )
            energy_fwhm_values.append(results["energy_parameters"]["energy_fwhm_mev"])

            # Calculate peak charge density and energy at peak
            peak_charge_den = calculate_peak_charge_density(
                hist, bins, results["total_charge_c"]
            )
            peak_charge_den_values.append(peak_charge_den)
            energy_at_peak_values.append(bins[np.argmax(hist)])

            # Store bins for later use (only need to do this once)
            if energy_bins is None:
                energy_bins = bins

        except Exception as e:
            print(f"Error processing {case_directory}: {e}")
            continue

    # Convert to numpy arrays and sort
    case_values = np.array(case_values) * args.factor
    if args.axis_offset is not None:
        case_values += args.axis_offset
    energy_histograms = np.array(energy_histograms)
    charge_values = np.array(charge_values)
    peak_charge_den_values = np.array(peak_charge_den_values)
    central_energy_values = np.array(central_energy_values)
    energy_at_peak_values = np.array(energy_at_peak_values)
    energy_fwhm_values = np.array(energy_fwhm_values)

    # Sort by case values
    sort_idx = np.argsort(case_values)
    case_values = case_values[sort_idx]
    energy_histograms = energy_histograms[sort_idx]
    charge_values = charge_values[sort_idx]
    peak_charge_den_values = peak_charge_den_values[sort_idx]
    central_energy_values = central_energy_values[sort_idx]
    energy_at_peak_values = energy_at_peak_values[sort_idx]
    energy_fwhm_values = energy_fwhm_values[sort_idx]

    return (
        case_values,
        energy_histograms,
        charge_values,
        peak_charge_den_values,
        central_energy_values,
        energy_at_peak_values,
        energy_fwhm_values,
    )


def create_waterfall_plot(
    case_values: np.ndarray,
    energy_histograms: np.ndarray,
    energy_bins: np.ndarray,
    charge_values: np.ndarray,
    set_name: str,
    units: str,
    show_title: bool,
    flip_y_axis: bool,
    top_right_annotation: Optional[str],
) -> None:
    """Create waterfall plot of energy spectra.

    Args:
        case_values: Array of case values.
        energy_histograms: Array of energy histograms.
        energy_bins: Energy bin edges.
        charge_values: Array of total charge values for each case.
        set_name: Name of the scan set.
        units: Units label for the scanned parameter.
        show_title: Whether to show plot title.
        flip_y_axis: Whether to flip the y-axis.
        top_right_annotation: Text to display in top-right corner, or None.

    Returns:
        None. Plot is displayed.
    """
    plt.figure(figsize=(12, 8))

    # Create 2D histogram-like waterfall plot
    energy_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
    x_mesh, y_mesh = np.meshgrid(energy_centers, case_values)

    # Calculate charge per MeV for color scale
    energy_slice_width = (
        energy_bins[1] - energy_bins[0]
    )  # Width of each energy slice in MeV

    # Convert histogram counts to actual charge in each slice
    # energy_histograms contains weighted particle counts, need to convert to charge
    # First, normalize by total charge to get fractional charge in each bin
    # Handle division by zero for cases with no electron beams
    hist_sums = np.sum(energy_histograms, axis=1, keepdims=True)
    normalized_histograms = np.where(
        hist_sums > 0, energy_histograms / hist_sums, np.zeros_like(energy_histograms)
    )

    # Multiply by total charge to get actual charge in each bin (in pC)
    charge_in_slices = normalized_histograms * charge_values[:, np.newaxis]

    # Calculate charge per MeV
    charge_per_mev = -1 * charge_in_slices / energy_slice_width

    # Plot as 2D histogram
    im = plt.pcolormesh(x_mesh, y_mesh, charge_per_mev, shading="auto", cmap="viridis")
    plt.colorbar(im, label="Charge per MeV (pC/MeV)")

    plt.xlabel("Energy (MeV)")
    plt.ylabel(f"{set_name} ({units})")

    # Add title if requested
    if show_title:
        plt.title(f"Energy Spectra Waterfall Plot vs {set_name}")

    # Add case value labels on y-axis
    plt.yticks(case_values, [f"{val:.2f}" for val in case_values])

    # Flip y-axis if requested
    if flip_y_axis:
        plt.gca().invert_yaxis()

    # Add top-right annotation if provided
    if top_right_annotation is not None and top_right_annotation:
        plt.text(
            0.04,
            0.98,
            top_right_annotation,
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.show()


def create_parameter_analysis_plots(
    case_values: np.ndarray,
    central_energy_values: np.ndarray,
    energy_at_peak_values: np.ndarray,
    peak_charge_den_values: np.ndarray,
    charge_values: np.ndarray,
    energy_fwhm_values: np.ndarray,
    set_name: str,
    units: str,
) -> None:
    """Create 2x2 subplot figure for parameter analysis.

    Args:
        case_values: Array of case values.
        central_energy_values: Array of central energy values.
        energy_at_peak_values: Array of energy at peak values.
        peak_charge_den_values: Array of peak charge density values.
        charge_values: Array of total charge values.
        energy_fwhm_values: Array of energy FWHM values.
        set_name: Name of the scan set.
        units: Units label for the scanned parameter.

    Returns:
        None. Plot is displayed.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Energy comparison
    ax1 = axes[0, 0]
    ax1.plot(case_values, central_energy_values, marker="o", label="Mean Energy (MeV)")
    ax1.plot(case_values, energy_at_peak_values, marker="s", label="Peak Energy (MeV)")
    ax1.set_xlabel(f"{set_name} ({units})")
    ax1.set_ylabel("Energy (MeV)")
    ax1.set_title("Energy vs Scanned Parameter")
    ax1.legend()

    # Plot 2: Charge density
    ax2 = axes[0, 1]
    ax2.plot(case_values, peak_charge_den_values, marker="^", color="tab:orange")
    ax2.set_xlabel(f"{set_name} ({units})")
    ax2.set_ylabel("Peak Charge Density (pC/MeV)")
    ax2.set_title("Peak Charge Density vs Scanned Parameter")

    # Plot 3: Total charge
    ax3 = axes[1, 0]
    ax3.plot(case_values, charge_values, marker="d", color="tab:green")
    ax3.set_xlabel(f"{set_name} ({units})")
    ax3.set_ylabel("Total Charge (pC)")
    ax3.set_title("Total Charge vs Scanned Parameter")

    # Plot 4: Energy FWHM
    ax4 = axes[1, 1]
    ax4.plot(case_values, energy_fwhm_values, marker="x", color="tab:red")
    ax4.set_xlabel(f"{set_name} ({units})")
    ax4.set_ylabel("Energy FWHM (MeV)")
    ax4.set_title("Energy FWHM vs Scanned Parameter")

    plt.tight_layout()
    plt.show()


def process(args: argparse.Namespace) -> None:
    """
    Main function to process simulation data and create plots.

    Args:
        args: Parsed command-line arguments. See module docstring for argument details.
    """
    (
        case_values,
        energy_histograms,
        charge_values,
        peak_charge_den_values,
        central_energy_values,
        energy_at_peak_values,
        energy_fwhm_values,
    ) = process_simulation_data(args)

    # Create waterfall plot
    energy_bins = np.linspace(
        args.energy_range_min, args.energy_range_max, args.energy_bins + 1
    )
    create_waterfall_plot(
        case_values,
        energy_histograms,
        energy_bins,
        charge_values,
        args.set_name,
        args.units,
        args.show_title,
        args.flip_y_axis,
        args.top_right_annotation,
    )

    # Create parameter analysis plots
    create_parameter_analysis_plots(
        case_values,
        central_energy_values,
        energy_at_peak_values,
        peak_charge_den_values,
        charge_values,
        energy_fwhm_values,
        args.set_name,
        args.units,
    )


def main() -> None:
    """Main entry point for script execution: parses command-line arguments and calls process()"""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
