"""Electron beam slice energy spread analysis and visualization.

This script loads an extracted ebeam diagnostic, prints the standard beam
analysis summary, and creates one figure with:
    - a weighted 2D histogram with z on the bottom axis and energy on the left axis
    - the relative RMS energy spread of each z slice plotted against z on the right axis

Usage:
    First, extract the ebeam from the full .h5 output file using either
    "extract_ebeam_from_hdf5.py" or "extract_ebeam_from_set.py". The extracted
    ebeam will be located in trimmed-down h5 files in an "ebeam" directory.

    Command-line usage:
        python plot_slice_energy_spread.py [OPTIONS]

    Examples:
        python plot_slice_energy_spread.py -d /path/to/ebeam
        python plot_slice_energy_spread.py -d /path/to/ebeam --min-uz 220 --slice-bins 100
        python plot_slice_energy_spread.py -d /path/to/ebeam --min-z 45018e-6 --max-z None
"""

from pathlib import Path
import argparse
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import inversion_fbpic.utils.analysis as an
from inversion_fbpic.utils.argparse_utils import build_selection_dict, float_or_none

DEFAULT_DIAG_FOLDER: Optional[Path] = None  # Path to ebeam .h5 files

DEFAULT_MIN_UZ: Optional[float] = None  # Minimum uz value for particle selection
DEFAULT_MAX_UZ: Optional[float] = None  # Maximum uz value for particle selection
DEFAULT_MIN_Z: Optional[float] = None  # Minimum z value for particle selection
DEFAULT_MAX_Z: Optional[float] = None  # Maximum z value for particle selection

DEFAULT_ITERATION_NUMBER: int = -1
DEFAULT_SPECIES: str = "n_elec"

DEFAULT_ANALYSIS_BINS: int = 200
DEFAULT_HIST_BINS: int = 300
DEFAULT_SLICE_BINS: int = 200
DEFAULT_MIN_PARTICLES_PER_SLICE: int = 2
DEFAULT_HIST_SMOOTHING_SIGMA: float = 0.7
DEFAULT_LINE_SMOOTHING_SIGMA: float = 0.7
DEFAULT_EMITTANCE_MAD_MULTIPLIER: Optional[float] = 5.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot an energy-vs-z histogram with relative z-slice RMS energy spread "
            "overlaid against z."
        )
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
        help="Which species to load from DIAG_FOLDER. Typically 'electrons' or 'n_elec'",
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
        "--analysis-bins",
        type=int,
        default=DEFAULT_ANALYSIS_BINS,
        help="Number of bins for the standard beam analysis current profile",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=DEFAULT_HIST_BINS,
        help="Number of bins for the 2D energy-vs-z histogram",
    )
    parser.add_argument(
        "--slice-bins",
        type=int,
        default=DEFAULT_SLICE_BINS,
        help="Number of z slices for slice energy spread analysis",
    )
    parser.add_argument(
        "--min-particles-per-slice",
        type=int,
        default=DEFAULT_MIN_PARTICLES_PER_SLICE,
        help="Minimum number of particles required to include a slice in the lineout",
    )
    parser.add_argument(
        "--hist-smoothing-sigma",
        type=float,
        default=DEFAULT_HIST_SMOOTHING_SIGMA,
        help="Gaussian smoothing sigma, in histogram bins, for the 2D histogram. Use 0 to disable",
    )
    parser.add_argument(
        "--line-smoothing-sigma",
        type=float,
        default=DEFAULT_LINE_SMOOTHING_SIGMA,
        help="Gaussian smoothing sigma, in slice bins, for the relative slice energy spread line. Use 0 to disable",
    )
    parser.add_argument(
        "--emittance-mad-multiplier",
        type=float_or_none,
        default=DEFAULT_EMITTANCE_MAD_MULTIPLIER,
        help=(
            "MAD multiplier used to remove x/y outliers before calculating "
            "normalized emittance for the plot legend. Use 'None', 'null', or 0 to disable"
        ),
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional path for saving the generated figure",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figure. Useful with --save-path for batch runs",
    )

    return parser.parse_args()


def weighted_mad(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Calculate weighted median and weighted median absolute deviation."""
    median = weighted_median(values, weights)
    mad = weighted_median(np.abs(values - median), weights)
    return median, mad


def weighted_median(data: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted median of a 1D numpy array."""
    data = np.asarray(data)
    weights = np.asarray(weights)

    if data.ndim != 1:
        raise TypeError("data must be a one dimensional array")
    if weights.ndim != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if np.sum(weights) <= 0:
        raise ValueError("the sum of the weights must be positive")

    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    cumulative_weights = np.cumsum(sorted_weights)
    median_positions = (cumulative_weights - 0.5 * sorted_weights) / cumulative_weights[
        -1
    ]
    return float(np.interp(0.5, median_positions, sorted_data))


def calculate_mad_filtered_normalized_emittance(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    w: np.ndarray,
    gamma_beam: float,
    mad_multiplier: Optional[float],
) -> tuple[dict[str, float], dict[str, int]]:
    """Calculate x/y normalized emittance after cutting x/y position outliers."""
    x_mask = calculate_mad_mask(x, w, mad_multiplier)
    y_mask = calculate_mad_mask(y, w, mad_multiplier)

    emittance_x = an.calculate_geometric_emittance(
        x[x_mask],
        y[x_mask],
        ux[x_mask],
        uy[x_mask],
        uz[x_mask],
        w[x_mask],
    )["x"]
    emittance_y = an.calculate_geometric_emittance(
        x[y_mask],
        y[y_mask],
        ux[y_mask],
        uy[y_mask],
        uz[y_mask],
        w[y_mask],
    )["y"]

    return (
        {"x": emittance_x * gamma_beam, "y": emittance_y * gamma_beam},
        {"x": int(np.count_nonzero(x_mask)), "y": int(np.count_nonzero(y_mask))},
    )


def calculate_mad_mask(
    values: np.ndarray,
    weights: np.ndarray,
    mad_multiplier: Optional[float],
) -> np.ndarray:
    """Return a MAD core mask, falling back to all particles if invalid."""
    mask = np.ones(len(values), dtype=bool)
    if mad_multiplier is not None and mad_multiplier > 0:
        median, mad = weighted_mad(values, weights)
        if mad > 0:
            mask = np.abs(values - median) <= mad_multiplier * mad

    if np.count_nonzero(mask) < 2 or np.sum(weights[mask]) <= 0:
        return np.ones(len(values), dtype=bool)

    return mask


def apply_emittance_mad_to_analysis(
    analysis_results: dict,
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    w: np.ndarray,
    mad_multiplier: Optional[float],
) -> dict:
    """Replace legend emittances with MAD-filtered values when enabled."""
    if mad_multiplier is None or mad_multiplier <= 0:
        analysis_results["emittance_mad_multiplier"] = mad_multiplier
        analysis_results["emittance_particles"] = {"x": len(w), "y": len(w)}
        return analysis_results

    emittance, particles = calculate_mad_filtered_normalized_emittance(
        x,
        y,
        ux,
        uy,
        uz,
        w,
        analysis_results["energy_parameters"]["gamma_beam"],
        mad_multiplier,
    )
    analysis_results["emittance"] = emittance
    analysis_results["emittance_particles"] = particles
    analysis_results["emittance_mad_multiplier"] = mad_multiplier
    return analysis_results


def calculate_z_slice_energy_spread(
    z: np.ndarray,
    energy_mev: np.ndarray,
    w: np.ndarray,
    bins: int = DEFAULT_SLICE_BINS,
    min_particles_per_slice: int = DEFAULT_MIN_PARTICLES_PER_SLICE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate weighted mean energy and RMS energy spread for each z slice.

    This mirrors the weighted RMS calculation used by
    ``analysis.calculate_slice_energy_spread`` while also returning the slice
    mean energy for summary statistics.
    """
    z_bin_edges = np.linspace(np.min(z), np.max(z), bins + 1)
    z_bin_centers = (z_bin_edges[:-1] + z_bin_edges[1:]) / 2

    slice_z: list[float] = []
    slice_mean_energy: list[float] = []
    slice_energy_spread: list[float] = []

    for i in range(bins):
        if i == bins - 1:
            slice_mask = (z >= z_bin_edges[i]) & (z <= z_bin_edges[i + 1])
        else:
            slice_mask = (z >= z_bin_edges[i]) & (z < z_bin_edges[i + 1])

        if np.count_nonzero(slice_mask) < min_particles_per_slice:
            continue

        energy_slice = energy_mev[slice_mask]
        w_slice = w[slice_mask]

        if np.sum(w_slice) <= 0:
            continue

        weighted_mean = np.average(energy_slice, weights=w_slice)
        weighted_variance = np.average(
            (energy_slice - weighted_mean) ** 2, weights=w_slice
        )

        slice_z.append(float(z_bin_centers[i]))
        slice_mean_energy.append(float(weighted_mean))
        slice_energy_spread.append(float(np.sqrt(weighted_variance)))

    return (
        np.array(slice_z),
        np.array(slice_mean_energy),
        np.array(slice_energy_spread),
    )


def calculate_slice_bin_indices(
    slice_z: np.ndarray,
    z: np.ndarray,
    bins: int,
) -> np.ndarray:
    """Return original uniform z-bin indices for retained slice centers."""
    if len(slice_z) == 0:
        return np.array([], dtype=int)

    z_min = np.min(z)
    z_max = np.max(z)
    if bins <= 0 or z_max == z_min:
        return np.arange(len(slice_z), dtype=int)

    bin_width = (z_max - z_min) / bins
    first_center = z_min + 0.5 * bin_width
    indices = np.rint((slice_z - first_center) / bin_width).astype(int)
    return np.clip(indices, 0, bins - 1)


def smooth_contiguous_slice_runs(
    values: np.ndarray,
    slice_bin_indices: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Smooth only across adjacent retained z bins, preserving sparse gaps."""
    if sigma <= 0 or len(values) < 2:
        return values

    smoothed = values.copy()
    run_start = 0
    run_breaks = np.flatnonzero(np.diff(slice_bin_indices) != 1) + 1

    for run_end in np.append(run_breaks, len(values)):
        if run_end - run_start > 1:
            smoothed[run_start:run_end] = gaussian_filter1d(
                values[run_start:run_end],
                sigma=sigma,
            )
        run_start = run_end

    return smoothed


def insert_gaps_between_noncontiguous_bins(
    x_values: np.ndarray,
    y_values: np.ndarray,
    slice_bin_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN separators so plotted lines do not bridge omitted z bins."""
    if len(x_values) < 2:
        return x_values, y_values

    gap_positions = np.flatnonzero(np.diff(slice_bin_indices) != 1) + 1
    if len(gap_positions) == 0:
        return x_values, y_values

    return (
        np.insert(x_values, gap_positions, np.nan),
        np.insert(y_values, gap_positions, np.nan),
    )


def create_slice_energy_spread_plot(
    z: np.ndarray,
    energy_mev: np.ndarray,
    w: np.ndarray,
    analysis_results: dict,
    slice_z: np.ndarray,
    slice_mean_energy: np.ndarray,
    slice_energy_spread: np.ndarray,
    slice_bin_indices: Optional[np.ndarray] = None,
    hist_bins: int = DEFAULT_HIST_BINS,
    hist_smoothing_sigma: float = DEFAULT_HIST_SMOOTHING_SIGMA,
    line_smoothing_sigma: float = DEFAULT_LINE_SMOOTHING_SIGMA,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Create the energy-vs-z histogram with slice energy spread overlay."""
    fig, ax_hist = plt.subplots(figsize=(9, 6))
    z_centroid = analysis_results["beam_sizes"]["z_mean_m"]
    z_centered_um = (z - z_centroid) * 1e6
    slice_z_centered_um = (slice_z - z_centroid) * 1e6

    hist, z_edges, energy_edges = np.histogram2d(
        z_centered_um,
        energy_mev,
        bins=hist_bins,
        weights=w,
    )
    if hist_smoothing_sigma > 0:
        hist = gaussian_filter(hist, sigma=hist_smoothing_sigma)

    ax_hist.pcolormesh(
        z_edges,
        energy_edges,
        hist.T,
        shading="auto",
        cmap="Reds",
    )

    ax_hist.set_xlabel(r"$z - \langle z \rangle$ (um)")
    ax_hist.set_ylabel("Energy (MeV)")
    ax_hist.set_title("Energy vs Z with Slice Energy Spread")
    ax_hist.grid(alpha=0.2)

    relative_slice_energy_spread = 100 * slice_energy_spread / slice_mean_energy
    if slice_bin_indices is None:
        slice_bin_indices = np.arange(len(relative_slice_energy_spread), dtype=int)
    if line_smoothing_sigma > 0:
        relative_slice_energy_spread = smooth_contiguous_slice_runs(
            relative_slice_energy_spread,
            slice_bin_indices,
            line_smoothing_sigma,
        )

    plot_slice_z_centered_um, plot_relative_slice_energy_spread = (
        insert_gaps_between_noncontiguous_bins(
            slice_z_centered_um,
            relative_slice_energy_spread,
            slice_bin_indices,
        )
    )

    ax_spread = ax_hist.twinx()
    (spread_line,) = ax_spread.plot(
        plot_slice_z_centered_um,
        plot_relative_slice_energy_spread,
        "o-",
        color="tab:blue",
        linewidth=1.5,
        markersize=3,
        label="Relative Slice Energy Spread",
    )
    ax_spread.set_ylabel("Relative Slice Energy Spread (%)", color="tab:blue")
    ax_spread.tick_params(axis="y", labelcolor="tab:blue")

    beam_stats = (
        f"Charge: {abs(analysis_results['total_charge_c']) * 1e12:.2f} pC\n"
        rf"$\epsilon_{{nx}}$: {analysis_results['emittance']['x'] * 1e6:.2f} um rad"
        "\n"
        rf"$\epsilon_{{ny}}$: {analysis_results['emittance']['y'] * 1e6:.2f} um rad"
        "\n"
        rf"$\sigma_z$: {analysis_results['beam_sizes']['sigma_z_m'] * 1e6:.2f} um"
    )
    stats_handle = Line2D([], [], linestyle="none", label=beam_stats)
    ax_spread.legend(handles=[spread_line, stats_handle], loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_slice_summary(
    slice_mean_energy: np.ndarray,
    slice_energy_spread: np.ndarray,
) -> None:
    """Print summary statistics for the z-slice energy spread analysis."""
    if len(slice_mean_energy) == 0:
        print("\n=== SLICE ENERGY SPREAD SUMMARY (UNSMOOTHED) ===")
        print("No valid slices found.")
        return

    print("\n=== SLICE ENERGY SPREAD SUMMARY (UNSMOOTHED) ===")
    print(f"Number of z slices analyzed: {len(slice_mean_energy)}")
    print(
        f"Slice mean energy range: {np.min(slice_mean_energy):.1f} - {np.max(slice_mean_energy):.1f} MeV"
    )
    relative_slice_energy_spread = 100 * slice_energy_spread / slice_mean_energy
    print(
        f"Average relative slice RMS energy spread: {np.mean(relative_slice_energy_spread):.2f}%"
    )
    print(
        f"Minimum relative slice RMS energy spread: {np.min(relative_slice_energy_spread):.2f}%"
    )
    print(
        f"Maximum relative slice RMS energy spread: {np.max(relative_slice_energy_spread):.2f}%"
    )


def process(args: argparse.Namespace) -> None:
    """Load beam data, analyze it, and generate the slice energy spread plot."""
    if args.diag_folder is None:
        raise RuntimeError(
            "Must specify path to the ebeam .h5 files directory using: -d, --diag-folder"
        )

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

    if len(w) == 0:
        print("Error: No particles found after applying selection.")
        return

    analysis = an.analyze_beam(x, y, z, ux, uy, uz, w, q, bins=args.analysis_bins)
    analysis = apply_emittance_mad_to_analysis(
        analysis,
        x,
        y,
        ux,
        uy,
        uz,
        w,
        args.emittance_mad_multiplier,
    )
    an.print_beam_summary(analysis)
    if (
        analysis["emittance_mad_multiplier"] is not None
        and analysis["emittance_mad_multiplier"] > 0
    ):
        print(
            "Emittance MAD filter: "
            f"x kept {analysis['emittance_particles']['x']:,}/{len(w):,}, "
            f"y kept {analysis['emittance_particles']['y']:,}/{len(w):,}"
        )
    z_centroid = analysis["beam_sizes"]["z_mean_m"]
    print(f"Beam z centroid: {z_centroid:.9e} m ({z_centroid * 1e6:.6f} um)")

    energy_mev = an.convert_to_mev(ux, uy, uz)
    slice_z, slice_mean_energy, slice_energy_spread = calculate_z_slice_energy_spread(
        z,
        energy_mev,
        w,
        bins=args.slice_bins,
        min_particles_per_slice=args.min_particles_per_slice,
    )
    slice_bin_indices = calculate_slice_bin_indices(slice_z, z, args.slice_bins)

    print_slice_summary(slice_mean_energy, slice_energy_spread)
    if len(slice_z) == 0:
        return

    create_slice_energy_spread_plot(
        z,
        energy_mev,
        w,
        analysis,
        slice_z,
        slice_mean_energy,
        slice_energy_spread,
        slice_bin_indices=slice_bin_indices,
        hist_bins=args.hist_bins,
        hist_smoothing_sigma=args.hist_smoothing_sigma,
        line_smoothing_sigma=args.line_smoothing_sigma,
        save_path=args.save_path,
        show=not args.no_show,
    )


def main() -> None:
    """Main entry point for script execution."""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
