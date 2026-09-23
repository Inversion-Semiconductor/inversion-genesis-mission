"""Script to analyze modulation wavelengths in electron beam current profiles.

This script loads electron beam data from FBPIC simulations, calculates the current
profile, and performs FFT analysis to identify dominant modulation wavelengths.
The analysis is presented in both wavelength and normalized wavenumber space.

Usage:
    Command-line usage:
        python modulated_ebeam_frequency.py [OPTIONS]

    Optional Arguments:
        -d, --diag-folder PATH
            Path to the directory containing beam data files. If not provided, uses
            the default value defined in the script.

        --min-z FLOAT|None
            Minimum z coordinate for filtering electrons (in meters). Use 'None' or
            'null' to explicitly set to None.
            Default: 589e-6

        --max-z FLOAT|None
            Maximum z coordinate for filtering electrons (in meters). Use 'None' or
            'null' to explicitly set to None.
            Default: 590.5e-6

        -i, --iteration-number INT
            Which dump to load (-1 for final dump).
            Default: -1

        -s, --species STR
            Species name to analyze.
            Default: "electrons"

        -b, --bins INT
            Number of bins for current calculation. Must be a positive integer;
            zero or negative values will raise an error.
            Default: 226

        --min-wavelength-um FLOAT
            Minimum wavelength to analyze (in micrometers).
            Default: 0.01

        --lambda-ref-um FLOAT
            Reference wavelength for normalization (in micrometers). Must be
            a positive value; zero or negative values will raise an error.
            Default: 1.2

        --peak-threshold FLOAT
            Minimum peak amplitude relative to maximum.
            Default: 0.1

        --num-dominant-peaks INT
            Number of dominant peaks to highlight.
            Default: 2

    Examples:
        # Use all defaults
        python modulated_ebeam_frequency.py

        # Specify a different diagnostics folder and z range
        python modulated_ebeam_frequency.py -d /path/to/ebeam --min-z 500e-6 --max-z 600e-6

        # Analyze with different reference wavelength
        python modulated_ebeam_frequency.py --lambda-ref-um 2.0 --num-dominant-peaks 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import inversion_fbpic.utils.analysis as an
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.fft import fft, fftfreq

from inversion_fbpic.utils.argparse_utils import float_or_none

# Default configuration constants
DEFAULT_DIAG_FOLDER: Path = Path(
    "../../../../../../inversion-fbpic-runscripts/simulations/pwfa/pwfa_modulated_test/analysis/diags/ebeam/"
)
DEFAULT_MIN_Z: Optional[float] = 589e-6  # Minimum z when selecting electrons (m)
DEFAULT_MAX_Z: Optional[float] = 590.5e-6  # Maximum z when selecting electrons (m)
DEFAULT_ITERATION_NUMBER: int = -1  # Which dump to load (-1 for final dump)
DEFAULT_SPECIES: str = "electrons"  # Species to analyze
DEFAULT_BINS: int = 226  # Number of bins for current calculation
DEFAULT_MIN_WAVELENGTH_UM: float = 0.01  # Minimum wavelength to analyze (μm)
DEFAULT_LAMBDA_REF_UM: float = 1.2  # Reference wavelength for normalization (μm)
DEFAULT_PEAK_THRESHOLD: float = 0.1  # Minimum peak amplitude relative to maximum
DEFAULT_NUM_DOMINANT_PEAKS: int = 2  # Number of dominant peaks to highlight

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze modulation wavelengths in electron beam current profiles."
    )

    parser.add_argument(
        "-d",
        "--diag-folder",
        type=Path,
        default=DEFAULT_DIAG_FOLDER,
        help="Path to the directory containing beam data files",
    )

    parser.add_argument(
        "--min-z",
        type=float_or_none,
        default=DEFAULT_MIN_Z,
        help="Minimum z coordinate for filtering electrons (in meters). Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "--max-z",
        type=float_or_none,
        default=DEFAULT_MAX_Z,
        help="Maximum z coordinate for filtering electrons (in meters). Use 'None' or 'null' to explicitly set to None",
    )

    parser.add_argument(
        "-i",
        "--iteration-number",
        type=int,
        default=DEFAULT_ITERATION_NUMBER,
        help="Which dump to load (-1 for final dump)",
    )

    parser.add_argument(
        "-s",
        "--species",
        type=str,
        default=DEFAULT_SPECIES,
        help="Species name to analyze",
    )

    parser.add_argument(
        "-b",
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of bins for current calculation (must be a positive integer)",
    )

    parser.add_argument(
        "--min-wavelength-um",
        type=float,
        default=DEFAULT_MIN_WAVELENGTH_UM,
        help="Minimum wavelength to analyze (in micrometers)",
    )

    parser.add_argument(
        "--lambda-ref-um",
        type=float,
        default=DEFAULT_LAMBDA_REF_UM,
        help="Reference wavelength for normalization (in micrometers). Must be positive.",
    )

    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=DEFAULT_PEAK_THRESHOLD,
        help="Minimum peak amplitude relative to maximum",
    )

    parser.add_argument(
        "--num-dominant-peaks",
        type=int,
        default=DEFAULT_NUM_DOMINANT_PEAKS,
        help="Number of dominant peaks to highlight",
    )

    args = parser.parse_args()

    # Validate bins is positive (required for histogram binning)
    if args.bins <= 0:
        parser.error("--bins must be a positive integer (got {})".format(args.bins))

    # Validate lambda_ref_um is positive (used in division for wavenumber normalization)
    if args.lambda_ref_um <= 0:
        parser.error(
            "--lambda-ref-um must be a positive value (got {})".format(args.lambda_ref_um)
        )

    return args


def load_and_filter_beam_data(
    diag_folder: Path,
    species: str,
    iteration: int,
    min_z: Optional[float] = None,
    max_z: Optional[float] = None,
) -> Tuple[ndarray, ...]:
    """Load and filter electron beam data.

    Args:
        diag_folder: Path to directory containing beam data files.
        species: Species name to load (typically "electrons").
        iteration: Iteration number to load (-1 for final dump).
        min_z: Minimum z coordinate for filtering (m).
        max_z: Maximum z coordinate for filtering (m).

    Returns:
        Tuple containing filtered beam data arrays (x, y, z, ux, uy, uz, w).

    Raises:
        FileNotFoundError: If data files cannot be found.
        ValueError: If data loading fails.
    """
    try:
        x, y, z, ux, uy, uz, w, q, ts = an.load_beam_data(
            diag_folder, species, iteration=iteration
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data in {diag_folder}")
    except Exception as e:
        raise ValueError(f"Error loading beam data: {e}")

    # Create data dictionary for filtering
    arrs: dict[str, ndarray] = {
        "x": x,
        "y": y,
        "z": z,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "w": w,
    }

    # Apply z-coordinate filtering
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
        q,
    )


def calculate_current_profile(
    z: ndarray, ux: ndarray, uy: ndarray, uz: ndarray, w: ndarray, q: float, bins: int
) -> Tuple[ndarray, ndarray]:
    """Calculate current profile from beam data.

    Args:
        z: Particle z-coordinates.
        ux: Particle x-velocities.
        uy: Particle y-velocities.
        uz: Particle z-velocities.
        w: Particle weights.
        q: Particle charge.
        bins: Number of bins for current calculation.

    Returns:
        Tuple of (current_profile, z_axis) arrays.
    """
    return an.calculate_current(z, ux, uy, uz, w, q, bins=bins)


def perform_fft_analysis(
    current: ndarray, z_axis: ndarray, min_wavelength_um: float, lambda_ref_um: float
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """Perform FFT analysis on current profile.

    Args:
        current: Current profile array.
        z_axis: Z-axis coordinates corresponding to current bins.
        min_wavelength_um: Minimum wavelength to include in analysis (μm).
        lambda_ref_um: Reference wavelength for normalization (μm).

    Returns:
        Tuple of (wavelengths_um, k_normalized, fft_amplitudes, z_spacing).
    """
    # Remove DC component and apply window to reduce spectral leakage
    current_detrended = current - np.mean(current)
    window = np.hanning(len(current_detrended))
    current_windowed = current_detrended * window

    # Perform FFT
    fft_result = fft(current_windowed)
    freqs = fftfreq(len(current_windowed), z_axis[1] - z_axis[0])

    # Take only positive frequencies and corresponding amplitudes
    positive_mask = freqs > 0
    freqs_positive = freqs[positive_mask]
    fft_positive = np.abs(fft_result[positive_mask])

    # Convert frequencies to wavelengths and wavenumbers
    wavelengths_um = 1e6 / freqs_positive  # Convert from m to μm
    k_values = 2 * np.pi / (wavelengths_um * 1e-6)  # wavenumber in rad/m

    # Reference wavenumber for normalization
    k_ref = 2 * np.pi / (lambda_ref_um * 1e-6)  # rad/m
    k_normalized = k_values / k_ref  # normalized wavenumber

    # Filter to show spectrum down to minimum wavelength
    wavelength_mask = wavelengths_um >= min_wavelength_um
    wavelengths_filtered = wavelengths_um[wavelength_mask]
    k_normalized_filtered = k_normalized[wavelength_mask]
    fft_filtered = fft_positive[wavelength_mask]

    return (
        wavelengths_filtered,
        k_normalized_filtered,
        fft_filtered,
        z_axis[1] - z_axis[0],
    )


def find_peaks(fft_amplitudes: ndarray, threshold: float) -> List[int]:
    """Find peaks in FFT spectrum.

    Args:
        fft_amplitudes: FFT amplitude array.
        threshold: Minimum peak amplitude relative to maximum.

    Returns:
        List of peak indices.
    """
    peak_indices = []
    max_amplitude = np.max(fft_amplitudes)
    threshold_value = threshold * max_amplitude

    for i in range(1, len(fft_amplitudes) - 1):
        if (
            fft_amplitudes[i] > fft_amplitudes[i - 1]
            and fft_amplitudes[i] > fft_amplitudes[i + 1]
            and fft_amplitudes[i] > threshold_value
        ):
            peak_indices.append(i)

    return peak_indices


def get_dominant_peaks(
    peak_indices: List[int],
    wavelengths: ndarray,
    k_normalized: ndarray,
    fft_amplitudes: ndarray,
    num_peaks: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Get the most dominant peaks from the FFT spectrum.

    Args:
        peak_indices: List of peak indices.
        wavelengths: Wavelength array (μm).
        k_normalized: Normalized wavenumber array.
        fft_amplitudes: FFT amplitude array.
        num_peaks: Number of dominant peaks to return.

    Returns:
        Tuple of (dominant_wavelengths, dominant_k_norm, dominant_amplitudes).
    """
    if not peak_indices:
        return [], [], []

    # Sort peaks by amplitude (descending order)
    peak_amplitudes = fft_amplitudes[peak_indices]
    sorted_indices = np.argsort(peak_amplitudes)[::-1]  # Sort in descending order

    # Get top N peaks (or fewer if less than N peaks exist)
    top_n_peaks = min(num_peaks, len(sorted_indices))
    top_peak_indices = [peak_indices[sorted_indices[i]] for i in range(top_n_peaks)]

    dominant_wavelengths = wavelengths[top_peak_indices].tolist()
    dominant_k_norm = k_normalized[top_peak_indices].tolist()
    dominant_amplitudes = fft_amplitudes[top_peak_indices].tolist()

    return dominant_wavelengths, dominant_k_norm, dominant_amplitudes


def print_peak_results(
    dominant_wavelengths: List[float],
    dominant_k_norm: List[float],
    dominant_amplitudes: List[float],
) -> None:
    """Print peak analysis results.

    Args:
        dominant_wavelengths: List of dominant wavelengths (μm).
        dominant_k_norm: List of normalized wavenumbers.
        dominant_amplitudes: List of peak amplitudes.
    """
    if not dominant_wavelengths:
        logger.info("No significant peaks found.")
        return

    logger.info("Top modulation wavelengths:")
    for i, (wavelength, k_norm, amplitude) in enumerate(
        zip(dominant_wavelengths, dominant_k_norm, dominant_amplitudes)
    ):
        wavelength_nm = wavelength * 1000  # Convert to nm
        logger.info(
            f"  #{i+1}: {wavelength:.3f} μm ({wavelength_nm:.1f} nm) - "
            f"k/k_ref = {k_norm:.2f} - amplitude: {amplitude:.2e}"
        )


def create_plots(
    z_axis: ndarray,
    current: ndarray,
    k_normalized: ndarray,
    fft_amplitudes: ndarray,
    dominant_wavelengths: List[float],
    dominant_k_norm: List[float],
    dominant_amplitudes: List[float],
    min_z: Optional[float],
    max_z: Optional[float],
    z_spacing: float,
    lambda_ref_um: float,
) -> None:
    """Create and display analysis plots.

    Args:
        z_axis: Z-axis coordinates.
        current: Current profile array.
        k_normalized: Normalized wavenumber array.
        fft_amplitudes: FFT amplitude array.
        dominant_wavelengths: List of dominant wavelengths (μm).
        dominant_k_norm: List of normalized wavenumbers.
        dominant_amplitudes: List of peak amplitudes.
        min_z: Minimum z coordinate used (m).
        max_z: Maximum z coordinate used (m).
        z_spacing: Spacing between z bins (m).
        lambda_ref_um: Reference wavelength for normalization (μm).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot current profile
    ax1.plot(z_axis * 1e6, current * 1e-3, "b-", linewidth=1)
    bin_size_nm = z_spacing * 1e9
    z_range_str = (
        f"{min_z*1e6:.1f}-{max_z*1e6:.1f}" if min_z and max_z else "full range"
    )
    ax1.set_title(
        f"Current Profile (Bin Size = {bin_size_nm:.03f} nm, z range: {z_range_str} μm)"
    )
    ax1.set_xlabel("z (μm)")
    ax1.set_ylabel("Current (kA)")
    ax1.grid(True, alpha=0.3)
    if min_z and max_z:
        ax1.set_xlim([min_z * 1e6, max_z * 1e6])

    # Plot FFT spectrum vs normalized wavenumber
    ax2.plot(k_normalized, fft_amplitudes, "r-", linewidth=1, label="FFT Spectrum")
    ax2.set_title(
        f"Fourier Transform of Current Profile (k normalized to λ = {lambda_ref_um} μm)"
    )
    ax2.set_xlabel(f"k/k_ref (where k_ref = 2π/λ_ref, λ_ref = {lambda_ref_um} μm)")
    ax2.set_ylabel("FFT Amplitude")
    ax2.grid(True, alpha=0.3)

    # Add vertical line at k/k_ref = 1 for reference
    ax2.axvline(
        1.0,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label=f"k_ref (λ = {lambda_ref_um} μm)",
    )

    # Highlight the dominant peaks
    if dominant_wavelengths:
        colors = ["green", "orange", "purple"]
        for i, (wavelength, amplitude, k_norm) in enumerate(
            zip(dominant_wavelengths, dominant_amplitudes, dominant_k_norm)
        ):
            if i < len(colors):  # Only plot up to available colors
                ax2.axvline(
                    k_norm,
                    color=colors[i],
                    linestyle="--",
                    alpha=0.7,
                    label=f"#{i+1}: k/k_ref = {k_norm:.2f} (λ = {wavelength:.3f} μm)",
                )

        ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def process(args: argparse.Namespace) -> None:
    """
    Main function to perform electron beam modulation analysis.

    Loads beam data, calculates current profile, performs FFT analysis,
    and displays results with plots.

    Args:
        args: Parsed command-line arguments. See module docstring for argument details.
    """
    try:
        # Load and filter beam data
        x, y, z, ux, uy, uz, w, q = load_and_filter_beam_data(
            args.diag_folder,
            args.species,
            args.iteration_number,
            args.min_z,
            args.max_z,
        )

        # Calculate current profile
        current, z_axis = calculate_current_profile(z, ux, uy, uz, w, q, args.bins)

        # Perform FFT analysis
        wavelengths, k_normalized, fft_amplitudes, z_spacing = perform_fft_analysis(
            current, z_axis, args.min_wavelength_um, args.lambda_ref_um
        )

        # Find peaks and get dominant ones
        peak_indices = find_peaks(fft_amplitudes, args.peak_threshold)
        dominant_wavelengths, dominant_k_norm, dominant_amplitudes = get_dominant_peaks(
            peak_indices,
            wavelengths,
            k_normalized,
            fft_amplitudes,
            args.num_dominant_peaks,
        )

        # Print results
        print_peak_results(dominant_wavelengths, dominant_k_norm, dominant_amplitudes)

        # Create plots
        create_plots(
            z_axis,
            current,
            k_normalized,
            fft_amplitudes,
            dominant_wavelengths,
            dominant_k_norm,
            dominant_amplitudes,
            args.min_z,
            args.max_z,
            z_spacing,
            args.lambda_ref_um,
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


def main() -> None:
    """Main entry point for script execution: parses command-line arguments and calls process()"""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
