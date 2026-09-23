"""
Expanding upon `visualize_ebeamparams_vs_scan.py`, this script summarizes the linear trends of all ebeam parameters
against all scanned parameters in a given "analysis_X" directory.

This script auto-discovers scan sets in a specified analysis directory, analyzes beam parameters for each scan,
fits a linear slope and R^2 for each parameter, normalizes scan parameters to percent-variation, prints a summary table,
and saves both the slopes and the raw parameter-vs-scan data to CSV files.

Usage:
    Command-line usage:
        python summarize_scan_sensitivities.py [OPTIONS]

    Optional Arguments:
        --directory-folder PATH
            Path to the directory containing all scans. If not provided, uses the default
            value defined in the script.

        --analysis-folder STR
            Directory name containing the specific scan set of interest, typically "analysis_X".
            If not provided, uses the default value defined in the script.

        --min-z FLOAT|None
            Minimum z position of ebeam particles to consider. Use 'None' or 'null' to
            explicitly set to None.
            Default: 4860e-6

    Examples:
        # Use all defaults
        python summarize_scan_sensitivities.py

        # Specify a different directory and analysis folder
        python summarize_scan_sensitivities.py --directory-folder /path/to/scans --analysis-folder analysis_1

        # Override min-z threshold
        python summarize_scan_sensitivities.py --min-z 5000e-6
"""

import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from inversion_fbpic.utils.analysis import analyze_beam, load_beam_data, apply_cut
from inversion_fbpic.utils.argparse_utils import float_or_none

# Default configuration variables
DEFAULT_DIRECTORY_FOLDER: Path = Path(
    "../../../../../../inversion-fbpic-runscripts/simulations/htu/version_1/htu_scans"
)  # Directory containing all scans
DEFAULT_ANALYSIS_FOLDER: str = (
    "analysis_3"  # Set of scans to analyze, typically "analysis_X"
)
DEFAULT_MIN_Z: Optional[float] = 4860e-6  # Default minimum z position

# Parameters to analyze
PARAMETERS: list[tuple[str, str]] = [
    ("central_energy", "Central Energy (MeV)"),
    ("energy_std", "Energy RMS (MeV)"),
    ("energy_fwhm", "Energy FWHM (MeV)"),
    ("total_charge", "Total Charge (pC)"),
    ("emittance_x", "Emittance x (mm-mrad)"),
    ("emittance_y", "Emittance y (mm-mrad)"),
    ("sigma_z", "Sigma z (μm)"),
    ("peak_current", "Peak Current (kA)"),
]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Summarize linear trends of ebeam parameters across parameter scans."
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
        help="Directory name containing the specific scan set of interest, typically 'analysis_X'",
    )

    parser.add_argument(
        "--min-z",
        type=float_or_none,
        default=DEFAULT_MIN_Z,
        help="Minimum z position of ebeam particles to consider. Use 'None' or 'null' to explicitly set to None",
    )

    return parser.parse_args()


def process(args: argparse.Namespace) -> None:
    """
    Main entry point for scan sensitivity summarization.

    Args:
        args: Parsed command-line arguments. See module docstring for argument details.
    """
    scan_root = args.directory_folder / args.analysis_folder
    scan_sets = [
        d for d in scan_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    summary_rows: List[List] = []
    raw_data_rows: List[List] = []
    print(f"{'Scan Set':<25} " + " ".join([f"{name:<20}" for _, name in PARAMETERS]))
    print("-" * (25 + 22 * len(PARAMETERS)))
    for scan_set in scan_sets:
        case_values, results_dict = analyze_scan(
            scan_set, args.analysis_folder, args.min_z
        )
        norm_case_values = normalize_to_percent_variation(case_values)
        row = [scan_set.name]
        slopes = []
        r2s = []
        for k, _ in PARAMETERS:
            slope, r2 = fit_slope_and_r2(norm_case_values, results_dict[k])
            slopes.append(slope)
            r2s.append(r2)
            row.append(f"{slope:.4g} (R2={r2:.2f})")
        print(f"{scan_set.name:<25} " + " ".join([f"{s:<20}" for s in row[1:]]))
        summary_rows.append([scan_set.name] + slopes + r2s)
        # Save raw data for this scan set
        for i in range(len(case_values)):
            raw_data_rows.append(
                [
                    scan_set.name,
                    case_values[i],
                    norm_case_values[i],
                    *[results_dict[k][i] for k, _ in PARAMETERS],
                ]
            )
    # Write summary CSV
    with open("scan_sensitivities_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Scan Set"]
            + [f"{k} Slope" for k, _ in PARAMETERS]
            + [f"{k} R2" for k, _ in PARAMETERS]
        )
        writer.writerows(summary_rows)
    # Write raw data CSV
    with open("scan_sensitivities_raw.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Scan Set",
                "Scan Value",
                "Normalized Scan Value",
                *[name for _, name in PARAMETERS],
            ]
        )
        writer.writerows(raw_data_rows)


def analyze_scan(
    scan_dir: Path, analysis_folder: str, min_z: Optional[float]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Analyze a single scan set directory.

    Args:
        scan_dir: Path to the scan set directory.
        analysis_folder: Name of the analysis folder (e.g., 'analysis_3').
        min_z: Minimum z position of ebeam particles to consider.

    Returns:
        Tuple of (case_values, results_dict), where case_values is a sorted numpy array of scan parameter values,
        and results_dict is a dict of parameter arrays sorted to match case_values.
    """
    case_values: List[float] = []
    results_dict: Dict[str, List[float]] = {k: [] for k, _ in PARAMETERS}
    for case_directory in scan_dir.iterdir():
        if case_directory.is_dir():
            data_folder = case_directory / "ebeam"
            if data_folder.is_dir():
                x, y, z, ux, uy, uz, w, q, _ = load_beam_data(
                    data_folder, "electrons", iteration=-1
                )
                arrs = {"x": x, "y": y, "z": z, "ux": ux, "uy": uy, "uz": uz, "w": w}
                arrs = apply_cut(arrs, "z", min_z, op="gt")

                x, y, z, ux, uy, uz, w = (
                    arrs["x"],
                    arrs["y"],
                    arrs["z"],
                    arrs["ux"],
                    arrs["uy"],
                    arrs["uz"],
                    arrs["w"],
                )
                if len(z) > 0:
                    results = analyze_beam(x, y, z, ux, uy, uz, w, q, bins=200)
                    # Extract scan value from folder name
                    case = float(case_directory.name.split("_")[0])
                    case_values.append(case)
                    results_dict["central_energy"].append(
                        results["energy_parameters"]["central_energy_mev"]
                    )
                    results_dict["energy_std"].append(
                        results["energy_parameters"]["energy_std_mev"]
                    )
                    results_dict["energy_fwhm"].append(
                        results["energy_parameters"]["energy_fwhm_mev"]
                    )
                    results_dict["total_charge"].append(
                        results["total_charge_c"] * 1e12
                    )  # pC
                    results_dict["emittance_x"].append(
                        results["emittance"]["x"] * 1e6
                    )  # mm-mrad
                    results_dict["emittance_y"].append(
                        results["emittance"]["y"] * 1e6
                    )  # mm-mrad
                    results_dict["sigma_z"].append(
                        results["beam_sizes"]["sigma_z_m"] * 1e6
                    )  # um
                    results_dict["peak_current"].append(
                        np.max(results["current"]) * 1e-3
                    )  # kA
    # Sort by case
    case_values_np = np.array(case_values)
    sort_idx = np.argsort(case_values_np)
    case_values_np = case_values_np[sort_idx]
    for k in results_dict:
        results_dict[k] = np.array(results_dict[k])[sort_idx]
    return case_values_np, results_dict


def normalize_to_percent_variation(values: np.ndarray) -> np.ndarray:
    """Normalize an array to percent-variation about its median value.

    Args:
        values: 1D numpy array of scan parameter values.

    Returns:
        1D numpy array of percent-variation, where 0 is the median, and endpoints are -0.5 and 0.5 for symmetric scans.
    """
    median = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val == min_val:
        return np.zeros_like(values)
    return (values - median) / (max_val - min_val)


def fit_slope_and_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit a linear slope and compute R^2 for y vs. x.

    Args:
        x: 1D numpy array of x values (scan parameter, normalized).
        y: 1D numpy array of y values (beam parameter).

    Returns:
        Tuple of (slope, r2), where slope is the linear fit slope and r2 is the coefficient of determination.
    """
    if len(x) < 2:
        return float("nan"), float("nan")
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    y_fit = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, r2


def main() -> None:
    """Main entry point for script execution: parses command-line arguments and calls process()"""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
