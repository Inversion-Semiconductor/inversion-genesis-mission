"""Script to fit density profile parameters to experimental lineout data.

This script uses scipy.optimize to find the best-fit parameters for the
build_generalized_gaussian_with_downramp function to match experimental lineout data.

Usage:
    Run `plasma_lineout_analysis.py` first to generate the output .npy file
    Adjust the initial parameter guesses in the INITIAL_PARAMS dictionary
    Set LINEOUT_SAMPLE to fit to a specific lineout, or None to fit to the average
    Run with any python interpreter.

The script will:
1. Load experimental lineout data
2. Fit the density function parameters using least squares optimization
3. Display the fitted parameters and correlation metrics
4. Plot the comparison between experimental data and fitted profile
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from inversion_fbpic.density_profiles.downramp_injection import (
    build_generalized_gaussian_with_downramp,
)

# Configuration constants
LINEOUT_FILE = "./plasma_lineouts_data.npy"  # Path to lineout data
LINEOUT_SAMPLE: Optional[int] = 5  # Set to None to fit to average of all lineouts
DATA_RANGE = 660 * 10.1e-3  # HASO's 660 wide pixel image at 10.1 um/pixel

# Initial parameter guesses (from the original script)
INITIAL_PARAMS = {
    "gauss_peak": 1.0,
    "gauss_alpha": 1.6e-3,
    "gauss_beta": 1.2,
    "gauss_z0": 3.0e-3,
    "ramp_z0": 2.55e-3,
    "ramp_left_tau": 1.50e-3,
    "ramp_right_width": 0.14e-3,
    "ramp_height": 1.5,
}

# Parameter bounds for optimization (to prevent unrealistic values)
PARAM_BOUNDS = {
    "gauss_peak": (0.1, 10.0),
    "gauss_alpha": (0.1e-3, 10.0e-3),
    "gauss_beta": (0.1, 5.0),
    "gauss_z0": (0.0, 10.0e-3),
    "ramp_z0": (0.0, 10.0e-3),
    "ramp_left_tau": (0.1e-3, 5.0e-3),
    "ramp_right_width": (0.01e-3, 1.0e-3),
    "ramp_height": (0.1, 5.0),
}

# Parameter order for optimization
PARAM_ORDER = [
    "gauss_peak", "gauss_alpha", "gauss_beta", "gauss_z0",
    "ramp_z0", "ramp_left_tau", "ramp_right_width", "ramp_height"
]


def load_lineout_data(file_path: str = LINEOUT_FILE) -> Optional[np.ndarray]:
    """Load lineout data from numpy file.

    Args:
        file_path: Path to the numpy file containing lineout data.

    Returns:
        Array of lineouts where each row is a lineout, or None if file not found.
    """
    if not Path(file_path).exists():
        print(f"Error: File {file_path} not found!")
        return None

    lineouts = np.load(file_path)
    print(f"Loaded lineout data: {lineouts.shape}")
    print(f"Number of lineouts: {lineouts.shape[0]}")
    print(f"Points per lineout: {lineouts.shape[1]}")

    return lineouts


def _evaluate_density_function(params: np.ndarray, z_positions: np.ndarray) -> np.ndarray:
    """Evaluate the density function with given parameters.

    Args:
        params: Array of parameters in the order defined by PARAM_ORDER.
        z_positions: Z positions to evaluate the function at.

    Returns:
        Density values at the given z positions.
    """
    # Unpack parameters
    (gauss_peak, gauss_alpha, gauss_beta, gauss_z0, ramp_z0, 
     ramp_left_tau, ramp_right_width, ramp_height) = params

    # Create the density function
    density_func = build_generalized_gaussian_with_downramp(
        gauss_peak=gauss_peak,
        gauss_alpha=gauss_alpha,
        gauss_beta=gauss_beta,
        gauss_z0=gauss_z0,
        ramp_z0=ramp_z0,
        ramp_left_tau=ramp_left_tau,
        ramp_right_width=ramp_right_width,
        ramp_height=ramp_height,
    )

    # Evaluate at r=0 (along the axis)
    r_positions = np.zeros_like(z_positions)
    return density_func(z_positions, r_positions)


def _calculate_objective(params: np.ndarray, z_positions: np.ndarray, 
                        target_data: np.ndarray) -> float:
    """Calculate the sum of squared residuals for optimization.

    Args:
        params: Array of parameters.
        z_positions: Z positions.
        target_data: Target experimental data.

    Returns:
        Sum of squared residuals.
    """
    # Get density values from the function
    density_values = _evaluate_density_function(params, z_positions)

    # Scale the density to match the target data range
    if target_data.max() > 0:
        scale_factor = target_data.max() / density_values.max()
        density_scaled = density_values * scale_factor
    else:
        density_scaled = density_values

    # Calculate sum of squared residuals
    residuals = target_data - density_scaled
    return np.sum(residuals**2)


def _prepare_target_data(lineouts: np.ndarray) -> np.ndarray:
    """Prepare target data for fitting.

    Args:
        lineouts: Array of experimental lineouts.

    Returns:
        Target data array (single lineout or average).
    """
    if LINEOUT_SAMPLE is not None:
        target_data = lineouts[LINEOUT_SAMPLE, :]
        print(f"Fitting to lineout {LINEOUT_SAMPLE + 1}")
    else:
        target_data = np.mean(lineouts, axis=0)
        print("Fitting to average of all lineouts")
    
    return target_data


def _prepare_optimization_setup() -> Tuple[np.ndarray, list]:
    """Prepare initial parameters and bounds for optimization.

    Returns:
        Tuple of (initial_params, bounds).
    """
    initial_params = np.array([INITIAL_PARAMS[key] for key in PARAM_ORDER])
    bounds = [PARAM_BOUNDS[key] for key in PARAM_ORDER]
    return initial_params, bounds


def _print_optimization_results(result: Any, fitted_params: Dict[str, float]) -> None:
    """Print optimization results and fitted parameters.

    Args:
        result: Optimization result object.
        fitted_params: Dictionary of fitted parameters.
    """
    if result.success:
        print("Optimization successful!")
        print(f"Final objective function value: {result.fun:.6f}")
        print(f"Number of iterations: {result.nit}")
    else:
        print("Optimization failed!")
        print(f"Reason: {result.message}")

    print(f"\nFitted parameters:")
    for key, value in fitted_params.items():
        print(f"  {key}: {value:.6f}")


def fit_density_profile(lineouts: np.ndarray) -> Dict[str, Any]:
    """Fit the density profile parameters to the experimental data.

    Args:
        lineouts: Array of experimental lineouts.

    Returns:
        Dictionary containing fitted parameters and optimization results.
    """
    # Prepare target data
    target_data = _prepare_target_data(lineouts)

    # Create z positions (convert from mm to meters)
    z_positions = np.linspace(0, DATA_RANGE / 1e3, len(target_data))

    # Prepare optimization setup
    initial_params, bounds = _prepare_optimization_setup()

    print(f"Initial parameters:")
    for i, key in enumerate(PARAM_ORDER):
        print(f"  {key}: {initial_params[i]:.6f}")

    print("\nStarting optimization...")

    # Perform optimization
    result = minimize(
        _calculate_objective,
        initial_params,
        args=(z_positions, target_data),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'disp': True}
    )

    # Extract fitted parameters
    fitted_params = {key: result.x[i] for i, key in enumerate(PARAM_ORDER)}

    # Print results
    _print_optimization_results(result, fitted_params)

    return {
        "fitted_params": fitted_params,
        "optimization_result": result,
        "target_data": target_data,
        "z_positions": z_positions,
    }


def _scale_density_to_target(density_values: np.ndarray, target_data: np.ndarray) -> np.ndarray:
    """Scale density values to match target data range.

    Args:
        density_values: Raw density values.
        target_data: Target experimental data.

    Returns:
        Scaled density values.
    """
    if target_data.max() > 0:
        scale_factor = target_data.max() / density_values.max()
        return density_values * scale_factor
    return density_values


def _calculate_correlation_metrics(target_data: np.ndarray, 
                                  fitted_density: np.ndarray) -> Dict[str, float]:
    """Calculate correlation and fit quality metrics.

    Args:
        target_data: Target experimental data.
        fitted_density: Fitted density values.

    Returns:
        Dictionary containing correlation, R-squared, and RMSE.
    """
    # Calculate correlation
    correlation = np.corrcoef(target_data, fitted_density)[0, 1]

    # Calculate R-squared
    ss_res = np.sum((target_data - fitted_density) ** 2)
    ss_tot = np.sum((target_data - np.mean(target_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((target_data - fitted_density) ** 2))

    return {
        "correlation": correlation,
        "r_squared": r_squared,
        "rmse": rmse,
    }


def calculate_fit_metrics(fitted_params: Dict[str, float], target_data: np.ndarray, 
                         z_positions: np.ndarray) -> Dict[str, Any]:
    """Calculate correlation and fit quality metrics.

    Args:
        fitted_params: Fitted parameters.
        target_data: Target experimental data.
        z_positions: Z positions.

    Returns:
        Dictionary containing fit metrics and scaled density values.
    """
    # Get fitted density values
    param_array = np.array([fitted_params[key] for key in PARAM_ORDER])
    fitted_density = _evaluate_density_function(param_array, z_positions)

    # Scale to match target data
    fitted_density_scaled = _scale_density_to_target(fitted_density, target_data)

    # Calculate metrics
    metrics = _calculate_correlation_metrics(target_data, fitted_density_scaled)
    metrics["fitted_density"] = fitted_density_scaled

    return metrics


def _create_comparison_plot(ax: plt.Axes, target_data: np.ndarray, 
                          fitted_density: np.ndarray, z_positions: np.ndarray,
                          metrics: Dict[str, float]) -> None:
    """Create the main comparison plot.

    Args:
        ax: Matplotlib axes object.
        target_data: Target experimental data.
        fitted_density: Fitted density values.
        z_positions: Z positions.
        metrics: Fit quality metrics.
    """
    # Convert z positions to mm for plotting
    z_mm = z_positions * 1000
    x_lineouts = np.linspace(0, DATA_RANGE, len(target_data))

    # Plot comparison
    ax.plot(x_lineouts, target_data, 'b-', linewidth=2, label='Experimental Data', alpha=0.8)
    ax.plot(z_mm, fitted_density, 'r-', linewidth=2, label='Fitted Profile', alpha=0.8)
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Intensity / Density')
    ax.set_title('Experimental Data vs Fitted Density Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, DATA_RANGE)

    # Add metrics text
    metrics_text = (f'Correlation: {metrics["correlation"]:.3f}\n'
                   f'R²: {metrics["r_squared"]:.3f}\n'
                   f'RMSE: {metrics["rmse"]:.3f}')
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def _create_residuals_plot(ax: plt.Axes, target_data: np.ndarray, 
                         fitted_density: np.ndarray) -> None:
    """Create the residuals plot.

    Args:
        ax: Matplotlib axes object.
        target_data: Target experimental data.
        fitted_density: Fitted density values.
    """
    x_lineouts = np.linspace(0, DATA_RANGE, len(target_data))
    residuals = target_data - fitted_density
    
    ax.plot(x_lineouts, residuals, 'g-', linewidth=1, alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals (Experimental - Fitted)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, DATA_RANGE)


def plot_fit_results(target_data: np.ndarray, fitted_density: np.ndarray,
                    z_positions: np.ndarray, metrics: Dict[str, float]) -> None:
    """Plot the comparison between experimental data and fitted profile.

    Args:
        target_data: Target experimental data.
        fitted_density: Fitted density values.
        z_positions: Z positions.
        metrics: Fit quality metrics.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    _create_comparison_plot(ax1, target_data, fitted_density, z_positions, metrics)
    _create_residuals_plot(ax2, target_data, fitted_density)

    plt.tight_layout()
    plt.show()


def _print_fit_metrics(metrics: Dict[str, Any]) -> None:
    """Print fit quality metrics.

    Args:
        metrics: Dictionary containing fit metrics.
    """
    print(f"\nFit Quality Metrics:")
    print(f"  Correlation: {metrics['correlation']:.3f}")
    print(f"  R-squared: {metrics['r_squared']:.3f}")
    print(f"  RMSE: {metrics['rmse']:.3f}")


def _print_fitted_parameters(fitted_params: Dict[str, float]) -> None:
    """Print fitted parameters in a format suitable for copying to the original script.

    Args:
        fitted_params: Dictionary of fitted parameters.
    """
    print("\nFitting complete!")
    print("\nTo use these fitted parameters in your original script, update the CASE==1 parameters:")
    for key, value in fitted_params.items():
        print(f"    {key}={value:.6f},")


def main() -> None:
    """Script entry point to fit density profile parameters to experimental data."""
    print("Loading experimental lineout data...")
    lineouts = load_lineout_data()

    if lineouts is None:
        print("Failed to load lineout data. Exiting.")
        return

    print("\nFitting density profile parameters...")
    fit_results = fit_density_profile(lineouts)

    print("\nCalculating fit metrics...")
    metrics = calculate_fit_metrics(
        fit_results["fitted_params"],
        fit_results["target_data"],
        fit_results["z_positions"]
    )

    _print_fit_metrics(metrics)

    print("\nCreating fit comparison plot...")
    plot_fit_results(
        fit_results["target_data"],
        metrics["fitted_density"],
        fit_results["z_positions"],
        metrics
    )

    _print_fitted_parameters(fit_results["fitted_params"])


if __name__ == "__main__":
    main()
