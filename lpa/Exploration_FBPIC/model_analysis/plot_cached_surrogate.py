"""Plot one-dimensional and two-dimensional scans from a cached scalar GP.

The cached model is never retrained. Reference parameters are loaded from a
simulation ``run_manifest.json`` and converted to the model coordinates recorded
in the cache before the requested axes are scanned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from exploration_fbpic.campaign import Campaign
from exploration_fbpic.exploration import XoptFBPICExplorer
from exploration_fbpic.surrogate_cache import CachedSurrogate, load_surrogate
from scipy.special import ndtr

DISPLAY_UNITS = {
    "laser_focal_position_m": (1.0e3, "Laser focal position (mm)"),
    "peak_neutral_density_cm3": (1.0e-18, "Peak neutral density (1e18 cm^-3)"),
    "laser_energy_J": (1.0, "Laser energy (J)"),
}


def parse_args() -> argparse.Namespace:
    """Parse cached-surrogate plotting arguments."""
    parser = argparse.ArgumentParser(
        description="Plot a cached one-output GP around a simulation reference point."
    )
    parser.add_argument("config", type=Path, help="campaign YAML configuration")
    parser.add_argument("--stage", default="stage_2")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--variable", required=True, help="cached scalar feature name")
    parser.add_argument(
        "--reference-run",
        type=Path,
        required=True,
        help="simulation directory or run_manifest.json used for fixed parameters",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--x-variable", default="laser_focal_position_m")
    parser.add_argument("--y-variable", default="peak_neutral_density_cm3")
    parser.add_argument("--scan-variable", default="laser_energy_J")
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--line-points", type=int, default=160)
    return parser.parse_args()


def load_reference_parameters(reference_run: Path) -> dict[str, float | str]:
    """Load full physical parameters from a run directory or manifest file."""
    manifest_path = (
        reference_run / "run_manifest.json" if reference_run.is_dir() else reference_run
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dict(payload["parameters"])


def physical_model_point(
    explorer: XoptFBPICExplorer, reference: dict[str, float | str]
) -> dict[str, float]:
    """Convert one full physical manifest vector to active model coordinates."""
    return {
        parameter.name: explorer._to_model_value(
            parameter, float(reference[parameter.name])
        )
        for parameter in explorer.parameters
    }


def physical_axis(explorer: XoptFBPICExplorer, variable: str, count: int) -> np.ndarray:
    """Return evenly spaced physical scan values for one active parameter."""
    if count < 2:
        raise ValueError("scan point count must be at least 2")
    try:
        parameter = explorer.campaign.config.parameters[variable]
    except KeyError as error:
        raise ValueError(f"Unknown campaign parameter {variable!r}") from error
    if variable not in explorer.variable_bounds:
        raise ValueError(f"{variable!r} is not active in the requested stage")
    assert parameter.lower is not None and parameter.upper is not None
    if parameter.log_scale:
        return np.geomspace(parameter.lower, parameter.upper, count)
    return np.linspace(parameter.lower, parameter.upper, count)


def predict(
    cached: CachedSurrogate,
    input_names: list[str],
    points: np.ndarray,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict mean and posterior standard deviation in bounded point batches."""
    if points.ndim != 2 or points.shape[1] != len(input_names):
        raise ValueError("points must have one column per cached model input")
    means, standard_deviations = [], []
    import gpytorch

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, len(points), chunk_size):
            inputs = torch.tensor(
                points[start : start + chunk_size], dtype=torch.float64
            )
            posterior = cached.model.posterior(inputs)
            means.append(posterior.mean.detach().cpu().numpy().reshape(-1))
            standard_deviations.append(
                posterior.variance.sqrt().detach().cpu().numpy().reshape(-1)
            )
    return np.concatenate(means), np.concatenate(standard_deviations)


def axis_label(variable: str) -> tuple[float, str]:
    """Return display scaling and a readable axis label for a parameter."""
    return DISPLAY_UNITS.get(variable, (1.0, variable.replace("_", " ")))


def displayed_prediction(
    cached: CachedSurrogate, mean: np.ndarray, standard_deviation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """Convert latent feasibility output to probability; leave scalars unchanged."""
    if cached.metadata.get("outcome_kind") != "probit_feasibility":
        return mean, standard_deviation, cached.variable
    probability = ndtr(mean / np.sqrt(1.0 + standard_deviation**2))
    lower = ndtr((mean - standard_deviation) / np.sqrt(2.0))
    upper = ndtr((mean + standard_deviation) / np.sqrt(2.0))
    return probability, (upper - lower) / 2.0, "Beam probability"


def plot_2d_scan(
    explorer: XoptFBPICExplorer,
    cached: CachedSurrogate,
    reference_point: dict[str, float],
    x_variable: str,
    y_variable: str,
    grid_size: int,
    output: Path,
) -> Path:
    """Write matched 2D posterior-mean and uncertainty contour panels."""
    x_physical = physical_axis(explorer, x_variable, grid_size)
    y_physical = physical_axis(explorer, y_variable, grid_size)
    x_grid, y_grid = np.meshgrid(x_physical, y_physical)
    names = list(cached.metadata["input_names"])
    points = np.asarray([[reference_point[name] for name in names]] * x_grid.size)
    x_parameter = explorer.campaign.config.parameters[x_variable]
    y_parameter = explorer.campaign.config.parameters[y_variable]
    points[:, names.index(x_variable)] = [
        explorer._to_model_value(x_parameter, value) for value in x_grid.ravel()
    ]
    points[:, names.index(y_variable)] = [
        explorer._to_model_value(y_parameter, value) for value in y_grid.ravel()
    ]
    mean, standard_deviation = predict(cached, names, points)
    mean = mean.reshape(x_grid.shape)
    standard_deviation = standard_deviation.reshape(x_grid.shape)
    mean, standard_deviation, output_label = displayed_prediction(
        cached, mean, standard_deviation
    )
    x_scale, x_label = axis_label(x_variable)
    y_scale, y_label = axis_label(y_variable)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    mean_kwargs = {"levels": 24, "cmap": "viridis"}
    if cached.metadata.get("outcome_kind") == "probit_feasibility":
        mean_kwargs.update({"vmin": 0.0, "vmax": 1.0})
    mean_plot = axes[0].contourf(
        x_grid * x_scale, y_grid * y_scale, mean, **mean_kwargs
    )
    uncertainty_plot = axes[1].contourf(
        x_grid * x_scale,
        y_grid * y_scale,
        standard_deviation,
        levels=24,
        cmap="magma",
    )
    figure.colorbar(mean_plot, ax=axes[0], label=f"GP mean {output_label}")
    figure.colorbar(
        uncertainty_plot,
        ax=axes[1],
        label=f"GP posterior uncertainty {output_label}",
    )
    for axis, title in zip(
        axes,
        ("Cached GP posterior mean", "Cached GP posterior uncertainty"),
        strict=True,
    ):
        axis.set(title=title, xlabel=x_label, ylabel=y_label)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_1d_scan(
    explorer: XoptFBPICExplorer,
    cached: CachedSurrogate,
    reference_point: dict[str, float],
    variable: str,
    count: int,
    output: Path,
) -> Path:
    """Write GP mean and one-standard-deviation band for one physical scan."""
    physical = physical_axis(explorer, variable, count)
    names = list(cached.metadata["input_names"])
    points = np.asarray([[reference_point[name] for name in names]] * len(physical))
    parameter = explorer.campaign.config.parameters[variable]
    points[:, names.index(variable)] = [
        explorer._to_model_value(parameter, value) for value in physical
    ]
    mean, standard_deviation = predict(cached, names, points)
    mean, standard_deviation, output_label = displayed_prediction(
        cached, mean, standard_deviation
    )
    scale, label = axis_label(variable)
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    axis.plot(physical * scale, mean, color="tab:blue", label="GP mean")
    axis.fill_between(
        physical * scale,
        mean - standard_deviation,
        mean + standard_deviation,
        color="tab:blue",
        alpha=0.22,
        label="GP posterior std. dev.",
    )
    axis.set(
        title=f"Cached GP scan: {output_label}",
        xlabel=label,
        ylabel=output_label,
    )
    if cached.metadata.get("outcome_kind") == "probit_feasibility":
        axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    """Generate general cached-GP scan plots around a reference simulation."""
    args = parse_args()
    explorer = XoptFBPICExplorer(Campaign.from_file(args.config), args.stage)
    cached = load_surrogate(explorer, args.cache_root, args.variable)
    reference = physical_model_point(
        explorer, load_reference_parameters(args.reference_run)
    )
    output_directory = args.output_directory
    two_dimensional = plot_2d_scan(
        explorer,
        cached,
        reference,
        args.x_variable,
        args.y_variable,
        args.grid_size,
        output_directory
        / f"{args.variable}_{args.x_variable}_vs_{args.y_variable}.png",
    )
    one_dimensional = plot_1d_scan(
        explorer,
        cached,
        reference,
        args.scan_variable,
        args.line_points,
        output_directory / f"{args.variable}_vs_{args.scan_variable}.png",
    )
    print(
        json.dumps(
            {
                "two_dimensional": str(two_dimensional),
                "one_dimensional": str(one_dimensional),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
