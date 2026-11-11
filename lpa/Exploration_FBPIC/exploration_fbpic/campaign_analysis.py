"""Compact scalar-data diagnostics for completed exploration campaigns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from .campaign import Campaign
from .exploration import XoptFBPICExplorer
from .campaign_rows import is_no_beam, model_input_matrix, physical_outcome_rows
from .feasibility import BeamFeasibilityModel


@dataclass(frozen=True)
class CampaignAnalysis:
    """Summary written by :func:`analyze_campaign`."""

    report: dict[str, object]
    report_path: Path
    plot_path: Path


def plot_input_pca_projections(
    campaign: Campaign,
    stage: str,
    output_directory: str | Path,
    data_path: str | Path | None = None,
) -> tuple[Path, ...]:
    """Write higher-component input PCA projections for feasibility inspection."""
    explorer = XoptFBPICExplorer(campaign, stage)
    raw = pd.read_parquet(data_path) if data_path is not None else explorer.store.load()
    _require_campaign_summary(raw)
    training = explorer.training_data(raw).replace([np.inf, -np.inf], np.nan)
    parameter_names = list(explorer.variable_bounds)
    complete = np.isfinite(training.loc[:, parameter_names].to_numpy(dtype=float)).all(
        axis=1
    )
    if not complete.any():
        raise ValueError("No compatible successful rows are available for input PCA.")
    X_unit = _unit_scale(
        training.loc[complete, parameter_names].to_numpy(dtype=float), explorer
    )
    input_center = X_unit.mean(axis=0)
    _, _, input_vectors = np.linalg.svd(X_unit - input_center, full_matrices=False)
    all_inputs, beam, no_beam, charge = _physical_outcome_inputs(explorer, raw)
    all_projection = (all_inputs - input_center) @ input_vectors.T
    feasibility = BeamFeasibilityModel(explorer)
    feasibility.fit_campaign_data(raw)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for first, second in ((0, 2), (0, 3), (1, 2), (1, 3)):
        output = output_directory / f"input-pca-pc{first + 1}-pc{second + 1}.png"
        _plot_input_pca_pair(
            all_projection,
            beam,
            no_beam,
            charge,
            feasibility,
            all_inputs,
            first,
            second,
            output,
        )
        paths.append(output)
    return tuple(paths)


def plot_feasibility_slice(
    campaign: Campaign,
    stage: str,
    output: str | Path,
    data_path: str | Path | None = None,
    grid_size: int = 100,
) -> Path:
    """Plot observed outcomes, beam probability, and classifier entropy."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    import matplotlib.pyplot as plt

    explorer = XoptFBPICExplorer(campaign, stage)
    raw = pd.read_parquet(data_path) if data_path is not None else explorer.store.load()
    _require_campaign_summary(raw)
    raw = raw.loc[raw["campaign_fingerprint"] == campaign.config.fingerprint()].copy()
    feasibility = BeamFeasibilityModel(explorer)
    feasibility.fit_campaign_data(raw)
    if not feasibility.is_ready:
        raise ValueError("Need both beam and no-beam observations to fit feasibility.")
    x_name = "laser_focal_position_m"
    y_name = "peak_neutral_density_cm3"
    x_parameter = campaign.config.parameters[x_name]
    y_parameter = campaign.config.parameters[y_name]
    assert x_parameter.lower is not None and x_parameter.upper is not None
    assert y_parameter.lower is not None and y_parameter.upper is not None
    x_physical = np.linspace(x_parameter.lower, x_parameter.upper, grid_size)
    y_physical = np.linspace(y_parameter.lower, y_parameter.upper, grid_size)
    x_grid, y_grid = np.meshgrid(x_physical, y_physical)
    parameter_names = list(explorer.variable_bounds)
    nominal = {
        parameter.name: explorer._to_model_value(parameter, float(parameter.nominal))
        for parameter in explorer.parameters
    }
    points = np.tile(
        np.asarray([nominal[name] for name in parameter_names], dtype=float),
        (x_grid.size, 1),
    )
    points[:, parameter_names.index(x_name)] = [
        explorer._to_model_value(x_parameter, value) for value in x_grid.ravel()
    ]
    points[:, parameter_names.index(y_name)] = [
        explorer._to_model_value(y_parameter, value) for value in y_grid.ravel()
    ]
    probability = feasibility.beam_probability_matrix(points).reshape(x_grid.shape)
    entropy = -(
        probability * np.log2(np.clip(probability, 1e-12, 1))
        + (1 - probability) * np.log2(np.clip(1 - probability, 1e-12, 1))
    )
    beam = raw["status"].eq("ok").to_numpy()
    no_beam = np.asarray(
        [
            is_no_beam(str(status), str(message))
            for status, message in zip(raw["status"], raw["message"], strict=True)
        ],
        dtype=bool,
    )
    x_data = raw[f"p:{x_name}"].to_numpy(dtype=float) * 1e3
    y_data = raw[f"p:{y_name}"].to_numpy(dtype=float) / 1e18
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    axes[0].scatter(x_data[beam], y_data[beam], c="tab:blue", s=34, label="Beam")
    axes[0].scatter(
        x_data[no_beam],
        y_data[no_beam],
        c="white",
        edgecolors="0.25",
        s=34,
        label="No beam",
    )
    axes[0].legend(loc="upper right")
    probability_image = axes[1].pcolormesh(
        x_grid * 1e3,
        y_grid / 1e18,
        probability,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    contours = axes[1].contour(
        x_grid * 1e3,
        y_grid / 1e18,
        probability,
        levels=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        colors="white",
        linewidths=0.9,
    )
    axes[1].clabel(contours, fmt=lambda value: f"{value:.0%}", fontsize=8)
    figure.colorbar(probability_image, ax=axes[1], label="P(beam)")
    entropy_image = axes[2].pcolormesh(
        x_grid * 1e3,
        y_grid / 1e18,
        entropy,
        shading="auto",
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    figure.colorbar(entropy_image, ax=axes[2], label="classification entropy (bits)")
    titles = (
        f"Cumulative physical outcomes through {stage.replace('_', ' ').title()}",
        "Beam probability",
        "Injection uncertainty",
    )
    for axis, title in zip(axes, titles, strict=True):
        axis.set(
            title=title,
            xlabel="laser focal position (mm)",
            ylabel="peak neutral density (1e18 cm^-3)",
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_feature_slice(
    campaign: Campaign,
    stage: str,
    output: str | Path,
    feature_name: str,
    data_path: str | Path | None = None,
    grid_size: int = 100,
) -> Path:
    """Plot observations and a feature GP on a two-parameter slice.

    Every active coordinate other than focal position and plasma density is held
    at its campaign nominal value. White observation markers denote entries whose
    selected feature is unavailable, which are absent from GP training.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    import matplotlib.pyplot as plt
    import torch

    explorer = XoptFBPICExplorer(campaign, stage)
    raw = pd.read_parquet(data_path) if data_path is not None else explorer.store.load()
    _require_campaign_summary(raw)
    raw = raw.loc[raw["campaign_fingerprint"] == campaign.config.fingerprint()].copy()
    x_name = "laser_focal_position_m"
    y_name = "peak_neutral_density_cm3"
    if feature_name not in explorer.feature_names:
        raise ValueError(
            f"{feature_name!r} is not an exploration feature; choose from "
            f"{', '.join(explorer.feature_names)}"
        )
    required = {f"p:{x_name}", f"p:{y_name}", f"f:{feature_name}"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(
            f"Campaign summary is missing required columns: {', '.join(missing)}"
        )

    training = explorer.training_data(raw)
    if training.empty:
        raise ValueError("No successful records are available to train the feature GP.")
    xopt = explorer.make_xopt(raw)
    generator = cast(Any, xopt.generator)
    model = generator.train_model(update_internal=False)
    parameter_names = list(explorer.variable_bounds)
    feature_index = list(explorer.feature_names).index(feature_name)
    x_parameter = campaign.config.parameters[x_name]
    y_parameter = campaign.config.parameters[y_name]
    assert x_parameter.lower is not None and x_parameter.upper is not None
    assert y_parameter.lower is not None and y_parameter.upper is not None
    x_physical = np.linspace(x_parameter.lower, x_parameter.upper, grid_size)
    y_physical = np.linspace(y_parameter.lower, y_parameter.upper, grid_size)
    x_grid, y_grid = np.meshgrid(x_physical, y_physical)
    nominal = {
        parameter.name: explorer._to_model_value(parameter, float(parameter.nominal))
        for parameter in explorer.parameters
    }
    points = np.tile(
        np.asarray([nominal[name] for name in parameter_names], dtype=float),
        (x_grid.size, 1),
    )
    points[:, parameter_names.index(x_name)] = [
        explorer._to_model_value(x_parameter, value) for value in x_grid.ravel()
    ]
    points[:, parameter_names.index(y_name)] = [
        explorer._to_model_value(y_parameter, value) for value in y_grid.ravel()
    ]
    model_inputs = torch.tensor(points, **generator.tkwargs)
    posterior = model.posterior(model_inputs)
    mean = posterior.mean.detach().cpu().numpy()[:, feature_index].reshape(x_grid.shape)
    stddev = (
        posterior.variance.sqrt()
        .detach()
        .cpu()
        .numpy()[:, feature_index]
        .reshape(x_grid.shape)
    )

    x_data = raw[f"p:{x_name}"].to_numpy(dtype=float) * 1e3
    y_data = raw[f"p:{y_name}"].to_numpy(dtype=float) / 1e18
    feature_data = raw[f"f:{feature_name}"].to_numpy(dtype=float)
    valid_feature = (raw["status"] == "ok").to_numpy() & np.isfinite(feature_data)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    scatter = axes[0].scatter(
        x_data[valid_feature],
        y_data[valid_feature],
        c=feature_data[valid_feature],
        cmap="viridis",
        s=34,
        edgecolors="black",
        linewidths=0.25,
    )
    axes[0].scatter(
        x_data[~valid_feature],
        y_data[~valid_feature],
        c="white",
        s=34,
        edgecolors="0.25",
        linewidths=0.65,
        label="No feature diagnostic",
    )
    axes[0].legend(loc="upper right")
    figure.colorbar(scatter, ax=axes[0], label=feature_name)
    mean_image = axes[1].pcolormesh(
        x_grid * 1e3, y_grid / 1e18, mean, shading="auto", cmap="viridis"
    )
    figure.colorbar(mean_image, ax=axes[1], label=f"GP mean {feature_name}")
    stddev_image = axes[2].pcolormesh(
        x_grid * 1e3, y_grid / 1e18, stddev, shading="auto", cmap="magma"
    )
    figure.colorbar(
        stddev_image, ax=axes[2], label=f"GP posterior std. dev. {feature_name}"
    )
    titles = (
        f"Cumulative observations through {stage.replace('_', ' ').title()}",
        f"{feature_name} GP mean at nominal slice",
        f"{feature_name} GP uncertainty at nominal slice",
    )
    for axis, title in zip(axes, titles, strict=True):
        axis.set(
            title=title,
            xlabel="laser focal position (mm)",
            ylabel="peak neutral density (1e18 cm^-3)",
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_charge_slice(
    campaign: Campaign,
    stage: str,
    output: str | Path,
    data_path: str | Path | None = None,
    grid_size: int = 100,
) -> Path:
    """Backward-compatible charge specialization of :func:`plot_feature_slice`."""
    return plot_feature_slice(
        campaign,
        stage,
        output,
        "total_beam_charge_pc",
        data_path,
        grid_size,
    )


def analyze_campaign(
    campaign: Campaign,
    stage: str,
    output_directory: str | Path,
    folds: int = 5,
    seed: int = 0,
    data_path: str | Path | None = None,
    vector_sim: int | None = None,
    input_pca_x: int = 1,
    input_pca_y: int = 2,
) -> CampaignAnalysis:
    """Assess compact scalar results without reading raw FBPIC diagnostics.

    The validation surrogate is an RBF interpolator over the same scaled inputs
    as the campaign GP. It intentionally measures predictive behavior on held-out
    simulations without requiring Xopt model checkpoint serialization.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")

    explorer = XoptFBPICExplorer(campaign, stage)
    raw = pd.read_parquet(data_path) if data_path is not None else explorer.store.load()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    _require_campaign_summary(raw)

    compatible = raw.loc[
        raw["campaign_fingerprint"] == campaign.config.fingerprint()
    ].copy()
    successful = compatible.loc[compatible["status"] == "ok"].copy()
    training = explorer.training_data(raw)
    parameter_names = list(explorer.variable_bounds)
    feature_names = list(explorer.feature_names)

    report: dict[str, object] = {
        "stage": stage,
        "campaign_fingerprint": campaign.config.fingerprint(),
        "total_records": int(len(raw)),
        "compatible_records": int(len(compatible)),
        "successful_records": int(len(successful)),
        "failed_or_incomplete_records": int(len(compatible) - len(successful)),
        "active_parameters": parameter_names,
        "exploration_features": feature_names,
    }
    if data_path is not None:
        report["data_path"] = str(Path(data_path).resolve())
    if training.empty:
        report["error"] = "No compatible successful rows with exploration features."
        report_path = output_directory / "campaign-analysis.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return CampaignAnalysis(
            report, report_path, output_directory / "campaign-projection.png"
        )

    finite = training.replace([np.inf, -np.inf], np.nan)
    valid_feature_names = [name for name in feature_names if finite[name].notna().any()]
    X = finite.loc[:, parameter_names].to_numpy(dtype=float)
    y = finite.loc[:, valid_feature_names].to_numpy(dtype=float)
    complete = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    X = X[complete]
    y = y[complete]
    report["complete_training_rows"] = int(len(X))
    report["dropped_rows_with_nonfinite_values"] = int(len(training) - len(X))
    report["feature_finite_counts"] = {
        name: int(finite[name].notna().sum()) for name in feature_names
    }

    X_unit = _unit_scale(X, explorer)
    report["space_coverage"] = _coverage(X_unit, seed)
    model_data = finite.loc[complete, [*parameter_names, *valid_feature_names]]
    report["cross_validation"] = _cross_validate(
        explorer, raw, model_data, valid_feature_names, folds, seed
    )
    report["sample_size_guidance"] = _sample_size_guidance(
        report["cross_validation"], len(X)
    )

    plot_path = output_directory / "campaign-projection.png"
    _plot_projection(
        explorer,
        raw,
        X_unit,
        y,
        valid_feature_names,
        plot_path,
        vector_sim,
        input_pca_x,
        input_pca_y,
    )
    report_path = output_directory / "campaign-analysis.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return CampaignAnalysis(report, report_path, plot_path)


def _require_campaign_summary(data: pd.DataFrame) -> None:
    required_columns = {"campaign_fingerprint", "status"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(
            "Campaign summary is missing required columns: "
            f"{', '.join(missing_columns)}. Pass campaign_runs.parquet created by this campaign."
        )


def _unit_scale(X: np.ndarray, explorer: XoptFBPICExplorer) -> np.ndarray:
    bounds = np.asarray(list(explorer.variable_bounds.values()), dtype=float)
    return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def _coverage(X_unit: np.ndarray, seed: int) -> dict[str, float | int]:
    if len(X_unit) < 2:
        return {"dimensions": int(X_unit.shape[1]), "points": int(len(X_unit))}
    distances = cdist(X_unit, X_unit)
    np.fill_diagonal(distances, np.inf)
    sobol = getattr(qmc, "Sobol")(d=X_unit.shape[1], scramble=True, seed=seed)
    reference = sobol.random(4096)
    reference_distances = cdist(reference, X_unit).min(axis=1)
    nearest = distances.min(axis=1)
    return {
        "dimensions": int(X_unit.shape[1]),
        "points": int(len(X_unit)),
        "nearest_neighbor_median": float(np.median(nearest)),
        "nearest_neighbor_minimum": float(np.min(nearest)),
        "reference_fill_distance_median": float(np.median(reference_distances)),
        "reference_fill_distance_p95": float(np.quantile(reference_distances, 0.95)),
        "reference_fill_distance_maximum": float(np.max(reference_distances)),
    }


def _cross_validate(
    explorer: XoptFBPICExplorer,
    raw: pd.DataFrame,
    data: pd.DataFrame,
    feature_names: list[str],
    folds: int,
    seed: int,
) -> dict[str, object]:
    if len(data) < folds + 2:
        return {
            "error": f"Need at least {folds + 2} complete rows for {folds}-fold validation."
        }
    import torch

    generator = np.random.default_rng(seed)
    fold_ids = generator.permutation(len(data)) % folds
    y = data.loc[:, feature_names].to_numpy(dtype=float)
    predictions = np.full_like(y, np.nan)
    xopt = explorer.make_xopt(raw)
    xopt_generator = cast(Any, xopt.generator)
    for fold in range(folds):
        train = fold_ids != fold
        test = ~train
        try:
            model = xopt_generator.train_model(
                data=data.iloc[train], update_internal=False
            )
            held_out_inputs = torch.tensor(
                data.iloc[test]
                .loc[:, list(explorer.variable_bounds)]
                .to_numpy(dtype=float),
                **xopt_generator.tkwargs,
            )
            predictions[test] = (
                model.posterior(held_out_inputs).mean.detach().cpu().numpy()
            )
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as error:
            return {"error": f"Xopt GP validation failed: {error}"}
    residual = predictions - y
    feature_metrics: dict[str, dict[str, float | None]] = {}
    result: dict[str, object] = {"folds": folds, "features": feature_metrics}
    for index, name in enumerate(feature_names):
        scale = float(np.std(y[:, index]))
        rmse = float(np.sqrt(np.mean(residual[:, index] ** 2)))
        feature_metrics[name] = {
            "rmse": rmse,
            "nrmse_by_observed_std": rmse / scale if scale else None,
            "r2": _r2(y[:, index], predictions[:, index]),
        }
    return result


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    if total == 0:
        return None
    return float(1 - np.sum((observed - predicted) ** 2) / total)


def _sample_size_guidance(validation: object, complete_rows: int) -> dict[str, object]:
    if not isinstance(validation, dict) or "features" not in validation:
        return {"status": "unavailable"}
    nrmse = [
        metrics["nrmse_by_observed_std"]
        for metrics in validation["features"].values()
        if metrics["nrmse_by_observed_std"] is not None
    ]
    if not nrmse:
        return {"status": "unavailable"}
    worst = float(max(nrmse))
    return {
        "status": "heuristic",
        "worst_feature_nrmse_by_observed_std": worst,
        "interpretation": (
            "Below 0.3 is a useful exploratory surrogate; 0.3 to 0.5 needs more data; "
            "above 0.5 is not yet predictive away from sampled points."
        ),
        "estimated_total_rows_for_0_3_nrmse": int(
            np.ceil(complete_rows * max(1.0, (worst / 0.3) ** 2))
        ),
        "note": "This is a rough learning-curve extrapolation, not an allocation estimate.",
    }


def _plot_projection(
    explorer: XoptFBPICExplorer,
    raw: pd.DataFrame,
    X_unit: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output: Path,
    vector_sim: int | None = None,
    input_pca_x: int = 1,
    input_pca_y: int = 2,
) -> None:
    import matplotlib.pyplot as plt

    input_center = X_unit.mean(axis=0)
    _, _, input_vectors = np.linalg.svd(X_unit - input_center, full_matrices=False)
    input_components = (input_pca_x - 1, input_pca_y - 1)
    if any(
        component < 0 or component >= len(input_vectors)
        for component in input_components
    ):
        raise ValueError(
            f"input PCA axes must be in [1, {len(input_vectors)}], got "
            f"{input_pca_x} and {input_pca_y}"
        )
    if input_components[0] == input_components[1]:
        raise ValueError("input PCA axes must select different components")
    all_inputs, beam, no_beam, charge = _physical_outcome_inputs(explorer, raw)
    all_projection = (all_inputs - input_center) @ input_vectors.T
    feasibility = BeamFeasibilityModel(explorer)
    feasibility.fit_campaign_data(raw)
    standardized_y = (y - y.mean(axis=0)) / np.where(
        y.std(axis=0) == 0, 1, y.std(axis=0)
    )
    response_center = standardized_y.mean(axis=0)
    _, _, response_vectors = np.linalg.svd(
        standardized_y - response_center, full_matrices=False
    )
    response_projection = (standardized_y - response_center) @ response_vectors[:2].T
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _plot_input_pca_pair(
        all_projection,
        beam,
        no_beam,
        charge,
        feasibility,
        all_inputs,
        *input_components,
        axis=axes[0],
    )
    colors = y[:, 0] if len(feature_names) else np.arange(len(y))
    scatter = axes[1].scatter(
        response_projection[:, 0],
        response_projection[:, 1],
        c=colors,
        cmap="plasma",
        vmin=0,
        vmax=1000,
        s=28,
    )
    axes[1].set(title="Response-space PCA", xlabel="PC 1", ylabel="PC 2")
    colorbar = figure.colorbar(
        scatter,
        ax=axes[1],
        label=feature_names[0] if feature_names else "row",
        ticks=[0, 500, 1000],
    )
    if feature_names and feature_names[0] == "total_beam_charge_pc":
        colorbar.ax.set_yticklabels(["0", "500", "1000+"])
    if vector_sim is not None:
        _plot_vector_simulations(
            axes,
            raw,
            explorer,
            feature_names,
            input_center,
            input_vectors,
            y.mean(axis=0),
            y.std(axis=0),
            response_center,
            response_vectors,
            vector_sim,
            input_components,
        )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_vector_simulations(
    axes: Any,
    raw: pd.DataFrame,
    explorer: XoptFBPICExplorer,
    feature_names: list[str],
    input_center: np.ndarray,
    input_vectors: np.ndarray,
    response_mean: np.ndarray,
    response_std: np.ndarray,
    response_center: np.ndarray,
    response_vectors: np.ndarray,
    vector_sim: int,
    input_components: tuple[int, int],
) -> None:
    """Mark the nominal and selected simulations in input and response PCA space."""
    if vector_sim < 0:
        raise ValueError("vector_sim must be nonnegative")
    parameter_names = list(explorer.variable_bounds)
    run_ids = ["sim_000000", f"sim_{vector_sim:06d}"]
    rows = raw.loc[
        (raw["campaign_fingerprint"] == explorer.campaign.config.fingerprint())
        & raw["run_id"].isin(run_ids)
    ].copy()
    if len(rows) != len(run_ids):
        found = set(rows["run_id"])
        missing = ", ".join(run_id for run_id in run_ids if run_id not in found)
        raise ValueError(f"vector simulation record not found: {missing}")
    rows = rows.set_index("run_id").loc[run_ids].reset_index()
    if not rows["status"].eq("ok").all():
        invalid = ", ".join(rows.loc[rows["status"] != "ok", "run_id"])
        raise ValueError(f"vector simulation is not beam-present: {invalid}")
    input_values = rows.loc[:, [f"p:{name}" for name in parameter_names]].to_numpy(
        dtype=float
    )
    model_inputs = np.asarray(
        [
            [
                explorer._to_model_value(parameter, value)
                for parameter, value in zip(explorer.parameters, row, strict=True)
            ]
            for row in input_values
        ],
        dtype=float,
    )
    input_projection = (
        _unit_scale(model_inputs, explorer) - input_center
    ) @ input_vectors[list(input_components)].T
    output_values = rows.loc[:, [f"f:{name}" for name in feature_names]].to_numpy(
        dtype=float
    )
    if not np.isfinite(output_values).all():
        invalid = ", ".join(rows.loc[~np.isfinite(output_values).all(axis=1), "run_id"])
        raise ValueError(f"vector simulation has incomplete features: {invalid}")
    standardized_outputs = (output_values - response_mean) / np.where(
        response_std == 0, 1, response_std
    )
    response_projection = (standardized_outputs - response_center) @ response_vectors[
        :2
    ].T
    for index, run_id in enumerate(run_ids):
        label = f"Sim {int(run_id.removeprefix('sim_')):04d}"
        for axis, coordinates in zip(
            axes, (input_projection[index], response_projection[index]), strict=True
        ):
            axis.scatter(
                coordinates[0],
                coordinates[1],
                marker="*",
                c="black",
                s=150,
                zorder=10,
            )
            axis.annotate(
                label,
                coordinates,
                xytext=(6, 6),
                textcoords="offset points",
                color="black",
                fontsize=8,
                zorder=11,
            )


def _physical_outcome_inputs(
    explorer: XoptFBPICExplorer, raw: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return full active-coordinate inputs and physical beam/no-beam masks."""
    outcomes, beam, no_beam = physical_outcome_rows(explorer, raw)
    model_inputs = model_input_matrix(explorer, outcomes)
    charge = outcomes.get(
        "f:total_beam_charge_pc", pd.Series(np.nan, index=outcomes.index)
    ).to_numpy(dtype=float)
    bounds = np.asarray(list(explorer.variable_bounds.values()), dtype=float)
    unit_inputs = (model_inputs - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return (
        unit_inputs,
        beam,
        no_beam,
        charge,
    )


def _plot_input_pca_pair(
    projection: np.ndarray,
    beam: np.ndarray,
    no_beam: np.ndarray,
    charge: np.ndarray,
    feasibility: BeamFeasibilityModel,
    unit_inputs: np.ndarray,
    first: int,
    second: int,
    output: Path | None = None,
    axis: Any | None = None,
) -> None:
    """Plot a pair of input PCs with observed-outcome feasibility annotations."""
    import matplotlib.pyplot as plt

    if axis is None:
        figure, axis = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    else:
        figure = axis.figure
    scatter = axis.scatter(
        projection[beam, first],
        projection[beam, second],
        c=charge[beam],
        cmap="winter",
        vmin=0,
        vmax=1000,
        s=28,
        edgecolors="none",
        label="Beam",
    )
    if feasibility.is_ready and beam.any():
        beam_probability = feasibility.beam_probability_matrix(
            _model_coordinates_from_unit(feasibility, unit_inputs[beam])
        )
        low_feasibility_beam = beam_probability < 0.30
        if low_feasibility_beam.any():
            axis.scatter(
                projection[beam][low_feasibility_beam, first],
                projection[beam][low_feasibility_beam, second],
                marker="x",
                c="limegreen",
                s=62,
                linewidths=1.35,
                label="Beam, P(beam) < 30%",
            )
    if no_beam.any():
        axis.scatter(
            projection[no_beam, first],
            projection[no_beam, second],
            c="white",
            edgecolors="0.2",
            linewidths=0.65,
            s=28,
            label="No beam",
        )
        if feasibility.is_ready:
            probability = feasibility.beam_probability_matrix(
                _model_coordinates_from_unit(feasibility, unit_inputs[no_beam])
            )
            for lower, upper, color, label in (
                (0.20, 0.30, "hotpink", "20% <= P(beam) < 30%"),
                (0.10, 0.20, "darkorange", "10% <= P(beam) < 20%"),
                (0.05, 0.10, "red", "5% <= P(beam) < 10%"),
                (0.00, 0.05, "maroon", "P(beam) < 5%"),
            ):
                band = (probability >= lower) & (probability < upper)
                if band.any():
                    axis.scatter(
                        projection[no_beam][band, first],
                        projection[no_beam][band, second],
                        marker="x",
                        c=color,
                        s=62,
                        linewidths=1.35,
                        label=f"No beam, {label}",
                    )
    axis.legend(loc="best", fontsize=8)
    axis.set(
        title=f"Input-space PCA: PC {first + 1} vs PC {second + 1}",
        xlabel=f"PC {first + 1}",
        ylabel=f"PC {second + 1}",
    )
    colorbar = figure.colorbar(
        scatter,
        ax=axis,
        label="total beam charge (pC)",
        ticks=[0, 500, 1000],
    )
    colorbar.ax.set_yticklabels(["0", "500", "1000+"])
    if output is not None:
        figure.savefig(output, dpi=180)
        plt.close(figure)


def _model_coordinates_from_unit(
    feasibility: BeamFeasibilityModel, unit_inputs: np.ndarray
) -> np.ndarray:
    bounds = np.asarray(
        list(feasibility.explorer.variable_bounds.values()), dtype=float
    )
    return bounds[:, 0] + unit_inputs * (bounds[:, 1] - bounds[:, 0])


def _pca(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vectors[:2].T
    if projection.shape[1] == 1:
        projection = np.column_stack((projection, np.zeros(len(projection))))
    return projection
