"""Rank one-dimensional cached-GP response curves through a reference run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from exploration_fbpic.campaign import Campaign
from exploration_fbpic.exploration import XoptFBPICExplorer
from exploration_fbpic.surrogate_cache import CachedSurrogate, load_surrogate

from model_analysis.plot_cached_surrogate import (
    axis_label,
    displayed_prediction,
    load_reference_parameters,
    physical_axis,
    physical_model_point,
    predict,
)


@dataclass(frozen=True)
class OneDimensionalResponse:
    """One cached-GP slice and its response magnitude diagnostics."""

    variable: str
    physical_axis: np.ndarray
    mean: np.ndarray
    uncertainty: np.ndarray
    response_range: float
    reference_uncertainty: float
    signal_to_uncertainty: float


def parse_args() -> argparse.Namespace:
    """Parse one-dimensional cached-GP sensitivity analysis arguments."""
    parser = argparse.ArgumentParser(
        description="Rank active parameters by their cached GP 1D response near a reference run."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", default="stage_2")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--variable", required=True, help="cached scalar or feasibility"
    )
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--points", type=int, default=60)
    return parser.parse_args()


def one_dimensional_response(
    explorer: XoptFBPICExplorer,
    cached: CachedSurrogate,
    reference_point: dict[str, float],
    variable: str,
    points: int,
) -> OneDimensionalResponse:
    """Evaluate one physical parameter across its campaign bounds."""
    physical = physical_axis(explorer, variable, points)
    names = list(cached.metadata["input_names"])
    model_points = np.asarray([[reference_point[name] for name in names]] * points)
    parameter = explorer.campaign.config.parameters[variable]
    model_points[:, names.index(variable)] = [
        explorer._to_model_value(parameter, value) for value in physical
    ]
    mean, uncertainty = predict(cached, names, model_points)
    mean, uncertainty, _ = displayed_prediction(cached, mean, uncertainty)
    response_range = float(np.ptp(mean))
    reference_index = int(np.argmin(np.abs(physical - float(parameter.nominal))))
    reference_uncertainty = float(uncertainty[reference_index])
    return OneDimensionalResponse(
        variable=variable,
        physical_axis=physical,
        mean=mean,
        uncertainty=uncertainty,
        response_range=response_range,
        reference_uncertainty=reference_uncertainty,
        signal_to_uncertainty=response_range / max(reference_uncertainty, 1e-12),
    )


def rank_responses(
    explorer: XoptFBPICExplorer,
    cached: CachedSurrogate,
    reference_point: dict[str, float],
    points: int,
) -> list[OneDimensionalResponse]:
    """Return every active 1D response sorted by decreasing response range."""
    names = list(cached.metadata["input_names"])
    axes = []
    model_point_sets = []
    for variable in explorer.variable_bounds:
        physical = physical_axis(explorer, variable, points)
        model_points = np.asarray([[reference_point[name] for name in names]] * points)
        parameter = explorer.campaign.config.parameters[variable]
        model_points[:, names.index(variable)] = [
            explorer._to_model_value(parameter, value) for value in physical
        ]
        axes.append((variable, physical, parameter))
        model_point_sets.append(model_points)
    means, uncertainties = predict(cached, names, np.concatenate(model_point_sets))
    responses = []
    for index, (variable, physical, parameter) in enumerate(axes):
        start = index * points
        mean, uncertainty, _ = displayed_prediction(
            cached, means[start : start + points], uncertainties[start : start + points]
        )
        reference_index = int(np.argmin(np.abs(physical - float(parameter.nominal))))
        response_range = float(np.ptp(mean))
        reference_uncertainty = float(uncertainty[reference_index])
        responses.append(
            OneDimensionalResponse(
                variable=variable,
                physical_axis=physical,
                mean=mean,
                uncertainty=uncertainty,
                response_range=response_range,
                reference_uncertainty=reference_uncertainty,
                signal_to_uncertainty=response_range
                / max(reference_uncertainty, 1e-12),
            )
        )
    return sorted(responses, key=lambda response: response.response_range, reverse=True)


def write_summary(responses: list[OneDimensionalResponse], output: Path) -> Path:
    """Write ranked 1D response diagnostics as a CSV table."""
    summary = pd.DataFrame(
        {
            "rank": np.arange(1, len(responses) + 1),
            "parameter": [response.variable for response in responses],
            "response_range": [response.response_range for response in responses],
            "reference_uncertainty": [
                response.reference_uncertainty for response in responses
            ],
            "signal_to_reference_uncertainty": [
                response.signal_to_uncertainty for response in responses
            ],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    return output


def plot_ranked_responses(
    cached: CachedSurrogate,
    responses: list[OneDimensionalResponse],
    output: Path,
) -> Path:
    """Plot all ranked 1D curves with posterior uncertainty bands."""
    _, _, output_label = displayed_prediction(cached, np.array([0.0]), np.array([0.0]))
    columns = 3
    rows = int(np.ceil(len(responses) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(15, 3.5 * rows), constrained_layout=True
    )
    for rank, (axis, response) in enumerate(
        zip(axes.flat, responses, strict=True), start=1
    ):
        scale, label = axis_label(response.variable)
        axis.plot(response.physical_axis * scale, response.mean, color="tab:blue")
        axis.fill_between(
            response.physical_axis * scale,
            response.mean - response.uncertainty,
            response.mean + response.uncertainty,
            color="tab:blue",
            alpha=0.2,
        )
        axis.set(
            title=(
                f"{rank}. {response.variable}\n"
                f"range={response.response_range:.3g}, "
                f"range/sigma={response.signal_to_uncertainty:.2f}"
            ),
            xlabel=label,
            ylabel=output_label,
        )
        axis.grid(True, alpha=0.25)
        if cached.metadata.get("outcome_kind") == "probit_feasibility":
            axis.set_ylim(0.0, 1.0)
    for axis in axes.flat[len(responses) :]:
        axis.set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    """Rank every active parameter's cached-GP 1D response near a reference run."""
    args = parse_args()
    explorer = XoptFBPICExplorer(Campaign.from_file(args.config), args.stage)
    cached = load_surrogate(explorer, args.cache_root, args.variable)
    reference = physical_model_point(
        explorer, load_reference_parameters(args.reference_run)
    )
    responses = rank_responses(explorer, cached, reference, args.points)
    output_directory = args.output_directory
    summary = write_summary(
        responses, output_directory / f"{args.variable}_1d_sensitivity_ranking.csv"
    )
    figure = plot_ranked_responses(
        cached, responses, output_directory / f"{args.variable}_1d_sensitivity.png"
    )
    print(f"Wrote {summary}")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
