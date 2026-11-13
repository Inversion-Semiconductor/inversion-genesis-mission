"""Overlay completed campaign evaluations on cached one-dimensional GP scans."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exploration_fbpic.campaign import Campaign
from exploration_fbpic.exploration import XoptFBPICExplorer
from exploration_fbpic.surrogate_cache import load_surrogate

from model_analysis.plot_cached_surrogate import (
    axis_label,
    displayed_prediction,
    physical_axis,
    physical_model_point,
    predict,
)


@dataclass(frozen=True)
class Observation:
    """One completed campaign result to show on a physical scan."""

    label: str
    parameters: dict[str, float | str]
    status: str
    charge_pc: float | None


def parse_args() -> argparse.Namespace:
    """Parse cached-GP observation-overlay arguments."""
    parser = argparse.ArgumentParser(
        description="Plot cached GP one-dimensional scans with completed observations."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", default="stage_2")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--variable", default="total_beam_charge_pc")
    parser.add_argument(
        "--feasibility-variable",
        default="feasibility",
        help="cached feasibility model to overlay on a 0-100%% right axis",
    )
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--energy-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--focus-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=160)
    parser.add_argument(
        "--charge-ylim",
        nargs=2,
        type=float,
        metavar=("MIN_PC", "MAX_PC"),
        help="fixed left charge-axis limits in pC",
    )
    parser.add_argument(
        "--result-file",
        default="results_shifted.json",
        help="preferred result record; native result.json is used as fallback",
    )
    return parser.parse_args()


def load_observation(run_path: Path, result_filename: str) -> Observation:
    """Load one result record, preferring shifted features when available."""
    native_path = run_path / "result.json"
    if not native_path.is_file():
        raise FileNotFoundError(f"Missing {native_path}")
    native = json.loads(native_path.read_text(encoding="utf-8"))
    selected_path = run_path / result_filename
    record = (
        json.loads(selected_path.read_text(encoding="utf-8"))
        if selected_path.is_file()
        else native
    )
    features = record["result"].get("features", {})
    charge = features.get("total_beam_charge_pc")
    return Observation(
        label=native["result"]["run_id"],
        parameters=dict(native["manifest"]["parameters"]),
        status=native["result"]["status"],
        charge_pc=None if charge is None else float(charge),
    )


def scan_prediction(
    explorer: XoptFBPICExplorer,
    cached,
    reference: dict[str, float],
    variable: str,
    points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Evaluate a cached GP across one physical active parameter range."""
    physical = physical_axis(explorer, variable, points)
    names = list(cached.metadata["input_names"])
    model_points = np.asarray([[reference[name] for name in names]] * points)
    parameter = explorer.campaign.config.parameters[variable]
    model_points[:, names.index(variable)] = [
        explorer._to_model_value(parameter, value) for value in physical
    ]
    mean, uncertainty = predict(cached, names, model_points)
    mean, uncertainty, label = displayed_prediction(cached, mean, uncertainty)
    return physical, mean, uncertainty, label


def plot_scan(
    axis,
    explorer: XoptFBPICExplorer,
    cached,
    feasibility_cached,
    reference: dict[str, float],
    variable: str,
    observations: list[Observation],
    points: int,
    charge_ylim: tuple[float, float] | None,
) -> None:
    """Plot one cached GP scan and completed beam/no-beam observations."""
    physical, mean, uncertainty, output_label = scan_prediction(
        explorer, cached, reference, variable, points
    )
    scale, x_label = axis_label(variable)
    axis.plot(physical * scale, mean, color="tab:blue", label="Cached GP mean")
    axis.fill_between(
        physical * scale,
        mean - uncertainty,
        mean + uncertainty,
        color="tab:blue",
        alpha=0.22,
        label="GP posterior std. dev.",
    )
    beam = [
        observation for observation in observations if observation.charge_pc is not None
    ]
    no_beam = [
        observation for observation in observations if observation.charge_pc is None
    ]
    if beam:
        x_values = [
            float(observation.parameters[variable]) * scale for observation in beam
        ]
        y_values = [observation.charge_pc for observation in beam]
        axis.scatter(
            x_values,
            y_values,
            color="black",
            s=48,
            zorder=5,
            label="Completed beam run",
        )
        for observation, x_value, y_value in zip(beam, x_values, y_values, strict=True):
            axis.annotate(
                observation.label.replace("scan_728_", ""),
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    if no_beam:
        baseline = 0.0
        x_values = [
            float(observation.parameters[variable]) * scale for observation in no_beam
        ]
        axis.scatter(
            x_values,
            [baseline] * len(no_beam),
            marker="x",
            color="tab:red",
            s=58,
            linewidths=1.5,
            zorder=5,
            label="Completed no-beam run",
        )
        for observation, x_value in zip(no_beam, x_values, strict=True):
            axis.annotate(
                observation.label.replace("scan_728_", ""),
                (x_value, baseline),
                xytext=(5, -12),
                textcoords="offset points",
                fontsize=8,
                color="tab:red",
            )
    axis.set(xlabel=x_label, ylabel=output_label)
    if charge_ylim is not None:
        axis.set_ylim(*charge_ylim)
    axis.grid(True, alpha=0.3)
    feasibility_physical, feasibility_mean, _, _ = scan_prediction(
        explorer, feasibility_cached, reference, variable, points
    )
    feasibility_axis = axis.twinx()
    feasibility_axis.plot(
        feasibility_physical * scale,
        100.0 * feasibility_mean,
        color="tab:red",
        linestyle="--",
        linewidth=1.8,
        label="Feasibility",
    )
    feasibility_axis.set_ylabel("Beam feasibility (%)", color="tab:red")
    feasibility_axis.tick_params(axis="y", labelcolor="tab:red")
    feasibility_axis.set_ylim(0.0, 100.0)
    lines, labels = axis.get_legend_handles_labels()
    feasibility_lines, feasibility_labels = feasibility_axis.get_legend_handles_labels()
    axis.legend(lines + feasibility_lines, labels + feasibility_labels, fontsize=8)


def main() -> None:
    """Write the requested energy and focal-position cached-GP comparison plot."""
    args = parse_args()
    explorer = XoptFBPICExplorer(Campaign.from_file(args.config), args.stage)
    cached = load_surrogate(explorer, args.cache_root, args.variable)
    feasibility_cached = load_surrogate(
        explorer, args.cache_root, args.feasibility_variable
    )
    reference = physical_model_point(
        explorer,
        load_observation(args.reference_run, args.result_file).parameters,
    )
    energy_observations = [
        load_observation(path, args.result_file) for path in args.energy_runs
    ]
    focus_observations = [
        load_observation(path, args.result_file) for path in args.focus_runs
    ]
    figure, axes = plt.subplots(2, 1, figsize=(9, 10), constrained_layout=True)
    plot_scan(
        axes[0],
        explorer,
        cached,
        feasibility_cached,
        reference,
        "laser_energy_J",
        energy_observations,
        args.points,
        None if args.charge_ylim is None else tuple(args.charge_ylim),
    )
    axes[0].set_title("Stage 2 charge GP: laser-energy validation scan")
    plot_scan(
        axes[1],
        explorer,
        cached,
        feasibility_cached,
        reference,
        "laser_focal_position_m",
        focus_observations,
        args.points,
        None if args.charge_ylim is None else tuple(args.charge_ylim),
    )
    axes[1].set_title("Stage 2 charge GP: focal-position validation scan")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
