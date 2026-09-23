"""Post-analysis of Optimas exploration outputs using Gaussian process models.

Loads Optimas ``ExplorationDiagnostics`` data, builds GP models, and optionally
prints parameter indices, contour plots, model-vs-data evaluation, or 1D slice
curves across varying parameters.

Usage:
    python gaussian_process_evaluation.py -d /path/to/optimas/run [OPTIONS]

Arguments:
    -d, --data-path PATH
        Path to Optimas exploration output (or parent containing ``exploration/``).
        Required.

    -p, --print-params
        Print indices of varying (contour) and analyzed (1D curve) parameters, then exit.

    --objective NAME
        Objective function name for the GP model. Default: ``f``.

    --use-obj-min
        Indicates the objective function is to be minimized during optimization

    -c, --show-contour
        Plot a 2D GP contour over two varying parameters (requires ``-cx`` and ``-cy``).

    -cx, --contour-x  -cy, --contour-y INT
        Indices into varying parameters for contour axes.

    -e, --evaluate-model
        Plot GP predictions vs observations for all history points.

    -s, --show-1d-curves
        Plot 1D GP slices for each varying parameter (side-by-side for two models).

    -s1, --model-1  -s2, --model-2 INT
        Indices into analyzed parameters for 1D curves. Omit to use the objective model.

    -s1m, --model-1-maximize  -s2m, --model-2-maximize
        Maximize (rather than minimize) when building the slice GP model.

Examples:
    python gaussian_process_evaluation.py -d /path/to/run -p

    python gaussian_process_evaluation.py -d /path/to/run -c -cx 0 -cy 1

    python gaussian_process_evaluation.py -d /path/to/run -s -s1 0 -s2m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from optimas.diagnostics import ExplorationDiagnostics

DEFAULT_OBJECTIVE_NAME: str = "f"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list for testing. If None, uses ``sys.argv``.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Post-analyze Optimas exploration outputs with Gaussian process models."
        ),
    )

    parser.add_argument(
        "-d",
        "--data-path",
        type=Path,
        required=True,
        help=(
            "Path to Optimas exploration output "
            "(or parent directory containing exploration/)"
        ),
    )

    parser.add_argument(
        "-p",
        "--print-params",
        action="store_true",
        help="Print varying and analyzed parameter indices, then exit",
    )

    parser.add_argument(
        "--objective",
        type=str,
        default=DEFAULT_OBJECTIVE_NAME,
        help="Objective function name for the GP model (default: %(default)s)",
    )

    parser.add_argument(
        "--use-obj-min",
        action="store_true",
        help="Set flag to indicate objective function should be minimized.",
    )

    parser.add_argument(
        "-c",
        "--show-contour",
        action="store_true",
        help="Plot a 2D GP contour over two varying parameters",
    )

    parser.add_argument(
        "-cx",
        "--contour-x",
        type=int,
        default=None,
        metavar="INDEX",
        help="Varying-parameter index for contour horizontal axis",
    )

    parser.add_argument(
        "-cy",
        "--contour-y",
        type=int,
        default=None,
        metavar="INDEX",
        help="Varying-parameter index for contour vertical axis",
    )

    parser.add_argument(
        "-e",
        "--evaluate-model",
        action="store_true",
        help="Plot GP model predictions vs observations for all history points",
    )

    parser.add_argument(
        "-s",
        "--show-1d-curves",
        action="store_true",
        help="Plot 1D GP slices for each varying parameter",
    )

    parser.add_argument(
        "-s1",
        "--model-1",
        type=int,
        default=None,
        metavar="INDEX",
        help="Analyzed-parameter index for left 1D curve (default: objective)",
    )

    parser.add_argument(
        "-s1m",
        "--model-1-maximize",
        action="store_true",
        help="Maximize (not minimize) when building model-1",
    )

    parser.add_argument(
        "-s2",
        "--model-2",
        type=int,
        default=None,
        metavar="INDEX",
        help="Analyzed-parameter index for right 1D curve (default: objective)",
    )

    parser.add_argument(
        "-s2m",
        "--model-2-maximize",
        action="store_true",
        help="Maximize (not minimize) when building model-2",
    )

    return parser.parse_args(argv)


def _resolve_data_path(data_path: Path) -> Path:
    """Resolve exploration directory from user-supplied path.

    Args:
        data_path: User path to exploration output or its parent.

    Returns:
        Path to the exploration diagnostics directory.

    Raises:
        FileNotFoundError: If exploration data cannot be located.
    """
    if (data_path / "exploration").exists():
        return data_path / "exploration"
    if data_path.exists():
        return data_path
    raise FileNotFoundError(f"Can not locate exploration data within '{data_path}'")


def _print_parameter_indices(
    varying_param_names: list[str],
    analyzed_params: list[Any],
) -> None:
    """Print indices of varying and analyzed parameters.

    Args:
        varying_param_names: Names of parameters that vary during exploration.
        analyzed_params: Optimas analyzed parameter objects.
    """
    print("Varying Parameters (Contour Axes):")
    for i, name in enumerate(varying_param_names):
        print(f" [{i}]: {name}")
    print()
    print("Analyzed Parameters (1D Curves):")
    for i, param in enumerate(analyzed_params):
        print(f" [{i}]: {param.name}")


def _plot_contour(
    obj_model: Any,
    varying_param_names: list[str],
    contour_x: int,
    contour_y: int,
) -> None:
    """Plot a 2D GP contour for two varying parameters.

    Args:
        obj_model: GP model for the objective.
        varying_param_names: Names of varying parameters.
        contour_x: Index of parameter for horizontal axis.
        contour_y: Index of parameter for vertical axis.
    """
    x_param = varying_param_names[contour_x]
    y_param = varying_param_names[contour_y]
    _, _ = obj_model.plot_contour(
        param_x=x_param,
        param_y=y_param,
        mode="both",
        figsize=(6, 3),
        dpi=200,
    )
    plt.tight_layout()
    plt.show()


def _evaluate_and_plot_model(
    obj_model: Any,
    diags: ExplorationDiagnostics,
    objective_name: str,
) -> None:
    """Plot GP model predictions vs observations.

    Args:
        obj_model: GP model for the objective.
        diags: Exploration diagnostics instance.
        objective_name: Name of the objective column in history.
    """
    mean, sem = obj_model.evaluate_model(diags.history)
    min_f = np.min(diags.history[objective_name])
    max_f = np.max(diags.history[objective_name])

    _, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.errorbar(
        diags.history[objective_name],
        mean,
        yerr=sem,
        fmt="o",
        ms=4,
        label="Data",
    )
    ax.plot(
        [min_f, max_f],
        [min_f, max_f],
        color="k",
        ls="--",
        label="Ideal correlation",
    )
    ax.set_xlabel("Observations")
    ax.set_ylabel("Model predictions")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def _build_slice_model(
    diags: ExplorationDiagnostics,
    obj_model: Any,
    objective_name: str,
    analyzed_params: list[Any],
    index: Optional[int],
    maximize: bool,
) -> tuple[Any, str]:
    """Build one GP model for a 1D slice plot panel.

    Args:
        diags: Exploration diagnostics instance.
        obj_model: GP model for the objective.
        objective_name: Name of the objective function.
        analyzed_params: Optimas analyzed parameter objects.
        index: Index into analyzed_params, or None to use the objective model.
        maximize: If True, pass minimize=False to build_gp_model.

    Returns:
        Tuple of (gp_model, label).
    """
    if index is not None:
        label = analyzed_params[index].name
        return diags.build_gp_model(label, minimize=not maximize), label
    return obj_model, objective_name


def _best_data_evaluation(
    diags: ExplorationDiagnostics,
    objective_name: str,
    analyzed_params: list[Any],
    varying_param_names: list[str],
    use_obj_min: bool = False,
) -> tuple[int, dict[str, float]]:
    """Find the history index and parameters at the best observed objective.

    Args:
        diags: Exploration diagnostics instance.
        objective_name: Name of the objective function.
        analyzed_params: Optimas analyzed parameter objects.
        varying_param_names: Names of varying parameters.

    Returns:
        Tuple of (history index, parameter dict at that index).
    """
    if use_obj_min:
        data_best_obj = min(diags.history[objective_name])
        data_best_i = int(np.argmin(diags.history[objective_name]))
    else:
        data_best_obj = max(diags.history[objective_name])
        data_best_i = int(np.argmax(diags.history[objective_name]))
    params_data_max: dict[str, float] = {objective_name: data_best_obj}
    for param in analyzed_params:
        params_data_max[param.name] = diags.history[param.name][data_best_i]
    for var in varying_param_names:
        params_data_max[var] = float(diags.history[var].values[data_best_i])
    return data_best_i, params_data_max


def _plot_1d_curves(
    diags: ExplorationDiagnostics,
    obj_model: Any,
    objective_name: str,
    varying_param_names: list[str],
    analyzed_params: list[Any],
    model_1_index: Optional[int],
    model_1_maximize: bool,
    model_2_index: Optional[int],
    model_2_maximize: bool,
    use_obj_min: bool = False,
) -> None:
    """Plot 1D GP slices for each varying parameter.

    Args:
        diags: Exploration diagnostics instance.
        obj_model: GP model for the objective.
        objective_name: Name of the objective function.
        varying_param_names: Names of varying parameters.
        analyzed_params: Optimas analyzed parameter objects.
        model_1_index: Index into analyzed_params for left panel, or None.
        model_1_maximize: If True, pass minimize=False to build_gp_model.
        model_2_index: Index into analyzed_params for right panel, or None.
        model_2_maximize: If True, pass minimize=False to build_gp_model.
    """
    model_1, model_1_label = _build_slice_model(
        diags,
        obj_model,
        objective_name,
        analyzed_params,
        model_1_index,
        model_1_maximize,
    )
    model_2, model_2_label = _build_slice_model(
        diags,
        obj_model,
        objective_name,
        analyzed_params,
        model_2_index,
        model_2_maximize,
    )

    best_model = obj_model.get_best_evaluation()
    best_data = _best_data_evaluation(
        diags,
        objective_name,
        analyzed_params,
        varying_param_names,
        use_obj_min=use_obj_min,
    )

    for name in varying_param_names:
        _ = plt.figure(figsize=(10, 3), dpi=150)
        gs = GridSpec(1, 2, wspace=0.4, bottom=0.2)

        _, ax1 = model_1.plot_slice(
            name, figsize=(6, 3), dpi=100, subplot_spec=gs[0, 0]
        )
        _, ax2 = model_2.plot_slice(
            name, figsize=(6, 3), dpi=100, subplot_spec=gs[0, 1]
        )

        model_location = best_model[1].get(name)
        data_location = best_data[1].get(name)

        ax1.vlines(
            [model_location, data_location],
            ymin=min(diags.history[model_1_label]),
            ymax=max(diags.history[model_1_label]),
            linestyles="dashed",
            color=["r", "b"],
        )
        ax1.scatter([data_location], [best_data[1][model_1_label]], c="b")
        ax1.text(
            model_location * 1.01,
            max(diags.history[model_1_label]) * 0.8,
            f"Model: {best_model[0]}",
        )
        ax1.text(
            data_location * 1.01,
            max(diags.history[model_1_label]) * 0.9,
            f"Data: {best_data[0]}",
        )

        ax2.vlines(
            [model_location, data_location],
            ymin=min(diags.history[model_2_label]),
            ymax=max(diags.history[model_2_label]),
            linestyles="dashed",
            color=["r", "b"],
        )
        ax2.scatter([data_location], [best_data[1][model_2_label]], c="b")
        ax2.text(
            model_location * 1.01,
            max(diags.history[model_2_label]) * 0.8,
            f"Model: {best_model[0]}",
        )
        ax2.text(
            data_location * 1.01,
            max(diags.history[model_2_label]) * 0.9,
            f"Data: {best_data[0]}",
        )

        ax1.set_title(f"{model_1_label} vs {name}")
        ax2.set_title(f"{model_2_label} vs {name}")

        plt.show()
        plt.close()


def process(args: argparse.Namespace) -> None:
    """Load diagnostics and run requested GP post-analysis.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If contour mode is enabled without axis indices.
    """
    data_path = _resolve_data_path(args.data_path)
    diags = ExplorationDiagnostics(str(data_path))

    varying_param_names = [var.name for var in diags.varying_parameters]
    analyzed_params = diags.analyzed_parameters

    if args.print_params:
        _print_parameter_indices(varying_param_names, analyzed_params)
        return

    obj_model = diags.build_gp_model(args.objective)

    if args.show_contour:
        if args.contour_x is None or args.contour_y is None:
            raise ValueError("Contour mode requires both -cx and -cy")
        _plot_contour(obj_model, varying_param_names, args.contour_x, args.contour_y)

    if args.evaluate_model:
        _evaluate_and_plot_model(obj_model, diags, args.objective)

    if args.show_1d_curves:
        _plot_1d_curves(
            diags,
            obj_model,
            args.objective,
            varying_param_names,
            analyzed_params,
            args.model_1,
            args.model_1_maximize,
            args.model_2,
            args.model_2_maximize,
            use_obj_min=args.use_obj_min,
        )


def main() -> None:
    """Entry point for Gaussian process evaluation script."""
    args = parse_args()
    try:
        process(args)
    except (ValueError, FileNotFoundError, IndexError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
