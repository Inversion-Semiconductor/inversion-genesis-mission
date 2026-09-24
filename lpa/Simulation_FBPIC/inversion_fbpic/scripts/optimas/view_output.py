"""Script to visualize Optimas optimization results.

This script loads exploration diagnostics from an Optimas optimization run
and creates various plots to analyze the optimization results. It supports
two data sources:

  - **exploration** (default): Loads from the Optimas "exploration" directory
    using ``ExplorationDiagnostics``.
  - **log**: Parses an ``optimization.log`` file directly. Useful when a run
    was interrupted before generating exploration history files.

Usage:
    Command-line usage:
        python view_output.py [OPTIONS]

    Optional Arguments:
        -d, -dir, --optimization-folder PATH
            Path to the optimization folder containing the "exploration" directory.
            Default: current working directory

        --source {exploration,log,auto}
            Where to read data from. "auto" uses the exploration folder if it
            exists, otherwise falls back to the log file.
            Default: auto

        --log-file PATH
            Path to the optimization.log file. If not given, defaults to
            <optimization-folder>/optimization.log.

        --list-parameters
            Print the available parameter names and exit.

        --plot-objective
            Plot the evolution of the objective function. Enabled by default.

        --no-plot-objective
            Disable plotting the objective function.

        --plot-parameter
            Plot correlation between a parameter and the objective function.

        --parameter-1 INT
            Parameter index used with --plot-parameter.

        --scatter
            Plot objective function against analyzed charge, labeling every trial.

    Examples:
        # Use all defaults (auto-detects exploration folder or log file)
        python view_output.py

        # Specify a different optimization folder
        python view_output.py -dir /path/to/optimization/folder

        # Read from the log file instead of the exploration folder
        python view_output.py --source log

        # Read from a specific log file
        python view_output.py --source log --log-file /path/to/optimization.log

        # List available parameters (works with any source)
        python view_output.py --list-parameters

        # Enable parameter correlation plot
        python view_output.py --plot-parameter --parameter-1 3

        # Plot objective function against charge
        python view_output.py --scatter

    For Gaussian process model analysis (contours, slices, model evaluation), use
    ``gaussian-process-evaluation`` instead.
"""

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Log parsing helper (for interrupted runs without exploration data)
# ---------------------------------------------------------------------------


def parse_log_file(
    log_path: Path,
) -> Tuple[
    List[int], List[float], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]
]:
    """Parse an optimization.log file to extract trial indices and results.

    Args:
        log_path: Path to the optimization.log file

    Returns:
        Tuple of (trial_indices, objective_values, varying_param_data,
        analyzed_param_data), where each parameter mapping associates a trial
        index with a {param_name: value} dict.

    Notes:
        This regular expression is valid for Optimas version 0.8.1.  Future versions may not
        be supported if the formatting of the output log file changes.
    """
    print(f"Parsing log file: {log_path}")

    objective_pattern = (
        r"Completed trial (\d+) with objective\(s\) \{'f': "
        r"\(np\.float64\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)"
    )
    param_pattern = r"Generated trial (\d+) with parameters \{(.+?)\}(?:\s|$)"
    analyzed_param_pattern = (
        r"Completed trial (\d+) with objective\(s\).*?"
        r"analyzed parameter\(s\) \{(.+?)\}(?:\s|$)"
    )
    value_pattern = (
        r"'([^']+)':\s*(?:\()?np\.float64\("
        r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)"
    )

    trial_data: Dict[int, float] = {}
    param_data: Dict[int, Dict[str, float]] = {}
    analyzed_param_data: Dict[int, Dict[str, float]] = {}

    with open(log_path, "r") as f:
        for line in f:
            match = re.search(objective_pattern, line)
            if match:
                trial_idx = int(match.group(1))
                objective_value = float(match.group(2))
                trial_data[trial_idx] = objective_value

            match = re.search(param_pattern, line)
            if match:
                trial_idx = int(match.group(1))
                params_str = match.group(2)
                params: Dict[str, float] = {}
                param_entries = re.findall(value_pattern, params_str)
                for param_name, param_value in param_entries:
                    params[param_name] = float(param_value)
                param_data[trial_idx] = params

            match = re.search(analyzed_param_pattern, line)
            if match:
                trial_idx = int(match.group(1))
                analyzed_params = {
                    name: float(value)
                    for name, value in re.findall(value_pattern, match.group(2))
                }
                analyzed_param_data[trial_idx] = analyzed_params

    if not trial_data:
        raise ValueError("No completed trials found in log file")

    sorted_trials = sorted(trial_data.items())
    trial_indices = [t[0] for t in sorted_trials]
    objective_values = [t[1] for t in sorted_trials]

    print(f"Found {len(trial_indices)} completed trials")
    if param_data:
        print(f"Found parameters for {len(param_data)} trials")
    if analyzed_param_data:
        print(f"Found analyzed parameters for {len(analyzed_param_data)} trials")

    return trial_indices, objective_values, param_data, analyzed_param_data


def load_trial_data(
    args: argparse.Namespace,
) -> Tuple[
    List[int], List[float], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]
]:
    """Resolve the --source flag and return trial indices, objectives, and parameters.

    Raises FileNotFoundError / ValueError on problems.
    """
    opt_dir: Path = args.optimization_folder
    source: str = args.source

    if source == "log":
        log_path = (
            Path(args.log_file) if args.log_file else opt_dir / "optimization.log"
        )
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        return parse_log_file(log_path)

    if source == "exploration":
        return _load_from_exploration(opt_dir)

    # --- auto mode: exploration -> log ----------------------------------
    exploration_dir = opt_dir / "exploration"
    if exploration_dir.exists():
        try:
            return _load_from_exploration(opt_dir)
        except Exception as exc:
            print(f"Warning: could not load exploration data ({exc}), trying log file…")

    log_path = Path(args.log_file) if args.log_file else opt_dir / "optimization.log"
    if log_path.exists():
        print("Auto-detected: using optimization.log")
        return parse_log_file(log_path)

    raise FileNotFoundError(f"No exploration directory or log file found in {opt_dir}")


def _load_from_exploration(
    opt_dir: Path,
) -> Tuple[
    List[int], List[float], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]
]:
    """Load trial data via ExplorationDiagnostics and convert to the common format."""
    from optimas.diagnostics import ExplorationDiagnostics

    diags = ExplorationDiagnostics(str(opt_dir / "exploration"))
    history = diags.history
    obj_name = diags.objectives[0].name
    objective_values_arr = history[obj_name].values

    trial_indices = list(range(len(objective_values_arr)))
    objective_values = objective_values_arr.tolist()

    param_data: Dict[int, Dict[str, float]] = {}
    analyzed_param_data: Dict[int, Dict[str, float]] = {}
    for idx in trial_indices:
        params = {}
        for p in diags.varying_parameters:
            params[p.name] = float(history[p.name].values[idx])
        param_data[idx] = params
        analyzed_param_data[idx] = {
            p.name: float(history[p.name].values[idx])
            for p in diags.analyzed_parameters
        }

    return trial_indices, objective_values, param_data, analyzed_param_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize Optimas optimization results."
    )

    parser.add_argument(
        "-d",
        "-dir",
        "--optimization-folder",
        type=Path,
        default=Path("./"),
        help="Path to the optimization folder containing the 'exploration' directory",
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["exploration", "log", "auto"],
        default="auto",
        help=(
            "Where to read data from. 'exploration' uses ExplorationDiagnostics, "
            "'log' parses optimization.log, "
            "'auto' tries exploration then log (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to optimization.log file (overrides default location)",
    )

    parser.add_argument(
        "--list-parameters",
        action="store_true",
        help="Print available parameter names and exit",
    )

    parser.add_argument(
        "--plot-objective",
        action="store_true",
        default=True,
        help="Plot the evolution of the objective function (default: enabled)",
    )

    parser.add_argument(
        "--no-plot-objective",
        action="store_false",
        dest="plot_objective",
        help="Disable plotting the objective function",
    )

    parser.add_argument(
        "--plot-parameter",
        action="store_true",
        default=False,
        help="Plot correlation between a parameter and the objective function",
    )

    parser.add_argument(
        "--parameter-1",
        type=int,
        default=None,
        help="Parameter index used with --plot-parameter",
    )

    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Plot objective function against analyzed charge",
    )

    return parser.parse_args()


def _get_sorted_param_names(
    param_data: Dict[int, Dict[str, float]],
) -> List[str]:
    """Return a sorted list of parameter names from the first available trial."""
    if not param_data:
        return []
    first = next(iter(param_data.values()))
    return sorted(first.keys())


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def process(args: argparse.Namespace) -> None:
    """Main function to run the optimization visualization.

    Args:
        args: Parsed command-line arguments. See module docstring for argument details.
    """
    trial_indices, objective_values, param_data, analyzed_param_data = load_trial_data(
        args
    )

    if args.list_parameters:
        names = _get_sorted_param_names(param_data)
        if names:
            print("\nAvailable parameters:")
            for i, name in enumerate(names):
                print(f"  [{i}] {name}")
            print()
        else:
            print("No parameters found in the data")
        return

    # Determine whether we can use ExplorationDiagnostics
    diags: Optional[Any] = None
    exploration_dir = args.optimization_folder / "exploration"
    if args.source == "exploration" or (
        args.source == "auto" and exploration_dir.exists()
    ):
        try:
            from optimas.diagnostics import ExplorationDiagnostics

            diags = ExplorationDiagnostics(str(exploration_dir))
        except Exception:
            pass

    if args.plot_objective:
        _plot_objective_evolution(
            trial_indices,
            objective_values,
            param_data,
            analyzed_param_data,
            diags,
        )

    if args.plot_parameter:
        if args.parameter_1 is None:
            print("Error: --parameter-1 is required when using --plot-parameter.")
            return
        _plot_parameter_correlation(
            trial_indices, objective_values, param_data, args.parameter_1
        )

    if args.scatter:
        _plot_charge_scatter(trial_indices, objective_values, analyzed_param_data)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_charge_scatter(
    trial_indices: List[int],
    objective_values: List[float],
    analyzed_param_data: Dict[int, Dict[str, float]],
) -> None:
    """Plot the objective function against analyzed charge for each trial."""
    charges: List[float] = []
    objectives: List[float] = []
    valid_trials: List[int] = []

    for trial_idx, objective_value in zip(trial_indices, objective_values):
        charge = analyzed_param_data.get(trial_idx, {}).get("charge")
        if (
            charge is None
            or not np.isfinite(charge)
            or not np.isfinite(objective_value)
        ):
            continue
        charges.append(charge)
        objectives.append(objective_value)
        valid_trials.append(trial_idx)

    if not charges:
        print(
            "Error: no finite charge and objective values available for scatter plot."
        )
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(charges, objectives, alpha=0.7, s=45, edgecolors="k", linewidth=0.4)

    for trial_idx, charge, objective_value in zip(valid_trials, charges, objectives):
        ax.annotate(
            str(trial_idx),
            xy=(charge, objective_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("Charge (pC)", fontsize=12)
    ax.set_ylabel("Objective Function (f)", fontsize=12)
    ax.set_title("Objective Function vs Charge", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_objective_evolution(
    trial_indices: List[int],
    objective_values: List[float],
    param_data: Dict[int, Dict[str, float]],
    analyzed_param_data: Dict[int, Dict[str, float]],
    diags: Optional[Any] = None,
) -> None:
    """Plot the evolution of the objective function over time.

    When *diags* (an ``ExplorationDiagnostics`` instance) is available the
    built-in ``plot_objective`` is used as a base; otherwise a standalone
    matplotlib plot is created from the raw trial data.
    """
    obj_array = np.array(objective_values)
    trial_array = np.array(trial_indices)
    finite_objectives = np.isfinite(obj_array)

    if diags is not None:
        diags.plot_objective(show_trace=True)
        plt.title("Optimas: Evolution of Objective Function")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        # The Optimas template maximizes f; minimization explorations need an inverse trace.
        best_so_far = np.maximum.accumulate(
            np.where(finite_objectives, obj_array, -np.inf)
        )

        ax.scatter(
            trial_array,
            obj_array,
            alpha=0.7,
            s=35,
            edgecolors="k",
            linewidth=0.4,
            label="Trial objective",
        )
        ax.plot(
            trial_array,
            best_so_far,
            color="tab:orange",
            linewidth=2,
            label="Best objective so far",
        )

        if finite_objectives.any():
            # Highlight the best finite trial.
            best_idx = int(np.nanargmax(obj_array))
            best_trial = trial_array[best_idx]
            best_value = obj_array[best_idx]
            ax.plot(
                best_trial,
                best_value,
                "r*",
                markersize=15,
                label=f"Best (trial {best_trial}: f={best_value:.2f})",
            )

        # Annotate top 5 trials
        top_5 = np.flatnonzero(finite_objectives)[
            np.argsort(obj_array[finite_objectives])[-5:][::-1]
        ]
        for rank, idx in enumerate(top_5, 1):
            t_num = trial_array[idx]
            o_val = obj_array[idx]
            off_x = 0.5
            off_y = 0.02 * (
                np.max(obj_array[finite_objectives])
                - np.min(obj_array[finite_objectives])
            )
            if rank % 2 == 0:
                off_y = -off_y
            ax.annotate(
                f"{t_num}",
                xy=(t_num, o_val),
                xytext=(t_num + off_x, o_val + off_y),
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="yellow",
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                ),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=0.5),
            )

        ax.set_xlabel("Trial Index", fontsize=12)
        ax.set_ylabel("Objective Function (f)", fontsize=12)
        ax.set_title(
            "Optimas: Evolution of Objective Function", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    # --- Summary statistics (shared by both paths) ----------------------
    if finite_objectives.any():
        best_idx = int(np.nanargmax(obj_array))
        best_trial = trial_array[best_idx]
        best_value = obj_array[best_idx]
    else:
        best_trial = None
        best_value = None

    print(f"\n{'=' * 60}")
    print("Summary Statistics")
    print(f"{'=' * 60}")
    print(f"Total trials completed: {len(trial_indices)}")
    print(f"Trial index range: {min(trial_indices)} to {max(trial_indices)}")
    if best_trial is None:
        print("\nBest trial: unavailable (no finite objective values)")
        print("Best objective value: unavailable")
        print("Mean objective value: unavailable")
        print("Std objective value:  unavailable")
    else:
        print(f"\nBest trial: {best_trial}")
        print(f"Best objective value: {best_value:.4f}")
        print(f"Mean objective value: {np.mean(obj_array[finite_objectives]):.4f}")
        print(f"Std objective value:  {np.std(obj_array[finite_objectives]):.4f}")
    print(
        f"Min objective value:  {np.min(obj_array):.4f} "
        f"(trial {trial_array[int(np.argmin(obj_array))]})"
    )

    top_5 = np.argsort(obj_array)[-5:][::-1]
    print("\nTop 5 trials:")
    for rank, idx in enumerate(top_5, 1):
        print(f"  {rank}. Trial {trial_array[idx]}: f = {obj_array[idx]:.4f}")

    if best_trial in param_data:
        print(f"\nParameters for best trial {best_trial}:")
        for pname, pval in sorted(param_data[best_trial].items()):
            print(f"  {pname}: {pval:.6e}")
    if best_trial in analyzed_param_data:
        print(f"\nAnalyzed parameters for best trial {best_trial}:")
        for pname, pval in sorted(analyzed_param_data[best_trial].items()):
            print(f"  {pname}: {pval:.6e}")
    print(f"{'=' * 60}\n")

    plt.show()


def _plot_parameter_correlation(
    trial_indices: List[int],
    objective_values: List[float],
    param_data: Dict[int, Dict[str, float]],
    parameter_idx: int,
) -> None:
    """Plot correlation between a parameter and the objective function.

    *parameter_idx* is an integer index into the sorted parameter-name list.
    """
    param_names = _get_sorted_param_names(param_data)
    if not param_names:
        print("Error: no parameter data available.")
        return
    if parameter_idx < 0 or parameter_idx >= len(param_names):
        print(
            f"Error: --parameter-1 index {parameter_idx} out of range "
            f"(0–{len(param_names) - 1}). Use --list-parameters to see names."
        )
        return

    param_name = param_names[parameter_idx]

    p_vals: List[float] = []
    o_vals: List[float] = []
    t_nums: List[int] = []

    for trial_idx, obj_val in zip(trial_indices, objective_values):
        if trial_idx in param_data and param_name in param_data[trial_idx]:
            p_vals.append(param_data[trial_idx][param_name])
            o_vals.append(obj_val)
            t_nums.append(trial_idx)

    if not p_vals:
        print(f"Error: parameter '{param_name}' not found in any trials")
        return

    param_array = np.array(p_vals)
    obj_array = np.array(o_vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(param_array, obj_array, alpha=0.6, s=50, edgecolors="k", linewidth=0.5)

    best_idx = int(np.argmax(obj_array))
    ax.scatter(
        param_array[best_idx],
        obj_array[best_idx],
        color="red",
        s=200,
        marker="*",
        edgecolors="k",
        linewidth=1,
        zorder=5,
        label=f"Best: {param_name}={param_array[best_idx]:.4e}, f={obj_array[best_idx]:.2f}",
    )

    # Annotate top 5
    top_5 = np.argsort(obj_array)[-5:][::-1]
    for rank, idx in enumerate(top_5, 1):
        off_x = 0.02 * (np.max(param_array) - np.min(param_array))
        off_y = 0.02 * (np.max(obj_array) - np.min(obj_array))
        if rank % 2 == 0:
            off_x = -off_x
        if rank > 2:
            off_y = -off_y
        ax.annotate(
            f"{t_nums[idx]}",
            xy=(param_array[idx], obj_array[idx]),
            xytext=(param_array[idx] + off_x, obj_array[idx] + off_y),
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="yellow",
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            ),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=0.5),
        )

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Objective Function (f)", fontsize=12)
    ax.set_title(f"Objective Function vs {param_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    correlation = np.corrcoef(param_array, obj_array)[0, 1]
    print(f"\n{'=' * 60}")
    print(f"Parameter Correlation Analysis: {param_name}")
    print(f"{'=' * 60}")
    print(f"Number of trials: {len(p_vals)}")
    print(f"Parameter range: [{np.min(param_array):.4e}, {np.max(param_array):.4e}]")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(
        f"Best value: {param_array[best_idx]:.4e} (objective: {obj_array[best_idx]:.4f})"
    )
    top_5 = np.argsort(obj_array)[-5:][::-1]
    print("\nTop 5 trials:")
    for rank, idx in enumerate(top_5, 1):
        print(
            f"  {rank}. Trial {t_nums[idx]}: {param_name} = {param_array[idx]:.4e}, "
            f"f = {obj_array[idx]:.4f}"
        )
    print(f"{'=' * 60}\n")

    plt.show()


def main() -> None:
    """Main entry point for script execution: parses command-line arguments and calls process()"""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
