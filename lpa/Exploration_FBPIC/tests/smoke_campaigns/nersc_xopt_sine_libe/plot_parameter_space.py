"""Plot completed Xopt sine runs in amplitude/phase parameter space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


RUN_ID = re.compile(r"sim_(\d+)$")


def load_results(run_root: Path, initial_design_size: int) -> pd.DataFrame:
    """Read completed toy inputs and results, deriving batch from simulation ID."""
    rows: list[dict[str, object]] = []
    for run_directory in sorted(run_root.glob("sim_*")):
        match = RUN_ID.fullmatch(run_directory.name)
        result_path = run_directory / "toy_result.json"
        if match is None or not result_path.is_file():
            continue
        run_id = int(match.group(1))
        inputs = json.loads((run_directory / "toy_input.json").read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        gpu_assignments = []
        for assignment_path in sorted(run_directory.glob("gpu_assignment_rank_*.json")):
            assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
            gpu_assignments.append(str(assignment.get("cuda_visible_devices", "")))
        batch = int(inputs.get("batch", 0 if run_id < initial_design_size else 1))
        rows.append(
            {
                "sim_id": run_id,
                "amplitude": float(inputs["amplitude"]),
                "phase_rad": float(inputs["phase_rad"]),
                "f": float(result["f"]),
                "batch": batch,
                "source": "initial" if batch == 0 else "xopt",
                "gpu_assignments": ", ".join(gpu_assignments),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no completed toy_result.json files found under {run_root}")
    return pd.DataFrame(rows).sort_values("sim_id").reset_index(drop=True)


def plot_results(results: pd.DataFrame, output_path: Path, snapshot_path: Path | None = None) -> None:
    """Create the annotated two-dimensional parameter-space scatter plot."""
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    phase_min, phase_max = results["phase_rad"].min(), results["phase_rad"].max()
    amplitude_min, amplitude_max = results["amplitude"].min(), results["amplitude"].max()
    if snapshot_path is not None and snapshot_path.is_file():
        snapshot = np.load(snapshot_path)
        phase, amplitude = snapshot["phase_rad"], snapshot["amplitude"]
        contours = axis.contourf(phase, amplitude, snapshot["stddev"], levels=18, cmap="Greys", alpha=0.45)
        uncertainty = figure.colorbar(contours, ax=axis, pad=0.02)
        uncertainty.set_label("Surrogate posterior standard deviation")
    scatter = axis.scatter(
        results["phase_rad"],
        results["amplitude"],
        c=results["f"],
        cmap="coolwarm",
        s=100,
        edgecolors="black",
        linewidths=0.7,
    )
    for row in results.itertuples():
        label = f"{row.sim_id} (B{row.batch})"
        if row.amplitude > amplitude_min + 0.82 * (amplitude_max - amplitude_min):
            offset, alignment = (5, -8), {"va": "top", "ha": "left"}
        elif row.phase_rad > phase_min + 0.88 * (phase_max - phase_min):
            offset, alignment = (-5, 5), {"va": "bottom", "ha": "right"}
        else:
            offset, alignment = (5, 5), {"va": "bottom", "ha": "left"}
        axis.annotate(label, (row.phase_rad, row.amplitude), xytext=offset, textcoords="offset points", fontsize=8, **alignment)
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.10 if snapshot_path is not None else 0.04)
    colorbar.set_label("f = A sin(B)")
    axis.set_xlabel("Phase B [rad]")
    axis.set_ylabel("Amplitude A")
    axis.set_title("Xopt/libEnsemble Sine Exploration")
    axis.grid(alpha=0.25)
    figure.savefig(output_path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Xopt sine parameter-space samples")
    parser.add_argument("--config", type=Path, default=Path("campaign.yaml"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("xopt_sine_parameter_space.png"))
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_root = args.run_root or Path(config["run_root"])
    results = load_results(run_root, len(config["initial_points"]))
    results_path = args.output.with_suffix(".csv")
    results.to_csv(results_path, index=False)
    snapshots = sorted((run_root / "surrogate_snapshots").glob("batch_*.npz"))
    snapshot_path = args.snapshot or (snapshots[-1] if snapshots else None)
    plot_results(results, args.output, snapshot_path)
    print(f"wrote {args.output} and {results_path}")


if __name__ == "__main__":
    main()