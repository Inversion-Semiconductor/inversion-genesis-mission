"""Figures for model comparisons."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..plotting import DATA_COLOR, SERIES_COLORS
from .evaluation import ModelComparison, ModelEvaluation

OBJECTIVE_COLORS = {"direct": SERIES_COLORS[0], "indirect": SERIES_COLORS[1]}


def plot_model_comparison(
    comparison: ModelComparison, *, title: str | None = None
) -> Figure:
    """NRMSE of predicted fits per model and objective, plus prediction vs target.

    Left: one horizontal box per (model, objective), best first, with the local
    fit's NRMSE as the reference row. Right: predicted against locally
    optimised unit-cube parameters for the best evaluation with targets.
    """
    ranked = [e for e in comparison.ranked() if not e.error]
    reference = comparison.training_set.reference_nrmse
    reference = reference[np.isfinite(reference)]

    fig = plt.figure(figsize=(13.0, 6.5))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.25, 1.0], wspace=0.35, left=0.2, right=0.97
    )
    ax_box = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[0, 1])

    rows = [f"{e.model_name} / {e.objective_name}" for e in ranked]
    data = [e.nrmse[np.isfinite(e.nrmse)] for e in ranked]
    colors = [OBJECTIVE_COLORS.get(e.objective_name, SERIES_COLORS[2]) for e in ranked]
    if reference.size:
        rows.append("local fit (reference)")
        data.append(reference)
        colors.append(DATA_COLOR)
    positions = np.arange(len(rows))
    boxes = ax_box.boxplot(
        data,
        positions=positions,
        orientation="horizontal",
        widths=0.55,
        patch_artist=True,
        medianprops={"color": DATA_COLOR, "lw": 1.2},
        flierprops={"marker": ".", "markersize": 4},
    )
    for patch, whisker_color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(whisker_color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(whisker_color)
    ax_box.set_yticks(positions)
    ax_box.set_yticklabels(rows, fontsize=8)
    ax_box.invert_yaxis()
    ax_box.set_xscale("log")
    ax_box.set_xlabel("NRMSE of the predicted parameters' fit")
    ax_box.set_title(
        f"{comparison.protocol}, {len(comparison.training_set)} samples", fontsize=10
    )

    best = next((e for e in ranked if e.parameter_error is not None), None)
    if best is not None:
        targets = best.test_set.targets
        for k, name in enumerate(best.test_set.space.names):
            ax_scatter.scatter(
                targets[:, k],
                best.predictions[:, k],
                s=16,
                color=SERIES_COLORS[k % len(SERIES_COLORS)],
                label=name,
                edgecolors="white",
                linewidths=0.5,
            )
        ax_scatter.plot([0, 1], [0, 1], color=DATA_COLOR, lw=0.8, alpha=0.5)
        ax_scatter.set_xlim(0, 1)
        ax_scatter.set_ylim(0, 1)
        ax_scatter.set_xlabel("local optimum (unit cube)")
        ax_scatter.set_ylabel("prediction (unit cube)")
        ax_scatter.set_title(f"{best.model_name} / {best.objective_name}", fontsize=10)
        ax_scatter.legend(fontsize=7.5, frameon=False, loc="upper left")
    else:
        ax_scatter.axis("off")
        ax_scatter.text(0.5, 0.5, "no local-fit targets", ha="center", va="center")

    for ax in (ax_box, ax_scatter):
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        title
        or f"{comparison.training_set.name}: {comparison.training_set.family.name} "
        "conditions -> parameters",
        fontsize=12,
    )
    return fig


def plot_training_history(evaluation: ModelEvaluation, model) -> Figure:
    """Loss per optimiser call for a model exposing a ``history`` list."""
    history = getattr(model, "history", None)
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    if history:
        ax.plot(
            history,
            color=OBJECTIVE_COLORS.get(evaluation.objective_name, SERIES_COLORS[0]),
            lw=1.5,
        )
        ax.set_yscale("log")
    ax.set_xlabel("objective call")
    ax.set_ylabel(f"{evaluation.objective_name} loss")
    ax.set_title(f"{evaluation.model_name} training", fontsize=10)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    return fig
