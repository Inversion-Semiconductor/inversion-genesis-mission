"""Plot a lineout against its best-fitting profile families."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .fitting import FitResult
from .lineout import Lineout

MM_PER_M = 1000.0
DATA_COLOR = "#0b0b0b"
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
"""Fixed categorical order: rank 1 is always blue, rank 2 orange, and so on."""


def plot_fit_comparison(
    lineout: Lineout,
    results: Sequence[FitResult],
    *,
    top: int = 3,
    title: str | None = None,
    n_model_points: int = 1000,
) -> Figure:
    """Data with the ``top`` ranked model curves above their residuals.

    ``results`` should already be ranked best-first (as in
    :class:`~fludat_fit.fitting.FamilyComparison`).
    """
    shown = list(results[: max(1, min(top, len(SERIES_COLORS)))])
    z_fine = np.linspace(lineout.z[0], lineout.z[-1], n_model_points)
    z_mm = lineout.z * MM_PER_M

    fig, (ax_fit, ax_res) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8.0, 6.0),
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.06},
    )
    ax_fit.plot(
        z_mm,
        lineout.density,
        color=DATA_COLOR,
        lw=1.2,
        alpha=0.85,
        label=f"lineout ({lineout.conditions.label()})",
    )
    for color, result in zip(SERIES_COLORS, shown, strict=False):
        model_fine = result.model_density(z_fine)
        label = f"{result.family_name}  (NRMSE {result.goodness.nrmse:.3g})"
        ax_fit.plot(z_fine * MM_PER_M, model_fine, color=color, lw=2.0, label=label)
        ax_res.plot(
            z_mm,
            (result.model_density() - lineout.density) / lineout.peak,
            color=color,
            lw=1.5,
        )
    ax_res.axhline(0.0, color=DATA_COLOR, lw=0.8, alpha=0.4)

    ax_fit.set_ylabel(f"density [{lineout.density_units}]")
    ax_res.set_ylabel("residual / peak")
    ax_res.set_xlabel("z [mm]")
    for ax in (ax_fit, ax_res):
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    ax_fit.legend(loc="upper left", fontsize=8, frameon=False)
    ax_fit.set_title(
        title
        if title is not None
        else f"{lineout.source or 'lineout'}: best {len(shown)} of {len(results)} families"
    )
    fig.align_ylabels()
    return fig
