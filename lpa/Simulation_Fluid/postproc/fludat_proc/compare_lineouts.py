#!/usr/bin/env python3
"""Compare density lineouts along ``z`` across several density cubes.

One figure is written per height ``x`` above the nozzle. Each figure has one panel per
backing pressure and one line per cube. Lineouts are read at the cubes' tabulated
pressures (no interpolation in pressure); a cube that lacks a requested pressure is
left out of that panel.

Examples (conda env inv-fbpic):

    python -m fludat_proc.compare_lineouts data/density_field/htu_dens_7_0.h5 \\
        data/htu_zeus/htu_short_density.h5 -o data/lineout_comparison
    python -m fludat_proc.compare_lineouts a.h5 b.h5 --labels "CFD" "Measured" \\
        --x 0.5 2 --pressure 15 35 --yscale log -o out_dir
"""

from __future__ import annotations

if __package__ in (
    None,
    "",
):  # run as a plain script, e.g. `python fludat_proc/compare_lineouts.py`
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fludat_proc"

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .common import MM_PER_M  # noqa: E402
from .density_cube import DensityCube, load_density_cube  # noqa: E402

DEFAULT_X_MM = (0.5, 1.0, 2.0, 3.0, 5.0)
DEFAULT_PRESSURES_BAR = (15.0, 35.0, 45.0)

# Categorical slots 1-4 of the reference palette, in fixed order. Orange and yellow are
# not separable by hue for every viewer, so each cube also gets its own line style.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SERIES_LINESTYLES = ("-", "--", "-.", ":")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def lineout_at(
    cube: DensityCube, x_mm: float, pressure_bar: float
) -> np.ndarray | None:
    """Return the density along ``z`` at ``x_mm`` and a tabulated pressure, else ``None``.

    ``None`` means the cube has no such pressure or does not reach ``x_mm``.
    """
    pressure_index = np.flatnonzero(np.isclose(cube.pressure, pressure_bar))
    x_min, x_max = cube.x_extent
    if pressure_index.size == 0 or not x_min <= x_mm <= x_max:
        return None
    upper = int(np.searchsorted(cube.x, x_mm, side="left"))
    plane = cube.density[:, :, pressure_index[0]]
    if upper == 0:
        return plane[:, 0].copy()
    weight = (x_mm - cube.x[upper - 1]) / (cube.x[upper] - cube.x[upper - 1])
    return (1.0 - weight) * plane[:, upper - 1] + weight * plane[:, upper]


def _style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)


def plot_lineout_comparison(
    cubes: list[DensityCube],
    labels: list[str],
    x_mm: float,
    pressures_bar: tuple[float, ...],
    *,
    yscale: str = "linear",
) -> Figure:
    """One figure at height ``x_mm``: a panel per pressure, a line per cube."""
    fig, axes = plt.subplots(
        1,
        len(pressures_bar),
        figsize=(4.6 * len(pressures_bar), 4.4),
        sharex=True,
        squeeze=False,
        facecolor=SURFACE,
    )
    units = cubes[0].density_units
    for ax, pressure in zip(axes[0], pressures_bar):
        _style_axes(ax)
        missing: list[str] = []
        positive_max = 0.0
        for index, (cube, label) in enumerate(zip(cubes, labels)):
            density = lineout_at(cube, x_mm, pressure)
            if density is None:
                missing.append(label)
                continue
            if yscale == "log":
                density = np.where(density > 0, density, np.nan)
            positive_max = max(positive_max, float(np.nanmax(density)))
            ax.plot(
                cube.z * MM_PER_M,
                density,
                color=SERIES_COLORS[index],
                linestyle=SERIES_LINESTYLES[index],
                linewidth=2,
                label=label,
            )
        ax.set_yscale(yscale)
        if yscale == "log" and positive_max > 0:
            ax.set_ylim(positive_max * 1e-3, positive_max * 2)
        ax.set_title(f"{pressure:g} bar", color=INK, fontsize=11, loc="left", pad=16)
        ax.set_xlabel("z [mm]", color=INK_SECONDARY)
        if missing:
            ax.text(
                0.03,
                0.97,
                "no data: " + ", ".join(missing),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color=INK_SECONDARY,
            )
    axes[0][0].set_ylabel(f"density [{units}]", color=INK_SECONDARY)
    handles, names = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for handle, name in zip(*ax.get_legend_handles_labels()):
            if name not in names:
                handles.append(handle)
                names.append(name)
    fig.legend(
        handles,
        names,
        loc="lower center",
        ncol=len(names),
        frameon=False,
        labelcolor=INK,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Density lineouts along z at x = {x_mm:g} mm above nozzle",
        color=INK,
        fontsize=13,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig


def default_label(path: Path) -> str:
    return path.stem.removesuffix("_density")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save lineout comparison figures for several density cubes."
    )
    parser.add_argument(
        "hdf5_paths", nargs="+", type=Path, help="HDF5 density cubes (at most 4)"
    )
    parser.add_argument(
        "--labels", nargs="+", default=None, help="Legend labels (default: file stems)"
    )
    parser.add_argument(
        "--x",
        nargs="+",
        type=float,
        default=list(DEFAULT_X_MM),
        metavar="MM",
        help="Heights above the nozzle [mm]; one figure each",
    )
    parser.add_argument(
        "--pressure",
        nargs="+",
        type=float,
        default=list(DEFAULT_PRESSURES_BAR),
        metavar="BAR",
        help="Backing pressures [bar]; one panel each (must be tabulated in the cubes)",
    )
    parser.add_argument("--yscale", choices=("linear", "log"), default="linear")
    parser.add_argument(
        "-o", "--output-dir", type=Path, required=True, help="Directory for the PNGs"
    )
    args = parser.parse_args()

    if len(args.hdf5_paths) > len(SERIES_COLORS):
        parser.error(f"at most {len(SERIES_COLORS)} cubes can be compared")
    labels = args.labels or [default_label(path) for path in args.hdf5_paths]
    if len(labels) != len(args.hdf5_paths):
        parser.error("--labels must match the number of cubes")

    cubes = [load_density_cube(path) for path in args.hdf5_paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for x_mm in args.x:
        fig = plot_lineout_comparison(
            cubes, labels, x_mm, tuple(args.pressure), yscale=args.yscale
        )
        output = args.output_dir / f"lineouts_x_{x_mm:g}_mm.png"
        fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
