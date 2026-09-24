#!/usr/bin/env python3
"""Plot interpolated density fields and lineouts.

Usage (conda env inv-fbpic):

    conda run -n inv-fbpic python plot_density.py /path/to/400_um.h5

    conda run -n inv-fbpic python plot_density.py /path/to/400_um.h5 \\
        --plot callable_lineout

    conda run -n inv-fbpic python plot_density.py /path/to/400_um.h5 \\
        --plot xz_map --pressure 12.5 -o density_map.png
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

from interpolate_density import (
    DensityInterpolation,
    InterpolationMethod,
    build_density_callable,
    build_density_interpolation,
)

PlotMode = Literal["xz_map", "lineout", "callable_lineout"]
DensityFn = Callable[[np.ndarray, float, float], np.ndarray]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


@dataclass
class _LineoutSelection:
    pressure: float
    x_position: float


def _resolve_bounds(
    bounds: tuple[float, float] | None,
    extent: tuple[float, float],
    *,
    axis_name: str,
) -> tuple[float, float]:
    if bounds is None:
        return extent

    lower, upper = bounds
    valid_lower, valid_upper = extent
    if not np.isfinite((lower, upper)).all() or lower >= upper:
        raise ValueError(f"{axis_name} bounds must be finite and increasing")
    if lower < valid_lower or upper > valid_upper:
        raise ValueError(
            f"{axis_name} bounds [{lower}, {upper}] are outside the valid range "
            f"[{valid_lower}, {valid_upper}]"
        )
    return lower, upper


def _validate_position(
    value: float,
    bounds: tuple[float, float],
    *,
    axis_name: str,
) -> float:
    lower, upper = bounds
    if not lower <= value <= upper:
        raise ValueError(
            f"{axis_name} position {value} is outside the plot bounds "
            f"[{lower}, {upper}]"
        )
    return value


def _default_lineout_selection(field: DensityInterpolation) -> _LineoutSelection:
    return _LineoutSelection(
        pressure=float(field.pressure[0]),
        x_position=float(field.x[0]),
    )


def _lineout_title(
    field: DensityInterpolation,
    selection: _LineoutSelection,
    *,
    label: str = "",
) -> str:
    prefix = f"{label}@ " if label else "@ "
    return (
        f"{field.geometry} {prefix}x = {selection.x_position:.2f} mm, "
        f"{selection.pressure:.1f} bar ({field.method})"
    )


def _style_lineout_axes(ax: Axes) -> None:
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("density [1e18 cm^-3]")
    ax.grid()


def _create_lineout_figure(
    interactive: bool,
    ax: Axes | None,
) -> tuple[Figure, Axes, Axes | None, Axes | None]:
    if ax is not None:
        return ax.figure, ax, None, None

    if interactive:
        fig = plt.figure()
        grid = gridspec.GridSpec(3, 1, height_ratios=[20, 1, 1], hspace=0.45)
        return (
            fig,
            fig.add_subplot(grid[0]),
            fig.add_subplot(grid[1]),
            fig.add_subplot(grid[2]),
        )

    fig, plot_ax = plt.subplots()
    return fig, plot_ax, None, None


def _attach_pressure_x_sliders(
    fig: Figure,
    field: DensityInterpolation,
    selection: _LineoutSelection,
    *,
    pressure_slider_ax: Axes,
    x_slider_ax: Axes,
    x_bounds: tuple[float, float],
    on_change: Callable[[_LineoutSelection], None],
) -> None:
    p_min, p_max = field.pressure_extent
    x_min, x_max = x_bounds
    state = _LineoutSelection(
        pressure=selection.pressure,
        x_position=selection.x_position,
    )

    pressure_slider = Slider(
        pressure_slider_ax,
        "backing pressure [bar]",
        p_min,
        p_max,
        valinit=state.pressure,
    )
    x_slider = Slider(
        x_slider_ax,
        "x [mm]",
        x_min,
        x_max,
        valinit=state.x_position,
    )

    def on_pressure_change(pressure_value: float) -> None:
        state.pressure = pressure_value
        on_change(state)

    def on_x_change(x_value: float) -> None:
        state.x_position = x_value
        on_change(state)

    pressure_slider.on_changed(on_pressure_change)
    x_slider.on_changed(on_x_change)
    fig._pressure_slider = pressure_slider
    fig._x_slider = x_slider


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_density_interpolation(
    field: DensityInterpolation,
    *,
    pressure: float | None = None,
    x_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
    x_points: int = 400,
    z_points: int = 400,
    ax: Axes | None = None,
    title: str | None = None,
    interactive: bool = True,
) -> Figure:
    """Plot interpolated density in the (x, z) plane with a pressure slider."""
    resolved_x_bounds = _resolve_bounds(x_bounds, field.x_extent, axis_name="x")
    resolved_z_bounds = _resolve_bounds(z_bounds, field.z_extent, axis_name="z")
    x_values = np.linspace(*resolved_x_bounds, x_points)
    z_values = np.linspace(*resolved_z_bounds, z_points)
    z_plot = z_values * 1000
    p_min, p_max = field.pressure_extent

    if pressure is None:
        pressure = float(field.pressure[0])

    if ax is None:
        if interactive:
            fig = plt.figure()
            grid = gridspec.GridSpec(2, 1, height_ratios=[20, 1], hspace=0.35)
            ax = fig.add_subplot(grid[0])
            pressure_slider_ax = fig.add_subplot(grid[1])
        else:
            fig, ax = plt.subplots()
            pressure_slider_ax = None
    else:
        fig = ax.figure
        pressure_slider_ax = None

    if interactive:
        xz_stack = field.xz_grids_at_pressures(x_values, z_values)
        density_grid = field.interpolate_xz_from_pressure_stack(xz_stack, pressure)
    else:
        density_grid = field.interpolate_xz_grid(x_values, z_values, pressure)

    mesh = ax.pcolormesh(
        z_plot,
        x_values,
        density_grid,
        shading="auto",
        cmap="viridis",
    )
    colorbar = fig.colorbar(mesh, ax=ax, label="density [1e18 cm^-3]")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [mm]")

    auto_title = title is None
    if auto_title:
        title = f"{field.geometry} @ {pressure:.1f} bar ({field.method})"
    ax.set_title(title)

    if interactive:
        assert pressure_slider_ax is not None
        pressure_slider = Slider(
            pressure_slider_ax,
            "backing pressure [bar]",
            p_min,
            p_max,
            valinit=pressure,
        )

        def on_pressure_change(pressure_value: float) -> None:
            density = field.interpolate_xz_from_pressure_stack(
                xz_stack, pressure_value
            )
            mesh.set_array(density.ravel())
            mesh.set_clim(vmin=density.min(), vmax=density.max())
            colorbar.update_normal(mesh)
            if auto_title:
                ax.set_title(
                    f"{field.geometry} @ {pressure_value:.1f} bar ({field.method})"
                )
            fig.canvas.draw_idle()

        pressure_slider.on_changed(on_pressure_change)
        fig._pressure_slider = pressure_slider
    else:
        fig.tight_layout()

    return fig


def plot_density_lineout(
    field: DensityInterpolation,
    *,
    pressure: float | None = None,
    x_position: float | None = None,
    x_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
    density_fn: DensityFn | None = None,
    title_label: str = "",
    ax: Axes | None = None,
    title: str | None = None,
    interactive: bool = True,
) -> Figure:
    """Plot a density lineout along z with pressure and x sliders."""
    resolved_x_bounds = _resolve_bounds(x_bounds, field.x_extent, axis_name="x")
    resolved_z_bounds = _resolve_bounds(z_bounds, field.z_extent, axis_name="z")
    selection = _default_lineout_selection(field)
    if pressure is not None:
        selection = _LineoutSelection(pressure=pressure, x_position=selection.x_position)
    if x_position is not None:
        selection = _LineoutSelection(
            pressure=selection.pressure,
            x_position=_validate_position(x_position, resolved_x_bounds, axis_name="x"),
        )
    elif x_bounds is not None:
        selection = _LineoutSelection(
            pressure=selection.pressure,
            x_position=resolved_x_bounds[0],
        )

    evaluate = density_fn or field.interpolate_along_z
    z_values = np.linspace(*resolved_z_bounds, field.z.size)
    z_plot = z_values * 1000

    fig, plot_ax, pressure_slider_ax, x_slider_ax = _create_lineout_figure(
        interactive, ax
    )
    (line,) = plot_ax.plot(z_plot, evaluate(z_values, selection.x_position, selection.pressure))
    _style_lineout_axes(plot_ax)

    auto_title = title is None
    if auto_title:
        title = _lineout_title(field, selection, label=title_label)
    plot_ax.set_title(title)

    if interactive:
        assert pressure_slider_ax is not None
        assert x_slider_ax is not None

        def refresh(selection: _LineoutSelection) -> None:
            line.set_ydata(evaluate(z_values, selection.x_position, selection.pressure))
            plot_ax.relim()
            plot_ax.autoscale_view()
            if auto_title:
                plot_ax.set_title(_lineout_title(field, selection, label=title_label))
            fig.canvas.draw_idle()

        _attach_pressure_x_sliders(
            fig,
            field,
            selection,
            pressure_slider_ax=pressure_slider_ax,
            x_slider_ax=x_slider_ax,
            x_bounds=resolved_x_bounds,
            on_change=refresh,
        )
    else:
        fig.tight_layout()

    return fig


def plot_density_callable_lineout(
    hdf5_path: Path,
    *,
    field: DensityInterpolation,
    method: InterpolationMethod = "linear",
    pressure: float | None = None,
    x_position: float | None = None,
    x_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    interactive: bool = True,
) -> Figure:
    """Plot a density lineout along z using ``build_density_callable``."""
    def density_fn(
        z_values: np.ndarray, x_value: float, pressure_value: float
    ) -> np.ndarray:
        density_callable = build_density_callable(
            hdf5_path,
            pressure_value,
            x_value,
            method=method,
            field=field,
        )
        return density_callable(z_values, np.zeros_like(z_values))

    return plot_density_lineout(
        field,
        pressure=pressure,
        x_position=x_position,
        x_bounds=x_bounds,
        z_bounds=z_bounds,
        density_fn=density_fn,
        title_label="callable ",
        ax=ax,
        title=title,
        interactive=interactive,
    )


def _plot_from_cli(
    plot_mode: PlotMode,
    hdf5_path: Path,
    field: DensityInterpolation,
    *,
    method: InterpolationMethod,
    pressure: float | None,
    x_position: float | None,
    x_bounds: tuple[float, float] | None,
    z_bounds: tuple[float, float] | None,
    title: str | None,
    interactive: bool,
) -> Figure:
    if plot_mode == "xz_map":
        return plot_density_interpolation(
            field,
            pressure=pressure,
            x_bounds=x_bounds,
            z_bounds=z_bounds,
            title=title,
            interactive=interactive,
        )
    if plot_mode == "lineout":
        return plot_density_lineout(
            field,
            pressure=pressure,
            x_position=x_position,
            x_bounds=x_bounds,
            z_bounds=z_bounds,
            title=title,
            interactive=interactive,
        )
    return plot_density_callable_lineout(
        hdf5_path,
        field=field,
        method=method,
        pressure=pressure,
        x_position=x_position,
        x_bounds=x_bounds,
        z_bounds=z_bounds,
        title=title,
        interactive=interactive,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a density interpolation and plot it interactively or save a figure."
        )
    )
    parser.add_argument(
        "hdf5_path",
        type=Path,
        help="HDF5 density cube for one nozzle",
    )
    parser.add_argument(
        "--plot",
        choices=("xz_map", "lineout", "callable_lineout"),
        default="xz_map",
        help=(
            "Plot mode: xz_map shows density in the (x, z) plane vs pressure; "
            "lineout shows density along z vs pressure and x; "
            "callable_lineout is like lineout but uses build_density_callable"
        ),
    )
    parser.add_argument(
        "--method",
        choices=("linear", "cubic", "quintic", "pchip"),
        default="linear",
        help="2D interpolation method in the (x, pressure) plane",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=None,
        help="Backing pressure [bar] for static output (default: lowest in dataset)",
    )
    parser.add_argument(
        "--x",
        type=float,
        default=None,
        help="x position [mm] for static lineout output (default: lowest in dataset)",
    )
    parser.add_argument(
        "--x-bounds",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="x plot bounds [mm] (default: full dataset extent)",
    )
    parser.add_argument(
        "--z-bounds",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="z plot bounds [mm] (default: full dataset extent)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save the figure to this path instead of displaying interactively",
    )
    parser.add_argument(
        "-t",
        "--title",
        default=None,
        help="Figure title",
    )
    args = parser.parse_args()

    field = build_density_interpolation(args.hdf5_path, method=args.method)
    x_bounds = (
        (float(args.x_bounds[0]), float(args.x_bounds[1]))
        if args.x_bounds is not None
        else None
    )
    z_bounds = (
        (float(args.z_bounds[0]) / 1000, float(args.z_bounds[1]) / 1000)
        if args.z_bounds is not None
        else None
    )
    fig = _plot_from_cli(
        args.plot,
        args.hdf5_path,
        field,
        method=args.method,
        pressure=args.pressure,
        x_position=args.x,
        x_bounds=x_bounds,
        z_bounds=z_bounds,
        title=args.title,
        interactive=args.output is None,
    )

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Wrote {args.output.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
