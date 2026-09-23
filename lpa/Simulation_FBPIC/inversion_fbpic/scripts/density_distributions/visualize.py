"""Visualization tools for FBPIC density profiles.

This module provides functions for visualizing density profiles used in FBPIC
simulations, including 2D plots and on-axis lineouts.

Typical usage example:
    plot_2d_density_profile(
        z_start=0e-3,
        z_end=4e-3,
        r_width=100e-6,
        density_function=my_density_func,
        units="mm",
        num=300,
    )
"""

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_2d_density_profile(
    z_start: float,
    z_end: float,
    r_width: float,
    density_function: Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike],
    peak_density_value: float = 1,
    units: str = "m",
    num: int = 100,
) -> None:
    """Plots a 2D density profile with an on-axis lineout.

    This function creates a visualization of a density profile over a specified
    domain, showing both the 2D distribution and a lineout along the r=0 axis.

    Args:
        z_start: Left (upstream) extent of domain in meters.
        z_end: Right (downstream) extent of domain in meters.
        r_width: Width of domain in meters, centered at r=0.
        density_function: Callable function that accepts (z, r) coordinates and
            returns density values. The function should accept numpy arrays and
            return a numpy array of the same shape.
        peak_density_value (float): value to normalize the peak to, defaults to 1
        units: Scale of the plot's axes. Can be "m", "mm", or "um".
            Defaults to "m".
        num: Number of points in the longitudinal axis for plotting.
            Defaults to 100.

    Returns:
        None. Displays the plot using matplotlib.
    """
    z_arr = np.linspace(z_start, z_end, num)
    r_arr = np.linspace(-r_width / 2, r_width / 2, 10)
    z_mesh, r_mesh = np.meshgrid(z_arr, r_arr)

    den_arr = np.vectorize(density_function)(z_mesh, r_mesh)
    den_on_axis = density_function(z_arr, r=0)

    # Calculate radial lineout at central z position
    z_center = (z_start + z_end) / 4 * 3
    r_arr_lineout = np.linspace(-r_width / 2, r_width / 2, num)
    den_radial = np.array([density_function(z_center, r) for r in r_arr_lineout])

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 5))

    if units == "mm":
        z_start_plot = z_start * 1e3
        z_end_plot = z_end * 1e3
        r_width_plot = r_width * 1e3
        z_arr_plot = z_arr * 1e3
        r_arr_lineout_plot = r_arr_lineout * 1e3
    elif units == "um":
        z_start_plot = z_start * 1e6
        z_end_plot = z_end * 1e6
        r_width_plot = r_width * 1e6
        z_arr_plot = z_arr * 1e6
        r_arr_lineout_plot = r_arr_lineout * 1e6
    else:
        z_start_plot = z_start
        z_end_plot = z_end
        r_width_plot = r_width
        z_arr_plot = z_arr
        r_arr_lineout_plot = r_arr_lineout

    # Left plot: 2D density profile with on-axis lineout
    ax1.imshow(
        den_arr,
        extent=(z_start_plot, z_end_plot, -r_width_plot / 2, r_width_plot / 2),
        aspect="auto",
        origin="lower",
    )
    ax1.set_ylabel(f"r ({units})")
    ax1.set_xlabel(f"z ({units})")
    ax1.set_title("2D Density Profile")

    ax2 = ax1.twinx()
    ax2.plot(z_arr_plot, den_on_axis * peak_density_value, c="r", ls="--")
    ax2.set_ylabel("On-Axis Density (rel.)", c="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # Right plot: Radial lineout at central z position
    ax3.plot(r_arr_lineout_plot, den_radial * peak_density_value, c="b", lw=2)
    ax3.set_xlabel(f"r ({units})")
    ax3.set_ylabel("Density (rel.)", c="b")
    ax3.set_title(
        f"Radial Lineout at z = {(z_start_plot + z_end_plot) *3/4:.2f} {units}"
    )
    ax3.tick_params(axis="y", labelcolor="b")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_iterative_1d_lineout(
    z_start: float,
    z_end: float,
    density_function: Callable[[npt.ArrayLike, npt.ArrayLike], npt.ArrayLike],
    color: str,
    line_style: str,
    fill_color: Optional[str] = None,
    label: Optional[str] = None,
    peak_density_value: float = 1,
    units: str = "m",
    num: int = 200,
    figure: Optional[plt.Figure] = None,
    show_plot: bool = False,
) -> Optional[plt.Figure]:
    if figure is None:
        figure = plt.figure(figsize=(7, 5))
    else:
        plt.figure(figure.number)

    z_arr = np.linspace(z_start, z_end, num)
    z_arr_plot = z_arr * 1e3
    den_on_axis = density_function(z=z_arr, r=0)

    plt.plot(
        z_arr_plot,
        den_on_axis * peak_density_value,
        c=color,
        ls=line_style,
        label=label,
    )

    if fill_color is not None:
        plt.fill_between(
            z_arr_plot, den_on_axis * peak_density_value, 0, color=fill_color, alpha=0.5
        )

    if show_plot:
        plt.xlabel(units)
        plt.ylabel("Density ($cm^{-3}$)")
        plt.legend()
        plt.xlim(z_start * 1e3, z_end * 1e3)
        plt.ylim(bottom=0)
        plt.show()

        return None

    else:
        return figure


def plot_laser_focal_plane(
    laser_focal_position: float,
    label: str = "Laser Focal Plane",
) -> None:
    """Add a vertical marker for the laser focal plane to a 1D lineout plot."""
    plt.axvline(laser_focal_position * 1e3, c="k", ls="--", label=label)
