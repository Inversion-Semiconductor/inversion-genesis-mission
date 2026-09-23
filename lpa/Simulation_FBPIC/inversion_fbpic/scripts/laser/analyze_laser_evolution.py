"""CLI front-end for the laser-evolution analysis.

This script parses command-line arguments, delegates the numerical work
to :mod:`inversion_fbpic.utils.field_analysis`, and produces plots:

  * Per-iteration ``imshow`` panels of the bandpassed and analytic-signal
    envelope energy densities (with the location where ``a0`` was
    sampled marked), shown only when ``--show-plots`` is set.
  * A summary energy / ``a0`` vs. ``z`` plot saved to disk.

For documentation of the underlying analysis (laser-centred bandpass,
cylindrical-symmetry energy integration, analytic-signal envelope,
engineering ``a0``) see
:func:`inversion_fbpic.utils.field_analysis.analyze_iteration`.

Usage:
    python analyze_laser_evolution.py [OPTIONS]

    Optional Arguments:
        -d, --series-path PATH
            Path to the OpenPMD HDF5 time series directory (or its parent
            -- an ``hdf5`` subdirectory will be tried automatically).
            Default: ./diags/hdf5

        -w, --laser-wavelength FLOAT
            Laser central wavelength in metres.
            Default: 0.8e-6

        --band-half-width-frac FLOAT
            Half-width of the laser-centred bandpass as a fraction of the
            laser wavenumber. With the default ``0.5`` the window spans
            ``[0.5*k_laser, 1.5*k_laser]``.
            Default: 0.5

        --rmax-window FLOAT
            Limit the cylindrical-symmetry energy integration to ``|r| <
            rmax-window`` (m). ``None`` -> full radial extent.
            Default: None

        --on-axis-rmax FLOAT
            Half-width (m) of the near-axis region used to find the
            longitudinal peak of the laser envelope.
            Default: 5e-6

        --energy-threshold FLOAT
            Drop iterations whose energy is below this value (J).
            Default: 1e-4

        --start-iteration INT
            First iteration to process (inclusive).
            Default: None

        --end-iteration INT
            Last iteration to process (inclusive).
            Default: None

        --results-file PATH
            Path to save the per-iteration results as JSON.
            Default: laser_evolution_results.json

        --output-plot PATH
            Path to save the energy / a0 vs. z summary plot.
            Default: laser_evolution_plot.png

        --show-plots
            Show per-iteration imshow plots.

        --log
            Use a logarithmic colour scale on per-iteration imshow plots
            (requires --show-plots).

        -j, --from-json PATH
            Skip the analysis entirely and regenerate the summary plot
            from a previously written results JSON file. All analysis-
            related arguments are ignored when this is set.

    Examples:
        # Run the full analysis (uses ./diags/hdf5/)
        python analyze_laser_evolution.py

        # Point at a specific diagnostics directory
        python analyze_laser_evolution.py -d /path/to/sim/diags/hdf5

        # Per-iteration plots with a log colour scale
        python analyze_laser_evolution.py --show-plots --log

        # Re-plot from a saved JSON file
        python analyze_laser_evolution.py --from-json laser_evolution_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from inversion_fbpic.utils.argparse_utils import float_or_none
from inversion_fbpic.utils.field_analysis import analyze_laser_evolution


DEFAULT_SERIES_PATH: Path = Path("./diags/hdf5")
DEFAULT_LASER_WAVELENGTH: float = 0.8e-6  # m
DEFAULT_BAND_HALF_WIDTH_FRAC: float = 0.5  # fraction of laser wavenumber
DEFAULT_RMAX_WINDOW: float | None = None  # m
DEFAULT_ON_AXIS_RMAX: float = 5e-6  # m
DEFAULT_ENERGY_THRESHOLD: float = 1e-4  # J
DEFAULT_RESULTS_FILE: Path = Path("laser_evolution_results.json")
DEFAULT_OUTPUT_PLOT: Path = Path("laser_evolution_plot.png")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the analysis script.

    Returns:
        Populated :class:`argparse.Namespace` after validation. Any
        validation failure raises ``SystemExit`` via
        :meth:`argparse.ArgumentParser.error`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Track laser energy and a0 through an FBPIC simulation by reading "
            "EM fields from an OpenPMD time series."
        )
    )

    parser.add_argument(
        "-d",
        "--series-path",
        type=Path,
        default=DEFAULT_SERIES_PATH,
        help="Path to the OpenPMD HDF5 time series (or its parent directory)",
    )
    parser.add_argument(
        "-w",
        "--laser-wavelength",
        type=float,
        default=DEFAULT_LASER_WAVELENGTH,
        help="Central laser wavelength in metres",
    )
    parser.add_argument(
        "--band-half-width-frac",
        type=float,
        default=DEFAULT_BAND_HALF_WIDTH_FRAC,
        help=(
            "Half-width of the laser-centred bandpass as a fraction of the "
            "laser wavenumber (default 0.5 -> window spans "
            "[0.5, 1.5] * k_laser)"
        ),
    )
    parser.add_argument(
        "--rmax-window",
        type=float_or_none,
        default=DEFAULT_RMAX_WINDOW,
        help="Radial limit for the energy integration in metres. Use 'None' for full range.",
    )
    parser.add_argument(
        "--on-axis-rmax",
        type=float,
        default=DEFAULT_ON_AXIS_RMAX,
        help="Half-width of the on-axis region used to locate the laser peak (m)",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=DEFAULT_ENERGY_THRESHOLD,
        help="Skip iterations whose computed laser energy falls below this value (J)",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=None,
        help="First iteration to process (inclusive)",
    )
    parser.add_argument(
        "--end-iteration",
        type=int,
        default=None,
        help="Last iteration to process (inclusive)",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="Path to save the per-iteration results JSON file",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=DEFAULT_OUTPUT_PLOT,
        help="Path to save the energy/a0 vs. z summary plot",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show per-iteration energy-density imshow with the a0 sampling location",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use logarithmic colour scale on per-iteration imshow plots (requires --show-plots)",
    )
    parser.add_argument(
        "-j",
        "--from-json",
        type=Path,
        default=None,
        help=(
            "Skip analysis and regenerate the summary plot from a previously "
            "written JSON results file"
        ),
    )

    args = parser.parse_args()
    if args.band_half_width_frac <= 0:
        parser.error("--band-half-width-frac must be positive")
    if args.on_axis_rmax <= 0:
        parser.error("--on-axis-rmax must be positive")
    if args.laser_wavelength <= 0:
        parser.error("--laser-wavelength must be positive")
    return args


def _imshow_norm(field: np.ndarray, log_scale: bool) -> LogNorm | None:
    """Returns a matplotlib colour norm for one of the imshow panels.

    Args:
        field: 2D field that will be passed to ``imshow``; used to
            derive sensible log-scale limits.
        log_scale: If ``True``, return a :class:`~matplotlib.colors.LogNorm`
            covering ``[vmin, vmax]`` with ``vmin`` clamped above the
            smallest positive value (and at least ``vmax / 1e6``). If
            ``False``, return ``None`` so ``imshow`` uses its default
            linear scale.

    Returns:
        A :class:`~matplotlib.colors.LogNorm` instance if ``log_scale``
        is ``True``, otherwise ``None``.
    """
    if not log_scale:
        return None
    positive = field[field > 0]
    vmax = float(np.max(field))
    if positive.size == 0 or vmax <= 0:
        print("Skipping log scale (no positive data)")
        return None
    vmin = max(float(positive.min()), vmax / 1e6)
    return LogNorm(vmin=vmin, vmax=vmax)


def _plot_iteration(record: dict[str, Any], *, log_scale: bool = False) -> None:
    """Shows the per-iteration energy-density panels with the ``a0`` marker.

    Args:
        record: Per-iteration record as returned by
            :func:`inversion_fbpic.utils.field_analysis.analyze_iteration`.
            Must contain both ``"summary"`` and ``"fields"`` sub-dicts.
        log_scale: If ``True``, use a logarithmic colour scale on both
            panels.
    """
    summary = record["summary"]
    fields = record["fields"]

    laser_energy_density = fields["laser_energy_density"]
    envelope_energy_density = fields["envelope_energy_density"]
    z_axis = fields["z_axis"]
    r_axis = fields["r_axis"]
    z_peak_idx = fields["z_peak_idx"]
    r_peak_idx = fields["r_peak_idx"]

    extent_um = (
        float(z_axis[0]) * 1e6,
        float(z_axis[-1]) * 1e6,
        float(r_axis[0]) * 1e6,
        float(r_axis[-1]) * 1e6,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    im0 = axes[0].imshow(
        laser_energy_density,
        extent=extent_um,
        origin="lower",
        aspect="auto",
        cmap="inferno",
        norm=_imshow_norm(laser_energy_density, log_scale),
    )
    axes[0].set_title("Bandpassed energy density (oscillating)")
    fig.colorbar(im0, ax=axes[0], label=r"$u$ (J/m$^3$)")

    im1 = axes[1].imshow(
        envelope_energy_density,
        extent=extent_um,
        origin="lower",
        aspect="auto",
        cmap="inferno",
        norm=_imshow_norm(envelope_energy_density, log_scale),
    )
    axes[1].set_title("Envelope energy density (analytic signal)")
    fig.colorbar(im1, ax=axes[1], label=r"$u_{env}$ (J/m$^3$)")

    for ax in axes:
        ax.scatter(
            [z_axis[z_peak_idx] * 1e6],
            [r_axis[r_peak_idx] * 1e6],
            marker="x",
            color="cyan",
            s=120,
            linewidths=2,
            label=rf"$a_0$ = {summary['a0']:.3f}",
        )
        ax.set_xlabel(r"$z$ ($\mu$m)")
        ax.legend(loc="upper right")

    axes[0].set_ylabel(r"$r$ ($\mu$m)")
    fig.suptitle(
        f"Iteration {summary['iteration']} | "
        f"$E_{{laser}}$ = {summary['energy_j'] * 1e3:.2f} mJ | "
        f"$z_{{peak}}$ = {z_axis[z_peak_idx] * 1e3:.3f} mm"
    )
    fig.tight_layout()
    plt.show()


def _plot_evolution(iterations: list[dict[str, Any]], output_path: Path) -> None:
    """Saves and shows the laser energy / ``a0`` vs. ``z`` summary plot.

    Args:
        iterations: List of per-iteration summary dicts (as stored under
            the ``"iterations"`` key of the JSON results file). Each
            entry must contain ``z_peak_m``, ``energy_j`` and ``a0``.
        output_path: Filesystem path where the PNG plot is written.
            Parent directories are created if necessary.
    """
    if not iterations:
        print("No valid results to plot; skipping summary plot.")
        return

    z_mm = np.array([r["z_peak_m"] for r in iterations]) * 1e3
    energy_mj = np.array([r["energy_j"] for r in iterations]) * 1e3
    a0 = np.array([r["a0"] for r in iterations])

    order = np.argsort(z_mm)
    z_mm = z_mm[order]
    energy_mj = energy_mj[order]
    a0 = a0[order]

    fig, (ax_energy, ax_a0) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_energy.plot(z_mm, energy_mj, "o-", color="#2E86AB", linewidth=2, markersize=5)
    ax_energy.set_ylabel("Laser energy (mJ)", fontsize=13)
    ax_energy.grid(True, linestyle="--", alpha=0.4)

    ax_a0.plot(z_mm, a0, "o-", color="#E07A5F", linewidth=2, markersize=5)
    ax_a0.set_xlabel("Laser-peak lab-frame $z$ (mm)", fontsize=13)
    ax_a0.set_ylabel(r"$a_0$", fontsize=13)
    ax_a0.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Laser pulse evolution across the plasma target", fontsize=15)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved summary plot to: {output_path}")
    plt.show()
    plt.close()


def process(args: argparse.Namespace) -> None:
    """Replays a JSON file or runs the analysis end-to-end.

    When ``args.from_json`` is set, the JSON file is loaded and only the
    summary plot is regenerated; all other analysis arguments are
    ignored. Otherwise the full analysis runs via
    :func:`inversion_fbpic.utils.field_analysis.analyze_laser_evolution`,
    optionally with a per-iteration plotting callback.

    Args:
        args: Parsed CLI arguments produced by :func:`parse_args`.
    """
    if args.from_json is not None:
        json_path = Path(args.from_json)
        print(f"Loading results from: {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            iterations = data.get("iterations", [])
        else:
            iterations = data  # legacy flat-list format
        _plot_evolution(iterations, args.output_plot)
        return

    on_iteration: Callable[[dict[str, Any]], None] | None
    if args.show_plots:

        def on_iteration(record: dict[str, Any]) -> None:
            _plot_iteration(record, log_scale=args.log)

    else:
        on_iteration = None

    output = analyze_laser_evolution(
        args.series_path,
        laser_wavelength=args.laser_wavelength,
        band_half_width_frac=args.band_half_width_frac,
        rmax_window=args.rmax_window,
        on_axis_rmax=args.on_axis_rmax,
        energy_threshold=args.energy_threshold,
        start_iteration=args.start_iteration,
        end_iteration=args.end_iteration,
        results_file=args.results_file,
        on_iteration=on_iteration,
        verbose=True,
    )

    _plot_evolution(output["iterations"], args.output_plot)


def main() -> None:
    """Script entry point: parses CLI arguments and runs the analysis."""
    args = parse_args()
    process(args)


if __name__ == "__main__":
    main()
