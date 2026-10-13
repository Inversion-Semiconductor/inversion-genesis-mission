#!/usr/bin/env python3
"""Fit a super-Gaussian and a Lorentzian across a feature of an Abel-inverted profile.

Pools several neighboring transverse lineouts (a window of z slices around the
center of the profile, by default) into one fitting dataset, so the fit is
less sensitive to the noise in any single lineout than fitting one slice alone.

``abel_invert`` output files carry no pixel-pitch calibration, so positions
here are in raw pixels, not physical length.

Example (conda env inv-fbpic):

    python -m fludat_proc.fit_profile data/U_HasoLift_average_processed_abel.h5 \\
        --plot-output data/U_HasoLift_average_processed_lineout_fit.png
"""

from __future__ import annotations

if __package__ in (
    None,
    "",
):  # run as a plain script, e.g. `python fludat_proc/fit_profile.py`
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fludat_proc"

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

DEFAULT_Z_HALF_WIDTH_PX = 50.0


def super_gaussian(
    r: np.ndarray, amplitude: float, width: float, order: float, offset: float
) -> np.ndarray:
    return offset + amplitude * np.exp(-np.abs(r / width) ** order)


def lorentzian(
    r: np.ndarray, amplitude: float, width: float, offset: float
) -> np.ndarray:
    return offset + amplitude / (1 + (r / width) ** 2)


def load_lineout_window(
    path: Path,
    *,
    z_px: float | None = None,
    z_half_width_px: float = DEFAULT_Z_HALF_WIDTH_PX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(x_px, window_data, z_window, z_center)`` for a z window of lineouts.

    ``window_data`` has shape ``(n_x, n_z_window)``: each column is one
    transverse lineout. ``z_px=None`` (default) centers the window at the
    middle of the z range. Abel inversion amplifies noise most strongly near
    the symmetry axis, so the center is not chosen by searching for the
    largest-magnitude on-axis value: that tends to land on single-pixel noise
    spikes rather than the real profile (visible as a spiky region in the
    on-axis trace at large z for this dataset).
    """
    with h5py.File(path, "r") as f:
        phase_inverted = f["phase_inverted"][()]
        x_px = f["x_px"][()]
        z_px_axis = f["z_px"][()]

    z_center = float(z_px_axis[len(z_px_axis) // 2]) if z_px is None else z_px
    in_window = np.abs(z_px_axis - z_center) <= max(z_half_width_px, 0.0)
    if not in_window.any():
        in_window[int(np.argmin(np.abs(z_px_axis - z_center)))] = True
    window_data = phase_inverted[:, in_window]
    z_window = z_px_axis[in_window]
    return x_px, window_data, z_window, z_center


def pool_lineouts(
    x_px: np.ndarray, window_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a ``(n_x, n_z)`` window of lineouts into ``(r_pooled, value_pooled)``."""
    n_z = window_data.shape[1]
    r_pooled = np.tile(x_px, n_z)
    value_pooled = window_data.T.ravel()
    return r_pooled, value_pooled


def _estimate_offset(lineout: np.ndarray, *, edge_fraction: float = 0.1) -> float:
    n_edge = max(1, int(len(lineout) * edge_fraction))
    return float(np.mean(np.concatenate([lineout[:n_edge], lineout[-n_edge:]])))


def _estimate_width(
    x_px: np.ndarray, lineout: np.ndarray, offset: float, amplitude: float
) -> float:
    above_half = np.abs(lineout - offset) >= np.abs(amplitude / 2)
    indices = np.flatnonzero(above_half)
    if indices.size < 2:
        return float((x_px[-1] - x_px[0]) / 4)
    half_width = (x_px[indices[-1]] - x_px[indices[0]]) / 2
    return float(max(half_width, (x_px[1] - x_px[0])))


def _initial_guess(
    x_px: np.ndarray, mean_profile: np.ndarray
) -> tuple[float, float, float]:
    center = len(mean_profile) // 2
    offset0 = _estimate_offset(mean_profile)
    amplitude0 = mean_profile[center] - offset0
    width0 = _estimate_width(x_px, mean_profile, offset0, amplitude0)
    return amplitude0, width0, offset0


def fit_super_gaussian(
    r_pooled: np.ndarray,
    value_pooled: np.ndarray,
    *,
    x_px: np.ndarray,
    mean_profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude0, width0, offset0 = _initial_guess(x_px, mean_profile)
    p0 = [amplitude0, width0, 2.0, offset0]
    bounds = ([-np.inf, 1e-6, 0.2, -np.inf], [np.inf, np.inf, 20.0, np.inf])
    popt, pcov = curve_fit(
        super_gaussian, r_pooled, value_pooled, p0=p0, bounds=bounds, maxfev=20000
    )
    return popt, pcov


def fit_lorentzian(
    r_pooled: np.ndarray,
    value_pooled: np.ndarray,
    *,
    x_px: np.ndarray,
    mean_profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude0, width0, offset0 = _initial_guess(x_px, mean_profile)
    p0 = [amplitude0, width0, offset0]
    bounds = ([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
    popt, pcov = curve_fit(
        lorentzian, r_pooled, value_pooled, p0=p0, bounds=bounds, maxfev=20000
    )
    return popt, pcov


def nrmse(data: np.ndarray, model: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((data - model) ** 2))
    return float(rmse / (np.max(data) - np.min(data)))


def plot_fit(
    x_px: np.ndarray,
    window_data: np.ndarray,
    z_window: np.ndarray,
    sg_params: np.ndarray,
    sg_nrmse: float,
    lor_params: np.ndarray,
    lor_nrmse: float,
    *,
    title: str | None = None,
) -> plt.Figure:
    r_pooled, value_pooled = pool_lineouts(x_px, window_data)
    mean_profile = window_data.mean(axis=1)

    x_fine = np.linspace(x_px[0], x_px[-1], 10 * len(x_px))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        r_pooled,
        value_pooled,
        s=4,
        color="gray",
        alpha=0.15,
        linewidths=0,
        label=f"{window_data.shape[1]} pooled lineouts",
    )
    ax.plot(x_px, mean_profile, "k-", lw=1.2, alpha=0.8, label="mean lineout")
    ax.plot(
        x_fine,
        super_gaussian(x_fine, *sg_params),
        "r-",
        lw=2,
        label=f"super-Gaussian (n={sg_params[2]:.2f}, NRMSE={sg_nrmse:.3g})",
    )
    ax.plot(
        x_fine,
        lorentzian(x_fine, *lor_params),
        "b-",
        lw=2,
        label=f"Lorentzian (NRMSE={lor_nrmse:.3g})",
    )
    ax.set_xlabel("r [px]")
    ax.set_ylabel("phase / px")
    ax.set_title(
        title
        or f"transverse lineouts, z = {z_window[0]:.0f}-{z_window[-1]:.0f} px "
        f"({window_data.shape[1]} slices)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="abel_invert output HDF5 file")
    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help="center z position [px] of the fitting window (default: middle of the z range)",
    )
    parser.add_argument(
        "--z-half-width",
        type=float,
        default=DEFAULT_Z_HALF_WIDTH_PX,
        help="half-width [px] of the z window pooled into the fit (default: %(default)s px); "
        "0 uses a single lineout",
    )
    parser.add_argument(
        "--plot-output", type=Path, help="Save the figure here instead of showing it"
    )
    parser.add_argument("-t", "--title", help="Figure title")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    x_px, window_data, z_window, z_center = load_lineout_window(
        args.input, z_px=args.z, z_half_width_px=args.z_half_width
    )
    r_pooled, value_pooled = pool_lineouts(x_px, window_data)
    mean_profile = window_data.mean(axis=1)

    sg_params, _ = fit_super_gaussian(
        r_pooled, value_pooled, x_px=x_px, mean_profile=mean_profile
    )
    sg_nrmse = nrmse(value_pooled, super_gaussian(r_pooled, *sg_params))
    lor_params, _ = fit_lorentzian(
        r_pooled, value_pooled, x_px=x_px, mean_profile=mean_profile
    )
    lor_nrmse = nrmse(value_pooled, lorentzian(r_pooled, *lor_params))

    print(
        f"fitting window: z = {z_window[0]:.0f}-{z_window[-1]:.0f} px "
        f"({window_data.shape[1]} lineouts, {r_pooled.size} pooled points)"
    )
    print(
        f"super-Gaussian: amplitude={sg_params[0]:.5g}, width={sg_params[1]:.5g} px, "
        f"order={sg_params[2]:.3g}, offset={sg_params[3]:.5g}  (NRMSE={sg_nrmse:.3g})"
    )
    print(
        f"Lorentzian:     amplitude={lor_params[0]:.5g}, width={lor_params[1]:.5g} px, "
        f"offset={lor_params[2]:.5g}  (NRMSE={lor_nrmse:.3g})"
    )

    fig = plot_fit(
        x_px,
        window_data,
        z_window,
        sg_params,
        sg_nrmse,
        lor_params,
        lor_nrmse,
        title=args.title,
    )
    if args.plot_output is not None:
        fig.savefig(args.plot_output, dpi=150)
        plt.close(fig)
        print(f"saved plot to {args.plot_output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
