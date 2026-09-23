"""Plot HDF5 density lineouts and their cutoff-derived longitudinal bounds.

Usage:
    python plot_profiles.py --files /path/to/profile1.h5 /path/to/profile2.h5 --output density_profiles.png
    python plot_profiles.py --files /path/to/profile.h5 --x-mm 2.0 --angle 45
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile


DEMO_DIRECTORY = Path(__file__).parent
DEFAULT_FILES = (DEMO_DIRECTORY / "800_um.h5", DEMO_DIRECTORY / "g_2_mm_400_um.h5")


def _pressure_samples(filename: Path) -> np.ndarray:
    """Return stored pressure points plus their midpoints for interpolation."""
    with h5py.File(filename, "r") as h5_file:
        pressure_points = np.asarray(h5_file["pressure_bar"], dtype=float)

    return np.sort(
        np.concatenate(
            (pressure_points, (pressure_points[:-1] + pressure_points[1:]) / 2.0)
        )
    )


def _build_profile(
    filename: Path, x_position_mm: float, angle_deg: float, pressure_bar: float
) -> InterpolateFromH5Profile:
    """Build one independent lineout at the requested transverse conditions."""
    return InterpolateFromH5Profile(
        filename=filename,
        density_name="density",
        density_scale=1e6,
        lineout_axis={
            "z_m": {"origin": 0.0, "coefficient": np.cos(angle_deg * np.pi / 180.0)},
            "x_mm": {
                "origin": x_position_mm,
                "coefficient": 1e3 * np.sin(angle_deg * np.pi / 180.0),
            },
        },
        interpolation_points={"pressure_bar": pressure_bar},
        p_nz=1,
        p_nr=1,
        p_nt=1,
        density_cutoff_ratio=5e-2,
        centering_mode="left",
        centroid_axis="z_m",
    )


def plot_file(
    ax: plt.Axes, filename: Path, x_position_mm: float, angle_deg: float
) -> None:
    """Plot sampled and interpolated pressure lineouts from one HDF5 file."""
    pressure_samples = _pressure_samples(filename)
    colors = plt.colormaps["viridis"](np.linspace(0.0, 1.0, len(pressure_samples)))

    for pressure_bar, color in zip(pressure_samples, colors, strict=True):
        profile = _build_profile(
            filename, x_position_mm, angle_deg, float(pressure_bar)
        )
        profile.plot_z_profile(
            ax,
            num=1_000,
            z_scale=1e3,
            label=f"{pressure_bar:g} bar",
            color=color,
        )

        # Each pair of vertical lines marks the cutoff-derived profile bounds.
        z_min, z_max = profile.get_z_extent()
        ax.axvline(z_min * 1e3, color=color, linestyle="--", alpha=0.55)
        ax.axvline(z_max * 1e3, color=color, linestyle="--", alpha=0.55)
        ax.axvline(profile.centroid * 1e3, color=color, linestyle="--", alpha=0.55)

    ax.set_title(filename.stem.replace("_", " "))
    ax.set_xlabel("z [mm]")
    ax.set_ylabel(r"density [m$^{-3}$]")
    ax.grid(alpha=0.25)
    ax.legend(title="pressure", ncols=2, fontsize="small")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        type=Path,
        nargs="+",
        default=DEFAULT_FILES,
        help="HDF5 density profile files to plot. Defaults to the two demo files.",
    )
    parser.add_argument(
        "--x-mm",
        type=float,
        default=2.0,
        help="Transverse x position used for every lineout. Defaults to 2.0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEMO_DIRECTORY / "density_profiles.png",
        help="Output image path. Defaults to density_profiles.png beside this script.",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=45.0,
        help="Angle with respect to z-axis in degrees. Defaults to 45 degrees.",
    )
    args = parser.parse_args()

    figure, axes = plt.subplots(
        nrows=len(args.files), figsize=(10, 4.5 * len(args.files)), sharex=False
    )
    for ax, filename in zip(np.atleast_1d(axes), args.files, strict=True):
        plot_file(ax, filename, args.x_mm, args.angle)

    figure.tight_layout()
    figure.savefig(args.output, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
