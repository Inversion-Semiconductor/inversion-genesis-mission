#!/usr/bin/env python3
"""Plot density lineout profiles stored as .npz files.

Usage (conda env inv-fbpic):

    conda run -n inv-fbpic python show_profile.py \\
        data/density_lineouts/800_um_conical_5_bar_density_lineout/x-2-0.npz

    conda run -n inv-fbpic python show_profile.py \\
        data/density_lineouts/800_um_conical_5_bar_density_lineout
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def collect_npz_paths(path: Path) -> list[Path]:
    """Return .npz file paths from a single file or directory input."""
    path = path.resolve()

    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got {path}")
        return [path]

    if path.is_dir():
        files = sorted(path.rglob("*.npz"))
        if not files:
            raise ValueError(f"No .npz files found in {path}")
        return files

    raise FileNotFoundError(path)


def load_profile(path: Path) -> np.ndarray:
    """Load a (z, density) array from an .npz file."""
    with np.load(path) as archive:
        if "data" not in archive:
            raise ValueError(f"{path} does not contain a 'data' array")
        return archive["data"]


def plot_profiles(paths: list[Path], title: str | None = None) -> plt.Figure:
    """Plot one or more density lineout profiles."""
    fig, ax = plt.subplots()

    for path in paths:
        data = load_profile(path)
        label = path.stem if len(paths) > 1 else None
        ax.plot(data[:, 0] * 1000, data[:, 1], label=label)

    ax.set_xlabel("z [mm]")
    ax.set_ylabel("density [1e18 cm^-3]")
    if title:
        ax.set_title(title)
    elif len(paths) == 1:
        ax.set_title(paths[0].stem)
    if len(paths) > 1:
        ax.legend()
    ax.grid()


    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot density lineout profiles from .npz file(s)."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a .npz file or a directory containing .npz files",
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
        help="Figure title (default: filename stem for a single profile)",
    )
    args = parser.parse_args()

    paths = collect_npz_paths(args.input)
    fig = plot_profiles(paths, title=args.title)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Wrote {args.output.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
