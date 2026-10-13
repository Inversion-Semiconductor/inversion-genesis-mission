#!/usr/bin/env python3
"""Plot every profile of a filled ``profile_parameters_*.json`` config on one figure.

Each named profile in ``profiles`` is reconstructed with the real
``inversion_fbpic`` classes from its ``kwargs`` (``center`` substitutes for
``start_position`` on conical profiles, exactly as the template documents),
drawn over its own ``get_z_extent()`` (its full defined support, not just a
narrow fit window), with a dashed vertical line at its ``center`` or
``centroid`` (the two names mean the same thing: where the profile is
centred). Showing the full extent matters because a profile can fit the data
well over the region it was compared against while still carrying an isolated,
badly-placed bump well outside it (see :class:`~fludat_fit.fitting.FitObjective`
for the loss term that discourages this). ``InterpolateFromH5Profile`` has
neither a builder here nor a centre: its curve is the density cube's own
lineout, sampled over the union of every other profile's extent, as the
reference the others were fitted against, with no line.

Each reconstructed profile is translated back out of its own left-edge-zero
frame (``generate_profile_configs``'s output convention) into the frame it was
actually fit in, where every profile shares one common anchor: the lineout's
own crossing of the gas target's physical ``z_m = 0`` plane. Without this,
profiles fit at different points would overlay in a coordinate frame the data
itself does not use, making the comparison meaningless.

Example (conda env inv-fbpic), from this directory:

    python -m fludat_fit.plot_profile_configs data/fit_cfgs/profile_parameters_x1mm_p20bar_a0deg.json
    python -m fludat_fit.plot_profile_configs data/fit_cfgs/profile_parameters_x*.json -o data/fit_cfgs
"""

from __future__ import annotations

if __package__ in (None, ""):  # run as a plain script
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fludat_fit"

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .cli_common import MM_PER_M
from .dataset import NozzleDataset
from .fitting import density_units_to_m3
from .lineout import Lineout, LineoutConditions
from .plotting import DATA_COLOR, SERIES_COLORS
from .profile_reconstruction import PROFILE_BUILDERS, build_profile, center_of

N_MODEL_POINTS = 2000


class ProfileCurve:
    """One profile's plotted curve: a name, a density(z) array, and an optional centre."""

    def __init__(
        self, name: str, z: np.ndarray, density: np.ndarray, center: float | None
    ) -> None:
        self.name = name
        self.z = z
        self.density = density
        self.center = center


def _build_profile_curves(
    config: dict[str, Any], target_centroid: float
) -> list[ProfileCurve]:
    """One curve per fittable profile, each over its own ``get_z_extent()``,
    translated back from its own left-edge-zero output frame to ``target_centroid``
    (the common anchor every profile was actually fit against; see the module
    docstring). This is an exact rigid shift, so it changes neither the curve's
    shape nor its density values."""
    curves = []
    for name, spec in config["profiles"].items():
        profile = build_profile(name, spec)
        if profile is None:
            if spec["class"] != "InterpolateFromH5Profile":
                print(
                    f"warning: don't know how to plot {name!r} (class {spec['class']!r}); "
                    "skipping",
                    file=sys.stderr,
                )
            continue  # InterpolateFromH5Profile has no builder; it is the reference curve
        z_min, z_max = profile.get_z_extent()
        z = np.linspace(z_min, z_max, N_MODEL_POINTS)
        density = np.asarray(profile.build_density_function()(z, np.zeros_like(z)))
        density = density * profile.nominal_density
        reported = center_of(spec)
        shift = reported - target_centroid
        curves.append(ProfileCurve(name, z - shift, density, reported - shift))
    return curves


def _reference_curve(
    dataset: NozzleDataset, conditions: LineoutConditions, z_values: np.ndarray
) -> Lineout:
    """The raw density-cube lineout at ``z_values``, converted to m^-3."""
    lineout = dataset.lineout(conditions, t_values=z_values)
    factor = density_units_to_m3(lineout.density_units)
    if factor is None:
        print(
            f"warning: unrecognised density units {lineout.density_units!r}; "
            "plotting the raw lineout unconverted",
            file=sys.stderr,
        )
        factor = 1.0
    return Lineout(
        z=lineout.z,
        density=lineout.density * factor,
        density_units="m^-3",
        conditions=conditions,
        source=lineout.source,
    )


def build_curves(
    config: dict[str, Any], config_path: Path
) -> tuple[list[ProfileCurve], Lineout]:
    """Every plottable curve of ``config``, plus the reference lineout they are compared to.

    Each fitted profile is drawn over its own full ``get_z_extent()``, shifted
    back to the common anchor (see :func:`_build_profile_curves`); the
    reference (``InterpolateFromH5Profile``/raw data) curve spans the union of
    those (now-common-frame) extents, so everything shares one comparable x axis.
    """
    if not any(
        spec["class"] in PROFILE_BUILDERS for spec in config["profiles"].values()
    ):
        raise ValueError(
            "no fittable profile in this config (only InterpolateFromH5Profile?)"
        )
    density_path = (config_path.parent / config["density_file"]).resolve()
    dataset = NozzleDataset.load(density_path)
    conditions = LineoutConditions(
        x_mm=config["x_mm"], pressure_bar=config["p_bar"], angle_deg=config["angle"]
    )
    target_centroid = dataset.path_for(conditions).t_at_physical_z(0.0)

    profile_curves = _build_profile_curves(config, target_centroid)
    if not profile_curves:
        raise ValueError(
            "no fittable profile in this config (only InterpolateFromH5Profile?)"
        )
    combined_min = min(curve.z.min() for curve in profile_curves)
    combined_max = max(curve.z.max() for curve in profile_curves)
    reference = _reference_curve(
        dataset, conditions, np.linspace(combined_min, combined_max, N_MODEL_POINTS)
    )
    return [
        ProfileCurve("h5_direct (data)", reference.z, reference.density, None),
        *profile_curves,
    ], reference


def plot_profile_config(
    config: dict[str, Any], config_path: Path, *, title: str | None = None
) -> Figure:
    """All profiles of a filled config on one axes, with dashed lines at their centres."""
    curves, reference = build_curves(config, config_path)
    conditions = LineoutConditions(
        x_mm=config["x_mm"], pressure_bar=config["p_bar"], angle_deg=config["angle"]
    )
    longitudinal_label = (
        "z [mm]" if conditions.angle_deg == 0.0 else "t along the lineout line [mm]"
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    colors = [DATA_COLOR, *SERIES_COLORS]
    for color, curve in zip(colors, curves, strict=False):
        ax.plot(
            curve.z * MM_PER_M,
            curve.density,
            color=color,
            lw=2.0 if curve.center is None else 1.6,
            alpha=0.85 if curve.center is None else 1.0,
            label=curve.name,
        )
        if curve.center is not None:
            ax.axvline(curve.center * MM_PER_M, color=color, ls="--", lw=1.2, alpha=0.8)

    ax.set_xlabel(longitudinal_label)
    ax.set_ylabel(f"density [{reference.density_units}]")
    ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title(title or f"{reference.source}: {conditions.label()}")
    fig.tight_layout()
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "configs",
        nargs="+",
        type=Path,
        help="Filled profile_parameters_*.json config(s).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save the figures; default: next to each config.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show the figure(s) as well."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    written = []
    for config_path in args.configs:
        config = json.loads(config_path.read_text())
        figure = plot_profile_config(config, config_path)
        output_dir = args.output_dir or config_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{config_path.stem}.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        written.append(output_path)
        print(f"wrote {output_path}")

    if args.show:
        plt.show()
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
