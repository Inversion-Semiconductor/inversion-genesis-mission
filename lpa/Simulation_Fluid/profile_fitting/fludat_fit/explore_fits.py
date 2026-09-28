#!/usr/bin/env python3
"""Interactively fit profile families to the lineout selected by x, pressure and angle sliders.

Examples (conda env inv-fbpic):

    python -m fludat_fit.explore_fits ../postproc/data/density_lineouts/400_um.h5
    python -m fludat_fit.explore_fits htu_dens_7_0.h5 --x 1.0 --pressure 20 --angle 5 \\
        --families conical:supergaussian generalized_lorentzian_sum[2] --live --output-dir fits
    python -m fludat_fit.explore_fits cube.h5 --x 0.5 -o explorer.png   # static figure

The sliders select the lineout line: its transverse offset ``x`` at ``z = 0``,
the backing pressure, and its angle to the ``z`` axis (0 is a plain lineout
along ``z``). Press *Fit* to fit the families (or pass ``--live`` to refit on
every slider change). *Save* writes the ranked results as JSON, including the
``InterpolateFromH5Profile`` arguments that reproduce the lineout in FBPIC, and
the best profile as an FBPIC YAML config into ``--output-dir``.
"""

from __future__ import annotations

if __package__ in (None, ""):  # run as a plain script
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fludat_fit"

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider

from .cli_common import (
    MM_PER_M,
    FitWindow,
    add_dataset_arguments,
    add_export_arguments,
    add_fit_arguments,
    add_window_arguments,
    build_scheme,
    export_profile_yaml,
    fit_window_from_args,
    handle_list_families,
    load_dataset,
    longitudinal_label,
    print_comparison,
    resolve_families,
    result_stem,
)
from .dataset import NozzleDataset
from .families import ProfileFamily
from .fitting import FamilyComparison, FittingScheme, compare_families
from .goodness_of_fit import DEFAULT_RANKING_METRIC
from .lineout import Lineout, LineoutConditions
from .plotting import DATA_COLOR, SERIES_COLORS

DEFAULT_ANGLE_RANGE = (-45.0, 45.0)


class FitExplorer:
    """A figure with condition sliders, a fit/save button pair, and the fit overlay.

    Args:
        dataset: The cube to take lineouts from.
        families: Families fitted on each *Fit*.
        scheme: The fitting scheme. A ``MultiStartLocalFit`` gets the previous
            fit of every family as a warm start, which makes small slider moves cheap.
        window: How lineouts are cropped and resampled before fitting.
        conditions: Initial slider values; default: lowest x and pressure, angle 0.
        angle_range: Slider range for the lineout angle [deg].
        top: Number of ranked model curves drawn.
        live: Refit on every slider change instead of on *Fit* only.
        output_dir: Where *Save* writes JSON and YAML; default ``fits``.
        rank_by: Ranking metric.
        nominal_density: [m^-3] for exported configs; default from the fitted amplitude.
        species: Species for exported configs.
    """

    def __init__(
        self,
        dataset: NozzleDataset,
        families: Sequence[ProfileFamily],
        scheme: FittingScheme,
        window: FitWindow | None = None,
        *,
        conditions: LineoutConditions | None = None,
        angle_range: tuple[float, float] = DEFAULT_ANGLE_RANGE,
        top: int = 3,
        live: bool = False,
        output_dir: Path | str = "fits",
        rank_by: str = DEFAULT_RANKING_METRIC,
        nominal_density: float | None = None,
        species: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.families = list(families)
        self.scheme = scheme
        self.window = window or FitWindow()
        self.top = max(1, min(top, len(SERIES_COLORS)))
        self.live = live
        self.output_dir = Path(output_dir)
        self.rank_by = rank_by
        self.nominal_density = nominal_density
        self.species = species
        self.comparison: FamilyComparison | None = None
        self.lineout: Lineout | None = None

        a_lo, a_hi = dataset.angle_extent
        lower, upper = sorted(angle_range)
        self.angle_range = (max(lower, a_lo), min(upper, a_hi))
        if not self.angle_range[0] < self.angle_range[1]:
            raise ValueError(
                f"angle_range {angle_range} must be increasing and within {dataset.angle_extent}"
            )

        self._build_figure(conditions or dataset.default_conditions())
        self.update_lineout()

    # ------------------------------------------------------------- layout
    def _build_figure(self, initial: LineoutConditions) -> None:
        n_sliders = 3
        self.figure: Figure = plt.figure(figsize=(12.5, 7.5))
        grid = gridspec.GridSpec(
            2 + n_sliders + 1,
            2,
            width_ratios=[3.0, 1.5],
            height_ratios=[3.0, 1.0] + [0.28] * n_sliders + [0.45],
            hspace=0.5,
            wspace=0.08,
            left=0.07,
            right=0.98,
            top=0.93,
            bottom=0.06,
        )
        self.ax_fit = self.figure.add_subplot(grid[0, 0])
        self.ax_res = self.figure.add_subplot(grid[1, 0], sharex=self.ax_fit)
        self.ax_text = self.figure.add_subplot(grid[0:2, 1])
        self.ax_text.axis("off")
        plt.setp(self.ax_fit.get_xticklabels(), visible=False)

        (self.data_line,) = self.ax_fit.plot(
            [], [], color=DATA_COLOR, lw=1.2, alpha=0.85, label="lineout"
        )
        self.model_lines = [
            self.ax_fit.plot([], [], color=color, lw=2.0)[0]
            for color in SERIES_COLORS[: self.top]
        ]
        self.residual_lines = [
            self.ax_res.plot([], [], color=color, lw=1.5)[0]
            for color in SERIES_COLORS[: self.top]
        ]
        self.ax_res.axhline(0.0, color=DATA_COLOR, lw=0.8, alpha=0.4)
        self.ax_fit.set_ylabel(f"density [{self.dataset.density_units}]")
        self.ax_res.set_ylabel("residual / peak")
        for ax in (self.ax_fit, self.ax_res):
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        self.status = self.ax_text.text(
            0.0,
            1.0,
            "",
            transform=self.ax_text.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=7.5,
        )

        slider_grid = grid[2 : 2 + n_sliders, 0].subgridspec(
            n_sliders, 2, width_ratios=[0.16, 0.84], hspace=0.6
        )
        x_lo, x_hi = self.dataset.x_extent
        p_lo, p_hi = self.dataset.pressure_extent
        a_lo, a_hi = self.angle_range
        self.x_slider = Slider(
            self.figure.add_subplot(slider_grid[0, 1]),
            "x [mm]",
            x_lo,
            x_hi,
            valinit=float(np.clip(initial.x_mm, x_lo, x_hi)),
        )
        self.pressure_slider = Slider(
            self.figure.add_subplot(slider_grid[1, 1]),
            "pressure [bar]",
            p_lo,
            p_hi,
            valinit=float(np.clip(initial.pressure_bar, p_lo, p_hi)),
        )
        self.angle_slider = Slider(
            self.figure.add_subplot(slider_grid[2, 1]),
            "angle [deg]",
            a_lo,
            a_hi,
            valinit=float(np.clip(initial.angle_deg, a_lo, a_hi)),
        )
        self.sliders = (self.x_slider, self.pressure_slider, self.angle_slider)
        for slider in self.sliders:
            slider.on_changed(self._on_slider)

        buttons = grid[2 + n_sliders, :].subgridspec(1, 8, wspace=0.3)
        self.fit_button = Button(self.figure.add_subplot(buttons[0, 6]), "Fit")
        self.save_button = Button(self.figure.add_subplot(buttons[0, 7]), "Save")
        self.fit_button.on_clicked(lambda _event: self.fit())
        self.save_button.on_clicked(lambda _event: self.save())
        self.widgets = (*self.sliders, self.fit_button, self.save_button)

    # ---------------------------------------------------------- behaviour
    @property
    def conditions(self) -> LineoutConditions:
        return LineoutConditions(
            x_mm=float(self.x_slider.val),
            pressure_bar=float(self.pressure_slider.val),
            angle_deg=float(self.angle_slider.val),
        )

    def set_conditions(self, conditions: LineoutConditions) -> None:
        """Move the sliders (each move triggers the slider callback)."""
        self.x_slider.set_val(conditions.x_mm)
        self.pressure_slider.set_val(conditions.pressure_bar)
        self.angle_slider.set_val(conditions.angle_deg)

    def _on_slider(self, _value: float) -> None:
        self.update_lineout()
        if self.live:
            self.fit()

    def update_lineout(self) -> Lineout:
        """Re-extract the lineout for the current sliders and clear stale fits."""
        conditions = self.conditions
        self.lineout = self.window.apply(self.dataset.lineout(conditions))
        self.comparison = None
        self.data_line.set_data(self.lineout.z * MM_PER_M, self.lineout.density)
        for line in (*self.model_lines, *self.residual_lines):
            line.set_data([], [])
        self.ax_fit.set_title(f"{self.dataset.name}: {conditions.label()}")
        self.ax_res.set_xlabel(longitudinal_label(self.lineout))
        self.ax_fit.legend(
            [self.data_line], ["lineout"], loc="upper left", fontsize=8, frameon=False
        )
        self._autoscale()
        self.status.set_text(
            f"{self.lineout.n_points} samples, peak {self.lineout.peak:.3g} "
            f"{self.lineout.density_units}\n\n"
            + ("(no density here)" if self.lineout.peak <= 0.0 else "press Fit")
        )
        self.figure.canvas.draw_idle()
        return self.lineout

    def fit(self) -> FamilyComparison | None:
        """Fit every family to the current lineout and redraw the overlay."""
        if self.lineout is None or self.lineout.peak <= 0.0:
            return None
        self.status.set_text("fitting ...")
        self.figure.canvas.draw_idle()
        self.comparison = compare_families(
            self.lineout, self.families, self.scheme, rank_by=self.rank_by
        )
        warm_starts = getattr(self.scheme, "warm_starts", None)
        if warm_starts is not None:
            for result in self.comparison:
                warm_starts[result.family_name] = result.parameters

        z_fine = np.linspace(self.lineout.z[0], self.lineout.z[-1], 1000)
        shown = self.comparison.results[: self.top]
        labels = ["lineout"]
        for model_line, residual_line, result in zip(
            self.model_lines, self.residual_lines, shown, strict=False
        ):
            model_line.set_data(z_fine * MM_PER_M, result.model_density(z_fine))
            residual_line.set_data(
                self.lineout.z * MM_PER_M,
                (result.model_density() - self.lineout.density) / self.lineout.peak,
            )
            labels.append(f"{result.family_name} (NRMSE {result.goodness.nrmse:.3g})")
        for line in (
            *self.model_lines[len(shown) :],
            *self.residual_lines[len(shown) :],
        ):
            line.set_data([], [])
        self.ax_fit.legend(
            [self.data_line, *self.model_lines[: len(shown)]],
            labels,
            loc="upper left",
            fontsize=8,
            frameon=False,
        )
        self._autoscale()
        self.status.set_text(self._ranking_text())
        self.figure.canvas.draw_idle()
        print_comparison(self.comparison, self.dataset.name)
        return self.comparison

    def save(self) -> list[Path]:
        """Write the current comparison (JSON) and its best profile (YAML)."""
        if self.comparison is None:
            self.comparison = self.fit()
        if self.comparison is None:
            return []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        best = self.comparison.best
        payload = self.comparison.to_dict()
        payload["interp_from_h5"] = self.dataset.h5_profile_kwargs(best.conditions)
        json_path = self.output_dir / f"{result_stem(best)}.json"
        json_path.write_text(json.dumps(payload, indent=2))
        yaml_path = export_profile_yaml(
            best,
            self.output_dir,
            nominal_density=self.nominal_density,
            species=self.species,
        )
        for path in (json_path, yaml_path):
            print(f"wrote {path}")
        self.status.set_text(self._ranking_text() + f"\n\nsaved to {self.output_dir}/")
        self.figure.canvas.draw_idle()
        return [json_path, yaml_path]

    # ------------------------------------------------------------ helpers
    def _autoscale(self) -> None:
        for ax in (self.ax_fit, self.ax_res):
            ax.relim()
            ax.autoscale_view()
        # set_ylim disables y autoscaling, so derive the top from the drawn curves
        # on every call instead of relying on autoscale_view.
        peaks = [
            np.nanmax(y)
            for line in (self.data_line, *self.model_lines)
            if len(y := np.asarray(line.get_ydata(), dtype=float))
        ]
        top = max(peaks, default=0.0)
        if not np.isfinite(top) or top <= 0.0:
            top = 1.0
        self.ax_fit.set_ylim(0.0, 1.05 * top)

    def _ranking_text(self) -> str:
        assert self.comparison is not None and self.lineout is not None
        lines = [
            f"{self.lineout.n_points} samples, ranked by {self.rank_by}",
            "",
            f"{'#':>2} {'family':<34} {'nrmse':>7} {'irel':>7} {'k':>2}",
        ]
        for index, result in enumerate(self.comparison, start=1):
            g = result.goodness
            name = result.family_name
            if len(name) > 34:
                name = name[:31] + "..."
            lines.append(
                f"{index:>2} {name:<34} {g.nrmse:>7.4f} "
                f"{g.integrated_relative_error:>7.4f} {g.n_parameters:>2}"
            )
        best = self.comparison.best
        lines += [
            "",
            f"best: {best.family_name}",
            f"amplitude = {best.amplitude:.4g} {self.lineout.density_units}",
        ]
        lines.extend(f"{name} = {value:.6g}" for name, value in best.parameters.items())
        return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_dataset_arguments(parser)
    add_window_arguments(parser)
    add_fit_arguments(parser, default_starts=4)
    add_export_arguments(parser)
    group = parser.add_argument_group("explorer")
    group.add_argument("--x", type=float, help="Initial x [mm]; default lowest.")
    group.add_argument(
        "--pressure", type=float, help="Initial backing pressure [bar]; default lowest."
    )
    group.add_argument(
        "--angle",
        type=float,
        default=0.0,
        help="Initial lineout angle [deg]; default 0.",
    )
    group.add_argument(
        "--angle-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(DEFAULT_ANGLE_RANGE),
        help=f"Angle slider range [deg]; default {DEFAULT_ANGLE_RANGE[0]:g} {DEFAULT_ANGLE_RANGE[1]:g}.",
    )
    group.add_argument(
        "--top", type=int, default=3, help="Ranked model curves drawn; default 3."
    )
    group.add_argument(
        "--live", action="store_true", help="Refit on every slider change."
    )
    group.add_argument(
        "--fit", action="store_true", help="Fit the initial point before showing."
    )
    group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fits"),
        help="Where the Save button writes JSON and YAML; default ./fits.",
    )
    group.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Fit the initial point and save the figure here instead of showing it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if handle_list_families(args):
        return 0
    dataset = load_dataset(args, parser)
    default = dataset.default_conditions()
    initial = LineoutConditions(
        x_mm=default.x_mm if args.x is None else args.x,
        pressure_bar=default.pressure_bar if args.pressure is None else args.pressure,
        angle_deg=args.angle,
    )
    explorer = FitExplorer(
        dataset,
        resolve_families(args),
        build_scheme(args),
        fit_window_from_args(args),
        conditions=initial,
        angle_range=tuple(args.angle_range),
        top=args.top,
        live=args.live,
        output_dir=args.output_dir,
        rank_by=args.rank_by,
        nominal_density=args.nominal_density,
        species=args.species,
    )
    if args.fit or args.live or args.output is not None:
        explorer.fit()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        explorer.figure.savefig(args.output, dpi=150)
        print(f"wrote {args.output}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
