#!/usr/bin/env python3
"""Fit every profile family over a distribution of (x, pressure, angle) points and summarise.

Examples (conda env inv-fbpic):

    python -m fludat_fit.fit_statistics htu_dens_7_0.h5 \\
        --x-range 0 2 --pressure-range 5 40 --angle-range -10 10 --samples 32 \\
        --workers 4 --json stats.json --csv stats.csv --plot stats.png
    python -m fludat_fit.fit_statistics cube.h5 --pressure-range 5 35 --sampling grid \\
        --samples 9 --families conical:supergaussian generalized_lorentzian_sum[2]

The sampled points are Latin-hypercube (default), uniform-random, or a grid over
the selected ranges (default: the cube's x and pressure extents, with the lineout
angle fixed at 0 unless ``--angle-range`` is given). Every family is fitted at
every point; the report gives, per family, the distribution of goodness-of-fit
metrics across points, how often it ranks first, its mean rank, and timing.
The CSV lists one row per (point, family) with the fitted parameters, which is
the raw material for a ``conditions -> parameters`` mapping.
"""

from __future__ import annotations

if __package__ in (None, ""):  # run as a plain script
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "fludat_fit"

import argparse
import csv
import json
import math
import sys
import time
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import qmc

from .cli_common import (
    FitWindow,
    add_dataset_arguments,
    add_fit_arguments,
    add_window_arguments,
    build_scheme,
    fit_window_from_args,
    handle_list_families,
    load_dataset,
    resolve_families,
)
from .dataset import NozzleDataset
from .families import ProfileFamily
from .fitting import FamilyComparison, FitResult, FittingScheme, compare_families
from .goodness_of_fit import DEFAULT_RANKING_METRIC, RANKING_METRICS
from .lineout import Lineout, LineoutConditions
from .plotting import DATA_COLOR, SERIES_COLORS

Sampling = Literal["lhs", "random", "grid"]
SAMPLING_METHODS: tuple[Sampling, ...] = ("lhs", "random", "grid")
CONDITION_AXES = ("x_mm", "pressure_bar", "angle_deg")
AXIS_LABELS = {
    "x_mm": "x [mm]",
    "pressure_bar": "pressure [bar]",
    "angle_deg": "angle [deg]",
}
METRIC_COLUMNS = (
    "nrmse",
    "rmse",
    "r_squared",
    "max_abs_error",
    "integrated_relative_error",
    "aic",
    "bic",
)


# ----------------------------------------------------------------- sampling
@dataclass(frozen=True)
class ConditionRanges:
    """The box of physical conditions to sample.

    Each axis is a ``(lower, upper)`` pair; a pair with equal values fixes that
    axis. The angle is the lineout line's angle to the ``z`` axis [deg].
    """

    x_mm: tuple[float, float]
    pressure_bar: tuple[float, float]
    angle_deg: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_dataset(
        cls,
        dataset: NozzleDataset,
        *,
        x_range: Sequence[float] | None = None,
        pressure_range: Sequence[float] | None = None,
        angle_range: Sequence[float] | None = None,
    ) -> ConditionRanges:
        """Ranges inside the dataset extents.

        ``None`` selects the full x or pressure extent; the angle defaults to a
        fixed ``0`` (axial lineouts). Ranges reaching outside are clamped with a
        warning.
        """
        return cls(
            x_mm=_validated_range(x_range, dataset.x_extent, "x"),
            pressure_bar=_validated_range(
                pressure_range, dataset.pressure_extent, "pressure"
            ),
            angle_deg=(
                (0.0, 0.0)
                if angle_range is None
                else _validated_range(angle_range, dataset.angle_extent, "angle")
            ),
        )

    @property
    def axes(self) -> dict[str, tuple[float, float]]:
        return {
            "x_mm": self.x_mm,
            "pressure_bar": self.pressure_bar,
            "angle_deg": self.angle_deg,
        }

    @property
    def varying_axes(self) -> list[str]:
        return [name for name, (lower, upper) in self.axes.items() if lower < upper]

    def sample(
        self, n: int, sampling: Sampling = "lhs", seed: int | None = 0
    ) -> list[LineoutConditions]:
        """``n`` condition points (a grid rounds ``n`` up to a full lattice)."""
        if n < 1:
            raise ValueError("n must be at least one")
        varying = self.varying_axes
        d = len(varying)
        if d == 0:
            unit = np.zeros((1, 0))
        elif sampling == "lhs":
            unit = qmc.LatinHypercube(d=d, seed=seed).random(n)
        elif sampling == "random":
            unit = np.random.default_rng(seed).random((n, d))
        elif sampling == "grid":
            per_axis = max(2, math.ceil(n ** (1.0 / d)))
            axes = np.meshgrid(*[np.linspace(0.0, 1.0, per_axis)] * d, indexing="ij")
            unit = np.column_stack([axis.ravel() for axis in axes])
        else:
            raise ValueError(f"unknown sampling {sampling!r}")

        points = []
        for row in unit:
            values: dict[str, float] = {}
            column = 0
            for name, (lower, upper) in self.axes.items():
                if name in varying:
                    values[name] = float(lower + row[column] * (upper - lower))
                    column += 1
                else:
                    values[name] = float(lower)
            points.append(LineoutConditions(**values))
        return points

    def to_dict(self) -> dict[str, Any]:
        return {name: list(bounds) for name, bounds in self.axes.items()}


def _validated_range(
    requested: Sequence[float] | None, extent: tuple[float, float], axis: str
) -> tuple[float, float]:
    """The requested range clamped to the dataset extent (with a warning)."""
    if requested is None:
        return (float(extent[0]), float(extent[1]))
    lower, upper = sorted(float(value) for value in requested)
    if upper < extent[0] or lower > extent[1]:
        raise ValueError(
            f"{axis} range [{lower:g}, {upper:g}] does not overlap the dataset extent "
            f"[{extent[0]:g}, {extent[1]:g}]"
        )
    clamped = (max(lower, extent[0]), min(upper, extent[1]))
    if clamped != (lower, upper):
        warnings.warn(
            f"{axis} range [{lower:g}, {upper:g}] clamped to the dataset extent "
            f"[{clamped[0]:g}, {clamped[1]:g}]",
            stacklevel=3,
        )
    return clamped


# ------------------------------------------------------------------ fitting
def _fit_one(
    payload: tuple[Lineout, list[ProfileFamily], FittingScheme, str],
) -> FamilyComparison:
    lineout, families, scheme, rank_by = payload
    return compare_families(lineout, families, scheme, rank_by=rank_by)


def fit_points(
    dataset: NozzleDataset,
    points: Iterable[LineoutConditions],
    families: Sequence[ProfileFamily],
    scheme: FittingScheme,
    window: FitWindow | None = None,
    *,
    rank_by: str = DEFAULT_RANKING_METRIC,
    workers: int = 1,
    progress: bool = False,
) -> list[FamilyComparison]:
    """Fit every family at every point; points with no density are skipped.

    Lineouts are extracted up front in this process; with ``workers > 1`` the
    fits are spread over a process pool (only the small lineouts are shipped).
    """
    window = window or FitWindow()
    lineouts = []
    for conditions in points:
        lineout = window.apply(dataset.lineout(conditions))
        if lineout.peak <= 0.0:
            if progress:
                print(f"  skip {conditions.label()}: no density", file=sys.stderr)
            continue
        lineouts.append(lineout)
    payloads = [(lineout, list(families), scheme, rank_by) for lineout in lineouts]

    comparisons: list[FamilyComparison] = []
    started = time.perf_counter()
    if workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            iterator = pool.map(_fit_one, payloads)
            for index, comparison in enumerate(iterator, start=1):
                comparisons.append(comparison)
                _report(progress, index, len(payloads), comparison, started)
    else:
        for index, payload in enumerate(payloads, start=1):
            comparison = _fit_one(payload)
            comparisons.append(comparison)
            _report(progress, index, len(payloads), comparison, started)
    return comparisons


def _report(
    progress: bool, index: int, total: int, comparison: FamilyComparison, started: float
) -> None:
    if not progress:
        return
    best = comparison.best
    print(
        f"  [{index}/{total}] {comparison.lineout.conditions.label()}: "
        f"best {best.family_name} (NRMSE {best.goodness.nrmse:.3g}) "
        f"[{time.perf_counter() - started:.0f}s]",
        file=sys.stderr,
    )


# --------------------------------------------------------------- statistics
@dataclass(frozen=True)
class MetricStats:
    median: float
    mean: float
    p90: float
    max: float

    @classmethod
    def of(cls, values: Sequence[float]) -> MetricStats:
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return cls(*(float("nan"),) * 4)
        return cls(
            median=float(np.median(finite)),
            mean=float(np.mean(finite)),
            p90=float(np.percentile(finite, 90)),
            max=float(np.max(finite)),
        )


@dataclass(frozen=True)
class FamilySummary:
    """One family's fitting performance across the sampled points."""

    family: str
    n_parameters: int
    n_points: int
    success_rate: float
    best_fraction: float
    mean_rank: float
    ranking_key: MetricStats
    """Statistics of the ranking metric's "lower is better" key across points."""
    metrics: dict[str, MetricStats]
    mean_seconds: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


class FitStatistics:
    """Comparisons at many condition points, summarised per family."""

    def __init__(
        self,
        comparisons: Sequence[FamilyComparison],
        *,
        ranges: ConditionRanges | None = None,
        sampling: str | None = None,
        dataset_name: str = "",
    ) -> None:
        if not comparisons:
            raise ValueError("no comparisons to summarise")
        self.comparisons = tuple(comparisons)
        self.ranges = ranges
        self.sampling = sampling
        self.dataset_name = dataset_name
        self.rank_by = comparisons[0].rank_by
        names: list[str] = []
        for comparison in comparisons:
            for result in comparison:
                if result.family_name not in names:
                    names.append(result.family_name)
        self.family_names = tuple(names)

    @property
    def n_points(self) -> int:
        return len(self.comparisons)

    @property
    def points(self) -> list[LineoutConditions]:
        return [comparison.lineout.conditions for comparison in self.comparisons]

    def results_of(self, family: str) -> list[tuple[int, FitResult]]:
        """``(rank, result)`` of ``family`` at every point where it was fitted."""
        found = []
        for comparison in self.comparisons:
            for rank, result in enumerate(comparison, start=1):
                if result.family_name == family:
                    found.append((rank, result))
        return found

    def summary(self) -> list[FamilySummary]:
        """Per-family summaries, best first by the median ranking key."""
        key = RANKING_METRICS[self.rank_by]
        summaries = []
        for family in self.family_names:
            ranked = self.results_of(family)
            results = [result for _, result in ranked]
            ranks = [rank for rank, _ in ranked]
            summaries.append(
                FamilySummary(
                    family=family,
                    n_parameters=results[0].goodness.n_parameters,
                    n_points=len(results),
                    success_rate=float(np.mean([r.success for r in results])),
                    best_fraction=float(np.mean([rank == 1 for rank in ranks])),
                    mean_rank=float(np.mean(ranks)),
                    ranking_key=MetricStats.of([key(r.goodness) for r in results]),
                    metrics={
                        name: MetricStats.of(
                            [getattr(r.goodness, name) for r in results]
                        )
                        for name in METRIC_COLUMNS
                    },
                    mean_seconds=float(np.mean([r.elapsed_seconds for r in results])),
                )
            )
        return sorted(
            summaries,
            key=lambda s: (
                math.inf
                if not np.isfinite(s.ranking_key.median)
                else s.ranking_key.median
            ),
        )

    def best_family_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for comparison in self.comparisons:
            name = comparison.best.family_name
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    # ------------------------------------------------------------ reports
    def table(self) -> str:
        header = (
            f"{'#':>2}  {'family':<44} {'k':>3} {'best%':>6} {'rank':>5} "
            f"{'nrmse med':>9} {'nrmse p90':>9} {'nrmse max':>9} {'irel med':>9} "
            f"{'bic med':>12} {'ok%':>5} {'s/fit':>6}"
        )
        lines = [header, "-" * len(header)]
        for index, s in enumerate(self.summary(), start=1):
            nrmse = s.metrics["nrmse"]
            lines.append(
                f"{index:>2}  {s.family:<44} {s.n_parameters:>3} "
                f"{100 * s.best_fraction:>6.1f} {s.mean_rank:>5.2f} "
                f"{nrmse.median:>9.4f} {nrmse.p90:>9.4f} {nrmse.max:>9.4f} "
                f"{s.metrics['integrated_relative_error'].median:>9.4f} "
                f"{s.metrics['bic'].median:>12.1f} {100 * s.success_rate:>5.0f} "
                f"{s.mean_seconds:>6.2f}"
            )
        return "\n".join(lines)

    def to_dict(self, include_profiles: bool = False) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "rank_by": self.rank_by,
            "sampling": self.sampling,
            "ranges": None if self.ranges is None else self.ranges.to_dict(),
            "n_points": self.n_points,
            "best_family_counts": self.best_family_counts(),
            "summary": [s.to_dict() for s in self.summary()],
            "points": [c.to_dict(include_profiles) for c in self.comparisons],
        }

    def rows(self) -> list[dict[str, Any]]:
        """One flat record per (point, family): conditions, metrics, parameters."""
        rows = []
        for point, comparison in enumerate(self.comparisons):
            for rank, result in enumerate(comparison, start=1):
                row: dict[str, Any] = {
                    "point": point,
                    **result.conditions.to_dict(),
                    "source": result.lineout.source,
                    "family": result.family_name,
                    "rank": rank,
                    "success": result.success,
                    "n_parameters": result.goodness.n_parameters,
                    **{name: getattr(result.goodness, name) for name in METRIC_COLUMNS},
                    "amplitude": result.amplitude,
                    "amplitude_units": result.lineout.density_units,
                    "elapsed_seconds": result.elapsed_seconds,
                }
                row.update(result.parameters)
                rows.append(row)
        return rows

    def write_csv(self, path: Path) -> Path:
        rows = self.rows()
        columns: list[str] = []
        for row in rows:
            columns.extend(name for name in row if name not in columns)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    # --------------------------------------------------------------- plot
    def plot(
        self,
        *,
        condition_axis: str = "pressure_bar",
        top: int = 3,
        title: str | None = None,
    ) -> Figure:
        """NRMSE distribution and rank-1 frequency per family, and NRMSE vs one condition."""
        summaries = self.summary()
        order = [s.family for s in summaries]
        positions = np.arange(len(order))
        fig = plt.figure(figsize=(13.0, 9.0))
        grid = fig.add_gridspec(
            2,
            2,
            height_ratios=[1.4, 1.0],
            width_ratios=[1.0, 0.7],
            hspace=0.35,
            wspace=0.06,
            left=0.24,
            right=0.97,
            top=0.92,
            bottom=0.07,
        )
        ax_box = fig.add_subplot(grid[0, 0])
        ax_bar = fig.add_subplot(grid[0, 1], sharey=ax_box)
        ax_scatter = fig.add_subplot(grid[1, :])

        # Both top panels list the families best-first from the top, sharing one label column.
        data = [
            [result.goodness.nrmse for _, result in self.results_of(name)]
            for name in order
        ]
        ax_box.boxplot(
            data,
            positions=positions,
            orientation="horizontal",
            widths=0.55,
            patch_artist=True,
            boxprops={"facecolor": "#cfe0f6", "edgecolor": SERIES_COLORS[0], "lw": 1.0},
            medianprops={"color": DATA_COLOR, "lw": 1.2},
            whiskerprops={"color": SERIES_COLORS[0], "lw": 1.0},
            capprops={"color": SERIES_COLORS[0], "lw": 1.0},
            flierprops={
                "marker": ".",
                "markersize": 4,
                "markeredgecolor": SERIES_COLORS[0],
            },
        )
        ax_box.set_xscale("log")
        ax_box.set_yticks(positions)
        ax_box.set_yticklabels(order, fontsize=8)
        ax_box.invert_yaxis()
        ax_box.set_xlabel("NRMSE (rmse / peak)")
        ax_box.set_title(f"fit error over {self.n_points} points", fontsize=10)

        fractions = [100.0 * s.best_fraction for s in summaries]
        ax_bar.barh(positions, fractions, color=SERIES_COLORS[0], height=0.6)
        ax_bar.tick_params(axis="y", labelleft=False, length=0)
        ax_bar.set_xlabel(f"ranked first by {self.rank_by} [% of points]")
        ax_bar.set_xlim(0, 100)
        for position, fraction in zip(positions, fractions, strict=True):
            if fraction > 0:
                ax_bar.text(
                    fraction + 1, position, f"{fraction:.0f}", va="center", fontsize=7.5
                )

        shown = order[: max(1, min(top, len(SERIES_COLORS)))]
        for color, name in zip(SERIES_COLORS, shown, strict=False):
            results = [result for _, result in self.results_of(name)]
            values = [getattr(r.conditions, condition_axis) for r in results]
            ax_scatter.scatter(
                values,
                [r.goodness.nrmse for r in results],
                s=18,
                color=color,
                label=name,
                edgecolors="white",
                linewidths=0.6,
            )
        ax_scatter.set_yscale("log")
        ax_scatter.set_xlabel(AXIS_LABELS.get(condition_axis, condition_axis))
        ax_scatter.set_ylabel("NRMSE")
        ax_scatter.set_title(
            f"best {len(shown)} families by median {self.rank_by}", fontsize=10
        )
        ax_scatter.legend(fontsize=8, frameon=False, loc="best")

        for ax in (ax_box, ax_bar, ax_scatter):
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(title or f"{self.dataset_name}: fitting performance", fontsize=12)
        return fig


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_dataset_arguments(parser)
    add_window_arguments(parser)
    add_fit_arguments(parser)
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="x range [mm]; default: dataset extent. Equal values fix the axis.",
    )
    group.add_argument(
        "--pressure-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Pressure range [bar]; default: dataset extent.",
    )
    group.add_argument(
        "--angle-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Lineout angle range [deg]; default: fixed at 0 (lineouts along z).",
    )
    group.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Number of condition points; default 16. A grid rounds up to a lattice.",
    )
    group.add_argument(
        "--sampling",
        choices=SAMPLING_METHODS,
        default="lhs",
        help="lhs (default), random, or grid.",
    )
    group.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed of the point sample; default 0.",
    )
    group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes fitting points in parallel; default 1.",
    )
    group = parser.add_argument_group("output")
    group.add_argument(
        "--json", type=Path, help="Write the summary and every fit as JSON."
    )
    group.add_argument(
        "--include-profiles",
        action="store_true",
        help="Include each fit's serialized FBPIC profile config in the JSON.",
    )
    group.add_argument(
        "--csv", type=Path, help="Write one row per (point, family) as CSV."
    )
    group.add_argument("--plot", type=Path, help="Save the summary figure here.")
    group.add_argument("--show", action="store_true", help="Show the summary figure.")
    group.add_argument(
        "--plot-axis",
        choices=CONDITION_AXES,
        default="pressure_bar",
        help="Condition on the x axis of the NRMSE scatter; default pressure_bar.",
    )
    group.add_argument("-t", "--title", help="Figure title.")
    group.add_argument(
        "-q", "--quiet", action="store_true", help="No per-point progress."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if handle_list_families(args):
        return 0
    dataset = load_dataset(args, parser)
    ranges = ConditionRanges.from_dataset(
        dataset,
        x_range=args.x_range,
        pressure_range=args.pressure_range,
        angle_range=args.angle_range,
    )
    points = ranges.sample(args.samples, args.sampling, seed=args.sample_seed)
    families = resolve_families(args)
    print(
        f"# {dataset.name}: {len(points)} {args.sampling} points over "
        + ", ".join(
            f"{name} in [{bounds[0]:g}, {bounds[1]:g}]"
            for name, bounds in ranges.axes.items()
        )
        + f"; {len(families)} families",
        file=sys.stderr,
    )
    comparisons = fit_points(
        dataset,
        points,
        families,
        build_scheme(args),
        fit_window_from_args(args),
        rank_by=args.rank_by,
        workers=args.workers,
        progress=not args.quiet,
    )
    if not comparisons:
        print("no lineouts fitted")
        return 1
    statistics = FitStatistics(
        comparisons, ranges=ranges, sampling=args.sampling, dataset_name=dataset.name
    )
    print(statistics.table())
    print(
        "# best family counts: "
        + ", ".join(
            f"{name} x{count}"
            for name, count in statistics.best_family_counts().items()
        )
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(statistics.to_dict(args.include_profiles), indent=2)
        )
        print(f"wrote {args.json}")
    if args.csv is not None:
        print(f"wrote {statistics.write_csv(args.csv)}")
    if args.plot is not None or args.show:
        figure = statistics.plot(condition_axis=args.plot_axis, title=args.title)
        if args.plot is not None:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.plot, dpi=150, bbox_inches="tight")
            print(f"wrote {args.plot}")
        if args.show:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
