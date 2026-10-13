"""Argument groups and helpers shared by the command-line tools.

Both ``explore_fits`` and ``fit_statistics`` take the same density cube,
fit-window and fitting options; this module defines them once and turns parsed
arguments into the library objects.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import NozzleDataset
from .families import DEFAULT_FAMILY_NAMES, ProfileFamily, get_families
from .fitting import FamilyComparison, FitResult, MultiStartLocalFit
from .goodness_of_fit import DEFAULT_RANKING_METRIC, RANKING_METRICS
from .lineout import DEFAULT_CUTOFF_RATIO, InterpolationMethod, Lineout

MM_PER_M = 1000.0
INTERPOLATION_METHODS: tuple[InterpolationMethod, ...] = (
    "linear",
    "cubic",
    "quintic",
    "pchip",
)
DEFAULT_MAX_POINTS = 1000


@dataclass(frozen=True)
class FitWindow:
    """How a raw lineout is cropped and resampled before fitting.

    Args:
        z_bounds: Explicit window in the lineout's longitudinal coordinate [m];
            when ``None`` the lineout is trimmed to its jet support at
            ``cutoff_ratio`` plus ``padding_fraction`` of vacuum on each side.
        cutoff_ratio: Peak fraction defining the jet support.
        padding_fraction: Vacuum kept on each side as a fraction of the support width.
        max_points: Resample lineouts with more samples than this; ``None`` keeps all.
    """

    z_bounds: tuple[float, float] | None = None
    cutoff_ratio: float = DEFAULT_CUTOFF_RATIO
    padding_fraction: float = 0.25
    max_points: int | None = DEFAULT_MAX_POINTS

    def apply(self, lineout: Lineout) -> Lineout:
        if self.z_bounds is not None:
            windowed = lineout.cropped(self.z_bounds)
        elif lineout.peak > 0.0:
            windowed = lineout.trimmed(self.cutoff_ratio, self.padding_fraction)
        else:
            windowed = lineout
        if self.max_points is not None:
            windowed = windowed.resampled(self.max_points)
        return windowed


# ------------------------------------------------------------- argument groups
def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "hdf5_path", nargs="?", type=Path, help="Density cube (fludat_proc HDF5)."
    )
    group.add_argument(
        "--method",
        choices=INTERPOLATION_METHODS,
        default="linear",
        help="(x, pressure) interpolation method; default linear (fast trilinear path).",
    )


def add_window_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("fit window")
    group.add_argument(
        "--z-bounds",
        type=float,
        nargs=2,
        metavar=("MIN_MM", "MAX_MM"),
        help="Fit window along the lineout [mm]; default: the jet support plus padding.",
    )
    group.add_argument(
        "--cutoff-ratio",
        type=float,
        default=DEFAULT_CUTOFF_RATIO,
        help=f"Peak fraction defining the jet support; default {DEFAULT_CUTOFF_RATIO:g}.",
    )
    group.add_argument(
        "--padding",
        type=float,
        default=0.25,
        help="Vacuum kept on each side, as a fraction of the support width; default 0.25.",
    )
    group.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help=f"Resample longer lineouts to this many samples; default {DEFAULT_MAX_POINTS}. "
        "0 keeps every sample.",
    )


def add_fit_arguments(
    parser: argparse.ArgumentParser, *, default_starts: int = 8
) -> None:
    group = parser.add_argument_group("fitting")
    group.add_argument(
        "--families",
        nargs="+",
        metavar="NAME",
        help="Families to fit; default all. See --list-families.",
    )
    group.add_argument(
        "--list-families",
        action="store_true",
        help="Print the family names and exit.",
    )
    group.add_argument(
        "--starts",
        type=int,
        default=default_starts,
        help=f"Sampled starting points per family; default {default_starts}.",
    )
    group.add_argument(
        "--optimizer",
        default="L-BFGS-B",
        help="Bounded scipy.optimize.minimize method; default L-BFGS-B.",
    )
    group.add_argument(
        "--seed", type=int, default=0, help="Start-sample seed; default 0."
    )
    group.add_argument(
        "--rank-by",
        choices=tuple(RANKING_METRICS),
        default=DEFAULT_RANKING_METRIC,
        help=f"Ranking metric; default {DEFAULT_RANKING_METRIC}.",
    )


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("FBPIC config export")
    group.add_argument(
        "--nominal-density",
        type=float,
        help="[m^-3] nominal density for exported YAML configs; default: the fitted "
        "amplitude converted from the cube units when recognised.",
    )
    group.add_argument(
        "--species",
        default=None,
        help="Species for exported YAML configs (e.g. H, with ionization 0); default none.",
    )


# ------------------------------------------------------------------ resolvers
def handle_list_families(args: argparse.Namespace) -> bool:
    """Print the registry and return ``True`` when ``--list-families`` was given."""
    if getattr(args, "list_families", False):
        print("\n".join(DEFAULT_FAMILY_NAMES))
        return True
    return False


def load_dataset(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> NozzleDataset:
    if args.hdf5_path is None:
        parser.error("a density cube is required")
    return NozzleDataset.load(args.hdf5_path, method=args.method)


def fit_window_from_args(args: argparse.Namespace) -> FitWindow:
    z_bounds = None
    if args.z_bounds is not None:
        lower, upper = sorted(args.z_bounds)
        z_bounds = (lower / MM_PER_M, upper / MM_PER_M)
    return FitWindow(
        z_bounds=z_bounds,
        cutoff_ratio=args.cutoff_ratio,
        padding_fraction=args.padding,
        max_points=args.max_points if args.max_points > 0 else None,
    )


def resolve_families(args: argparse.Namespace) -> list[ProfileFamily]:
    return get_families(args.families)


def build_scheme(args: argparse.Namespace) -> MultiStartLocalFit:
    return MultiStartLocalFit(
        n_starts=args.starts, method=args.optimizer, seed=args.seed
    )


# ------------------------------------------------------------------- reporting
def longitudinal_label(lineout: Lineout) -> str:
    """Axis label for the lineout coordinate: ``z`` for axial lines, ``t`` otherwise."""
    if lineout.conditions.angle_deg == 0.0:
        return "z [mm]"
    return "t along the lineout line [mm]"


def describe_lineout(lineout: Lineout, dataset_name: str, rank_by: str) -> str:
    z_lo, z_hi = lineout.z_extent
    return (
        f"# {dataset_name}: {lineout.conditions.label()}, {lineout.n_points} points, "
        f"t in [{z_lo * MM_PER_M:.3f}, {z_hi * MM_PER_M:.3f}] mm, ranked by {rank_by}"
    )


def describe_best(result: FitResult) -> str:
    lines = [
        f"# best: {result.family_name}, amplitude {result.amplitude:.4g} "
        f"{result.lineout.density_units}"
    ]
    lines.extend(
        f"#   {name} = {value:.6g}" for name, value in result.parameters.items()
    )
    return "\n".join(lines)


def print_comparison(comparison: FamilyComparison, dataset_name: str) -> None:
    print(describe_lineout(comparison.lineout, dataset_name, comparison.rank_by))
    print(comparison.table())
    print(describe_best(comparison.best))
    print()


def export_profile_yaml(
    result: FitResult,
    directory: Path,
    *,
    nominal_density: float | None = None,
    species: str | None = None,
) -> Path:
    """Write ``result`` as an FBPIC YAML density-profile config and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    overrides: dict[str, Any] = {}
    if species is not None:
        overrides.update(species=species, ionization=0)
    profile = result.build_profile(nominal_density=nominal_density, **overrides)
    return profile.to_yaml_file(directory / f"{result_stem(result)}.yaml")


def result_stem(result: FitResult) -> str:
    conditions = result.conditions
    stem = (
        f"{result.family_name.replace(':', '_').replace('+', '_')}"
        f"_x{conditions.x_mm:g}mm_p{conditions.pressure_bar:g}bar"
        f"_a{conditions.angle_deg:g}deg"
    )
    return stem.replace("[", "").replace("]", "")
