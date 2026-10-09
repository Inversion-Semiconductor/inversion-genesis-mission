#!/usr/bin/env python3
"""Evaluate models that predict a profile family's parameters from (x, pressure, angle).

Examples (conda env inv-fbpic):

    python -m fludat_fit.evaluate_models htu_dens_7_0.h5 --family conical:supergaussian \\
        --x-range 0 3 --angle-range -10 10 --samples 64 --workers 4 \\
        --training-set htu_sg.npz --json models.json --csv models.csv --plot models.png
    python -m fludat_fit.evaluate_models --training-set htu_sg.npz --models poly1 knn3 \\
        --objectives direct --folds 0 --test-fraction 0.3 --refine

The training set is built once (lineouts at sampled conditions, each fitted by
the local multi-start optimiser to give the *indirect* targets) and cached in
``--training-set``. Every requested model is then trained under every
requested objective (*direct*: the local optimiser's own fit loss of the
predicted parameters; *indirect*: distance to the local optimum) and scored on
held-out conditions by the fit quality of its predictions, its parameter
error, and, with ``--refine``, how much local optimisation it saves as a warm
start. The built-in models are baselines; plug in your own through the library.
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
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from .cli_common import (
    add_dataset_arguments,
    add_window_arguments,
    fit_window_from_args,
    load_dataset,
)
from .families import DEFAULT_FAMILY_NAMES, get_families
from .fitting import MultiStartLocalFit
from .learning import (
    OBJECTIVES,
    TrainingSet,
    build_training_set,
    builtin_model_factories,
    compare_models,
    get_model_factories,
    get_objective,
)
from .learning.plotting import plot_model_comparison
from .sampling import SAMPLING_METHODS, ConditionRanges


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_dataset_arguments(parser)
    add_window_arguments(parser)

    group = parser.add_argument_group("training set")
    group.add_argument(
        "--family",
        default="conical:supergaussian",
        help="Profile family whose parameters are learned; default conical:supergaussian.",
    )
    group.add_argument(
        "--training-set",
        type=Path,
        help="Cache file (.npz). Loaded when it exists (then the cube is optional); "
        "otherwise the set is built from the cube and saved here.",
    )
    group.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="x range [mm]; default: cube extent.",
    )
    group.add_argument(
        "--pressure-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Pressure range [bar]; default: cube extent.",
    )
    group.add_argument(
        "--angle-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Lineout angle range [deg]; default: fixed at 0.",
    )
    group.add_argument(
        "--samples", type=int, default=32, help="Condition points; default 32."
    )
    group.add_argument(
        "--sampling",
        choices=SAMPLING_METHODS,
        default="lhs",
        help="lhs (default), random, grid.",
    )
    group.add_argument(
        "--sample-seed", type=int, default=0, help="Point-sample seed; default 0."
    )
    group.add_argument(
        "--starts",
        type=int,
        default=8,
        help="Local-fit starting points per sample; default 8.",
    )
    group.add_argument(
        "--optimizer",
        default="L-BFGS-B",
        help="Local-fit scipy method; default L-BFGS-B.",
    )
    group.add_argument(
        "--no-targets",
        action="store_true",
        help="Skip the local fits (direct objective only, no reference or indirect metrics).",
    )
    group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes for the local fits; default 1.",
    )

    group = parser.add_argument_group("evaluation")
    group.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        help=f"Built-in models to evaluate; default all: {' '.join(builtin_model_factories())}.",
    )
    group.add_argument(
        "--objectives",
        nargs="+",
        choices=tuple(OBJECTIVES),
        default=list(OBJECTIVES),
        help="Training objectives; default both.",
    )
    group.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Cross-validation folds; 0 for one split. Default 4.",
    )
    group.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Test fraction with --folds 0; default 0.25.",
    )
    group.add_argument("--seed", type=int, default=0, help="Split seed; default 0.")
    group.add_argument(
        "--refine",
        action="store_true",
        help="Also polish each prediction with a local optimiser and report the savings.",
    )

    group = parser.add_argument_group("output")
    group.add_argument("--json", type=Path, help="Write the summaries as JSON.")
    group.add_argument(
        "--csv", type=Path, help="Write one row per (model, objective, test sample)."
    )
    group.add_argument("--plot", type=Path, help="Save the comparison figure.")
    group.add_argument(
        "--show", action="store_true", help="Show the comparison figure."
    )
    group.add_argument("-t", "--title", help="Figure title.")
    group.add_argument("-q", "--quiet", action="store_true", help="No progress output.")
    group.add_argument(
        "--list-families", action="store_true", help="Print family names and exit."
    )
    return parser


def load_or_build_training_set(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> TrainingSet:
    if args.training_set is not None and args.training_set.is_file():
        training_set = TrainingSet.load(args.training_set)
        if (
            training_set.family.name != args.family
            and args.family != parser.get_default("family")
        ):
            parser.error(
                f"{args.training_set} holds {training_set.family.name!r}, not {args.family!r}"
            )
        if not args.quiet:
            print(
                f"# loaded {len(training_set)} samples from {args.training_set}",
                file=sys.stderr,
            )
        return training_set

    dataset = load_dataset(args, parser)
    family = get_families([args.family])[0]
    ranges = ConditionRanges.from_dataset(
        dataset,
        x_range=args.x_range,
        pressure_range=args.pressure_range,
        angle_range=args.angle_range,
    )
    points = ranges.sample(args.samples, args.sampling, seed=args.sample_seed)
    if not args.quiet:
        print(
            f"# {dataset.name}: building {len(points)} {args.sampling} samples of "
            f"{family.name}"
            + ("" if args.no_targets else f" with local fits ({args.starts} starts)"),
            file=sys.stderr,
        )
    training_set = build_training_set(
        dataset,
        points,
        family,
        ranges=ranges,
        window=fit_window_from_args(args),
        scheme=MultiStartLocalFit(n_starts=args.starts, method=args.optimizer),
        fit_targets=not args.no_targets,
        workers=args.workers,
        progress=not args.quiet,
    )
    if args.training_set is not None:
        training_set.save(args.training_set)
        if not args.quiet:
            print(f"# saved training set to {args.training_set}", file=sys.stderr)
    return training_set


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_families:
        print("\n".join(DEFAULT_FAMILY_NAMES))
        return 0

    try:
        factories = get_model_factories(args.models)
    except KeyError as exc:
        parser.error(str(exc))
    objectives = [get_objective(name) for name in args.objectives]
    training_set = load_or_build_training_set(args, parser)
    comparison = compare_models(
        factories,
        training_set,
        objectives,
        folds=args.folds if args.folds > 0 else None,
        test_fraction=args.test_fraction,
        seed=args.seed,
        refine=args.refine,
    )
    print(
        f"# {training_set.name}: {training_set.family.name}, {len(training_set)} samples, "
        f"parameters {list(training_set.space.names)}"
    )
    print(comparison.table())

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(comparison.to_dict(), indent=2))
        print(f"wrote {args.json}")
    if args.csv is not None:
        rows = comparison.rows()
        columns: list[str] = []
        for row in rows:
            columns.extend(name for name in row if name not in columns)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    if args.plot is not None or args.show:
        figure = plot_model_comparison(comparison, title=args.title)
        if args.plot is not None:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.plot, dpi=150, bbox_inches="tight")
            print(f"wrote {args.plot}")
        if args.show:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
