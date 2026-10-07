"""Command line tools for preparing and running staged exploration campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import Campaign
from .exploration import XoptFBPICExplorer


DEFAULT_FEATURES = (
    "total_beam_charge_pc",
    "mean_uz",
    "cov_x_x",
    "cov_y_y",
    "cov_uz_uz",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="FBPIC Bayesian exploration campaign tools")
    parser.add_argument("config", type=Path, help="campaign YAML configuration")
    parser.add_argument("--stage", default="stage_1")
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    subparsers = parser.add_subparsers(dest="command", required=True)
    initial = subparsers.add_parser("initial-design", help="write a Sobol-plus-anchor initial design")
    initial.add_argument("--count", type=int, help="default: stage initial_design_size")
    initial.add_argument("--seed", type=int, default=0)
    initial.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate", help="run one FBPIC point locally")
    evaluate.add_argument("--point", required=True, help="JSON mapping of active parameter values")
    subparsers.add_parser("status", help="show reusable successful evaluations")
    args = parser.parse_args()

    explorer = XoptFBPICExplorer(Campaign.from_file(args.config), args.stage, args.features)
    if args.command == "initial-design":
        count = args.count if args.count is not None else explorer.campaign.config.stage_settings(args.stage).initial_design_size
        points = explorer.initial_design(count, args.seed)
        args.output.write_text(json.dumps(points, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(points)} initial points to {args.output}")
    elif args.command == "evaluate":
        print(json.dumps(explorer.evaluate(json.loads(args.point)), indent=2))
    else:
        data = explorer.training_data()
        print(json.dumps({"stage": args.stage, "reusable_successful_rows": len(data), "active_parameters": list(explorer.variable_bounds)}, indent=2))


if __name__ == "__main__":
    main()
