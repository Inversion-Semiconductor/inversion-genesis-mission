"""Command line tools for preparing and running staged exploration campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import Campaign
from .executor import FBPICRunExecutor
from .exploration import XoptFBPICExplorer
from .store import CampaignStore


def main() -> None:
    parser = argparse.ArgumentParser(description="FBPIC Bayesian exploration campaign tools")
    parser.add_argument("config", type=Path, help="campaign YAML configuration")
    parser.add_argument("--stage", default="stage_1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initial = subparsers.add_parser("initial-design", help="write a Sobol-plus-anchor initial design")
    initial.add_argument("--count", type=int, help="default: stage initial_design_size")
    initial.add_argument("--seed", type=int, default=0)
    initial.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate", help="run one FBPIC point locally")
    evaluate.add_argument("--point", required=True, help="JSON mapping of active parameter values")
    evaluate.add_argument("--run-id", help="explicit run ID; required for concurrent dispatch")
    status = subparsers.add_parser("status", help="show reusable successful evaluations")
    args = parser.parse_args()

    explorer = XoptFBPICExplorer(Campaign.from_file(args.config), args.stage)
    if args.command == "initial-design":
        count = args.count if args.count is not None else explorer.campaign.config.stage_settings(args.stage).initial_design_size
        points = explorer.initial_design(count, args.seed)
        args.output.write_text(json.dumps(points, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(points)} initial points to {args.output}")
    elif args.command == "evaluate":
        point = json.loads(args.point)
        if args.run_id is None:
            print(json.dumps(explorer.evaluate(point), indent=2))
        else:
            result = FBPICRunExecutor(explorer.campaign).evaluate(args.run_id, args.stage, point)
            CampaignStore(explorer.campaign.config.run_root).append(result)
            print(json.dumps(result.as_dict(), indent=2))
    else:
        data = explorer.training_data()
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "reusable_successful_rows": len(data),
                    "active_parameters": list(explorer.variable_bounds),
                    "exploration_features": list(explorer.feature_names),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
