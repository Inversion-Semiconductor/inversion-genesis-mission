"""Command line tools for preparing and running staged exploration campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .campaign import Campaign
from .campaign_analysis import (
    analyze_campaign,
    plot_charge_slice,
    plot_feature_slice,
    plot_feasibility_slice,
    plot_input_pca_projections,
)
from .density_envelope import plot_density_envelope
from .density_ramps import plot_density_ramp_parameter_space, plot_density_ramps
from .executor import FBPICRunExecutor
from .exploration import XoptFBPICExplorer
from .libensemble_driver import preview_xopt_batch
from .store import CampaignStore
from inversion_fbpic.utils.virtual_waist import shift_run_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FBPIC Bayesian exploration campaign tools"
    )
    parser.add_argument("config", type=Path, help="campaign YAML configuration")
    parser.add_argument("--stage", default="stage_1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initial = subparsers.add_parser(
        "initial-design", help="write a Sobol-plus-anchor initial design"
    )
    initial.add_argument("--count", type=int, help="default: stage initial_design_size")
    initial.add_argument("--seed", type=int, default=0)
    initial.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate", help="run one FBPIC point locally")
    evaluate.add_argument(
        "--point", required=True, help="JSON mapping of active parameter values"
    )
    evaluate.add_argument(
        "--run-id", help="explicit run ID; required for concurrent dispatch"
    )
    density_ramps = subparsers.add_parser(
        "plot-density-ramps",
        help="plot the four lower/upper left density-ramp extremes",
    )
    density_ramps.add_argument(
        "--simulation-half-length",
        type=float,
        required=True,
        metavar="METRES",
        help="left plotting extent in metres; sets density_center_location_m",
    )
    density_ramps.add_argument("--output", type=Path, required=True)
    density_ramps.add_argument(
        "--parameter-space-output",
        type=Path,
        help="default: OUTPUT with '-parameter-space' appended to its stem",
    )
    density_ramps.add_argument("--samples", type=int, default=1000)
    density_envelope = subparsers.add_parser(
        "plot-density-envelope",
        help="plot the nominal density profile with random campaign samples",
    )
    density_envelope.add_argument("--output", type=Path, required=True)
    density_envelope.add_argument("--random-samples", type=int, default=20)
    density_envelope.add_argument("--seed", type=int, default=0)
    density_envelope.add_argument("--samples", type=int, default=1000)
    analyze = subparsers.add_parser(
        "analyze-campaign",
        help="assess compact scalar results and write a JSON report plus PCA plots",
    )
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument(
        "--data",
        type=Path,
        help="explicit campaign_runs.parquet path; use for a compact local export",
    )
    analyze.add_argument("--folds", type=int, default=5)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument(
        "--vector-sim",
        type=int,
        help="mark sim_000000 and this simulation on both campaign PCA panels",
    )
    analyze.add_argument(
        "--input-pca-x",
        type=int,
        default=1,
        help="one-based input-space PCA component for the horizontal axis",
    )
    analyze.add_argument(
        "--input-pca-y",
        type=int,
        default=2,
        help="one-based input-space PCA component for the vertical axis",
    )
    charge_slice = subparsers.add_parser(
        "plot-charge-slice",
        help="plot charge observations plus the Xopt GP mean and uncertainty slice",
    )
    charge_slice.add_argument("--output", type=Path, required=True)
    charge_slice.add_argument("--data", type=Path)
    charge_slice.add_argument("--grid-size", type=int, default=100)
    feature_slice = subparsers.add_parser(
        "plot-feature-slice",
        help="plot one feature's observations plus Xopt GP mean and uncertainty slice",
    )
    feature_slice.add_argument("--feature", required=True)
    feature_slice.add_argument("--output", type=Path, required=True)
    feature_slice.add_argument("--data", type=Path)
    feature_slice.add_argument("--grid-size", type=int, default=100)
    feasibility_slice = subparsers.add_parser(
        "plot-feasibility-slice",
        help="plot beam existence probability and injection uncertainty slice",
    )
    feasibility_slice.add_argument("--output", type=Path, required=True)
    feasibility_slice.add_argument("--data", type=Path)
    feasibility_slice.add_argument("--grid-size", type=int, default=100)
    input_projections = subparsers.add_parser(
        "plot-input-pca-projections",
        help="write higher-component input PCA projections with feasibility markers",
    )
    input_projections.add_argument("--output", type=Path, required=True)
    input_projections.add_argument("--data", type=Path)
    preview = subparsers.add_parser(
        "preview-batch",
        help="log a no-submit Xopt/feasibility batch proposal",
    )
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--data", type=Path)
    preview.add_argument("--count", type=int, default=32)
    preview.add_argument("--candidate-pool-factor", type=int, default=8)
    preview.add_argument("--minimum-beam-feasibility", type=float)
    subparsers.add_parser("status", help="show reusable successful evaluations")
    rebuild = subparsers.add_parser(
        "rebuild-store",
        help="rebuild campaign_runs.parquet from per-run result.json files",
    )
    rebuild.add_argument(
        "--result-file",
        help="per-run result filename; defaults to the configured shifted file when enabled",
    )
    rebuild.add_argument(
        "--output",
        type=Path,
        help="aggregate output path; defaults to campaign_runs.parquet",
    )
    rebuild.add_argument(
        "--run-root",
        type=Path,
        help="override the stage run root, for example a downloaded NERSC campaign",
    )
    shift_results = subparsers.add_parser(
        "shift-results",
        help="write virtual-waist-shifted result records for all successful runs",
    )
    shift_results.add_argument(
        "--run-root",
        type=Path,
        help="override the stage run root, for example a downloaded NERSC campaign",
    )
    args = parser.parse_args()

    explorer = XoptFBPICExplorer(Campaign.from_file(args.config), args.stage)
    if args.command == "analyze-campaign":
        analysis = analyze_campaign(
            explorer.campaign,
            args.stage,
            args.output,
            args.folds,
            args.seed,
            args.data,
            args.vector_sim,
            args.input_pca_x,
            args.input_pca_y,
        )
        print(
            json.dumps(
                {
                    "report": str(analysis.report_path),
                    "projection": str(analysis.plot_path),
                    "complete_training_rows": analysis.report.get(
                        "complete_training_rows", 0
                    ),
                },
                indent=2,
            )
        )
    elif args.command == "plot-charge-slice":
        output = plot_charge_slice(
            explorer.campaign, args.stage, args.output, args.data, args.grid_size
        )
        print(f"wrote charge GP slice to {output}")
    elif args.command == "plot-feature-slice":
        output = plot_feature_slice(
            explorer.campaign,
            args.stage,
            args.output,
            args.feature,
            args.data,
            args.grid_size,
        )
        print(f"wrote {args.feature} GP slice to {output}")
    elif args.command == "plot-feasibility-slice":
        output = plot_feasibility_slice(
            explorer.campaign, args.stage, args.output, args.data, args.grid_size
        )
        print(f"wrote feasibility slice to {output}")
    elif args.command == "plot-input-pca-projections":
        outputs = plot_input_pca_projections(
            explorer.campaign, args.stage, args.output, args.data
        )
        print(
            json.dumps({"projections": [str(output) for output in outputs]}, indent=2)
        )
    elif args.command == "preview-batch":
        data = (
            pd.read_parquet(args.data)
            if args.data is not None
            else explorer.store.load()
        )
        preview = preview_xopt_batch(
            explorer,
            data,
            args.count,
            args.minimum_beam_feasibility,
            args.candidate_pool_factor,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(preview, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "candidate_pool_count": preview["candidate_pool_count"],
                    "selected_count": len(preview["selected"]),
                },
                indent=2,
            )
        )
    elif args.command == "plot-density-ramps":
        campaign = Campaign.from_file(args.config)
        output = plot_density_ramps(
            campaign,
            args.simulation_half_length,
            args.output,
            args.samples,
        )
        parameter_space_output = args.parameter_space_output
        if parameter_space_output is None:
            parameter_space_output = args.output.with_name(
                f"{args.output.stem}-parameter-space{args.output.suffix}"
            )
        plot_density_ramp_parameter_space(
            campaign, args.simulation_half_length, parameter_space_output
        )
        print(f"wrote density-ramp plot to {output}")
        print(f"wrote density-ramp parameter-space plot to {parameter_space_output}")
    elif args.command == "plot-density-envelope":
        output = plot_density_envelope(
            Campaign.from_file(args.config),
            args.stage,
            args.output,
            args.random_samples,
            args.seed,
            args.samples,
        )
        print(f"wrote density-profile envelope to {output}")
    elif args.command == "initial-design":
        count = (
            args.count
            if args.count is not None
            else explorer.campaign.config.stage_settings(args.stage).initial_design_size
        )
        points = explorer.initial_design(count, args.seed)
        args.output.write_text(json.dumps(points, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(points)} initial points to {args.output}")
    elif args.command == "evaluate":
        explorer.require_compatible_run_root()
        point = json.loads(args.point)
        if args.run_id is None:
            result = FBPICRunExecutor(explorer.campaign).evaluate_next(
                args.stage, point
            )
        else:
            result = FBPICRunExecutor(explorer.campaign).evaluate(
                args.run_id, args.stage, point
            )
        explorer.refresh_store()
        print(json.dumps(result.as_dict(), indent=2))
    elif args.command == "rebuild-store":
        result_file = (
            args.result_file or explorer.campaign.config.campaign_result_filename
        )
        run_root = args.run_root or explorer.campaign.config.run_root
        output = args.output or run_root / "campaign_runs.parquet"
        rebuilt = CampaignStore(run_root).rebuild(
            result_filename=result_file,
            output_path=output,
        )
        print(
            json.dumps(
                {
                    "rebuilt_rows": len(rebuilt),
                    "result_file": result_file,
                    "output": str(output),
                },
                indent=2,
            )
        )
    elif args.command == "shift-results":
        settings = explorer.campaign.config.virtual_waist_shift
        if not settings["enabled"]:
            raise ValueError(
                "analysis.virtual_waist_shift must be enabled to shift campaign results"
            )
        run_root = args.run_root or explorer.campaign.config.run_root
        completed = []
        failed = []
        skipped = 0
        for run_directory in sorted(
            path for path in run_root.glob("sim_*") if path.is_dir()
        ):
            result_path = run_directory / "result.json"
            if not result_path.is_file():
                skipped += 1
                continue
            record = json.loads(result_path.read_text(encoding="utf-8"))
            if record["result"]["status"] != "ok":
                skipped += 1
                continue
            try:
                _, metadata, output = shift_run_result(
                    run_directory,
                    species=settings["species"],
                    target_z_m=settings["target_z_m"],
                    uz_min=settings["uz_min"],
                    central_fraction=settings["central_fraction"],
                    charge_fraction=settings["charge_fraction"],
                    output_filename=settings["result_filename"],
                )
            except (FileNotFoundError, ValueError, OSError) as error:
                failed.append({"run_id": run_directory.name, "error": str(error)})
            else:
                completed.append(
                    {
                        "run_id": run_directory.name,
                        "result": str(output),
                        "waist_average_m": metadata["waist_average_m"],
                    }
                )
        print(
            json.dumps(
                {
                    "run_root": str(run_root),
                    "shifted": len(completed),
                    "skipped": skipped,
                    "failed": failed,
                },
                indent=2,
            )
        )
        if failed:
            raise SystemExit(1)
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
