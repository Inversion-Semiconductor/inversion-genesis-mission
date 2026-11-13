"""Prepare and finalize one manual candidate through the normal campaign executor.

The Slurm script runs the generated ``run_fbpic.py`` with one MPI rank per GPU
between the ``prepare`` and ``finalize`` commands. Finalization applies the
campaign's virtual-waist shift and rebuilds an isolated shifted campaign table.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from exploration_fbpic.campaign import Campaign
from exploration_fbpic.executor import FBPICRunExecutor
from exploration_fbpic.store import CampaignStore

CAMPAIGN_PATH = Path(__file__).parents[1] / "campaign.yaml"


def parse_args() -> argparse.Namespace:
    """Parse prepare/finalize workflow commands."""
    parser = argparse.ArgumentParser(
        description="Run one manual Stage 2 candidate with campaign persistence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "finalize"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--candidate", type=Path, required=True)
        subparser.add_argument("--run-root", type=Path, required=True)
        subparser.add_argument("--run-id", required=True)
    return parser.parse_args()


def load_candidate(path: Path) -> tuple[str, dict[str, float]]:
    """Load the stage and active-parameter mapping from a candidate JSON file."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    stage = payload.get("stage")
    parameters = payload.get("parameters")
    if not isinstance(stage, str):
        raise TypeError("candidate JSON requires a string 'stage'")
    if not isinstance(parameters, dict) or not all(
        isinstance(name, str) and isinstance(value, (int, float))
        for name, value in parameters.items()
    ):
        raise ValueError("candidate JSON requires numeric 'parameters' mapping")
    return stage, {name: float(value) for name, value in parameters.items()}


def one_off_campaign(run_root: Path) -> Campaign:
    """Load the standard campaign with an isolated one-off durable run root."""
    campaign = Campaign.from_file(CAMPAIGN_PATH)
    return Campaign(replace(campaign.config, run_root=run_root.resolve()))


def prepare(candidate_path: Path, run_root: Path, run_id: str) -> Path:
    """Write the normal campaign manifest and generated FBPIC entrypoint."""
    stage, parameters = load_candidate(candidate_path)
    executor = FBPICRunExecutor(one_off_campaign(run_root))
    run_directory, _ = executor.prepare_run(run_id, stage, parameters)
    return run_directory


def finalize(candidate_path: Path, run_root: Path, run_id: str) -> dict[str, Any]:
    """Extract native and shifted features after the MPI FBPIC process completes."""
    stage, parameters = load_candidate(candidate_path)
    campaign = one_off_campaign(run_root)
    executor = FBPICRunExecutor(campaign)
    run_directory = campaign.config.run_root / run_id
    manifest = campaign.make_manifest(run_id, stage, parameters)
    result = executor.complete_prepared_run(
        run_id, run_directory, manifest, lambda _: None
    )
    CampaignStore(campaign.config.run_root).rebuild(
        result_filename=campaign.config.campaign_result_filename
    )
    return result.as_dict()


def main() -> None:
    """Dispatch a one-off prepare or finalize command."""
    args = parse_args()
    if args.command == "prepare":
        print(prepare(args.candidate, args.run_root, args.run_id))
    else:
        print(
            json.dumps(finalize(args.candidate, args.run_root, args.run_id), indent=2)
        )


if __name__ == "__main__":
    main()
