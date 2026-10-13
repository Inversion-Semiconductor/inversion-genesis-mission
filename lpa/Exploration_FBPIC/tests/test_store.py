from pathlib import Path
import json

import pandas as pd

from exploration_fbpic.campaign import Campaign
from exploration_fbpic.executor import EvaluationResult
from exploration_fbpic.store import CampaignStore


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_store_persists_full_parameter_vector(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    run_directory = tmp_path / "sim_000001"
    run_directory.mkdir()
    campaign.make_manifest("sim_000001", "stage_1").write(run_directory)
    result = EvaluationResult("sim_000001", "ok", {"total_beam_charge_pc": 5.0}, run_directory)

    store = CampaignStore(tmp_path)
    store.append(result)

    row = store.load().iloc[0]
    assert row["p:laser_energy_J"] == 2.5
    assert row["p:zernike_secondary_trefoil_x"] == 0.0
    assert row["f:total_beam_charge_pc"] == 5.0


def test_store_rebuilds_from_per_run_results(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    run_directory = tmp_path / "sim_000001"
    run_directory.mkdir()
    manifest = campaign.make_manifest("sim_000001", "stage_1")
    manifest.write(run_directory)
    result = EvaluationResult("sim_000001", "ok", {"total_beam_charge_pc": 5.0}, run_directory)
    (run_directory / "result.json").write_text(
        json.dumps({"manifest": manifest.__dict__, "result": result.as_dict()}, default=str),
        encoding="utf-8",
    )

    store = CampaignStore(tmp_path)
    rebuilt = store.rebuild()

    assert len(rebuilt) == 1
    assert store.load().iloc[0]["f:total_beam_charge_pc"] == 5.0


def test_store_load_repairs_corrupt_parquet_from_run_results(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    run_directory = tmp_path / "sim_000001"
    run_directory.mkdir()
    manifest = campaign.make_manifest("sim_000001", "stage_1")
    manifest.write(run_directory)
    result = EvaluationResult("sim_000001", "ok", {"total_beam_charge_pc": 5.0}, run_directory)
    (run_directory / "result.json").write_text(
        json.dumps({"manifest": manifest.__dict__, "result": result.as_dict()}, default=str),
        encoding="utf-8",
    )
    store = CampaignStore(tmp_path)
    store.path.write_bytes(b"not a parquet file")

    repaired = store.load()

    assert len(repaired) == 1
    assert pd.read_parquet(store.path).iloc[0]["f:total_beam_charge_pc"] == 5.0


def test_store_append_repairs_corrupt_parquet_from_run_results(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    store = CampaignStore(tmp_path)
    first_directory = tmp_path / "sim_000001"
    first_directory.mkdir()
    first_manifest = campaign.make_manifest("sim_000001", "stage_1")
    first_manifest.write(first_directory)
    first_result = EvaluationResult("sim_000001", "ok", {"total_beam_charge_pc": 5.0}, first_directory)
    (first_directory / "result.json").write_text(
        json.dumps({"manifest": first_manifest.__dict__, "result": first_result.as_dict()}, default=str),
        encoding="utf-8",
    )
    store.append(first_result)
    store.path.write_bytes(b"corrupt")

    second_directory = tmp_path / "sim_000002"
    second_directory.mkdir()
    second_manifest = campaign.make_manifest("sim_000002", "stage_1")
    second_manifest.write(second_directory)
    second_result = EvaluationResult("sim_000002", "ok", {"total_beam_charge_pc": 6.0}, second_directory)
    (second_directory / "result.json").write_text(
        json.dumps({"manifest": second_manifest.__dict__, "result": second_result.as_dict()}, default=str),
        encoding="utf-8",
    )
    store.append(second_result)

    repaired = store.load()

    assert set(repaired["run_id"]) == {"sim_000001", "sim_000002"}
