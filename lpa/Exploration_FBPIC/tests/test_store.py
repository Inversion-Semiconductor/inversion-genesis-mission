from pathlib import Path
import json

import pandas as pd
import pytest

from exploration_fbpic.campaign import Campaign
from exploration_fbpic.executor import EvaluationResult
from exploration_fbpic.store import CampaignStore


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_store_persists_full_parameter_vector(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    run_directory = tmp_path / "sim_000001"
    run_directory.mkdir()
    campaign.make_manifest("sim_000001", "stage_1").write(run_directory)
    result = EvaluationResult(
        "sim_000001", "ok", {"total_beam_charge_pc": 5.0}, run_directory
    )
    manifest = campaign.make_manifest("sim_000001", "stage_1")
    (run_directory / "result.json").write_text(
        json.dumps(
            {"manifest": manifest.__dict__, "result": result.as_dict()}, default=str
        ),
        encoding="utf-8",
    )

    store = CampaignStore(tmp_path)
    store.rebuild()

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
    result = EvaluationResult(
        "sim_000001", "ok", {"total_beam_charge_pc": 5.0}, run_directory
    )
    (run_directory / "result.json").write_text(
        json.dumps(
            {"manifest": manifest.__dict__, "result": result.as_dict()}, default=str
        ),
        encoding="utf-8",
    )

    store = CampaignStore(tmp_path)
    rebuilt = store.rebuild()

    assert len(rebuilt) == 1
    assert store.load().iloc[0]["f:total_beam_charge_pc"] == 5.0


def test_store_rebuilds_shifted_results_and_retains_native_no_beam_records(
    tmp_path: Path,
) -> None:
    campaign = Campaign.from_file(CONFIG)
    successful_directory = tmp_path / "sim_000001"
    successful_directory.mkdir()
    successful_manifest = campaign.make_manifest("sim_000001", "stage_1")
    successful_manifest.write(successful_directory)
    successful_result = EvaluationResult(
        "sim_000001", "ok", {"total_beam_charge_pc": 5.0}, successful_directory
    )
    successful_record = {
        "manifest": successful_manifest.__dict__,
        "result": successful_result.as_dict(),
    }
    (successful_directory / "result.json").write_text(
        json.dumps(successful_record, default=str), encoding="utf-8"
    )
    shifted_record = json.loads(json.dumps(successful_record, default=str))
    shifted_record["result"]["features"] = {"total_beam_charge_pc": 7.0}
    (successful_directory / "results_shifted.json").write_text(
        json.dumps(shifted_record), encoding="utf-8"
    )

    no_beam_directory = tmp_path / "sim_000002"
    no_beam_directory.mkdir()
    no_beam_manifest = campaign.make_manifest("sim_000002", "stage_1")
    no_beam_manifest.write(no_beam_directory)
    no_beam_result = EvaluationResult(
        "sim_000002", "no_beam", {}, no_beam_directory, "no scalar diagnostics"
    )
    (no_beam_directory / "result.json").write_text(
        json.dumps(
            {"manifest": no_beam_manifest.__dict__, "result": no_beam_result.as_dict()},
            default=str,
        ),
        encoding="utf-8",
    )

    rebuilt = CampaignStore(tmp_path).rebuild(result_filename="results_shifted.json")

    rows = rebuilt.set_index("run_id")
    assert rows.loc["sim_000001", "f:total_beam_charge_pc"] == 7.0
    assert rows.loc["sim_000002", "status"] == "no_beam"


def test_store_requires_shifted_result_for_successful_run(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    run_directory = tmp_path / "sim_000001"
    run_directory.mkdir()
    manifest = campaign.make_manifest("sim_000001", "stage_1")
    manifest.write(run_directory)
    result = EvaluationResult("sim_000001", "ok", {}, run_directory)
    (run_directory / "result.json").write_text(
        json.dumps(
            {"manifest": manifest.__dict__, "result": result.as_dict()}, default=str
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="results_shifted.json"):
        CampaignStore(tmp_path).rebuild(result_filename="results_shifted.json")


def test_store_load_repairs_corrupt_parquet_from_run_results(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    run_directory = tmp_path / "sim_000001"
    run_directory.mkdir()
    manifest = campaign.make_manifest("sim_000001", "stage_1")
    manifest.write(run_directory)
    result = EvaluationResult(
        "sim_000001", "ok", {"total_beam_charge_pc": 5.0}, run_directory
    )
    (run_directory / "result.json").write_text(
        json.dumps(
            {"manifest": manifest.__dict__, "result": result.as_dict()}, default=str
        ),
        encoding="utf-8",
    )
    store = CampaignStore(tmp_path)
    store.path.write_bytes(b"not a parquet file")

    repaired = store.load()

    assert len(repaired) == 1
    assert pd.read_parquet(store.path).iloc[0]["f:total_beam_charge_pc"] == 5.0


def test_store_rebuild_repairs_corrupt_parquet_from_run_results(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    store = CampaignStore(tmp_path)
    first_directory = tmp_path / "sim_000001"
    first_directory.mkdir()
    first_manifest = campaign.make_manifest("sim_000001", "stage_1")
    first_manifest.write(first_directory)
    first_result = EvaluationResult(
        "sim_000001", "ok", {"total_beam_charge_pc": 5.0}, first_directory
    )
    (first_directory / "result.json").write_text(
        json.dumps(
            {"manifest": first_manifest.__dict__, "result": first_result.as_dict()},
            default=str,
        ),
        encoding="utf-8",
    )
    store.rebuild()
    store.path.write_bytes(b"corrupt")

    second_directory = tmp_path / "sim_000002"
    second_directory.mkdir()
    second_manifest = campaign.make_manifest("sim_000002", "stage_1")
    second_manifest.write(second_directory)
    second_result = EvaluationResult(
        "sim_000002", "ok", {"total_beam_charge_pc": 6.0}, second_directory
    )
    (second_directory / "result.json").write_text(
        json.dumps(
            {"manifest": second_manifest.__dict__, "result": second_result.as_dict()},
            default=str,
        ),
        encoding="utf-8",
    )
    store.rebuild()

    repaired = store.load()

    assert set(repaired["run_id"]) == {"sim_000001", "sim_000002"}


def test_store_rebuilds_all_durable_records_after_a_corrupt_cache(
    tmp_path: Path,
) -> None:
    campaign = Campaign.from_file(CONFIG)
    store = CampaignStore(tmp_path)
    for run_id, charge in [("sim_000001", 5.0), ("sim_000002", 6.0)]:
        run_directory = tmp_path / run_id
        run_directory.mkdir()
        manifest = campaign.make_manifest(run_id, "stage_1")
        manifest.write(run_directory)
        result = EvaluationResult(
            run_id, "ok", {"total_beam_charge_pc": charge}, run_directory
        )
        (run_directory / "result.json").write_text(
            json.dumps(
                {"manifest": manifest.__dict__, "result": result.as_dict()}, default=str
            ),
            encoding="utf-8",
        )
    store.path.write_bytes(b"corrupt")

    store.rebuild()

    repaired = store.load()
    assert set(repaired["run_id"]) == {"sim_000001", "sim_000002"}
