from pathlib import Path
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from exploration_fbpic.campaign import Campaign
from exploration_fbpic.exploration import XoptFBPICExplorer


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_initial_design_has_active_parameters_only() -> None:
    explorer = XoptFBPICExplorer(
        Campaign.from_file(CONFIG),
        "stage_1",
    )

    points = explorer.initial_design(20, seed=7)

    assert len(points) == 20
    assert set(points[0]) == set(explorer.variable_bounds)
    assert points[0]["laser_energy_J"] == 2.5


def test_stage_two_training_data_reuses_compatible_stage_one_records(
    monkeypatch,
) -> None:
    campaign = Campaign.from_file(CONFIG)
    explorer = XoptFBPICExplorer(campaign, "stage_2")
    parameters = campaign.config.complete_parameters("stage_1")
    fingerprint = campaign.config.fingerprint()
    compatible = {
        "run_id": "sim_000000",
        "stage": "stage_1",
        "campaign_fingerprint": fingerprint,
        "status": "ok",
        **{f"p:{name}": value for name, value in parameters.items()},
        "f:total_beam_charge_pc": 100.0,
        "f:mean_uz": 200.0,
        "f:cov_x_x": 1.0,
        "f:cov_y_y": 2.0,
        "f:cov_uz_uz": 3.0,
    }
    incompatible = compatible | {
        "run_id": "sim_000001",
        "campaign_fingerprint": "other",
    }
    monkeypatch.setattr(
        explorer.store, "load", lambda: pd.DataFrame([compatible, incompatible])
    )

    training = explorer.training_data()

    assert len(training) == 1
    assert set(training) == {*explorer.variable_bounds, *explorer.feature_names}
    assert training.iloc[0]["peak_neutral_density_cm3"] == 1.4
    assert training.iloc[0]["density_fwhm_right_m"] == 2.0
    assert explorer.completed_evaluation_count() == 0


def test_stage_one_training_data_rejects_off_nominal_later_stage_rows(
    monkeypatch,
) -> None:
    campaign = Campaign.from_file(CONFIG)
    explorer = XoptFBPICExplorer(campaign, "stage_1")
    parameters = campaign.config.complete_parameters("stage_2")
    fingerprint = campaign.config.fingerprint()
    compatible = {
        "run_id": "sim_000000",
        "stage": "stage_1",
        "campaign_fingerprint": fingerprint,
        "status": "ok",
        **{f"p:{name}": value for name, value in parameters.items()},
        "f:total_beam_charge_pc": 100.0,
        "f:mean_uz": 200.0,
        "f:cov_x_x": 1.0,
        "f:cov_y_y": 2.0,
        "f:cov_uz_uz": 3.0,
    }
    later_stage = compatible | {
        "run_id": "sim_000001",
        "stage": "stage_2",
        "p:laser_gdd_relative": 0.2,
    }
    monkeypatch.setattr(
        explorer.store, "load", lambda: pd.DataFrame([compatible, later_stage])
    )

    with pytest.raises(
        ValueError, match="vary inactive parameter 'laser_gdd_relative'"
    ):
        explorer.training_data()


def test_stage_one_training_data_ignores_failed_later_stage_rows(monkeypatch) -> None:
    campaign = Campaign.from_file(CONFIG)
    explorer = XoptFBPICExplorer(campaign, "stage_1")
    parameters = campaign.config.complete_parameters("stage_2")
    failed_later_stage = {
        "run_id": "sim_000001",
        "stage": "stage_2",
        "campaign_fingerprint": campaign.config.fingerprint(),
        "status": "failed",
        **{f"p:{name}": value for name, value in parameters.items()},
        "p:density_fwhm_right_m": 1.3e-3,
        "f:total_beam_charge_pc": 100.0,
    }
    monkeypatch.setattr(
        explorer.store, "load", lambda: pd.DataFrame([failed_later_stage])
    )

    assert explorer.training_data().empty


def test_non_continuation_stage_rejects_any_existing_run_directory(
    tmp_path: Path,
) -> None:
    campaign = Campaign.from_file(CONFIG)
    stages = dict(campaign.config.stages)
    stages["stage_2"] = replace(stages["stage_2"], continuation=False)
    config = replace(campaign.config, run_root=tmp_path, stages=stages)
    explorer = XoptFBPICExplorer(Campaign(config), "stage_2")
    run_directory = tmp_path / "sim_000001"
    run_directory.mkdir()
    Campaign(config).make_manifest("sim_000001", "stage_1").write(run_directory)

    with pytest.raises(FileExistsError, match="continuation disabled"):
        explorer.require_fresh_run_root()


def test_compatible_run_root_rejects_foreign_campaign_manifest(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    config = replace(campaign.config, run_root=tmp_path)
    explorer = XoptFBPICExplorer(Campaign(config), "stage_1")
    run_directory = tmp_path / "sim_000001"
    run_directory.mkdir()
    manifest = Campaign(config).make_manifest("sim_000001", "stage_1")
    manifest_path = manifest.write(run_directory)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["campaign_fingerprint"] = "different-fingerprint"
    manifest_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="runs from a different campaign"):
        explorer.require_compatible_run_root()


def test_next_run_number_skips_orphaned_run_directories(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    config = campaign.config.__class__(
        name=campaign.config.name,
        template_script=campaign.config.template_script,
        template_class=campaign.config.template_class,
        run_root=tmp_path,
        parameters=campaign.config.parameters,
        stages=campaign.config.stages,
        analysis=campaign.config.analysis,
        execution=campaign.config.execution,
    )
    explorer = XoptFBPICExplorer(Campaign(config), "stage_1")
    (tmp_path / "sim_000002").mkdir()
    (tmp_path / "sim_000010").mkdir()

    assert explorer._next_run_number() == 11


def test_log_scale_uses_log_model_coordinates_and_physical_manifests() -> None:
    campaign = Campaign.from_file(CONFIG)
    parameters = dict(campaign.config.parameters)
    parameters["peak_neutral_density_cm3"] = replace(
        parameters["peak_neutral_density_cm3"], log_scale=True
    )
    config = replace(campaign.config, parameters=parameters)
    explorer = XoptFBPICExplorer(Campaign(config), "stage_1")
    physical = explorer.initial_design(1)[0]
    model = explorer.model_initial_design(1)[0]

    assert physical["peak_neutral_density_cm3"] == 1.4e18
    assert model["peak_neutral_density_cm3"] == np.log(1.4)
    assert model["laser_focal_position_m"] == 3.0e-3
    assert explorer.model_to_physical(model)[
        "peak_neutral_density_cm3"
    ] == pytest.approx(1.4e18)


def test_fixed_center_is_excluded_from_exploration_coordinates() -> None:
    explorer = XoptFBPICExplorer(Campaign.from_file(CONFIG), "stage_1")
    physical = explorer.initial_design(1)[0]
    model = explorer.model_initial_design(1)[0]

    assert "density_center_location_m" not in physical
    assert "density_center_location_m" not in model
    assert Campaign.from_file(CONFIG).config.complete_parameters("stage_1")[
        "density_center_location_m"
    ] == pytest.approx(4.0e-3)
