from pathlib import Path

import pandas as pd

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


def test_stage_two_training_data_reuses_compatible_stage_one_records(monkeypatch) -> None:
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
    incompatible = compatible | {"run_id": "sim_000001", "campaign_fingerprint": "other"}
    monkeypatch.setattr(explorer.store, "load", lambda: pd.DataFrame([compatible, incompatible]))

    training = explorer.training_data()

    assert len(training) == 1
    assert set(training) == {*explorer.variable_bounds, *explorer.feature_names}
    assert training.iloc[0]["density_peak"] == 1.1
    assert explorer.completed_evaluation_count() == 0


def test_next_run_number_skips_orphaned_run_directories(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    config = campaign.config.__class__(
        name=campaign.config.name,
        template_script=campaign.config.template_script,
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
