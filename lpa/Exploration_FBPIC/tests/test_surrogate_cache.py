from pathlib import Path

import numpy as np
import pandas as pd
import torch
from exploration_fbpic.campaign import Campaign
from exploration_fbpic.exploration import XoptFBPICExplorer
from exploration_fbpic.surrogate_cache import (
    INDUCING_POINTS,
    MODEL_STATE_FILENAME,
    load_surrogate,
    train_surrogates,
)

CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_charge_surrogate_cache_round_trip(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    explorer = XoptFBPICExplorer(campaign, "stage_2")
    rows = []
    for index in range(8):
        parameters = campaign.config.complete_parameters("stage_2")
        fraction = index / 7
        for parameter in explorer.parameters:
            assert parameter.lower is not None and parameter.upper is not None
            parameters[parameter.name] = parameter.lower + fraction * (
                parameter.upper - parameter.lower
            )
        rows.append(
            {
                "campaign_fingerprint": "archived-stage-2",
                "status": "ok",
                **{f"p:{name}": value for name, value in parameters.items()},
                "f:total_beam_charge_pc": 50.0 + 100.0 * fraction,
                "f:mean_uz": 150.0 + 50.0 * fraction,
            }
        )
    data_path = tmp_path / "campaign_runs.parquet"
    pd.DataFrame(rows).to_parquet(data_path, index=False)

    result = train_surrogates(
        explorer, data_path, tmp_path / "cache", "total_beam_charge_pc"
    )
    caches = result.trained
    cached = load_surrogate(explorer, tmp_path / "cache", "total_beam_charge_pc")
    inputs = torch.tensor(
        cached.training_data.loc[:, list(explorer.variable_bounds)].to_numpy(),
        dtype=torch.float64,
    )

    posterior = cached.model.posterior(inputs)

    assert caches == (tmp_path / "cache" / "var_total_beam_charge_pc",)
    assert (caches[0] / "metadata.json").is_file()
    assert (caches[0] / "training.parquet").is_file()
    assert (caches[0] / MODEL_STATE_FILENAME).is_file()
    assert cached.metadata["source_campaign_fingerprint"] == "archived-stage-2"
    assert cached.metadata["training_rows"] == 8
    assert cached.metadata["model_family"] == "approximate_variational_gp"
    assert cached.metadata["inducing_points"] == INDUCING_POINTS
    assert posterior.mean.shape == (8, 1)
    assert np.isfinite(posterior.mean.detach().numpy()).all()


def test_all_variables_create_separate_cache_directories(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    explorer = XoptFBPICExplorer(campaign, "stage_2")
    rows = []
    for index in range(4):
        parameters = campaign.config.complete_parameters("stage_2")
        fraction = index / 3
        for parameter in explorer.parameters:
            assert parameter.lower is not None and parameter.upper is not None
            parameters[parameter.name] = parameter.lower + fraction * (
                parameter.upper - parameter.lower
            )
        rows.append(
            {
                "campaign_fingerprint": "archived-stage-2",
                "status": "ok",
                **{f"p:{name}": value for name, value in parameters.items()},
                "f:mean_ux": fraction,
                "f:mean_uy": 1.0 - fraction,
            }
        )
    data_path = tmp_path / "campaign_runs.parquet"
    pd.DataFrame(rows).to_parquet(data_path, index=False)

    result = train_surrogates(explorer, data_path, tmp_path / "cache", "all")
    caches = result.trained

    assert caches == (
        tmp_path / "cache" / "var_mean_ux",
        tmp_path / "cache" / "var_mean_uy",
    )
    assert all((cache / MODEL_STATE_FILENAME).is_file() for cache in caches)

    skipped = train_surrogates(explorer, data_path, tmp_path / "cache", "all")

    assert skipped.trained == ()
    assert skipped.skipped == caches

    overwritten = train_surrogates(
        explorer, data_path, tmp_path / "cache", "all", overwrite=True
    )

    assert overwritten.trained == caches
    assert overwritten.skipped == ()


def test_feasibility_surrogate_uses_beam_and_no_beam_outcomes(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    explorer = XoptFBPICExplorer(campaign, "stage_2")
    rows = []
    for index in range(4):
        parameters = campaign.config.complete_parameters("stage_2")
        fraction = index / 3
        for parameter in explorer.parameters:
            assert parameter.lower is not None and parameter.upper is not None
            parameters[parameter.name] = parameter.lower + fraction * (
                parameter.upper - parameter.lower
            )
        rows.append(
            {
                "campaign_fingerprint": "archived-stage-2",
                "status": "ok" if index % 2 else "no_beam",
                "message": "",
                **{f"p:{name}": value for name, value in parameters.items()},
            }
        )
    data_path = tmp_path / "campaign_runs.parquet"
    pd.DataFrame(rows).to_parquet(data_path, index=False)

    result = train_surrogates(explorer, data_path, tmp_path / "cache", "feasibility")
    caches = result.trained
    cached = load_surrogate(explorer, tmp_path / "cache", "feasibility")

    assert caches == (tmp_path / "cache" / "var_feasibility",)
    assert set(cached.training_data["feasibility"]) == {-1.0, 1.0}
    assert cached.metadata["outcome_kind"] == "probit_feasibility"
    assert cached.metadata["beam_rows"] == 2
    assert cached.metadata["no_beam_rows"] == 2
    assert cached.metadata["sampling"] == "deterministic_class_balanced"
