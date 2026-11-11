from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd

from exploration_fbpic.campaign import Campaign
from exploration_fbpic.campaign_rows import model_input_matrix, physical_outcome_rows
from exploration_fbpic.exploration import XoptFBPICExplorer


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_physical_outcomes_include_beam_and_legacy_no_beam_only() -> None:
    campaign = Campaign.from_file(CONFIG)
    explorer = XoptFBPICExplorer(campaign, "stage_1")
    parameters = campaign.config.complete_parameters("stage_1")
    base = {
        "campaign_fingerprint": campaign.config.fingerprint(),
        **{f"p:{name}": value for name, value in parameters.items()},
    }
    data = pd.DataFrame(
        [
            base | {"run_id": "sim_000000", "status": "ok", "message": ""},
            base
            | {
                "run_id": "sim_000001",
                "status": "analysis_error",
                "message": "No scalar diagnostic results found in datapoint.json",
            },
            base
            | {
                "run_id": "sim_000002",
                "status": "error",
                "message": "unrelated failure",
            },
            base
            | {
                "run_id": "sim_000003",
                "campaign_fingerprint": "foreign",
                "status": "ok",
                "message": "",
            },
        ]
    )

    rows, beam, no_beam = physical_outcome_rows(explorer, data)

    assert list(rows["run_id"]) == ["sim_000000", "sim_000001"]
    assert np.array_equal(beam, [True, False])
    assert np.array_equal(no_beam, [False, True])


def test_model_input_matrix_uses_log_scaled_active_coordinates() -> None:
    campaign = Campaign.from_file(CONFIG)
    parameters = dict(campaign.config.parameters)
    parameters["peak_neutral_density_cm3"] = replace(
        parameters["peak_neutral_density_cm3"], log_scale=True
    )
    explorer = XoptFBPICExplorer(
        Campaign(replace(campaign.config, parameters=parameters)),
        "stage_1",
    )
    physical = campaign.config.complete_parameters("stage_1")
    rows = pd.DataFrame([{f"p:{name}": value for name, value in physical.items()}])

    matrix = model_input_matrix(explorer, rows)

    density_index = list(explorer.variable_bounds).index("peak_neutral_density_cm3")
    assert matrix.shape == (1, len(explorer.variable_bounds))
    assert matrix[0, density_index] == np.log(1.4)
