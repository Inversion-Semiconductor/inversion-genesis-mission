import sys
from pathlib import Path

import pytest

from exploration_fbpic import cli
from exploration_fbpic.campaign import Campaign
from exploration_fbpic.density_ramps import build_density_ramp_cases
from exploration_fbpic.exploration import XoptFBPICExplorer


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_density_ramp_cases_use_all_extremes_and_requested_half_length() -> None:
    campaign = Campaign.from_file(CONFIG)

    cases = build_density_ramp_cases(campaign, simulation_half_length=4.0e-3)

    assert {(case.fwhm_left_m, case.beta_left) for case in cases} == {
        (0.8e-3, 1.0),
        (0.8e-3, 3.0),
        (3.0e-3, 1.0),
        (3.0e-3, 3.0),
    }
    assert {case.density.center for case in cases} == {4.0e-3}


def test_evaluate_validates_run_root_compatibility(monkeypatch) -> None:
    def fail_compatibility_check(_: XoptFBPICExplorer) -> None:
        raise ValueError("runs from a different campaign")

    monkeypatch.setattr(
        XoptFBPICExplorer, "require_compatible_run_root", fail_compatibility_check
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "explore-fbpic",
            str(CONFIG),
            "--stage",
            "stage_1",
            "evaluate",
            "--point",
            '{"laser_energy_J": 2.5}',
        ],
    )

    with pytest.raises(ValueError, match="runs from a different campaign"):
        cli.main()
