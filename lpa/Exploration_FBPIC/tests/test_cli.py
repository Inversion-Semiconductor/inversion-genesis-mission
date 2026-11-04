import sys
from pathlib import Path

import numpy as np
import pytest

from exploration_fbpic import cli
from exploration_fbpic.campaign import Campaign
from exploration_fbpic.density_envelope import build_density_envelope_cases
from exploration_fbpic.density_ramps import (
    build_density_ramp_cases,
    build_density_ramp_parameter_space,
)
from exploration_fbpic.exploration import XoptFBPICExplorer


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_density_ramp_cases_use_all_extremes_and_requested_half_length() -> None:
    campaign = Campaign.from_file(CONFIG)
    fwhm = campaign.config.parameters["density_fwhm_left_m"]
    ramp_percent = campaign.config.parameters["density_left_ramp_edge_percent"]

    cases = build_density_ramp_cases(campaign, simulation_half_length=4.0e-3)

    assert {(case.fwhm_left_m, case.requested_ramp_edge_percent) for case in cases} == {
        (fwhm.lower, ramp_percent.lower),
        (fwhm.lower, ramp_percent.upper),
        (fwhm.upper, ramp_percent.lower),
        (fwhm.upper, ramp_percent.upper),
    }
    assert {case.density.center for case in cases} == {4.0e-3}
    for case in cases:
        assert case.density.build_density_function()(
            case.one_percent_location_m, 0.0
        ) == pytest.approx(0.01)


def test_density_ramp_parameter_space_samples_each_edge() -> None:
    campaign = Campaign.from_file(CONFIG)
    fwhm = campaign.config.parameters["density_fwhm_left_m"]
    ramp_percent = campaign.config.parameters["density_left_ramp_edge_percent"]

    points = build_density_ramp_parameter_space(campaign, simulation_half_length=5.0e-3)

    assert len(points) == 16
    assert {
        (point.fwhm_left_m, point.requested_ramp_edge_percent) for point in points
    } >= {
        (fwhm.lower, ramp_percent.lower),
        (fwhm.lower, ramp_percent.upper),
        (fwhm.upper, ramp_percent.lower),
        (fwhm.upper, ramp_percent.upper),
    }
    low_percent_beta = next(
        point.beta_left
        for point in points
        if point.fwhm_left_m == fwhm.lower
        and point.requested_ramp_edge_percent == ramp_percent.lower
    )
    high_percent_beta = next(
        point.beta_left
        for point in points
        if point.fwhm_left_m == fwhm.lower
        and point.requested_ramp_edge_percent == ramp_percent.upper
    )
    assert low_percent_beta > high_percent_beta
    assert next(
        point.effective_ramp_edge_percent
        for point in points
        if point.fwhm_left_m == fwhm.lower
        and point.requested_ramp_edge_percent == ramp_percent.lower
    ) == pytest.approx(float(ramp_percent.lower))
    assert next(
        point.effective_ramp_edge_percent
        for point in points
        if point.fwhm_left_m == fwhm.upper
        and point.requested_ramp_edge_percent == ramp_percent.lower
    ) == pytest.approx(
        float(
            campaign.config.execution["hyperparameters"]["ramp_percent_min_high_fwhm"]
        )
    )
    low_fwhm_percentages = np.array(
        [
            point.requested_ramp_edge_percent
            for point in points
            if point.fwhm_left_m == fwhm.lower
        ]
    )
    assert np.allclose(
        np.sort(low_fwhm_percentages),
        np.geomspace(ramp_percent.lower, ramp_percent.upper, 5),
    )


def test_density_envelope_samples_stage_parameters_within_bounds() -> None:
    campaign = Campaign.from_file(CONFIG)

    cases = build_density_envelope_cases(campaign, "stage_2", random_samples=3, seed=4)

    assert len(cases) == 4
    for parameter in campaign.config.active_parameters("stage_2"):
        assert parameter.lower is not None and parameter.upper is not None
        values = [case.parameters[parameter.name] for case in cases[1:]]
        assert all(parameter.lower <= value <= parameter.upper for value in values)
    for case in cases:
        density = case.density
        left_one_percent_distance = (
            density.fwhm_left
            / 2.0
            * (np.log(100.0) / np.log(2.0)) ** (1.0 / density.beta_left)
        )
        right_one_percent_distance = (
            density.fwhm_right
            / 2.0
            * (np.log(100.0) / np.log(2.0)) ** (1.0 / density.beta_right)
        )
        profile = density.build_density_function()
        assert profile(
            density.center - left_one_percent_distance, 0.0
        ) == pytest.approx(0.01)
        assert profile(
            density.center + right_one_percent_distance, 0.0
        ) == pytest.approx(0.01)
        assert density.center == pytest.approx(4.0e-3)
        assert density.p_zmin == 0.0
        assert density.p_zmax == 9.0e-3


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
