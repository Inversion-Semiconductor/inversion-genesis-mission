from pathlib import Path

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
