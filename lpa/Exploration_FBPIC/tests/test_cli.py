import sys
from pathlib import Path

import pytest

from exploration_fbpic import cli
from exploration_fbpic.exploration import XoptFBPICExplorer


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_evaluate_validates_run_root_compatibility(monkeypatch) -> None:
    def fail_compatibility_check(_: XoptFBPICExplorer) -> None:
        raise ValueError("runs from a different campaign")

    monkeypatch.setattr(XoptFBPICExplorer, "require_compatible_run_root", fail_compatibility_check)
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