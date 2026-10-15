"""Run the nominal initial-study FBPIC case with inspection diagnostics."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from exploration_fbpic.campaign import Campaign


CAMPAIGN_FILE = Path(__file__).with_name("campaign.yaml")
PSCRATCH = os.environ.get("PSCRATCH")
if PSCRATCH is None:
    raise RuntimeError("PSCRATCH must be set for the nominal NERSC run")
RUN_LABEL = os.environ.get("NOMINAL_RUN_LABEL", "nominal_test")
if not RUN_LABEL.replace("_", "").replace("-", "").isalnum():
    raise ValueError("NOMINAL_RUN_LABEL must contain only letters, digits, '_' or '-'")
RUN_DIRECTORY = Path(PSCRATCH) / "ionization_injection_exploration" / RUN_LABEL


def load_template_class(campaign: Campaign) -> type:
    """Load the campaign's configured simulation template class."""
    specification = importlib.util.spec_from_file_location(
        "nominal_initial_study_template", campaign.config.template_script
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"could not import {campaign.config.template_script}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return getattr(module, campaign.config.template_class)


def main() -> None:
    campaign = Campaign.from_file(CAMPAIGN_FILE)
    parameters = campaign.config.complete_parameters("stage_1")
    hyperparameters = campaign.config.template_hyperparameters()
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    os.chdir(RUN_DIRECTORY)
    hyperparameters.update(
        {
            "dump_fields": True,
            "number_dumps_lab": 10,
            "lab_diagnostic_directory": RUN_DIRECTORY / "lab_diags",
            "results_directory": RUN_DIRECTORY / "results",
            "lasy_file": RUN_DIRECTORY / "experimental_laser",
        }
    )
    simulation_class = load_template_class(campaign)
    simulation_class(parameters, hyperparameters).run()


if __name__ == "__main__":
    main()
