"""Run the observed OOM envelope as a reproducible four-GPU stress case."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

from exploration_fbpic.campaign import Campaign


CAMPAIGN_FILE = Path(__file__).parents[2] / "runs" / "initial_study" / "campaign.yaml"
PSCRATCH = os.environ.get("PSCRATCH")
if PSCRATCH is None:
    raise RuntimeError("PSCRATCH must be set for the NERSC memory-stress run")
RUN_LABEL = os.environ.get("MEMORY_STRESS_RUN_LABEL", "memory_stress_4gpu")
if not RUN_LABEL.replace("_", "").replace("-", "").isalnum():
    raise ValueError(
        "MEMORY_STRESS_RUN_LABEL must contain only letters, digits, '_' or '-'"
    )
RUN_DIRECTORY = (
    Path(PSCRATCH) / "ionization_injection_exploration_stage1_4gpu" / RUN_LABEL
)

# The extrema below occurred across distinct Stage 0 OOM cases. Combining them
# produces an intentionally harsher memory stress case than any single run.
OBSERVED_OOM_ENVELOPE = {
    "peak_neutral_density_cm3": 2.718370894528925e18,  # sim_000023
    "density_fwhm_left_m": 4.001246253587306e-3,  # sim_000021
    "laser_energy_J": 2.7,  # sim_000006
    "laser_pulse_duration_fwhm_s": 45.0e-15,  # sim_000008
    "laser_spot_size_m": 32.0e-6,  # sim_000010
    "laser_super_gaussian_order": 4.0,  # sim_000012
    "laser_focal_position_m": 5.0e-3,  # sim_000014
}


def load_template_class(campaign: Campaign) -> type:
    """Load the simulation class configured by the campaign."""
    specification = importlib.util.spec_from_file_location(
        "memory_stress_initial_study_template", campaign.config.template_script
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"could not import {campaign.config.template_script}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return getattr(module, campaign.config.template_class)


def main() -> None:
    campaign = Campaign.from_file(CAMPAIGN_FILE)
    parameters = campaign.config.complete_parameters("stage_2", OBSERVED_OOM_ENVELOPE)
    hyperparameters = campaign.config.template_hyperparameters()
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (RUN_DIRECTORY / "memory_stress_parameters.json").write_text(
        json.dumps(
            {
                "observed_oom_envelope": OBSERVED_OOM_ENVELOPE,
                "parameters": parameters,
                "hyperparameters": hyperparameters,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chdir(RUN_DIRECTORY)
    hyperparameters.update(
        {
            "lab_diagnostic_directory": RUN_DIRECTORY / "lab_diags",
            "results_directory": RUN_DIRECTORY / "results",
            "lasy_file": RUN_DIRECTORY / "experimental_laser",
        }
    )
    simulation_class = load_template_class(campaign)
    simulation_class(parameters, hyperparameters).run()


if __name__ == "__main__":
    main()
