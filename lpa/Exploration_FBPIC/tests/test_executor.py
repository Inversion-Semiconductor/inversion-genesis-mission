from pathlib import Path

from exploration_fbpic.campaign import Campaign
from exploration_fbpic.executor import FBPICRunExecutor


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_prepare_run_writes_isolated_full_manifest(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    config = campaign.config
    copied_config = config.__class__(
        name=config.name,
        template_script=config.template_script,
        run_root=tmp_path,
        parameters=config.parameters,
        stages=config.stages,
        analysis=config.analysis,
        execution=config.execution,
    )
    executor = FBPICRunExecutor(Campaign(copied_config))

    run_directory, manifest = executor.prepare_run("sim_000001", "stage_1")

    assert (run_directory / "run_manifest.json").is_file()
    assert (run_directory / "run_fbpic.py").is_file()
    assert manifest.parameters["laser_spectral_bandwidth_rad_s"] == "auto"
