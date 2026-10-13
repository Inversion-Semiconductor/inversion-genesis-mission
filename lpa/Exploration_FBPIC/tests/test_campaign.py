from pathlib import Path

from exploration_fbpic.campaign import Campaign


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_stage_one_keeps_all_physical_parameters() -> None:
    campaign = Campaign.from_file(CONFIG)

    manifest = campaign.make_manifest(
        "sim_000001",
        "stage_1",
        {"laser_energy_J": 3.0},
    )

    assert len(manifest.parameters) == len(campaign.config.parameters)
    assert manifest.parameters["laser_energy_J"] == 3.0
    assert manifest.parameters["zernike_coma_5_x"] == 0.0
    assert manifest.parameters["laser_spectral_bandwidth_rad_s"] == "auto"
    assert manifest.stage_settings.gpus_per_simulation == 8
    assert manifest.stage_settings.concurrent_simulations == 4


def test_stage_resource_settings_are_consistent() -> None:
    campaign = Campaign.from_file(CONFIG)
    settings = campaign.config.stage_settings("stage_1")

    assert settings.max_evaluations == 200
    assert settings.initial_design_size == 24
    assert settings.total_gpus == settings.gpus_per_simulation * settings.concurrent_simulations


def test_stage_one_rejects_inactive_parameters() -> None:
    campaign = Campaign.from_file(CONFIG)

    try:
        campaign.make_manifest("sim_000001", "stage_1", {"laser_cep_phase_rad": 0.2})
    except ValueError as error:
        assert "inactive" in str(error)
    else:
        raise AssertionError("expected an inactive parameter to be rejected")


def test_fingerprint_is_independent_of_template_checkout_path(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    template_copy = tmp_path / "copied_template.py"
    template_copy.write_bytes(campaign.config.template_script.read_bytes())
    relocated = campaign.config.__class__(
        name=campaign.config.name,
        template_script=template_copy,
        template_class=campaign.config.template_class,
        run_root=campaign.config.run_root,
        parameters=campaign.config.parameters,
        stages=campaign.config.stages,
        analysis=campaign.config.analysis,
        execution=campaign.config.execution,
    )

    assert relocated.fingerprint() == campaign.config.fingerprint()
