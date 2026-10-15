from pathlib import Path
import importlib.util

import pytest

from exploration_fbpic.campaign import Campaign


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"
FBPIC_CONFIGS = (
    CONFIG,
    Path(__file__).parent / "smoke_campaigns" / "nersc_smoke" / "campaign.yaml",
    Path(__file__).parent
    / "smoke_campaigns"
    / "nersc_surrogate_smoke"
    / "campaign.yaml",
)


@pytest.mark.parametrize("config_path", FBPIC_CONFIGS)
def test_campaign_parameters_match_the_simulation_template(config_path: Path) -> None:
    campaign = Campaign.from_file(config_path)
    specification = importlib.util.spec_from_file_location(
        "campaign_template", campaign.config.template_script
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    template = getattr(module, campaign.config.template_class)
    accepted = (
        template._PHYSICAL_PARAMETER_KEYS | template._OPTIONAL_PHYSICAL_PARAMETER_KEYS
    )

    assert set(campaign.config.parameters) <= accepted
    assert template._PHYSICAL_PARAMETER_KEYS <= set(campaign.config.parameters)
    template(
        campaign.config.complete_parameters(next(iter(campaign.config.stages))),
        campaign.config.execution["hyperparameters"],
    )


def test_stage_one_keeps_all_physical_parameters() -> None:
    campaign = Campaign.from_file(CONFIG)

    manifest = campaign.make_manifest(
        "sim_000001",
        "stage_1",
        {"laser_energy_J": 2.6},
    )

    assert len(manifest.parameters) == len(campaign.config.parameters)
    assert manifest.parameters["laser_energy_J"] == 2.6
    assert manifest.parameters["zernike_coma_5_x"] == 0.0
    assert manifest.parameters["laser_spectral_bandwidth_rad_s"] == "auto"
    assert manifest.stage_settings.gpus_per_simulation == 8
    assert manifest.stage_settings.concurrent_simulations == 4


def test_campaign_derives_independent_right_ramp_inputs() -> None:
    campaign = Campaign.from_file(CONFIG)
    hyperparameters = campaign.config.template_hyperparameters()

    assert hyperparameters["right_ramp_percent_fwhm_range_m"] == [0.8e-3, 5.0e-3]
    assert hyperparameters["right_ramp_percent_min_low_fwhm"] == 0.05


def test_stage_resource_settings_are_consistent() -> None:
    campaign = Campaign.from_file(CONFIG)
    settings = campaign.config.stage_settings("stage_1")

    assert settings.max_evaluations == 200
    assert settings.initial_design_size == 24
    assert (
        settings.total_gpus
        == settings.gpus_per_simulation * settings.concurrent_simulations
    )


def test_production_density_geometry_is_fixed_to_the_simulation_domain() -> None:
    campaign = Campaign.from_file(CONFIG)
    parameters = campaign.config.parameters
    hyperparameters = campaign.config.execution["hyperparameters"]

    assert parameters["density_center_location_m"].nominal == 5.0e-3
    assert not parameters["density_center_location_m"].variable
    assert (
        "density_center_location_m"
        not in campaign.config.stage_settings("stage_1").active_parameters
    )
    assert (
        "density_center_location_m"
        not in campaign.config.stage_settings("stage_2").active_parameters
    )
    assert hyperparameters["zmin"] == -150.0e-6
    assert hyperparameters["zmax"] == 0.0
    assert hyperparameters["nz"] == 3840
    assert (hyperparameters["zmax"] - hyperparameters["zmin"]) / hyperparameters[
        "nz"
    ] <= 800.0e-9 / 20.0
    assert hyperparameters["p_zmin"] == 0.0
    assert hyperparameters["p_zmax"] == 10.0e-3


def test_stage_one_rejects_inactive_parameters() -> None:
    campaign = Campaign.from_file(CONFIG)

    try:
        campaign.make_manifest("sim_000001", "stage_1", {"laser_cep_phase_rad": 0.2})
    except ValueError as error:
        assert "inactive" in str(error)
    else:
        raise AssertionError("expected an inactive parameter to be rejected")


def test_production_nitrogen_dopant_fraction_is_fixed() -> None:
    campaign = Campaign.from_file(CONFIG)

    parameter = campaign.config.parameters["nitrogen_dopant_fraction"]
    assert parameter.nominal == 0.005
    assert not parameter.variable
    with pytest.raises(ValueError, match="inactive or unknown"):
        campaign.make_manifest(
            "sim_000001", "stage_2", {"nitrogen_dopant_fraction": 0.01}
        )


def test_production_spherical_aberration_is_fixed() -> None:
    campaign = Campaign.from_file(CONFIG)

    parameter = campaign.config.parameters["zernike_spherical_3"]
    assert parameter.nominal == 0.0
    assert not parameter.variable
    assert (
        "zernike_spherical_3"
        not in campaign.config.stage_settings("stage_2").active_parameters
    )


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


def test_campaign_rejects_nominal_outside_bounds(tmp_path: Path) -> None:
    config = tmp_path / "campaign.yaml"
    config.write_text(
        """\
name: invalid
template_script: template.py
run_root: runs
parameters:
    x: {nominal: 2.0, lower: 0.0, upper: 1.0}
stages:
    stage_1:
        active_parameters: [x]
        max_evaluations: 1
        initial_design_size: 1
        gpus_per_simulation: 1
        total_gpus: 1
        concurrent_simulations: 1
analysis:
    exploration_features: [f]
execution: {hyperparameters: {}}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside"):
        Campaign.from_file(config)


def test_campaign_expands_absolute_environment_run_root(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "campaign.yaml"
    config.write_text(
        """\
name: test
template_script: template.py
run_root: ${PSCRATCH}/campaign
parameters:
    x: {nominal: 0.5, lower: 0.0, upper: 1.0}
stages:
    stage_1:
        active_parameters: [x]
        max_evaluations: 1
        initial_design_size: 1
        gpus_per_simulation: 1
        total_gpus: 1
        concurrent_simulations: 1
analysis:
    exploration_features: [f]
execution: {hyperparameters: {}}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PSCRATCH", str(tmp_path / "scratch"))

    campaign = Campaign.from_file(config)

    assert campaign.config.run_root == (tmp_path / "scratch" / "campaign").resolve()
