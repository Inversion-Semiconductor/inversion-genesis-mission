from pathlib import Path
import importlib.util

import pytest

from exploration_fbpic.campaign import Campaign


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"
PRODUCTION_CONFIG = Path(__file__).parents[1] / "runs" / "production_study" / "campaign.yaml"
FBPIC_CONFIGS = (
    CONFIG,
    PRODUCTION_CONFIG,
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
    assert (
        settings.total_gpus
        == settings.gpus_per_simulation * settings.concurrent_simulations
    )


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


def _load_template(campaign: Campaign) -> type:
    specification = importlib.util.spec_from_file_location(
        "campaign_template", campaign.config.template_script
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return getattr(module, campaign.config.template_class)


def _write_profile_campaign(tmp_path: Path, name: str, hyperparameters: str) -> Path:
    profiles = Path(__file__).parents[1] / "runs" / "resolution_profiles.yaml"
    config = tmp_path / "campaign.yaml"
    config.write_text(
        f"""\
name: test
template_script: template.py
run_root: runs
parameters:
    x: {{nominal: 0.5, lower: 0.0, upper: 1.0}}
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
execution:
    resolution_profile: {{file: {profiles}, name: {name}}}
    hyperparameters: {hyperparameters}
""",
        encoding="utf-8",
    )
    return config


def test_production_campaign_uses_minimum_resolution_profile() -> None:
    hyperparameters = Campaign.from_file(PRODUCTION_CONFIG).config.execution[
        "hyperparameters"
    ]

    assert hyperparameters["gamma_boost"] == 3.0
    assert hyperparameters["nz"] == 2592
    assert hyperparameters["nr"] == 1080
    assert hyperparameters["zmin"] == pytest.approx(-100e-6)
    assert hyperparameters["rmax"] == pytest.approx(400e-6)
    assert hyperparameters["p_rmax"] == pytest.approx(90e-6)
    assert hyperparameters["helium_ppc"] == [1, 2, 10]
    assert hyperparameters["nitrogen_ppc"] == [2, 2, 15]
    assert hyperparameters["n_order"] == 32


@pytest.mark.parametrize(
    ("name", "cells_per_wavelength", "dr"),
    [("minimum", 20.74, 0.370e-6), ("ideal", 30.72, 0.320e-6)],
)
def test_resolution_profiles_match_the_specification(
    tmp_path: Path, name: str, cells_per_wavelength: float, dr: float
) -> None:
    config = _write_profile_campaign(tmp_path, name, "{}")
    hyperparameters = Campaign.from_file(config).config.execution["hyperparameters"]

    length = hyperparameters["zmax"] - hyperparameters["zmin"]
    assert 800e-9 * hyperparameters["nz"] / length == pytest.approx(
        cells_per_wavelength, abs=5e-3
    )
    assert hyperparameters["rmax"] / hyperparameters["nr"] == pytest.approx(
        dr, abs=5e-10
    )


def test_resolution_profile_rejects_duplicate_hyperparameters(tmp_path: Path) -> None:
    config = _write_profile_campaign(tmp_path, "minimum", "{nz: 100}")

    with pytest.raises(ValueError, match="nz"):
        Campaign.from_file(config)


def test_resolution_profile_rejects_unknown_name(tmp_path: Path) -> None:
    config = _write_profile_campaign(tmp_path, "huge", "{}")

    with pytest.raises(KeyError, match="huge"):
        Campaign.from_file(config)


def test_template_species_ppc_falls_back_to_shared_values() -> None:
    campaign = Campaign.from_file(PRODUCTION_CONFIG)
    template = _load_template(campaign)
    parameters = campaign.config.complete_parameters("stage_1")

    production = template(parameters, campaign.config.execution["hyperparameters"])
    shared = template(parameters, {"p_nz": 3, "p_nr": 4, "p_nt": 5})

    assert production._species_ppc("helium_ppc") == (1, 2, 10)
    assert production._species_ppc("nitrogen_ppc") == (2, 2, 15)
    assert shared._species_ppc("helium_ppc") == (3, 4, 5)
    assert shared._species_ppc("nitrogen_ppc") == (3, 4, 5)
