from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from exploration_fbpic.campaign import Campaign
from exploration_fbpic import executor
from exploration_fbpic.executor import FBPICRunExecutor, mean_kinetic_energy_mev
from inversion_fbpic.lib.config_container import ConfigContainer
from inversion_fbpic.lib.datapoint import Parameters
from inversion_fbpic.lib.diagnostics import (
    _Diagnostic,
    _ParticleDiagnostic,
    MeanKineticEnergy,
    MomentDescriptorDiagnostic,
)


CONFIG = Path(__file__).parents[1] / "runs" / "initial_study" / "campaign.yaml"


def test_prepare_run_writes_isolated_full_manifest(tmp_path: Path) -> None:
    campaign = Campaign.from_file(CONFIG)
    config = campaign.config
    copied_config = config.__class__(
        name=config.name,
        template_script=config.template_script,
        template_class=config.template_class,
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
    assert (
        "from ionization_injection_template import IonizationInjectionSimulation"
        in (run_directory / "run_fbpic.py").read_text(encoding="utf-8")
    )
    assert manifest.parameters["laser_spectral_bandwidth_rad_s"] == "auto"


def test_prepare_run_imports_the_configured_template_module_and_class(
    tmp_path: Path,
) -> None:
    campaign = Campaign.from_file(CONFIG)
    alternate_template = tmp_path / "alternate_template.py"
    alternate_template.write_text(
        "class AlternateSimulation:\n    pass\n", encoding="utf-8"
    )
    config = campaign.config.__class__(
        name=campaign.config.name,
        template_script=alternate_template,
        template_class="AlternateSimulation",
        run_root=tmp_path / "runs",
        parameters=campaign.config.parameters,
        stages=campaign.config.stages,
        analysis=campaign.config.analysis,
        execution=campaign.config.execution,
    )

    run_directory, _ = FBPICRunExecutor(Campaign(config)).prepare_run(
        "sim_000001", "stage_1"
    )
    entrypoint = (run_directory / "run_fbpic.py").read_text(encoding="utf-8")

    assert "from alternate_template import AlternateSimulation" in entrypoint
    assert (
        "AlternateSimulation(PHYSICAL_PARAMETERS, HYPERPARAMETERS).run()" in entrypoint
    )


def test_mean_kinetic_energy_uses_all_normalized_momentum_components() -> None:
    particles = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(3.0)]])
    energy = mean_kinetic_energy_mev(particles, np.array([1.0]))

    assert energy == pytest.approx(0.51099895)


@pytest.fixture()
def result_executor(tmp_path: Path) -> FBPICRunExecutor:
    campaign = Campaign.from_file(CONFIG)
    config = replace(
        campaign.config,
        run_root=tmp_path,
        analysis={
            "exploration_features": ["total_beam_charge_pc", "mean_kinetic_energy_MeV"]
        },
    )
    return FBPICRunExecutor(Campaign(config))


@pytest.fixture()
def saved_datapoint() -> ConfigContainer:
    return ConfigContainer(
        configs=[
            Parameters(label="inputs", data={"mean_uz": -999.0}),
            MomentDescriptorDiagnostic(
                target=("all_of_species", "e"),
                data={"total_beam_charge_pc": 2, "mean_uz": 3.0},
            ),
            MeanKineticEnergy(
                target=("all_of_species", "e"),
                data={"mean_kinetic_energy_MeV": 1.25},
            ),
        ]
    )


@pytest.mark.parametrize("format", ["json", "yaml", "h5"])
@pytest.mark.parametrize("directory", ["results", "custom_results", "absolute"])
def test_extract_features_reads_saved_diagnostics_without_reanalysis(
    tmp_path: Path, result_executor, saved_datapoint, monkeypatch, format, directory
) -> None:
    results_directory = tmp_path / directory
    if directory != "results":
        configured = str(results_directory) if directory == "absolute" else directory
        config = result_executor.config
        result_executor.config = replace(
            config,
            execution={
                **config.execution,
                "hyperparameters": {
                    **config.execution["hyperparameters"],
                    "results_directory": configured,
                },
            },
        )
    saved_datapoint.to_file(
        results_directory / f"datapoint.{format}", format, group_path="/results"
    )
    reanalyze = Mock(
        side_effect=AssertionError("Saved results must not be recalculated")
    )
    monkeypatch.setattr(_Diagnostic, "analyze", reanalyze)
    monkeypatch.setattr(_ParticleDiagnostic, "load_particles", reanalyze)
    monkeypatch.setattr(executor, "mean_kinetic_energy_mev", reanalyze)

    features = result_executor.extract_features(tmp_path)

    assert features == {
        "total_beam_charge_pc": 2.0,
        "mean_uz": 3.0,
        "mean_kinetic_energy_MeV": 1.25,
    }
    assert all(isinstance(value, float) for value in features.values())
    assert not (tmp_path / "lab_diags").exists()
    reanalyze.assert_not_called()


def test_extract_features_supports_nested_containers(
    tmp_path, result_executor, saved_datapoint
) -> None:
    ConfigContainer(configs=[ConfigContainer(configs=[saved_datapoint])]).to_json_file(
        tmp_path / "results" / "datapoint.json"
    )
    assert result_executor.extract_features(tmp_path)["mean_kinetic_energy_MeV"] == 1.25


def test_extract_features_does_not_fall_back_to_raw_particles(
    tmp_path, result_executor
) -> None:
    raw_directory = tmp_path / "lab_diags" / "hdf5"
    raw_directory.mkdir(parents=True)
    (raw_directory / "data00000000.h5").touch()
    with pytest.raises(FileNotFoundError, match="No saved datapoint config"):
        result_executor.extract_features(tmp_path)


def test_extract_features_rejects_multiple_datapoint_files(
    tmp_path, result_executor, saved_datapoint
) -> None:
    for format in ("json", "yaml"):
        saved_datapoint.to_file(tmp_path / "results" / f"datapoint.{format}", format)
    with pytest.raises(ValueError, match="Multiple saved datapoint configs"):
        result_executor.extract_features(tmp_path)


def test_extract_features_requires_a_config_container(
    tmp_path, result_executor
) -> None:
    Parameters(data={"total_beam_charge_pc": 2.0}).to_json_file(
        tmp_path / "results" / "datapoint.json"
    )
    with pytest.raises(ValueError, match="ConfigContainer"):
        result_executor.extract_features(tmp_path)


def test_extract_features_requires_the_results_hdf5_group(
    tmp_path, result_executor, saved_datapoint
) -> None:
    saved_datapoint.to_hdf5_file(tmp_path / "results" / "datapoint.h5")
    with pytest.raises(ValueError, match="/results"):
        result_executor.extract_features(tmp_path)


def test_extract_features_ignores_metadata_and_rejects_empty_analysis(
    tmp_path, result_executor
) -> None:
    ConfigContainer(
        configs=[
            Parameters(data={"total_beam_charge_pc": 2.0}),
            MeanKineticEnergy(target=("all_of_species", "e")),
        ]
    ).to_json_file(tmp_path / "results" / "datapoint.json")
    with pytest.raises(ValueError, match="No scalar diagnostic results"):
        result_executor.extract_features(tmp_path)


def test_extract_features_requires_configured_exploration_outputs(
    tmp_path, result_executor
) -> None:
    ConfigContainer(
        configs=[
            MomentDescriptorDiagnostic(
                target=("all_of_species", "e"), data={"total_beam_charge_pc": 2.0}
            )
        ]
    ).to_json_file(tmp_path / "results" / "datapoint.json")
    with pytest.raises(
        ValueError, match="Missing saved exploration features.*mean_kinetic_energy_MeV"
    ):
        result_executor.extract_features(tmp_path)


def test_extract_features_rejects_duplicate_feature_names(
    tmp_path, result_executor
) -> None:
    ConfigContainer(
        configs=[
            MeanKineticEnergy(
                target=("all_of_species", "e"),
                data={"mean_kinetic_energy_MeV": value},
            )
            for value in (1.0, 2.0)
        ]
    ).to_json_file(tmp_path / "results" / "datapoint.json")
    with pytest.raises(ValueError, match="Duplicate saved diagnostic feature"):
        result_executor.extract_features(tmp_path)


@pytest.mark.parametrize("value", [None, True, "1.25", [1.25], np.nan, np.inf, -np.inf])
def test_extract_features_requires_finite_numeric_scalars(
    tmp_path, result_executor, value
) -> None:
    ConfigContainer(
        configs=[
            MeanKineticEnergy(
                target=("all_of_species", "e"),
                data={"mean_kinetic_energy_MeV": value},
            )
        ]
    ).to_json_file(tmp_path / "results" / "datapoint.json")
    with pytest.raises(
        ValueError, match="Saved diagnostic feature mean_kinetic_energy_MeV"
    ):
        result_executor.extract_features(tmp_path)


def test_evaluate_uses_dumped_results(
    tmp_path, result_executor, saved_datapoint, monkeypatch
):
    monkeypatch.setattr(
        result_executor,
        "_execute",
        lambda directory: saved_datapoint.to_json_file(
            directory / "results" / "datapoint.json"
        ),
    )
    result = result_executor.evaluate("sim_000001", "stage_1")
    assert result.status == "ok"
    assert result.features["mean_kinetic_energy_MeV"] == 1.25
    stored = json.loads(
        (result.run_directory / "result.json").read_text(encoding="utf-8")
    )
    assert stored["result"]["features"] == result.features


def test_evaluate_records_missing_dump_as_analysis_error(result_executor, monkeypatch):
    monkeypatch.setattr(result_executor, "_execute", Mock())
    result = result_executor.evaluate("sim_000001", "stage_1")
    assert result.status == "analysis_error"
    assert "No saved datapoint config" in result.message
    assert result.features == {}
