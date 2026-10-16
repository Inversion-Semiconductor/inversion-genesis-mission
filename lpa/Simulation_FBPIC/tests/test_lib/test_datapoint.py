"""Dictionary-backed inputs and the abstract diagnostics analysis lifecycle."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import Mock

import attrs
import numpy as np
import pytest

from inversion_fbpic.lib.config_container import ConfigContainer
from inversion_fbpic.lib.datapoint import _Datapoint, Parameters
from inversion_fbpic.lib.diagnostics import (
    _Diagnostic,
    _ParticleDiagnostic,
    MeanKineticEnergy,
    MomentDescriptorDiagnostic,
)
from inversion_fbpic.lib.serializable_config import SerializableConfig

SERIALIZERS = ("to_dict", "to_json", "to_yaml", "to_json_file", "to_yaml_file")
LOGGER = "inversion_fbpic.lib.diagnostics"


@pytest.fixture()
def diagnostic_class(monkeypatch: pytest.MonkeyPatch):
    """Register a lightweight diagnostic without leaking its tag to other tests."""
    monkeypatch.setattr(
        _Datapoint, "_CONCRETE_REGISTRY", dict(_Datapoint._CONCRETE_REGISTRY)
    )

    @attrs.define(kw_only=True, slots=False)
    class SampleDiagnostics(_Diagnostic):
        SUBCLASS: ClassVar[str] = "test_sample_diagnostics"

        charge: float = attrs.field(default=3.0)
        calls: int = attrs.field(default=0, init=False)

        def _analyze(self) -> dict[str, Any]:
            self.calls += 1
            return {"total_charge": self.charge, "calls": self.calls}

    return SampleDiagnostics


def _serialize(config: SerializableConfig, method: str, tmp_path: Path) -> Any:
    serializer = getattr(config, method)
    if method.endswith("_file"):
        extension = "json" if method == "to_json_file" else "yaml"
        return serializer(tmp_path / "output" / f"config.{extension}")
    return serializer()


def _round_trip(config: SerializableConfig, method: str, tmp_path: Path):
    value = _serialize(config, method, tmp_path)
    if method == "to_dict":
        return SerializableConfig.from_dict(value)
    if method == "to_json":
        return SerializableConfig.from_json(value)
    if method == "to_yaml":
        return SerializableConfig.from_yaml(value)
    return SerializableConfig.from_file(value)


@pytest.mark.parametrize("label", ["", "scan inputs"])
def test_parameters_retain_live_dictionary(label: str) -> None:
    from inversion_fbpic.lib import serializable_config as module

    data = {"energy": 5.0, "nested": {"selection": [10.0, None]}}
    parameters = Parameters(label=label, data=data)
    assert parameters.label == label
    assert parameters.data is data
    data["energy"] = 6.0
    assert parameters.data["energy"] == 6.0
    parameters.data["new"] = "value"
    assert data["new"] == "value"
    assert parameters.to_dict() == {
        "config_type": "datapoint",
        "subclass": "parameters",
        "git_hash": module._git_hash(),
        "parameters": {"label": label, "data": data},
    }


def test_parameters_require_dictionary_at_construction() -> None:
    with pytest.raises(TypeError, match="data"):
        Parameters()  # type: ignore[call-arg]


@pytest.mark.parametrize("data", [None, [], "values", 1])
def test_parameters_reject_non_dictionary(data: Any) -> None:
    with pytest.raises(TypeError, match="data"):
        Parameters(data=data)


@pytest.mark.parametrize("label", ["", "scan inputs"])
@pytest.mark.parametrize("method", SERIALIZERS)
def test_parameters_round_trip(method: str, tmp_path: Path, label: str) -> None:
    original = Parameters(
        label=label,
        data={"energy": 5.0, "nested": {"selection": [10.0, None]}, "label": "run"},
    )
    loaded = _round_trip(original, method, tmp_path)
    assert type(loaded) is Parameters
    assert loaded.label == label
    assert loaded.data == original.data


@pytest.mark.parametrize("label", [None, [], {}, 1])
def test_parameters_reject_non_string_label(label: Any) -> None:
    with pytest.raises(TypeError, match="label"):
        Parameters(label=label, data={})


def test_parameters_default_label_for_construction_and_legacy_payload() -> None:
    parameters = Parameters(data={"energy": 5.0})
    assert parameters.label == ""
    payload = parameters.to_dict()
    del payload["parameters"]["label"]
    loaded = Parameters.from_dict(payload)
    assert isinstance(loaded, Parameters)
    assert loaded.label == ""
    assert loaded.data == parameters.data


def test_parameters_keep_config_like_user_dictionaries_as_data() -> None:
    user_data = {
        "config_type": "not_a_domain",
        "subclass": "not_a_config",
        "parameters": {"file": "does_not_exist.yaml"},
        "registered_payload": Parameters(data={"energy": 5.0}).to_dict(),
    }
    loaded = _Datapoint.from_json(Parameters(data=user_data).to_json())
    assert isinstance(loaded, Parameters)
    assert loaded.data == user_data
    assert isinstance(loaded.data["registered_payload"], dict)


def test_parameters_use_standard_serialization_conversions(tmp_path: Path) -> None:
    original = Parameters(
        data={
            "array": np.array([1.0, 2.0]),
            "scalar": np.float64(3.0),
            "complex": 1.0 + 2.0j,
            "tuple": (4, None),
            "path": tmp_path / "result.h5",
        }
    )
    loaded = Parameters.from_json(original.to_json())
    assert isinstance(loaded, Parameters)
    assert loaded.data == {
        "array": [1.0, 2.0],
        "scalar": 3.0,
        "complex": [1.0, 2.0],
        "tuple": [4, None],
        "path": (tmp_path / "result.h5").as_posix(),
    }


@pytest.mark.parametrize("label", ["", "scan inputs"])
@pytest.mark.parametrize("include_nones", [True, False])
def test_empty_parameters_load_with_collection_fallback(
    include_nones: bool, label: str
) -> None:
    payload = Parameters(label=label, data={}).to_dict(include_nones=include_nones)
    assert ("data" in payload["parameters"]) is include_nones
    first = Parameters.from_dict(payload)
    second = Parameters.from_dict(payload)
    assert isinstance(first, Parameters) and isinstance(second, Parameters)
    assert first.data == second.data == {}
    assert first.label == second.label == label
    if include_nones:
        # An explicitly supplied dict stays live, including a payload's dict.
        assert first.data is second.data is payload["parameters"]["data"]
    else:
        assert first.data is not second.data


def test_parameters_preserve_nested_nulls_when_omitting_optional_fields() -> None:
    parameters = Parameters(data={"optional": None, "selection": [10.0, None]})
    payload = parameters.to_dict(include_nones=False)
    assert payload["parameters"]["data"] == parameters.data


def test_datapoint_domain_and_diagnostics_are_not_concrete() -> None:
    assert SerializableConfig._DOMAIN_REGISTRY["datapoint"] is _Datapoint
    assert "dataset" not in SerializableConfig._DOMAIN_REGISTRY
    assert _Datapoint._CONCRETE_REGISTRY["parameters"] is Parameters
    assert _Diagnostic.__module__ == "inversion_fbpic.lib.diagnostics"
    assert _Diagnostic._CONCRETE_REGISTRY is _Datapoint._CONCRETE_REGISTRY
    assert _Diagnostic not in _Datapoint._CONCRETE_REGISTRY.values()
    assert "SUBCLASS" not in _Datapoint.__dict__
    assert "SUBCLASS" not in _Diagnostic.__dict__
    assert inspect.isabstract(_Diagnostic)
    with pytest.raises(TypeError, match="abstract"):
        _Diagnostic()  # type: ignore[abstract]


def test_diagnostics_store_returned_dictionary_by_identity(
    diagnostic_class, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostic = diagnostic_class()
    assert not diagnostic.analysis_complete
    assert diagnostic.calls == 0
    results = {"total_charge": 7.0}
    monkeypatch.setattr(diagnostic, "_analyze", lambda: results)
    assert diagnostic.analyze() is None
    assert diagnostic.data is results
    assert diagnostic.analysis_complete
    with pytest.raises(AttributeError):
        diagnostic.analysis_complete = False


def test_each_analyze_call_reruns_analysis(diagnostic_class) -> None:
    diagnostic = diagnostic_class()
    diagnostic.analyze()
    assert diagnostic.calls == 1
    assert diagnostic.data == {"total_charge": 3.0, "calls": 1}
    diagnostic.charge = 5.0
    diagnostic.analyze()
    assert diagnostic.calls == 2
    assert diagnostic.data == {"total_charge": 5.0, "calls": 2}
    assert diagnostic.analysis_complete


@pytest.mark.parametrize("previously_complete", [False, True])
def test_failed_analysis_keeps_previous_data_and_clears_completion(
    diagnostic_class, monkeypatch: pytest.MonkeyPatch, previously_complete: bool
) -> None:
    diagnostic = diagnostic_class(data={"previous": 1.0})
    if previously_complete:
        diagnostic.analyze()
    previous = diagnostic.data

    def fail() -> dict[str, Any]:
        raise RuntimeError("analysis failed")

    monkeypatch.setattr(diagnostic, "_analyze", fail)
    with pytest.raises(RuntimeError, match="analysis failed"):
        diagnostic.analyze()
    assert not diagnostic.analysis_complete
    assert diagnostic.data is previous


@pytest.mark.parametrize("invalid_result", [None, [], 3.0, "results"])
def test_invalid_analysis_result_is_rejected(
    diagnostic_class, monkeypatch: pytest.MonkeyPatch, invalid_result: Any
) -> None:
    diagnostic = diagnostic_class()
    diagnostic.analyze()
    previous = diagnostic.data
    monkeypatch.setattr(diagnostic, "_analyze", lambda: invalid_result)
    with pytest.raises(TypeError, match=r"_analyze\(\) must return a dict"):
        diagnostic.analyze()
    assert not diagnostic.analysis_complete
    assert diagnostic.data is previous


@pytest.mark.parametrize("method", SERIALIZERS)
@pytest.mark.parametrize("nested", [False, True])
def test_serializing_incomplete_diagnostics_logs_warning_without_analyzing(
    diagnostic_class, method: str, nested: bool, tmp_path: Path, caplog
) -> None:
    diagnostic = diagnostic_class(data={"saved": 1.0})
    config = (
        ConfigContainer(configs=[ConfigContainer(configs=[diagnostic])])
        if nested
        else diagnostic
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _serialize(config, method, tmp_path)
    assert diagnostic.calls == 0
    assert diagnostic.data == {"saved": 1.0}
    assert not diagnostic.analysis_complete
    records = [record for record in caplog.records if record.name == LOGGER]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "SampleDiagnostics" in records[0].message
    assert "call analyze()" in records[0].message


@pytest.mark.parametrize("method", SERIALIZERS)
def test_completed_diagnostics_do_not_warn(
    diagnostic_class, method: str, tmp_path: Path, caplog
) -> None:
    diagnostic = diagnostic_class()
    diagnostic.analyze()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _serialize(ConfigContainer(configs=[diagnostic]), method, tmp_path)
    assert not caplog.records
    assert diagnostic.calls == 1


@pytest.mark.parametrize("method", SERIALIZERS)
def test_loaded_diagnostics_keep_results_but_reset_runtime_completion(
    diagnostic_class, method: str, tmp_path: Path, caplog
) -> None:
    diagnostic = diagnostic_class(charge=7.0)
    diagnostic.analyze()
    loaded = _round_trip(diagnostic, method, tmp_path)
    assert type(loaded) is diagnostic_class
    assert loaded.data == diagnostic.data
    assert loaded.charge == 7.0
    assert loaded.calls == 0
    assert not loaded.analysis_complete
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        payload = loaded.to_dict()
    assert "incomplete" in caplog.text
    assert set(payload["parameters"]) == {"data", "charge"}
    loaded.analyze()
    assert loaded.analysis_complete


@pytest.mark.parametrize("name", ["analysis_complete", "_analysis_complete"])
def test_serialized_payload_cannot_supply_completion_state(
    diagnostic_class, name
) -> None:
    payload = {
        "config_type": "datapoint",
        "subclass": diagnostic_class.SUBCLASS,
        "parameters": {name: True},
    }
    with pytest.raises(ValueError, match="Unknown parameter"):
        SerializableConfig.from_dict(payload)


def test_diagnostics_defaults_and_examples_are_independent_and_do_not_warn(
    diagnostic_class, caplog
) -> None:
    assert diagnostic_class().data is not diagnostic_class().data
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        payload = diagnostic_class.example_dict()
        diagnostic_class.example_json()
        diagnostic_class.example_yaml()
    assert payload["parameters"] == {"data": {}, "charge": 3.0}
    assert not caplog.records


@pytest.fixture()
def moment_simulation(
    minimal_hyperparameters, minimal_density, minimal_laser, tmp_path: Path
):
    from inversion_fbpic.lib.simulation import Simulation

    diagnostic = MomentDescriptorDiagnostic(target=("elec_name", "electrons"))
    minimal_density = attrs.evolve(minimal_density, elec_name="electrons")
    simulation = Simulation(
        elements=ConfigContainer(
            configs=[
                minimal_hyperparameters,
                ConfigContainer(configs=[minimal_density, minimal_laser, diagnostic]),
            ]
        )
    )
    simulation.working_directory = tmp_path
    return simulation, diagnostic


@pytest.fixture()
def moment_particles():
    particles = np.random.default_rng(17).normal(size=(256, 6))
    weights = np.linspace(1.0, 3.0, len(particles))
    return particles, weights


def test_particle_diagnostic_base_is_abstract_and_not_registered() -> None:
    assert inspect.isabstract(_ParticleDiagnostic)
    assert "SUBCLASS" not in _ParticleDiagnostic.__dict__
    assert _ParticleDiagnostic not in _Datapoint._CONCRETE_REGISTRY.values()
    assert issubclass(MomentDescriptorDiagnostic, _ParticleDiagnostic)
    with pytest.raises(TypeError, match="abstract"):
        _ParticleDiagnostic(target=("all_of_species", "e"))  # type: ignore[abstract]


def test_particle_loading_does_not_mutate_analysis_state(
    moment_simulation, moment_particles, monkeypatch
) -> None:
    from inversion_fbpic.lib import diagnostics as module

    simulation, diagnostic = moment_simulation
    directory = simulation._save_directory / "hdf5"
    directory.mkdir(parents=True)
    latest = directory / "data00000010.h5"
    latest.touch()
    loader = Mock(return_value=moment_particles)
    calculation = Mock(
        side_effect=AssertionError("Loading must not compute a descriptor")
    )
    cropper = Mock(side_effect=AssertionError("Default loading must not crop"))
    monkeypatch.setattr(module, "load_openpmd_particles", loader)
    monkeypatch.setattr(module, "compute_moment_descriptor", calculation)
    monkeypatch.setattr(module, "crop_central_particles", cropper)
    assert diagnostic.central_fraction is None
    previous = {"saved": 1.0}
    diagnostic.data = previous
    particles, weights = diagnostic.load_particles()
    np.testing.assert_array_equal(particles, moment_particles[0])
    np.testing.assert_array_equal(weights, moment_particles[1])
    assert diagnostic.data is previous
    assert not diagnostic.analysis_complete
    loader.assert_called_once_with(latest, "electrons", select=None)
    calculation.assert_not_called()
    cropper.assert_not_called()


@pytest.mark.parametrize(
    "diagnostic_class", [MeanKineticEnergy, MomentDescriptorDiagnostic]
)
@pytest.mark.parametrize("central_fraction", [0.0, -0.1, 1.1, np.nan, np.inf, -np.inf])
def test_particle_diagnostic_rejects_invalid_central_fraction(
    diagnostic_class, central_fraction
) -> None:
    with pytest.raises(ValueError, match="central_fraction"):
        diagnostic_class(
            target=("all_of_species", "e"), central_fraction=central_fraction
        )


@pytest.mark.parametrize(
    "diagnostic_class", [MeanKineticEnergy, MomentDescriptorDiagnostic]
)
def test_particle_diagnostic_crops_combined_recordings(
    moment_simulation, moment_particles, monkeypatch, diagnostic_class
) -> None:
    from inversion_fbpic.lib import diagnostics as module
    from inversion_fbpic.utils.distributions import crop_central_particles

    simulation, _ = moment_simulation
    simulation.densities.append(
        attrs.evolve(simulation.densities[0], elec_name="second")
    )
    diagnostic = diagnostic_class(
        target=("elec_name", ["electrons", "second"]),
        select={"uz": [0.0, None]},
        central_fraction=0.8,
    )
    simulation._sort_component(diagnostic)
    directory = simulation._save_directory / "hdf5"
    directory.mkdir(parents=True)
    latest = directory / "data10.h5"
    latest.touch()
    particles, weights = moment_particles
    loader = Mock(
        side_effect=[
            (particles[:128], weights[:128]),
            (particles[128:], weights[128:]),
        ]
    )
    cropper = Mock(wraps=crop_central_particles)
    monkeypatch.setattr(module, "load_openpmd_particles", loader)
    monkeypatch.setattr(module, "crop_central_particles", cropper)
    expected_particles, expected_weights = crop_central_particles(
        particles, weights, central_fraction=0.8
    )

    loaded_particles, loaded_weights = diagnostic.load_particles()

    assert len(loaded_particles) < len(particles)
    np.testing.assert_array_equal(loaded_particles, expected_particles)
    np.testing.assert_array_equal(loaded_weights, expected_weights)
    assert [entry.args for entry in loader.call_args_list] == [
        (latest, "electrons"),
        (latest, "second"),
    ]
    assert all(
        entry.kwargs == {"select": diagnostic.select} for entry in loader.call_args_list
    )
    cropper.assert_called_once()
    np.testing.assert_array_equal(cropper.call_args.args[0], particles)
    np.testing.assert_array_equal(cropper.call_args.args[1], weights)
    assert cropper.call_args.kwargs == {"central_fraction": 0.8}


@pytest.mark.parametrize(
    "diagnostic_class", [MeanKineticEnergy, MomentDescriptorDiagnostic]
)
@pytest.mark.parametrize("central_fraction", [None, 0.95])
@pytest.mark.parametrize("method", SERIALIZERS)
def test_particle_diagnostic_central_fraction_round_trip(
    diagnostic_class, central_fraction, method, tmp_path
) -> None:
    diagnostic = diagnostic_class(
        target=("all_of_species", "e"), central_fraction=central_fraction
    )
    loaded = _round_trip(diagnostic, method, tmp_path)
    assert type(loaded) is diagnostic_class
    assert loaded.central_fraction == central_fraction
    assert diagnostic.to_dict()["parameters"]["central_fraction"] == central_fraction
    assert (
        "central_fraction" in diagnostic.to_dict(include_nones=False)["parameters"]
    ) is (central_fraction is not None)


def test_another_particle_diagnostic_reuses_data_gathering(
    moment_simulation, moment_particles, monkeypatch
) -> None:
    from inversion_fbpic.lib import diagnostics as module

    monkeypatch.setattr(
        _Datapoint, "_CONCRETE_REGISTRY", dict(_Datapoint._CONCRETE_REGISTRY)
    )

    @attrs.define(kw_only=True, slots=False)
    class ParticleCountDiagnostic(_ParticleDiagnostic):
        SUBCLASS: ClassVar[str] = "test_particle_count"

        def _analyze(self) -> dict[str, Any]:
            particles, weights = self.load_particles()
            return {
                "count": float(len(particles)),
                "total_weight": float(weights.sum()),
            }

    simulation, _ = moment_simulation
    diagnostic = ParticleCountDiagnostic(target=("all_of_species", "e"))
    simulation._sort_component(diagnostic)
    directory = simulation._save_directory / "hdf5"
    directory.mkdir(parents=True)
    (directory / "data00000010.h5").touch()
    monkeypatch.setattr(
        module, "load_openpmd_particles", Mock(return_value=moment_particles)
    )
    diagnostic.analyze()
    assert diagnostic.data == {
        "count": float(len(moment_particles[0])),
        "total_weight": float(moment_particles[1].sum()),
    }
    assert diagnostic.analysis_complete


def test_mean_kinetic_energy_analyzes_latest_selected_particles(
    moment_simulation, monkeypatch
) -> None:
    from inversion_fbpic.lib import diagnostics as module

    simulation, _ = moment_simulation
    diagnostic = MeanKineticEnergy(
        target=("elec_name", "electrons"), select={"uz": [1.0, None]}
    )
    simulation._sort_component(diagnostic)
    assert isinstance(diagnostic, _ParticleDiagnostic)
    assert diagnostic.RUN_BEFORE_SIMULATION is False
    assert diagnostic.data == {}
    assert not diagnostic.analysis_complete
    directory = simulation._save_directory / "hdf5"
    directory.mkdir(parents=True)
    (directory / "data9.h5").touch()
    latest = directory / "data10.h5"
    latest.touch()
    particles = np.zeros((2, 6))
    particles[1, (1, 3, 5)] = 1.0
    loader = Mock(return_value=(particles, np.array([1.0, 3.0])))
    monkeypatch.setattr(module, "load_openpmd_particles", loader)

    diagnostic.analyze()

    assert diagnostic.analysis_complete
    assert diagnostic.data == {
        "mean_kinetic_energy_MeV": pytest.approx(0.51099895 * 3.0 / 4.0)
    }
    loader.assert_called_once_with(latest, "electrons", select=diagnostic.select)


@pytest.mark.parametrize("method", SERIALIZERS)
def test_mean_kinetic_energy_round_trip(tmp_path: Path, method: str) -> None:
    diagnostic = MeanKineticEnergy(
        target=("elec_name", ["electrons_he", "electrons_n2"]),
        select={"uz": [10.0, None]},
        data={"mean_kinetic_energy_MeV": 1.25},
    )
    loaded = _round_trip(diagnostic, method, tmp_path)
    assert type(loaded) is MeanKineticEnergy
    assert loaded.SUBCLASS == "mean_kinetic_energy"
    assert loaded.target == diagnostic.target
    assert loaded.select == diagnostic.select
    assert loaded.data == diagnostic.data
    assert not loaded.analysis_complete
    with pytest.raises(ValueError, match="Attach"):
        loaded.attached_simulation


def test_mean_kinetic_energy_failure_preserves_previous_data(monkeypatch) -> None:
    diagnostic = MeanKineticEnergy(target=("all_of_species", "e"))
    loader = Mock(return_value=(np.zeros((2, 6)), np.ones(2)))
    monkeypatch.setattr(diagnostic, "load_particles", loader)
    diagnostic.analyze()
    previous = diagnostic.data
    assert diagnostic.analysis_complete
    loader.return_value = (np.zeros((2, 6)), np.zeros(2))
    with pytest.raises(ZeroDivisionError):
        diagnostic.analyze()
    assert diagnostic.data is previous
    assert not diagnostic.analysis_complete


def test_simulation_attaches_nested_diagnostic(moment_simulation) -> None:
    simulation, diagnostic = moment_simulation
    assert diagnostic.attached_simulation is simulation
    diagnostic.attach(simulation)
    assert diagnostic.attached_simulation is simulation
    assert not diagnostic.analysis_complete


def test_diagnostic_attachment_is_required_and_type_checked() -> None:
    diagnostic = MomentDescriptorDiagnostic(target=("elec_name", "electrons"))
    with pytest.raises(ValueError, match="Attach"):
        diagnostic.analyze()
    with pytest.raises(TypeError, match="Simulation"):
        diagnostic.attach(object())  # type: ignore[arg-type]
    assert not diagnostic.analysis_complete


def test_diagnostic_rejects_attachment_to_another_simulation(
    moment_simulation, minimal_hyperparameters, minimal_density, minimal_laser
) -> None:
    from inversion_fbpic.lib.simulation import Simulation

    original, diagnostic = moment_simulation
    with pytest.raises(ValueError, match="another simulation"):
        Simulation(
            elements=[
                minimal_hyperparameters,
                minimal_density,
                minimal_laser,
                diagnostic,
            ]
        )
    assert diagnostic.attached_simulation is original


def test_moment_descriptor_selects_highest_iteration_and_stores_results(
    moment_simulation, moment_particles, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inversion_fbpic.lib import diagnostics as module
    from inversion_fbpic.utils.distributions import compute_moment_descriptor

    simulation, diagnostic = moment_simulation
    directory = simulation._save_directory / "hdf5"
    directory.mkdir(parents=True)
    for name in ("data9.h5", "data10.h5", "data_999.h5", "data100.h5.tmp"):
        (directory / name).touch()
    (directory / "data999.h5").mkdir()
    loader = Mock(return_value=moment_particles)
    monkeypatch.setattr(module, "load_openpmd_particles", loader)
    diagnostic.analyze()
    loader.assert_called_once_with(directory / "data10.h5", "electrons", select=None)
    assert diagnostic.data == compute_moment_descriptor(*moment_particles)
    assert len(diagnostic.data) == 33
    assert diagnostic.analysis_complete
    (directory / "data11.h5").touch()
    loader.reset_mock()
    diagnostic.analyze()
    loader.assert_called_once_with(directory / "data11.h5", "electrons", select=None)


def test_moment_descriptor_exposes_constants_as_class_variables() -> None:
    from inversion_fbpic.utils import distributions

    diagnostic = MomentDescriptorDiagnostic(target=("elec_name", "electrons"))
    fields = {field.name for field in attrs.fields(MomentDescriptorDiagnostic)}
    constructor_parameters = inspect.signature(MomentDescriptorDiagnostic).parameters
    serialized_parameters = diagnostic.to_dict()["parameters"]
    for name in ("OFF", "MOMENTS", "SPLINE", "LONGITUDINAL_PROFILE_BINS"):
        expected = getattr(distributions, name)
        assert getattr(MomentDescriptorDiagnostic, name) == expected
        assert getattr(diagnostic, name) == expected
        assert name not in fields
        assert name not in constructor_parameters
        assert name not in serialized_parameters
    assert diagnostic.longitudinal_mode == MomentDescriptorDiagnostic.SPLINE
    assert (
        diagnostic.longitudinal_bins
        == MomentDescriptorDiagnostic.LONGITUDINAL_PROFILE_BINS
    )


@pytest.mark.parametrize(
    "mode",
    [
        MomentDescriptorDiagnostic.OFF,
        MomentDescriptorDiagnostic.MOMENTS,
        MomentDescriptorDiagnostic.SPLINE,
    ],
)
def test_moment_descriptor_forwards_all_options(
    moment_simulation, moment_particles, monkeypatch, mode: int
) -> None:
    from inversion_fbpic.lib import diagnostics as module
    from inversion_fbpic.utils.distributions import compute_moment_descriptor

    simulation, diagnostic = moment_simulation
    directory = simulation._save_directory / "hdf5"
    directory.mkdir(parents=True)
    (directory / "data00000010.h5").touch()
    diagnostic.longitudinal_mode = mode
    diagnostic.longitudinal_bins = 6
    diagnostic.include_total_weight = True
    diagnostic.include_higher_moments = True
    monkeypatch.setattr(
        module, "load_openpmd_particles", Mock(return_value=moment_particles)
    )
    diagnostic.analyze()
    assert diagnostic.data == compute_moment_descriptor(
        *moment_particles,
        longitudinal_mode=mode,
        longitudinal_bins=6,
        include_total_weight=True,
        include_higher_moments=True,
    )


def test_moment_descriptor_uses_absolute_save_directory(
    moment_simulation, moment_particles, monkeypatch, tmp_path
) -> None:
    from inversion_fbpic.lib import diagnostics as module

    simulation, diagnostic = moment_simulation
    hyperparameters = attrs.evolve(
        simulation.hyparams, save_directory=tmp_path / "external"
    )
    simulation.hyparams = hyperparameters
    simulation.working_directory = tmp_path / "different-run"
    directory = tmp_path / "external" / "hdf5"
    directory.mkdir(parents=True)
    last_file = directory / "data00000012.h5"
    last_file.touch()
    loader = Mock(return_value=moment_particles)
    monkeypatch.setattr(module, "load_openpmd_particles", loader)
    diagnostic.analyze()
    loader.assert_called_once_with(last_file, "electrons", select=None)


@pytest.mark.parametrize(
    "select",
    [
        None,
        {},
        *(
            {coordinate: [-0.5, 0.5]}
            for coordinate in ("x", "ux", "y", "uy", "z", "uz")
        ),
        {"uz": [0.0, None], "x": [None, 0.5], "z": [None, None]},
    ],
)
def test_moment_descriptor_reads_real_openpmd_file(
    moment_simulation, moment_particles, select
) -> None:
    import h5py
    from scipy.constants import c, m_e
    from inversion_fbpic.utils.distributions import compute_moment_descriptor

    simulation, diagnostic = moment_simulation
    particles, weights = moment_particles
    directory = simulation._save_directory / "hdf5"
    directory.mkdir(parents=True)
    with h5py.File(directory / "data00000010.h5", "w") as file:
        file.attrs["openPMD"] = np.bytes_("1.1.0")
        file.attrs["openPMDextension"] = np.uint32(0)
        file.attrs["basePath"] = np.bytes_("/data/%T/")
        file.attrs["particlesPath"] = np.bytes_("particles/")
        file.attrs["iterationEncoding"] = np.bytes_("fileBased")
        file.attrs["iterationFormat"] = np.bytes_("data%T.h5")
        iteration = file.create_group("data/10")
        iteration.attrs.update(time=0.0, dt=1.0, timeUnitSI=1.0)
        species = iteration.create_group("particles/electrons")
        for index, coordinate in enumerate(("x", "y", "z")):
            for name, values, scale, dimensions in (
                ("position", particles[:, index * 2], 1.0, [1, 0, 0, 0, 0, 0, 0]),
                (
                    "positionOffset",
                    np.zeros(len(particles)),
                    1.0,
                    [1, 0, 0, 0, 0, 0, 0],
                ),
                (
                    "momentum",
                    particles[:, index * 2 + 1],
                    m_e * c,
                    [1, 1, -1, 0, 0, 0, 0],
                ),
            ):
                record = species.require_group(name)
                record.attrs.update(
                    unitDimension=np.asarray(dimensions, dtype=float),
                    timeOffset=0.0,
                    macroWeighted=np.uint32(0),
                    weightingPower=0.0,
                )
                component = record.create_dataset(coordinate, data=values)
                component.attrs["unitSI"] = scale
        for name, values, dimensions in (
            ("weighting", weights, [0, 0, 0, 0, 0, 0, 0]),
            ("mass", np.full(len(particles), m_e), [0, 1, 0, 0, 0, 0, 0]),
        ):
            record = species.create_dataset(name, data=values)
            record.attrs.update(
                unitSI=1.0,
                unitDimension=np.asarray(dimensions, dtype=float),
                timeOffset=0.0,
                macroWeighted=np.uint32(0),
                weightingPower=0.0,
            )
    diagnostic.select = select
    retained = np.ones(len(particles), dtype=bool)
    coordinates = ("x", "ux", "y", "uy", "z", "uz")
    for coordinate, (lower, upper) in (select or {}).items():
        values = particles[:, coordinates.index(coordinate)]
        if lower is not None:
            retained &= values > lower
        if upper is not None:
            retained &= values < upper
    loaded_particles, loaded_weights = diagnostic.load_particles()
    np.testing.assert_allclose(loaded_particles, particles[retained])
    np.testing.assert_allclose(loaded_weights, weights[retained])
    diagnostic.analyze()
    assert diagnostic.analysis_complete
    assert diagnostic.data == pytest.approx(
        compute_moment_descriptor(particles[retained], weights[retained])
    )


def test_moment_descriptor_finds_no_output_only_at_analysis_time(
    moment_simulation,
) -> None:
    simulation, diagnostic = moment_simulation
    assert diagnostic.attached_simulation is simulation
    with pytest.raises(ValueError, match="No openPMD particle diagnostic files"):
        diagnostic.analyze()
    assert not diagnostic.analysis_complete


@pytest.mark.parametrize("failure", ["loading", "calculation"])
def test_moment_descriptor_failure_retains_previous_data(
    moment_simulation, moment_particles, monkeypatch, failure: str
) -> None:
    from inversion_fbpic.lib import diagnostics as module

    simulation, diagnostic = moment_simulation
    directory = simulation._save_directory / "hdf5"
    directory.mkdir(parents=True)
    (directory / "data00000010.h5").touch()
    monkeypatch.setattr(
        module, "load_openpmd_particles", Mock(return_value=moment_particles)
    )
    diagnostic.analyze()
    previous = diagnostic.data
    target = (
        "load_openpmd_particles"
        if failure == "loading"
        else "compute_moment_descriptor"
    )
    monkeypatch.setattr(
        module, target, Mock(side_effect=ValueError("invalid beam data"))
    )
    with pytest.raises(ValueError, match="invalid beam data"):
        diagnostic.analyze()
    assert diagnostic.data is previous
    assert not diagnostic.analysis_complete


@pytest.mark.parametrize("method", SERIALIZERS)
def test_moment_descriptor_round_trip_excludes_attachment(
    moment_simulation, tmp_path: Path, method: str
) -> None:
    simulation, diagnostic = moment_simulation
    diagnostic.data = {"total_beam_charge_pc": 12.0}
    loaded = _round_trip(diagnostic, method, tmp_path)
    assert type(loaded) is MomentDescriptorDiagnostic
    assert loaded.data == diagnostic.data
    assert loaded.target == ("elec_name", "electrons")
    assert isinstance(loaded.target, tuple)
    assert loaded.select is None
    assert not loaded.analysis_complete
    with pytest.raises(ValueError, match="Attach"):
        loaded.attached_simulation
    assert "_simulation" not in diagnostic.to_dict()["parameters"]
    loaded.attach(simulation)
    assert loaded.attached_simulation is simulation


def test_simulation_round_trip_reattaches_descriptor(moment_simulation) -> None:
    from inversion_fbpic.lib.simulation import Simulation

    simulation, diagnostic = moment_simulation
    diagnostic.data = {"total_beam_charge_pc": 12.0}
    loaded = SerializableConfig.from_json(simulation.to_json())
    assert isinstance(loaded, Simulation)
    descriptor = loaded.diagnostics[0]
    assert isinstance(descriptor, MomentDescriptorDiagnostic)
    assert descriptor.attached_simulation is loaded
    assert descriptor.data == diagnostic.data


@pytest.mark.parametrize(
    "arguments",
    [
        {"target": "electrons"},
        {"target": ()},
        {"target": ("elec_name",)},
        {"target": ("elec_name", "electrons", None, {})},
        {"target": ("elec_name", "electrons", {"uz": [10.0, None]})},
        {"target": ("unknown", "electrons")},
        {"target": ("elec_name", "")},
        {"target": ("elec_name", [])},
        {"target": ("ion_name", ["ions", None])},
        {"target": ("all_of_species", ["He"])},
        {"target": ("all_of_species", "")},
        {"target": ("elec_name", "electrons"), "select": []},
        {"target": ("elec_name", "electrons"), "select": 0},
        {"target": ("elec_name", "electrons"), "select": "extra"},
        {"target": ("elec_name", "electrons"), "longitudinal_mode": 99},
        {"target": ("elec_name", "electrons"), "longitudinal_bins": 3},
        {"target": ("elec_name", "electrons"), "include_total_weight": "true"},
    ],
)
def test_moment_descriptor_validates_inputs(arguments) -> None:
    with pytest.raises((TypeError, ValueError)):
        MomentDescriptorDiagnostic(**arguments)


@pytest.mark.parametrize(
    "target,expected_names",
    [
        (("elec_name", ["e_He", "e_H"]), ["e_He", "e_H"]),
        (("elec_name", "e_He"), ["e_He"]),
        (("ion_name", ["ions_He", "ions_H"]), ["ions_He", "ions_H"]),
        (("ion_name", "ions_He"), ["ions_He"]),
        (("all_of_species", "e"), ["e_He", "e_H", "e_bare"]),
        (("all_of_species", "He"), ["ions_He", "ions_He_second"]),
        (("elec_name", ["e_H", "e_He", "e_H"]), ["e_H", "e_He"]),
    ],
)
@pytest.mark.parametrize(
    "select",
    [None, {}, {"uz": [10.0, None], "x": [-1.0e-6, 1.0e-6]}],
)
def test_moment_descriptor_concatenates_selected_recordings(
    moment_simulation,
    moment_particles,
    monkeypatch,
    target,
    expected_names,
    select,
) -> None:
    from inversion_fbpic.lib import diagnostics as module
    from inversion_fbpic.utils.distributions import compute_moment_descriptor

    simulation, diagnostic = moment_simulation
    base = simulation.densities[0]
    simulation.densities = [
        attrs.evolve(base, species="He", elec_name="e_He", ion_name="ions_He"),
        attrs.evolve(base, species="H", elec_name="e_H", ion_name="ions_H"),
        attrs.evolve(
            base, species=None, elec_name="e_bare", ion_name="unrecorded_ions"
        ),
        attrs.evolve(base, species="He", elec_name=None, ion_name="ions_He_second"),
        attrs.evolve(base, species="He", elec_name=None, ion_name=None),
    ]
    diagnostic.target = target
    diagnostic.select = select
    directory = simulation._save_directory / "hdf5"
    directory.mkdir(parents=True)
    latest = directory / "data00000010.h5"
    latest.touch()
    particles, weights = moment_particles
    distributions = {
        name: (particles + index, weights * (index + 1))
        for index, name in enumerate(
            ["e_He", "e_H", "e_bare", "ions_He", "ions_H", "ions_He_second"]
        )
    }
    loader = Mock(side_effect=lambda path, name, **kwargs: distributions[name])
    monkeypatch.setattr(module, "load_openpmd_particles", loader)
    diagnostic.analyze()
    assert [call.args for call in loader.call_args_list] == [
        (latest, name) for name in expected_names
    ]
    assert [call.kwargs for call in loader.call_args_list] == [
        {"select": select} for name in expected_names
    ]
    combined_particles = np.concatenate(
        [distributions[name][0] for name in expected_names]
    )
    combined_weights = np.concatenate(
        [distributions[name][1] for name in expected_names]
    )
    assert diagnostic.data == compute_moment_descriptor(
        combined_particles, combined_weights
    )
    assert diagnostic.analysis_complete


@pytest.mark.parametrize(
    "target",
    [
        ("elec_name", "unknown"),
        ("elec_name", ["electrons", "unknown"]),
        ("ion_name", "electrons"),
        ("all_of_species", "He"),
    ],
)
def test_moment_descriptor_rejects_unrecorded_target(moment_simulation, target) -> None:
    _, diagnostic = moment_simulation
    previous = {"saved": 1.0}
    diagnostic.data = previous
    diagnostic.target = target
    with pytest.raises(ValueError, match="recorded"):
        diagnostic.analyze()
    assert diagnostic.data is previous
    assert not diagnostic.analysis_complete


@pytest.mark.parametrize("method", SERIALIZERS)
@pytest.mark.parametrize("select", [None, {}, {"uz": [10.0, None]}])
def test_moment_descriptor_multiple_names_target_and_select_round_trip(
    moment_simulation, tmp_path, method, select
) -> None:
    _, diagnostic = moment_simulation
    diagnostic.target = ("elec_name", ["electrons", "second"])
    diagnostic.select = select
    loaded = _round_trip(diagnostic, method, tmp_path)
    assert isinstance(loaded, MomentDescriptorDiagnostic)
    assert loaded.target == diagnostic.target
    assert isinstance(loaded.target, tuple)
    assert isinstance(loaded.target[1], list)
    assert loaded.select == select
    parameters = diagnostic.to_dict()["parameters"]
    assert "target" in parameters and "select" in parameters
    assert "selection" not in parameters


@pytest.mark.parametrize("select", [None, {}, {"uz": [10.0, None]}])
def test_moment_descriptor_target_and_select_hdf5_round_trip(
    moment_simulation, tmp_path, select
) -> None:
    _, diagnostic = moment_simulation
    diagnostic.target = ("ion_name", ["ions_He", "ions_H"])
    diagnostic.select = select
    path = diagnostic.to_hdf5_file(tmp_path / "diagnostic.h5")
    loaded = SerializableConfig.from_file(path)
    assert isinstance(loaded, MomentDescriptorDiagnostic)
    assert loaded.target == diagnostic.target
    assert isinstance(loaded.target, tuple)
    assert loaded.select == select


@pytest.mark.parametrize("select", [None, {"uz": [10.0, None]}])
def test_moment_descriptor_separate_inputs_preserve_open_bounds(select) -> None:
    diagnostic = MomentDescriptorDiagnostic(
        target=("all_of_species", "e"), select=select
    )
    payload = diagnostic.to_dict(include_nones=False)
    assert ("select" in payload["parameters"]) is (select is not None)
    loaded = SerializableConfig.from_dict(payload)
    assert isinstance(loaded, MomentDescriptorDiagnostic)
    assert loaded.target == diagnostic.target
    assert loaded.select == select
