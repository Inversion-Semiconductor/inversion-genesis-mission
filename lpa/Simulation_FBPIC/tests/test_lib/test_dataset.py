"""Dictionary-backed inputs and the abstract diagnostics analysis lifecycle."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, ClassVar

import attrs
import numpy as np
import pytest

from inversion_fbpic.lib.config_container import ConfigContainer
from inversion_fbpic.lib.datapoint import _Datapoint, Parameters
from inversion_fbpic.lib.diagnostics import _Diagnostic
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


def test_parameters_retain_live_dictionary() -> None:
    from inversion_fbpic.lib import serializable_config as module

    data = {"energy": 5.0, "nested": {"selection": [10.0, None]}}
    parameters = Parameters(data=data)
    assert parameters.data is data
    data["energy"] = 6.0
    assert parameters.data["energy"] == 6.0
    parameters.data["new"] = "value"
    assert data["new"] == "value"
    assert parameters.to_dict() == {
        "config_type": "datapoint",
        "subclass": "parameters",
        "git_hash": module._git_hash(),
        "parameters": {"data": data},
    }


def test_parameters_require_dictionary_at_construction() -> None:
    with pytest.raises(TypeError, match="data"):
        Parameters()  # type: ignore[call-arg]


@pytest.mark.parametrize("data", [None, [], "values", 1])
def test_parameters_reject_non_dictionary(data: Any) -> None:
    with pytest.raises(TypeError, match="data"):
        Parameters(data=data)


@pytest.mark.parametrize("method", SERIALIZERS)
def test_parameters_round_trip(method: str, tmp_path: Path) -> None:
    original = Parameters(
        data={"energy": 5.0, "nested": {"selection": [10.0, None]}, "label": "run"}
    )
    loaded = _round_trip(original, method, tmp_path)
    assert type(loaded) is Parameters
    assert loaded.data == original.data


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


@pytest.mark.parametrize("include_nones", [True, False])
def test_empty_parameters_load_with_collection_fallback(include_nones: bool) -> None:
    payload = Parameters(data={}).to_dict(include_nones=include_nones)
    assert ("data" in payload["parameters"]) is include_nones
    first = Parameters.from_dict(payload)
    second = Parameters.from_dict(payload)
    assert isinstance(first, Parameters) and isinstance(second, Parameters)
    assert first.data == second.data == {}
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
