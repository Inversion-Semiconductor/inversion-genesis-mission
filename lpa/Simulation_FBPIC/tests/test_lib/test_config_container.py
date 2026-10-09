"""Generic config management, nested round-trips, and relative file references."""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Any

import pytest

from inversion_fbpic.lib.config_container import ConfigContainer
from inversion_fbpic.lib.datapoint import Parameters
from inversion_fbpic.lib.serializable_config import SerializableConfig


def test_empty_containers_have_independent_defaults() -> None:
    first = ConfigContainer()
    second = ConfigContainer()
    assert first.configs == second.configs == []
    assert first.configs is not second.configs
    first.configs.append(Parameters(data={}))
    assert second.configs == []


def test_container_preserves_live_objects_order_and_repeated_references() -> None:
    first = Parameters(data={"energy": 5.0})
    second = Parameters(data={"energy": 7.0})
    sources = [first, second, first]
    container = ConfigContainer(configs=sources)
    assert container.configs is not sources
    assert container.configs[0] is first
    assert container.configs[1] is second
    assert container.configs[2] is first
    first.data["energy"] = 9.0
    serialized = container.to_dict()["parameters"]["configs"]
    assert [entry["parameters"]["data"]["energy"] for entry in serialized] == [9, 7, 9]


def test_container_resolves_all_supported_sources(tmp_path: Path) -> None:
    parameters = Parameters(data={"long_value": "x" * 512})
    json_file = parameters.to_json_file(tmp_path / "parameters.json")
    yaml_file = parameters.to_yaml_file(tmp_path / "parameters.yaml")
    container = ConfigContainer(
        configs=[
            parameters,
            parameters.to_dict(),
            parameters.to_json(),
            parameters.to_yaml(),
            json_file,
            str(yaml_file),
        ]
    )
    assert container.configs[0] is parameters
    assert len(container.configs) == 6
    for config in container.configs:
        assert isinstance(config, Parameters)
        assert config.data == parameters.data
    from_json = container.configs[4]
    from_yaml = container.configs[5]
    assert isinstance(from_json, Parameters) and isinstance(from_yaml, Parameters)
    assert from_json.source_file == json_file
    assert from_yaml.source_file == yaml_file


@pytest.mark.parametrize("format", ["dict", "json", "yaml"])
def test_heterogeneous_nested_container_round_trip(
    format: str, minimal_hyperparameters, minimal_density, minimal_laser
) -> None:
    parameters = Parameters(data={"energy": 5.0})
    nested = ConfigContainer(configs=[parameters, parameters])
    original = ConfigContainer(
        configs=[minimal_hyperparameters, minimal_density, minimal_laser, nested]
    )
    assert original.configs[3] is nested
    if format == "dict":
        loaded = SerializableConfig.from_dict(original.to_dict())
    elif format == "json":
        loaded = SerializableConfig.from_json(original.to_json())
    else:
        loaded = SerializableConfig.from_yaml(original.to_yaml())
    assert isinstance(loaded, ConfigContainer)
    assert [type(entry) for entry in loaded.configs] == [
        type(entry) for entry in original.configs
    ]
    assert loaded.to_dict() == original.to_dict()
    loaded_nested = loaded.configs[3]
    assert isinstance(loaded_nested, ConfigContainer)
    assert len(loaded_nested.configs) == 2
    assert loaded_nested.configs[0] is not loaded_nested.configs[1]


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_container_files_embed_loaded_configs_and_record_live_mutations(
    extension: str, tmp_path: Path
) -> None:
    data = {"energy": 5.0}
    original = Parameters(data=data)
    child_file = original.to_yaml_file(tmp_path / "child.yaml")
    container = ConfigContainer(configs=[original, child_file, original])
    data["energy"] = 7.0
    writer = container.to_json_file if extension == "json" else container.to_yaml_file
    saved = writer(tmp_path / "output" / f"container.{extension}")
    child_file.unlink()
    loaded = SerializableConfig.from_file(saved)
    assert isinstance(loaded, ConfigContainer)
    assert loaded.source_file == saved
    entries = loaded.configs
    assert all(isinstance(entry, Parameters) for entry in entries)
    assert [
        entry.data["energy"] for entry in entries if isinstance(entry, Parameters)
    ] == [7, 5, 7]
    assert entries[0] is not entries[2]


@pytest.mark.parametrize("configs", [None, {}, "child.yaml", (1,), 3])
def test_container_rejects_non_list_inputs(configs: Any) -> None:
    with pytest.raises(TypeError, match="configs"):
        ConfigContainer(configs=configs)


@pytest.mark.parametrize("entry", [None, 1, [], object()])
def test_container_rejects_unsupported_members_with_index(entry: Any) -> None:
    with pytest.raises(TypeError, match=r"configs\[1\]"):
        ConfigContainer(configs=[Parameters(data={}), entry])


@pytest.mark.parametrize(
    "entry",
    [
        {"config_type": "unknown", "subclass": "unknown", "parameters": {}},
        {"config_type": "datapoint", "subclass": "unknown", "parameters": {}},
        "not a config",
        Path("does_not_exist.yaml"),
    ],
)
def test_container_reports_invalid_source_index(entry) -> None:
    with pytest.raises(ValueError, match=r"configs\[0\]"):
        ConfigContainer(configs=[entry])


def test_container_does_not_scan_directories(tmp_path: Path) -> None:
    Parameters(data={}).to_yaml_file(tmp_path / "child.yaml")
    with pytest.raises(ValueError, match=r"configs\[0\]"):
        ConfigContainer(configs=[tmp_path])


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_nested_file_references_resolve_against_each_containing_file(
    extension: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "project" / "cfg"
    cfg.mkdir(parents=True)
    child = Parameters(data={"label": "child"}).to_yaml_file(
        cfg / "nested" / "child.yaml"
    )
    root_child = Parameters(data={"label": "root"}).to_yaml_file(cfg / "child.yaml")
    nested = cfg / "nested" / "container.yaml"
    nested.write_text(
        "config_type: config_container\nsubclass: config_container\n"
        "parameters:\n  configs: [child.yaml]\n",
        encoding="utf-8",
    )
    outer = cfg / f"outer.{extension}"
    payload = {
        "config_type": "config_container",
        "subclass": "config_container",
        "parameters": {"configs": ["nested/container.yaml", "child.yaml"]},
    }
    if extension == "json":
        import json

        text = json.dumps(payload)
    else:
        import yaml

        text = yaml.safe_dump(payload)
    outer.write_text(text, encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = SerializableConfig.from_file(outer)
    assert isinstance(loaded, ConfigContainer)
    inner = loaded.configs[0]
    root = loaded.configs[1]
    assert isinstance(inner, ConfigContainer) and isinstance(root, Parameters)
    assert inner.source_file == nested.resolve()
    inner_child = inner.configs[0]
    assert isinstance(inner_child, Parameters)
    assert inner_child.source_file == child
    assert inner_child.data == {"label": "child"}
    assert root.source_file == root_child
    assert root.data == {"label": "root"}


def test_container_uses_programmatic_path_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "cfg"
    child = Parameters(data={"energy": 5.0}).to_json_file(cfg / "child.json")
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    with SerializableConfig.resolving_paths_relative_to(cfg):
        container = ConfigContainer(configs=["child.json", Path("child.json")])
    assert all(isinstance(entry, Parameters) for entry in container.configs)
    assert all(
        entry.source_file == child
        for entry in container.configs
        if isinstance(entry, Parameters)
    )
    assert container.configs[0] is not container.configs[1]


def test_explicit_anchor_yields_to_nested_source_and_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "cfg"
    root_child = Parameters(data={"label": "root"}).to_yaml_file(cfg / "child.yaml")
    nested_child = Parameters(data={"label": "nested"}).to_yaml_file(
        cfg / "nested" / "child.yaml"
    )
    nested = cfg / "nested" / "container.yaml"
    nested.write_text(
        "config_type: config_container\nsubclass: config_container\n"
        "parameters:\n  configs: [child.yaml]\n",
        encoding="utf-8",
    )
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    with SerializableConfig.resolving_paths_relative_to(cfg):
        container = ConfigContainer(configs=["nested/container.yaml", "child.yaml"])
    inner, outer_child = container.configs
    assert isinstance(inner, ConfigContainer) and isinstance(outer_child, Parameters)
    inner_child = inner.configs[0]
    assert isinstance(inner_child, Parameters)
    assert inner_child.data == {"label": "nested"}
    assert inner_child.source_file == nested_child
    assert outer_child.source_file == root_child

    # No same-named root file should be required for a nested relative reference.
    root_child.unlink()
    with SerializableConfig.resolving_paths_relative_to(cfg):
        loaded = ConfigContainer.from_file("nested/container.yaml")
    assert isinstance(loaded, ConfigContainer)
    assert isinstance(loaded.configs[0], Parameters)
    assert loaded.configs[0].data == {"label": "nested"}


@pytest.mark.parametrize("include_nones", [True, False])
def test_container_empty_collection_fallback_and_examples(include_nones: bool) -> None:
    payload = ConfigContainer().to_dict(include_nones=include_nones)
    assert ("configs" in payload["parameters"]) is include_nones
    loaded = ConfigContainer.from_dict(payload)
    assert isinstance(loaded, ConfigContainer)
    assert loaded.configs == []
    assert ConfigContainer.example_dict()["parameters"] == {"configs": []}


def test_container_yaml_preserves_nested_comments() -> None:
    source = (
        "# Curated run configs.\n"
        "config_type: config_container\n"
        "subclass: config_container\n"
        "parameters:\n"
        "  configs:\n"
        "    - config_type: datapoint\n"
        "      subclass: parameters\n"
        "      parameters:\n"
        "        data: {energy: 5.0} # measured energy\n"
    )
    loaded = ConfigContainer.from_yaml(source)
    assert isinstance(loaded, ConfigContainer)
    dumped = loaded.to_yaml(round_trip_comments=True)
    assert "Curated run configs." in dumped
    assert "measured energy" in dumped
    assert ConfigContainer.from_yaml(dumped).to_dict() == loaded.to_dict()
    documented = loaded.to_yaml(round_trip_comments=False)
    assert "Manage an ordered collection" in documented
    assert "Store a curated set" in documented


def test_from_any_does_not_mask_unrelated_filesystem_errors(monkeypatch) -> None:
    def denied(cls, path, **kwargs):
        raise PermissionError(errno.EACCES, "access denied")

    monkeypatch.setattr(SerializableConfig, "from_file", classmethod(denied))
    with pytest.raises(PermissionError, match="access denied"):
        SerializableConfig.from_any(Parameters(data={}).to_json())
