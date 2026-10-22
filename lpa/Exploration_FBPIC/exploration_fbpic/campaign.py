"""Canonical campaign configuration and immutable run manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from . import __version__ as _PACKAGE_VERSION


_NUMERIC_STRING = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _normalize_numeric_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_numeric_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_numeric_strings(item) for item in value]
    if isinstance(value, str) and _NUMERIC_STRING.fullmatch(value):
        return float(value)
    return value


def _resolve_resolution_profile(execution: dict[str, Any], base: Path) -> dict[str, Any]:
    """Merge the selected resolution profile into the execution hyperparameters."""
    selection = execution.get("resolution_profile")
    if selection is None:
        return execution
    profile_path = base / selection["file"]
    profiles = yaml.safe_load(profile_path.read_text(encoding="utf-8"))["profiles"]
    name = selection["name"]
    if name not in profiles:
        raise KeyError(f"unknown resolution profile {name!r}; choose from {sorted(profiles)}")
    hyperparameters = dict(execution.get("hyperparameters") or {})
    overlap = set(profiles[name]) & set(hyperparameters)
    if overlap:
        raise ValueError(
            f"hyperparameters {sorted(overlap)} are set by resolution profile {name!r}"
        )
    return {**execution, "hyperparameters": {**profiles[name], **hyperparameters}}


def _exploration_package_version() -> str:
    """Return installed package metadata, with an in-tree source fallback."""
    try:
        return version("exploration_fbpic")
    except PackageNotFoundError:
        return _PACKAGE_VERSION


@dataclass(frozen=True)
class Parameter:
    """One physical FBPIC input, retained in every campaign record."""

    name: str
    nominal: float | str
    lower: float | None = None
    upper: float | None = None
    unit: str = ""
    linear_scale: float = 1.0
    log_scale: bool = False
    description: str = ""

    @property
    def variable(self) -> bool:
        return self.lower is not None and self.upper is not None

    def validate(self, value: float | str) -> float | str:
        if self.linear_scale <= 0:
            raise ValueError(f"{self.name}.linear_scale must be positive")
        if not self.variable:
            if value != self.nominal:
                raise ValueError(f"{self.name} is fixed at {self.nominal!r}")
            return value
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be numeric")
        numeric_value = float(value)
        assert self.lower is not None and self.upper is not None
        if not self.lower <= numeric_value <= self.upper:
            raise ValueError(
                f"{self.name}={numeric_value:g} is outside [{self.lower:g}, {self.upper:g}]"
            )
        return numeric_value


@dataclass(frozen=True)
class StageConfig:
    """Active dimensions and resource budget for one campaign stage."""

    active_parameters: tuple[str, ...]
    max_evaluations: int
    initial_design_size: int
    gpus_per_simulation: int
    total_gpus: int
    concurrent_simulations: int
    continuation: bool = False

    def validate(self, name: str, parameters: Mapping[str, Parameter]) -> None:
        unknown = set(self.active_parameters) - set(parameters)
        if unknown:
            raise ValueError(f"stage {name}: unknown parameters {sorted(unknown)}")
        nonnumeric = [parameter for parameter in self.active_parameters if not parameters[parameter].variable]
        if nonnumeric:
            raise ValueError(f"stage {name}: fixed parameters cannot be active: {nonnumeric}")
        if self.initial_design_size < 0:
            raise ValueError(f"stage {name}: initial_design_size cannot be negative")
        if self.max_evaluations < self.initial_design_size:
            raise ValueError(f"stage {name}: max_evaluations must cover the initial design")
        if min(self.max_evaluations, self.gpus_per_simulation, self.total_gpus, self.concurrent_simulations) < 1:
            raise ValueError(f"stage {name}: resource settings must be positive")
        if self.total_gpus % self.gpus_per_simulation:
            raise ValueError(f"stage {name}: total_gpus must be divisible by gpus_per_simulation")
        if self.concurrent_simulations > self.total_gpus // self.gpus_per_simulation:
            raise ValueError(f"stage {name}: concurrent_simulations exceeds GPU capacity")


@dataclass(frozen=True)
class CampaignConfig:
    """Configuration shared by all stages of one physically comparable campaign."""

    name: str
    template_script: Path
    template_class: str
    run_root: Path
    parameters: dict[str, Parameter]
    stages: dict[str, StageConfig]
    analysis: dict[str, Any]
    execution: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "CampaignConfig":
        path = Path(path).expanduser().resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        parameters = {}
        for name, raw_values in data["parameters"].items():
            values = dict(raw_values)
            scale = float(values.get("linear_scale", 1.0))
            if scale <= 0:
                raise ValueError(f"{name}.linear_scale must be positive")
            for key in {"nominal", "lower", "upper"}:
                value = values.get(key)
                if isinstance(value, str) and value != "auto":
                    value = float(value)
                if isinstance(value, (int, float)):
                    values[key] = float(value) * scale
            values["linear_scale"] = scale
            parameters[name] = Parameter(name=name, **values)
        for parameter in parameters.values():
            parameter.validate(parameter.nominal)
        stages = {
            name: StageConfig(
                active_parameters=tuple(values["active_parameters"]),
                max_evaluations=int(values["max_evaluations"]),
                initial_design_size=int(values["initial_design_size"]),
                gpus_per_simulation=int(values["gpus_per_simulation"]),
                total_gpus=int(values["total_gpus"]),
                concurrent_simulations=int(values["concurrent_simulations"]),
                continuation=bool(values.get("continuation", False)),
            )
            for name, values in data["stages"].items()
        }
        for stage, settings in stages.items():
            settings.validate(stage, parameters)
        run_root = Path(os.path.expandvars(str(data["run_root"]))).expanduser()
        if not run_root.is_absolute():
            run_root = path.parent / run_root
        return cls(
            name=data["name"],
            template_script=(path.parent / data["template_script"]).resolve(),
            template_class=str(data.get("template_class", "IonizationInjectionSimulation")),
            run_root=run_root.resolve(),
            parameters=parameters,
            stages=stages,
            analysis=dict(data["analysis"]),
            execution=_normalize_numeric_strings(
                _resolve_resolution_profile(dict(data["execution"]), path.parent)
            ),
        )

    def active_parameters(self, stage: str) -> tuple[Parameter, ...]:
        try:
            names = self.stages[stage].active_parameters
        except KeyError as error:
            raise KeyError(f"unknown stage {stage!r}; choose from {sorted(self.stages)}") from error
        return tuple(self.parameters[name] for name in names)

    def stage_settings(self, stage: str) -> StageConfig:
        try:
            return self.stages[stage]
        except KeyError as error:
            raise KeyError(f"unknown stage {stage!r}; choose from {sorted(self.stages)}") from error

    @property
    def exploration_features(self) -> tuple[str, ...]:
        """Scalar outputs selected by this campaign for exploration."""
        features = self.analysis.get("exploration_features", ())
        if not isinstance(features, list) or not all(isinstance(feature, str) for feature in features):
            raise ValueError("analysis.exploration_features must be a list of scalar feature names")
        if not features:
            raise ValueError("analysis.exploration_features must select at least one scalar feature")
        return tuple(features)

    def complete_parameters(
        self, stage: str, proposed: Mapping[str, float] | None = None
    ) -> dict[str, float | str]:
        """Expand a stage proposal into the full, permanent physical input vector."""
        proposed = proposed or {}
        active_names = set(self.stage_settings(stage).active_parameters)
        unknown = set(proposed) - active_names
        if unknown:
            raise ValueError(f"proposal changes inactive or unknown parameters: {sorted(unknown)}")
        resolved: dict[str, float | str] = {}
        for name, parameter in self.parameters.items():
            value = proposed.get(name, parameter.nominal)
            resolved[name] = parameter.validate(value)
        return resolved

    def fingerprint(self) -> str:
        """Hash settings that must remain fixed to compare all campaign outputs."""
        payload = {
            "exploration_fbpic_version": _exploration_package_version(),
            "template_sha256": sha256(self.template_script.read_bytes()).hexdigest(),
            "template_class": self.template_class,
            "analysis": self.analysis,
            "execution": self.execution,
            "parameters": {name: asdict(parameter) for name, parameter in self.parameters.items()},
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class RunManifest:
    """The persistent record required to reuse an evaluation in a later stage."""

    run_id: str
    stage: str
    campaign_fingerprint: str
    parameters: dict[str, float | str]
    template_script: str
    stage_settings: StageConfig

    def write(self, run_directory: str | Path) -> Path:
        path = Path(run_directory) / "run_manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


class Campaign:
    """Creates complete parameter vectors and manifests for one staged campaign."""

    def __init__(self, config: CampaignConfig):
        self.config = config

    @classmethod
    def from_file(cls, path: str | Path) -> "Campaign":
        return cls(CampaignConfig.from_file(path))

    def make_manifest(
        self, run_id: str, stage: str, proposed: Mapping[str, float] | None = None
    ) -> RunManifest:
        return RunManifest(
            run_id=run_id,
            stage=stage,
            campaign_fingerprint=self.config.fingerprint(),
            parameters=self.config.complete_parameters(stage, proposed),
            template_script=str(self.config.template_script),
            stage_settings=self.config.stage_settings(stage),
        )
