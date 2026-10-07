"""Canonical campaign configuration and immutable run manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class Parameter:
    """One physical FBPIC input, retained in every campaign record."""

    name: str
    nominal: float | str
    lower: float | None = None
    upper: float | None = None
    unit: str = ""
    log_scale: bool = False
    description: str = ""

    @property
    def variable(self) -> bool:
        return self.lower is not None and self.upper is not None

    def validate(self, value: float | str) -> float | str:
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
    run_root: Path
    parameters: dict[str, Parameter]
    stages: dict[str, StageConfig]
    analysis: dict[str, Any]
    execution: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "CampaignConfig":
        path = Path(path).expanduser().resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        parameters = {
            name: Parameter(
                name=name,
                **{
                    key: float(value)
                    if key in {"nominal", "lower", "upper"}
                    and isinstance(value, str)
                    and value != "auto"
                    else value
                    for key, value in values.items()
                },
            )
            for name, values in data["parameters"].items()
        }
        stages = {
            name: StageConfig(
                active_parameters=tuple(values["active_parameters"]),
                max_evaluations=int(values["max_evaluations"]),
                initial_design_size=int(values["initial_design_size"]),
                gpus_per_simulation=int(values["gpus_per_simulation"]),
                total_gpus=int(values["total_gpus"]),
                concurrent_simulations=int(values["concurrent_simulations"]),
            )
            for name, values in data["stages"].items()
        }
        for stage, settings in stages.items():
            settings.validate(stage, parameters)
        return cls(
            name=data["name"],
            template_script=(path.parent / data["template_script"]).resolve(),
            run_root=(path.parent / data["run_root"]).resolve(),
            parameters=parameters,
            stages=stages,
            analysis=dict(data["analysis"]),
            execution=dict(data["execution"]),
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
            "template_script": str(self.template_script),
            "template_sha256": sha256(self.template_script.read_bytes()).hexdigest(),
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
