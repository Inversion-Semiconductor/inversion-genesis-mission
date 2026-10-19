"""Isolated FBPIC run execution and scalar diagnostic extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

from inversion_fbpic.lib.config_container import ConfigContainer
from inversion_fbpic.lib.diagnostics import _Diagnostic
from inversion_fbpic.utils.distributions import (
    mean_kinetic_energy_mev as mean_kinetic_energy_mev,
)

from .campaign import Campaign, RunManifest


@dataclass(frozen=True)
class EvaluationResult:
    """One complete FBPIC evaluation, including failed simulations."""

    run_id: str
    status: str
    features: dict[str, float]
    run_directory: Path
    message: str = ""
    wall_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "features": self.features,
            "run_directory": str(self.run_directory),
            "message": self.message,
            "wall_s": self.wall_s,
        }


class FBPICRunExecutor:
    """Create, execute, and extract one independently reproducible FBPIC run."""

    def __init__(self, campaign: Campaign):
        self.campaign = campaign
        self.config = campaign.config

    def prepare_run(
        self, run_id: str, stage: str, proposed: Mapping[str, float] | None = None
    ) -> tuple[Path, RunManifest]:
        run_directory = self.config.run_root / run_id
        if run_directory.exists():
            raise FileExistsError(f"run directory already exists: {run_directory}")
        run_directory.mkdir(parents=True)
        manifest = self.campaign.make_manifest(run_id, stage, proposed)
        manifest.write(run_directory)
        self._write_entrypoint(run_directory, manifest)
        return run_directory, manifest

    def evaluate(
        self, run_id: str, stage: str, proposed: Mapping[str, float] | None = None
    ) -> EvaluationResult:
        run_directory, manifest = self.prepare_run(run_id, stage, proposed)
        started = time.perf_counter()
        try:
            self._execute(run_directory)
            features = self.extract_features(run_directory)
        except subprocess.TimeoutExpired as error:
            result = EvaluationResult(
                run_id,
                "timeout",
                {},
                run_directory,
                str(error),
                time.perf_counter() - started,
            )
        except subprocess.CalledProcessError as error:
            result = EvaluationResult(
                run_id,
                "fbpic_error",
                {},
                run_directory,
                str(error),
                time.perf_counter() - started,
            )
        except (FileNotFoundError, ValueError, OSError) as error:
            result = EvaluationResult(
                run_id,
                "analysis_error",
                {},
                run_directory,
                str(error),
                time.perf_counter() - started,
            )
        except Exception as error:
            result = EvaluationResult(
                run_id,
                "error",
                {},
                run_directory,
                repr(error),
                time.perf_counter() - started,
            )
        else:
            result = EvaluationResult(
                run_id,
                "ok",
                features,
                run_directory,
                wall_s=time.perf_counter() - started,
            )
        self.finalize(run_directory, manifest, result)
        return result

    def _write_entrypoint(self, run_directory: Path, manifest: RunManifest) -> None:
        hyperparameters = self.config.template_hyperparameters()
        hyperparameters["lasy_file"] = str(run_directory / "experimental_laser")
        hyperparameters["lab_diagnostic_directory"] = str(run_directory / "lab_diags")
        contents = f'''"""Generated FBPIC entry point. Do not edit; see run_manifest.json."""
from pathlib import Path
import sys

TEMPLATE_DIRECTORY = Path({str(self.config.template_script.parent)!r})
sys.path.insert(0, str(TEMPLATE_DIRECTORY))
from {self.config.template_script.stem} import {self.config.template_class}

PHYSICAL_PARAMETERS = {manifest.parameters!r}
HYPERPARAMETERS = {hyperparameters!r}

if __name__ == "__main__":
    {self.config.template_class}(PHYSICAL_PARAMETERS, HYPERPARAMETERS).run()
'''
        (run_directory / "run_fbpic.py").write_text(contents, encoding="utf-8")

    def _execute(self, run_directory: Path) -> None:
        command = [*self.config.execution.get("launcher", ["python"]), "run_fbpic.py"]
        with (
            (run_directory / "stdout.log").open("w", encoding="utf-8") as stdout,
            (run_directory / "stderr.log").open("w", encoding="utf-8") as stderr,
        ):
            subprocess.run(
                command,
                cwd=run_directory,
                check=True,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )

    def extract_features(self, run_directory: Path) -> dict[str, float]:
        """Read scalar diagnostic results saved by the template, without analyzing.

        Load the sole ``datapoint`` config in the configured results directory
        (``results`` by default). HDF5 datapoints use the template's ``/results``
        group. Parameter metadata is ignored, and nested containers are supported.
        """
        hyperparameters = self.config.execution.get("hyperparameters", {})
        results_directory = Path(hyperparameters.get("results_directory", "results"))
        if not results_directory.is_absolute():
            results_directory = run_directory / results_directory
        suffixes = {".json", ".jsn", ".yaml", ".yml", ".h5", ".hdf5"}
        paths = sorted(
            path
            for path in results_directory.glob("datapoint.*")
            if path.is_file() and path.suffix.lower() in suffixes
        )
        if not paths:
            raise FileNotFoundError(
                f"No saved datapoint config found in {results_directory}"
            )
        if len(paths) != 1:
            raise ValueError(
                f"Multiple saved datapoint configs found in {results_directory}"
            )
        path = paths[0]
        if path.suffix.lower() in {".h5", ".hdf5"}:
            datapoint = ConfigContainer.from_hdf5_file(path, group_path="/results")
        else:
            datapoint = ConfigContainer.from_file(path)
        if not isinstance(datapoint, ConfigContainer):
            raise ValueError(f"Saved datapoint {path} must be a ConfigContainer")

        features: dict[str, float] = {}

        def collect(container: ConfigContainer) -> None:
            for config in container.configs:
                if isinstance(config, ConfigContainer):
                    collect(config)
                elif isinstance(config, _Diagnostic):
                    for name, value in config.data.items():
                        if name in features:
                            raise ValueError(
                                f"Duplicate saved diagnostic feature: {name}"
                            )
                        if not isinstance(value, Real) or isinstance(value, bool):
                            raise ValueError(
                                f"Saved diagnostic feature {name} must be numeric"
                            )
                        scalar = float(value)
                        if not math.isfinite(scalar):
                            raise ValueError(
                                f"Saved diagnostic feature {name} must be finite"
                            )
                        features[name] = scalar

        collect(datapoint)
        if not features:
            raise ValueError(f"No scalar diagnostic results found in {path}")
        missing = sorted(set(self.config.exploration_features) - features.keys())
        if missing:
            raise ValueError(f"Missing saved exploration features: {missing}")
        return features

    @staticmethod
    def finalize(
        run_directory: Path, manifest: RunManifest, result: EvaluationResult
    ) -> None:
        record = {
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest": asdict(manifest),
            "result": result.as_dict(),
        }
        (run_directory / "result.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
