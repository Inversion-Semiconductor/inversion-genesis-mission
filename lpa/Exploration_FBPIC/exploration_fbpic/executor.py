"""Isolated FBPIC run execution and scalar diagnostic extraction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np
from scipy.constants import c, e, m_e

from inversion_fbpic.utils.distributions import (
    MOMENTS,
    OFF,
    SPLINE,
    compute_moment_descriptor,
    crop_central_particles,
    load_openpmd_particles,
    select_by_uz,
)

from .campaign import Campaign, RunManifest


def mean_kinetic_energy_mev(particles: np.ndarray, weights: np.ndarray) -> float:
    """Return the charge-weighted mean electron kinetic energy in MeV."""
    gamma = np.sqrt(1.0 + np.sum(particles[:, (1, 3, 5)] ** 2, axis=1))
    kinetic_energy_mev = (gamma - 1.0) * m_e * c**2 / e / 1.0e6
    return float(np.average(kinetic_energy_mev, weights=weights))


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
            result = EvaluationResult(run_id, "timeout", {}, run_directory, str(error), time.perf_counter() - started)
        except subprocess.CalledProcessError as error:
            result = EvaluationResult(run_id, "fbpic_error", {}, run_directory, str(error), time.perf_counter() - started)
        except (FileNotFoundError, ValueError, OSError) as error:
            result = EvaluationResult(run_id, "analysis_error", {}, run_directory, str(error), time.perf_counter() - started)
        else:
            result = EvaluationResult(run_id, "ok", features, run_directory, wall_s=time.perf_counter() - started)
        self._write_result(run_directory, manifest, result)
        return result

    def _write_entrypoint(self, run_directory: Path, manifest: RunManifest) -> None:
        hyperparameters = dict(self.config.execution["hyperparameters"])
        hyperparameters["lasy_file"] = str(run_directory / "experimental_laser")
        hyperparameters["lab_diagnostic_directory"] = str(run_directory / "lab_diags")
        contents = f'''"""Generated FBPIC entry point. Do not edit; see run_manifest.json."""
from pathlib import Path
import sys

TEMPLATE_DIRECTORY = Path({str(self.config.template_script.parent)!r})
sys.path.insert(0, str(TEMPLATE_DIRECTORY))
from ionization_injection_template import IonizationInjectionSimulation

PHYSICAL_PARAMETERS = {manifest.parameters!r}
HYPERPARAMETERS = {hyperparameters!r}

if __name__ == "__main__":
    IonizationInjectionSimulation(PHYSICAL_PARAMETERS, HYPERPARAMETERS).run()
'''
        (run_directory / "run_fbpic.py").write_text(contents, encoding="utf-8")

    def _execute(self, run_directory: Path) -> None:
        command = [*self.config.execution.get("launcher", ["python"]), "run_fbpic.py"]
        timeout_s = float(self.config.execution.get("timeout_s", 21600))
        with (run_directory / "stdout.log").open("w", encoding="utf-8") as stdout, (
            run_directory / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            subprocess.run(
                command,
                cwd=run_directory,
                check=True,
                timeout=timeout_s,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )

    def extract_features(self, run_directory: Path) -> dict[str, float]:
        """Extract the complete scalar schema from the final particle diagnostic."""
        analysis = self.config.analysis
        diagnostics = sorted((run_directory / "lab_diags" / "hdf5").glob("data*.h5"))
        if not diagnostics:
            raise FileNotFoundError("no particle diagnostic produced by FBPIC")
        particles, weights = load_openpmd_particles(diagnostics[-1], analysis["species"])
        particles, weights = select_by_uz(particles, weights, uz_min=float(analysis["uz_min"]))
        particles, weights = crop_central_particles(
            particles, weights, central_fraction=float(analysis["central_fraction"])
        )
        features = compute_moment_descriptor(
            particles,
            weights,
            longitudinal_mode=OFF,
            longitudinal_bins=int(analysis["longitudinal_bins"]),
            include_higher_moments=True,
        )
        features.update(
            compute_moment_descriptor(
                particles,
                weights,
                longitudinal_mode=MOMENTS,
                longitudinal_bins=int(analysis["longitudinal_bins"]),
            )
        )
        features.update(
            compute_moment_descriptor(
                particles,
                weights,
                longitudinal_mode=SPLINE,
                longitudinal_bins=int(analysis["longitudinal_bins"]),
            )
        )
        features["mean_kinetic_energy_MeV"] = mean_kinetic_energy_mev(particles, weights)
        return features

    @staticmethod
    def _write_result(run_directory: Path, manifest: RunManifest, result: EvaluationResult) -> None:
        record = {
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest": manifest.__dict__,
            "result": result.as_dict(),
        }
        (run_directory / "result.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
