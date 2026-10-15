"""Xopt Bayesian exploration over a staged FBPIC campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .campaign import Campaign
from .executor import FBPICRunExecutor
from .store import CampaignStore


class XoptFBPICExplorer:
    """Bridge a canonical campaign dataset to Xopt Bayesian exploration.

    Xopt chooses active variables. The executor expands each proposal to every
    physical parameter, preserving old data when a later stage activates more
    dimensions.
    """

    def __init__(
        self,
        campaign: Campaign,
        stage: str,
    ) -> None:
        self.campaign = campaign
        self.stage = stage
        self.parameters = campaign.config.active_parameters(stage)
        self.feature_names = campaign.config.exploration_features
        self.executor = FBPICRunExecutor(campaign)
        self.store = CampaignStore(campaign.config.run_root)

    @property
    def variable_bounds(self) -> dict[str, list[float]]:
        return {
            parameter.name: [
                self._to_model_value(parameter, float(parameter.lower)),
                self._to_model_value(parameter, float(parameter.upper)),
            ]
            for parameter in self.parameters
            if parameter.lower is not None and parameter.upper is not None
        }

    def initial_design(self, n: int, seed: int = 0) -> list[dict[str, float]]:
        """Physical-unit nominal, anchor, and Sobol points for CLI inspection."""
        nominal = {parameter.name: float(parameter.nominal) for parameter in self.parameters}
        points = [nominal]
        for parameter in self.parameters:
            assert parameter.lower is not None and parameter.upper is not None
            for value in (parameter.lower, parameter.upper):
                point = nominal.copy()
                point[parameter.name] = float(value)
                points.append(point)
        fill = max(0, n - len(points))
        if fill:
            unit = qmc.Sobol(d=len(self.parameters), scramble=True, seed=seed).random(fill)
            low = np.array([self._to_model_value(parameter, float(parameter.lower)) for parameter in self.parameters])
            high = np.array([self._to_model_value(parameter, float(parameter.upper)) for parameter in self.parameters])
            for row in low + unit * (high - low):
                points.append(
                    {
                        parameter.name: self._to_physical_value(parameter, value)
                        for parameter, value in zip(self.parameters, row, strict=True)
                    }
                )
        return points[:n]

    def model_initial_design(self, n: int, seed: int = 0) -> list[dict[str, float]]:
        """Return the initial design transformed into Xopt/GP coordinates."""
        return [self.physical_to_model(point) for point in self.initial_design(n, seed)]

    def physical_to_model(self, point: Mapping[str, float]) -> dict[str, float]:
        """Transform active physical inputs into the coordinates seen by Xopt."""
        return {
            parameter.name: self._to_model_value(parameter, float(point[parameter.name]))
            for parameter in self.parameters
        }

    def model_to_physical(self, point: Mapping[str, float]) -> dict[str, float]:
        """Transform Xopt/GP coordinates back to physical inputs for FBPIC."""
        return {
            parameter.name: self._to_physical_value(parameter, float(point[parameter.name]))
            for parameter in self.parameters
        }

    @staticmethod
    def _to_model_value(parameter, value: float) -> float:
        scaled = value / parameter.linear_scale
        return float(np.log(scaled)) if parameter.log_scale else scaled

    @staticmethod
    def _to_physical_value(parameter, value: float) -> float:
        scaled = float(np.exp(value)) if parameter.log_scale else value
        return scaled * parameter.linear_scale

    def evaluate(self, proposal: Mapping[str, float]) -> dict[str, float | bool | str]:
        """Xopt evaluator function: launch FBPIC and return selected scalar outputs."""
        run_id = f"sim_{self._next_run_number():06d}"
        result = self.executor.evaluate(run_id, self.stage, self.model_to_physical(proposal))
        self.store.append(result)
        outputs: dict[str, float | bool | str] = {
            "status_ok": result.status == "ok",
            "run_id": result.run_id,
        }
        for feature in self.feature_names:
            outputs[feature] = result.features.get(feature, float("nan"))
        return outputs

    def require_fresh_run_root(self) -> None:
        """Require non-continuation stages to start in an empty run root."""
        run_root = self.campaign.config.run_root
        previous_runs = sorted(
            path for path in run_root.glob("sim_*") if path.is_dir()
        )
        if not self.campaign.config.stage_settings(self.stage).continuation and previous_runs:
            raise FileExistsError(
                f"stage {self.stage!r} has continuation disabled, but run root "
                f"{run_root} already contains previous runs: "
                f"{', '.join(path.name for path in previous_runs)}"
            )

    def require_compatible_run_root(self) -> None:
        """Require every available run manifest to belong to this campaign."""
        run_root = self.campaign.config.run_root
        previous_runs = sorted(
            path for path in run_root.glob("sim_*") if path.is_dir()
        )
        expected_fingerprint = self.campaign.config.fingerprint()
        incompatible = []
        for run_directory in previous_runs:
            manifest_path = run_directory / "run_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("campaign_fingerprint") != expected_fingerprint:
                incompatible.append(run_directory.name)
        if incompatible:
            names = ", ".join(incompatible[:5])
            suffix = ", ..." if len(incompatible) > 5 else ""
            raise ValueError(
                f"run root {run_root} contains runs from a different campaign "
                f"(fingerprint != {expected_fingerprint}): {names}{suffix}"
            )

    def make_xopt(self):
        """Create a generator trained on compatible completed campaign rows."""
        try:
            from xopt import Evaluator, VOCS, Xopt
            from xopt.generators.bayesian import BayesianExplorationGenerator
        except ImportError as error:
            raise RuntimeError("install exploration_fbpic with xopt before creating an explorer") from error

        vocs = VOCS(
            variables=self.variable_bounds,
            objectives={feature: "EXPLORE" for feature in self.feature_names},
        )
        generator = BayesianExplorationGenerator(vocs=vocs)
        xopt = Xopt(generator=generator, evaluator=Evaluator(function=self.evaluate), vocs=vocs)
        existing = self.training_data()
        if not existing.empty:
            xopt.add_data(existing)
        return xopt

    def training_data(self) -> pd.DataFrame:
        """Load all successful historical rows with the currently active inputs.

        Rows from older stages are valid because every run stored every input;
        inactive values form the nominal slice of the expanded model.
        """
        data = self.store.load()
        if data.empty:
            return data
        data = data.loc[data["campaign_fingerprint"] == self.campaign.config.fingerprint()]
        data = data.loc[data["status"] == "ok"]
        if data.empty:
            return data
        active_names = {parameter.name for parameter in self.parameters}
        for name, parameter in self.campaign.config.parameters.items():
            if name in active_names:
                continue
            column = f"p:{name}"
            if column not in data:
                return pd.DataFrame()
            if isinstance(parameter.nominal, (int, float)):
                nominal_slice = np.isclose(data[column], float(parameter.nominal))
            else:
                nominal_slice = data[column] == parameter.nominal
            if not nominal_slice.all():
                raise ValueError(
                    f"cannot train {self.stage!r} from records that vary inactive parameter {name!r}"
                )
        required = [f"p:{parameter.name}" for parameter in self.parameters]
        feature_columns = [f"f:{feature}" for feature in self.feature_names]
        available = [column for column in feature_columns if column in data]
        if not available:
            return pd.DataFrame()
        selected = data.loc[:, [*required, *available]].copy()
        selected.columns = [parameter.name for parameter in self.parameters] + [
            column.removeprefix("f:") for column in available
        ]
        for parameter in self.parameters:
            selected[parameter.name] = [
                self._to_model_value(parameter, float(value))
                for value in selected[parameter.name]
            ]
        for feature in self.feature_names:
            if feature not in selected:
                selected[feature] = np.nan
        return selected

    def completed_evaluation_count(self) -> int:
        """Count compatible persisted evaluations that consume this stage's budget."""
        data = self.store.load()
        if data.empty:
            return 0
        compatible = data.loc[
            (data["stage"] == self.stage)
            & (data["campaign_fingerprint"] == self.campaign.config.fingerprint())
        ]
        return len(compatible)

    def _next_run_number(self) -> int:
        numbers = [
            int(path.name.removeprefix("sim_"))
            for path in self.campaign.config.run_root.glob("sim_*")
            if path.is_dir() and path.name.removeprefix("sim_").isdigit()
        ]
        return max(numbers, default=-1) + 1
