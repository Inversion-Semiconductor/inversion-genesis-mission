"""Xopt Bayesian exploration over a staged FBPIC campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

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
        exploration_features: Iterable[str],
    ) -> None:
        self.campaign = campaign
        self.stage = stage
        self.parameters = campaign.config.active_parameters(stage)
        self.feature_names = tuple(exploration_features)
        if not self.feature_names:
            raise ValueError("at least one scalar feature must be explored")
        self.executor = FBPICRunExecutor(campaign)
        self.store = CampaignStore(campaign.config.run_root)

    @property
    def variable_bounds(self) -> dict[str, list[float]]:
        return {
            parameter.name: [float(parameter.lower), float(parameter.upper)]
            for parameter in self.parameters
            if parameter.lower is not None and parameter.upper is not None
        }

    def initial_design(self, n: int, seed: int = 0) -> list[dict[str, float]]:
        """Nominal + active axis bounds + Sobol fill for a transparent first batch."""
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
            low = np.array([self.variable_bounds[name][0] for name in self.variable_bounds])
            high = np.array([self.variable_bounds[name][1] for name in self.variable_bounds])
            for row in low + unit * (high - low):
                points.append(dict(zip(self.variable_bounds, row, strict=True)))
        return points[:n]

    def evaluate(self, proposal: Mapping[str, float]) -> dict[str, float | bool | str]:
        """Xopt evaluator function: launch FBPIC and return selected scalar outputs."""
        run_id = f"sim_{self._next_run_number():06d}"
        result = self.executor.evaluate(run_id, self.stage, proposal)
        self.store.append(result)
        outputs: dict[str, float | bool | str] = {
            "status_ok": result.status == "ok",
            "run_id": result.run_id,
        }
        for feature in self.feature_names:
            outputs[feature] = result.features.get(feature, float("nan"))
        return outputs

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
        required = [f"p:{parameter.name}" for parameter in self.parameters]
        feature_columns = [f"f:{feature}" for feature in self.feature_names]
        available = [column for column in feature_columns if column in data]
        if not available:
            return pd.DataFrame()
        selected = data.loc[data["status"] == "ok", [*required, *available]].copy()
        selected.columns = [parameter.name for parameter in self.parameters] + [
            column.removeprefix("f:") for column in available
        ]
        for feature in self.feature_names:
            if feature not in selected:
                selected[feature] = np.nan
        return selected

    def _next_run_number(self) -> int:
        data = self.store.load()
        if data.empty:
            return 0
        numbers = pd.to_numeric(data["run_id"].str.extract(r"(\d+)$")[0], errors="coerce")
        return int(numbers.max()) + 1
