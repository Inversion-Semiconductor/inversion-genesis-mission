"""Beam-existence classification used to gate conditional feature exploration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .exploration import XoptFBPICExplorer


NO_BEAM_MESSAGE = "No scalar diagnostic results found"


def is_no_beam(status: str, message: str = "") -> bool:
    """Identify a completed physical no-injection outcome, including legacy rows."""
    return status == "no_beam" or (
        status in {"error", "analysis_error"} and NO_BEAM_MESSAGE in message
    )


@dataclass
class BeamFeasibilityModel:
    """Probabilistic beam-existence model in active Xopt coordinates.

    The feature GPs stay conditional on beam-present records. This lightweight
    ensemble supplies a stable probability gate for candidate selection and can
    be retrained after each completed resource-set batch.
    """

    explorer: "XoptFBPICExplorer"
    _inputs: list[np.ndarray] = field(default_factory=list)
    _labels: list[int] = field(default_factory=list)
    _classifier: object | None = None

    @property
    def is_ready(self) -> bool:
        return self._classifier is not None

    def fit_campaign_data(self, data: pd.DataFrame) -> None:
        """Load physical beam/no-beam outcomes from the durable campaign table."""
        if data.empty:
            return
        compatible = data.loc[
            data["campaign_fingerprint"] == self.explorer.campaign.config.fingerprint()
        ]
        parameter_names = list(self.explorer.variable_bounds)
        required = [f"p:{name}" for name in parameter_names]
        if any(column not in compatible for column in required):
            return
        for _, row in compatible.iterrows():
            status = str(row["status"])
            message = str(row.get("message", ""))
            if status == "ok":
                label = 1
            elif is_no_beam(status, message):
                label = 0
            else:
                continue
            model_point = np.asarray(
                [
                    self.explorer._to_model_value(
                        self.explorer.campaign.config.parameters[name],
                        float(row[f"p:{name}"]),
                    )
                    for name in parameter_names
                ],
                dtype=float,
            )
            self._inputs.append(model_point)
            self._labels.append(label)
        self._fit()

    def add_completed(self, points: pd.DataFrame, outcome_codes: np.ndarray) -> None:
        """Add model-coordinate outcomes from the live libEnsemble batch."""
        parameter_names = list(self.explorer.variable_bounds)
        for index, outcome in enumerate(outcome_codes):
            if int(outcome) not in {0, 1}:
                continue
            self._inputs.append(
                points.iloc[index].loc[parameter_names].to_numpy(dtype=float)
            )
            self._labels.append(1 if int(outcome) == 0 else 0)
        self._fit()

    def beam_probability(self, points: list[Mapping[str, float]]) -> np.ndarray:
        """Return $P(beam | x)$ for Xopt-coordinate candidate mappings."""
        if not self.is_ready:
            return np.ones(len(points))
        matrix = np.asarray(
            [
                [point[name] for name in self.explorer.variable_bounds]
                for point in points
            ],
            dtype=float,
        )
        return self._classifier.predict_proba(matrix)[:, 1]  # type: ignore[union-attr]

    def beam_probability_matrix(self, points: np.ndarray) -> np.ndarray:
        """Return beam probability for a matrix in active Xopt coordinates."""
        if not self.is_ready:
            return np.ones(len(points))
        return self._classifier.predict_proba(points)[:, 1]  # type: ignore[union-attr]

    def _fit(self) -> None:
        if len(set(self._labels)) < 2:
            self._classifier = None
            return
        from sklearn.ensemble import RandomForestClassifier

        self._classifier = RandomForestClassifier(
            n_estimators=256,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=0,
        ).fit(np.asarray(self._inputs), np.asarray(self._labels))
