"""Beam-existence classification used to gate conditional feature exploration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .campaign_rows import model_input_matrix, physical_outcome_rows

if TYPE_CHECKING:
    from .exploration import XoptFBPICExplorer


@dataclass
class BeamFeasibilityModel:
    """Probabilistic beam-existence model in active Xopt coordinates.

    The feature GPs stay conditional on beam-present records. This lightweight
    ensemble supplies a stable probability gate for candidate selection and can
    be retrained after each completed resource-set batch.
    """

    explorer: XoptFBPICExplorer
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
        try:
            outcomes, beam, _ = physical_outcome_rows(self.explorer, data)
        except ValueError:
            return
        for model_point, beam_present in zip(
            model_input_matrix(self.explorer, outcomes), beam, strict=True
        ):
            self._inputs.append(model_point)
            self._labels.append(1 if beam_present else 0)
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
