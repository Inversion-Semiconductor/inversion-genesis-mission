"""Losses for ``conditions -> parameters`` models.

Two ways to score a predicted unit-cube parameter vector ``u`` for a sample:

* :class:`DirectObjective` evaluates the profile with ``u`` against the lineout
  through the very same :class:`~fludat_fit.fitting.FitObjective` the local
  optimiser minimises. It needs no local fits, and a model trained on it
  learns to minimise the fit error itself. Its gradient with respect to ``u``
  is obtained by central finite differences, so any gradient-based model can
  use it.
* :class:`IndirectObjective` compares ``u`` with the locally optimised
  parameters ``u*`` of the sample (a regression target). It is cheap and has
  an analytic gradient, but inherits the local optimiser's mistakes and treats
  every parameter direction as equally important.

Both return per-sample losses; models decide how to reduce them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from .data import TrainingSet


class Objective(ABC):
    """Per-sample loss of unit-cube predictions on a :class:`TrainingSet`."""

    name: str
    needs_targets: bool

    @abstractmethod
    def per_sample(
        self,
        predictions: np.ndarray,
        training_set: TrainingSet,
        indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Loss of ``predictions[i]`` on sample ``indices[i]`` (default: all, in order)."""

    @abstractmethod
    def gradient(
        self,
        predictions: np.ndarray,
        training_set: TrainingSet,
        indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        """``(N, d)`` derivative of each per-sample loss with respect to its prediction."""

    def loss(
        self,
        predictions: np.ndarray,
        training_set: TrainingSet,
        indices: Sequence[int] | None = None,
    ) -> float:
        """Mean per-sample loss."""
        return float(np.mean(self.per_sample(predictions, training_set, indices)))

    def check(self, training_set: TrainingSet) -> None:
        if self.needs_targets and not training_set.has_targets:
            raise ValueError(
                f"objective {self.name!r} needs local-fit targets; build the training "
                "set with fit_targets=True"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


def _resolve(
    predictions: np.ndarray, training_set: TrainingSet, indices: Sequence[int] | None
) -> tuple[np.ndarray, list[int]]:
    predictions = np.atleast_2d(np.asarray(predictions, dtype=np.float64))
    resolved = (
        list(range(len(training_set))) if indices is None else [int(i) for i in indices]
    )
    if predictions.shape != (len(resolved), training_set.dimension):
        raise ValueError(
            f"predictions have shape {predictions.shape}; expected "
            f"({len(resolved)}, {training_set.dimension})"
        )
    return predictions, resolved


class DirectObjective(Objective):
    """The local-fit loss of the predicted parameters (no targets needed).

    Args:
        eps: Finite-difference step in unit-cube coordinates for :meth:`gradient`.
    """

    name = "direct"
    needs_targets = False

    def __init__(self, eps: float = 1.0e-4) -> None:
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def per_sample(self, predictions, training_set, indices=None):
        predictions, resolved = _resolve(predictions, training_set, indices)
        return np.array(
            [
                training_set.fit_objective(index)(u)
                for u, index in zip(predictions, resolved, strict=True)
            ]
        )

    def gradient(self, predictions, training_set, indices=None):
        predictions, resolved = _resolve(predictions, training_set, indices)
        gradients = np.empty_like(predictions)
        for row, (u, index) in enumerate(zip(predictions, resolved, strict=True)):
            objective = training_set.fit_objective(index)
            for k in range(u.size):
                step = np.zeros_like(u)
                step[k] = self.eps
                # Keep both stencil points inside the cube.
                up = np.clip(u + step, 0.0, 1.0)
                down = np.clip(u - step, 0.0, 1.0)
                spacing = up[k] - down[k]
                gradients[row, k] = (
                    (objective(up) - objective(down)) / spacing
                    if spacing > 0.0
                    else 0.0
                )
        return gradients


class IndirectObjective(Objective):
    """Squared unit-cube distance to the locally optimised parameters.

    Args:
        weights: Per-parameter weights (length ``d``); default all one. The
            loss is ``mean_k w_k (u_k - u*_k)**2``.
    """

    name = "indirect"
    needs_targets = True

    def __init__(self, weights: Sequence[float] | np.ndarray | None = None) -> None:
        self.weights = (
            None if weights is None else np.asarray(weights, dtype=np.float64)
        )
        if self.weights is not None and np.any(self.weights < 0.0):
            raise ValueError("weights must be non-negative")

    def _weights(self, dimension: int) -> np.ndarray:
        if self.weights is None:
            return np.ones(dimension)
        if self.weights.shape != (dimension,):
            raise ValueError(f"weights must have length {dimension}")
        return self.weights

    def per_sample(self, predictions, training_set, indices=None):
        self.check(training_set)
        predictions, resolved = _resolve(predictions, training_set, indices)
        targets = np.array([training_set.samples[i].target for i in resolved])
        w = self._weights(training_set.dimension)
        return np.mean(w * (predictions - targets) ** 2, axis=1)

    def gradient(self, predictions, training_set, indices=None):
        self.check(training_set)
        predictions, resolved = _resolve(predictions, training_set, indices)
        targets = np.array([training_set.samples[i].target for i in resolved])
        w = self._weights(training_set.dimension)
        return 2.0 * w * (predictions - targets) / training_set.dimension


OBJECTIVES: dict[str, type[Objective]] = {
    DirectObjective.name: DirectObjective,
    IndirectObjective.name: IndirectObjective,
}


def get_objective(name: str) -> Objective:
    if name not in OBJECTIVES:
        raise KeyError(f"unknown objective {name!r}; choose from {list(OBJECTIVES)}")
    return OBJECTIVES[name]()
