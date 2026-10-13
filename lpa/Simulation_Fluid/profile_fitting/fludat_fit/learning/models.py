"""The model interface, and reference models that exercise the framework.

A :class:`ParameterModel` maps normalised conditions (``(N, 3)`` features in
``[0, 1]``) to unit-cube profile parameters (``(N, d)``). It is trained on a
:class:`~fludat_fit.learning.data.TrainingSet` under an
:class:`~fludat_fit.learning.objectives.Objective`, which it may query for
per-sample losses and gradients however it likes: closed-form regression for
the indirect loss, gradient descent on either loss, or anything else. The
framework never looks inside a model.

The models here are deliberately simple baselines that a real learned model
should beat: a constant, a nearest-neighbour lookup, and a polynomial map
trained by L-BFGS-B on whichever objective it is given.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import minimize

from .data import TrainingSet
from .objectives import IndirectObjective, Objective


@runtime_checkable
class ParameterModel(Protocol):
    """Anything that learns ``features -> unit-cube parameters``."""

    name: str

    def fit(self, training_set: TrainingSet, objective: Objective) -> None:
        """Train on ``training_set`` to minimise ``objective``.

        Raise ``ValueError`` for an unsupported combination (for example a
        pure regression model given the direct objective on a set without targets).
        """

    def predict(self, features: np.ndarray) -> np.ndarray:
        """``(N, d)`` unit-cube parameters for ``(N, 3)`` normalised conditions."""


ModelFactory = Callable[[], ParameterModel]
"""Creates a fresh, untrained model; evaluation trains one per fold and objective."""


def clip_unit(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


# ------------------------------------------------------------------ baselines
class ConstantModel:
    """Predicts one parameter vector for every condition.

    Indirect: the mean target. Direct: the single vector minimising the mean
    fit loss over the training set, by L-BFGS-B from the cube centre (or from
    the mean target when targets exist).
    """

    name = "constant"

    def __init__(self, maxiter: int = 200) -> None:
        self.maxiter = maxiter
        self.value: np.ndarray | None = None

    def fit(self, training_set: TrainingSet, objective: Objective) -> None:
        objective.check(training_set)
        d = training_set.dimension
        start = (
            training_set.targets.mean(axis=0)
            if training_set.has_targets
            else np.full(d, 0.5)
        )
        if isinstance(objective, IndirectObjective):
            self.value = clip_unit(start)
            return

        n = len(training_set)

        def loss_and_gradient(u: np.ndarray) -> tuple[float, np.ndarray]:
            tiled = np.tile(u, (n, 1))
            return (
                objective.loss(tiled, training_set),
                objective.gradient(tiled, training_set).mean(axis=0),
            )

        result = minimize(
            loss_and_gradient,
            clip_unit(start),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * d,
            options={"maxiter": self.maxiter},
        )
        self.value = clip_unit(result.x)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.value is None:
            raise RuntimeError("model is not trained")
        n = np.atleast_2d(features).shape[0]
        return np.tile(self.value, (n, 1))


class NearestNeighborModel:
    """Inverse-distance-weighted average of the ``k`` nearest training targets.

    A pure lookup: it needs local-fit targets and ignores the objective apart
    from that requirement.
    """

    name = "knn"

    def __init__(self, k: int = 3, power: float = 2.0) -> None:
        if k < 1:
            raise ValueError("k must be at least one")
        self.k = k
        self.power = power
        self.name = f"knn{k}"
        self._features: np.ndarray | None = None
        self._targets: np.ndarray | None = None

    def fit(self, training_set: TrainingSet, objective: Objective) -> None:
        if not training_set.has_targets:
            raise ValueError(f"{self.name} needs local-fit targets")
        self._features = training_set.features
        self._targets = training_set.targets

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._features is None or self._targets is None:
            raise RuntimeError("model is not trained")
        features = np.atleast_2d(np.asarray(features, dtype=np.float64))
        k = min(self.k, self._features.shape[0])
        distances = np.linalg.norm(
            features[:, None, :] - self._features[None, :, :], axis=2
        )
        nearest = np.argsort(distances, axis=1)[:, :k]
        predictions = np.empty((features.shape[0], self._targets.shape[1]))
        for row in range(features.shape[0]):
            d = distances[row, nearest[row]]
            if d[0] == 0.0:
                predictions[row] = self._targets[nearest[row][0]]
                continue
            w = 1.0 / d**self.power
            predictions[row] = (w[:, None] * self._targets[nearest[row]]).sum(
                axis=0
            ) / w.sum()
        return clip_unit(predictions)


class PolynomialModel:
    """Polynomial map from features to a sigmoid-squashed unit cube.

    ``u = sigmoid(Phi(features) @ W)`` with all monomials up to ``degree``.
    Under the indirect objective ``W`` starts from ridge least squares on the
    logit of the targets and is then refined by L-BFGS-B on the objective; under
    the direct objective it starts from that solution when targets exist (else
    from the cube centre) and descends the fit loss through the objective's
    gradient and the chain rule. This is the template for any differentiable
    model: it only ever calls ``objective.loss`` and ``objective.gradient``.
    """

    def __init__(
        self, degree: int = 1, ridge: float = 1.0e-6, maxiter: int = 300
    ) -> None:
        if degree < 0:
            raise ValueError("degree must be non-negative")
        self.degree = degree
        self.ridge = ridge
        self.maxiter = maxiter
        self.name = f"poly{degree}"
        self.weights: np.ndarray | None = None
        self.history: list[float] = []

    # -- features ----------------------------------------------------------
    def basis(self, features: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(np.asarray(features, dtype=np.float64))
        centred = features - 0.5
        columns = [np.ones(features.shape[0])]
        for total in range(1, self.degree + 1):
            for combo in itertools.combinations_with_replacement(
                range(features.shape[1]), total
            ):
                columns.append(np.prod(centred[:, combo], axis=1))
        return np.column_stack(columns)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    @staticmethod
    def _logit(u: np.ndarray, margin: float = 1.0e-3) -> np.ndarray:
        u = np.clip(u, margin, 1.0 - margin)
        return np.log(u / (1.0 - u))

    # -- training ----------------------------------------------------------
    def fit(self, training_set: TrainingSet, objective: Objective) -> None:
        objective.check(training_set)
        phi = self.basis(training_set.features)
        m, d = phi.shape[1], training_set.dimension
        if training_set.has_targets:
            gram = phi.T @ phi + self.ridge * np.eye(m)
            weights = np.linalg.solve(gram, phi.T @ self._logit(training_set.targets))
        else:
            weights = np.zeros((m, d))
        self.history = []

        def loss_and_gradient(flat: np.ndarray) -> tuple[float, np.ndarray]:
            w = flat.reshape(m, d)
            logits = phi @ w
            u = self._sigmoid(logits)
            loss = objective.loss(u, training_set) + self.ridge * float(np.sum(w**2))
            du = objective.gradient(u, training_set) / len(training_set)
            dlogits = du * u * (1.0 - u)
            grad = phi.T @ dlogits + 2.0 * self.ridge * w
            self.history.append(loss)
            return loss, grad.ravel()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = minimize(
                loss_and_gradient,
                weights.ravel(),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": self.maxiter},
            )
        self.weights = result.x.reshape(m, d)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("model is not trained")
        return self._sigmoid(self.basis(features) @ self.weights)


# ------------------------------------------------------------------- registry
def builtin_model_factories() -> dict[str, ModelFactory]:
    """Name -> factory for the reference models."""
    return {
        "constant": ConstantModel,
        "knn3": lambda: NearestNeighborModel(3),
        "poly1": lambda: PolynomialModel(1),
        "poly2": lambda: PolynomialModel(2),
    }


def get_model_factories(names: Sequence[str] | None = None) -> dict[str, ModelFactory]:
    registry = builtin_model_factories()
    if names is None:
        return registry
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise KeyError(f"unknown models {unknown}; choose from {list(registry)}")
    return {name: registry[name] for name in names}


def describe_model(model: Any) -> str:
    return getattr(model, "name", type(model).__name__)
