"""Evaluate and compare ``conditions -> parameters`` models.

Every model is scored the same way regardless of how it was trained:

* **direct**: the fit quality of the predicted parameters against the held-out
  lineouts (through :class:`~fludat_fit.fitting.FitObjective`, so NRMSE and
  friends are the usual :class:`~fludat_fit.goodness_of_fit.GoodnessOfFit`),
  compared with the locally optimised fit of the same lineout;
* **indirect**: the unit-cube distance between the prediction and the local
  optimum, overall and per parameter;
* **refinement** (optional): how many objective evaluations a local optimiser
  needs when started from the prediction, versus the cold multi-start fit.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..fitting import MultiStartLocalFit
from ..goodness_of_fit import GoodnessOfFit
from .data import TrainingSet
from .models import ModelFactory, ParameterModel, describe_model
from .objectives import Objective


@dataclass(frozen=True)
class RefinementResult:
    """Local optimisation warm-started from a prediction."""

    nrmse: np.ndarray
    evaluations: np.ndarray
    reference_evaluations: np.ndarray

    @property
    def evaluation_ratio(self) -> np.ndarray:
        """Warm evaluations / cold evaluations per sample (``nan`` without a cold fit)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                self.reference_evaluations > 0,
                self.evaluations / self.reference_evaluations,
                np.nan,
            )


@dataclass(frozen=True)
class ModelEvaluation:
    """One trained model scored on one test set."""

    model_name: str
    objective_name: str
    training_size: int
    test_set: TrainingSet = field(repr=False)
    predictions: np.ndarray = field(repr=False)
    goodness: tuple[GoodnessOfFit, ...] = field(repr=False)
    direct_loss: np.ndarray = field(repr=False)
    train_seconds: float = 0.0
    predict_seconds: float = 0.0
    refinement: RefinementResult | None = field(default=None, repr=False)
    error: str | None = None

    # -- per-sample arrays -------------------------------------------------
    @property
    def n_test(self) -> int:
        return len(self.test_set)

    @property
    def nrmse(self) -> np.ndarray:
        return np.array([g.nrmse for g in self.goodness])

    @property
    def reference_nrmse(self) -> np.ndarray:
        return self.test_set.reference_nrmse

    @property
    def nrmse_ratio(self) -> np.ndarray:
        """Predicted-fit NRMSE over local-fit NRMSE (``nan`` without a local fit)."""
        reference = self.reference_nrmse
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(reference > 0.0, self.nrmse / reference, np.nan)

    @property
    def parameter_error(self) -> np.ndarray | None:
        """``(N, d)`` absolute unit-cube error per parameter, if targets exist."""
        if not self.test_set.has_targets:
            return None
        return np.abs(self.predictions - self.test_set.targets)

    @property
    def indirect_loss(self) -> np.ndarray | None:
        """Mean squared unit-cube distance to the local optimum, if targets exist."""
        error = self.parameter_error
        return None if error is None else np.mean(error**2, axis=1)

    # -- summaries -----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        def stats(values: np.ndarray | None) -> dict[str, float] | None:
            if values is None:
                return None
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return None
            return {
                "median": float(np.median(finite)),
                "mean": float(np.mean(finite)),
                "p90": float(np.percentile(finite, 90)),
                "max": float(np.max(finite)),
            }

        payload: dict[str, Any] = {
            "model": self.model_name,
            "objective": self.objective_name,
            "training_size": self.training_size,
            "test_size": self.n_test,
            "error": self.error,
            "train_seconds": self.train_seconds,
            "predict_seconds": self.predict_seconds,
            "direct_loss": stats(self.direct_loss),
            "nrmse": stats(self.nrmse),
            "reference_nrmse": stats(self.reference_nrmse),
            "nrmse_ratio": stats(self.nrmse_ratio),
            "indirect_loss": stats(self.indirect_loss),
            "parameter_error": None,
            "refinement": None,
        }
        error = self.parameter_error
        if error is not None:
            payload["parameter_error"] = {
                name: float(np.median(error[:, k]))
                for k, name in enumerate(self.test_set.space.names)
            }
        if self.refinement is not None:
            payload["refinement"] = {
                "nrmse": stats(self.refinement.nrmse),
                "evaluations": stats(self.refinement.evaluations.astype(float)),
                "evaluation_ratio": stats(self.refinement.evaluation_ratio),
            }
        return payload

    def rows(self) -> list[dict[str, Any]]:
        """One record per test sample."""
        rows = []
        indirect = self.indirect_loss
        error = self.parameter_error
        for i, sample in enumerate(self.test_set.samples):
            row: dict[str, Any] = {
                "model": self.model_name,
                "objective": self.objective_name,
                **sample.conditions.to_dict(),
                "direct_loss": float(self.direct_loss[i]),
                "nrmse": float(self.nrmse[i]),
                "reference_nrmse": float(self.reference_nrmse[i]),
                "nrmse_ratio": float(self.nrmse_ratio[i]),
                "indirect_loss": None if indirect is None else float(indirect[i]),
            }
            for k, name in enumerate(self.test_set.space.names):
                row[f"pred_{name}"] = float(self.predictions[i, k])
                if error is not None:
                    row[f"err_{name}"] = float(error[i, k])
            if self.refinement is not None:
                row["refined_nrmse"] = float(self.refinement.nrmse[i])
                row["refined_evaluations"] = int(self.refinement.evaluations[i])
                row["reference_evaluations"] = int(
                    self.refinement.reference_evaluations[i]
                )
            rows.append(row)
        return rows


# ---------------------------------------------------------------- evaluation
def score_predictions(
    predictions: np.ndarray, test_set: TrainingSet
) -> tuple[np.ndarray, tuple[GoodnessOfFit, ...]]:
    """Direct loss and goodness of fit of unit-cube ``predictions`` on ``test_set``."""
    predictions = np.clip(np.atleast_2d(predictions), 0.0, 1.0)
    losses = np.empty(len(test_set))
    goodness = []
    for i in range(len(test_set)):
        objective = test_set.fit_objective(i)
        losses[i] = objective(predictions[i])
        goodness.append(objective.result(predictions[i], scheme="prediction").goodness)
    return losses, tuple(goodness)


def refine_predictions(
    predictions: np.ndarray,
    test_set: TrainingSet,
    scheme: MultiStartLocalFit | None = None,
) -> RefinementResult:
    """Polish each prediction with a local optimiser started only from it."""
    scheme = scheme or MultiStartLocalFit(n_starts=0, include_initial_guess=False)
    nrmse = np.empty(len(test_set))
    evaluations = np.empty(len(test_set), dtype=int)
    for i, sample in enumerate(test_set.samples):
        result = scheme.fit(
            sample.lineout,
            test_set.family,
            space=test_set.space,
            warm_start=predictions[i],
        )
        nrmse[i] = result.goodness.nrmse
        evaluations[i] = result.n_function_evaluations
    return RefinementResult(
        nrmse=nrmse,
        evaluations=evaluations,
        reference_evaluations=np.array(
            [s.reference_evaluations for s in test_set.samples], dtype=int
        ),
    )


def evaluate_model(
    model: ParameterModel,
    train_set: TrainingSet,
    test_set: TrainingSet,
    objective: Objective,
    *,
    refine: bool = False,
    refine_scheme: MultiStartLocalFit | None = None,
) -> ModelEvaluation:
    """Train ``model`` on ``train_set`` under ``objective`` and score it on ``test_set``.

    An unsupported model/objective combination (``ValueError`` from ``fit``)
    yields an evaluation with ``error`` set and centre-of-cube predictions.
    """
    name = describe_model(model)
    error = None
    started = time.perf_counter()
    try:
        model.fit(train_set, objective)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predictions = np.clip(np.atleast_2d(model.predict(test_set.features)), 0.0, 1.0)
        predict_seconds = time.perf_counter() - started
    except ValueError as exc:
        error = str(exc)
        train_seconds = time.perf_counter() - started
        predict_seconds = 0.0
        predictions = np.full((len(test_set), test_set.dimension), 0.5)
    if predictions.shape != (len(test_set), test_set.dimension):
        raise ValueError(
            f"{name}.predict returned shape {predictions.shape}; expected "
            f"({len(test_set)}, {test_set.dimension})"
        )
    losses, goodness = score_predictions(predictions, test_set)
    refinement = (
        refine_predictions(predictions, test_set, refine_scheme)
        if refine and error is None
        else None
    )
    return ModelEvaluation(
        model_name=name,
        objective_name=objective.name,
        training_size=len(train_set),
        test_set=test_set,
        predictions=predictions,
        goodness=goodness,
        direct_loss=losses,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        refinement=refinement,
        error=error,
    )


def cross_validate(
    factory: ModelFactory,
    training_set: TrainingSet,
    objective: Objective,
    *,
    k: int = 4,
    seed: int | None = 0,
    refine: bool = False,
) -> list[ModelEvaluation]:
    """One :class:`ModelEvaluation` per fold, each with a freshly built model."""
    return [
        evaluate_model(factory(), train, test, objective, refine=refine)
        for train, test in training_set.k_folds(k, seed)
    ]


def merge_evaluations(evaluations: Sequence[ModelEvaluation]) -> ModelEvaluation:
    """Concatenate fold evaluations of one model/objective into one record."""
    if not evaluations:
        raise ValueError("nothing to merge")
    first = evaluations[0]
    if any(
        e.model_name != first.model_name or e.objective_name != first.objective_name
        for e in evaluations
    ):
        raise ValueError("can only merge evaluations of one model and objective")
    base = first.test_set
    samples = [s for e in evaluations for s in e.test_set.samples]
    merged_set = TrainingSet(
        base.family, base.space, base.ranges, samples, name="folds"
    )
    refinement = None
    if all(e.refinement is not None for e in evaluations):
        refinement = RefinementResult(
            nrmse=np.concatenate([e.refinement.nrmse for e in evaluations]),
            evaluations=np.concatenate([e.refinement.evaluations for e in evaluations]),
            reference_evaluations=np.concatenate(
                [e.refinement.reference_evaluations for e in evaluations]
            ),
        )
    errors = [e.error for e in evaluations if e.error]
    return ModelEvaluation(
        model_name=first.model_name,
        objective_name=first.objective_name,
        training_size=int(round(np.mean([e.training_size for e in evaluations]))),
        test_set=merged_set,
        predictions=np.concatenate([e.predictions for e in evaluations]),
        goodness=tuple(g for e in evaluations for g in e.goodness),
        direct_loss=np.concatenate([e.direct_loss for e in evaluations]),
        train_seconds=float(np.sum([e.train_seconds for e in evaluations])),
        predict_seconds=float(np.sum([e.predict_seconds for e in evaluations])),
        refinement=refinement,
        error=errors[0] if errors else None,
    )


# ---------------------------------------------------------------- comparison
@dataclass(frozen=True)
class ModelComparison:
    """Evaluations of several models under several objectives on one training set."""

    training_set: TrainingSet = field(repr=False)
    evaluations: tuple[ModelEvaluation, ...]
    protocol: str

    def ranked(self, by: str = "nrmse") -> list[ModelEvaluation]:
        """Evaluations best-first by the median of ``by`` (``nrmse``, ``direct_loss``, ``indirect_loss``)."""

        def key(evaluation: ModelEvaluation) -> float:
            values = getattr(evaluation, by)
            if values is None or evaluation.error:
                return float("inf")
            finite = values[np.isfinite(values)]
            return float(np.median(finite)) if finite.size else float("inf")

        return sorted(self.evaluations, key=key)

    def table(self) -> str:
        reference = self.training_set.reference_nrmse
        finite = reference[np.isfinite(reference)]
        header = (
            f"{'#':>2}  {'model':<12} {'objective':<9} {'n_train':>7} {'nrmse med':>9} "
            f"{'nrmse p90':>9} {'ratio med':>9} {'indirect':>9} {'refined':>8} {'evals%':>7} "
            f"{'train[s]':>8}"
        )
        lines = [header, "-" * len(header)]
        if finite.size:
            lines.append(
                f"{'':>2}  {'local fit':<12} {'(reference)':<9} {'':>7} "
                f"{np.median(finite):>9.4f} {np.percentile(finite, 90):>9.4f} "
                f"{1.0:>9.2f} {'':>9} {'':>8} {100:>7.0f} {'':>8}"
            )
        for index, e in enumerate(self.ranked(), start=1):
            s = e.summary()
            if e.error:
                lines.append(
                    f"{index:>2}  {e.model_name:<12} {e.objective_name:<9} "
                    f"{e.training_size:>7} unsupported: {e.error}"
                )
                continue

            def med(block: dict | None) -> str:
                return "" if block is None else f"{block['median']:.4f}"

            refined = (
                ""
                if s["refinement"] is None
                else f"{s['refinement']['nrmse']['median']:.4f}"
            )
            evals = (
                ""
                if s["refinement"] is None
                or s["refinement"]["evaluation_ratio"] is None
                else f"{100 * s['refinement']['evaluation_ratio']['median']:.0f}"
            )
            lines.append(
                f"{index:>2}  {e.model_name:<12} {e.objective_name:<9} {e.training_size:>7} "
                f"{s['nrmse']['median']:>9.4f} {s['nrmse']['p90']:>9.4f} "
                f"{med(s['nrmse_ratio']):>9} {med(s['indirect_loss']):>9} {refined:>8} "
                f"{evals:>7} {e.train_seconds:>8.2f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.training_set.family.name,
            "dataset": self.training_set.name,
            "protocol": self.protocol,
            "n_samples": len(self.training_set),
            "parameters": list(self.training_set.space.names),
            "ranges": self.training_set.ranges.to_dict(),
            "evaluations": [e.summary() for e in self.ranked()],
        }

    def rows(self) -> list[dict[str, Any]]:
        return [row for e in self.evaluations for row in e.rows()]


def compare_models(
    factories: Mapping[str, ModelFactory],
    training_set: TrainingSet,
    objectives: Iterable[Objective],
    *,
    folds: int | None = 4,
    test_fraction: float = 0.25,
    seed: int | None = 0,
    refine: bool = False,
) -> ModelComparison:
    """Train and score every model under every objective.

    With ``folds`` the score is ``k``-fold cross-validation merged over folds;
    with ``folds=None`` a single random split of ``test_fraction`` is used.
    """
    objectives = list(objectives)
    evaluations = []
    if folds is None:
        train, test = training_set.split(test_fraction, seed)
        protocol = f"split test_fraction={test_fraction} seed={seed}"
        for name, factory in factories.items():
            for objective in objectives:
                evaluation = evaluate_model(
                    factory(), train, test, objective, refine=refine
                )
                evaluations.append(_renamed(evaluation, name))
    else:
        protocol = f"{folds}-fold cross-validation seed={seed}"
        for name, factory in factories.items():
            for objective in objectives:
                per_fold = cross_validate(
                    factory, training_set, objective, k=folds, seed=seed, refine=refine
                )
                evaluations.append(_renamed(merge_evaluations(per_fold), name))
    return ModelComparison(training_set, tuple(evaluations), protocol)


def _renamed(evaluation: ModelEvaluation, name: str) -> ModelEvaluation:
    if evaluation.model_name == name:
        return evaluation
    from dataclasses import replace

    return replace(evaluation, model_name=name)
