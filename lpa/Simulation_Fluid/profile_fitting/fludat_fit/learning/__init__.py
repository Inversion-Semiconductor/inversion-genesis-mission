"""Evaluation framework for models that map lineout conditions to profile parameters.

Pieces:

* :mod:`~fludat_fit.learning.data` - :class:`TrainingSet` (lineouts at sampled
  conditions plus their local-fit parameters in one fixed unit cube),
  :func:`build_training_set`, splits and ``.npz`` storage.
* :mod:`~fludat_fit.learning.objectives` - :class:`DirectObjective` (the local
  optimiser's own fit loss of the predicted parameters) and
  :class:`IndirectObjective` (distance to the locally optimised parameters).
* :mod:`~fludat_fit.learning.models` - the :class:`ParameterModel` protocol and
  baseline models (constant, nearest neighbour, polynomial).
* :mod:`~fludat_fit.learning.evaluation` - :func:`evaluate_model`,
  :func:`cross_validate`, :func:`compare_models` and their reports.

The framework is agnostic to the model: anything with ``fit(training_set,
objective)`` and ``predict(features)`` can be evaluated.
"""

from .data import (
    TrainingSample,
    TrainingSet,
    build_training_set,
    reference_parameter_space,
)
from .evaluation import (
    ModelComparison,
    ModelEvaluation,
    RefinementResult,
    compare_models,
    cross_validate,
    evaluate_model,
    merge_evaluations,
    refine_predictions,
    score_predictions,
)
from .models import (
    ConstantModel,
    ModelFactory,
    NearestNeighborModel,
    ParameterModel,
    PolynomialModel,
    builtin_model_factories,
    get_model_factories,
)
from .objectives import (
    OBJECTIVES,
    DirectObjective,
    IndirectObjective,
    Objective,
    get_objective,
)

__all__ = [
    "OBJECTIVES",
    "ConstantModel",
    "DirectObjective",
    "IndirectObjective",
    "ModelComparison",
    "ModelEvaluation",
    "ModelFactory",
    "NearestNeighborModel",
    "Objective",
    "ParameterModel",
    "PolynomialModel",
    "RefinementResult",
    "TrainingSample",
    "TrainingSet",
    "build_training_set",
    "builtin_model_factories",
    "compare_models",
    "cross_validate",
    "evaluate_model",
    "get_model_factories",
    "get_objective",
    "merge_evaluations",
    "reference_parameter_space",
    "refine_predictions",
    "score_predictions",
]
