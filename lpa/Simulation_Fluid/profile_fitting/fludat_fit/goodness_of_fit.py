"""Goodness-of-fit metrics and ranking, independent of how a fit was obtained."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

_TINY = np.finfo(np.float64).tiny


@dataclass(frozen=True)
class GoodnessOfFit:
    """Residual statistics of ``model`` against ``data`` on a common ``z`` grid.

    ``sse``/``rmse`` are in squared and plain density units; ``nrmse`` and
    ``max_abs_error`` are normalised by the data peak; ``integrated_relative_error``
    is ``∫|model - data| dz / ∫|data| dz`` (the metric ``fludat_proc.convergence``
    uses); ``aic``/``bic`` assume Gaussian residuals with ``n_parameters``
    including the amplitude.
    """

    n_points: int
    n_parameters: int
    sse: float
    rmse: float
    nrmse: float
    r_squared: float
    max_abs_error: float
    integrated_relative_error: float
    aic: float
    bic: float

    @classmethod
    def compute(
        cls,
        z: np.ndarray,
        data: np.ndarray,
        model: np.ndarray,
        *,
        n_parameters: int,
        weights: np.ndarray | None = None,
    ) -> GoodnessOfFit:
        z = np.asarray(z, dtype=np.float64)
        data = np.asarray(data, dtype=np.float64)
        model = np.asarray(model, dtype=np.float64)
        w = (
            np.ones_like(data)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        residual = model - data
        n = int(data.size)

        sse = float(np.sum(w * residual**2))
        total_weight = float(np.sum(w))
        rmse = float(np.sqrt(sse / total_weight))
        peak = float(np.max(np.abs(data)))
        peak = peak if peak > 0.0 else 1.0
        mean = float(np.sum(w * data) / total_weight)
        total_ss = float(np.sum(w * (data - mean) ** 2))
        r_squared = 1.0 - sse / total_ss if total_ss > 0.0 else float("nan")

        data_mass = float(np.trapezoid(np.abs(data), z))
        integrated = (
            float(np.trapezoid(np.abs(residual), z)) / data_mass
            if data_mass > 0.0
            else float("nan")
        )

        log_likelihood_term = n * np.log(max(sse, _TINY) / n)
        return cls(
            n_points=n,
            n_parameters=int(n_parameters),
            sse=sse,
            rmse=rmse,
            nrmse=rmse / peak,
            r_squared=r_squared,
            max_abs_error=float(np.max(np.abs(residual))) / peak,
            integrated_relative_error=integrated,
            aic=float(log_likelihood_term + 2 * n_parameters),
            bic=float(log_likelihood_term + n_parameters * np.log(n)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HasGoodness(Protocol):
    goodness: GoodnessOfFit


RANKING_METRICS: dict[str, Callable[[GoodnessOfFit], float]] = {
    # Each key function returns "lower is better".
    "bic": lambda g: g.bic,
    "aic": lambda g: g.aic,
    "nrmse": lambda g: g.nrmse,
    "rmse": lambda g: g.rmse,
    "sse": lambda g: g.sse,
    "integrated_relative_error": lambda g: g.integrated_relative_error,
    "max_abs_error": lambda g: g.max_abs_error,
    "r_squared": lambda g: -g.r_squared,
}
"""Metric name -> key such that smaller keys rank better."""

DEFAULT_RANKING_METRIC = "bic"


def rank_results(
    results: Iterable[HasGoodness], by: str = DEFAULT_RANKING_METRIC
) -> list:
    """Sort fit results best-first by one of :data:`RANKING_METRICS`."""
    if by not in RANKING_METRICS:
        raise KeyError(
            f"unknown ranking metric {by!r}; choose from {list(RANKING_METRICS)}"
        )
    key = RANKING_METRICS[by]

    def sort_key(result: HasGoodness) -> float:
        value = key(result.goodness)
        return float("inf") if not np.isfinite(value) else value

    return sorted(results, key=sort_key)
