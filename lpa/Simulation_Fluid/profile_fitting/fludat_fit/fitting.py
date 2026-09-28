"""Fitting schemes: turn ``(lineout, family)`` into a :class:`FitResult`.

:class:`FittingScheme` is the interface. :class:`MultiStartLocalFit` is the
current implementation: bounded L-BFGS-B from Latin-hypercube starts in the
family's unit cube, with the amplitude solved in closed form at every
evaluation. A learned model that predicts parameters from
:class:`~fludat_fit.lineout.LineoutConditions` would implement the same
protocol and return the same :class:`FitResult`, so everything downstream
(goodness of fit, ranking, profile export, plots) is shared.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from inversion_fbpic.lib.density_core import _DensityProfile
from scipy.optimize import OptimizeResult, minimize

from .families import ProfileFamily, get_families
from .goodness_of_fit import DEFAULT_RANKING_METRIC, GoodnessOfFit, rank_results
from .lineout import Lineout, LineoutConditions
from .parameters import ParameterSpace

DENSITY_UNITS_TO_M3: dict[str, float] = {
    "m^-3": 1.0,
    "m**-3": 1.0,
    "1/m^3": 1.0,
    "cm^-3": 1.0e6,
    "cm**-3": 1.0e6,
    "1/cm^3": 1.0e6,
    "mm^-3": 1.0e9,
    "mm**-3": 1.0e9,
    "1/mm^3": 1.0e9,
}
"""Conversion factors from density-cube units to m^-3 for the units we recognise."""

INVALID_OBJECTIVE = 1.0e6
"""Objective value returned when a parameter vector cannot build a profile."""


def density_units_to_m3(units: str) -> float | None:
    """Factor converting ``units`` to m^-3, or ``None`` for unrecognised units."""
    return DENSITY_UNITS_TO_M3.get(units.strip().replace(" ", ""))


# ------------------------------------------------------------------- result
@dataclass(frozen=True)
class FitResult:
    """A fitted family: shape parameters, amplitude and goodness of fit.

    ``amplitude`` is in the lineout's density units; the model curve is
    ``amplitude * family.relative_density(parameters, z)``.
    """

    family: ProfileFamily = field(repr=False, compare=False)
    lineout: Lineout = field(repr=False, compare=False)
    space: ParameterSpace = field(repr=False, compare=False)
    theta: np.ndarray
    amplitude: float
    goodness: GoodnessOfFit
    scheme: str
    success: bool = True
    message: str = ""
    n_starts: int = 1
    n_function_evaluations: int = 0
    elapsed_seconds: float = 0.0

    @property
    def family_name(self) -> str:
        return self.family.name

    @property
    def conditions(self) -> LineoutConditions:
        return self.lineout.conditions

    @property
    def parameters(self) -> dict[str, float]:
        return self.space.as_dict(self.theta)

    def model_density(self, z: np.ndarray | None = None) -> np.ndarray:
        """The fitted curve in lineout units, on ``z`` [m] (default: the lineout grid)."""
        z = self.lineout.z if z is None else np.asarray(z, dtype=np.float64)
        return self.amplitude * self.family.relative_density(self.parameters, z)

    def nominal_density_m3(self) -> float | None:
        """The amplitude converted to m^-3, or ``None`` if the units are unknown."""
        factor = density_units_to_m3(self.lineout.density_units)
        return None if factor is None else self.amplitude * factor

    def build_profile(
        self, nominal_density: float | None = None, **overrides: Any
    ) -> _DensityProfile:
        """The FBPIC density-profile object for this fit.

        ``nominal_density`` [m^-3] defaults to the amplitude converted from the
        lineout units; pass it explicitly when the units are not recognised.
        ``overrides`` (``species``, ``ionization``, ``p_nz``, ...) replace the
        family's default constructor arguments.
        """
        if nominal_density is None:
            nominal_density = self.nominal_density_m3()
            if nominal_density is None:
                raise ValueError(
                    f"cannot convert density units {self.lineout.density_units!r} to m^-3; "
                    "pass nominal_density explicitly"
                )
        return self.family.build(
            self.parameters, nominal_density=nominal_density, **overrides
        )

    def to_dict(self, include_profile: bool = True) -> dict[str, Any]:
        """JSON-friendly summary; ``profile_config`` is the profile's serialized form."""
        payload: dict[str, Any] = {
            "family": self.family_name,
            "scheme": self.scheme,
            "conditions": self.conditions.to_dict(),
            "source": self.lineout.source,
            "parameters": self.parameters,
            "parameter_bounds": {
                spec.name: [spec.lower, spec.upper] for spec in self.space.specs
            },
            "amplitude": float(self.amplitude),
            "amplitude_units": self.lineout.density_units,
            "nominal_density_m3": self.nominal_density_m3(),
            "goodness_of_fit": self.goodness.to_dict(),
            "success": bool(self.success),
            "message": self.message,
            "n_starts": int(self.n_starts),
            "n_function_evaluations": int(self.n_function_evaluations),
            "elapsed_seconds": float(self.elapsed_seconds),
        }
        if include_profile:
            nominal = self.nominal_density_m3()
            profile = self.family.build(
                self.parameters, nominal_density=nominal if nominal else 1.0
            )
            payload["profile_config"] = profile.to_dict()
            payload["profile_config_nominal_density_is_placeholder"] = nominal is None
        return payload


# ------------------------------------------------------------------ scheme
@runtime_checkable
class FittingScheme(Protocol):
    """Anything that fits a family to a lineout."""

    name: str

    def fit(self, lineout: Lineout, family: ProfileFamily) -> FitResult: ...


def optimal_amplitude(
    relative: np.ndarray, data: np.ndarray, weights: np.ndarray
) -> float:
    """Non-negative least-squares amplitude for ``data ~ amplitude * relative``."""
    denominator = float(np.sum(weights * relative * relative))
    if denominator <= 0.0:
        return 0.0
    return max(float(np.sum(weights * relative * data)) / denominator, 0.0)


class MultiStartLocalFit:
    """Bounded local optimisation from several starting points.

    Each start is a point of a Latin-hypercube sample in the family's unit cube;
    the family's :meth:`~fludat_fit.families.ProfileFamily.initial_guess` is
    added when available. The objective is the weighted sum of squared
    residuals, normalised by the weighted sum of squared data so tolerances are
    unit-free, with the amplitude eliminated in closed form (variable projection).

    Args:
        n_starts: Number of sampled starting points.
        method: Any bounded ``scipy.optimize.minimize`` method (``L-BFGS-B``,
            ``Powell``, ``TNC``, ``Nelder-Mead``, ...).
        seed: Seed for the start sample.
        include_initial_guess: Add the family heuristic as one more start.
        options: Extra solver options; merged over the per-method defaults.
        fit_amplitude: If ``False`` the amplitude is fixed to the lineout peak.
    """

    name = "multistart_local"

    _DEFAULT_OPTIONS: dict[str, dict[str, Any]] = {
        "L-BFGS-B": {"maxiter": 1000, "eps": 1.0e-6},
        "TNC": {"maxfun": 5000, "eps": 1.0e-6},
        "Powell": {"maxiter": 5000, "xtol": 1.0e-4, "ftol": 1.0e-8},
        "Nelder-Mead": {"maxiter": 5000, "xatol": 1.0e-4, "fatol": 1.0e-10},
    }

    def __init__(
        self,
        n_starts: int = 8,
        method: str = "L-BFGS-B",
        seed: int | None = 0,
        include_initial_guess: bool = True,
        options: Mapping[str, Any] | None = None,
        fit_amplitude: bool = True,
    ) -> None:
        if n_starts < 0 or (n_starts == 0 and not include_initial_guess):
            raise ValueError("need at least one start")
        self.n_starts = int(n_starts)
        self.method = method
        self.seed = seed
        self.include_initial_guess = include_initial_guess
        self.options = {**self._DEFAULT_OPTIONS.get(method, {}), **(options or {})}
        self.fit_amplitude = fit_amplitude
        self.warm_starts: dict[str, Mapping[str, float]] = {}
        """Family name -> parameters used as one more start (e.g. the previous fit)."""

    # -- public --------------------------------------------------------------
    def fit(self, lineout: Lineout, family: ProfileFamily) -> FitResult:
        started = time.perf_counter()
        space = family.parameter_space(lineout.summary())
        z, data, weights = lineout.z, lineout.density, lineout.effective_weights()
        normalisation = float(np.sum(weights * data * data)) or 1.0
        fixed_amplitude = None if self.fit_amplitude else lineout.peak
        evaluations = 0

        def objective(u: np.ndarray) -> float:
            nonlocal evaluations
            evaluations += 1
            parameters = space.as_dict(space.from_unit(np.clip(u, 0.0, 1.0)))
            try:
                relative = family.relative_density(parameters, z)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                return INVALID_OBJECTIVE
            if not np.all(np.isfinite(relative)):
                return INVALID_OBJECTIVE
            amplitude = (
                optimal_amplitude(relative, data, weights)
                if fixed_amplitude is None
                else fixed_amplitude
            )
            residual = amplitude * relative - data
            return float(np.sum(weights * residual * residual)) / normalisation

        best: OptimizeResult | None = None
        starts = self._starts(space, lineout, family)
        for u0 in starts:
            result = self._minimize(objective, u0, space.dimension)
            if best is None or result.fun < best.fun:
                best = result
        assert best is not None

        theta = space.from_unit(np.clip(best.x, 0.0, 1.0))
        parameters = space.as_dict(theta)
        relative = family.relative_density(parameters, z)
        amplitude = (
            optimal_amplitude(relative, data, weights)
            if fixed_amplitude is None
            else fixed_amplitude
        )
        goodness = GoodnessOfFit.compute(
            z,
            data,
            amplitude * relative,
            n_parameters=space.dimension + (1 if fixed_amplitude is None else 0),
            weights=lineout.weights,
        )
        return FitResult(
            family=family,
            lineout=lineout,
            space=space,
            theta=theta,
            amplitude=amplitude,
            goodness=goodness,
            scheme=f"{self.name}[{self.method}]",
            success=bool(best.success) and best.fun < INVALID_OBJECTIVE,
            message=str(best.message),
            n_starts=len(starts),
            n_function_evaluations=evaluations,
            elapsed_seconds=time.perf_counter() - started,
        )

    # -- internals -----------------------------------------------------------
    def _starts(
        self, space: ParameterSpace, lineout: Lineout, family: ProfileFamily
    ) -> list[np.ndarray]:
        starts: list[np.ndarray] = []
        warm = self.warm_starts.get(family.name)
        if warm is not None and set(warm) >= set(space.names):
            starts.append(self._unit_point(space, warm))
        if self.include_initial_guess:
            guess = family.initial_guess(lineout.summary())
            if guess is not None:
                starts.append(self._unit_point(space, guess))
        starts.extend(space.sample_unit(self.n_starts, seed=self.seed))
        return starts

    @staticmethod
    def _unit_point(
        space: ParameterSpace, parameters: Mapping[str, float]
    ) -> np.ndarray:
        theta = space.clip(space.from_dict(parameters))
        return np.clip(space.to_unit(theta), 0.0, 1.0)

    def _minimize(self, objective, u0: np.ndarray, dimension: int) -> OptimizeResult:
        bounds = [(0.0, 1.0)] * dimension
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                return minimize(
                    objective,
                    u0,
                    method=self.method,
                    bounds=bounds,
                    options=self.options,
                )
            except ValueError:  # e.g. NaN in the line search
                return OptimizeResult(
                    x=u0, fun=objective(u0), success=False, message="optimizer raised"
                )


# -------------------------------------------------------------- comparisons
def fit_family(
    lineout: Lineout, family: ProfileFamily, scheme: FittingScheme | None = None
) -> FitResult:
    """Fit one family; the default scheme is :class:`MultiStartLocalFit`."""
    scheme = MultiStartLocalFit() if scheme is None else scheme
    return scheme.fit(lineout, family)


@dataclass(frozen=True)
class FamilyComparison:
    """Every family's fit to one lineout, ranked best-first."""

    lineout: Lineout = field(repr=False)
    results: tuple[FitResult, ...]
    rank_by: str

    @property
    def best(self) -> FitResult:
        return self.results[0]

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def table(self, max_rows: int | None = None) -> str:
        """A fixed-width text table of the ranked metrics."""
        rows = self.results if max_rows is None else self.results[:max_rows]
        header = (
            f"{'rank':>4}  {'family':<44} {'k':>3} {'nrmse':>9} {'max|err|':>9} "
            f"{'int.rel.err':>11} {'R^2':>8} {'BIC':>12} {'time[s]':>8}"
        )
        lines = [header, "-" * len(header)]
        for index, result in enumerate(rows, start=1):
            g = result.goodness
            flag = "" if result.success else " !"
            lines.append(
                f"{index:>4}  {result.family_name + flag:<44} {g.n_parameters:>3} "
                f"{g.nrmse:>9.4f} {g.max_abs_error:>9.4f} {g.integrated_relative_error:>11.4f} "
                f"{g.r_squared:>8.4f} {g.bic:>12.1f} {result.elapsed_seconds:>8.2f}"
            )
        return "\n".join(lines)

    def to_dict(self, include_profiles: bool = True) -> dict[str, Any]:
        return {
            "conditions": self.lineout.conditions.to_dict(),
            "source": self.lineout.source,
            "density_units": self.lineout.density_units,
            "n_points": self.lineout.n_points,
            "z_extent_m": list(self.lineout.z_extent),
            "rank_by": self.rank_by,
            "results": [result.to_dict(include_profiles) for result in self.results],
        }


def compare_families(
    lineout: Lineout,
    families: Iterable[ProfileFamily] | Sequence[str] | None = None,
    scheme: FittingScheme | None = None,
    rank_by: str = DEFAULT_RANKING_METRIC,
) -> FamilyComparison:
    """Fit each family to ``lineout`` and rank them by ``rank_by``.

    ``families`` may be family objects, registry names, or ``None`` for the
    whole default registry.
    """
    scheme = MultiStartLocalFit() if scheme is None else scheme
    resolved = _resolve_families(families)
    results = [scheme.fit(lineout, family) for family in resolved]
    return FamilyComparison(
        lineout=lineout,
        results=tuple(rank_results(results, by=rank_by)),
        rank_by=rank_by,
    )


def fit_lineouts(
    lineouts: Iterable[Lineout],
    family: ProfileFamily,
    scheme: FittingScheme | None = None,
) -> list[FitResult]:
    """Fit one family to several lineouts.

    The returned results pair each lineout's :class:`LineoutConditions` with a
    parameter vector in the same :class:`ParameterSpace` ordering, which is the
    raw material for learning ``conditions -> parameters`` later.
    """
    scheme = MultiStartLocalFit() if scheme is None else scheme
    return [scheme.fit(lineout, family) for lineout in lineouts]


def _resolve_families(
    families: Iterable[ProfileFamily] | Sequence[str] | None,
) -> list[ProfileFamily]:
    if families is None:
        return get_families()
    items = list(families)
    if all(isinstance(item, str) for item in items):
        return get_families(items)
    if all(isinstance(item, ProfileFamily) for item in items):
        return items
    raise TypeError("families must be all names or all ProfileFamily objects")
