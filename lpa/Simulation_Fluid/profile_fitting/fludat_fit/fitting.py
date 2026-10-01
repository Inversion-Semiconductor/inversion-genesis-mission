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
from typing import Any, NamedTuple, Protocol, runtime_checkable

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


MAX_DEVIATION_EXPONENT = 8
"""Even power whose weighted power-mean stands in for ``max(|residual|)`` in
:class:`FitObjective`, smoothly (see there for why)."""


class _Evaluated(NamedTuple):
    """The built profile's contribution to :class:`FitObjective` at one unit point."""

    amplitude: float
    relative: np.ndarray
    z_extent: tuple[float, float]


class FitObjective:
    """The local-fit objective for one lineout and family, on a unit cube.

    ``__call__(u)`` returns three terms added together:

    * the weighted sum of squared residuals normalised by the weighted sum of
      squared data, so it is unit-free (this alone was the whole objective
      before the other two terms were added);
    * ``max_deviation_weight`` times the squared, weighted
      ``MAX_DEVIATION_EXPONENT``-power mean of ``|residual| / peak``: a smooth
      stand-in for the squared *relative* worst-point error
      ``(max(|residual|) / peak) ** 2``, which it approaches as the exponent
      grows. The exact max is avoided because it is only piecewise
      differentiable (which point attains it can change discontinuously as
      parameters move), and ``MultiStartLocalFit`` estimates gradients by
      finite differences; that combination made ``scipy.optimize.minimize``'s
      line search fail intermittently in practice.
    * ``extent_weight`` times the squared excess of the built profile's own
      ``get_z_extent()`` width over ``allowed_extent_ratio`` times the fit
      window's width (a hinge: zero unless that ratio is exceeded).

    The power-mean's degree does not change the calibration
    ``max_deviation_weight = 1.0`` promises: if ``model - data`` were a
    nonzero constant ``c`` everywhere over a flat ``data`` of value ``d``,
    every term in the power mean is identical, so it equals ``|c|`` exactly
    (for any exponent, and independent of weights) and both terms equal
    ``(c / d) ** 2``. A pure sum-of-squares fit can leave an isolated,
    badly-missed region far from the bulk of the data (its contribution to
    the *sum* of squares is small next to everywhere else); this term
    penalises that worst region directly.

    The extent term addresses a different failure mode: a family like
    ``GeneralizedLorentzianSum`` can fit the data inside the window perfectly
    while one term's shape parameters give it an enormous, numerically
    negligible tail, since nothing in the first two terms is evaluated outside
    the window. ``get_z_extent()`` is the same quantity
    :meth:`~fludat_fit.families.ProfileFamily.build` (and downstream plotting)
    report as the profile's support, so a fit this term accepts will not later
    turn out to carry a wildly disproportionate tail. Unlike
    ``max_deviation_weight``, there is no flat-data identity calibrating
    ``allowed_extent_ratio``; a fit window is already padded around the jet's
    support, so a ratio of ``2.0`` (built profile at most twice as wide as
    the window it was compared against) is a threshold choice, not a
    derivation.

    With the amplitude eliminated in closed form (variable projection), this
    is the exact quantity :class:`MultiStartLocalFit` minimises, and the
    *direct* loss a learned ``conditions -> parameters`` model is judged and
    trained on.

    Args:
        lineout: The data.
        family: The profile family.
        space: The unit cube's physical bounds; default: the family's space for
            this lineout. A learned model uses one fixed space for a whole dataset.
        fit_amplitude: If ``False`` the amplitude is fixed to the lineout peak.
        max_deviation_weight: Weight of the worst-point term relative to the
            sum-of-squares term; ``0.0`` recovers the original sum-of-squares-only
            objective.
        extent_weight: Weight of the excess-extent term; ``0.0`` disables it.
        allowed_extent_ratio: How many multiples of the fit window's width the
            built profile's ``get_z_extent()`` may reach before the extent term
            becomes nonzero.
    """

    def __init__(
        self,
        lineout: Lineout,
        family: ProfileFamily,
        space: ParameterSpace | None = None,
        *,
        fit_amplitude: bool = True,
        max_deviation_weight: float = 1.0,
        extent_weight: float = 1.0,
        allowed_extent_ratio: float = 2.0,
    ) -> None:
        if max_deviation_weight < 0.0:
            raise ValueError("max_deviation_weight must be non-negative")
        if extent_weight < 0.0:
            raise ValueError("extent_weight must be non-negative")
        if allowed_extent_ratio <= 0.0:
            raise ValueError("allowed_extent_ratio must be positive")
        self.lineout = lineout
        self.family = family
        self.space = space or family.parameter_space(lineout.summary())
        self.fixed_amplitude = None if fit_amplitude else lineout.peak
        self.max_deviation_weight = float(max_deviation_weight)
        self.extent_weight = float(extent_weight)
        self.allowed_extent_ratio = float(allowed_extent_ratio)
        self._weights = lineout.effective_weights()
        self._sum_weights = float(np.sum(self._weights))
        self._normalisation = float(np.sum(self._weights * lineout.density**2)) or 1.0
        self._peak = lineout.peak if lineout.peak > 0.0 else 1.0
        self._window_width = float(lineout.z[-1] - lineout.z[0])
        self.evaluations = 0

    @property
    def dimension(self) -> int:
        return self.space.dimension

    @property
    def n_parameters(self) -> int:
        """Shape parameters plus the amplitude when it is fitted."""
        return self.space.dimension + (1 if self.fixed_amplitude is None else 0)

    def evaluate(self, u: np.ndarray) -> _Evaluated | None:
        """The built profile's amplitude, relative density and extent; ``None`` if invalid."""
        parameters = self.space.as_dict(self.space.from_unit(np.clip(u, 0.0, 1.0)))
        try:
            profile = self.family.build(parameters)
            relative = np.asarray(
                profile.build_density_function()(
                    self.lineout.z, np.zeros_like(self.lineout.z)
                ),
                dtype=np.float64,
            )
            z_extent = profile.get_z_extent()
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return None
        if not np.all(np.isfinite(relative)) or not np.all(np.isfinite(z_extent)):
            return None
        amplitude = (
            optimal_amplitude(relative, self.lineout.density, self._weights)
            if self.fixed_amplitude is None
            else self.fixed_amplitude
        )
        return _Evaluated(amplitude, relative, z_extent)

    def __call__(self, u: np.ndarray) -> float:
        self.evaluations += 1
        evaluated = self.evaluate(u)
        if evaluated is None:
            return INVALID_OBJECTIVE
        amplitude, relative, z_extent = evaluated
        residual = amplitude * relative - self.lineout.density
        total = float(np.sum(self._weights * residual * residual)) / self._normalisation
        if self.max_deviation_weight > 0.0:
            relative_residual = np.abs(residual) / self._peak
            power_mean = (
                float(np.sum(self._weights * relative_residual**MAX_DEVIATION_EXPONENT))
                / self._sum_weights
            ) ** (1.0 / MAX_DEVIATION_EXPONENT)
            total += self.max_deviation_weight * power_mean**2
        if self.extent_weight > 0.0:
            extent_ratio = (z_extent[1] - z_extent[0]) / self._window_width
            excess = max(0.0, extent_ratio - self.allowed_extent_ratio)
            total += self.extent_weight * excess * excess
        return total

    def result(
        self,
        u: np.ndarray,
        *,
        scheme: str,
        success: bool = True,
        message: str = "",
        n_starts: int = 0,
        n_function_evaluations: int | None = None,
        elapsed_seconds: float = 0.0,
    ) -> FitResult:
        """Package unit point ``u`` as a :class:`FitResult` with its goodness of fit."""
        u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        theta = self.space.from_unit(u)
        evaluated = self.evaluate(u)
        if evaluated is None:
            amplitude, model = 0.0, np.zeros_like(self.lineout.density)
            success = False
        else:
            amplitude, relative, _z_extent = evaluated
            model = amplitude * relative
        goodness = GoodnessOfFit.compute(
            self.lineout.z,
            self.lineout.density,
            model,
            n_parameters=self.n_parameters,
            weights=self.lineout.weights,
        )
        return FitResult(
            family=self.family,
            lineout=self.lineout,
            space=self.space,
            theta=theta,
            amplitude=amplitude,
            goodness=goodness,
            scheme=scheme,
            success=success,
            message=message,
            n_starts=n_starts,
            n_function_evaluations=(
                self.evaluations
                if n_function_evaluations is None
                else n_function_evaluations
            ),
            elapsed_seconds=elapsed_seconds,
        )


class MultiStartLocalFit:
    """Bounded local optimisation from several starting points.

    Each start is a point of a Latin-hypercube sample in the family's unit cube;
    the family's :meth:`~fludat_fit.families.ProfileFamily.initial_guess` and
    any warm start are added. The objective is :class:`FitObjective`.

    Args:
        n_starts: Number of sampled starting points.
        method: Any bounded ``scipy.optimize.minimize`` method (``L-BFGS-B``,
            ``Powell``, ``TNC``, ``Nelder-Mead``, ...).
        seed: Seed for the start sample.
        include_initial_guess: Add the family heuristic as one more start.
        options: Extra solver options; merged over the per-method defaults.
        fit_amplitude: If ``False`` the amplitude is fixed to the lineout peak.
        max_deviation_weight: Weight of the worst-point term in
            :class:`FitObjective`; see there for the calibration behind its
            default of ``1.0``. ``0.0`` recovers a plain sum-of-squares fit.
        extent_weight: Weight of :class:`FitObjective`'s excess-extent term;
            ``0.0`` disables it.
        allowed_extent_ratio: See :class:`FitObjective`.
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
        max_deviation_weight: float = 1.0,
        extent_weight: float = 1.0,
        allowed_extent_ratio: float = 2.0,
    ) -> None:
        if n_starts < 0:
            raise ValueError("n_starts must be non-negative")
        self.n_starts = int(n_starts)
        self.method = method
        self.seed = seed
        self.include_initial_guess = include_initial_guess
        self.options = {**self._DEFAULT_OPTIONS.get(method, {}), **(options or {})}
        self.fit_amplitude = fit_amplitude
        self.max_deviation_weight = max_deviation_weight
        self.extent_weight = extent_weight
        self.allowed_extent_ratio = allowed_extent_ratio
        self.warm_starts: dict[str, Mapping[str, float]] = {}
        """Family name -> parameters used as one more start (e.g. the previous fit)."""

    # -- public --------------------------------------------------------------
    def fit(
        self,
        lineout: Lineout,
        family: ProfileFamily,
        *,
        space: ParameterSpace | None = None,
        warm_start: Mapping[str, float] | np.ndarray | None = None,
    ) -> FitResult:
        """Fit ``family`` to ``lineout``.

        Args:
            space: Unit-cube bounds to optimise in; default: the family's space
                for this lineout.
            warm_start: One more starting point, as physical parameters by name
                or as a unit-cube vector (used by learned-model refinement).
        """
        started = time.perf_counter()
        objective = FitObjective(
            lineout,
            family,
            space,
            fit_amplitude=self.fit_amplitude,
            max_deviation_weight=self.max_deviation_weight,
            extent_weight=self.extent_weight,
            allowed_extent_ratio=self.allowed_extent_ratio,
        )
        starts = self._starts(objective.space, lineout, family, warm_start)
        if not starts:
            raise ValueError(
                "no starting points: increase n_starts or give a warm start"
            )

        best: OptimizeResult | None = None
        for u0 in starts:
            result = self._minimize(objective, u0, objective.dimension)
            if best is None or result.fun < best.fun:
                best = result
        assert best is not None
        return objective.result(
            best.x,
            scheme=f"{self.name}[{self.method}]",
            success=bool(best.success) and best.fun < INVALID_OBJECTIVE,
            message=str(best.message),
            n_starts=len(starts),
            elapsed_seconds=time.perf_counter() - started,
        )

    # -- internals -----------------------------------------------------------
    def _starts(
        self,
        space: ParameterSpace,
        lineout: Lineout,
        family: ProfileFamily,
        warm_start: Mapping[str, float] | np.ndarray | None,
    ) -> list[np.ndarray]:
        starts: list[np.ndarray] = []
        if warm_start is not None:
            if isinstance(warm_start, Mapping):
                starts.append(self._unit_point(space, warm_start))
            else:
                starts.append(
                    np.clip(np.asarray(warm_start, dtype=np.float64), 0.0, 1.0)
                )
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
