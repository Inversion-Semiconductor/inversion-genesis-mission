"""Bounded parameter vectors and their unit-cube representation.

Optimizers work on ``u`` in ``[0, 1]**n``; families work on physical values
``theta``. :class:`ParameterSpace` converts between the two, applying a log
transform to scale-like parameters so that widths spanning orders of magnitude
are searched evenly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class ParameterSpec:
    """One bounded scalar parameter.

    Args:
        name: Parameter name; unique within a family.
        lower: Lower bound (inclusive) in physical units.
        upper: Upper bound (inclusive) in physical units.
        log_scale: Search uniformly in ``log(value)`` instead of ``value``.
            Requires ``lower > 0``.
        unit: Unit label for reports (``"m"``, ``"m^-1"``, ``""``).
    """

    name: str
    lower: float
    upper: float
    log_scale: bool = False
    unit: str = ""

    def __post_init__(self) -> None:
        lower, upper = float(self.lower), float(self.upper)
        if not (np.isfinite(lower) and np.isfinite(upper)) or lower >= upper:
            raise ValueError(
                f"{self.name}: bounds must be finite and increasing, got ({lower}, {upper})"
            )
        if self.log_scale and lower <= 0.0:
            raise ValueError(f"{self.name}: log-scale parameters need lower > 0")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_unit(self, value: np.ndarray) -> np.ndarray:
        if self.log_scale:
            return (np.log(value) - np.log(self.lower)) / (
                np.log(self.upper) - np.log(self.lower)
            )
        return (value - self.lower) / (self.upper - self.lower)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        if self.log_scale:
            return np.exp(
                np.log(self.lower) + u * (np.log(self.upper) - np.log(self.lower))
            )
        return self.lower + u * (self.upper - self.lower)


@dataclass(frozen=True)
class ParameterSpace:
    """An ordered collection of :class:`ParameterSpec` with vector transforms."""

    specs: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        specs = tuple(self.specs)
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names: {names}")
        if not specs:
            raise ValueError("a parameter space needs at least one parameter")
        object.__setattr__(self, "specs", specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def dimension(self) -> int:
        return len(self.specs)

    @property
    def lower(self) -> np.ndarray:
        return np.array([spec.lower for spec in self.specs])

    @property
    def upper(self) -> np.ndarray:
        return np.array([spec.upper for spec in self.specs])

    def __getitem__(self, name: str) -> ParameterSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def index(self, name: str) -> int:
        return self.names.index(name)

    # ------------------------------------------------------------ transforms
    def to_unit(self, theta: Sequence[float] | np.ndarray) -> np.ndarray:
        theta = self._as_vector(theta)
        return np.array(
            [spec.to_unit(value) for spec, value in zip(self.specs, theta, strict=True)]
        )

    def from_unit(self, u: Sequence[float] | np.ndarray) -> np.ndarray:
        u = self._as_vector(u)
        return np.array(
            [spec.from_unit(value) for spec, value in zip(self.specs, u, strict=True)]
        )

    def clip(self, theta: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.clip(self._as_vector(theta), self.lower, self.upper)

    def contains(self, theta: Sequence[float] | np.ndarray) -> bool:
        theta = self._as_vector(theta)
        return bool(np.all(theta >= self.lower) and np.all(theta <= self.upper))

    def center(self) -> np.ndarray:
        """The physical point at the middle of the unit cube."""
        return self.from_unit(np.full(self.dimension, 0.5))

    def sample_unit(
        self, n: int, seed: int | np.random.Generator | None = None
    ) -> np.ndarray:
        """``(n, dimension)`` Latin-hypercube points in the unit cube."""
        if n <= 0:
            return np.empty((0, self.dimension))
        return qmc.LatinHypercube(d=self.dimension, seed=seed).random(n)

    def sample(
        self, n: int, seed: int | np.random.Generator | None = None
    ) -> np.ndarray:
        return np.array([self.from_unit(u) for u in self.sample_unit(n, seed)])

    # ------------------------------------------------------------ dict forms
    def as_dict(self, theta: Sequence[float] | np.ndarray) -> dict[str, float]:
        theta = self._as_vector(theta)
        return {
            name: float(value) for name, value in zip(self.names, theta, strict=True)
        }

    def from_dict(self, values: Mapping[str, float]) -> np.ndarray:
        missing = set(self.names) - set(values)
        if missing:
            raise KeyError(f"missing parameters: {sorted(missing)}")
        return np.array([float(values[name]) for name in self.names])

    def _as_vector(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"expected a vector of length {self.dimension}, got shape {vector.shape}"
            )
        return vector
