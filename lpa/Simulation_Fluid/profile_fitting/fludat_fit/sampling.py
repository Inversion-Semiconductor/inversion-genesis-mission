"""Boxes of physical conditions and how to sample them."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import qmc

from .dataset import NozzleDataset
from .lineout import LineoutConditions

Sampling = Literal["lhs", "random", "grid"]
SAMPLING_METHODS: tuple[Sampling, ...] = ("lhs", "random", "grid")
CONDITION_AXES = ("x_mm", "pressure_bar", "angle_deg")
AXIS_LABELS = {
    "x_mm": "x [mm]",
    "pressure_bar": "pressure [bar]",
    "angle_deg": "angle [deg]",
}


@dataclass(frozen=True)
class ConditionRanges:
    """The box of physical conditions to sample.

    Each axis is a ``(lower, upper)`` pair; a pair with equal values fixes that
    axis. The angle is the lineout line's angle to the ``z`` axis [deg].
    """

    x_mm: tuple[float, float]
    pressure_bar: tuple[float, float]
    angle_deg: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_dataset(
        cls,
        dataset: NozzleDataset,
        *,
        x_range: Sequence[float] | None = None,
        pressure_range: Sequence[float] | None = None,
        angle_range: Sequence[float] | None = None,
    ) -> ConditionRanges:
        """Ranges inside the dataset extents.

        ``None`` selects the full x or pressure extent; the angle defaults to a
        fixed ``0`` (axial lineouts). Ranges reaching outside are clamped with a
        warning.
        """
        return cls(
            x_mm=_validated_range(x_range, dataset.x_extent, "x"),
            pressure_bar=_validated_range(
                pressure_range, dataset.pressure_extent, "pressure"
            ),
            angle_deg=(
                (0.0, 0.0)
                if angle_range is None
                else _validated_range(angle_range, dataset.angle_extent, "angle")
            ),
        )

    @property
    def axes(self) -> dict[str, tuple[float, float]]:
        return {
            "x_mm": self.x_mm,
            "pressure_bar": self.pressure_bar,
            "angle_deg": self.angle_deg,
        }

    @property
    def varying_axes(self) -> list[str]:
        return [name for name, (lower, upper) in self.axes.items() if lower < upper]

    def sample(
        self, n: int, sampling: Sampling = "lhs", seed: int | None = 0
    ) -> list[LineoutConditions]:
        """``n`` condition points (a grid rounds ``n`` up to a full lattice)."""
        if n < 1:
            raise ValueError("n must be at least one")
        varying = self.varying_axes
        d = len(varying)
        if d == 0:
            unit = np.zeros((1, 0))
        elif sampling == "lhs":
            unit = qmc.LatinHypercube(d=d, seed=seed).random(n)
        elif sampling == "random":
            unit = np.random.default_rng(seed).random((n, d))
        elif sampling == "grid":
            per_axis = max(2, math.ceil(n ** (1.0 / d)))
            axes = np.meshgrid(*[np.linspace(0.0, 1.0, per_axis)] * d, indexing="ij")
            unit = np.column_stack([axis.ravel() for axis in axes])
        else:
            raise ValueError(f"unknown sampling {sampling!r}")

        points = []
        for row in unit:
            values: dict[str, float] = {}
            column = 0
            for name, (lower, upper) in self.axes.items():
                if name in varying:
                    values[name] = float(lower + row[column] * (upper - lower))
                    column += 1
                else:
                    values[name] = float(lower)
            points.append(LineoutConditions(**values))
        return points

    def normalize(self, conditions: np.ndarray) -> np.ndarray:
        """Map ``(N, 3)`` condition vectors to ``[0, 1]`` per axis; fixed axes give 0."""
        conditions = np.atleast_2d(np.asarray(conditions, dtype=np.float64))
        lower = np.array([bounds[0] for bounds in self.axes.values()])
        upper = np.array([bounds[1] for bounds in self.axes.values()])
        width = upper - lower
        safe = np.where(width > 0.0, width, 1.0)
        return np.where(width > 0.0, (conditions - lower) / safe, 0.0)

    def denormalize(self, features: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(np.asarray(features, dtype=np.float64))
        lower = np.array([bounds[0] for bounds in self.axes.values()])
        upper = np.array([bounds[1] for bounds in self.axes.values()])
        return lower + features * (upper - lower)

    def to_dict(self) -> dict[str, Any]:
        return {name: list(bounds) for name, bounds in self.axes.items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConditionRanges:
        return cls(**{name: tuple(payload[name]) for name in CONDITION_AXES})


def _validated_range(
    requested: Sequence[float] | None, extent: tuple[float, float], axis: str
) -> tuple[float, float]:
    """The requested range clamped to the dataset extent (with a warning)."""
    if requested is None:
        return (float(extent[0]), float(extent[1]))
    lower, upper = sorted(float(value) for value in requested)
    if upper < extent[0] or lower > extent[1]:
        raise ValueError(
            f"{axis} range [{lower:g}, {upper:g}] does not overlap the dataset extent "
            f"[{extent[0]:g}, {extent[1]:g}]"
        )
    clamped = (max(lower, extent[0]), min(upper, extent[1]))
    if clamped != (lower, upper):
        warnings.warn(
            f"{axis} range [{lower:g}, {upper:g}] clamped to the dataset extent "
            f"[{clamped[0]:g}, {clamped[1]:g}]",
            stacklevel=3,
        )
    return clamped
