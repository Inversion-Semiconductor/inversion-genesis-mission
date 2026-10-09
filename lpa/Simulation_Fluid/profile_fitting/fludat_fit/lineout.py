"""Density lineouts along ``z`` at fixed physical conditions.

A :class:`Lineout` is the thing a profile family is fitted against: one
``density(z)`` curve taken from a ``fludat_proc`` density cube at a transverse
position and backing pressure. The conditions travel with the lineout in a
:class:`LineoutConditions` record so that fitted parameters can later be
related back to ``(x_mm, pressure_bar, angle_deg)``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from fludat_proc.interpolate_density import (
        DensityInterpolation,
        InterpolationMethod,
        build_density_interpolation,
    )
except ImportError:  # sibling checkout without an installed fludat_proc
    _POSTPROC_DIR = Path(__file__).resolve().parents[2] / "postproc"
    if not (_POSTPROC_DIR / "fludat_proc").is_dir():
        raise
    sys.path.insert(0, str(_POSTPROC_DIR))
    from fludat_proc.interpolate_density import (
        DensityInterpolation,
        InterpolationMethod,
        build_density_interpolation,
    )

DEFAULT_CUTOFF_RATIO = 5.0e-3
"""Fraction of the peak below which a lineout is considered vacuum."""


@dataclass(frozen=True)
class LineoutConditions:
    """The three physical parameters that select one lineout from a density cube.

    Args:
        x_mm: Transverse offset [mm] of the lineout line where it crosses ``z = 0``.
        pressure_bar: Backing pressure [bar].
        angle_deg: Angle [deg] of the lineout line to the ``z`` axis in the
            ``(z, x)`` plane, positive towards ``+x``: the line is
            ``z_m(t) = cos(angle) t``, ``x_mm(t) = x_mm + 1000 sin(angle) t`` with the
            path parameter ``t`` in metres, as in ``InterpolateFromH5Profile``'s
            ``lineout_axis``. ``0`` is a plain lineout along ``z`` at fixed ``x``.
    """

    x_mm: float
    pressure_bar: float
    angle_deg: float = 0.0

    FIELD_NAMES = ("x_mm", "pressure_bar", "angle_deg")

    def __post_init__(self) -> None:
        for name in self.FIELD_NAMES:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            object.__setattr__(self, name, value)

    def as_vector(self) -> np.ndarray:
        """``[x_mm, pressure_bar, angle_deg]``."""
        return np.array([self.x_mm, self.pressure_bar, self.angle_deg], dtype=float)

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.FIELD_NAMES}

    def label(self) -> str:
        return f"x={self.x_mm:g} mm, p={self.pressure_bar:g} bar, angle={self.angle_deg:g} deg"


@dataclass(frozen=True)
class LineoutSummary:
    """Scale information a family uses to bound its parameters.

    All lengths are in metres. ``support`` is the ``z`` interval where the
    density is at least ``cutoff_ratio`` of the peak; ``fwhm`` is the width of
    the half-maximum crossing around the peak.
    """

    z_min: float
    z_max: float
    peak: float
    z_peak: float
    centroid: float
    fwhm: float
    support: tuple[float, float]
    cutoff_ratio: float

    @property
    def width(self) -> float:
        return self.z_max - self.z_min

    @property
    def support_width(self) -> float:
        return self.support[1] - self.support[0]

    @property
    def z_scale(self) -> float:
        """A length scale of the jet, never zero: the support width or the window."""
        return self.support_width if self.support_width > 0.0 else self.width


@dataclass(frozen=True)
class Lineout:
    """One ``density(z)`` curve at fixed conditions.

    Args:
        z: Strictly increasing sample positions [m].
        density: Density samples in ``density_units``.
        density_units: Units string as stored in the density cube.
        conditions: The physical parameters the lineout was taken at.
        source: Nozzle name or file the lineout came from.
        weights: Optional non-negative fit weights, one per sample.
    """

    z: np.ndarray
    density: np.ndarray
    density_units: str
    conditions: LineoutConditions
    source: str | None = None
    weights: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=np.float64)
        density = np.asarray(self.density, dtype=np.float64)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "density", density)
        if z.ndim != 1 or z.size < 2:
            raise ValueError("z must be one-dimensional with at least two samples")
        if density.shape != z.shape:
            raise ValueError(
                f"density has shape {density.shape}; expected {z.shape} like z"
            )
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(density)):
            raise ValueError("z and density must be finite")
        if np.any(np.diff(z) <= 0.0):
            raise ValueError("z must be strictly increasing")
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=np.float64)
            if weights.shape != z.shape or np.any(weights < 0.0):
                raise ValueError("weights must match z and be non-negative")
            if not np.any(weights > 0.0):
                raise ValueError("at least one weight must be positive")
            object.__setattr__(self, "weights", weights)

    # ------------------------------------------------------------------ basics
    @property
    def n_points(self) -> int:
        return int(self.z.size)

    @property
    def z_extent(self) -> tuple[float, float]:
        return float(self.z[0]), float(self.z[-1])

    @property
    def peak(self) -> float:
        return float(self.density.max())

    @property
    def z_peak(self) -> float:
        return float(self.z[int(np.argmax(self.density))])

    def effective_weights(self) -> np.ndarray:
        return np.ones_like(self.z) if self.weights is None else self.weights

    # ---------------------------------------------------------------- derived
    def support(
        self, cutoff_ratio: float = DEFAULT_CUTOFF_RATIO
    ) -> tuple[float, float]:
        """The ``z`` interval where density is at least ``cutoff_ratio * peak``."""
        if self.peak <= 0.0:
            return self.z_extent
        above = np.flatnonzero(self.density >= cutoff_ratio * self.peak)
        return float(self.z[above[0]]), float(self.z[above[-1]])

    def fwhm(self) -> float:
        """Full width at half maximum around the peak sample, by linear crossing."""
        if self.peak <= 0.0:
            return self.z_extent[1] - self.z_extent[0]
        half = 0.5 * self.peak
        i_peak = int(np.argmax(self.density))
        left = self.z[0]
        for i in range(i_peak, 0, -1):
            if self.density[i - 1] < half <= self.density[i]:
                left = _crossing(
                    self.z[i - 1], self.z[i], self.density[i - 1], self.density[i], half
                )
                break
        right = self.z[-1]
        for i in range(i_peak, self.z.size - 1):
            if self.density[i] >= half > self.density[i + 1]:
                right = _crossing(
                    self.z[i], self.z[i + 1], self.density[i], self.density[i + 1], half
                )
                break
        return float(right - left)

    def centroid(self) -> float:
        """Density-weighted centre; the peak position if the lineout has no mass."""
        positive = np.clip(self.density, 0.0, None)
        mass = np.trapezoid(positive, self.z)
        if mass <= 0.0:
            return self.z_peak
        return float(np.trapezoid(positive * self.z, self.z) / mass)

    def summary(self, cutoff_ratio: float = DEFAULT_CUTOFF_RATIO) -> LineoutSummary:
        return LineoutSummary(
            z_min=self.z_extent[0],
            z_max=self.z_extent[1],
            peak=self.peak,
            z_peak=self.z_peak,
            centroid=self.centroid(),
            fwhm=self.fwhm(),
            support=self.support(cutoff_ratio),
            cutoff_ratio=cutoff_ratio,
        )

    # ------------------------------------------------------------ transforms
    def trimmed(
        self,
        cutoff_ratio: float = DEFAULT_CUTOFF_RATIO,
        padding_fraction: float = 0.25,
    ) -> Lineout:
        """Crop to the jet: the support at ``cutoff_ratio`` plus padding on each side.

        The padding is a fraction of the support width and keeps some vacuum in
        the fit so that profiles are also required to vanish outside the jet.
        """
        lower, upper = self.support(cutoff_ratio)
        pad = padding_fraction * (upper - lower)
        keep = (self.z >= lower - pad) & (self.z <= upper + pad)
        if keep.sum() < 2:
            return self
        return self._subset(keep)

    def cropped(self, z_bounds: tuple[float, float]) -> Lineout:
        """Keep the samples inside ``z_bounds`` [m]."""
        lower, upper = z_bounds
        keep = (self.z >= lower) & (self.z <= upper)
        if keep.sum() < 2:
            raise ValueError(f"fewer than two samples inside z bounds {z_bounds}")
        return self._subset(keep)

    def with_weights(self, weights: np.ndarray | None) -> Lineout:
        return replace(self, weights=weights)

    def resampled(self, n_points: int) -> Lineout:
        """Linearly resample onto ``n_points`` uniform samples over the same extent.

        Returns ``self`` unchanged when it already has at most ``n_points`` samples.
        """
        if n_points < 2:
            raise ValueError("n_points must be at least two")
        if self.n_points <= n_points:
            return self
        z = np.linspace(self.z[0], self.z[-1], n_points)
        return replace(
            self,
            z=z,
            density=np.interp(z, self.z, self.density),
            weights=(
                None if self.weights is None else np.interp(z, self.z, self.weights)
            ),
        )

    def _subset(self, keep: np.ndarray) -> Lineout:
        return replace(
            self,
            z=self.z[keep],
            density=self.density[keep],
            weights=None if self.weights is None else self.weights[keep],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "z_m": self.z.tolist(),
            "density": self.density.tolist(),
            "density_units": self.density_units,
            "conditions": self.conditions.to_dict(),
            "source": self.source,
        }


def _crossing(z0: float, z1: float, d0: float, d1: float, level: float) -> float:
    return z0 + (level - d0) / (d1 - d0) * (z1 - z0)


# ---------------------------------------------------------------- extraction
def extract_lineout(
    field: DensityInterpolation,
    conditions: LineoutConditions | Mapping[str, float],
    z_values: np.ndarray | None = None,
) -> Lineout:
    """Take one lineout along ``z`` at fixed ``x`` from an interpolated density cube.

    ``z_values`` defaults to the cube's tabulated ``z`` grid. Points outside the
    cube evaluate to zero density, as in ``fludat_proc``. Oblique lineouts
    (``angle_deg != 0``) need :class:`fludat_fit.dataset.NozzleDataset`.
    """
    if not isinstance(conditions, LineoutConditions):
        conditions = LineoutConditions(**dict(conditions))
    if conditions.angle_deg != 0.0:
        raise ValueError(
            "extract_lineout handles angle_deg=0 only; use NozzleDataset.lineout for "
            "oblique lines"
        )
    z = field.z if z_values is None else np.asarray(z_values, dtype=np.float64)
    density = field.interpolate_along_z(z, conditions.x_mm, conditions.pressure_bar)
    return Lineout(
        z=z,
        density=density,
        density_units=field.density_units,
        conditions=conditions,
        source=field.geometry,
    )


def extract_lineouts(
    field: DensityInterpolation,
    conditions: Iterable[LineoutConditions | Mapping[str, float]],
    z_values: np.ndarray | None = None,
) -> list[Lineout]:
    return [extract_lineout(field, item, z_values) for item in conditions]


def load_lineout(
    hdf5_path: str | Path,
    x_mm: float,
    pressure_bar: float,
    *,
    method: InterpolationMethod = "linear",
    z_values: np.ndarray | None = None,
) -> Lineout:
    """Load a density cube and extract one lineout along ``z`` from it."""
    field = build_density_interpolation(hdf5_path, method=method)
    return extract_lineout(field, LineoutConditions(x_mm, pressure_bar), z_values)
