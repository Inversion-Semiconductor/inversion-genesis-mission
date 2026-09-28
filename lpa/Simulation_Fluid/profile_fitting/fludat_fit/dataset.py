"""A nozzle dataset: one density cube queried along oblique lines in the ``(z, x)`` plane.

The three physical conditions of a lineout are the transverse offset ``x_mm``,
the backing pressure ``pressure_bar`` and the angle ``angle_deg`` of the
lineout line to the ``z`` axis. The line is the affine path

    z_m(t)  = cos(angle) * t
    x_mm(t) = x_mm + 1000 * sin(angle) * t

with the path parameter ``t`` in metres, which is exactly the ``lineout_axis``
form of ``inversion_fbpic``'s ``InterpolateFromH5Profile``. Lineouts follow that
class's conventions: the path is clipped to the part inside the cube, sampled
with one point per participating grid point, and the density is trilinear in
``(z, x, pressure)``. :meth:`NozzleDataset.h5_profile_kwargs` gives the matching
``InterpolateFromH5Profile`` constructor arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .lineout import (
    DensityInterpolation,
    InterpolationMethod,
    Lineout,
    LineoutConditions,
    build_density_interpolation,
)

MM_PER_M = 1000.0
MAX_ANGLE_DEG = 89.0
"""Lines steeper than this run (almost) along ``x`` and have no longitudinal extent."""


@dataclass(frozen=True)
class LineoutPath:
    """The affine lineout line for one set of conditions, clipped to the cube.

    ``t`` runs from ``t_min`` to ``t_max`` [m]; ``z_m(t) = z_origin + z_coefficient * t``
    and ``x_mm(t) = x_origin + x_coefficient * t``.
    """

    z_origin: float
    z_coefficient: float
    x_origin: float
    x_coefficient: float
    t_min: float
    t_max: float
    n_points: int

    @property
    def is_axial(self) -> bool:
        """``True`` for a plain lineout along ``z`` at fixed ``x``."""
        return self.x_coefficient == 0.0

    def t_values(self) -> np.ndarray:
        return np.linspace(self.t_min, self.t_max, self.n_points)

    def z_m(self, t: np.ndarray) -> np.ndarray:
        return self.z_origin + self.z_coefficient * np.asarray(t, dtype=np.float64)

    def x_mm(self, t: np.ndarray) -> np.ndarray:
        return self.x_origin + self.x_coefficient * np.asarray(t, dtype=np.float64)

    def lineout_axis(self) -> dict[str, dict[str, float]]:
        """The ``InterpolateFromH5Profile.lineout_axis`` dictionary for this line."""
        return {
            "z_m": {"origin": self.z_origin, "coefficient": self.z_coefficient},
            "x_mm": {"origin": self.x_origin, "coefficient": self.x_coefficient},
        }


def _bracket(
    grid: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bracketing indices ``(i0, i1)`` and the weight of ``i1`` for sorted ``grid``."""
    values = np.asarray(values, dtype=np.float64)
    if grid.size == 1:
        zeros = np.zeros(values.shape, dtype=int)
        return zeros, zeros, np.zeros(values.shape)
    i1 = np.clip(np.searchsorted(grid, values, side="right"), 1, grid.size - 1)
    i0 = i1 - 1
    weight = (values - grid[i0]) / (grid[i1] - grid[i0])
    return i0, i1, np.clip(weight, 0.0, 1.0)


def trilinear_density(
    field: DensityInterpolation,
    z_m: np.ndarray,
    x_mm: np.ndarray,
    pressure_bar: float,
) -> np.ndarray:
    """Trilinear density at points ``(z_m[i], x_mm[i], pressure_bar)`` inside the cube.

    Identical to ``RegularGridInterpolator((z, x, pressure), density)`` with the
    linear method (the ``InterpolateFromH5Profile`` path) and, for points on the
    ``z`` grid, to ``fludat_proc.interpolate_along_z`` with ``method="linear"``.
    Points are clipped to the cube, so callers restrict the query to it.
    """
    z_m = np.asarray(z_m, dtype=np.float64)
    x_mm = np.asarray(x_mm, dtype=np.float64)
    iz0, iz1, wz = _bracket(field.z, z_m)
    ix0, ix1, wx = _bracket(field.x, x_mm)
    ip0, ip1, wp = _bracket(field.pressure, np.array([pressure_bar]))
    ip0, ip1, wp = int(ip0[0]), int(ip1[0]), float(wp[0])
    density = field.cube.density

    def slab(ip: int) -> np.ndarray:
        return (
            (1.0 - wz) * (1.0 - wx) * density[iz0, ix0, ip]
            + wz * (1.0 - wx) * density[iz1, ix0, ip]
            + (1.0 - wz) * wx * density[iz0, ix1, ip]
            + wz * wx * density[iz1, ix1, ip]
        )

    if wp == 0.0 or ip0 == ip1:
        return slab(ip0)
    return (1.0 - wp) * slab(ip0) + wp * slab(ip1)


class NozzleDataset:
    """One density cube, queried along oblique lineout lines at fixed pressure.

    Args:
        field: The interpolated cube. ``method="linear"`` uses the vectorised
            trilinear path; other methods fall back to the per-point
            ``fludat_proc`` interpolation, which is much slower.
        path: Where the cube was loaded from, if known (recorded in exports).
    """

    def __init__(self, field: DensityInterpolation, path: Path | None = None) -> None:
        self.field = field
        self.path = None if path is None else Path(path)

    @classmethod
    def load(
        cls, hdf5_path: str | Path, *, method: InterpolationMethod = "linear"
    ) -> NozzleDataset:
        return cls(
            build_density_interpolation(hdf5_path, method=method), Path(hdf5_path)
        )

    # ------------------------------------------------------------- extents
    @property
    def name(self) -> str:
        return self.field.geometry

    @property
    def density_units(self) -> str:
        return self.field.density_units

    @property
    def method(self) -> InterpolationMethod:
        return self.field.method

    @property
    def z(self) -> np.ndarray:
        return self.field.z

    @property
    def x_extent(self) -> tuple[float, float]:
        return self.field.x_extent

    @property
    def pressure_extent(self) -> tuple[float, float]:
        return self.field.pressure_extent

    @property
    def angle_extent(self) -> tuple[float, float]:
        """Angles with a longitudinal component: ``(-MAX_ANGLE_DEG, MAX_ANGLE_DEG)``."""
        return (-MAX_ANGLE_DEG, MAX_ANGLE_DEG)

    def default_conditions(self) -> LineoutConditions:
        """Lowest x and pressure, along ``z``."""
        return LineoutConditions(
            x_mm=self.x_extent[0], pressure_bar=self.pressure_extent[0], angle_deg=0.0
        )

    def contains(self, conditions: LineoutConditions) -> bool:
        x_lo, x_hi = self.x_extent
        p_lo, p_hi = self.pressure_extent
        return bool(
            x_lo <= conditions.x_mm <= x_hi
            and p_lo <= conditions.pressure_bar <= p_hi
            and abs(conditions.angle_deg) <= MAX_ANGLE_DEG
        )

    # ---------------------------------------------------------------- path
    def path_for(self, conditions: LineoutConditions) -> LineoutPath:
        """The lineout line through ``(z=0, x=x_mm)`` at ``angle_deg``, clipped to the cube.

        As in ``InterpolateFromH5Profile``, the path parameter range is the
        intersection of the ranges over which each varying coordinate stays inside
        its grid, and the sample count is the sum of the participating grid sizes
        (just the ``z`` grid for an axial line).
        """
        if abs(conditions.angle_deg) > MAX_ANGLE_DEG:
            raise ValueError(
                f"|angle_deg| must be at most {MAX_ANGLE_DEG:g}, got {conditions.angle_deg:g}"
            )
        x_lo, x_hi = self.x_extent
        if not x_lo <= conditions.x_mm <= x_hi:
            raise ValueError(
                f"x_mm={conditions.x_mm:g} is outside the cube range [{x_lo:g}, {x_hi:g}]"
            )
        angle = np.deg2rad(conditions.angle_deg)
        z_coefficient = float(np.cos(angle))
        x_coefficient = (
            0.0 if conditions.angle_deg == 0.0 else float(MM_PER_M * np.sin(angle))
        )

        z_lo, z_hi = self.field.z_extent
        bounds = [sorted((z_lo / z_coefficient, z_hi / z_coefficient))]
        n_points = self.field.z.size
        if x_coefficient != 0.0:
            bounds.append(
                sorted(
                    (
                        (x_lo - conditions.x_mm) / x_coefficient,
                        (x_hi - conditions.x_mm) / x_coefficient,
                    )
                )
            )
            n_points += self.field.x.size
        t_min = max(lower for lower, _ in bounds)
        t_max = min(upper for _, upper in bounds)
        if not t_min < t_max:
            raise ValueError("the lineout line does not intersect the cube")
        return LineoutPath(
            z_origin=0.0,
            z_coefficient=z_coefficient,
            x_origin=float(conditions.x_mm),
            x_coefficient=x_coefficient,
            t_min=float(t_min),
            t_max=float(t_max),
            n_points=int(n_points),
        )

    def h5_profile_kwargs(self, conditions: LineoutConditions) -> dict[str, Any]:
        """``InterpolateFromH5Profile`` arguments that reproduce this lineout in FBPIC."""
        return {
            "filename": None if self.path is None else str(self.path),
            "density_name": "density",
            "lineout_axis": self.path_for(conditions).lineout_axis(),
            "interpolation_points": {"pressure_bar": float(conditions.pressure_bar)},
        }

    # ------------------------------------------------------------ lineouts
    def lineout(
        self, conditions: LineoutConditions, t_values: np.ndarray | None = None
    ) -> Lineout:
        """The density along the lineout line, as a function of the path parameter.

        The returned ``Lineout.z`` is the path parameter ``t`` [m], which is the
        longitudinal coordinate FBPIC sees. ``t_values`` overrides the sampling;
        values whose point leaves the cube evaluate to zero.
        """
        p_lo, p_hi = self.pressure_extent
        if not p_lo <= conditions.pressure_bar <= p_hi:
            raise ValueError(
                f"pressure_bar={conditions.pressure_bar:g} is outside [{p_lo:g}, {p_hi:g}]"
            )
        path = self.path_for(conditions)
        if t_values is None:
            t = self.field.z if path.is_axial else path.t_values()
            inside = np.ones(t.shape, dtype=bool)
        else:
            t = np.asarray(t_values, dtype=np.float64)
            inside = (t >= path.t_min) & (t <= path.t_max)
        density = np.zeros(t.shape, dtype=np.float64)
        z_m, x_mm = path.z_m(t[inside]), path.x_mm(t[inside])
        if self.method == "linear":
            density[inside] = trilinear_density(
                self.field, z_m, x_mm, conditions.pressure_bar
            )
        else:
            density[inside] = [
                self.field.interpolate(float(z), float(x), conditions.pressure_bar)
                for z, x in zip(z_m, x_mm, strict=True)
            ]
        return Lineout(
            z=t,
            density=density,
            density_units=self.density_units,
            conditions=conditions,
            source=self.name,
        )
