#!/usr/bin/env python3
"""Build 2D density interpolations over (x, backing pressure) from HDF5.

Expects one nozzle HDF5 file containing coordinate datasets ``z_m``, ``x_mm``,
and ``pressure_bar``, plus a ``density`` dataset indexed as ``(z, x, pressure)``.

For plotting, see ``plot_density.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

InterpolationMethod = Literal["linear", "cubic", "quintic", "pchip"]


# ---------------------------------------------------------------------------
# HDF5 data loading
# ---------------------------------------------------------------------------

REQUIRED_DATASETS = ("z_m", "x_mm", "pressure_bar", "density")
EXPECTED_AXIS_ORDER = "z_m,x_mm,pressure_bar"


def _as_text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _validate_coordinates(name: str, values: np.ndarray, path: Path) -> None:
    if values.ndim != 1:
        raise ValueError(f"{path} dataset {name!r} must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"{path} dataset {name!r} must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} dataset {name!r} must contain finite values")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{path} dataset {name!r} must be strictly increasing")


def _validate_dimension_scales(hdf5_file: h5py.File, path: Path) -> None:
    """Ensure each density axis is explicitly linked to its coordinate dataset."""
    density_dataset = hdf5_file["density"]
    for axis, name in enumerate(("z_m", "x_mm", "pressure_bar")):
        dimension = density_dataset.dims[axis]
        if dimension.label != name:
            raise ValueError(
                f"{path} density axis {axis} has label {dimension.label!r}; "
                f"expected {name!r}"
            )

        scales = list(dimension.values())
        if len(scales) != 1 or scales[0].name != hdf5_file[name].name:
            attached_names = [scale.name for scale in scales]
            raise ValueError(
                f"{path} density axis {axis} must be linked only to {name!r}; "
                f"found {attached_names}"
            )


def load_density_hdf5(
    path: str | Path,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate a nozzle density cube from an HDF5 file."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        with h5py.File(path, "r") as hdf5_file:
            missing = [name for name in REQUIRED_DATASETS if name not in hdf5_file]
            if missing:
                raise ValueError(f"{path} is missing required dataset(s): {missing}")

            axis_order = hdf5_file.attrs.get("density_axis_order")
            if axis_order is not None and _as_text(axis_order) != EXPECTED_AXIS_ORDER:
                raise ValueError(
                    f"{path} has density_axis_order={_as_text(axis_order)!r}; "
                    f"expected {EXPECTED_AXIS_ORDER!r}"
                )

            _validate_dimension_scales(hdf5_file, path)

            z = np.asarray(hdf5_file["z_m"], dtype=np.float64)
            x = np.asarray(hdf5_file["x_mm"], dtype=np.float64)
            pressure = np.asarray(hdf5_file["pressure_bar"], dtype=np.float64)
            density = np.asarray(hdf5_file["density"], dtype=np.float64)
            nozzle = _as_text(hdf5_file.attrs.get("nozzle", path.stem))
    except OSError as exc:
        raise ValueError(f"Could not read HDF5 file {path}: {exc}") from exc

    _validate_coordinates("z_m", z, path)
    _validate_coordinates("x_mm", x, path)
    _validate_coordinates("pressure_bar", pressure, path)
    expected_shape = (z.size, x.size, pressure.size)
    if density.shape != expected_shape:
        raise ValueError(
            f"{path} density has shape {density.shape}; expected {expected_shape} "
            "for (z_m, x_mm, pressure_bar)"
        )
    if not np.all(np.isfinite(density)):
        raise ValueError(f"{path} density must contain finite values")

    return nozzle, z, x, pressure, density


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DensityInterpolation:
    """Density samples and 2D interpolators in the (x, pressure) plane."""

    geometry: str
    z: np.ndarray
    x: np.ndarray
    pressure: np.ndarray
    density: np.ndarray
    method: InterpolationMethod

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.density.shape

    @property
    def x_extent(self) -> tuple[float, float]:
        return float(self.x.min()), float(self.x.max())

    @property
    def z_extent(self) -> tuple[float, float]:
        return float(self.z.min()), float(self.z.max())

    @property
    def pressure_extent(self) -> tuple[float, float]:
        return float(self.pressure.min()), float(self.pressure.max())

    def _explain_interpolation_error(
        self,
        exc: Exception,
        *,
        z_value: float | None = None,
        x_value: float | None = None,
        pressure_value: float | None = None,
    ) -> None:
        details = [f"geometry={self.geometry!r}"]
        if z_value is not None:
            z_min, z_max = self.z_extent
            details.append(f"z={z_value} (valid range [{z_min}, {z_max}] m)")
        if x_value is not None:
            x_min, x_max = self.x_extent
            details.append(f"x={x_value} (valid range [{x_min}, {x_max}] mm)")
        if pressure_value is not None:
            p_min, p_max = self.pressure_extent
            details.append(
                f"pressure={pressure_value} (valid range [{p_min}, {p_max}] bar)"
            )
        print(
            "Density interpolation failed: "
            + "; ".join(details)
            + f". ({type(exc).__name__}: {exc})"
        )

    def density_at_z(self, z_value: float) -> np.ndarray:
        """Return a (n_x, n_pressure) slice interpolated along z."""
        z_interpolator = interp1d(
            self.z, self.density[:, 0, 0], bounds_error=True, assume_sorted=True
        )
        z_interpolator(z_value)

        slice_2d = np.empty((self.x.size, self.pressure.size), dtype=np.float64)
        for ix in range(self.x.size):
            for ip in range(self.pressure.size):
                slice_2d[ix, ip] = np.interp(
                    z_value, self.z, self.density[:, ix, ip]
                )
        return slice_2d

    def _make_2d_interpolator(self, values_2d: np.ndarray) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            (self.x, self.pressure),
            values_2d,
            method=self.method,
            bounds_error=True,
        )

    def _evaluate_2d(
        self,
        interpolator: RegularGridInterpolator,
        x_value: float,
        pressure_value: float,
    ) -> float:
        return float(interpolator((x_value, pressure_value)))

    def _density_at_x_pressure(
        self, z_value: float, x_value: float, pressure_value: float
    ) -> float:
        values_2d = self.density_at_z(z_value)
        interpolator = self._make_2d_interpolator(values_2d)
        return self._evaluate_2d(interpolator, x_value, pressure_value)

    def interpolate(self, z_value: float, x_value: float, pressure_value: float) -> float:
        """Interpolate density at a single (z, x, pressure) point."""
        try:
            return self._density_at_x_pressure(z_value, x_value, pressure_value)
        except ValueError as exc:
            self._explain_interpolation_error(
                exc,
                z_value=z_value,
                x_value=x_value,
                pressure_value=pressure_value,
            )
            raise

    def interpolate_along_z(
        self,
        z_values: np.ndarray,
        x_value: float,
        pressure_value: float,
    ) -> np.ndarray:
        """Interpolate density along z at fixed x and backing pressure."""
        z_array = np.asarray(z_values, dtype=np.float64)
        result = np.empty_like(z_array)
        try:
            for index, z_value in enumerate(z_array.flat):
                result.flat[index] = self._density_at_x_pressure(
                    z_value, x_value, pressure_value
                )
        except ValueError as exc:
            self._explain_interpolation_error(
                exc,
                z_value=z_value,
                x_value=x_value,
                pressure_value=pressure_value,
            )
            raise
        return result.reshape(z_array.shape)

    def density_profile_along_z(
        self, x_value: float, pressure_value: float
    ) -> np.ndarray:
        """Return density on the stored z grid at fixed x and backing pressure."""
        return self.interpolate_along_z(self.z, x_value, pressure_value)

    def interpolate_grid(
        self,
        z_value: float,
        x_values: np.ndarray,
        pressure_values: np.ndarray,
    ) -> np.ndarray:
        """Interpolate density on a regular (x, pressure) grid at fixed z."""
        x_grid, pressure_grid = np.meshgrid(x_values, pressure_values, indexing="ij")
        query = np.column_stack([x_grid.ravel(), pressure_grid.ravel()])
        try:
            values_2d = self.density_at_z(z_value)
            interpolator = self._make_2d_interpolator(values_2d)
            return interpolator(query).reshape(x_grid.shape)
        except ValueError as exc:
            self._explain_interpolation_error(exc, z_value=z_value)
            raise

    def interpolate_xz_grid(
        self,
        x_values: np.ndarray,
        z_values: np.ndarray,
        pressure_value: float,
    ) -> np.ndarray:
        """Interpolate density on a regular (x, z) grid at fixed backing pressure."""
        density_grid = np.empty((x_values.size, z_values.size), dtype=np.float64)
        query = np.column_stack(
            [x_values, np.full(x_values.size, pressure_value)]
        )
        try:
            for iz, z_value in enumerate(z_values):
                values_2d = self.density_at_z(z_value)
                interpolator = self._make_2d_interpolator(values_2d)
                density_grid[:, iz] = interpolator(query)
        except ValueError as exc:
            self._explain_interpolation_error(
                exc,
                z_value=z_value,
                pressure_value=pressure_value,
            )
            raise
        return density_grid

    def xz_grids_at_pressures(
        self,
        x_values: np.ndarray,
        z_values: np.ndarray,
    ) -> np.ndarray:
        """Return density on an (x, z) grid for each tabulated backing pressure."""
        return np.stack(
            [
                self.interpolate_xz_grid(x_values, z_values, float(pressure_value))
                for pressure_value in self.pressure
            ],
            axis=0,
        )

    def interpolate_xz_from_pressure_stack(
        self,
        xz_stack: np.ndarray,
        pressure_value: float,
    ) -> np.ndarray:
        """Interpolate precomputed (pressure, x, z) grids to a backing pressure."""
        interpolator = interp1d(
            self.pressure,
            xz_stack,
            axis=0,
            bounds_error=True,
            assume_sorted=True,
        )
        return np.asarray(interpolator(pressure_value), dtype=np.float64)


def build_density_interpolation(
    hdf5_path: Path,
    *,
    method: InterpolationMethod = "linear",
) -> DensityInterpolation:
    """Load an HDF5 density cube and prepare a (x, pressure) interpolator."""
    nozzle, z, x, pressure, density = load_density_hdf5(hdf5_path)

    return DensityInterpolation(
        geometry=nozzle,
        z=z,
        x=x,
        pressure=pressure,
        density=density,
        method=method,
    )


def build_density_callable(
    hdf5_path: Path,
    backing_pressure: float,
    x_position: float,
    *,
    method: InterpolationMethod = "linear",
    field: DensityInterpolation | None = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return an FBPIC-compatible density function of z at fixed x and pressure.

    The returned callable accepts ``(z, r)`` where ``r`` is unused but required by
    FBPIC. The samples are loaded from ``hdf5_path`` if ``field`` is not supplied.
    ``z`` and ``r`` may be 1-D arrays; the returned density array matches the
    shape of ``z``.
    """
    if field is None:
        field = build_density_interpolation(hdf5_path, method=method)
    elif method != field.method:
        raise ValueError(
            f"method={method!r} does not match field.method={field.method!r}"
        )

    def density(z: np.ndarray, r: np.ndarray) -> np.ndarray:
        del r
        return field.interpolate_along_z(z, x_position, backing_pressure)

    return density
