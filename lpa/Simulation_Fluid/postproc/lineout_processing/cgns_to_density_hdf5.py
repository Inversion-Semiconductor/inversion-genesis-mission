#!/usr/bin/env python3
"""Convert unstructured CGNS density fields to a regular nozzle HDF5 grid.

The output follows the density-cube convention used by ``interpolate_density``:
``density[z_m, x_mm, pressure_bar]``. Directory input combines one CGNS file per
backing pressure into the pressure axis; source files must be named
``<pressure>_bar.cgns``.

Example (conda env inv-fbpic):

    python cgns_to_density_hdf5.py \
        data/density_field_raw/htu_density \
        --output data/density_lineouts/htu_density.h5 \
        --density-units kg/m^3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import griddata

from process_symmetric import write_hdf5_density_cube

DEFAULT_ZONE_PATH = "Base/Zone"
DEFAULT_FLOW_SOLUTION = "FlowSolution.N:1"
DEFAULT_X_COORDINATE = "CoordinateX"
DEFAULT_Z_COORDINATE = "CoordinateY"
DEFAULT_DENSITY_NAME = "Density"
PRESSURE_FILE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)_bar\.cgns$", re.IGNORECASE)


class CgnsDataError(ValueError):
    """Raised when required pointwise data is absent or invalid in a CGNS file."""


def parse_pressure_filename(filename: str) -> float:
    """Parse backing pressure in bar from a CGNS filename such as ``10_bar.cgns``."""
    match = PRESSURE_FILE_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(
            f"Could not parse backing pressure from {filename!r}; expected "
            "<pressure>_bar.cgns"
        )
    return float(match.group(1))


def collect_cgns_inputs(
    input_path: str | Path,
    *,
    pressure_bar: float | None,
) -> list[tuple[float, Path]]:
    """Return pressure-tagged CGNS source paths for one nozzle conversion."""
    input_path = Path(input_path).resolve()
    if input_path.is_file():
        if pressure_bar is None:
            raise ValueError("pressure_bar is required when input is one CGNS file")
        if not np.isfinite(pressure_bar):
            raise ValueError("pressure_bar must be finite")
        return [(float(pressure_bar), input_path)]

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    if pressure_bar is not None:
        raise ValueError("pressure_bar is only valid when input is one CGNS file")

    sources = [
        (parse_pressure_filename(path.name), path)
        for path in sorted(input_path.rglob("*.cgns"))
        if path.is_file()
    ]
    if not sources:
        raise ValueError(f"No pressure-tagged .cgns files found in {input_path}")
    sources.sort(key=lambda item: item[0])

    pressures = [pressure for pressure, _ in sources]
    if len(set(pressures)) != len(pressures):
        raise ValueError(f"Duplicate backing pressure file in {input_path}")
    return sources


def _read_cgns_data(node: h5py.Group, description: str) -> np.ndarray:
    """Read the data child used by the CGNS HDF5 mapping."""
    for name in (" data", "data"):
        if name in node:
            return np.asarray(node[name], dtype=np.float64)
    raise CgnsDataError(f"{description} does not contain a CGNS data array")


def load_cgns_density_points(
    input_path: str | Path,
    *,
    zone_path: str = DEFAULT_ZONE_PATH,
    flow_solution: str = DEFAULT_FLOW_SOLUTION,
    x_coordinate: str = DEFAULT_X_COORDINATE,
    z_coordinate: str = DEFAULT_Z_COORDINATE,
    density_name: str = DEFAULT_DENSITY_NAME,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x position, z position, and density from one CGNS zone."""
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    try:
        with h5py.File(input_path, "r") as cgns_file:
            zone = cgns_file[zone_path]
            coordinates = zone["GridCoordinates"]
            x_points = _read_cgns_data(
                coordinates[x_coordinate],
                f"{input_path} coordinate {x_coordinate!r}",
            )
            z_points = _read_cgns_data(
                coordinates[z_coordinate],
                f"{input_path} coordinate {z_coordinate!r}",
            )
            density_points = _read_cgns_data(
                zone[f"{flow_solution}/{density_name}"],
                f"{input_path} density {flow_solution}/{density_name}",
            )
    except KeyError as exc:
        raise CgnsDataError(
            f"{input_path} is missing required CGNS node {exc.args[0]!r}"
        ) from exc
    except OSError as exc:
        raise CgnsDataError(f"Could not read CGNS file {input_path}: {exc}") from exc

    if z_points.ndim != 1 or x_points.ndim != 1 or density_points.ndim != 1:
        raise CgnsDataError("CGNS coordinates and density must be one-dimensional")
    if z_points.size < 3:
        raise CgnsDataError("CGNS field must contain at least three points")
    if not (z_points.size == x_points.size == density_points.size):
        raise CgnsDataError(
            "CGNS x coordinate, z coordinate, and density arrays "
            "must have the same length"
        )
    if not all(np.all(np.isfinite(values)) for values in (z_points, x_points, density_points)):
        raise CgnsDataError("CGNS coordinates and density must contain finite values")

    return x_points, z_points, density_points


def select_and_mirror_half_plane(
    x_points: np.ndarray,
    z_points: np.ndarray,
    density_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep nonnegative x samples and mirror z data across zero."""
    x_mask = x_points >= 0.0
    x_points = x_points[x_mask]
    z_points = z_points[x_mask]
    density_points = density_points[x_mask]
    if x_points.size < 3:
        raise CgnsDataError("CGNS field has fewer than three samples at x >= 0")

    positive_z_mask = z_points > 0.0
    return (
        np.concatenate((z_points, -z_points[positive_z_mask])),
        np.concatenate((x_points, x_points[positive_z_mask])),
        np.concatenate((density_points, density_points[positive_z_mask])),
    )


def regular_grid(
    z_points: np.ndarray,
    x_points: np.ndarray,
    *,
    z_count: int,
    x_count: int,
    z_bounds: tuple[float, float] | None = None,
    x_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create regular output z and x grids in source-coordinate meters."""
    if z_count < 2 or x_count < 2:
        raise ValueError("--z-points and --x-points must both be at least 2")

    z_min, z_max = z_bounds or (float(z_points.min()), float(z_points.max()))
    x_min, x_max = x_bounds or (float(x_points.min()), float(x_points.max()))
    if not z_min < z_max:
        raise ValueError("z bounds must be strictly increasing")
    if not x_min < x_max:
        raise ValueError("x bounds must be strictly increasing")

    return (
        _symmetric_z_grid(z_min, z_max, z_count),
        np.linspace(x_min, x_max, x_count, dtype=np.float64),
    )


def _symmetric_z_grid(z_min: float, z_max: float, count: int) -> np.ndarray:
    """Create an exactly mirrored z grid over symmetric bounds."""
    if not np.isclose(z_min, -z_max):
        return np.linspace(z_min, z_max, count, dtype=np.float64)

    if count % 2:
        positive = np.linspace(0.0, z_max, count // 2 + 1, dtype=np.float64)
        return np.concatenate((-positive[:0:-1], positive))

    positive = np.linspace(z_max / count, z_max, count // 2, dtype=np.float64)
    return np.concatenate((-positive[::-1], positive))


def shared_regular_grid(
    fields: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    z_count: int,
    x_count: int,
    z_bounds: tuple[float, float] | None = None,
    x_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a regular grid over the common extent of all CGNS fields."""
    if not fields:
        raise ValueError("At least one CGNS field is required")

    default_z_bounds = (
        max(float(z_points.min()) for z_points, _, _ in fields),
        min(float(z_points.max()) for z_points, _, _ in fields),
    )
    default_x_bounds = (
        max(float(x_points.min()) for _, x_points, _ in fields),
        min(float(x_points.max()) for _, x_points, _ in fields),
    )
    z_limits = z_bounds or default_z_bounds
    x_limits = x_bounds or default_x_bounds
    if not z_limits[0] < z_limits[1]:
        raise ValueError("CGNS fields do not have a nonzero common z range")
    if not x_limits[0] < x_limits[1]:
        raise ValueError("CGNS fields do not have a nonzero common x range")

    return regular_grid(
        np.asarray(z_limits, dtype=np.float64),
        np.asarray(x_limits, dtype=np.float64),
        z_count=z_count,
        x_count=x_count,
        z_bounds=z_limits,
        x_bounds=x_limits,
    )


def grid_density(
    z_points: np.ndarray,
    x_points: np.ndarray,
    density_points: np.ndarray,
    z_grid: np.ndarray,
    x_grid: np.ndarray,
    *,
    method: str = "linear",
    outside_fill: str = "zero",
) -> np.ndarray:
    """Interpolate unstructured density samples onto a regular ``(z, x)`` grid."""
    z_mesh, x_mesh = np.meshgrid(z_grid, x_grid, indexing="ij")
    field = griddata(
        np.column_stack((z_points, x_points)),
        density_points,
        (np.abs(z_mesh), x_mesh),
        method=method,
    )

    missing = np.isnan(field)
    if not np.any(missing):
        return np.asarray(field, dtype=np.float64)
    if outside_fill == "raise":
        raise ValueError(
            f"Interpolation left {missing.sum()} grid points outside the CGNS domain"
        )
    if outside_fill == "zero":
        field[missing] = 0.0
    elif outside_fill == "nearest":
        field[missing] = griddata(
            np.column_stack((z_points, x_points)),
            density_points,
            (np.abs(z_mesh[missing]), x_mesh[missing]),
            method="nearest",
        )
    else:
        raise ValueError(f"Unknown outside-fill mode {outside_fill!r}")

    if np.array_equal(z_grid, -z_grid[::-1]):
        negative_count = z_grid.size // 2
        field[:negative_count, :] = field[-1:-(negative_count + 1):-1, :]

    return np.asarray(field, dtype=np.float64)


def convert_cgns_to_density_hdf5(
    input_path: str | Path,
    output_path: str | Path,
    *,
    pressure_bar: float | None = None,
    density_scale: float,
    density_units: str,
    z_count: int = 1000,
    x_count: int = 500,
    z_bounds: tuple[float, float] | None = None,
    x_bounds: tuple[float, float] | None = None,
    method: str = "linear",
    outside_fill: str = "zero",
    nozzle: str | None = None,
    zone_path: str = DEFAULT_ZONE_PATH,
    flow_solution: str = DEFAULT_FLOW_SOLUTION,
    x_coordinate: str = DEFAULT_X_COORDINATE,
    z_coordinate: str = DEFAULT_Z_COORDINATE,
    density_name: str = DEFAULT_DENSITY_NAME,
) -> Path:
    """Convert one CGNS field or pressure-tagged CGNS directory to HDF5.

    CGNS coordinates are interpreted as meters. ``CoordinateY`` becomes
    ``z_m`` and ``CoordinateX`` becomes ``x_mm`` in the output. For
    directory input, each ``<pressure>_bar.cgns`` file becomes one pressure axis
    sample, and all fields are interpolated onto a shared grid.
    """
    if not np.isfinite(density_scale):
        raise ValueError("density_scale must be finite")
    if not density_units:
        raise ValueError("density_units must not be empty")

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    sources = collect_cgns_inputs(input_path, pressure_bar=pressure_bar)
    fields = [
        select_and_mirror_half_plane(
            *load_cgns_density_points(
                source_path,
                zone_path=zone_path,
                flow_solution=flow_solution,
                x_coordinate=x_coordinate,
                z_coordinate=z_coordinate,
                density_name=density_name,
            )
        )
        for _, source_path in sources
    ]
    z_grid, x_grid_m = shared_regular_grid(
        fields,
        z_count=z_count,
        x_count=x_count,
        z_bounds=z_bounds,
        x_bounds=x_bounds,
    )
    density = np.empty((z_grid.size, x_grid_m.size, len(fields)), dtype=np.float64)
    for pressure_index, (z_points, x_points, density_points) in enumerate(fields):
        density[:, :, pressure_index] = density_scale * grid_density(
            z_points,
            x_points,
            density_points,
            z_grid,
            x_grid_m,
            method=method,
            outside_fill=outside_fill,
        )

    pressures = np.array([pressure for pressure, _ in sources], dtype=np.float64)
    output = write_hdf5_density_cube(
        output_path,
        nozzle=nozzle or input_path.name,
        density_scale=density_scale,
        density_units=density_units,
        z=z_grid,
        x=x_grid_m * 1000.0,
        pressure=pressures,
        density=density,
    )

    with h5py.File(output, "a") as hdf5_file:
        hdf5_file.attrs["source_format"] = "CGNS"
        hdf5_file.attrs["source_files"] = np.asarray(
            [str(source_path) for _, source_path in sources],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        hdf5_file.attrs["source_pressures_bar"] = pressures
        hdf5_file.attrs["source_coordinate_mapping"] = (
            f"z_m={z_coordinate};x_mm={x_coordinate}*1000"
        )
        hdf5_file.attrs["source_x_selection"] = f"{x_coordinate} >= 0"
        hdf5_file.attrs["source_z_symmetry"] = (
            f"mirrored across {z_coordinate}=0"
        )
        hdf5_file.attrs["interpolation_method"] = method
        hdf5_file.attrs["outside_fill"] = outside_fill

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert unstructured CGNS density fields to a regular HDF5 grid."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing <pressure>_bar.cgns files",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output HDF5 file")
    parser.add_argument(
        "--density-scale",
        type=float,
        default=1.0,
        help="Factor applied to CGNS density values before writing (default: 1)",
    )
    parser.add_argument(
        "--density-units",
        required=True,
        help="Units for the output density dataset, for example cm^-3 or m^-3",
    )
    parser.add_argument("--z-points", type=int, default=1000, help="Output axial grid points")
    parser.add_argument("--x-points", type=int, default=500, help="Output transverse grid points")
    parser.add_argument(
        "--z-bounds",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=None,
        help="Optional axial output bounds in m (default: CGNS extent)",
    )
    parser.add_argument(
        "--x-bounds",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=None,
        help="Optional transverse output bounds in m (default: CGNS extent)",
    )
    args = parser.parse_args()

    output = convert_cgns_to_density_hdf5(
        args.input_dir,
        args.output,
        density_scale=args.density_scale,
        density_units=args.density_units,
        z_count=args.z_points,
        x_count=args.x_points,
        z_bounds=tuple(args.z_bounds) if args.z_bounds is not None else None,
        x_bounds=tuple(args.x_bounds) if args.x_bounds is not None else None,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
