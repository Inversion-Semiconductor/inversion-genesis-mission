#!/usr/bin/env python3
"""Parse density lineout files, mirror them across z=0, and save profiles.

Usage (conda env inv-fbpic):

    conda run -n inv-fbpic python process_symmetric.py 1.0e20 \\
        --input-dir data/density_lineouts_raw \\
        --output-dir data/density_lineouts

    conda run -n inv-fbpic python process_symmetric.py 1.0e20 \\
        data/density_lineouts_raw/800_um_conical/20_bar \\
        --output-dir data/density_lineouts

    conda run -n inv-fbpic python process_symmetric.py 1.0e20 \\
        --input-dir data/density_lineouts_raw/400_um \\
        --output-dir data/density_lineouts \\
        --hdf5-output data/density_lineouts/400_um.h5 \\
        --density-units cm^-3
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np

from ansys_lineouts import parse_lineout_file

X_POSITION_PATTERN = re.compile(r"(?:l-)?x-(\d+)(?:-(\d))?(?:-y-0)?$")
PRESSURE_FILE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)_bar(?:\.[^.]+)?$")


def sort_by_z(data: np.ndarray) -> np.ndarray:
    """Sort a (z, density) array by the z column."""
    order = np.argsort(data[:, 0])
    return data[order]


def mirror_across_z0(data: np.ndarray) -> np.ndarray:
    """Mirror a (z, density) profile across z=0."""
    data = sort_by_z(data)
    z, density = data[:, 0], data[:, 1]

    pos_mask = z >= 0
    z_pos, d_pos = z[pos_mask], density[pos_mask]

    mirror_mask = z_pos > 0
    z_neg = -z_pos[mirror_mask][::-1]
    d_neg = d_pos[mirror_mask][::-1]

    mirrored = np.column_stack(
        [np.concatenate([z_neg, z_pos]), np.concatenate([d_neg, d_pos])]
    )
    return sort_by_z(mirrored)


def parse_x_position(label: str) -> float:
    """Parse an x coordinate in mm from a raw lineout section label."""
    match = X_POSITION_PATTERN.fullmatch(label)
    if match is None:
        raise ValueError(f"Could not parse x position from {label!r}")

    integer = int(match.group(1))
    if match.group(2) is not None:
        return integer + int(match.group(2)) / 10.0
    return float(integer)


def parse_pressure_filename(filename: str) -> float:
    """Parse backing pressure in bar from a raw filename such as ``5_bar.txt``."""
    match = PRESSURE_FILE_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Could not parse backing pressure from {filename!r}")
    return float(match.group(1))


def _deduplicate_z(profile: np.ndarray) -> np.ndarray:
    """Keep the last density sample for every repeated z coordinate."""
    z = profile[:, 0]
    _, reverse_indices = np.unique(z[::-1], return_index=True)
    return profile[np.sort(profile.shape[0] - 1 - reverse_indices)]


def build_density_cube(
    input_dir: Path,
    density_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a density cube indexed as ``(z, x, backing_pressure)``.

    ``input_dir`` must contain the raw pressure files for exactly one nozzle.
    All pressures must provide the same set of x positions.
    """
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    pressure_files: list[tuple[float, Path]] = []
    for path in iter_lineout_inputs(input_dir):
        pressure_files.append((parse_pressure_filename(path.name), path))
    pressure_files.sort(key=lambda item: item[0])

    pressures = np.array([pressure for pressure, _ in pressure_files], dtype=np.float64)
    if len(set(pressures)) != pressures.size:
        raise ValueError(f"Duplicate backing pressure file in {input_dir}")

    profiles: dict[tuple[float, float], np.ndarray] = {}
    x_positions: list[float] | None = None
    for pressure, path in pressure_files:
        sections = parse_lineout_file(path)
        parsed_sections = sorted(
            (parse_x_position(label), data) for label, data in sections.items()
        )
        current_x_positions = [x_position for x_position, _ in parsed_sections]
        if len(set(current_x_positions)) != len(current_x_positions):
            raise ValueError(f"Duplicate x position in {path}")
        if x_positions is None:
            x_positions = current_x_positions
        elif x_positions != current_x_positions:
            raise ValueError(
                f"x positions in {path} do not match other pressures: "
                f"{x_positions} vs {current_x_positions}"
            )

        for x_position, data in parsed_sections:
            profile = mirror_across_z0(data)
            profile[:, 1] *= density_scale
            profiles[pressure, x_position] = _deduplicate_z(profile)

    assert x_positions is not None
    z = profiles[pressures[0], x_positions[0]][:, 0].copy()
    if np.any(np.diff(z) <= 0):
        raise ValueError("Mirrored z coordinates must be strictly increasing")

    density = np.empty((z.size, len(x_positions), pressures.size), dtype=np.float64)
    for ip, pressure in enumerate(pressures):
        for ix, x_position in enumerate(x_positions):
            profile = profiles[pressure, x_position]
            profile_z, profile_density = profile[:, 0], profile[:, 1]
            if np.any(np.diff(profile_z) <= 0):
                raise ValueError(
                    f"Mirrored z coordinates must be strictly increasing for "
                    f"x={x_position}, pressure={pressure}"
                )
            if profile_z.shape == z.shape and np.allclose(profile_z, z):
                density[:, ix, ip] = profile_density
            else:
                density[:, ix, ip] = np.interp(z, profile_z, profile_density)

    return z, np.array(x_positions, dtype=np.float64), pressures, density


def write_hdf5_density_cube(
    output_path: Path,
    *,
    nozzle: str,
    density_scale: float,
    density_units: str,
    z: np.ndarray,
    x: np.ndarray,
    pressure: np.ndarray,
    density: np.ndarray,
) -> Path:
    """Write one nozzle's coordinate-indexed density cube to HDF5."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as hdf5_file:
        hdf5_file.attrs["nozzle"] = nozzle
        hdf5_file.attrs["density_scale"] = density_scale
        hdf5_file.attrs["density_axis_order"] = "z_m,x_mm,pressure_bar"

        z_dataset = hdf5_file.create_dataset("z_m", data=z)
        x_dataset = hdf5_file.create_dataset("x_mm", data=x)
        pressure_dataset = hdf5_file.create_dataset("pressure_bar", data=pressure)
        density_dataset = hdf5_file.create_dataset("density", data=density)

        z_dataset.attrs["units"] = "m"
        x_dataset.attrs["units"] = "mm"
        pressure_dataset.attrs["units"] = "bar"
        density_dataset.attrs["units"] = density_units

        for axis, (name, coordinate_dataset) in enumerate(
            (
                ("z_m", z_dataset),
                ("x_mm", x_dataset),
                ("pressure_bar", pressure_dataset),
            )
        ):
            coordinate_dataset.make_scale(name)
            density_dataset.dims[axis].label = name
            density_dataset.dims[axis].attach_scale(coordinate_dataset)

    return output_path


def iter_lineout_inputs(path: Path, *, recursive: bool = False) -> list[Path]:
    """Return lineout file paths from a single file or a directory of files."""
    path = path.resolve()

    if path.is_file():
        return [path]

    if path.is_dir():
        iterator = path.rglob("*") if recursive else path.iterdir()
        files = sorted(
            p for p in iterator if p.is_file() and not p.name.startswith(".")
        )
        if not files:
            raise ValueError(f"No files found in {path}")
        return files

    raise FileNotFoundError(path)


def output_dir_for_input(
    input_path: Path,
    output_dir: Path,
    input_root: Path | None = None,
) -> Path:
    """Map an input file to its output directory, preserving relative structure."""
    input_path = input_path.resolve()
    if input_root is not None:
        relative = input_path.relative_to(input_root.resolve())
        return (output_dir / relative.with_suffix("")).resolve()
    return (output_dir / input_path.stem).resolve()


def process_file(
    input_path: Path,
    density_scale: float,
    output_dir: Path,
    input_root: Path | None = None,
) -> list[Path]:
    """Process one lineout file and write one .npz array per lineout section."""
    input_path = input_path.resolve()
    file_output_dir = output_dir_for_input(input_path, output_dir, input_root)
    file_output_dir.mkdir(parents=True, exist_ok=True)

    sections = parse_lineout_file(input_path)
    output_paths: list[Path] = []

    for label, data in sections.items():
        output_path = file_output_dir / f"{label}.npz"
        mirrored = mirror_across_z0(data)
        mirrored[:, 1] *= density_scale
        np.savez(output_path, data=mirrored)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse density lineout files, sort by z, mirror across z=0, "
            "and save each lineout as a separate .npz file."
        )
    )
    parser.add_argument(
        "density_scale",
        type=float,
        help="Factor to multiply raw file densities by before saving",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Path to a single raw density lineout file",
    )
    input_group.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        help="Directory containing raw density lineout files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Base output directory; files are written to "
            "<output-dir>/<relative-path>/<label>.npz"
        ),
    )
    parser.add_argument(
        "--hdf5-output",
        type=Path,
        default=None,
        help=(
            "Write a single HDF5 density cube for one nozzle input directory; "
            "the cube is indexed by z, x position, and backing pressure"
        ),
    )
    parser.add_argument(
        "--density-units",
        default=None,
        help=(
            "Units for the HDF5 density dataset, for example cm^-3 or m^-3; "
            "required with --hdf5-output"
        ),
    )
    args = parser.parse_args()

    using_input_dir = args.input_dir is not None
    input_path = args.input_dir if using_input_dir else args.input
    if args.hdf5_output is not None and not using_input_dir:
        parser.error("--hdf5-output requires --input-dir for one nozzle")
    if args.hdf5_output is not None and not args.density_units:
        parser.error("--density-units is required with --hdf5-output")
    input_root = input_path.resolve() if using_input_dir else None
    output_paths: list[Path] = []
    for lineout_path in iter_lineout_inputs(input_path, recursive=using_input_dir):
        output_paths.extend(
            process_file(
                lineout_path,
                args.density_scale,
                args.output_dir,
                input_root,
            )
        )

    for output_path in output_paths:
        print(f"Wrote {output_path}")

    if args.hdf5_output is not None:
        z, x, pressure, density = build_density_cube(
            args.input_dir,
            args.density_scale,
        )
        hdf5_path = write_hdf5_density_cube(
            args.hdf5_output,
            nozzle=args.input_dir.resolve().name,
            density_scale=args.density_scale,
            density_units=args.density_units,
            z=z,
            x=x,
            pressure=pressure,
            density=density,
        )
        print(f"Wrote {hdf5_path}")


if __name__ == "__main__":
    main()
