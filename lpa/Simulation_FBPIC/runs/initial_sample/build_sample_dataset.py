#!/usr/bin/env python3
"""Build a compact JSON dataset from the bundled sample FBPIC raw data.

From the repository root, build the seven-run example dataset with:

    python build_sample_dataset.py

This writes ``sample_dataset/sample_dataset.json``. Each simulation contains
its ``[VaryingParameters]`` input dictionary and the 33-scalar phase-space
descriptor under ``output``. The default selection is ``uz >= 30`` for the
``nitrogen_electrons`` species, followed by a central 95% crop on each
phase-space axis.

To plot one raw sample diagnostic with the same openPMD reader and selection:

    python phase_space_moments_from_h5.py \
        sample_dataset/raw_data/sim_0000/lab_diags/hdf5/data00000049.h5 \
        --species nitrogen_electrons \
        --uz-min 30 \
        --output sample_dataset/sim_0000_phase_space_moments.png \
        --all
"""
from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path

import numpy as np

from inversion_fbpic.utils.distributions import (
    compute_moment_descriptor,
    crop_central_particles,
    load_openpmd_particles,
    select_by_uz,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert sample raw FBPIC runs into an input/output JSON dataset."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("sample_dataset/raw_data"),
        help="Directory containing sim_*/input.ini folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sample_dataset/sample_dataset.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--uz-min",
        type=float,
        default=30.0,
        help="Retain particles with uz at or above this value (default: 30)",
    )
    parser.add_argument(
        "--species",
        default="nitrogen_electrons",
        help="openPMD particle species used to calculate outputs",
    )
    parser.add_argument(
        "--central-fraction",
        type=float,
        default=0.95,
        help=(
            "Central fraction retained independently on each phase-space axis "
            "(default: 0.95)"
        ),
    )
    return parser.parse_args()


def parse_ini_value(value: str) -> bool | int | float | str:
    """Parse scalar INI values while preserving non-numeric configuration strings."""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        numeric = float(value)
    except ValueError:
        return value
    return int(numeric) if numeric.is_integer() else numeric


def read_input(path: Path) -> dict[str, dict[str, bool | int | float | str]]:
    """Read every INI section and convert scalar values where possible."""
    parser = configparser.ConfigParser()
    with path.open(encoding="utf-8") as input_file:
        parser.read_file(input_file)
    return {
        section: {key: parse_ini_value(value) for key, value in parser[section].items()}
        for section in parser.sections()
    }


def build_output(
    diagnostic_file: Path, species: str, uz_min: float, central_fraction: float
) -> dict[str, float]:
    """Calculate the shared 33-scalar descriptor from an openPMD diagnostic."""
    particles, weights = load_openpmd_particles(diagnostic_file, species)
    particles, weights = select_by_uz(particles, weights, uz_min=uz_min)
    if weights is None:
        raise ValueError(f"{diagnostic_file}: {species} has no weights")
    particles, weights = crop_central_particles(
        particles, weights, central_fraction=central_fraction
    )
    return compute_moment_descriptor(particles, weights)


def main() -> None:
    args = parse_args()
    records: dict[str, dict[str, object]] = {}
    for run_dir in sorted(path for path in args.root.glob("sim_*") if path.is_dir()):
        diagnostic_files = sorted((run_dir / "lab_diags" / "hdf5").glob("data*.h5"))
        if not diagnostic_files:
            raise ValueError(f"{run_dir}: no particle diagnostics found")

        input_sections = read_input(run_dir / "input.ini")
        if "VaryingParameters" not in input_sections:
            raise ValueError(f"{run_dir}: missing [VaryingParameters] in input.ini")
        records[run_dir.name] = {
            "input": input_sections["VaryingParameters"],
            "output": build_output(
                diagnostic_files[-1], args.species, args.uz_min, args.central_fraction
            ),
        }

    if not records:
        raise SystemExit(f"No sim_* folders found below {args.root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, indent=2, allow_nan=False)
        output_file.write("\n")
    print(f"Wrote {args.output} with {len(records)} simulations")


if __name__ == "__main__":
    main()