"""Recalculate campaign beam scalars at a virtual-waist-relative drift plane.

This is a command-line wrapper around :mod:`inversion_fbpic.utils.virtual_waist`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inversion_fbpic.utils.virtual_waist import (
    shift_run_result,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Drift a beam to a virtual-waist-relative plane and recalculate scalars."
    )
    parser.add_argument("simulation_directory", type=Path)
    parser.add_argument(
        "--species",
        nargs="+",
        required=True,
        help="Electron species to combine from the final particle dump.",
    )
    parser.add_argument(
        "--target-z-mm",
        type=float,
        default=5.0,
        help="Target plane in mm downstream of the averaged virtual waist (default: 5).",
    )
    parser.add_argument(
        "--uz-min",
        type=float,
        default=20.0,
        help="Minimum normalized longitudinal momentum (default: 20).",
    )
    parser.add_argument(
        "--central-fraction",
        type=float,
        default=0.95,
        help="Six-dimensional central crop used for campaign scalars (default: 0.95).",
    )
    return parser.parse_args()


def main() -> None:
    """Drift one simulation's electron beam and write shifted campaign scalars."""
    args = parse_args()
    simulation_directory = args.simulation_directory.resolve()
    _, metadata, output_result = shift_run_result(
        simulation_directory,
        species=tuple(args.species),
        target_z_m=args.target_z_mm * 1.0e-3,
        uz_min=args.uz_min,
        central_fraction=args.central_fraction,
    )
    print(f"Wrote {output_result}")
    print(f"Averaged virtual waist: {metadata['waist_average_m'] * 1.0e3:.3f} mm")
    print(
        "Forward drift from final diagnostic: "
        f"{metadata['drift_distance_from_final_diagnostic_m'] * 1.0e3:.3f} mm"
    )


if __name__ == "__main__":
    main()
