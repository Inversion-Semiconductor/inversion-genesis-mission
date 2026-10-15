"""Build lume-fbpic archives from the `runs/initial_sample` sample dataset.

`sample_dataset.json` holds seven ionization-injection runs (`sim_0000` to `sim_0006`), each as
24 physical inputs and the 33 moment-descriptor outputs. For every run this writes
`<output dir>/<run>.h5`, a `LUMEFBPICModel` archive (`LUMEFBPICModel.archive()`) with

- the config rebuilt from the run's inputs, on the production hyperparameters of
  `ionization_injection_runscript_00.py` (`ionization_injection.PRODUCTION`);
- the action inputs, read from that config;
- the 33 `descriptor_*` outputs as recorded in the dataset (the charge converted from C to pC);
- the dataset record itself, as an `inversion_fbpic` `Parameters` (the simulator's `parameters`,
  group `parameters` of the archive): the run name, the dataset file name, the descriptor
  selection the dataset used, and the run's inputs and outputs exactly as recorded, so an archive
  says what it was rebuilt from.

`LUMEFBPICModel.from_archive()` loads one and its `get()` returns the recorded descriptor values,
so it can be served with `python -m lume_fbpic_twin.serve`.

These are RECONSTRUCTED archives, not the runs themselves:

1. There are no final particles and no `stats`; the dataset keeps only the 33 descriptor scalars.
2. The config approximates the config that produced the numbers. `lume_fbpic` cannot match the
   original interaction length (`right_buffer` must be positive, see `ionization_injection.py`
   docstring), does not pass `p_zmin`/`p_rmax` to the particle loading, and the runs
   used 8 MPI ranks on GPUs, which this records only as `use_mpi=True`.
3. Only inputs that have an action (laser energy, pulse duration, focal position, the nitrogen
   dopant fraction and the 12 Zernike coefficients) may differ from the `ionization_injection.py`
   baseline, which equals `sim_0000`. Any other input that differs raises an error rather than
   being dropped silently; the dataset's runs vary only five of the 24 inputs.

Usage: `python initial_sample_archives.py [sample_dataset.json] [output dir]`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import typing

import attrs

from lume_fbpic.actions import (
    DESCRIPTOR_PREFIX,
    MomentDescriptorAction,
    evolve_laser,
)
from lume_fbpic.density_profiles import atom_density
from lume_fbpic.model import LUMEFBPICModel

from inversion_fbpic.lib.datapoint import Parameters

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import ionization_injection  # noqa: E402  (docs/examples/ionization_injection.py)

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[4]
    / "lpa/Simulation_FBPIC/runs/initial_sample/sample_dataset/sample_dataset.json"
)


def build_archive(
    record: dict[str, typing.Any],
    dataset_metadata: dict[str, typing.Any],
    path: Path,
    *,
    run: str,
    dataset: str,
) -> Path:
    """Write the archive for one dataset run and return its path. `run` is the run's name in
    the dataset and `dataset` the dataset's file name; both go into the archive's `parameters`.
    """
    model = ionization_injection.build_model()
    model.dummy_run = True
    _use_production_settings(model)
    _check_descriptor_settings(model, dataset_metadata)
    _check_fixed_inputs(model, record["input"])
    _apply_inputs(model, record["input"])
    model.simulator.recorded_outputs = {
        DESCRIPTOR_PREFIX + name: value
        for name, value in _descriptor_outputs(record["output"]).items()
    }
    model.simulator.parameters = Parameters(
        data={
            "run": run,
            "dataset": dataset,
            "selection": dict(dataset_metadata),
            "inputs": dict(record["input"]),
            "outputs": dict(record["output"]),
        }
    )
    model.archive(path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "output_dir", nargs="?", type=Path, default=Path("initial_sample_archives")
    )
    args = parser.parse_args(argv)

    dataset = json.loads(args.dataset.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for run, record in dataset["runs"].items():
        path = build_archive(
            record,
            dataset["metadata"],
            args.output_dir / f"{run}.h5",
            run=run,
            dataset=args.dataset.name,
        )
        print(f"wrote {path}")


def _apply_inputs(model: LUMEFBPICModel, inputs: dict[str, float]) -> None:
    """Set the inputs that have an action (the rest were checked to equal the baseline)."""
    values = {}
    for name, value in inputs.items():
        if name in ionization_injection.DATASET_ACTIONS:
            values[ionization_injection.DATASET_ACTIONS[name]] = value
        elif name.startswith(ionization_injection.ZERNIKE_PREFIX):
            values[name] = value
    model.set(values)


def _check_descriptor_settings(
    model: LUMEFBPICModel, dataset_metadata: dict[str, typing.Any]
) -> None:
    """The model's descriptor actions must select particles as the dataset's did."""
    wanted = {
        "uz_min": dataset_metadata["uz_min"],
        "central_fraction": dataset_metadata["central_fraction"],
        "longitudinal_bins": dataset_metadata["longitudinal_bins"],
    }
    for action in model.supported_variables.values():
        if isinstance(action, MomentDescriptorAction):
            have = {key: getattr(action, key) for key in wanted}
            if have != wanted:
                raise ValueError(
                    f"descriptor actions use {have}, the dataset used {wanted}"
                )


def _check_fixed_inputs(model: LUMEFBPICModel, inputs: dict[str, float]) -> None:
    """Raise if an input without an action differs from what the model already has."""
    simulator = model.simulator
    he, nitrogen = simulator.densities
    baseline = {
        "density_alpha_m": he.alpha,
        "density_beta": he.beta,
        "density_center_location_m": he.z0,
        "density_peak": he.gauss_peak,
        "laser_spot_size_m": simulator.laser.waist,
        "laser_super_gaussian_order": simulator.laser.super_gaussian_order,
        "laser_wavelength_m": simulator.laser.wavelength,
        # n_plasma = 2 * n_gas, and n_gas is the He + N atom density
        "peak_plasma_density_cm3": 2.0
        * (atom_density(he) + atom_density(nitrogen))
        / 1.0e6,
    }
    for name, value in inputs.items():
        if name in baseline and not math.isclose(value, baseline[name], rel_tol=1e-9):
            raise ValueError(
                f"input {name}={value} differs from the baseline {baseline[name]} and has no "
                "action to set it"
            )
    unknown = (
        set(inputs)
        - set(baseline)
        - set(ionization_injection.DATASET_ACTIONS)
        - {k for k in inputs if k.startswith(ionization_injection.ZERNIKE_PREFIX)}
    )
    if unknown:
        raise ValueError(
            f"inputs this script does not know how to apply: {sorted(unknown)}"
        )


def _descriptor_outputs(output: dict[str, float]) -> dict[str, float]:
    """The dataset's recorded descriptor values under the feature names `lume_fbpic` uses.

    The dataset records the charge in coulombs as `total_beam_charge_c`; the descriptor feature
    is `total_beam_charge_pc`, in picocoulombs.
    """
    converted = dict(output)
    if "total_beam_charge_c" in converted:
        converted["total_beam_charge_pc"] = converted.pop("total_beam_charge_c") * 1e12
    return converted


def _use_production_settings(model: LUMEFBPICModel) -> None:
    """Replace the example's coarsened grid and laser grid by the production ones."""
    simulator = model.simulator
    production = ionization_injection.PRODUCTION
    simulator.hyparams = attrs.evolve(
        simulator.hyparams,
        nz=production["nz"],
        nr=production["nr"],
        nm=production["nm"],
        number_dumps=production["number_dumps"],
        write_period=production["write_period"],
        use_mpi=production["use_mpi"],
    )
    simulator.densities = [
        attrs.evolve(density, p_nt=production["p_nt"])
        for density in simulator.densities
    ]
    simulator.laser = evolve_laser(
        simulator.laser,
        n_azimuthal_modes=production["laser_n_azimuthal_modes"],
        num_points=production["laser_num_points"],
    )


if __name__ == "__main__":
    main()
