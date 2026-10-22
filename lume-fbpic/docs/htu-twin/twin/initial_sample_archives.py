"""Build lume-fbpic archives from the `runs/initial_sample` sample dataset.

`sample_dataset.json` holds seven ionization-injection runs (`sim_0000` to `sim_0006`), each as
24 physical inputs and the 33 moment-descriptor outputs. For every run this writes
`<output dir>/<run>.h5`, a `LUMEFBPICModel` archive (`LUMEFBPICModel.archive()`) with

- the config rebuilt from the run's inputs (`build_model()`, then the inputs that vary);
- the action inputs, read from that config;
- the 33 `descriptor_*` outputs as recorded in the dataset (the charge converted from C to pC);
- the dataset record itself, as an `inversion_fbpic` `Parameters` (the simulator's `parameters`,
  group `parameters` of the archive): the run name, the dataset file name, the descriptor
  selection the dataset used, and the run's inputs and outputs exactly as recorded, so an archive
  says what it was rebuilt from.

`LUMEFBPICModel.from_archive()` loads one and its `get()` returns the recorded descriptor values,
so it can be served with `python -m lume_fbpic_twin.serve`.

`build_model()` is the configuration the dataset's runs were made with: `sim_0000` of
`runs/initial_sample/ionization_injection_runscript_00.py` and its template as they were when the
dataset was written (inversion-genesis-mission #14, `5dfff13`), on that runscript's production
hyperparameters. Both files have changed since, so this script holds its own copy rather than
following them or `docs/examples/ionization_injection.py`.

These are RECONSTRUCTED archives, not the runs themselves:

1. There are no final particles and no `stats`; the dataset keeps only the 33 descriptor scalars.
2. The config approximates the config that produced the numbers:
   - The gas: the template called `add_new_species()` directly for He, N and two empty electron
     species. Here two `GeneralizedGaussianProfile`s (He and N, same shape) do it, with
     `nominal_density` set to the atom density times the species' ionization levels, because
     `_DensityProfile` divides by the level count. The nitrogen is seeded as atoms at
     `n_gas * nitrogen_dopant_fraction`, as the template did then.
   - The moving window: `beta_window` is the template's own formula, not `Simulation`'s.
   - The interaction length: the template's is `z0 + 2.8 alpha`, from `p_zmin=0` to the end of
     the jet. `p_zmin` clips the profiles' extent to that span, but `SimulationHyperparameters`
     adds a `right_buffer` it requires to be positive, left at its default (the 70 um window), so
     the run is about 1% longer.
   - The laser: `LasyLaserPulse`, with `z0` set as in `demo_lasy_laser_simulation`.
   - The runs used 8 MPI ranks on GPUs, which this records only as `use_mpi=True`.
3. Only inputs that have an action (laser energy, pulse duration, focal position, the nitrogen
   dopant fraction and the 12 Zernike coefficients) may differ from the `build_model()`
   baseline, which equals `sim_0000`. Any other input that differs raises an error rather than
   being dropped silently; the dataset's runs vary only five of the 24 inputs.

Usage: `python initial_sample_archives.py [sample_dataset.json] [output dir]`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import typing

from scipy.constants import c

from lume_fbpic.actions import (
    DESCRIPTOR_PREFIX,
    DensityFieldAction,
    DopantFractionAction,
    FinalParticlesAction,
    LaserFieldAction,
    MomentDescriptorAction,
    ZernikeCoefficientAction,
    make_descriptor_actions,
    make_stat_actions,
)
from lume_fbpic.density_profiles import (
    GeneralizedGaussianProfile,
    atom_density,
    nominal_density,
)
from lume_fbpic.lwfa_simulator import LWFASimulator
from lume_fbpic.model import LUMEFBPICModel

from inversion_fbpic.lib.datapoint import Parameters
from inversion_fbpic.lib.laser import LasyLaserPulse
from inversion_fbpic.lib.simulation import SimulationHyperparameters

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[4]
    / "lpa/Simulation_FBPIC/runs/initial_sample/sample_dataset/sample_dataset.json"
)

# --- physical parameters: sim_0000 of sample_dataset.json ---
PEAK_PLASMA_DENSITY_CM3 = 4.0e18
NITROGEN_DOPANT_FRACTION = 0.03
DENSITY_PEAK = 1.1
DENSITY_ALPHA_M = 1.1e-3
DENSITY_BETA = 1.1
DENSITY_CENTER_LOCATION_M = 2.3e-3
LASER_WAVELENGTH_M = 800e-9
LASER_ENERGY_J = 2.5
LASER_PULSE_DURATION_FWHM_S = 30e-15
LASER_SPOT_SIZE_M = 24e-6
LASER_SUPER_GAUSSIAN_ORDER = 3.0
LASER_FOCAL_POSITION_M = 3.5e-3
# --- hyperparameters: the production settings of ionization_injection_runscript_00.py at 5dfff13
NZ, NR, N_AZIMUTHAL_MODES = 1500, 300, 5
ZMIN, ZMAX, RMAX = -70.0e-6, 0.0, 200.0e-6
GAMMA_BOOST = 1.5
NUMBER_DUMPS = 50
WRITE_PERIOD = 50
USE_MPI = True  # submission_script_00.sh runs 8 ranks under srun
P_NZ, P_NR, P_NT = 2, 2, 15
# sim_0000's particle-loading bounds, as passed to `add_new_species()`.
P_ZMIN, P_RMAX = 0.0, 90.0e-6
# The species of the electrons from nitrogen, as the template's particle diagnostic named it (and
# the dataset's `metadata` records).
NITROGEN_ELECTRONS = "nitrogen_electrons"
# --- LASY laser grid ---
LASER_N_AZIMUTHAL_MODES = 5
LASER_NUM_POINTS = (600, 900)
Z0_ANTENNA = 0.0
T_START = 0.0
# The actions of `build_model()` that set an input of sample_dataset.json, by the input's name;
# the Zernike coefficients' actions are named after their inputs, `ZERNIKE_PREFIX` + coefficient.
DATASET_ACTIONS = {
    "laser_energy_J": "laser_energy",
    "laser_focal_position_m": "laser_focal_position",
    "laser_pulse_duration_fwhm_s": "laser_temporal_width",
    "nitrogen_dopant_fraction": "nitrogen_dopant_fraction",
}
ZERNIKE_PREFIX = "zernike_"


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
    model = build_model()
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


def build_model(working_directory: str | None = None) -> LUMEFBPICModel:
    """The dataset's `sim_0000` as a `LUMEFBPICModel`, with `dummy_run=True`: the config the
    dataset's runs were made with (see the module docstring), and the actions that set the inputs
    the runs vary, the run statistics, the final particles and the 33 moment-descriptor
    features."""
    # The template's own v_window formula (n_crit at 800 nm ~= 1.75e27 m^-3, its own
    # approximation), not Simulation's more general calculate_mean_group_beta(). right_buffer is
    # left at its default -- see the module docstring.
    n_plasma = PEAK_PLASMA_DENSITY_CM3 * 1.0e6  # cm^-3 -> m^-3
    n_gas = n_plasma / 2.0
    beta_window = 1.0 - 0.5 * n_gas / 1.75e27
    hyparams = SimulationHyperparameters(
        zmin=ZMIN,
        zmax=ZMAX,
        rmax=RMAX,
        nz=NZ,
        nr=NR,
        nm=N_AZIMUTHAL_MODES,
        use_mpi=USE_MPI,
        number_dumps=NUMBER_DUMPS,
        write_period=WRITE_PERIOD,
        gamma_boost=GAMMA_BOOST,
        field_diagnostics=["rho", "E", "B"],
        random_seed=0,
        beta_window=beta_window,
    )
    laser = LasyLaserPulse(
        energy=LASER_ENERGY_J,
        wavelength=LASER_WAVELENGTH_M,
        tau_fwhm=LASER_PULSE_DURATION_FWHM_S,
        waist=LASER_SPOT_SIZE_M,
        focal_position=LASER_FOCAL_POSITION_M,
        super_gaussian_order=LASER_SUPER_GAUSSIAN_ORDER,
        zernike_coefficients={},  # all zero -- sim_0000's own baseline
        z0=Z0_ANTENNA - c * (T_START + 3 * LASER_PULSE_DURATION_FWHM_S),
        z0_antenna=Z0_ANTENNA,
        t_start=T_START,
        n_azimuthal_modes=LASER_N_AZIMUTHAL_MODES,
        num_points=LASER_NUM_POINTS,
        hi_range=10.0,  # sim_0000's runscript overrides the template's 8.0
        center_and_remove_tilt=True,
        centering_angles=72,
        lasy_file=Path("diags") / "lasy_laser",
    )
    # Two profiles (see the module docstring), of the same shape, with the He and N atom
    # densities of the gas.
    density_he = GeneralizedGaussianProfile(
        nominal_density=nominal_density("He", n_gas * (1.0 - NITROGEN_DOPANT_FRACTION)),
        species="He",
        ionization=0,
        p_zmin=P_ZMIN,
        p_rmax=P_RMAX,
        p_nz=P_NZ,
        p_nr=P_NR,
        p_nt=P_NT,
        gauss_peak=DENSITY_PEAK,
        z0=DENSITY_CENTER_LOCATION_M,
        alpha=DENSITY_ALPHA_M,
        beta=DENSITY_BETA,
        elec_name=None,  # never diagnosed upstream either
    )
    density_n = GeneralizedGaussianProfile(
        nominal_density=nominal_density("N", n_gas * NITROGEN_DOPANT_FRACTION),
        species="N",
        ionization=0,
        p_zmin=P_ZMIN,
        p_rmax=P_RMAX,
        p_nz=P_NZ,
        p_nr=P_NR,
        p_nt=P_NT,
        gauss_peak=DENSITY_PEAK,
        z0=DENSITY_CENTER_LOCATION_M,
        alpha=DENSITY_ALPHA_M,
        beta=DENSITY_BETA,
        elec_name=NITROGEN_ELECTRONS,
        elec_select={"uz": [10.0, None]},  # the template's particle diagnostic
    )
    simulator = LWFASimulator(
        hyparams,
        laser,
        [density_he, density_n],
        target_species=NITROGEN_ELECTRONS,
        working_directory=working_directory,
    )
    actions = [
        LaserFieldAction(name="laser_energy", field_name="energy"),
        LaserFieldAction(name="laser_focal_position", field_name="focal_position"),
        LaserFieldAction(name="laser_temporal_width", field_name="tau_fwhm"),
        DensityFieldAction(
            name="nitrogen_dopant_density",
            density_index=1,
            field_name="nominal_density",
        ),
        # Scan variable `nitrogen_dopant_fraction` of sample_dataset.json: rewrites BOTH gas
        # densities at constant total atom density (profile 0 = He, profile 1 = N).
        DopantFractionAction(
            name="nitrogen_dopant_fraction", host_index=0, dopant_index=1
        ),
        # One action per Zernike coefficient, named like the dataset's inputs (for example
        # `zernike_astigmatism_4`, `zernike_coma_x`); phase amplitudes in radians.
        *[
            ZernikeCoefficientAction(
                name=ZERNIKE_PREFIX + coefficient, coefficient=coefficient
            )
            for coefficient in laser.zernike_coefficients
        ],
        *make_stat_actions(),
        FinalParticlesAction(name="final_particles", read_only=True),
        # The 33 moment-descriptor scalars build_dataset.py writes (uz >= 30, central 95%).
        *make_descriptor_actions(),
    ]
    return LUMEFBPICModel(simulator, actions, dummy_run=True)


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
        if name in DATASET_ACTIONS:
            values[DATASET_ACTIONS[name]] = value
        elif name.startswith(ZERNIKE_PREFIX):
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
        - set(DATASET_ACTIONS)
        - {k for k in inputs if k.startswith(ZERNIKE_PREFIX)}
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


if __name__ == "__main__":
    main()
