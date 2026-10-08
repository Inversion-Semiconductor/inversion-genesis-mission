from inversion_fbpic.lib import density_profiles as dn, laser as ls, simulation as sm
from inversion_fbpic.lib.config_container import ConfigContainer
from inversion_fbpic.lib.datapoint import Parameters
from inversion_fbpic.lib.diagnostics import MomentDescriptorDiagnostic
from mpi4py import MPI
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
import os
from typing import Literal
from pathlib import Path
from inversion_fbpic.utils.simulation_setup_tools import (
    calculate_acceleration_gradient,
    calculate_plasma_wavelength,
)
from inversion_fbpic.utils.plotting import plot_from_hdf5_series
from inversion_fbpic.utils.make_movie import make_movie
from inversion_fbpic.utils import analysis


MPI_SIZE = MPI.COMM_WORLD.Get_size()
MPI_RANK = MPI.COMM_WORLD.Get_rank()

CONFIG_TYPE: Literal["h5", "yaml", "json"] = "h5"

if MPI_SIZE <= 1:
    USE_MPI = False
else:
    USE_MPI = True


def main() -> None:
    RUN_DIR = Path(__file__).resolve().parent
    DIAGS_DIR = RUN_DIR / "diags"
    PLOTS_DIR = RUN_DIR / "plots"
    CONFIGS_DIR = RUN_DIR / "cfgs"
    RESULTS_FILE = RUN_DIR / "results.h5"
    # define parameters
    TARGET_ENERGY = 430e6  # eV
    LASER_ENERGY = 4.5  # J
    WAVELENGTH = 800e-9  # m
    LASER_WAIST = 30e-6  # m
    TAU_FWHM = 40e-15  # s
    FLATTOP_PLASMA_DENSITY = 1.0e18 * 1e6  # m^-3
    NEUTRAL_DOPANT_FRACTION = 0.03  # unitless

    WINDOW_SIZE = max(
        3.0 * calculate_plasma_wavelength(FLATTOP_PLASMA_DENSITY), 6 * TAU_FWHM * c
    )  # m
    LASER_CENTROID = -2 * TAU_FWHM * c  # m

    laser = ls.GaussianLaserPulse(
        energy=LASER_ENERGY,
        wavelength=WAVELENGTH,
        waist=LASER_WAIST,
        tau_fwhm=TAU_FWHM,
        z0=LASER_CENTROID,
        cep=0.0,
        focal_position=3.0e-3,
        polarization=0.0,
    )

    if laser.a0 is not None:
        accel_gradient = calculate_acceleration_gradient(
            laser.a0, FLATTOP_PLASMA_DENSITY
        )
    else:
        raise ValueError(
            "Laser a0 was not evaluated. Please check the laser parameters."
        )

    flattop_length = TARGET_ENERGY / accel_gradient
    print(f"Flattop length: {flattop_length * 1e6} um")

    background_profile = dn.SmoothSineFlattop(
        nominal_density=FLATTOP_PLASMA_DENSITY,
        p_nz=1,
        p_nr=1,
        p_nt=2,
        species="He",
        elec_name="electrons_He",
        elec_select={"uz": [10.0, None]},
        flattop_width=flattop_length,
        upramp_length=100e-6,
        downramp_length=100e-6,
    )

    dopant_profile = dn.SmoothSineFlattop(
        nominal_density=FLATTOP_PLASMA_DENSITY * NEUTRAL_DOPANT_FRACTION * 7.0 / 2.0,
        p_nz=4,
        p_nr=4,
        p_nt=8,
        species="N",
        elec_name="electrons_N",
        elec_select={"uz": [10.0, None]},
        flattop_width=flattop_length,
        upramp_length=100e-6,
        downramp_length=100e-6,
    )

    hyparams = sm.SimulationHyperparameters(
        zmin=-WINDOW_SIZE,
        zmax=0.0,
        rmax=120e-6,
        nz=1024,
        nr=300,
        nm=2,
        use_mpi=USE_MPI,
        number_dumps=100,
        gamma_boost=2,
        field_diagnostics=["E", "B", "rho"],
    )

    physical_parameters = Parameters(
        data={
            "target_energy_eV": TARGET_ENERGY,
            "laser_energy_J": LASER_ENERGY,
            "wavelength_m": WAVELENGTH,
            "laser_waist_m": LASER_WAIST,
            "tau_fwhm_s": TAU_FWHM,
            "flattop_plasma_density_m3": FLATTOP_PLASMA_DENSITY,
            "dopant_plasma_density_m3": dopant_profile.nominal_density,
            "neutral_dopant_fraction": NEUTRAL_DOPANT_FRACTION,
            "electron_diagnostic_selection": background_profile.elec_select,
            "background_species": background_profile.species,
            "dopant_species": dopant_profile.species,
            "focal_position_m": laser.focal_position,
            "cep_rad": laser.cep,
            "polarization_rad": laser.polarization,
            "upramp_length_m": background_profile.upramp_length,
            "downramp_length_m": background_profile.downramp_length,
            "radial_extent_m": hyparams.rmax,
            "gamma_boost": hyparams.gamma_boost,
            "window_size_m": WINDOW_SIZE,
            "laser_centroid_m": LASER_CENTROID,
            "laser_a0": laser.a0,
            "acceleration_gradient_eV_per_m": accel_gradient,
            "flattop_length_m": flattop_length,
        }
    )
    beam_moments = MomentDescriptorDiagnostic(selection=("all_of_species", "e"))
    results = ConfigContainer(configs=[physical_parameters, beam_moments])

    # plot density profiles
    if not USE_MPI or MPI_RANK == 0:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        fig, ax = plt.subplots()
        background_profile.plot_z_profile(ax=ax, label="Flattop", num=600)
        dopant_profile.plot_z_profile(ax=ax, label="Downramp", num=600)
        ax.legend()
        ax.grid()
        ax.set_xlabel("z (m)")
        ax.set_ylabel("Density (m^-3)")
        ax.set_title("Density Profiles")
        plt.savefig(PLOTS_DIR / "density_profiles.png")

    sim = sm.Simulation(
        elements=[hyparams, laser, background_profile, dopant_profile, results]
    )

    # save configs
    if not USE_MPI or MPI_RANK == 0:
        os.makedirs(CONFIGS_DIR, exist_ok=True)
        hyparams.grid_parameters_yaml(str(CONFIGS_DIR / "grid_parameters.yaml"))
        if CONFIG_TYPE == "yaml":
            hyparams.to_yaml_file(CONFIGS_DIR / "hyparams.yaml")
            laser.to_yaml_file(CONFIGS_DIR / "laser.yaml")
            background_profile.to_yaml_file(CONFIGS_DIR / "flattop_profile.yaml")
            dopant_profile.to_yaml_file(CONFIGS_DIR / "downramp_profile.yaml")
        elif CONFIG_TYPE == "h5":
            sim.to_hdf5_file(
                CONFIGS_DIR / "simulation.h5", include_nones=False, overwrite=True
            )
        elif CONFIG_TYPE == "json":
            sim.to_json_file(CONFIGS_DIR / "simulation.json", include_nones=False)
        else:
            raise ValueError(f"Unknown config type: {CONFIG_TYPE}")

    sim.setup_simulation(working_directory=RUN_DIR)
    sim.run_simulation()

    # analysis
    if not USE_MPI or MPI_RANK == 0:
        if beam_moments.analysis_complete:
            results.to_hdf5_file(RESULTS_FILE, overwrite=True)

        # plot some movies
        for field in ["rho", "eme"]:
            stills_dir, prefix = plot_from_hdf5_series(
                Path(DIAGS_DIR) / "hdf5",
                PLOTS_DIR / "stills",
                field_name=field if field == "rho" else None,
                component=field if field == "eme" else None,
                vminmax=(0, 1e18) if field == "rho" else None,
                cmap="magma",
            )
            make_movie(stills_dir, prefix, filename=field, movie_dir=PLOTS_DIR)

        # do a beam analysis
        bd1 = analysis.load_beam_data(Path(DIAGS_DIR) / "hdf5", "electrons_He")
        bd2 = analysis.load_beam_data(Path(DIAGS_DIR) / "hdf5", "electrons_N")
        bds = list(bd1[:7])
        for i, (e1, e2) in enumerate(zip(bd1[:7], bd2[:7])):
            bds[i] = np.concatenate((e1, e2), axis=0)
        ba = analysis.analyze_beam(*bds, bd1[7])
        analysis.print_beam_summary(ba, PLOTS_DIR / "beam_summary.txt")
        analysis.plot_beam_analysis(*bds, ba, save_path=PLOTS_DIR / "beam_analysis.png")


if __name__ == "__main__":
    main()
