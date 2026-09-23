"""
Optimas simulation template.

This template mirrors `../demo_downramp_simulation/run_simulation.py`
but exposes parameters for Optimas. Fill in the placeholders when launching
an Optimas job.
"""

from inversion_fbpic.utils.simulation_setup_tools import calculate_acceleration_gradient
from inversion_fbpic.utils.simulation_setup_tools import calculate_plasma_wavelength
from inversion_fbpic.utils.plotting import plot_from_hdf5_series
from inversion_fbpic.utils.make_movie import make_movie
from inversion_fbpic.lib import density_profiles as dn, laser as ls, simulation as sm
from inversion_fbpic.utils import analysis

from scipy.constants import c
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from mpi4py import MPI

# use MPI in FBPIC (Optimas will still use MPI)
USE_MPI = False

# simulation resolution placeholders
NZ = 2048
NR = 500
NM = 3
GAMMA_BOOST = 4.0
NUMBER_DUMPS = 2  # could also be a parameter for an Optimas script to set, for a higher-quality final movie
MAKE_MOVIE = False

if __name__ == "__main__":
    MPI_RANK = MPI.COMM_WORLD.Get_rank()

    # --- Optimas placeholders (match variables used in template.py) ---
    TARGET_ENERGY = 430e6  # eV
    LASER_ENERGY = {{laser_energy}}  # J #noqa: F821
    WAVELENGTH = 800e-9  # m
    LASER_WAIST = {{laser_waist}}  # m #noqa: F821
    TAU_FWHM = {{tau_fwhm_fs}} * 1e-15  # s #noqa: F821
    FLATTOP_PLASMA_DENSITY = (
        {{flattop_plasma_density_e18}} * 1e18 * 1e6  # noqa: F821
    )  # m^-3
    DOWNRAMP_LENGTH = {{downramp_length}}  # m #noqa: F821
    DOWNRAMP_POSITION = {{downramp_position}}  # m #noqa: F821
    DOWNRAMP_HEIGHT = {{downramp_height}}  # unitless #noqa: F821
    FOCAL_POSITION = {{focal_position}}  # m #noqa: F821

    DIAGS_DIR = "diags"
    PLOTS_DIR = Path("plots")

    # compute WINDOW_SIZE and LASER_CENTROID like template.py
    WINDOW_SIZE = max(
        3.0 * calculate_plasma_wavelength(FLATTOP_PLASMA_DENSITY), 6 * TAU_FWHM * c
    )  # m
    LASER_CENTROID = -WINDOW_SIZE / 3.0  # m

    # --- build laser like the single-sim example ---
    laser = ls.GaussianLaserPulse(
        energy=LASER_ENERGY,
        wavelength=WAVELENGTH,
        waist=LASER_WAIST,
        tau_fwhm=TAU_FWHM,
        z0=LASER_CENTROID,
        cep=0.0,
        focal_position=FOCAL_POSITION,
        polarization=0.0,
    )

    # compute a flattop length from the desired target energy (mirrors single_sim)
    accel_gradient = calculate_acceleration_gradient(laser.a0, FLATTOP_PLASMA_DENSITY)
    flattop_length = TARGET_ENERGY / accel_gradient
    print(f"Computed flattop length: {flattop_length*1e6:.3f} um")

    # build density profiles using SmoothSineFlattop as in the single example
    flattop_profile = dn.SmoothSineFlattop(
        nominal_density=FLATTOP_PLASMA_DENSITY,
        p_nz=2,
        p_nr=2,
        p_nt=4,
        elec_name="electrons_flattop",
        elec_select={"uz": [10.0, None]},
        flattop_width=flattop_length,
        upramp_length=100e-6,
        downramp_length=100e-6,
    )

    downramp_profile = dn.SmoothSineFlattop(
        nominal_density=FLATTOP_PLASMA_DENSITY * DOWNRAMP_HEIGHT,
        p_nz=2,
        p_nr=2,
        p_nt=4,
        elec_name="electrons_downramp",
        elec_select={"uz": [10.0, None]},
        upramp_length=250e-6,
        flattop_width=DOWNRAMP_POSITION - 250e-6,
        downramp_length=DOWNRAMP_LENGTH,
        offset_length=0.0,
    )

    # choose hyperparameters - mirror the single simulation
    hyparams = sm.SimulationHyperparameters(
        zmin=-WINDOW_SIZE,
        zmax=0.0,
        rmax=120e-6,
        nz=NZ,
        nr=NR,
        nm=NM,
        use_mpi=USE_MPI,
        number_dumps=NUMBER_DUMPS,
        gamma_boost=GAMMA_BOOST,
        field_diagnostics=["E", "B", "rho"],
    )

    # save YAMLs for recordkeeping (only on rank 0 in MPI setups)
    if not USE_MPI or MPI_RANK == 0:
        os.makedirs("cfgs", exist_ok=True)
        hyparams.to_yaml_file("cfgs/hyparams.yaml")
        hyparams.grid_parameters_yaml("cfgs/grid_parameters.yaml")
        laser.to_yaml_file("cfgs/laser.yaml")
        flattop_profile.to_yaml_file("cfgs/flattop_profile.yaml")
        downramp_profile.to_yaml_file("cfgs/downramp_profile.yaml")

        # plot density profiles for quick inspection
        fig, ax = plt.subplots()
        flattop_profile.plot_z_profile(ax=ax, label="Flattop", num=600)
        downramp_profile.plot_z_profile(ax=ax, label="Downramp", num=600)
        ax.legend()
        ax.grid()
        ax.set_xlabel("z (m)")
        ax.set_ylabel("Density (m^-3)")
        ax.set_title("Density Profiles")
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "density_profiles.png")
        plt.close()

    # build and run the simulation (mirrors the single-sim flow)
    sim = sm.Simulation(elements=[hyparams, laser, flattop_profile, downramp_profile])
    sim.setup_simulation(working_directory=Path(__file__).parent)
    sim.run_simulation(logger_friendly_progress=True)

    if not USE_MPI or MPI_RANK == 0:
        # generate a small rho movie (same helper as single-sim)
        if MAKE_MOVIE:
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
                movie = make_movie(
                    stills_dir, prefix, filename=field, movie_dir=PLOTS_DIR
                )

        # do a beam analysis
        try:
            bd1 = analysis.load_beam_data(Path(DIAGS_DIR) / "hdf5", "electrons_flattop")
            bd2 = analysis.load_beam_data(
                Path(DIAGS_DIR) / "hdf5", "electrons_downramp"
            )
            bds = list(bd1[:7])
            for i, (e1, e2) in enumerate(zip(bd1[:7], bd2[:7])):
                bds[i] = np.concatenate((e1, e2), axis=0)
            ba = analysis.analyze_beam(*bds, bd1[7])
            analysis.print_beam_summary(ba, PLOTS_DIR / "beam_summary.txt")
            analysis.plot_beam_analysis(
                *bds, ba, save_path=PLOTS_DIR / "beam_analysis.png"
            )
        except Exception as e:
            print(f"Beam analysis failed: {e}")
