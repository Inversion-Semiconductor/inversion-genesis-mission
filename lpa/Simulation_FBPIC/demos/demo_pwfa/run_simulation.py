"""Run a coarse beam-driven FBPIC demonstration."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e, pi

from inversion_fbpic.lib.density_profiles import SmoothSineFlattop
from inversion_fbpic.lib.particle_beam import GaussianParticleBeam
from inversion_fbpic.lib.simulation import Simulation, SimulationHyperparameters
from inversion_fbpic.utils import analysis
from inversion_fbpic.utils.make_movie import make_movie
from inversion_fbpic.utils.plotting import plot_from_hdf5_series

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

RUN_DIR = Path(__file__).resolve().parent
CONFIG_DIR = RUN_DIR / "cfgs"
DIAGS_DIR = RUN_DIR / "diags"
PLOTS_DIR = RUN_DIR / "plots"

# The downramp is the sum of a primary flattop and a lower-density, delayed
# profile. Their overlap lowers the wake phase velocity for injection.
PLASMA_DENSITY = 1.0e24  # m^-3
FLATTOP_LENGTH = 0.5e-3  # m
DOWNRAMP_LENGTH = 25.0e-6  # m

# Electron-driver parameters. These are intentionally coarse so the example is
# suitable for a small exploratory run rather than a production-quality case.
DRIVER_DENSITY = PLASMA_DENSITY * 2.0
DRIVER_GAMMA = 1_000.0
DRIVER_SIG_R = 4.0e-6
DRIVER_SIG_Z = 4.0e-6
DRIVER_N_EMIT = 1.0e-6
DRIVER_PARTICLE_COUNT = (
    DRIVER_DENSITY * (2.0 * pi) ** (3.0 / 2) * (DRIVER_SIG_R**2 * DRIVER_SIG_Z)
)


def main() -> Simulation:
    """Build, record, and run the beam-driven downramp-injection simulation."""
    comm = MPI.COMM_WORLD if MPI is not None else None
    use_mpi = comm is not None and comm.Get_size() > 1
    is_root = comm is None or comm.Get_rank() == 0

    flattop_profile = SmoothSineFlattop(
        nominal_density=PLASMA_DENSITY,
        p_nz=2,
        p_nr=2,
        p_nt=4,
        species="H",
        elec_name="electrons_flattop",
        elec_select={"uz": [10.0, None]},
        flattop_width=FLATTOP_LENGTH,
        upramp_length=100.0e-6,
        downramp_length=100.0e-6,
    )
    downramp_profile = SmoothSineFlattop(
        nominal_density=PLASMA_DENSITY * 0.3,
        p_nz=2,
        p_nr=2,
        p_nt=4,
        species="H",
        elec_name="electrons_downramp",
        elec_select={"uz": [10.0, None]},
        upramp_length=100.0e-6,
        flattop_width=100.0e-6,
        downramp_length=DOWNRAMP_LENGTH,
    )
    driver = GaussianParticleBeam(
        q=-e,
        sig_r=DRIVER_SIG_R,
        sig_z=DRIVER_SIG_Z,
        n_emit=DRIVER_N_EMIT,
        gamma0=DRIVER_GAMMA,
        sig_gamma=10.0,
        n_physical_particles=DRIVER_PARTICLE_COUNT,
        n_macroparticles=20_000,
        zf=0.0,
        particle_name="driver_electrons",
        particle_select={"uz": [100.0, None]},
    )
    hyperparameters = SimulationHyperparameters(
        zmin=-50.0e-6,
        zmax=30.0e-6,
        rmax=50.0e-6,
        nz=512,
        nr=300,
        nm=1,
        use_mpi=use_mpi,
        number_dumps=100,
        field_diagnostics=["E", "B", "rho"],
        save_directory="diags",
    )
    simulation = Simulation(
        elements=[hyperparameters, flattop_profile, downramp_profile, driver]
    )

    if is_root:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        simulation.to_hdf5_file(
            CONFIG_DIR / "simulation.h5", include_nones=False, overwrite=True
        )
        figure = simulation.plot_densities()
        figure.savefig(PLOTS_DIR / "plasma_density.png", dpi=150)
        plt.close(figure)
    if comm is not None:
        comm.Barrier()

    simulation.setup_simulation(working_directory=RUN_DIR)
    simulation.run_simulation()

    if is_root:
        for field in ["rho", "eme"]:
            stills_dir, prefix = plot_from_hdf5_series(
                DIAGS_DIR / "hdf5",
                PLOTS_DIR / "stills",
                field_name=field if field == "rho" else None,
                component=field if field == "eme" else None,
                vminmax=(1e15, 1e17) if field == "rho" else None,
                cmap="magma",
                scale="log",
            )
            make_movie(stills_dir, prefix, filename=field, movie_dir=PLOTS_DIR)

        flattop_beam = analysis.load_beam_data(DIAGS_DIR / "hdf5", "electrons_flattop")
        downramp_beam = analysis.load_beam_data(
            DIAGS_DIR / "hdf5", "electrons_downramp"
        )
        beam_data = list(flattop_beam[:7])
        for index, (flattop_data, downramp_data) in enumerate(
            zip(flattop_beam[:7], downramp_beam[:7])
        ):
            beam_data[index] = np.concatenate((flattop_data, downramp_data), axis=0)
        beam_analysis = analysis.analyze_beam(*beam_data, flattop_beam[7])
        analysis.print_beam_summary(beam_analysis, PLOTS_DIR / "beam_summary.txt")
        analysis.plot_beam_analysis(
            *beam_data, beam_analysis, save_path=PLOTS_DIR / "beam_analysis.png"
        )

    return simulation


if __name__ == "__main__":
    main()
