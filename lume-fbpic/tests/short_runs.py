"""Real fbpic runs small enough for a unit test, a fraction of a second each. Tests use them to run
a simulator or a model for real instead of mocking the run.

- PWFA: 3 steps of a flat-top driver in a bare-electron plasma on a 40 x 20 grid. The driver is
  in the last dump, so a run reads back a bunch.
- LWFA: a laser through 40 um of plasma on an 8 x 8 grid, 40 steps. A run lasts until the window
  has passed the plasma, and nothing is trapped in so short a run, so the bunch read back is
  empty.
"""

from __future__ import annotations

from inversion_fbpic.lib.density_profiles import SmoothSineFlattop
from inversion_fbpic.lib.laser import GaussianLaserPulse
from inversion_fbpic.lib.simulation import SimulationHyperparameters

from downramp_actions import make_actions
from lume_fbpic.actions import make_pwfa_actions
from lume_fbpic.density_profiles import LinearRampFlattop
from lume_fbpic.lwfa_simulator import LWFASimulator
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.pwfa_config import FlatTopBunch, PWFAGrid
from lume_fbpic.pwfa_simulator import PWFASimulator


def short_lwfa_model(directory, *, dummy_run: bool = False) -> LUMEFBPICModel:
    """A model of `short_lwfa_simulator(directory)` with the downramp actions (`laser_energy`,
    ...); by default `set()` runs it."""
    simulator = short_lwfa_simulator(directory)
    return LUMEFBPICModel(simulator, make_actions(simulator), dummy_run=dummy_run)


def short_lwfa_simulator(directory) -> LWFASimulator:
    """A 40-step LWFA run working in `directory`: a laser through a 20 um flat-top and a 20 um
    downramp section, each with 10 um ramps, on an 8 x 8 grid, with the downramp profiles the
    downramp actions expect."""
    hyparams = SimulationHyperparameters(
        zmin=-1.0e-5,
        zmax=0.0,
        rmax=1.0e-5,
        nz=8,
        nr=8,
        nm=1,
        use_mpi=False,
        number_dumps=2,
    )
    laser = GaussianLaserPulse(
        energy=5.0,
        z0=-3.0e-5,
        wavelength=8.0e-7,
        tau_fwhm=3.8e-14,
        cep=0.0,
        waist=2.8e-5,
        focal_position=3.0e-3,
        polarization=0.0,
    )
    section = dict(
        p_nz=1,
        p_nr=1,
        p_nt=1,
        flattop_width=2.0e-5,
        upramp_length=1.0e-5,
        downramp_length=1.0e-5,
    )
    densities = [
        SmoothSineFlattop(
            nominal_density=1.0e24, elec_name="electrons_flattop", **section
        ),
        SmoothSineFlattop(
            nominal_density=0.3e24, elec_name="electrons_downramp", **section
        ),
    ]
    return LWFASimulator(
        hyparams,
        laser,
        densities,
        target_species="electrons_flattop",
        working_directory=directory,
    )


def short_pwfa_model(directory, *, dummy_run: bool = False) -> LUMEFBPICModel:
    """A model of `short_pwfa_simulator(directory)` with the PWFA actions; by default `set()` runs
    it."""
    simulator = short_pwfa_simulator(directory)
    return LUMEFBPICModel(simulator, make_pwfa_actions(simulator), dummy_run=dummy_run)


def short_pwfa_simulator(directory, **grid_changes) -> PWFASimulator:
    """A configured 3-step run of a flat-top driver in a bare-electron plasma, working in
    `directory`, with dumps at steps 0 and 2 and at the end, step 3."""
    grid = PWFAGrid(
        zmin=-10.0e-6,
        zmax=30.0e-6,
        nz=40,
        rmax=20.0e-6,
        nr=20,
        n_steps=3,
        write_period=2,
        **grid_changes,
    )
    plasma = LinearRampFlattop(
        nominal_density=1.0e24,
        species=None,
        ionization=-1,
        p_nz=2,
        p_nr=2,
        p_nt=4,
        ramp_start=0.0,
        ramp_length=5.0e-5,
    )
    driver = FlatTopBunch(
        gamma=2.0e3, density=5.0e24, radius=2.0e-6, zmin=15.0e-6, zmax=20.0e-6
    )
    simulator = PWFASimulator(
        grid, plasma, driver, target_species="driver", working_directory=directory
    )
    simulator.configure()
    return simulator
