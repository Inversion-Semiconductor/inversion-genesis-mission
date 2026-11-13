"""Real fbpic runs small enough for a unit test, a fraction of a second each. Tests use them to run
a simulator or a model for real instead of mocking the run.

- PWFA: 3 steps of a flat-top driver (a uniform `ExpressionParticleBeam`) in a bare-electron
  plasma on a 40 x 20 grid. The driver is in the last dump, so a run reads back a bunch.
- A beam: a Gaussian electron beam drifting 20 um through vacuum on a 16 x 8 grid, with no
  laser and no plasma, and optionally external fields. A run reads the beam back.
- LWFA: a laser through 40 um of plasma on an 8 x 8 grid, 40 steps. A run lasts until the window
  has passed the plasma, and nothing is trapped in so short a run, so the bunch read back is
  empty.

All run on the CPU (`use_cuda=False`), whatever the machine has, so that they behave the same
everywhere; `on_cpu()` does the same for a model built by an example, whose default is a GPU when
there is one.
"""

from __future__ import annotations

import attrs
from inversion_fbpic.lib.density_profiles import SmoothSineFlattop
from inversion_fbpic.lib.laser import GaussianLaserPulse
from inversion_fbpic.lib.particle_beam import (
    ExpressionParticleBeam,
    GaussianParticleBeam,
)
from inversion_fbpic.lib.simulation import SimulationHyperparameters
from scipy.constants import e, pi

from downramp_actions import make_actions
from lume_fbpic.actions import (
    DensityFieldAction,
    FinalParticlesAction,
    make_beam_actions,
    make_stat_actions,
)
from lume_fbpic.density_profiles import LinearRampFlattop
from lume_fbpic.simulator import FBPICSimulator
from lume_fbpic.model import LUMEFBPICModel


def short_beam_simulator(
    directory, *, external_fields=(), **beam_changes
) -> FBPICSimulator:
    """A run of a Gaussian beam (`particle_name="beam"`) drifting 20 um through vacuum, working in
    `directory`, with `external_fields` and the beam's fields changed by `beam_changes`. There is no
    laser and no density profile, so `simulation_length` sets the length of the run."""
    hyparams = SimulationHyperparameters(
        zmin=-1.0e-5,
        zmax=0.0,
        rmax=1.0e-5,
        nz=16,
        nr=8,
        nm=1,
        use_mpi=False,
        use_cuda=False,
        number_dumps=2,
        simulation_length=2.0e-5,
        random_seed=0,
    )
    beam = GaussianParticleBeam(
        **{
            "sig_r": 1.0e-6,
            "sig_z": 1.0e-6,
            "n_emit": 1.0e-7,
            "gamma0": 100.0,
            "sig_gamma": 1.0,
            "n_physical_particles": 1.0e6,
            "n_macroparticles": 200,
            "zf": -5.0e-6,
            "particle_name": "beam",
            **beam_changes,
        }
    )
    return FBPICSimulator(
        hyparams,
        beams=[beam],
        external_fields=list(external_fields),
        target_species="beam",
        working_directory=directory,
    )


def short_lwfa_model(directory, *, dummy_run: bool = False) -> LUMEFBPICModel:
    """A model of `short_lwfa_simulator(directory)` with the downramp actions (`laser_energy`,
    ...); by default `set()` runs it."""
    simulator = short_lwfa_simulator(directory)
    return LUMEFBPICModel(simulator, make_actions(simulator), dummy_run=dummy_run)


def short_lwfa_simulator(directory) -> FBPICSimulator:
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
        use_cuda=False,
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
    return FBPICSimulator(
        hyparams,
        lasers=[laser],
        densities=densities,
        target_species="electrons_flattop",
        working_directory=directory,
    )


def on_cpu(model: LUMEFBPICModel) -> LUMEFBPICModel:
    """`model`, its simulator set to run on the CPU (`use_cuda=False`)."""
    s = model.simulator
    s.hyparams = attrs.evolve(s.hyparams, use_cuda=False)
    return model


def short_pwfa_model(directory, *, dummy_run: bool = False) -> LUMEFBPICModel:
    """A model of `short_pwfa_simulator(directory)` with the plasma density (`plasma_density`), the
    driver's actions (`driver_gamma`, `driver_density`, ...), the statistics and the final
    particles; by default `set()` runs it."""
    simulator = short_pwfa_simulator(directory)
    actions = [
        DensityFieldAction(
            name="plasma_density", density_index=0, field_name="nominal_density"
        ),
        *make_beam_actions(simulator),
        *make_stat_actions(),
        FinalParticlesAction(name="final_particles", read_only=True),
    ]
    return LUMEFBPICModel(simulator, actions, dummy_run=dummy_run)


def short_pwfa_simulator(directory, *, n_steps: int = 3, **changes) -> FBPICSimulator:
    """A configured `n_steps`-step run of a flat-top driver (`particle_name="driver"`) in a
    bare-electron plasma, working in `directory`, with three dumps: the start, the middle and the
    end. `changes` are changes to its `SimulationHyperparameters`."""
    dz = 1.0e-6  # 40 cells over the 40 um window
    hyparams = SimulationHyperparameters(
        **{
            "zmin": -10.0e-6,
            "zmax": 30.0e-6,
            "nz": 40,
            "rmax": 20.0e-6,
            "nr": 20,
            "nm": 1,
            "use_mpi": False,
            "use_cuda": False,
            "number_dumps": 3,
            # POSIT: the lab-frame step is one cell, dz / c, and the window moves at c; half a cell
            # of right_buffer keeps the step count from rounding down
            "simulation_length": n_steps * dz,
            "right_buffer": 0.5 * dz,
            "beta_window": 1.0,
            "field_diagnostics": ["rho", "E", "B"],
            **changes,
        }
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
    driver = ExpressionParticleBeam(
        gamma0=2.0e3,
        n=5.0e24,
        p_zmin=15.0e-6,
        p_zmax=20.0e-6,
        p_rmax=2.0e-6,
        particle_name="driver",
    )
    simulator = FBPICSimulator(
        hyparams,
        densities=[plasma],
        beams=[driver],
        target_species="driver",
        working_directory=directory,
    )
    simulator.configure()
    return simulator


def uniform_beam_charge(beam: ExpressionParticleBeam) -> float:
    """The charge [C] of a uniform `ExpressionParticleBeam` (no `dens_func`), a positive
    magnitude: its density times the volume of its cylinder."""
    return (
        e
        * beam.n
        * pi
        * (beam.p_rmax**2 - beam.p_rmin**2)
        * (beam.p_zmax - beam.p_zmin)
    )
