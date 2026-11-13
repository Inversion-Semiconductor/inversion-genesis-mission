"""lume_fbpic conversion of fbpic's `boosted_frame_script.py`
(fbpic/docs/source/example_input/boosted_frame_script.py): a boosted-frame (gamma 10) LWFA in a
guiding plasma channel, with an externally injected electron bunch accelerated in the wake.

- Laser: a Gaussian pulse (a0 = 2, 50 um waist, 16 fs, 0.8 um), injected by an antenna at z = 0
  (`GaussianLaserPulse(method="antenna")`), focused at z = 0.
- Plasma: fully ionized hydrogen (`species="H"`, `ionization=-1`: electrons and mobile protons) at
  3e24 m^-3, a 0.5 mm up-ramp, a 3.5 mm plateau and a 0.5 mm down-ramp, times the parabolic
  channel `1 + r^2 / (pi r_e w^4 n_e)` matched to the 50 um waist. It is one
  `ExpressionDensityProfile`, the original's `dens_func` written as a `MathExpression`.
- Bunch: the original's flat-top bunch (`add_particle_bunch`), gamma 400 and 5e23 m^-3, 3 um long
  and 10 um in radius, 15 um behind the laser: a uniform `ExpressionParticleBeam`
  (`particle_name="bunch"`), the `target_species` read back.
- Boost: `gamma_boost=10`; `Simulation` sets the time step as the original does,
  `min(rmax / (2 gamma Nr) / c, dz / c)`, and the Galilean frame and the back-transformed lab-frame
  dumps (`number_dumps`, the original's 11) itself.

Substitutions made in the conversion:

- Moving window: `beta_window` is the original's `1 - 0.5 n_e / 1.75e27`. Left to `Simulation`, it
  would take the faster of the laser's group velocity and the bunch's (gamma 400), the bunch's.
- Interaction length: the original runs the window across the plasma, 4.5 mm; `Simulation` adds a
  `right_buffer`, by default the window length (30 um): 619 boosted steps against the original's
  617. The lab-frame dumps are at the same times.
- Diagnostics: only the back-transformed lab-frame dumps, of the fields (`rho`, `E`, `B`) and of
  the bunch with `uz >= 0` (the original's `select`); the original's boosted-frame dumps and its
  plasma-electron output are left out.
- The antenna: the original leaves `v_antenna` at fbpic's default, 0. `inversion_fbpic`'s default
  is None, which fbpic refuses; `FBPICSimulator` runs an unset `v_antenna` as 0.
- `use_cuda` is left at its default, True, as in the original: a GPU when there is one, the CPU
  otherwise.

The model's inputs are the laser's `a0` and focus, the plasma density and the bunch's fields
(`make_beam_actions()`: `bunch_gamma`, `bunch_density`, `bunch_radius`, ...); its outputs the
bunch's charge and energy statistics and its final particles. `build_model()`'s `nz`, `nr` and
`plateau` make a smaller run (the tests use them).

Run against the original, both on the CPU (the original's `use_cuda` set to False; the machine had
no GPU), on 8 threads each, about 4 min 50 s each, the bunch in the 11 lab-frame dumps agreed at
every stage:

                          original                      this example
    start                 4800, 75.55 pC, 203.87 MeV    4800, 75.55 pC, 203.87 MeV
    middle, z = 2.24 mm   341.48 MeV, spread 15.31      341.48 MeV, spread 15.31
    end, z = 4.51 mm      3349, 55.93 pC                3348, 55.91 pC
                          71.41 MeV, spread 40.16       71.42 MeV, spread 40.15

(macroparticles with `uz >= 0`, charge, mean kinetic energy and its spread in MeV.) The bunch
gains energy over the first half of the plasma, then dephases and is decelerated, and part of it
falls out of the `uz >= 0` selection: the original's physics, not the conversion's.
"""

from __future__ import annotations

import math

from inversion_fbpic.lib.density_profiles import ExpressionDensityProfile
from inversion_fbpic.lib.laser import GaussianLaserPulse
from inversion_fbpic.lib.math_expression import MathExpression
from inversion_fbpic.lib.particle_beam import ExpressionParticleBeam
from inversion_fbpic.lib.simulation import SimulationHyperparameters
from inversion_fbpic.utils.simulation_setup_tools import (
    LASER_TAU_TO_SIGMA,
    SIGMA_TO_FWHM_INTENSITY,
)
from lume_fbpic.actions import (
    DensityFieldAction,
    FinalParticlesAction,
    LaserFieldAction,
    make_beam_actions,
    make_stat_actions,
)
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.simulator import FBPICSimulator

# --- the box and the boost: the original's ---
GAMMA_BOOST = 10.0
NZ, ZMIN, ZMAX = 600, -30.0e-6, 0.0
NR, RMAX = 75, 150.0e-6
NM = 2
NUMBER_DUMPS = 11  # the original's N_lab_diag
# --- the laser ---
A0 = 2.0
WAIST = 50.0e-6
UPSTREAM_TAU = (
    16.0e-15  # the original's tau; as in lwfa.py, converted exactly to a FWHM
)
TAU_FWHM = UPSTREAM_TAU * SIGMA_TO_FWHM_INTENSITY * LASER_TAU_TO_SIGMA
LASER_Z0 = -10.0e-6
LASER_FOCUS = 0.0
WAVELENGTH = 0.8e-6
# --- the plasma ---
N_E = 3.0e24  # [m^-3]
RAMP_UP, PLATEAU, RAMP_DOWN = 0.5e-3, 3.5e-3, 0.5e-3
P_RMAX = 100.0e-6
W_MATCHED = 50.0e-6
# the original's 2.81e-15 for the classical electron radius
CHANNEL = 1.0 / (math.pi * 2.81e-15 * W_MATCHED**4 * N_E)  # [m^-2]
# --- the bunch: 15 um behind the laser's centroid, 3 um long ---
BUNCH_GAMMA = 400.0
BUNCH_N = 5.0e23
BUNCH_ZMIN = LASER_Z0 - 15.0e-6
BUNCH_LENGTH = 3.0e-6
BUNCH_RMAX = 10.0e-6


def build_model(
    working_directory: str | None = None,
    *,
    nz: int = NZ,
    nr: int = NR,
    plateau: float = PLATEAU,
) -> LUMEFBPICModel:
    """The model of the module docstring; `nz`, `nr` and `plateau` [m] shrink it."""
    hyparams = SimulationHyperparameters(
        zmin=ZMIN,
        zmax=ZMAX,
        nz=nz,
        rmax=RMAX,
        nr=nr,
        nm=NM,
        use_mpi=False,
        gamma_boost=GAMMA_BOOST,
        beta_window=1.0 - 0.5 * N_E / 1.75e27,  # the original's v_window
        number_dumps=NUMBER_DUMPS,
        field_diagnostics=["rho", "E", "B"],
        r_boundary="reflective",
    )
    laser = GaussianLaserPulse(
        a0=A0,
        waist=WAIST,
        tau_fwhm=TAU_FWHM,
        z0=LASER_Z0,
        focal_position=LASER_FOCUS,
        wavelength=WAVELENGTH,
        cep=0.0,
        polarization=0.0,
        method="antenna",
        z0_antenna=0.0,  # at rest: `FBPICSimulator` runs an unset `v_antenna` as 0, fbpic's default
    )
    length = RAMP_UP + plateau + RAMP_DOWN
    plasma = ExpressionDensityProfile(
        nominal_density=N_E,
        species="H",
        ionization=-1,  # fully ionized: electrons and protons
        function=MathExpression(
            expression="np.clip(np.minimum(z / up, (length - z) / down), 0.0, 1.0)"
            + " * (1.0 + channel * r**2)",
            variables=("z", "r"),
            parameters={
                "up": RAMP_UP,
                "down": RAMP_DOWN,
                "length": length,
                "channel": CHANNEL,
            },
        ),
        z_min=0.0,
        z_max=length,
        p_zmin=0.0,
        p_zmax=length,
        p_rmax=P_RMAX,
        p_nz=2,
        p_nr=2,
        p_nt=6,
    )
    bunch = ExpressionParticleBeam(
        gamma0=BUNCH_GAMMA,
        n=BUNCH_N,
        p_zmin=BUNCH_ZMIN,
        p_zmax=BUNCH_ZMIN + BUNCH_LENGTH,
        p_rmax=BUNCH_RMAX,
        particle_name="bunch",
        particle_select={"uz": [0.0, None]},
    )
    simulator = FBPICSimulator(
        hyparams,
        lasers=[laser],
        densities=[plasma],
        beams=[bunch],
        target_species="bunch",
        working_directory=working_directory,
    )
    actions = [
        LaserFieldAction(name="laser_a0", field_name="a0"),
        LaserFieldAction(name="laser_focal_position", field_name="focal_position"),
        DensityFieldAction(
            name="plasma_density", density_index=0, field_name="nominal_density"
        ),
        *make_beam_actions(simulator),
        *make_stat_actions(),
        FinalParticlesAction(name="final_particles", read_only=True),
    ]
    return LUMEFBPICModel(simulator, actions)


if __name__ == "__main__":
    model = build_model()
    model.set({})  # runs the simulation (an empty set changes nothing first)
    print(model.get(["charge_pc", "energy_mean_mev", "energy_std_mev"]))
