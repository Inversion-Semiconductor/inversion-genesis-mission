"""A particle beam through external magnetic fields in vacuum: no laser and no plasma.

A Gaussian electron beam (`GaussianParticleBeam`, 25 MeV, 160 fC) drifts 0.5 mm through two
external magnetic fields that act on it alone (`species="beam"`):

- a dipole corrector, a `UniformExternalField` with a vertical field `By`, which deflects the
  beam horizontally. Its strength is the action `corrector_by` [T] (`ExternalFieldAction`);
- a quadrupole, an `ExpressionExternalField` with `Bx = G y` and `By = G x` (focusing in y,
  defocusing in x for an electron moving in +z). Its gradient `G` is a fixed parameter of the
  expression: actions do not reach expression parameters.

It shows what `FBPICSimulator` runs without a laser or a density profile:

- `hyparams.simulation_length` sets how far the moving window travels, since there is no plasma
  to take the length from; the run lasts that plus the window's own length (`right_buffer`'s
  default);
- `beta_window` is left to `Simulation`, which takes it from the beam's `gamma0`;
- the beam is written to the diagnostics as `particle_name="beam"`, the `target_species` read
  back, and the external fields name it as theirs.

The model's inputs are the corrector and the beam's fields (`make_beam_actions()`: `beam_gamma`,
`beam_sigma_r`, `beam_charge`, ...); its outputs the run statistics and the final particles.

Running it prints the beam's horizontal deflection next to the dipole's prediction,
`theta = c By L / (p c)` for a travel `L` (`predicted_deflection_rad()`); the quadrupole moves
the centroid much less than that, because the beam starts on axis. The run is small (a 20 um box
on a 32 x 16 grid, about 830 steps, 2000 macroparticles) and takes about 20 s on a 16-core CPU;
it gave 6.065 mrad against a predicted 6.103 mrad. With the corrector off, the quadrupole changes
the beam's x-x' slope by +2.8 /m and its y-y' slope by -2.8 /m, against the thin-lens
`c G L / (p c)` of 3.05 /m. Nothing here models a real magnet: the fields are uniform along z and
have no fringes.
"""

from __future__ import annotations

import math

import numpy
from scipy.constants import c, m_e, e

from inversion_fbpic.lib.external_field import (
    ExpressionExternalField,
    UniformExternalField,
)
from inversion_fbpic.lib.math_expression import MathExpression
from inversion_fbpic.lib.particle_beam import GaussianParticleBeam
from inversion_fbpic.lib.simulation import SimulationHyperparameters
from lume_fbpic.actions import (
    ExternalFieldAction,
    FinalParticlesAction,
    make_beam_actions,
    make_stat_actions,
)
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.simulator import FBPICSimulator

# --- the beam ---
BEAM_GAMMA = 50.0
BEAM_GAMMA_SPREAD = 0.05  # absolute, in units of gamma
BEAM_SIGMA_R_M = 2.0e-6
BEAM_SIGMA_Z_M = 2.0e-6
BEAM_EMITTANCE_M = 1.0e-7  # normalized
BEAM_ELECTRONS = 1.0e6
BEAM_MACROPARTICLES = 2000
BEAM_FOCUS_M = -1.0e-5  # the middle of the box
# --- the fields ---
CORRECTOR_BY_T = 1.0
QUADRUPOLE_GRADIENT_T_PER_M = 500.0
# --- the box and the run ---
ZMIN, ZMAX, RMAX = -2.0e-5, 0.0, 2.0e-5
NZ, NR = 32, 16
SIMULATION_LENGTH_M = 5.0e-4
NUMBER_DUMPS = 2


def build_model(
    working_directory: str | None = None,
    *,
    simulation_length: float = SIMULATION_LENGTH_M,
) -> LUMEFBPICModel:
    """The model of the module docstring, its run `simulation_length` long."""
    hyparams = SimulationHyperparameters(
        zmin=ZMIN,
        zmax=ZMAX,
        rmax=RMAX,
        nz=NZ,
        nr=NR,
        nm=1,
        use_mpi=False,
        number_dumps=NUMBER_DUMPS,
        simulation_length=simulation_length,
        random_seed=0,
    )
    beam = GaussianParticleBeam(
        sig_r=BEAM_SIGMA_R_M,
        sig_z=BEAM_SIGMA_Z_M,
        n_emit=BEAM_EMITTANCE_M,
        gamma0=BEAM_GAMMA,
        sig_gamma=BEAM_GAMMA_SPREAD,
        n_physical_particles=BEAM_ELECTRONS,
        n_macroparticles=BEAM_MACROPARTICLES,
        zf=BEAM_FOCUS_M,
        particle_name="beam",
    )
    corrector = UniformExternalField(
        field_functions={"By": CORRECTOR_BY_T}, species="beam"
    )
    quadrupole = ExpressionExternalField(
        field_functions={
            "Bx": MathExpression(
                expression="gradient * y",
                variables=("y",),
                parameters={"gradient": QUADRUPOLE_GRADIENT_T_PER_M},
            ),
            "By": MathExpression(
                expression="gradient * x",
                variables=("x",),
                parameters={"gradient": QUADRUPOLE_GRADIENT_T_PER_M},
            ),
        },
        species="beam",
    )
    simulator = FBPICSimulator(
        hyparams,
        beams=[beam],
        external_fields=[corrector, quadrupole],
        target_species="beam",
        working_directory=working_directory,
    )
    actions = [
        ExternalFieldAction(name="corrector_by", field_index=0, field_type="By"),
        *make_beam_actions(simulator),
        *make_stat_actions(),
        FinalParticlesAction(name="final_particles", read_only=True),
    ]
    return LUMEFBPICModel(simulator, actions)


def deflection_rad(model: LUMEFBPICModel) -> float:
    """The beam's horizontal deflection after the run: its mean `px` over its mean `pz`."""
    p = model.simulator.final_particles
    return float(numpy.mean(p.px) / numpy.mean(p.pz))


def predicted_deflection_rad(model: LUMEFBPICModel) -> float:
    """The corrector's deflection of the beam's centroid, `c By L / (p c)` with `By` in T, `L` the
    distance the beam travels and `p c` in eV: the force on an electron moving in +z through
    `By > 0` is in +x."""
    s = model.simulator
    h = s.hyparams
    # POSIT: the run lasts until the window has moved `simulation_length` plus its own length
    # (`right_buffer`'s default), and the beam moves with the window
    travel = h.simulation_length + (h.zmax - h.zmin)
    by = s.external_fields[0].field_functions["By"]
    pc = math.sqrt(s.beams[0].gamma0 ** 2 - 1.0) * m_e * c**2 / e
    return c * by * travel / pc


if __name__ == "__main__":
    model = build_model()
    model.set({})  # runs the simulation (an empty set changes nothing first)
    print(model.get(["charge_pc", "energy_mean_mev", "energy_std_mev"]))
    print(
        f"deflection {deflection_rad(model) * 1e3:.3f} mrad, "
        + f"the corrector's prediction {predicted_deflection_rad(model) * 1e3:.3f} mrad"
    )
