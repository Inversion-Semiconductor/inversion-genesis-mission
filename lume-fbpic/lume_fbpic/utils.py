"""Particle conversions: fbpic's openPMD output to an openPMD-beamphysics `ParticleGroup`, and the
statistics of a run's bunch. Also `evolve_laser()`, used by both `simulator` and `actions`.

fbpic and `inversion_fbpic` hold a particle's momentum normalized, `u = gamma * beta`, and its
weight as a number of electrons; a `ParticleGroup` holds the momentum in eV/c and the weight in
coulombs. `particle_group()` and `normalized_phase_space()` convert between the two.
"""

from __future__ import annotations

from pathlib import Path
import typing

import attrs
import numpy
from scipy.constants import e

from beamphysics import ParticleGroup
from beamphysics.units import mec2
from openpmd_viewer.addons import LpaDiagnostics

from inversion_fbpic.utils.distributions import COORD_NAMES

# The electron rest energy [eV], `p [eV/c] = u * ELECTRON_MC2_EV`: beamphysics's own, the mass its
# energies use.
ELECTRON_MC2_EV = mec2


# A laser's amplitude field -> the other one, derived from it (see `evolve_laser()`).
_AMPLITUDE_SIBLING = {"a0": "energy", "energy": "a0"}


class Statistic(typing.NamedTuple):
    """A statistic of a run's bunch: its unit, what it is, and how to compute it from the
    `ParticleGroup`."""

    unit: str
    description: str
    function: typing.Callable[[ParticleGroup], float]


# The statistics of a run's bunch, by name.
STATISTICS = {
    "charge_pc": Statistic(
        "pC", "Charge, all electrons", lambda particles: particles.charge * 1e12
    ),
    "energy_mean_mev": Statistic(
        "MeV",
        "Mean kinetic energy",
        lambda particles: particles["mean_kinetic_energy"] / 1e6,
    ),
    "energy_std_mev": Statistic(
        "MeV",
        "Kinetic energy rms",
        lambda particles: particles["sigma_energy"] / 1e6,
    ),
}


def evolve_laser(laser: typing.Any, **changes: typing.Any) -> typing.Any:
    """`attrs.evolve(laser, **changes)` for a `_LaserPulse`.

    Every `_LaserPulse` holds *both* `energy` and `a0` -- whichever wasn't given at construction
    is derived and stored too -- but its constructor takes only one of them. So the one that
    should not survive the change is dropped first: the other amplitude when `changes` sets
    `energy` or `a0`, otherwise the derived one. A laser built from `a0` reports the energy it
    derived as `out_energy`, and no other laser sets it, so that tells which one was given.
    """
    changed = [name for name in _AMPLITUDE_SIBLING if name in changes]
    given = "a0" if laser.out_energy is not None else "energy"
    derived = _AMPLITUDE_SIBLING[changed[0] if changed else given]
    return attrs.evolve(laser, **{derived: None, **changes})


def load_final_particles(
    diags_path: Path | str, species: str
) -> tuple[ParticleGroup, dict[str, float]]:
    """The electrons of `species` in the last dump of the openPMD diagnostics in `diags_path`.

    Returns the particles as a `ParticleGroup` (all at `t = 0`) and their `STATISTICS`: the
    charge, and the charge-weighted mean and spread of the kinetic energy (all 0 when there are
    no particles).
    """
    ts = LpaDiagnostics(str(diags_path), check_all_files=True)
    *coordinates, electrons = ts.get_particle(
        [*COORD_NAMES, "w"],
        iteration=ts.iterations[-1],
        species=species,
        plot=False,
    )
    particles = particle_group(numpy.stack(coordinates, axis=-1), electrons)
    return particles, {
        name: float(statistic.function(particles)) if len(electrons) else 0.0
        for name, statistic in STATISTICS.items()
    }


def normalized_phase_space(
    particles: ParticleGroup,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The phase space of `particles`, one row of `COORD_NAMES` (`x, ux, y, uy, z, uz`) per
    particle, positions in metres and momenta normalized to `m c`, and their weights in
    electrons; the inverse of `particle_group()`."""
    return (
        numpy.stack(
            [
                (
                    numpy.asarray(particles[f"p{name[1]}"], dtype=numpy.float64)
                    / ELECTRON_MC2_EV
                    if name.startswith("u")
                    else numpy.asarray(particles[name], dtype=numpy.float64)
                )
                for name in COORD_NAMES
            ],
            axis=-1,
        ),
        numpy.asarray(particles.weight, dtype=numpy.float64) / e,
    )


def particle_group(
    phase_space: numpy.ndarray, electrons: numpy.ndarray, t: typing.Any = 0.0
) -> ParticleGroup:
    """Electrons with the phase space `phase_space`, a row of `COORD_NAMES` per particle as
    `normalized_phase_space()` returns it, `electrons` physical electrons per macroparticle and
    time `t` [s] (one for all, or one each), all alive."""
    columns = dict(zip(COORD_NAMES, numpy.asarray(phase_space, dtype=numpy.float64).T))
    n = len(electrons)
    return ParticleGroup(
        data={
            **{name: columns[name] for name in ("x", "y", "z")},
            **{f"p{name}": columns[f"u{name}"] * ELECTRON_MC2_EV for name in "xyz"},
            "t": numpy.broadcast_to(numpy.asarray(t, dtype=numpy.float64), n).copy(),
            "status": numpy.ones(n, dtype=int),
            "weight": numpy.asarray(electrons, dtype=numpy.float64) * e,
            "species": "electron",
        }
    )
