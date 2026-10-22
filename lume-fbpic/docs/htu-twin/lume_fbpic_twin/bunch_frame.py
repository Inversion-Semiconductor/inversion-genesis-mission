"""The bunch frame the twin takes its source bunch in: every particle on one plane `z`, and `t` its
arrival time there (on the clock of the snapshot it came from).

fbpic writes lab-frame snapshots (every particle at one `t`, spread in `z`); Cheetah reads only
`t` of a `ParticleGroup` (its `tau = c t`), so a bunch has to be moved to one plane before it goes
in. The move is the ballistic one of `ParticleGroup.drift_to_z()`: each particle travels in a
straight line at its own velocity, without fields, space charge or scattering. A particle ahead
of the bunch's centre arrives first, with a negative `t`, as in Cheetah.
"""

from __future__ import annotations

from beamphysics import ParticleGroup
from lume_fbpic import descriptor
from lume_fbpic.utils import particle_group


def bunch_frame_particles(
    particles: ParticleGroup,
    *,
    central_fraction: float | None = descriptor.CENTRAL_FRACTION,
    uz_min: float | None = descriptor.UZ_MIN,
) -> ParticleGroup:
    """Select the bunch from a lab-frame snapshot and put it in the bunch frame, on the plane of
    the selection's charge-weighted mean `z` (`to_bunch_frame()`).

    The selection is the moment descriptor's (`lume_fbpic.descriptor.select()`): keep
    `uz = pz / (m_e c) >= uz_min`, then the central `central_fraction` of the charge-weighted
    distance from the mean on each of the six phase-space axes (`None` skips a step). Momenta and
    charge weights are the selection's, and `status` is 1.

    Raises:
        ValueError: If fewer than two particles remain after the selection.
    """
    return to_bunch_frame(
        particle_group(
            *descriptor.select(
                particles, uz_min=uz_min, central_fraction=central_fraction
            )
        )
    )


def to_bunch_frame(particles: ParticleGroup, z: float | None = None) -> ParticleGroup:
    """A copy of `particles` drifted ballistically to the plane `z` [m] (the charge-weighted mean
    `z` if `None`).

    Each particle moves by `(z - z_i) / (beta_z c)` in time, so `x` and `y` change by
    `(z - z_i) px / pz`. A bunch already in the bunch frame is drifted by `z` less its plane.
    Momenta, charges and `status` are unchanged.

    Raises:
        ValueError: If a particle does not move forward (`pz <= 0`): it has no arrival time.
    """
    if (particles.pz <= 0).any():
        raise ValueError(
            "a particle with pz <= 0 does not move forward and has no arrival time"
        )
    result = particles.copy()
    result.drift_to_z(z)
    return result
