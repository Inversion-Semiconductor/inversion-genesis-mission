"""The moment descriptor of a `ParticleGroup`, computed directly with `inversion_fbpic`'s library,
as `build_dataset.py` computes it: a reference the tests compare lume_fbpic's results with.
"""

from __future__ import annotations

import numpy
from scipy.constants import e

from inversion_fbpic.utils import distributions
from lume_fbpic.utils import ELECTRON_MC2_EV


def phase_space(particles) -> numpy.ndarray:
    """`(x, ux, y, uy, z, uz)` of each particle, momenta normalized to `m c`."""
    return numpy.stack(
        [
            particles.x,
            particles.px / ELECTRON_MC2_EV,
            particles.y,
            particles.py / ELECTRON_MC2_EV,
            particles.z,
            particles.pz / ELECTRON_MC2_EV,
        ],
        axis=-1,
    )


def reference_descriptor(
    particles, *, uz_min: float | None, central_fraction: float | None
) -> dict[str, float]:
    """The descriptor of `particles` after the selection (`None` skips a step)."""
    selected, weights = phase_space(particles), numpy.asarray(particles.weight) / e
    if uz_min is not None:
        selected, weights = distributions.select_by_uz(selected, weights, uz_min=uz_min)
    if central_fraction is not None:
        selected, weights = distributions.crop_central_particles(
            selected, weights, central_fraction=central_fraction
        )
    return distributions.compute_moment_descriptor(selected, weights)
