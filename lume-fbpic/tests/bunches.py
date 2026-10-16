"""Synthetic electron bunches for the tests."""

from __future__ import annotations

import numpy

from beamphysics import ParticleGroup

from lume_fbpic.utils import particle_group


def lab_snapshot(
    n: int = 4000,
    *,
    seed: int = 0,
    uz_mean: float = 150.0,
    uz_std: float = 40.0,
    electrons: float = 1.0e3,
) -> ParticleGroup:
    """A Gaussian bunch as fbpic's diagnostics give it, all at `t = 0`: `x`, `y`, `z` about
    (0, 0, 3 mm) with rms 3, 1.5 and 4 um, `ux`, `uy`, `uz` (normalized) about (0, 0, `uz_mean`)
    with rms 2, 1 and `uz_std`, and `electrons` electrons per macroparticle."""
    rng = numpy.random.default_rng(seed)
    x, y, z = (
        rng.normal(mean, rms, n)
        for mean, rms in ((0.0, 3.0e-6), (0.0, 1.5e-6), (0.003, 4.0e-6))
    )
    ux, uy, uz = (
        rng.normal(mean, rms, n)
        for mean, rms in ((0.0, 2.0), (0.0, 1.0), (uz_mean, uz_std))
    )
    return particle_group(
        numpy.column_stack([x, ux, y, uy, z, uz]), numpy.full(n, electrons)
    )
