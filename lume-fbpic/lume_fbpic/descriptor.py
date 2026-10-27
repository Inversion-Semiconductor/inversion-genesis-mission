"""The moment descriptor of a bunch, as `build_dataset.py` computes it for the dataset.

The features are `inversion_fbpic.utils.distributions.compute_moment_descriptor()`'s, in its spline
mode, of the particles a selection keeps: `uz >= uz_min`, then the central `central_fraction` on
each of the six phase-space axes (`select()`). `features()` names them, with their units.
"""

from __future__ import annotations

import numpy

from inversion_fbpic.utils.distributions import (
    COORD_NAMES,
    LONGITUDINAL_PROFILE_BINS,
    SPLINE,
    compute_moment_descriptor,
    crop_central_particles,
    select_by_uz,
)
from lume_fbpic import utils

# POSIT: the defaults of build_dataset.py's --central-fraction and --uz-min, which the dataset
# used. The uz cut is tuned to the ionization-injection beams.
CENTRAL_FRACTION = 0.95
UZ_MIN = 30.0

# The unit of a covariance, by the number of positions in its pair (momenta are normalized).
_COVARIANCE_UNITS = ("1", "m", "m^2")


def compute(
    particles,
    *,
    uz_min: float | None = UZ_MIN,
    central_fraction: float | None = CENTRAL_FRACTION,
    longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS,
) -> dict[str, float]:
    """The descriptor of the particles of the `ParticleGroup` `particles` that `select()` keeps.

    Raises:
        ValueError: If fewer than two particles are left.
    """
    return compute_moment_descriptor(
        *select(particles, uz_min=uz_min, central_fraction=central_fraction),
        longitudinal_mode=SPLINE,
        longitudinal_bins=longitudinal_bins,
    )


def covariance_feature(row: str, column: str) -> str:
    """The name of the covariance of the phase-space coordinates `row` and `column`
    (`COORD_NAMES`, `row` first in their order)."""
    return f"cov_{row}_{column}"


def features(longitudinal_bins: int = LONGITUDINAL_PROFILE_BINS) -> dict[str, str]:
    """The unit of each feature, by name, in `compute_moment_descriptor()`'s order.

    The phase space is `x`, `ux`, `y`, `uy`, `z`, `uz`: positions in metres and momenta normalized
    to `m c`. A momentum centroid or a longitudinal `uz` feature is dimensionless (`"1"`), a
    covariance is `m` to the power of the number of positions in the pair (`cov_x_x` in m^2,
    `cov_x_ux` in m, `cov_ux_ux` dimensionless), and the charge is in pC.
    """
    result = {f"mean_{name}": "1" for name in COORD_NAMES if name.startswith("u")}
    for index, row in enumerate(COORD_NAMES):
        for column in COORD_NAMES[index:]:
            result[covariance_feature(row, column)] = _COVARIANCE_UNITS[
                _is_position(row) + _is_position(column)
            ]
    for index in range(longitudinal_bins):
        result |= dict.fromkeys(slice_features(index), "1")
    result["total_beam_charge_pc"] = "pC"
    return result


def select(
    particles,
    *,
    uz_min: float | None = UZ_MIN,
    central_fraction: float | None = CENTRAL_FRACTION,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The particles of the `ParticleGroup` `particles` with `uz >= uz_min`, then the central
    `central_fraction` of those on each axis (`None` skips a step), as
    `utils.normalized_phase_space()` returns them.

    Raises:
        ValueError: If fewer than two particles are left, or `central_fraction` is not in (0, 1].
    """
    phase_space, electrons = utils.normalized_phase_space(particles)
    if uz_min is not None:
        phase_space, electrons = select_by_uz(phase_space, electrons, uz_min=uz_min)
    if central_fraction is not None:
        phase_space, electrons = crop_central_particles(
            phase_space, electrons, central_fraction=central_fraction
        )
    return phase_space, electrons


def slice_features(index: int) -> tuple[str, str]:
    """The names of the mean and the rms `uz` of longitudinal bin `index`."""
    return f"longitudinal_mean_uz_{index:02d}", f"longitudinal_rms_uz_{index:02d}"


def _is_position(name: str) -> bool:
    return not name.startswith("u")
