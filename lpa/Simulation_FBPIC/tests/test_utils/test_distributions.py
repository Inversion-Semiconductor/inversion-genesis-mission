import numpy as np
import pytest

from inversion_fbpic.utils.distributions import (
    mean_kinetic_energy_mev,
    weighted_quantile,
)


@pytest.mark.parametrize(
    "momentum",
    [
        (np.sqrt(3.0), 0.0, 0.0),
        (0.0, np.sqrt(3.0), 0.0),
        (0.0, 0.0, np.sqrt(3.0)),
        (1.0, 1.0, 1.0),
    ],
)
def test_mean_kinetic_energy_uses_normalized_momenta_not_positions(momentum) -> None:
    ux, uy, uz = momentum
    particles = np.array([[1e6, ux, -2e6, uy, 3e6, uz]])
    result = mean_kinetic_energy_mev(particles, np.array([1.0]))
    assert isinstance(result, float)
    assert result == pytest.approx(0.51099895)


def test_mean_kinetic_energy_is_charge_weighted() -> None:
    particles = np.zeros((2, 6))
    particles[1, 5] = np.sqrt(3.0)
    assert mean_kinetic_energy_mev(particles, np.array([1.0, 3.0])) == pytest.approx(
        0.51099895 * 3.0 / 4.0
    )


def test_mean_kinetic_energy_of_stationary_particles_is_zero() -> None:
    assert mean_kinetic_energy_mev(np.zeros((2, 6)), np.array([1.0, 2.0])) == 0.0


def test_weighted_quantile_uses_macro_charge() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0, 97.0])

    result = weighted_quantile(values, weights, np.array([0.25, 0.5, 0.75]))

    assert np.allclose(result, [2.22680412, 2.48453608, 2.74226804])


def test_weighted_quantile_supports_coordinatewise_particle_data() -> None:
    values = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    weights = np.array([1.0, 1.0, 1.0, 97.0])

    result = weighted_quantile(values, weights, 0.5, axis=0)

    assert np.allclose(result, [2.48453608, 0.0])
