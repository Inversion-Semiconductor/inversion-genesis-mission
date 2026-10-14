import numpy as np

from inversion_fbpic.utils.distributions import weighted_quantile


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
