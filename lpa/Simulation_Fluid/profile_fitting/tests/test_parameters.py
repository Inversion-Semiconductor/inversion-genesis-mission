from __future__ import annotations

import numpy as np
import pytest

from fludat_fit.parameters import ParameterSpace, ParameterSpec


def test_spec_validation():
    with pytest.raises(ValueError, match="increasing"):
        ParameterSpec("a", 1.0, 1.0)
    with pytest.raises(ValueError, match="log-scale"):
        ParameterSpec("a", 0.0, 1.0, log_scale=True)
    with pytest.raises(ValueError, match="duplicate"):
        ParameterSpace((ParameterSpec("a", 0, 1), ParameterSpec("a", 0, 2)))


def test_unit_round_trip_linear_and_log():
    space = ParameterSpace(
        (
            ParameterSpec("position", -2.0, 3.0),
            ParameterSpec("width", 1.0e-5, 1.0e-1, log_scale=True),
        )
    )
    theta = np.array([0.5, 1.0e-3])
    u = space.to_unit(theta)
    assert u[0] == pytest.approx(0.5)
    assert u[1] == pytest.approx(0.5)  # geometric midpoint of the log range
    np.testing.assert_allclose(space.from_unit(u), theta)
    np.testing.assert_allclose(space.center(), [0.5, 1.0e-3])
    assert space.names == ("position", "width")
    assert space["width"].log_scale
    assert space.index("width") == 1


def test_clip_contains_and_dicts():
    space = ParameterSpace((ParameterSpec("a", 0.0, 1.0), ParameterSpec("b", 2.0, 4.0)))
    assert space.contains([0.5, 3.0])
    assert not space.contains([1.5, 3.0])
    np.testing.assert_allclose(space.clip([1.5, 1.0]), [1.0, 2.0])
    assert space.as_dict([0.25, 2.5]) == {"a": 0.25, "b": 2.5}
    np.testing.assert_allclose(space.from_dict({"b": 2.5, "a": 0.25}), [0.25, 2.5])
    with pytest.raises(KeyError):
        space.from_dict({"a": 0.0})
    with pytest.raises(ValueError, match="length 2"):
        space.to_unit([0.0])


def test_samples_are_inside_bounds_and_reproducible():
    space = ParameterSpace(
        (ParameterSpec("a", -1.0, 1.0), ParameterSpec("b", 1.0, 100.0, log_scale=True))
    )
    samples = space.sample(16, seed=3)
    assert samples.shape == (16, 2)
    assert np.all(samples >= space.lower) and np.all(samples <= space.upper)
    np.testing.assert_array_equal(samples, space.sample(16, seed=3))
    assert space.sample_unit(0).shape == (0, 2)
