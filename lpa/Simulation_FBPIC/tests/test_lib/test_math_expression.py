"""Tests for safe, declarative serializable numerical functions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.density_profiles import ExpressionDensityProfile
from inversion_fbpic.lib.serializable_config import SerializableConfig
from inversion_fbpic.lib.math_expression import MathExpression


def test_multivariate_function_evaluates_scalars_and_arrays() -> None:
    function = MathExpression(
        expression="amplitude * np.exp(-((z - center) / width) ** 2) * np.where(r <= radius, 1, 0)",
        variables=("z", "r"),
        parameters={"amplitude": 2.0, "center": 1.0, "width": 2.0, "radius": 3.0},
    )

    assert function(1.0, 0.0) == 2.0
    np.testing.assert_allclose(
        function(np.array([1.0, 3.0]), np.array([0.0, 4.0])),
        np.array([2.0, 0.0]),
    )


def test_compilation_namespace_includes_expression_parameters() -> None:
    expression = MathExpression(
        expression="scale * np.sin(z)",
        variables=("z",),
        parameters={"scale": 2.0},
    )

    namespace = expression.compilation_namespace()

    assert namespace["np"] is np
    assert namespace["scale"] == 2.0
    assert namespace["pi"] == np.pi


def test_function_round_trips_without_pickling() -> None:
    function = MathExpression(
        expression="scale * np.sin(z)", variables=("z",), parameters={"scale": 2.0}
    )

    payload = function.to_dict()
    assert payload["config_type"] == "math_expression"
    assert json.dumps(payload)
    reloaded = SerializableConfig.from_dict(payload)

    assert isinstance(reloaded, MathExpression)
    assert reloaded.expression == function.expression
    assert reloaded(0.5) == pytest.approx(function(0.5))

    yaml_reloaded = SerializableConfig.from_yaml(function.to_yaml(comments=False))
    assert isinstance(yaml_reloaded, MathExpression)
    assert yaml_reloaded(0.5) == pytest.approx(function(0.5))


def test_long_expression_round_trips_with_yaml_comments() -> None:
    expression = (
        "np.where(z < ramp_start, 0.0, "
        "np.where(z < ramp_start + ramp_length, "
        "(z - ramp_start) / ramp_length, 1.0))"
    )
    function = MathExpression(
        expression=expression,
        variables=("z",),
        parameters={"ramp_start": 1.0e-5, "ramp_length": 4.0e-5},
    )

    reloaded = SerializableConfig.from_yaml(function.to_yaml())

    assert isinstance(reloaded, MathExpression)
    assert reloaded.expression == expression


def test_function_exposes_allowlisted_scipy_numpy_and_math_symbols() -> None:
    function = MathExpression(
        expression="scipy.special.jv(0, z) + numpy.linalg.norm([z, 0]) + math.log(1)",
        variables=("z",),
    )

    assert function(1.0) == pytest.approx(1.7651976866)


def test_function_exposes_constants_without_module_prefixes() -> None:
    function = MathExpression(expression="e + pi", variables=("z",))

    assert function(0.0) == pytest.approx(np.e + np.pi)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo unsafe')",
        "z.__class__",
        "[z for z in (1, 2)]",
        "np.ctypeslib.load_library(z, z)",
        "np.memmap(z, dtype=np.uint8)",
        "np.load(z, allow_pickle=True)",
        "np.ndarray.tofile(np.array([z]), z)",
        "scipy.io.loadmat(z)",
        "np.random.seed(z)",
        "np.e",
        "numpy.pi",
        "math.inf",
    ],
)
def test_function_rejects_executable_python_syntax(expression: str) -> None:
    with pytest.raises(ValueError):
        MathExpression(expression=expression, variables=("z",))


def test_function_rejects_out_keyword_to_preserve_input_arrays() -> None:
    with pytest.raises(ValueError, match="cannot use 'out'.*modify input arrays"):
        MathExpression(expression="np.multiply(z, 0, out=z)", variables=("z",))


def test_general_density_profile_round_trips_and_builds_2d_function() -> None:
    profile = ExpressionDensityProfile(
        nominal_density=1.0e24,
        p_nz=1,
        p_nr=1,
        p_nt=1,
        function=MathExpression(
            expression="np.where((z >= 0) * (r <= radius), 1, 0)",
            variables=("z", "r"),
            parameters={"radius": 2.0e-6},
        ),
        z_min=0.0,
        z_max=4.0e-6,
        r_max=2.0e-6,
    )

    reloaded = _DensityProfile.from_json(profile.to_json())

    assert isinstance(reloaded, ExpressionDensityProfile)
    assert reloaded.get_z_extent() == (0.0, 4.0e-6)
    assert reloaded.get_r_extent() == 2.0e-6
    np.testing.assert_array_equal(
        reloaded.build_density_function()(
            np.array([-1.0e-6, 1.0e-6]), np.array([0.0, 3.0e-6])
        ),
        np.array([0, 0]),
    )


@pytest.mark.parametrize(
    ("variables", "expression", "z", "r", "expected"),
    [
        (
            ("z",),
            "2 * z",
            np.array([1.0, 2.0]),
            np.array([4.0, 5.0]),
            np.array([2.0, 4.0]),
        ),
        (
            ("r",),
            "r ** 2",
            np.array([1.0, 2.0]),
            np.array([4.0, 5.0]),
            np.array([16.0, 25.0]),
        ),
    ],
)
def test_general_density_profile_accepts_one_dimensional_functions(
    variables: tuple[str, ...],
    expression: str,
    z: np.ndarray,
    r: np.ndarray,
    expected: np.ndarray,
) -> None:
    profile = ExpressionDensityProfile(
        nominal_density=1.0e24,
        p_nz=1,
        p_nr=1,
        p_nt=1,
        function=MathExpression(expression=expression, variables=variables),
        z_min=0.0,
        z_max=1.0,
    )

    np.testing.assert_array_equal(profile.build_density_function()(z, r), expected)


def test_general_density_profile_rejects_unrelated_function_variables() -> None:
    with pytest.raises(ValueError, match="drawn"):
        ExpressionDensityProfile(
            nominal_density=1.0e24,
            p_nz=1,
            p_nr=1,
            p_nt=1,
            function=MathExpression(expression="x", variables=("x",)),
            z_min=0.0,
            z_max=1.0,
        )
