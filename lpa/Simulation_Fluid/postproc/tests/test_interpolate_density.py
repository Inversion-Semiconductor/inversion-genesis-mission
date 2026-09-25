from __future__ import annotations

import numpy as np
import pytest

from fludat_proc.interpolate_density import (
    DensityInterpolation,
    build_density_callable,
    build_density_interpolation,
)


def expected_density(z, x, pressure):
    return (1 + z) * (1 + x) * pressure


def test_density_at_z_matches_tabulated_and_interpolated_slices(small_cube):
    field = DensityInterpolation(small_cube)

    np.testing.assert_allclose(field.density_at_z(-2.0), small_cube.density[0])
    np.testing.assert_allclose(field.density_at_z(2.0), small_cube.density[-1])
    np.testing.assert_allclose(
        field.density_at_z(0.5),
        expected_density(0.5, small_cube.x[:, None], small_cube.pressure),
    )


def test_point_and_profile_interpolation_are_exact_for_trilinear_data(small_cube):
    field = DensityInterpolation(small_cube)

    assert field.interpolate(0.25, 1.5, 7.5) == pytest.approx(
        expected_density(0.25, 1.5, 7.5)
    )
    z_values = np.linspace(-1.0, 1.0, 7)
    np.testing.assert_allclose(
        field.interpolate_along_z(z_values, 0.5, 6.0),
        expected_density(z_values, 0.5, 6.0),
    )


def test_xz_grid_and_pressure_stack_agree(small_cube):
    field = DensityInterpolation(small_cube)
    x_values = np.array([0.0, 0.5, 2.0])
    z_values = np.array([-1.0, 0.0, 1.5])

    direct = field.interpolate_xz_grid(x_values, z_values, 7.5)
    stack = field.xz_grids_at_pressures(x_values, z_values)

    assert direct.shape == (3, 3)
    assert stack.shape == (2, 3, 3)
    np.testing.assert_allclose(
        field.interpolate_xz_from_pressure_stack(stack, 7.5), direct
    )
    np.testing.assert_allclose(
        direct, expected_density(z_values[None, :], x_values[:, None], 7.5)
    )


def test_out_of_range_queries_raise_with_valid_extents(small_cube):
    field = DensityInterpolation(small_cube)

    with pytest.raises(ValueError, match=r"z=3.0.*valid range \[-2.0, 2.0\] m"):
        field.interpolate(3.0, 0.0, 5.0)
    with pytest.raises(ValueError, match=r"pressure=50.*\[5.0, 10.0\] bar"):
        field.interpolate(0.0, 0.0, 50.0)
    with pytest.raises(ValueError, match="x=-1"):
        field.interpolate_along_z(np.array([0.0]), -1.0, 5.0)


def test_build_from_hdf5_and_fbpic_callable(small_cube_path):
    field = build_density_interpolation(small_cube_path, method="linear")
    assert field.geometry == "test"
    assert field.density_units == "cm^-3"

    density = build_density_callable(
        small_cube_path, backing_pressure=7.5, x_position=1.0
    )
    z = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        density(z, np.zeros_like(z)), expected_density(z, 1.0, 7.5)
    )

    with pytest.raises(ValueError, match="does not match field.method"):
        build_density_callable(small_cube_path, 7.5, 1.0, method="cubic", field=field)
