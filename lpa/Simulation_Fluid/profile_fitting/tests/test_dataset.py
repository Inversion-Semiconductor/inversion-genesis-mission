from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from fludat_fit.dataset import MAX_ANGLE_DEG, NozzleDataset, trilinear_density
from fludat_fit.lineout import LineoutConditions
from fludat_proc.interpolate_density import build_density_interpolation


def fbpic_reference(field, path, pressure):
    """What InterpolateFromH5Profile computes: a full trilinear interpolator on the path."""
    interpolator = RegularGridInterpolator(
        (field.z, field.x, field.pressure), field.cube.density, bounds_error=True
    )
    t = path.t_values()
    query = np.column_stack(
        (
            np.clip(path.z_m(t), field.z[0], field.z[-1]),
            np.clip(path.x_mm(t), field.x[0], field.x[-1]),
            np.full(t.size, pressure),
        )
    )
    return t, interpolator(query)


def test_trilinear_matches_fludat_proc_on_the_z_grid(small_cube_path):
    field = build_density_interpolation(small_cube_path)
    for x_mm, pressure in ((0.3, 7.5), (1.0, 5.0), (2.0, 10.0), (0.0, 6.1)):
        np.testing.assert_allclose(
            trilinear_density(field, field.z, np.full(field.z.size, x_mm), pressure),
            field.interpolate_along_z(field.z, x_mm, pressure),
            rtol=1e-12,
            atol=0.0,
        )


def test_dataset_basics(small_cube_path):
    dataset = NozzleDataset.load(small_cube_path)
    assert dataset.name == "test" and dataset.path == small_cube_path
    assert dataset.x_extent == (0.0, 2.0) and dataset.pressure_extent == (5.0, 10.0)
    assert dataset.angle_extent == (-MAX_ANGLE_DEG, MAX_ANGLE_DEG)
    assert dataset.default_conditions() == LineoutConditions(0.0, 5.0, 0.0)
    assert dataset.contains(LineoutConditions(1.0, 10.0, 30.0))
    assert not dataset.contains(LineoutConditions(3.0, 10.0))
    assert not dataset.contains(LineoutConditions(1.0, 10.0, 90.0))


def test_axial_lineout_uses_the_z_grid(small_cube_path):
    dataset = NozzleDataset.load(small_cube_path)
    lineout = dataset.lineout(LineoutConditions(1.0, 10.0))
    np.testing.assert_array_equal(lineout.z, dataset.z)
    np.testing.assert_allclose(
        lineout.density, dataset.field.interpolate_along_z(dataset.z, 1.0, 10.0)
    )
    assert lineout.source == "test" and lineout.density_units == "cm^-3"
    path = dataset.path_for(LineoutConditions(1.0, 10.0))
    assert path.is_axial and path.n_points == dataset.z.size
    assert (path.t_min, path.t_max) == dataset.field.z_extent
    assert path.lineout_axis() == {
        "z_m": {"origin": 0.0, "coefficient": 1.0},
        "x_mm": {"origin": 1.0, "coefficient": 0.0},
    }


@pytest.mark.parametrize("angle", [0.0, 15.0, -30.0, 60.0])
@pytest.mark.parametrize("x_mm", [0.0, 1.0, 2.0])
def test_t_at_physical_z_is_always_zero_since_z_origin_is_always_zero(
    small_cube_path, x_mm, angle
):
    """Every lineout line passes through (z_m=0, x_mm=x_mm) by construction, so
    the path parameter at the physical z_m=0 plane is always t=0, regardless
    of x or angle. This is the anchor fludat_fit.generate_profile_configs pins
    every profile family's centroid to, since the gas target is symmetric
    about z_m=0 but an oblique or off-axis lineout samples it asymmetrically
    (its own density-weighted centroid need not be, and generally is not, 0)."""
    dataset = NozzleDataset.load(small_cube_path)
    path = dataset.path_for(LineoutConditions(x_mm, 10.0, angle))

    assert path.z_m(path.t_at_physical_z(0.0)) == pytest.approx(0.0, abs=1.0e-12)
    assert path.t_at_physical_z(0.0) == pytest.approx(0.0, abs=1.0e-12)
    # A non-zero target still resolves correctly, from the same (origin=0) line.
    other = 3.0e-4
    t_other = path.t_at_physical_z(other)
    assert path.z_m(t_other) == pytest.approx(other)


@pytest.mark.parametrize("angle", [15.0, -30.0, 60.0])
def test_oblique_lineout_matches_the_fbpic_convention(small_cube_path, angle):
    dataset = NozzleDataset.load(small_cube_path)
    conditions = LineoutConditions(1.0, 7.5, angle)
    path = dataset.path_for(conditions)
    field = dataset.field

    # Coefficients: unit direction in (z [m], x [mm]) at `angle` from the z axis.
    assert path.z_coefficient == pytest.approx(np.cos(np.deg2rad(angle)))
    assert path.x_coefficient == pytest.approx(1000.0 * np.sin(np.deg2rad(angle)))
    assert path.n_points == field.z.size + field.x.size
    # The path stays inside the cube in both z and x ...
    t = path.t_values()
    z_m, x_mm = path.z_m(t), path.x_mm(t)
    assert field.z[0] - 1e-12 <= z_m.min() and z_m.max() <= field.z[-1] + 1e-12
    assert field.x[0] - 1e-9 <= x_mm.min() and x_mm.max() <= field.x[-1] + 1e-9
    # ... and touches a bound at each end (clipped as tightly as possible).
    for end in (0, -1):
        assert (
            np.isclose(z_m[end], field.z_extent).any()
            or np.isclose(x_mm[end], field.x_extent).any()
        )

    lineout = dataset.lineout(conditions)
    t_ref, density_ref = fbpic_reference(field, path, 7.5)
    np.testing.assert_allclose(lineout.z, t_ref)
    np.testing.assert_allclose(lineout.density, density_ref, rtol=1e-10, atol=1e-6)


def test_oblique_lineout_is_clipped_by_the_x_bounds(small_cube_path):
    dataset = NozzleDataset.load(small_cube_path)
    # At 45 deg from x = 1 mm the line leaves x in [0, 2] after 1 mm / sin(45 deg) of
    # path, long before it leaves z in [-4, 4] mm.
    path = dataset.path_for(LineoutConditions(1.0, 5.0, 45.0))
    half = 1.0e-3 / np.sin(np.deg2rad(45.0))
    assert path.t_min == pytest.approx(-half) and path.t_max == pytest.approx(half)


def test_custom_t_values_are_zero_outside_the_path(small_cube_path):
    dataset = NozzleDataset.load(small_cube_path)
    t = np.linspace(-6e-3, 6e-3, 25)
    lineout = dataset.lineout(LineoutConditions(1.0, 10.0, 10.0), t_values=t)
    np.testing.assert_array_equal(lineout.z, t)
    assert lineout.density[0] == 0.0 and lineout.density[-1] == 0.0
    assert lineout.density[12] > 0.0


def test_non_linear_methods_fall_back_to_fludat_proc(small_cube_path):
    dataset = NozzleDataset.load(small_cube_path)
    nearest = NozzleDataset(replace(dataset.field, method="nearest"))
    conditions = LineoutConditions(0.7, 8.0, 20.0)
    path = nearest.path_for(conditions)
    t = path.t_values()[::20]
    lineout = nearest.lineout(conditions, t_values=t)
    expected = [
        nearest.field.interpolate(float(z), float(x), 8.0)
        for z, x in zip(path.z_m(t), path.x_mm(t), strict=True)
    ]
    np.testing.assert_allclose(lineout.density, expected)
    assert not np.allclose(
        lineout.density, dataset.lineout(conditions, t_values=t).density
    )


def test_h5_profile_kwargs(small_cube_path):
    dataset = NozzleDataset.load(small_cube_path)
    kwargs = dataset.h5_profile_kwargs(LineoutConditions(1.0, 7.5, 30.0))
    assert kwargs["filename"] == str(small_cube_path)
    assert kwargs["density_name"] == "density"
    assert kwargs["interpolation_points"] == {"pressure_bar": 7.5}
    axis = kwargs["lineout_axis"]
    assert axis["z_m"]["coefficient"] == pytest.approx(np.cos(np.pi / 6))
    assert axis["x_mm"] == {"origin": 1.0, "coefficient": pytest.approx(500.0)}


def test_validation(small_cube_path):
    dataset = NozzleDataset.load(small_cube_path)
    with pytest.raises(ValueError, match="angle_deg"):
        dataset.path_for(LineoutConditions(1.0, 5.0, 89.5))
    with pytest.raises(ValueError, match="x_mm"):
        dataset.path_for(LineoutConditions(3.0, 5.0))
    with pytest.raises(ValueError, match="pressure_bar"):
        dataset.lineout(LineoutConditions(1.0, 50.0))
    with pytest.raises(ValueError, match="finite"):
        LineoutConditions(1.0, float("nan"))
