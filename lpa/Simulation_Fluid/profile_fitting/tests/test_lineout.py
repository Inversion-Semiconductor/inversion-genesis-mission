from __future__ import annotations

import numpy as np
import pytest

from fludat_fit.lineout import (
    Lineout,
    LineoutConditions,
    extract_lineouts,
    load_lineout,
)
from fludat_proc.interpolate_density import build_density_interpolation

from .conftest import CONDITIONS


def gaussian_lineout(sigma=1.0e-3, center=0.5e-3, n=401):
    z = np.linspace(-5.0e-3, 5.0e-3, n)
    return Lineout(
        z=z,
        density=3.0 * np.exp(-((z - center) ** 2) / (2 * sigma**2)),
        density_units="cm^-3",
        conditions=CONDITIONS,
    )


def test_conditions_vector_and_dict():
    conditions = LineoutConditions(x_mm=1.0, pressure_bar=5.0)
    assert conditions.angle_deg == 0.0
    assert conditions.as_vector().tolist() == [1.0, 5.0, 0.0]
    assert conditions.to_dict() == {"x_mm": 1.0, "pressure_bar": 5.0, "angle_deg": 0.0}
    assert LineoutConditions(1.0, 5.0, 30.0).as_vector().tolist() == [1.0, 5.0, 30.0]
    assert LineoutConditions(1, 5, 30).label() == "x=1 mm, p=5 bar, angle=30 deg"


def test_validation():
    z = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="strictly increasing"):
        Lineout(z=z[::-1], density=z, density_units="u", conditions=CONDITIONS)
    with pytest.raises(ValueError, match="shape"):
        Lineout(z=z, density=z[:3], density_units="u", conditions=CONDITIONS)
    with pytest.raises(ValueError, match="weights"):
        Lineout(z=z, density=z, density_units="u", conditions=CONDITIONS, weights=-z)


def test_summary_of_a_gaussian():
    lineout = gaussian_lineout()
    summary = lineout.summary()
    assert summary.peak == pytest.approx(3.0, rel=1e-3)
    assert summary.z_peak == pytest.approx(0.5e-3, abs=3.0e-5)
    assert summary.centroid == pytest.approx(0.5e-3, abs=1.0e-5)
    assert summary.fwhm == pytest.approx(2.3548e-3, rel=0.02)
    lower, upper = summary.support
    # density >= 1e-3 * peak within about 3.7 sigma of the centre
    assert lower == pytest.approx(0.5e-3 - 3.72e-3, abs=5.0e-5)
    assert upper == pytest.approx(0.5e-3 + 3.72e-3, abs=5.0e-5)
    assert summary.z_scale == pytest.approx(upper - lower)


def test_trim_and_crop():
    lineout = gaussian_lineout(sigma=0.3e-3, center=0.0)
    trimmed = lineout.trimmed(cutoff_ratio=1e-3, padding_fraction=0.0)
    lower, upper = lineout.support(1e-3)
    assert trimmed.z_extent == (lower, upper)
    assert trimmed.n_points < lineout.n_points
    padded = lineout.trimmed(cutoff_ratio=1e-3, padding_fraction=0.5)
    assert padded.z_extent[0] < lower and padded.z_extent[1] > upper

    cropped = lineout.cropped((-1.0e-3, 1.0e-3))
    assert cropped.z.min() >= -1.0e-3 and cropped.z.max() <= 1.0e-3
    with pytest.raises(ValueError):
        lineout.cropped((10.0, 11.0))


def test_extract_from_cube(small_cube_path):
    lineout = load_lineout(small_cube_path, x_mm=1.0, pressure_bar=10.0)
    assert lineout.source == "test"
    assert lineout.density_units == "cm^-3"
    assert lineout.conditions == LineoutConditions(1.0, 10.0)
    assert lineout.peak == pytest.approx(2.0e19)
    assert lineout.z_peak == pytest.approx(0.0, abs=1e-12)

    field = build_density_interpolation(small_cube_path)
    lineouts = extract_lineouts(
        field, [{"x_mm": 0.0, "pressure_bar": 5.0}, LineoutConditions(2.0, 7.5)]
    )
    assert [item.peak for item in lineouts] == pytest.approx([0.5e19, 2.25e19])

    # A wider z grid sees vacuum beyond the cube.
    wide = load_lineout(
        small_cube_path, 1.0, 10.0, z_values=np.linspace(-6e-3, 6e-3, 25)
    )
    assert wide.density[0] == 0.0 and wide.density[-1] == 0.0
    with pytest.raises(ValueError, match="angle_deg=0 only"):
        extract_lineouts(field, [LineoutConditions(1.0, 5.0, 10.0)])
