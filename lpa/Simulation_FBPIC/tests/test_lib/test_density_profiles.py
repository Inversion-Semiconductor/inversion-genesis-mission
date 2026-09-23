"""
Tests for density profile physics: build_density_function, extents,
get_plasma_wavelength, ModifiedDensityProfile, and MatchedRadialModifier.

Species/ionization validation lives in the sibling ``test_density.py``.

Run from Simulation_FBPIC::

    pytest tests/test_lib/test_density_profiles.py -v
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import shutil


# ---------------------------------------------------------------------------
# Constructor kwargs (mirrored from conftest for importability)
# ---------------------------------------------------------------------------

_COMMON = {"nominal_density": 1.0e24, "p_nz": 1, "p_nr": 1, "p_nt": 1}

EXAMPLE_DENSITY_KWARGS = {**_COMMON, "length": 5.0e-6, "start_position": 0.0}

ASYMMETRIC_SINE_KWARGS = {
    **_COMMON,
    "peak_z0": 2.5e-3,
    "upramp_length": 2.5e-3,
    "downramp_length": 1.5e-3,
}

SMOOTH_SINE_FLATTOP_KWARGS = {
    **_COMMON,
    "flattop_width": 1.0e-3,
    "upramp_length": 0.5e-3,
    "downramp_length": 0.5e-3,
    "offset_length": 0.0,
}

GAUSSIAN_PLUS_TRIANGLE_KWARGS = {
    **_COMMON,
    "gauss_sigma": 3.0e-6,
    "gauss_z0": 10.0e-6,
    "tri_z0": 15.0e-6,
    "tri_left_width": 5.0e-6,
    "tri_right_width": 5.0e-6,
    "tri_height": 1.0,
}

GENERALIZED_GAUSSIAN_PLUS_TRIANGLE_KWARGS = {
    **_COMMON,
    "gauss_peak": 1.0,
    "gauss_alpha": 3.0e-6,
    "gauss_beta": 2.0,
    "gauss_z0": 10.0e-6,
    "tri_z0": 15.0e-6,
    "tri_left_width": 5.0e-6,
    "tri_right_width": 5.0e-6,
    "tri_height": 1.0,
}

H5_DENSITY_KWARGS = {
    "p_nz": 1,
    "p_nr": 1,
    "p_nt": 1,
    "density_name": "density",
    "lineout_axis": "z_m",
    "interpolation_points": {"x_mm": 1.0, "pressure_bar": 0.0},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_example():
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile

    return ExampleDensityProfile(**EXAMPLE_DENSITY_KWARGS)


def _build_asymmetric_sine():
    from inversion_fbpic.lib.density_profiles import AsymmetricSine

    return AsymmetricSine(**ASYMMETRIC_SINE_KWARGS)


def _build_smooth_sine_flattop():
    from inversion_fbpic.lib.density_profiles import SmoothSineFlattop

    return SmoothSineFlattop(**SMOOTH_SINE_FLATTOP_KWARGS)


def _build_gaussian_plus_triangle():
    from inversion_fbpic.lib.density_profiles import GaussianPlusTriangle

    return GaussianPlusTriangle(**GAUSSIAN_PLUS_TRIANGLE_KWARGS)


def _build_generalized_gaussian_plus_triangle():
    from inversion_fbpic.lib.density_profiles import GeneralizedGaussianPlusTriangle

    return GeneralizedGaussianPlusTriangle(**GENERALIZED_GAUSSIAN_PLUS_TRIANGLE_KWARGS)


def _build_modified():
    from inversion_fbpic.lib.density_profiles import SmoothSineFlattop
    from inversion_fbpic.lib.density_modifiers import (
        MatchedRadialModifier,
        ModifiedDensityProfile,
    )

    base = SmoothSineFlattop(**SMOOTH_SINE_FLATTOP_KWARGS)
    modifier = MatchedRadialModifier(matched_density=1.0e24, radial_extent=50.0e-6)
    return ModifiedDensityProfile(base_density_profile=base, modifiers=[modifier])


def _write_h5_density(tmp_path, dimension_labels: list[bytes] | None = None):
    filename = tmp_path / "density.h5"
    density = np.zeros((4, 2, 2))
    density[:, 1, 0] = [0.0, 1.0, 4.0, 0.0]

    with h5py.File(filename, "w") as h5_file:
        dataset = h5_file.create_dataset("density", data=density)
        if dimension_labels is not None:
            dataset.attrs["DIMENSION_LABELS"] = dimension_labels
        h5_file.create_dataset("z_m", data=[0.0, 1.0, 2.0, 3.0])
        h5_file.create_dataset("x_mm", data=[0.0, 1.0])
        h5_file.create_dataset("pressure_bar", data=[0.0, 1.0])

    return filename


def _write_oblique_h5_density(tmp_path):
    filename = tmp_path / "oblique_density.h5"
    z_m = np.array([0.0, 1.0, 2.0, 3.0])
    x_mm = np.array([0.0, 1.0, 2.0])
    pressure_bar = np.array([0.0, 1.0])
    density = np.empty((z_m.size, x_mm.size, pressure_bar.size))
    for z_index, z_value in enumerate(z_m):
        for x_index, x_value in enumerate(x_mm):
            density[z_index, x_index, 0] = z_value + x_value
            density[z_index, x_index, 1] = 2.0 * (z_value + x_value)

    with h5py.File(filename, "w") as h5_file:
        dataset = h5_file.create_dataset("density", data=density)
        dataset.attrs["DIMENSION_LABELS"] = [b"z_m", b"x_mm", b"pressure_bar"]
        h5_file.create_dataset("z_m", data=z_m)
        h5_file.create_dataset("x_mm", data=x_mm)
        h5_file.create_dataset("pressure_bar", data=pressure_bar)

    return filename


_PROFILE_BUILDERS = {
    "sine_squared_bump": _build_example,
    "asymmetric_sine": _build_asymmetric_sine,
    "smooth_sine_flattop": _build_smooth_sine_flattop,
    "gaussian_plus_triangle": _build_gaussian_plus_triangle,
    "generalized_gaussian_plus_triangle": _build_generalized_gaussian_plus_triangle,
    "modified_density_profile": _build_modified,
}


@pytest.fixture(params=list(_PROFILE_BUILDERS.keys()))
def profile_instance(request):
    """Parametrized fixture: one instance of each concrete profile subclass."""
    return _PROFILE_BUILDERS[request.param]()


# ===================================================================
# Unit tests — ExampleDensityProfile
# ===================================================================


class TestExampleDensityProfile:
    def test_peak_near_center(self) -> None:
        profile = _build_example()
        dens = profile.build_density_function()
        z_center = profile.start_position + profile.length / 2.0
        assert dens(z_center, 0.0) == pytest.approx(1.0, abs=1e-10)

    def test_zero_outside_support(self) -> None:
        profile = _build_example()
        dens = profile.build_density_function()
        z_before = profile.start_position - 1e-6
        z_after = profile.start_position + profile.length + 1e-6
        assert dens(z_before, 0.0) == pytest.approx(0.0, abs=1e-15)
        assert dens(z_after, 0.0) == pytest.approx(0.0, abs=1e-15)

    def test_radially_uniform(self) -> None:
        profile = _build_example()
        dens = profile.build_density_function()
        z_center = profile.start_position + profile.length / 2.0
        val_r0 = dens(z_center, 0.0)
        val_r1 = dens(z_center, 1e-3)
        assert val_r0 == pytest.approx(val_r1)

    def test_array_input(self) -> None:
        profile = _build_example()
        dens = profile.build_density_function()
        z = np.linspace(
            profile.start_position - 1e-6,
            profile.start_position + profile.length + 1e-6,
            50,
        )
        r = np.zeros_like(z)
        result = dens(z, r)
        assert isinstance(result, np.ndarray)
        assert result.shape == z.shape
        assert np.all(result >= 0.0)


# ===================================================================
# Unit tests — InterpolateFromH5Profile
# ===================================================================


class TestInterpolateFromH5Profile:
    def _build(self, tmp_path, **overrides):
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        return InterpolateFromH5Profile(
            filename=filename, **(H5_DENSITY_KWARGS | overrides)
        )

    @pytest.mark.parametrize("extension", ["yaml", "json"])
    def test_file_round_trip_resolves_h5_relative_to_config_source(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
        extension: str,
    ) -> None:
        """A config and its sibling data tree remain loadable after relocation."""
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile
        from inversion_fbpic.lib.serializable_config import SerializableConfig

        work_dir = tmp_path / "work"
        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True)
        filename = _write_h5_density(
            data_dir, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        run_dir = work_dir / "out"
        run_dir.mkdir()
        monkeypatch.chdir(work_dir)
        profile = InterpolateFromH5Profile(
            filename=filename.relative_to(work_dir), **H5_DENSITY_KWARGS
        )
        config_path = run_dir / f"profile.{extension}"
        if extension == "yaml":
            profile.to_yaml_file(config_path, comments=False)
        else:
            profile.to_json_file(config_path)

        serialized = config_path.read_text(encoding="utf-8")
        if extension == "yaml":
            import yaml

            payload = yaml.safe_load(serialized)
        else:
            import json

            payload = json.loads(serialized)
        assert payload["parameters"]["filename"] == "../data/density.h5"

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        monkeypatch.chdir(other_dir)

        relocated_work_dir = tmp_path / "relocated"
        shutil.move(str(work_dir), relocated_work_dir)
        relocated_config_path = relocated_work_dir / "out" / config_path.name
        relocated_filename = relocated_work_dir / "data" / filename.name

        loaded = SerializableConfig.from_file(relocated_config_path)
        assert isinstance(loaded, InterpolateFromH5Profile)
        assert loaded.filename == relocated_filename.resolve()
        assert loaded.nominal_density == pytest.approx(profile.nominal_density)

    def test_interpolates_normalizes_and_derives_metadata(self, tmp_path) -> None:
        profile = self._build(tmp_path)
        density = profile.build_density_function()

        assert density(1.0, 0.0) == pytest.approx(1.0)
        assert density(np.array([-1.0, 0.0, 1.0, 2.0]), 0.0) == pytest.approx(
            [0.0, 0.25, 1.0, 0.0]
        )
        assert profile.nominal_density == pytest.approx(4.0)
        assert profile.centroid == pytest.approx(0.8)
        assert profile.get_z_extent() == pytest.approx((0.0, 1.0))

    def test_offset_and_broadcasting(self, tmp_path) -> None:
        profile = self._build(tmp_path, longitudinal_offset=10.0)
        density = profile.build_density_function()

        assert density(11.0, np.zeros(3)) == pytest.approx(np.ones(3))
        assert density(np.array([10.0, 11.0]), np.array([0.0, 1.0])) == pytest.approx(
            [0.25, 1.0]
        )
        assert density(9.0, 0.0) == pytest.approx(0.0)
        assert profile.centroid == pytest.approx(10.8)
        assert profile.get_z_extent() == pytest.approx((10.0, 11.0))

    def test_warns_when_data_bounds_truncate_cutoff_support(self, tmp_path) -> None:
        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            h5_file["density"][:, 1, 0] = [2.0, 4.0, 2.0, 1.0]

        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        with pytest.warns(
            UserWarning, match="extent is truncated by the available data"
        ):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    def test_preserves_nonuniform_scalar_axis_samples(self, tmp_path) -> None:
        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            h5_file["z_m"][:] = [0.0, 0.5, 2.0, 3.0]

        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        profile = InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)
        assert profile.lineout.grid[0] == pytest.approx([-0.5, 0.0, 1.5, 2.5])

    @pytest.mark.parametrize(
        ("centering_mode", "expected_centroid", "expected_extent"),
        [
            ("centroid", 10.0, (9.2, 10.2)),
            ("left", 10.8, (10.0, 11.0)),
            ("right", 9.8, (9.0, 10.0)),
        ],
    )
    def test_centers_profile_at_selected_anchor(
        self, tmp_path, centering_mode, expected_centroid, expected_extent
    ) -> None:
        profile = self._build(
            tmp_path,
            centering_mode=centering_mode,
            longitudinal_offset=10.0,
        )

        assert profile.centroid == pytest.approx(expected_centroid)
        assert profile.get_z_extent() == pytest.approx(expected_extent)

    def test_samples_oblique_affine_path_and_preserves_centering(
        self, tmp_path
    ) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        profile = InterpolateFromH5Profile(
            filename=_write_oblique_h5_density(tmp_path),
            p_nz=1,
            p_nr=1,
            p_nt=1,
            density_name="density",
            neutral=False,
            lineout_axis={
                "z_m": {"origin": 0.0, "coefficient": 1.0},
                "x_mm": {"origin": 1.0, "coefficient": 1.0},
            },
            interpolation_points={"pressure_bar": 0.0},
            centering_mode="centroid",
            longitudinal_offset=10.0,
        )
        density = profile.build_density_function()

        assert profile.lineout.grid[0].size == 7
        assert profile.centroid == pytest.approx(10.0)
        assert density(profile.lineout.grid[0], 0.0) == pytest.approx(
            [
                1.0 / 3.0,
                4.0 / 9.0,
                5.0 / 9.0,
                2.0 / 3.0,
                7.0 / 9.0,
                8.0 / 9.0,
                1.0,
            ]
        )
        assert density(profile.lineout.grid[0][0] - 1.0e-6, 0.0) == pytest.approx(0.0)
        assert density(profile.lineout.grid[0][-1] + 1.0e-6, 0.0) == pytest.approx(0.0)

    def test_samples_affine_path_with_fixed_axis(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        profile = InterpolateFromH5Profile(
            filename=_write_h5_density(
                tmp_path,
                dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"],
            ),
            p_nz=1,
            p_nr=1,
            p_nt=1,
            density_name="density",
            neutral=False,
            lineout_axis={
                "z_m": {"origin": 0.0, "coefficient": 1.0},
                "x_mm": {"origin": 1.0, "coefficient": 0.0},
            },
            interpolation_points={"pressure_bar": 0.0},
        )
        density = profile.build_density_function()

        assert profile.lineout.grid[0] == pytest.approx([-1.0, 0.0, 1.0, 2.0])
        assert density(profile.lineout.grid[0], 0.0) == pytest.approx(
            [0.0, 0.25, 1.0, 0.0]
        )

    def test_rejects_out_of_bounds_origin_for_fixed_axis(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        with pytest.raises(ValueError, match="coordinate dataset bounds"):
            InterpolateFromH5Profile(
                filename=_write_h5_density(
                    tmp_path,
                    dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"],
                ),
                p_nz=1,
                p_nr=1,
                p_nt=1,
                density_name="density",
                neutral=False,
                lineout_axis={
                    "z_m": {"origin": 0.0, "coefficient": 1.0},
                    "x_mm": {"origin": 99.0, "coefficient": 0.0},
                },
                interpolation_points={"pressure_bar": 0.0},
            )

    def test_centroid_axis_uses_a_fixed_slice(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        profile = InterpolateFromH5Profile(
            filename=_write_oblique_h5_density(tmp_path),
            p_nz=1,
            p_nr=1,
            p_nt=1,
            density_name="density",
            neutral=False,
            lineout_axis={
                "z_m": {"origin": 0.0, "coefficient": 1.0},
                "x_mm": {"origin": 1.0, "coefficient": -1.0},
            },
            centroid_axis="z_m",
            interpolation_points={"pressure_bar": 0.0},
            centering_mode="centroid",
            longitudinal_offset=10.0,
        )

        # At x_mm=1, the z_m density weights are [1, 2, 3, 4], centering at t=2.
        assert profile.centroid == pytest.approx(10.0)
        assert profile.lineout.grid[0] == pytest.approx(
            [8.0 + sample_index / 6.0 for sample_index in range(7)]
        )

    def test_rejects_out_of_bounds_origin_for_centroid_slice(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        with pytest.raises(ValueError, match="lineout_axis origins"):
            InterpolateFromH5Profile(
                filename=_write_oblique_h5_density(tmp_path),
                p_nz=1,
                p_nr=1,
                p_nt=1,
                density_name="density",
                neutral=False,
                lineout_axis={
                    "z_m": {"origin": 0.0, "coefficient": 1.0},
                    "x_mm": {"origin": 3.0, "coefficient": -1.0},
                },
                centroid_axis="z_m",
                interpolation_points={"pressure_bar": 0.0},
                centering_mode="centroid",
            )

    def test_rejects_fixed_centroid_axis(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        with pytest.raises(ValueError, match="centroid_axis must have a nonzero"):
            InterpolateFromH5Profile(
                filename=_write_h5_density(
                    tmp_path,
                    dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"],
                ),
                p_nz=1,
                p_nr=1,
                p_nt=1,
                density_name="density",
                neutral=False,
                lineout_axis={
                    "z_m": {"origin": 0.0, "coefficient": 0.0},
                    "x_mm": {"origin": 1.0, "coefficient": 1.0},
                },
                centroid_axis="z_m",
                interpolation_points={"pressure_bar": 0.0},
            )

    def test_serializes_affine_path(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_core import _DensityProfile

        profile = self._build(
            tmp_path,
            lineout_axis={
                "z_m": {"origin": 0.0, "coefficient": 1.0},
                "x_mm": {"origin": 1.0, "coefficient": -1.0},
            },
            centroid_axis="z_m",
            interpolation_points={"pressure_bar": 0.0},
        )

        restored = _DensityProfile.from_yaml(profile.to_yaml(comments=False))
        assert restored.lineout_axis == profile.lineout_axis
        assert restored.centroid_axis == profile.centroid_axis

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"interpolation_points": {"x_mm": 1.0}}, "missing interpolation"),
            (
                {
                    "interpolation_points": {
                        "x_mm": 1.0,
                        "pressure_bar": 0.0,
                        "y_mm": 0.0,
                    }
                },
                "unexpected interpolation",
            ),
            ({"lineout_axis": "y_mm"}, "not in DIMENSION_LABELS"),
            ({"centroid_axis": "y_mm"}, "centroid_axis.*not in DIMENSION_LABELS"),
            (
                {"centroid_axis": "x_mm"},
                "centroid_axis must be included in lineout_axis",
            ),
            ({"lineout_axis": {}}, "non-empty dictionary"),
            (
                {
                    "lineout_axis": {
                        "z_m": {"origin": 0.0, "coefficient": 0.0},
                    },
                    "interpolation_points": {"x_mm": 1.0, "pressure_bar": 0.0},
                },
                "must contain at least one nonzero coefficient",
            ),
            (
                {
                    "lineout_axis": {
                        "z_m": {"origin": 0.0, "coefficient": 1.0},
                        "x_mm": {"origin": float("nan"), "coefficient": 1.0},
                    },
                    "interpolation_points": {"pressure_bar": 0.0},
                },
                "finite values",
            ),
            ({"density_cutoff_ratio": -0.1}, "must be >= 0.0"),
            ({"density_cutoff_ratio": 1.0}, "must be < 1.0"),
            (
                {
                    "interpolation_points": {
                        "x_mm": 2.0,
                        "pressure_bar": 0.0,
                    }
                },
                "coordinate dataset bounds",
            ),
        ],
    )
    def test_rejects_invalid_configuration(self, tmp_path, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            self._build(tmp_path, **kwargs)

    def test_rejects_missing_dimension_labels(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(tmp_path)
        with pytest.raises(ValueError, match="must define DIMENSION_LABELS"):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    @pytest.mark.parametrize(
        ("dimension_labels", "match"),
        [
            ([b"z_m", b"x_mm"], "one label for each density dimension"),
            (
                [b"z_m", b"x_mm", b"z_m"],
                "must not contain duplicate labels",
            ),
        ],
    )
    def test_rejects_invalid_dimension_labels(
        self, tmp_path, dimension_labels, match
    ) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(tmp_path, dimension_labels=dimension_labels)
        with pytest.raises(ValueError, match=match):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    def test_rejects_non_finite_density_samples(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            h5_file["density"][0, 0, 0] = np.nan

        with pytest.raises(ValueError, match="must contain only finite values"):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    def test_rejects_negative_density_samples(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            h5_file["density"][0, 0, 0] = -1.0

        with pytest.raises(ValueError, match="must not contain negative values"):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    def test_rejects_single_point_coordinate_axis(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = tmp_path / "single_point_axis.h5"
        with h5py.File(filename, "w") as h5_file:
            density = h5_file.create_dataset("density", data=np.ones((1, 2, 2)))
            density.attrs["DIMENSION_LABELS"] = [b"z_m", b"x_mm", b"pressure_bar"]
            h5_file.create_dataset("z_m", data=[0.0])
            h5_file.create_dataset("x_mm", data=[0.0, 1.0])
            h5_file.create_dataset("pressure_bar", data=[0.0, 1.0])

        with pytest.raises(
            ValueError,
            match="Coordinate dataset 'z_m' must contain at least two values",
        ):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    @pytest.mark.parametrize(
        ("coordinate", "replacement", "match"),
        [
            ("z_m", [0.0, 1.0, 1.0, 3.0], "strictly increasing"),
            ("x_mm", [0.0, np.inf], "must be finite"),
        ],
    )
    def test_rejects_invalid_coordinate_values(
        self, tmp_path, coordinate, replacement, match
    ) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            h5_file[coordinate][:] = replacement

        with pytest.raises(ValueError, match=match):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    @pytest.mark.parametrize(
        ("coordinate", "replacement", "match"),
        [
            ("z_m", [0.0, 1.0], "length 4"),
            ("x_mm", [[0.0, 1.0]], "one-dimensional"),
        ],
    )
    def test_rejects_malformed_coordinate_datasets(
        self, tmp_path, coordinate, replacement, match
    ) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            del h5_file[coordinate]
            h5_file.create_dataset(coordinate, data=replacement)

        with pytest.raises(ValueError, match=match):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)

    def test_rejects_missing_coordinate_dataset(self, tmp_path) -> None:
        from inversion_fbpic.lib.density_profiles import InterpolateFromH5Profile

        filename = _write_h5_density(
            tmp_path, dimension_labels=[b"z_m", b"x_mm", b"pressure_bar"]
        )
        with h5py.File(filename, "r+") as h5_file:
            del h5_file["pressure_bar"]

        with pytest.raises(ValueError, match="Coordinate dataset 'pressure_bar'"):
            InterpolateFromH5Profile(filename=filename, **H5_DENSITY_KWARGS)


# ===================================================================
# Unit tests — get_plasma_wavelength
# ===================================================================


class TestGetPlasmaWavelength:
    def test_nominal(self) -> None:
        profile = _build_example()
        expected = 3.3e7 / np.sqrt(profile.nominal_density)
        assert profile.get_plasma_wavelength() == pytest.approx(expected, rel=1e-6)

    def test_position_dependent(self) -> None:
        profile = _build_example()
        z_center = profile.start_position + profile.length / 2.0
        result = profile.get_plasma_wavelength(z=z_center, r=0.0)
        assert result > 0.0
        assert np.isfinite(result)

    def test_zero_density_returns_inf(self) -> None:
        profile = _build_example()
        z_outside = profile.start_position - 1e-3
        result = profile.get_plasma_wavelength(z=z_outside, r=0.0)
        assert result == float("inf")


# ===================================================================
# Unit tests — MatchedRadialModifier
# ===================================================================


class TestMatchedRadialModifier:
    def test_modify_increases_density_with_r(self) -> None:
        from inversion_fbpic.lib.density_modifiers import MatchedRadialModifier

        def flat_z_profile(z, r):
            return np.ones_like(np.asarray(z), dtype=float)

        modifier = MatchedRadialModifier(matched_density=1.0e24, radial_extent=50e-6)
        modified = modifier.modify_density_function(flat_z_profile)

        val_r0 = modified(0.0, 0.0)
        val_r_nonzero = modified(0.0, 20e-6)
        assert val_r_nonzero > val_r0

    def test_get_r_extent_returns_value(self) -> None:
        from inversion_fbpic.lib.density_modifiers import MatchedRadialModifier

        modifier = MatchedRadialModifier(matched_density=1.0e24, radial_extent=50e-6)
        assert modifier.get_r_extent() == pytest.approx(50e-6)

    def test_get_r_extent_none(self) -> None:
        from inversion_fbpic.lib.density_modifiers import MatchedRadialModifier

        modifier = MatchedRadialModifier(matched_density=1.0e24)
        assert modifier.get_r_extent() is None

    def test_get_z_extent_is_none(self) -> None:
        from inversion_fbpic.lib.density_modifiers import MatchedRadialModifier

        modifier = MatchedRadialModifier(matched_density=1.0e24)
        assert modifier.get_z_extent() is None


# ===================================================================
# Unit tests — ModifiedDensityProfile
# ===================================================================


class TestModifiedDensityProfile:
    def test_inherits_base_attributes(self) -> None:
        profile = _build_modified()
        base = profile.resolved_base_density_profile
        assert profile.nominal_density == base.nominal_density
        assert profile.p_nz == base.p_nz
        assert profile.species == base.species
        assert profile.ionization == base.ionization

    def test_z_extent_at_least_as_wide_as_base(self) -> None:
        profile = _build_modified()
        base_extent = profile.resolved_base_density_profile.get_z_extent()
        mod_extent = profile.get_z_extent()
        assert mod_extent[0] <= base_extent[0]
        assert mod_extent[1] >= base_extent[1]

    def test_r_extent_from_modifier(self) -> None:
        profile = _build_modified()
        assert profile.get_r_extent() == pytest.approx(50e-6)

    def test_build_density_function_callable(self) -> None:
        profile = _build_modified()
        dens = profile.build_density_function()
        z = np.linspace(0.0, 1e-3, 20)
        r = np.zeros_like(z)
        result = dens(z, r)
        assert result.shape == z.shape
        assert np.all(np.isfinite(result))

    def test_modified_density_has_radial_dependence(self) -> None:
        profile = _build_modified()
        dens = profile.build_density_function()
        z_mid = (
            SMOOTH_SINE_FLATTOP_KWARGS["offset_length"]
            + SMOOTH_SINE_FLATTOP_KWARGS["upramp_length"]
            + SMOOTH_SINE_FLATTOP_KWARGS["flattop_width"] / 2.0
        )
        val_r0 = dens(z_mid, 0.0)
        val_r_nonzero = dens(z_mid, 20e-6)
        if val_r0 > 0:
            assert val_r_nonzero > val_r0

    def test_from_dict_round_trip(self) -> None:
        profile = _build_modified()
        d = profile.to_dict()

        from inversion_fbpic.lib.density_modifiers import ModifiedDensityProfile

        reloaded = ModifiedDensityProfile.from_dict(d)
        assert reloaded.nominal_density == profile.nominal_density
        assert reloaded.get_z_extent() == profile.get_z_extent()


# ===================================================================
# Parametrized whole-class instance tests
# ===================================================================


class TestAllDensityProfiles:
    def test_build_density_function_returns_callable(self, profile_instance) -> None:
        dens = profile_instance.build_density_function()
        z = np.linspace(-1e-3, 5e-3, 100)
        r = np.zeros_like(z)
        result = dens(z, r)
        assert isinstance(result, np.ndarray)
        assert result.shape == z.shape

    def test_density_non_negative(self, profile_instance) -> None:
        dens = profile_instance.build_density_function()
        z_min, z_max = profile_instance.get_z_extent()
        z = np.linspace(z_min, z_max, 200)
        r = np.zeros_like(z)
        result = dens(z, r)
        assert np.all(result >= -1e-15), f"Negative density found: {result.min()}"

    def test_density_zero_far_outside_support(self, profile_instance) -> None:
        z_min, z_max = profile_instance.get_z_extent()
        span = z_max - z_min
        dens = profile_instance.build_density_function()
        z_far_left = z_min - 10 * span
        z_far_right = z_max + 10 * span
        assert dens(z_far_left, 0.0) == pytest.approx(0.0, abs=1e-6)
        assert dens(z_far_right, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_get_z_extent_returns_ordered_pair(self, profile_instance) -> None:
        z_min, z_max = profile_instance.get_z_extent()
        assert z_min < z_max

    def test_get_r_extent_none_or_positive(self, profile_instance) -> None:
        r_ext = profile_instance.get_r_extent()
        assert r_ext is None or r_ext > 0.0

    def test_yaml_round_trip(self, profile_instance) -> None:
        yaml_str = profile_instance.to_yaml(comments=False)
        from inversion_fbpic.lib.density_core import _DensityProfile

        reloaded = _DensityProfile.from_yaml(yaml_str)
        assert type(reloaded) is type(profile_instance)
        orig_params = profile_instance.to_dict()["parameters"]
        reload_params = reloaded.to_dict()["parameters"]
        for key in orig_params:
            val = orig_params[key]
            if val is None or isinstance(val, (dict, list)):
                continue
            assert reload_params[key] == pytest.approx(
                val, rel=1e-6
            ), f"Mismatch on {key}: {val} vs {reload_params[key]}"

    def test_get_plasma_wavelength_nominal(self, profile_instance) -> None:
        wl = profile_instance.get_plasma_wavelength()
        expected = 3.3e7 / np.sqrt(profile_instance.nominal_density)
        assert wl == pytest.approx(expected, rel=1e-6)
