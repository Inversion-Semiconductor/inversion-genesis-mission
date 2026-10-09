from __future__ import annotations

import json

import pytest

from fludat_fit.generate_profile_configs import main as generate_main
from fludat_fit.plot_profile_configs import build_curves, main, plot_profile_config


@pytest.fixture
def config_and_path(small_cube_path, tmp_path):
    """A filled config (2 profiles, one of each fittable class) for the small cube."""
    template = {
        "_comment": "test template",
        "density_file": None,
        "x_mm": None,
        "p_bar": None,
        "angle": None,
        "seeds": [0],
        "profiles": {
            "lorentzian_sum_1term": {
                "class": "GeneralizedLorentzianSum",
                "centroid": None,
                "kwargs": {
                    "parameters": [
                        {"A": None, "c": None, "w": None, "b": None, "m": None}
                    ]
                },
            },
            "conical_supergaussian_no_fringes": {
                "class": "GenericConicalTarget",
                "kwargs": {
                    "nominal_density": None,
                    "center": None,
                    "main_profile_type": "supergaussian",
                    "main_profile_parameters": {"fwhm": None, "beta": None},
                    "skew_rate": None,
                },
            },
            "h5_direct": {
                "class": "InterpolateFromH5Profile",
                "kwargs": {
                    "density_name": "density",
                    "lineout_axis": "z_m",
                    "centering_mode": "left",
                },
            },
        },
    }
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template))
    output_dir = tmp_path / "cfgs"
    code = generate_main(
        [
            str(small_cube_path),
            str(template_path),
            "--point",
            "1.0",
            "10.0",
            "0.0",
            "--starts",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert code == 0
    config_path = output_dir / "profile_parameters_x1mm_p10bar_a0deg.json"
    return json.loads(config_path.read_text()), config_path


def test_build_curves_reconstructs_every_fitted_profile(config_and_path):
    config, config_path = config_and_path
    curves, reference = build_curves(config, config_path)

    assert [c.name for c in curves] == [
        "h5_direct (data)",
        "lorentzian_sum_1term",
        "conical_supergaussian_no_fringes",
    ]
    assert reference.density_units == "m^-3"
    # cube is in cm^-3; the reference curve must be converted to m^-3.
    assert reference.density.max() == pytest.approx(2.0e19 * 1.0e6, rel=0.05)

    data_curve, lorentzian_curve, conical_curve = curves
    assert data_curve.center is None
    assert lorentzian_curve.center is not None and conical_curve.center is not None

    # Each fitted profile is drawn over its own get_z_extent(), not a shared window,
    # so their ranges need not match; the reference spans their union exactly.
    assert reference.z.min() == pytest.approx(
        min(lorentzian_curve.z.min(), conical_curve.z.min())
    )
    assert reference.z.max() == pytest.approx(
        max(lorentzian_curve.z.max(), conical_curve.z.max())
    )
    # The conical profile's own extent should be tighter than the Lorentzian sum's
    # (a generalized Lorentzian's tails decay slowly under its density cutoff).
    conical_width = conical_curve.z.max() - conical_curve.z.min()
    lorentzian_width = lorentzian_curve.z.max() - lorentzian_curve.z.min()
    assert conical_width < lorentzian_width

    # The config's own "center"/"centroid" fields are in each profile's own
    # left-edge-zero output frame (see generate_profile_configs); for plotting,
    # build_curves translates every profile back to the one anchor they were
    # all actually fit against, the lineout's z_m=0 crossing -- t=0 here, since
    # this is an axial (angle=0) lineout at x_mm=1.0.
    assert lorentzian_curve.center == pytest.approx(0.0, abs=1.0e-9)
    assert conical_curve.center == pytest.approx(0.0, abs=1.0e-9)


def test_plot_profile_config_draws_a_line_and_vline_per_profile(config_and_path):
    config, config_path = config_and_path
    figure = plot_profile_config(config, config_path)
    (ax,) = figure.axes
    # 1 data line (no vline) + 2 fitted profiles, each with a curve and a dashed vline.
    assert len(ax.get_lines()) == 1 + 2 * 2
    assert ax.get_legend() is not None
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == [
        "h5_direct (data)",
        "lorentzian_sum_1term",
        "conical_supergaussian_no_fringes",
    ]
    assert ax.get_xlabel() == "z [mm]"
    assert "x=1 mm" in ax.get_title()


def test_cli_writes_one_png_per_config(config_and_path, tmp_path):
    _, config_path = config_and_path
    output_dir = tmp_path / "plots"
    code = main([str(config_path), "--output-dir", str(output_dir)])
    assert code == 0
    assert (output_dir / f"{config_path.stem}.png").is_file()


def test_unrecognised_profile_class_is_skipped_with_a_warning(config_and_path, capsys):
    config, config_path = config_and_path
    config = dict(config)
    config["profiles"] = {
        **config["profiles"],
        "mystery": {"class": "SomeFutureProfile", "kwargs": {}},
    }
    curves, _ = build_curves(config, config_path)
    assert "mystery" not in [c.name for c in curves]
    assert "don't know how to plot 'mystery'" in capsys.readouterr().err


def test_build_curves_requires_at_least_one_fittable_profile(small_cube_path, tmp_path):
    template_path = tmp_path / "t.json"
    template_path.write_text(
        json.dumps(
            {
                "density_file": "does_not_matter.h5",
                "x_mm": 1.0,
                "p_bar": 10.0,
                "angle": 0.0,
                "profiles": {
                    "h5_direct": {
                        "class": "InterpolateFromH5Profile",
                        "kwargs": {
                            "density_name": "density",
                            "lineout_axis": "z_m",
                            "centering_mode": "left",
                        },
                    }
                },
            }
        )
    )
    config = json.loads(template_path.read_text())
    with pytest.raises(ValueError, match="no fittable profile"):
        build_curves(config, template_path)


def test_angled_point_uses_the_lineout_label(small_cube_path, tmp_path):
    template_path = tmp_path / "t.json"
    template_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "conical_supergaussian_no_fringes": {
                        "class": "GenericConicalTarget",
                        "kwargs": {
                            "nominal_density": None,
                            "center": None,
                            "main_profile_type": "supergaussian",
                            "main_profile_parameters": {"fwhm": None, "beta": None},
                            "skew_rate": None,
                        },
                    }
                }
            }
        )
    )
    output_dir = tmp_path / "out"
    generate_main(
        [
            str(small_cube_path),
            str(template_path),
            "--point",
            "1.0",
            "10.0",
            "20.0",
            "--starts",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )
    config_path = output_dir / "profile_parameters_x1mm_p10bar_a20deg.json"
    config = json.loads(config_path.read_text())
    figure = plot_profile_config(config, config_path)
    (ax,) = figure.axes
    assert ax.get_xlabel().startswith("t along")
