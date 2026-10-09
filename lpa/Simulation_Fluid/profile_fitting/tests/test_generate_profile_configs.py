from __future__ import annotations

import copy
from dataclasses import replace
import json

import pytest

from fludat_fit.cli_common import FitWindow
from fludat_fit.dataset import NozzleDataset
from fludat_fit.families import (
    FixedParameterFamily,
    GeneralizedLorentzianSumFamily,
    GenericConicalTargetFamily,
)
from fludat_fit.fitting import MultiStartLocalFit
from fludat_fit.generate_profile_configs import (
    _conical_kwargs,
    _lorentzian_sum_kwargs,
    fill_template,
    main,
)
from fludat_fit.lineout import LineoutConditions
from fludat_fit.profile_reconstruction import build_profile, center_of

TEMPLATE = {
    "profiles": {
        "lorentzian_sum_1term": {
            "class": "GeneralizedLorentzianSum",
            "centroid": None,
            "kwargs": {
                "parameters": [{"A": None, "c": None, "w": None, "b": None, "m": None}],
                "density_cutoff_ratio": None,
            },
        },
        "lorentzian_sum_3term": {
            "class": "GeneralizedLorentzianSum",
            "centroid": None,
            "kwargs": {
                "parameters": [
                    {"A": None, "c": None, "w": None, "b": None, "m": None},
                    {"A": None, "c": None, "w": None, "b": None, "m": None},
                    {"A": None, "c": None, "w": None, "b": None, "m": None},
                ],
                "density_cutoff_ratio": None,
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


def _left_edge(name: str, spec: dict) -> float:
    """``get_z_extent()[0]`` of the real profile reconstructed from ``spec``."""
    z_min, _z_max = build_profile(name, spec).get_z_extent()
    return z_min


def _fitted_profiles(config: dict) -> dict:
    return {
        name: spec
        for name, spec in config["profiles"].items()
        if spec["class"] != "InterpolateFromH5Profile"
    }


@pytest.fixture
def windowed_lineout(small_cube_path):
    dataset = NozzleDataset.load(small_cube_path)
    conditions = LineoutConditions(1.0, 10.0, 0.0)
    return FitWindow(max_points=150).apply(dataset.lineout(conditions))


def _fill(small_cube_path, conditions, *, starts=3):
    dataset = NozzleDataset.load(small_cube_path)
    scheme = MultiStartLocalFit(n_starts=starts, seed=0)
    config = fill_template(
        copy.deepcopy(TEMPLATE),
        dataset,
        conditions,
        window=FitWindow(max_points=150),
        scheme=scheme,
        density_file="does_not_matter.h5",
    )
    lineout = FitWindow(max_points=150).apply(dataset.lineout(conditions))
    return config, lineout


def test_every_fitted_profile_has_its_left_edge_at_zero(small_cube_path):
    """The re-basing convention: after fitting (pinned to the z_m=0 crossing)
    and shifting, every profile's own get_z_extent() starts at exactly 0,
    matching h5_direct's own centering_mode="left" convention (see the module
    docstring). This holds regardless of where each shape's support happens to
    sit relative to its pinned anchor."""
    config, _lineout = _fill(small_cube_path, LineoutConditions(1.0, 10.0, 0.0))
    fitted = _fitted_profiles(config)
    for name, spec in fitted.items():
        assert _left_edge(name, spec) == pytest.approx(0.0, abs=1.0e-9), name
    assert len(fitted) == 3  # every fittable template entry was checked


def test_reported_center_matches_the_rebased_profiles_own_notion_of_center(
    small_cube_path,
):
    """``center``/``centroid`` in the written config is not an arbitrary
    number: for GenericConicalTarget it is (by class construction) exactly
    ``profile.centroid``, and for GeneralizedLorentzianSum it is the pinned
    dominant term's position — both after the same left-edge-zero shift."""
    config, _lineout = _fill(small_cube_path, LineoutConditions(2.0, 10.0, 30.0))
    for name, spec in _fitted_profiles(config).items():
        profile = build_profile(name, spec)
        reported = center_of(spec)
        if spec["class"] == "GenericConicalTarget":
            assert reported == pytest.approx(profile.centroid), name
        else:
            assert reported == pytest.approx(profile.parameters[0].c), name


def test_fill_template_warns_when_a_fit_fails(small_cube_path, monkeypatch):
    conditions = LineoutConditions(1.0, 10.0, 0.0)
    template = {
        "profiles": {
            "conical": copy.deepcopy(
                TEMPLATE["profiles"]["conical_supergaussian_no_fringes"]
            )
        }
    }
    dataset = NozzleDataset.load(small_cube_path)
    scheme = MultiStartLocalFit(n_starts=1, seed=0)
    fit = scheme.fit

    def failed_fit(lineout, family):
        return replace(
            fit(lineout, family),
            success=False,
            message="forced fit failure",
        )

    monkeypatch.setattr(scheme, "fit", failed_fit)
    with pytest.warns(RuntimeWarning, match=r"conical.*forced fit failure"):
        config = fill_template(
            template,
            dataset,
            conditions,
            window=FitWindow(max_points=150),
            scheme=scheme,
            density_file="does_not_matter.h5",
        )

    assert config["profiles"]["conical"]["kwargs"]["nominal_density"] > 0.0


def test_oblique_lineout_profiles_are_each_rebased_to_their_own_left_edge(
    small_cube_path,
):
    """An oblique line samples the (z-symmetric) target asymmetrically, so the
    lineout's own density-weighted centroid is measurably nonzero even though
    every profile is pinned to the target's true centre (z_m = 0) before being
    shifted. Each shape's support is generally asymmetric about that anchor
    differently (skew, fringes, satellite terms), so the shift — and hence the
    final reported center — differs per profile even though they share the
    same pre-shift anchor."""
    conditions = LineoutConditions(2.0, 10.0, 30.0)
    config, lineout = _fill(small_cube_path, conditions)

    assert (
        abs(lineout.centroid()) > 1.0e-5
    )  # confirms the sampling really is asymmetric

    reported_centers = []
    for name, spec in _fitted_profiles(config).items():
        assert _left_edge(name, spec) == pytest.approx(0.0, abs=1.0e-9), name
        reported_centers.append(center_of(spec))
    assert len({round(c, 12) for c in reported_centers}) > 1  # shifts differ per shape


def test_conical_kwargs_shifts_center_so_the_profile_starts_at_zero(windowed_lineout):
    target = 9.87e-4
    family = FixedParameterFamily(
        GenericConicalTargetFamily("supergaussian"), "center", target
    )
    result = MultiStartLocalFit(n_starts=3, seed=0).fit(windowed_lineout, family)
    template_kwargs = TEMPLATE["profiles"]["conical_supergaussian_no_fringes"]["kwargs"]

    kwargs = _conical_kwargs(result, template_kwargs, target)
    spec = {"class": "GenericConicalTarget", "kwargs": kwargs}
    profile = build_profile("conical", spec)

    assert _left_edge("conical", spec) == pytest.approx(0.0, abs=1.0e-9)
    assert profile.centroid == pytest.approx(kwargs["center"])
    assert kwargs["main_profile_parameters"]["fwhm"] == pytest.approx(
        result.parameters["fwhm"]
    )  # shape parameters are untouched by the shift


def test_conical_kwargs_treats_null_fringe_type_as_no_fringe(windowed_lineout):
    target = 9.87e-4
    family = FixedParameterFamily(
        GenericConicalTargetFamily("supergaussian"), "center", target
    )
    result = MultiStartLocalFit(n_starts=3, seed=0).fit(windowed_lineout, family)
    template_kwargs = {
        "main_profile_type": "supergaussian",
        "main_profile_parameters": {"fwhm": None, "beta": None},
        "fringe_profile_type": None,
        "fringe_profile_parameters": {},
    }

    kwargs = _conical_kwargs(result, template_kwargs, target)

    assert "fringe_profile_type" not in kwargs
    assert "fringe_profile_parameters" not in kwargs
    assert "fringe_center_offset" not in kwargs
    assert "fringe_relative_height" not in kwargs


def test_conical_kwargs_converts_cosine_squared_ramp_fraction(windowed_lineout):
    target = 9.87e-4
    family = FixedParameterFamily(
        GenericConicalTargetFamily("cosine_squared_flattop"), "center", target
    )
    result = MultiStartLocalFit(n_starts=3, seed=0).fit(windowed_lineout, family)
    template_kwargs = {
        "main_profile_type": "cosine_squared_flattop",
        "main_profile_parameters": {"fwhm": None, "ramp_length": None},
    }

    kwargs = _conical_kwargs(result, template_kwargs, target)

    assert kwargs["main_profile_parameters"] == pytest.approx(
        {
            "fwhm": result.parameters["fwhm"],
            "ramp_length": result.parameters["ramp_fraction"]
            * result.parameters["fwhm"],
        }
    )


def test_lorentzian_sum_kwargs_shifts_every_term_by_the_same_constant(
    windowed_lineout,
):
    """The dominant term was already pinned to ``target`` during fitting (see
    ``_family_for``); every term's position is then shifted by one common
    constant — an exact rigid translation, so no shape parameter changes — and
    that same shift is added to ``centroid``, so the reported anchor and the
    profile's own pinned term stay in agreement."""
    target = 1.23e-3
    family = FixedParameterFamily(
        GeneralizedLorentzianSumFamily(n_terms=3), "c_0", target
    )
    result = MultiStartLocalFit(n_starts=3, seed=0).fit(windowed_lineout, family)
    before = result.build_profile()
    fitted_terms = before.parameters

    kwargs, centroid = _lorentzian_sum_kwargs(result, target)

    shift = kwargs["parameters"][0]["c"] - fitted_terms[0].c
    assert centroid == pytest.approx(target + shift)
    for term_before, term_after in zip(fitted_terms, kwargs["parameters"], strict=True):
        assert term_after["c"] == pytest.approx(term_before.c + shift)
        assert term_after["A"] == term_before.A
        assert term_after["w"] == term_before.w
        assert term_after["b"] == term_before.b
        assert term_after["m"] == term_before.m
    assert kwargs["density_cutoff_ratio"] == before.density_cutoff_ratio

    spec = {"class": "GeneralizedLorentzianSum", "centroid": centroid, "kwargs": kwargs}
    assert _left_edge("lorentzian", spec) == pytest.approx(0.0, abs=1.0e-9)


def test_cli_end_to_end_rebases_every_profile_to_its_own_left_edge(
    small_cube_path, tmp_path
):
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(TEMPLATE))
    output_dir = tmp_path / "cfgs"

    code = main(
        [
            str(small_cube_path),
            str(template_path),
            "--point",
            "2.0",
            "10.0",
            "30.0",
            "--starts",
            "3",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert code == 0

    config_path = output_dir / "profile_parameters_x2mm_p10bar_a30deg.json"
    config = json.loads(config_path.read_text())

    for name, spec in _fitted_profiles(config).items():
        assert _left_edge(name, spec) == pytest.approx(0.0, abs=1.0e-9), name
