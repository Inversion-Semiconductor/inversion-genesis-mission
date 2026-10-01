from __future__ import annotations

import pytest
from inversion_fbpic.lib.density_profiles import (
    GeneralizedLorentzianSum,
    GenericConicalTarget,
)

from fludat_fit.profile_reconstruction import (
    build_conical,
    build_lorentzian_sum,
    build_profile,
    center_of,
)

CONICAL_SPEC = {
    "class": "GenericConicalTarget",
    "kwargs": {
        "nominal_density": 3.0e24,
        "center": 4.0e-4,
        "main_profile_type": "supergaussian",
        "main_profile_parameters": {"fwhm": 2.0e-3, "beta": 2.5},
        "skew_rate": 12.0,
    },
}

LORENTZIAN_SUM_SPEC = {
    "class": "GeneralizedLorentzianSum",
    "centroid": -1.0e-4,
    "kwargs": {
        "parameters": [
            {"A": 2.0e24, "c": -1.0e-4, "w": 1.0e-3, "b": 2.0, "m": 1.5},
            {"A": 5.0e23, "c": 3.0e-4, "w": 5.0e-4, "b": 2.0, "m": 1.0},
        ]
    },
}

H5_SPEC = {
    "class": "InterpolateFromH5Profile",
    "kwargs": {
        "density_name": "density",
        "lineout_axis": "z_m",
        "centering_mode": "left",
    },
}


def test_build_conical_sets_center_as_the_profiles_own_centroid():
    profile = build_conical(CONICAL_SPEC["kwargs"])
    assert isinstance(profile, GenericConicalTarget)
    assert profile.centroid == pytest.approx(CONICAL_SPEC["kwargs"]["center"])
    assert profile.nominal_density == pytest.approx(3.0e24)
    assert profile.species is None and profile.ionization == 0


def test_build_lorentzian_sum_matches_the_given_terms():
    profile = build_lorentzian_sum(LORENTZIAN_SUM_SPEC["kwargs"])
    assert isinstance(profile, GeneralizedLorentzianSum)
    assert [term.c for term in profile.parameters] == pytest.approx([-1.0e-4, 3.0e-4])
    assert profile.nominal_density == pytest.approx(2.0e24)  # max |A|


def test_build_profile_dispatches_on_class_and_skips_h5_direct():
    assert isinstance(build_profile("c", CONICAL_SPEC), GenericConicalTarget)
    assert isinstance(build_profile("l", LORENTZIAN_SUM_SPEC), GeneralizedLorentzianSum)
    assert build_profile("h", H5_SPEC) is None


def test_center_of_reads_the_right_field_per_class():
    assert center_of(CONICAL_SPEC) == pytest.approx(CONICAL_SPEC["kwargs"]["center"])
    assert center_of(LORENTZIAN_SUM_SPEC) == pytest.approx(
        LORENTZIAN_SUM_SPEC["centroid"]
    )
    assert center_of(H5_SPEC) is None
