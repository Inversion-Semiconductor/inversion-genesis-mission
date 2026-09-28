from __future__ import annotations

import numpy as np
import pytest

from fludat_fit.families import (
    DEFAULT_FAMILY_NAMES,
    FRINGE_PROFILE_TYPES,
    MAIN_PROFILE_TYPES,
    GeneralizedLorentzianSumFamily,
    GenericConicalTargetFamily,
    ProfileFamily,
    family_registry,
    get_families,
)

from .conftest import make_lineout

ALL_FAMILIES = list(family_registry().values()) + [
    GenericConicalTargetFamily(main, fringe, fringe_side=side)
    for main in MAIN_PROFILE_TYPES
    for fringe in FRINGE_PROFILE_TYPES
    for side in ("left", "right")
]


def _summary(z_grid):
    lineout = make_lineout(np.exp(-(z_grid**2) / (2 * (1.0e-3) ** 2)), z_grid)
    return lineout.summary()


def test_registry_names_are_unique_and_resolvable():
    assert len(set(DEFAULT_FAMILY_NAMES)) == len(DEFAULT_FAMILY_NAMES)
    assert [f.name for f in get_families()] == list(DEFAULT_FAMILY_NAMES)
    assert [f.name for f in get_families(["asymmetric_sine"])] == ["asymmetric_sine"]
    with pytest.raises(KeyError, match="unknown profile family"):
        get_families(["nope"])
    assert "conical:supergaussian+cosine_squared" in DEFAULT_FAMILY_NAMES


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.name)
def test_family_builds_valid_profiles_across_its_space(family: ProfileFamily, z_grid):
    summary = _summary(z_grid)
    space = family.parameter_space(summary)
    assert space.dimension >= 3

    guess = family.initial_guess(summary)
    assert guess is not None and set(guess) == set(space.names)
    assert space.contains(space.clip(space.from_dict(guess)))

    for theta in [space.center(), *space.sample(12, seed=1)]:
        parameters = space.as_dict(theta)
        relative = family.relative_density(parameters, z_grid)
        assert relative.shape == z_grid.shape
        assert np.all(np.isfinite(relative)) and np.all(relative >= 0.0)
        profile = family.build(
            parameters, nominal_density=3.0e24, species="H", ionization=0
        )
        assert isinstance(profile, family.profile_class)
        assert profile.nominal_density == pytest.approx(3.0e24)
        assert profile.species == "H"
        z_min, z_max = profile.get_z_extent()
        assert z_min < z_max


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.name)
def test_profile_round_trips_through_its_config_dict(family: ProfileFamily, z_grid):
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    summary = _summary(z_grid)
    space = family.parameter_space(summary)
    parameters = space.as_dict(space.center())
    profile = family.build(parameters, nominal_density=1.0e24)
    rebuilt = SerializableConfig.from_dict(profile.to_dict())
    np.testing.assert_allclose(
        rebuilt.build_density_function()(z_grid, 0 * z_grid),
        profile.build_density_function()(z_grid, 0 * z_grid),
    )


def test_centered_families_put_the_centroid_where_asked(z_grid):
    summary = _summary(z_grid)
    for family in [
        get_families(["power_law_flattop"])[0],
        GenericConicalTargetFamily("cosine_squared_flattop"),
    ]:
        space = family.parameter_space(summary)
        parameters = space.as_dict(space.center())
        parameters["center"] = 1.25e-3
        profile = family.build(parameters)
        assert profile.centroid == pytest.approx(1.25e-3)
        assert profile.species is None and profile.ionization == 0


def test_lorentzian_sum_amplitudes_scale_with_nominal_density(z_grid):
    family = GeneralizedLorentzianSumFamily(n_terms=2)
    space = family.parameter_space(_summary(z_grid))
    assert space.dimension == 9 and "A_1" in space.names and "A_0" not in space.names
    parameters = space.as_dict(space.center())
    parameters["A_1"] = -0.4
    profile = family.build(parameters, nominal_density=5.0e24)
    assert [term.A for term in profile.parameters] == pytest.approx([5.0e24, -2.0e24])
    assert profile.nominal_density == pytest.approx(5.0e24)
    with pytest.raises(ValueError):
        GeneralizedLorentzianSumFamily(n_terms=0)


def test_cosine_flattop_ramp_constraint_is_a_bound(z_grid):
    family = GenericConicalTargetFamily("cosine_squared_flattop", fit_skew=False)
    space = family.parameter_space(_summary(z_grid))
    assert "skew_rate" not in space.names
    parameters = space.as_dict(space.upper)  # ramp_fraction = 1 -> ramp_length == fwhm
    profile = family.build(parameters)
    assert profile.main_profile_parameters["ramp_length"] == pytest.approx(
        profile.main_profile_parameters["fwhm"]
    )


def test_conical_family_rejects_unknown_types():
    with pytest.raises(ValueError):
        GenericConicalTargetFamily("gaussian")
    with pytest.raises(ValueError):
        GenericConicalTargetFamily("supergaussian", "triangle")
    with pytest.raises(ValueError):
        GenericConicalTargetFamily("supergaussian", "lorentzian", fringe_side="up")
