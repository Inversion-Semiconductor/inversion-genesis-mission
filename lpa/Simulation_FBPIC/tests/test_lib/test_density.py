"""
Tests for species, ionization, and particle bounds on DensityProfile.

Covers the outer product of:

* species: ``None``, ``"H"``, and ``"He"`` (too few levels for high ionization)
* ionization: ``None``, ``-1`` (fully ionized), ``2``, and a value too large for the species

Run from Simulation_FBPIC::

    pytest tests/test_lib/test_density.py -v
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from inversion_fbpic.lib.density_core import _DensityProfile
from inversion_fbpic.lib.density_profiles import ExampleDensityProfile

# Minimal constructor kwargs shared by all ExampleDensityProfile instances.
_MINIMAL_PROFILE_KWARGS = {
    "nominal_density": 1.0e24,
    "length": 1.0e-5,
    "p_nz": 1,
    "p_nr": 1,
    "p_nt": 1,
}

# Ionization level that exceeds helium's capacity (2 levels) but is valid for heavier ions.
_IONIZATION_TOO_LARGE = 99

_SPECIES_IONIZATION_CASES: list[pytest.param] = [
    pytest.param(
        None,
        None,
        None,
        1,
        False,
        False,
        id="species_none-ionization_none",
    ),
    pytest.param(
        None,
        0,
        None,
        1,
        False,
        False,
        id="species_none-ionization_zero_is_the_default",
    ),
    pytest.param(
        None,
        -1,
        None,
        1,
        False,
        False,
        id="species_none-ionization_fully_ionized",
    ),
    pytest.param(
        None,
        1,
        None,
        1,
        False,
        False,
        id="species_none-ionization_one_is_the_stored_form",
    ),
    pytest.param(
        None,
        2,
        None,
        None,
        True,
        False,
        id="species_none-ionization_2",
    ),
    pytest.param(
        None,
        _IONIZATION_TOO_LARGE,
        None,
        None,
        True,
        False,
        id="species_none-ionization_too_large",
    ),
    pytest.param(
        "H",
        None,
        "H",
        0,
        False,
        False,
        id="species_h-ionization_none",
    ),
    pytest.param(
        "H",
        -1,
        "H",
        1,
        False,
        False,
        id="species_h-ionization_fully_ionized",
    ),
    pytest.param(
        "H",
        2,
        "H",
        None,
        True,
        False,
        id="species_h-ionization_2",
    ),
    pytest.param(
        "H",
        _IONIZATION_TOO_LARGE,
        "H",
        None,
        True,
        False,
        id="species_h-ionization_too_large",
    ),
    pytest.param(
        "He",
        None,
        "He",
        0,
        False,
        False,
        id="species_too_small-ionization_none",
    ),
    pytest.param(
        "He",
        -1,
        "He",
        2,
        False,
        False,
        id="species_too_small-ionization_fully_ionized",
    ),
    pytest.param(
        "He",
        2,
        "He",
        2,
        False,
        False,
        id="species_too_small-ionization_2",
    ),
    pytest.param(
        "He",
        _IONIZATION_TOO_LARGE,
        "He",
        None,
        True,
        False,
        id="species_too_small-ionization_too_large",
    ),
]


def _build_profile(
    species: str | None, ionization: int | None
) -> ExampleDensityProfile:
    return ExampleDensityProfile(
        species=species,
        ionization=ionization,
        **_MINIMAL_PROFILE_KWARGS,
    )


@pytest.mark.parametrize(
    (
        "species",
        "ionization",
        "expected_species",
        "expected_ionization",
        "expect_error",
        "expect_hydrogen_assumption_warning",
    ),
    _SPECIES_IONIZATION_CASES,
)
def test_species_ionization_outer_product(
    species: str | None,
    ionization: int | None,
    expected_species: str | None,
    expected_ionization: int | None,
    expect_error: bool,
    expect_hydrogen_assumption_warning: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validate species/ionization normalization and error handling."""
    if expect_error:
        with pytest.raises(ValueError, match="ionization"):
            _build_profile(species, ionization)
        return

    profile = _build_profile(species, ionization)
    assert profile.species == expected_species
    assert profile.ionization == expected_ionization

    captured = capsys.readouterr()
    if expect_hydrogen_assumption_warning:
        assert "assuming Hydrogen" in captured.out
    else:
        assert "assuming Hydrogen" not in captured.out


def test_species_none_without_explicit_ionization_is_bare_electrons() -> None:
    """The default `ionization` (0) must not make `species=None` raise."""
    profile = ExampleDensityProfile(species=None, **_MINIMAL_PROFILE_KWARGS)
    assert profile.species is None
    assert profile.ionization == 1


@pytest.mark.parametrize(
    "species,expected_levels",
    [
        ("H", 1),
        ("He", 2),
    ],
)
def test_get_num_ionization_levels(species: str, expected_levels: int) -> None:
    assert _DensityProfile._get_num_ionization_levels(species) == expected_levels


def test_get_num_ionization_levels_unrecognized_species() -> None:
    with pytest.raises(ValueError, match="Error getting ionization levels for species"):
        _DensityProfile._get_num_ionization_levels("NotAnElement")


def test_hydrogen_too_small_for_ionization_two() -> None:
    """Hydrogen has one ionization level; ionization=2 must be rejected."""
    with pytest.raises(ValueError, match="Species H has only 1 ionization levels"):
        _build_profile("H", 2)


def test_helium_too_small_for_excessive_ionization() -> None:
    """Helium has two ionization levels; ionization=99 must be rejected."""
    with pytest.raises(ValueError, match="Species He has only 2 ionization levels"):
        _build_profile("He", _IONIZATION_TOO_LARGE)


_PARTICLE_BOUND_NAMES = ("p_rmin", "p_rmax", "p_zmin", "p_zmax")


@pytest.mark.parametrize(
    "minimum,maximum", [("p_zmin", "p_zmax"), ("p_rmin", "p_rmax")]
)
@pytest.mark.parametrize(
    "lower,upper", [(2.0e-6, 1.0e-6), (1.0e-6, 1.0e-6), (np.inf, np.inf)]
)
@pytest.mark.parametrize("species", [None, "He"])
def test_particle_bound_pairs_must_be_strictly_ordered(
    minimum, maximum, lower, upper, species
) -> None:
    with pytest.raises(ValueError, match=f"{minimum} must be less than {maximum}"):
        ExampleDensityProfile(
            **_MINIMAL_PROFILE_KWARGS,
            species=species,
            **{minimum: lower, maximum: upper},
        )


@pytest.mark.parametrize(
    "minimum,maximum", [("p_zmin", "p_zmax"), ("p_rmin", "p_rmax")]
)
def test_particle_bound_pairs_are_validated_on_deserialization(
    minimum, maximum
) -> None:
    payload = ExampleDensityProfile(**_MINIMAL_PROFILE_KWARGS).to_dict()
    payload["parameters"].update({minimum: 2.0e-6, maximum: 1.0e-6})
    with pytest.raises(ValueError, match=f"{minimum} must be less than {maximum}"):
        _DensityProfile.from_dict(payload)


@pytest.mark.parametrize(
    "bounds,edge",
    [
        ({"p_zmin": 9.0e-6}, 8.0e-6),
        ({"p_zmin": _MINIMAL_PROFILE_KWARGS["length"] - 2.0e-6}, 8.0e-6),
        ({"p_zmin": 9.0e-6, "p_zmax": 10.0e-6}, 8.0e-6),
        ({"p_zmax": -3.0e-6}, -2.0e-6),
        ({"p_zmax": -2.0e-6}, -2.0e-6),
        ({"p_zmin": -4.0e-6, "p_zmax": -3.0e-6}, -2.0e-6),
        ({"p_zmin": np.inf}, 8.0e-6),
        ({"p_zmax": -np.inf}, -2.0e-6),
    ],
)
def test_empty_z_extent_warns_and_collapses_to_intrinsic_edge(bounds, edge) -> None:
    profile = ExampleDensityProfile(
        **_MINIMAL_PROFILE_KWARGS, start_position=-2.0e-6, **bounds
    )
    with pytest.warns(RuntimeWarning, match="empty longitudinal extent"):
        extent = profile.get_z_extent()
    assert extent == pytest.approx((edge, edge))
    assert extent[1] - extent[0] == 0.0


def test_z_extent_preserves_positive_width_near_edge(recwarn) -> None:
    base = ExampleDensityProfile(**_MINIMAL_PROFILE_KWARGS, start_position=-2.0e-6)
    upper = base._get_z_extent()[1]
    lower = float(np.nextafter(upper, -np.inf))
    profile = ExampleDensityProfile(
        **_MINIMAL_PROFILE_KWARGS, start_position=-2.0e-6, p_zmin=lower
    )
    assert profile.get_z_extent() == (lower, upper)
    assert upper - lower > 0.0
    assert not recwarn


@pytest.mark.parametrize(
    "p_rmin,p_rmax,expected",
    [(5.0e-6, None, 5.0e-6), (6.0e-6, None, 6.0e-6), (6.0e-6, 8.0e-6, 6.0e-6)],
)
def test_empty_r_extent_warns_and_collapses_to_particle_minimum(
    p_rmin, p_rmax, expected, monkeypatch
) -> None:
    monkeypatch.setattr(ExampleDensityProfile, "_get_r_extent", lambda self: 5.0e-6)
    profile = ExampleDensityProfile(
        **_MINIMAL_PROFILE_KWARGS, p_rmin=p_rmin, p_rmax=p_rmax
    )
    with pytest.warns(RuntimeWarning, match="empty radial extent"):
        extent = profile.get_r_extent()
    assert extent == pytest.approx(expected)
    assert extent - profile.p_rmin == 0.0


def test_radial_minimum_preserves_unknown_upper_extent(recwarn) -> None:
    profile = ExampleDensityProfile(**_MINIMAL_PROFILE_KWARGS, p_rmin=6.0e-6)
    assert profile.get_r_extent() is None
    assert not recwarn


@pytest.mark.parametrize(
    "bounds,method,expected",
    [
        ({"p_zmin": 11.0e-6}, "get_z_extent", (1.0e-5, 1.0e-5)),
        ({"p_rmin": 6.0e-6}, "get_r_extent", 6.0e-6),
    ],
)
def test_modified_profile_warns_for_empty_clipped_extents(
    bounds, method, expected, monkeypatch
) -> None:
    from inversion_fbpic.lib.density_modifiers import ModifiedDensityProfile

    monkeypatch.setattr(ExampleDensityProfile, "_get_r_extent", lambda self: 5.0e-6)
    base = ExampleDensityProfile(**_MINIMAL_PROFILE_KWARGS, **bounds)
    modified = ModifiedDensityProfile(base_density_profile=base)
    with pytest.warns(RuntimeWarning, match="empty"):
        extent = getattr(modified, method)()
    assert extent == pytest.approx(expected)


@pytest.mark.parametrize("name", _PARTICLE_BOUND_NAMES)
@pytest.mark.parametrize("value", [None, "2.5e-6", 1])
def test_particle_bounds_default_and_conversion(name, value) -> None:
    default_profile = ExampleDensityProfile(**_MINIMAL_PROFILE_KWARGS)
    assert getattr(default_profile, name) is None

    profile = ExampleDensityProfile(**_MINIMAL_PROFILE_KWARGS, **{name: value})
    converted = getattr(profile, name)
    if value is None:
        assert converted is None
    else:
        assert isinstance(converted, float)
        assert converted == float(value)


@pytest.mark.parametrize(
    "name,value", [("p_rmin", -1.0), ("p_rmax", -1.0), ("p_rmax", 0.0)]
)
def test_invalid_radial_particle_bounds(name, value) -> None:
    with pytest.raises(ValueError, match=name):
        ExampleDensityProfile(**_MINIMAL_PROFILE_KWARGS, **{name: value})


@pytest.mark.parametrize("name", ["p_rmin", "p_zmin", "p_zmax"])
def test_particle_bounds_accept_zero(name) -> None:
    profile = ExampleDensityProfile(**_MINIMAL_PROFILE_KWARGS, **{name: 0})
    assert getattr(profile, name) == 0.0


@pytest.mark.parametrize("name", ["p_zmin", "p_zmax"])
def test_longitudinal_particle_bounds_accept_negative_positions(name) -> None:
    profile = ExampleDensityProfile(**_MINIMAL_PROFILE_KWARGS, **{name: "-2.5e-6"})
    assert getattr(profile, name) == -2.5e-6


@pytest.mark.parametrize(
    "bounds,expected",
    [
        ({}, (-2.0e-6, 8.0e-6)),
        ({"p_zmin": 0.0}, (0.0, 8.0e-6)),
        ({"p_zmax": 0.0}, (-2.0e-6, 0.0)),
        ({"p_zmin": 1.0e-6, "p_zmax": 4.0e-6}, (1.0e-6, 4.0e-6)),
        ({"p_zmin": -5.0e-6, "p_zmax": 12.0e-6}, (-2.0e-6, 8.0e-6)),
        ({"p_zmin": -2.0e-6, "p_zmax": 8.0e-6}, (-2.0e-6, 8.0e-6)),
        ({"p_zmin": -np.inf, "p_zmax": np.inf}, (-2.0e-6, 8.0e-6)),
    ],
)
def test_z_extent_intersects_particle_bounds(bounds, expected) -> None:
    profile = ExampleDensityProfile(
        **_MINIMAL_PROFILE_KWARGS, start_position=-2.0e-6, **bounds
    )
    assert profile._get_z_extent() == pytest.approx((-2.0e-6, 8.0e-6))
    assert profile.get_z_extent() == pytest.approx(expected)


@pytest.mark.parametrize(
    "p_rmax,expected",
    [
        (None, 5.0e-6),
        (2.0e-6, 2.0e-6),
        (5.0e-6, 5.0e-6),
        (8.0e-6, 5.0e-6),
        (np.inf, 5.0e-6),
    ],
)
def test_r_extent_intersects_particle_bound(p_rmax, expected, monkeypatch) -> None:
    monkeypatch.setattr(ExampleDensityProfile, "_get_r_extent", lambda self: 5.0e-6)
    profile = ExampleDensityProfile(
        **_MINIMAL_PROFILE_KWARGS, p_rmin=1.0e-6, p_rmax=p_rmax
    )
    assert profile._get_r_extent() == pytest.approx(5.0e-6)
    assert profile.get_r_extent() == pytest.approx(expected)


@pytest.mark.parametrize(
    "p_rmax,expected", [(None, None), (2.0e-6, 2.0e-6), (np.inf, np.inf)]
)
def test_r_extent_without_intrinsic_radial_extent(p_rmax, expected) -> None:
    profile = ExampleDensityProfile(**_MINIMAL_PROFILE_KWARGS, p_rmax=p_rmax)
    assert profile._get_r_extent() is None
    assert profile.get_r_extent() == expected


@pytest.mark.parametrize(
    "bounds",
    [
        dict.fromkeys(_PARTICLE_BOUND_NAMES),
        {"p_rmin": 0.0, "p_rmax": 3.0e-6, "p_zmin": -2.0e-6, "p_zmax": 0.0},
    ],
)
def test_particle_bounds_serialization_round_trip(bounds) -> None:
    profile = ExampleDensityProfile(**_MINIMAL_PROFILE_KWARGS, **bounds)
    payload = profile.to_dict()
    for name, value in bounds.items():
        assert payload["parameters"][name] == value

    restored = _DensityProfile.from_dict(payload)
    assert isinstance(restored, ExampleDensityProfile)
    for name, value in bounds.items():
        assert getattr(restored, name) == value
    if bounds.get("p_zmax") == 0.0:
        with pytest.warns(RuntimeWarning, match="empty longitudinal extent"):
            original_extent = profile.get_z_extent()
        with pytest.warns(RuntimeWarning, match="empty longitudinal extent"):
            assert restored.get_z_extent() == original_extent
    else:
        assert restored.get_z_extent() == profile.get_z_extent()


@pytest.mark.parametrize(
    "species,ionization,is_boosted",
    [(None, 1, False), ("H", 0, False), ("He", 1, True), ("He", -1, True)],
)
@pytest.mark.parametrize(
    "bounds,expected",
    [
        pytest.param(
            {},
            {"p_rmin": 0.0, "p_rmax": np.inf, "p_zmin": -np.inf, "p_zmax": np.inf},
            id="defaults",
        ),
        pytest.param(
            dict.fromkeys(_PARTICLE_BOUND_NAMES),
            {"p_rmin": 0.0, "p_rmax": np.inf, "p_zmin": -np.inf, "p_zmax": np.inf},
            id="explicit-none",
        ),
        pytest.param(
            {"p_rmin": 0.0, "p_rmax": np.inf, "p_zmin": -np.inf, "p_zmax": np.inf},
            {"p_rmin": 0.0, "p_rmax": np.inf, "p_zmin": -np.inf, "p_zmax": np.inf},
            id="explicit-infinities",
        ),
        pytest.param(
            {"p_rmin": 1.0e-6, "p_rmax": 3.0e-6, "p_zmin": -2.0e-6, "p_zmax": 7.0e-6},
            {"p_rmin": 1.0e-6, "p_rmax": 3.0e-6, "p_zmin": -2.0e-6, "p_zmax": 7.0e-6},
            id="finite-bounds",
        ),
        pytest.param(
            {"p_rmin": 0.0, "p_zmin": 0.0},
            {"p_rmin": 0.0, "p_rmax": np.inf, "p_zmin": 0.0, "p_zmax": np.inf},
            id="zero-zmin",
        ),
        pytest.param(
            {"p_zmax": 0.0},
            {"p_rmin": 0.0, "p_rmax": np.inf, "p_zmin": -np.inf, "p_zmax": 0.0},
            id="zero-zmax",
        ),
    ],
)
def test_add_to_simulation_passes_particle_bounds(
    species, ionization, is_boosted, bounds, expected
) -> None:
    from scipy.constants import e, m_e

    profile = ExampleDensityProfile(
        **_MINIMAL_PROFILE_KWARGS, species=species, ionization=ionization, **bounds
    )
    simulation = Mock()
    electrons, ions = Mock(), Mock()
    simulation.add_new_species.side_effect = [electrons, ions]

    added_electrons, added_ions = profile.add_to_simulation(simulation, is_boosted)

    assert added_electrons is electrons
    assert added_ions is (None if species is None else ions)
    assert simulation.add_new_species.call_count == (1 if species is None else 2)
    for call in simulation.add_new_species.call_args_list:
        for name, value in expected.items():
            assert call.kwargs[name] == value
        assert call.kwargs["boost_positions_in_dens_func"] is is_boosted
        assert callable(call.kwargs["dens_func"])
        for name in ("p_nz", "p_nr", "p_nt"):
            assert call.kwargs[name] == getattr(profile, name)

    electron_kwargs = simulation.add_new_species.call_args_list[0].kwargs
    assert electron_kwargs["q"] == -e
    assert electron_kwargs["m"] == m_e
    levels = 1 if species is None else profile._get_num_ionization_levels(species)
    assert electron_kwargs["n"] == pytest.approx(
        profile.nominal_density * profile.ionization / levels
    )
    if species is not None:
        ion_kwargs = simulation.add_new_species.call_args_list[1].kwargs
        assert ion_kwargs["q"] == e * profile.ionization
        assert ion_kwargs["n"] == pytest.approx(profile.nominal_density / levels)
        if profile.ionization < levels:
            ions.make_ionizable.assert_called_once_with(
                species, target_species=electrons, level_start=profile.ionization
            )
        else:
            ions.make_ionizable.assert_not_called()
