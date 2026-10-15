"""Tests for the units an action gets from the documentation of the config field it acts on."""

from __future__ import annotations

import re

import pytest

from lume_fbpic import archive
from lume_fbpic.actions import (
    DensityFieldAction,
    DopantFractionAction,
    HyperparameterFieldAction,
    LaserFieldAction,
    StatAction,
)
from lume_fbpic.model import LUMEFBPICModel
from lume_fbpic.units import UNIT_TAG, is_unit

# Bracketed tags in config docstrings that are not units, and why. Type hints, lists of allowed
# values and ranges are recognized by their form (see `_is_not_a_unit()`); these are the others.
_NOT_UNITS = {
    "waists": "LasyLaserPulse.hi_range: a size in laser waists, relative, not a physical unit",
}


def test_a_density_action_takes_the_unit_of_its_field(model):
    assert _unit(model, "flattop_density") == "m^-3"
    assert _unit(model, "downramp_density") == "m^-3"
    assert _unit(model, "downramp_length") == "m"


def test_a_dopant_fraction_is_dimensionless(simulator):
    action = DopantFractionAction(
        name="fraction", host_index=0, dopant_index=1, read_only=False
    )

    model = LUMEFBPICModel(simulator, [action], dummy_run=True)

    assert _unit(model, "fraction") == "1"


@pytest.mark.parametrize(
    "action_class, kwargs",
    [(DensityFieldAction, {"density_index": 0, "field_name": "no_such_field"})],
)
def test_a_field_that_does_not_exist_has_no_unit(simulator, action_class, kwargs):
    model = LUMEFBPICModel(
        simulator, [action_class(name="x", **kwargs)], dummy_run=True
    )

    assert _unit(model, "x") is None


def test_a_field_without_a_unit_tag_has_none(simulator):
    model = LUMEFBPICModel(
        simulator, [LaserFieldAction(name="a0", field_name="a0")], dummy_run=True
    )

    assert _unit(model, "a0") is None


def test_a_hyperparameter_action_takes_the_unit_of_its_field(simulator):
    model = LUMEFBPICModel(
        simulator,
        [
            HyperparameterFieldAction(name="start", field_name="zmin"),
            HyperparameterFieldAction(name="cells", field_name="nz"),
        ],
        dummy_run=True,
    )

    assert _unit(model, "start") == "m"
    assert _unit(model, "cells") is None  # a count: its tag is a type, `[int]`


def test_a_laser_action_takes_the_unit_of_its_field(model):
    assert _unit(model, "laser_energy") == "J"
    assert _unit(model, "laser_focal_position") == "m"
    assert _unit(model, "laser_temporal_width") == "s"


def test_a_pwfa_action_takes_the_unit_of_its_field():
    import pwfa_gaussian

    model = pwfa_gaussian.build_model(dummy_run=True)

    assert _unit(model, "plasma_density") == "m^-3"
    assert _unit(model, "driver_charge") == "C"
    assert _unit(model, "driver_sigma_r") == "m"
    assert _unit(model, "witness_position") == "m"
    assert _unit(model, "driver_gamma") is None


def test_a_stat_action_takes_the_unit_of_its_statistic(model):
    assert _unit(model, "charge_pc") == "pC"
    assert _unit(model, "energy_mean_mev") == "MeV"
    assert _unit(model, "energy_std_mev") == "MeV"


def test_a_unit_given_to_an_action_is_kept(simulator):
    model = LUMEFBPICModel(
        simulator,
        [
            LaserFieldAction(name="energy", field_name="energy", unit="mJ"),
            LaserFieldAction(name="focus", field_name="focal_position", unit=None),
            StatAction(name="charge", stat_name="charge_pc", unit="C", read_only=True),
        ],
        dummy_run=True,
    )

    assert _unit(model, "energy") == "mJ"
    assert _unit(model, "focus") is None
    assert _unit(model, "charge") == "C"


def test_a_zernike_coefficient_is_in_radians():
    import ionization_injection

    model = ionization_injection.build_model()

    assert _unit(model, "zernike_coma_x") == "rad"
    assert _unit(model, "nitrogen_dopant_fraction") == "1"


def test_an_explicit_none_unit_survives_an_archive(simulator, tmp_path):
    model = LUMEFBPICModel(
        simulator,
        [
            LaserFieldAction(name="energy", field_name="energy", unit=None),
            LaserFieldAction(name="focus", field_name="focal_position"),
            StatAction(name="charge", stat_name="charge_pc", unit=None, read_only=True),
        ],
        dummy_run=True,
    )
    assert _unit(model, "energy") is None and _unit(model, "focus") == "m"
    archive.save(model, tmp_path / "a.h5")

    loaded = archive.load(tmp_path / "a.h5")

    assert _unit(loaded, "energy") is None  # given as None: not replaced by the default
    assert _unit(loaded, "charge") is None
    assert _unit(loaded, "focus") == "m"  # left unset: the default, as before


def test_every_unit_tag_in_a_config_docstring_is_a_unit_beamphysics_knows():
    import inversion_fbpic.lib  # noqa: F401  (registers its config classes)
    import lume_fbpic.density_profiles  # noqa: F401
    import lume_fbpic.pwfa_config  # noqa: F401
    from inversion_fbpic.lib.serializable_config import SerializableConfig

    unknown = _unknown_unit_tags(_subclasses(SerializableConfig))

    assert not unknown, (
        f"unit tags that beamphysics does not know, so their fields get no unit: {unknown}; "
        "write a symbol it knows (for example µm, not um), or list the tag in "
        "_NOT_UNITS if it is not a unit"
    )


def test_the_unit_tag_guard_catches_a_unit_beamphysics_does_not_know():
    class Config:
        @classmethod
        def _parameter_descriptions(cls):
            return {
                "size": ["(float) [um] A size."],
                "name": ["(str) [str] A name."],
                "mode": ['(str) ["a", "b"] A mode.'],
                "length": ["(float) [m] A length."],
            }

    assert _unknown_unit_tags([Config]) == [("Config", "size", "um")]


def test_the_derived_units_are_valid_and_survive_an_archive(model, tmp_path):
    from beamphysics.units import pmd_unit

    archive.save(model, tmp_path / "a.h5")
    loaded = archive.load(tmp_path / "a.h5")

    for name, variable in model.supported_variables.items():
        unit = getattr(variable, "unit", None)
        assert _unit(loaded, name) == unit
        if unit is not None:
            pmd_unit(unit)  # raises for an unknown unit


def _unit(model: LUMEFBPICModel, name: str) -> str | None:
    return getattr(model.supported_variables[name], "unit", None)


def _is_not_a_unit(tag: str) -> bool:
    """Whether a bracketed tag is plainly not a unit: a list of allowed values, a range or a
    type hint (it holds a quote, a comma, a `|` or a bracket), a Python type, a class name, or one
    of `_NOT_UNITS`."""
    return (
        tag in _NOT_UNITS
        or re.search(r"[\"',|\[]", tag) is not None
        or tag in ("bool", "dict", "float", "int", "list", "str", "tuple")
        or re.fullmatch(r"[A-Z][A-Za-z]+", tag) is not None
    )


def _subclasses(cls: type) -> list[type]:
    found = []
    for subclass in cls.__subclasses__():
        found += [subclass, *_subclasses(subclass)]
    return found


def _unknown_unit_tags(classes) -> list[tuple[str, str, str]]:
    """The (class, field, tag) of each bracketed tag in the field documentation of `classes`
    that is neither a unit `beamphysics` knows nor plainly something else."""
    found = set()
    for cls in classes:
        for field, text in cls._parameter_descriptions().items():
            for tag in UNIT_TAG.findall(" ".join(text)):
                if not is_unit(tag) and not _is_not_a_unit(tag):
                    found.add((cls.__name__, field, tag))
    return sorted(found)


def test_a_list_of_actions_can_serve_several_models(simulator):
    actions = [LaserFieldAction(name="focus", field_name="focal_position")]

    first = LUMEFBPICModel(simulator, actions, dummy_run=True)

    assert actions[0].unit is None  # the action given is left as it is
    assert _unit(first, "focus") == "m"
    assert _unit(LUMEFBPICModel(simulator, actions, dummy_run=True), "focus") == "m"
