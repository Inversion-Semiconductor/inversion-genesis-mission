"""Tests for lume_fbpic action/variable classes, through the model's `get()` and `set()`."""

from __future__ import annotations

import math

import pytest

from conftest import LASER_KWARGS
from downramp_actions import make_actions
from inversion_fbpic.lib.laser import GaussianLaserPulse
from lume_fbpic.actions import HyperparameterFieldAction, LaserFieldAction
from lume_fbpic.model import LUMEFBPICModel


@pytest.fixture()
def model(simulator) -> LUMEFBPICModel:
    """The downramp action set plus a laser `a0` and a hyperparameter action."""
    return LUMEFBPICModel(
        simulator,
        [
            *make_actions(simulator),
            LaserFieldAction(name="laser_a0", field_name="a0"),
            HyperparameterFieldAction(name="nz", field_name="nz"),
        ],
        dummy_run=True,
    )


def _get(model, name):
    return model.get([name])[name]


def test_a_laser_field_reads_the_laser(model):
    assert _get(model, "laser_energy") == model.simulator.laser.energy


def test_setting_a_laser_field_replaces_the_laser_and_leaves_the_old_one(model):
    original = model.simulator.laser

    model.set({"laser_energy": 7.5})

    assert model.simulator.laser.energy == 7.5
    assert _get(model, "laser_energy") == 7.5
    assert original.energy == 5.0


def test_setting_another_laser_field_keeps_the_energy_and_derives_a0_again(model):
    # Every laser holds both energy and a0, one given and one derived; the derived one must be
    # dropped before the laser is rebuilt, or the rebuild sees both.
    model.set({"laser_focal_position": 20.0e-6})

    expected = GaussianLaserPulse(
        energy=5.0, **{**LASER_KWARGS, "focal_position": 20.0e-6}
    )
    assert model.simulator.laser.focal_position == 20.0e-6
    assert model.simulator.laser.energy == 5.0
    assert model.simulator.laser.a0 == expected.a0


def test_after_a0_is_set_another_laser_field_keeps_a0_and_derives_the_energy(model):
    model.set({"laser_a0": 3.0})
    model.set({"laser_focal_position": 20.0e-6})

    expected = GaussianLaserPulse(a0=3.0, **{**LASER_KWARGS, "focal_position": 20.0e-6})
    assert model.simulator.laser.a0 == 3.0
    assert model.simulator.laser.focal_position == 20.0e-6
    assert model.simulator.laser.energy == expected.energy


def test_a_hyperparameter_field_reads_and_sets_the_hyperparameters(model):
    assert _get(model, "nz") == model.simulator.hyparams.nz

    model.set({"nz": 16})

    assert model.simulator.hyparams.nz == 16


def test_a_density_field_sets_only_the_profile_it_indexes(model):
    other = model.simulator.densities[0]

    model.set({"downramp_length": 0.25e-3})

    assert model.simulator.densities[1].downramp_length == 0.25e-3
    assert model.simulator.densities[0] == other


def test_a_stat_reads_the_simulators_stats(model):
    model.simulator.stats = {"charge_pc": 12.3}

    assert _get(model, "charge_pc") == 12.3


def test_a_stat_is_nan_before_any_run(model):
    assert math.isnan(_get(model, "energy_mean_mev"))


def test_the_final_particles_are_the_simulators(model, particle_group):
    model.simulator.final_particles = particle_group

    assert _get(model, "final_particles") is particle_group
