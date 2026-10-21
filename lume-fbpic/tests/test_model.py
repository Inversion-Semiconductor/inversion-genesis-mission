"""Tests for LUMEFBPICModel."""

from __future__ import annotations

import os

import pytest

from lume.exceptions import ReadOnlyError

from downramp_actions import make_actions
from lume_fbpic.model import LUMEFBPICModel
from short_runs import short_lwfa_model, short_pwfa_model


def test_the_model_has_the_actions_it_was_given(simulator):
    model = LUMEFBPICModel(simulator, make_actions(simulator))
    assert "laser_energy" in model.supported_variables
    assert "downramp_length" in model.supported_variables
    assert "final_particles" in model.supported_variables


def test_dummy_run_set_updates_parameter_without_running(model, simulator):
    model.set({"laser_energy": 6.0})

    assert model.get(["laser_energy"])["laser_energy"] == 6.0
    assert simulator.final_particles is None
    assert not simulator.diagnostics_directory(simulator.working_directory).exists()


def test_set_runs_the_simulation_with_the_new_values(tmp_path):
    model = short_pwfa_model(tmp_path)
    simulator = model.simulator
    assert simulator.final_particles is None

    model.set({"driver_density": 1.0e25})

    assert simulator.driver.density == 1.0e25
    # the run saw the new density: the driver's charge is proportional to it
    assert model.get(["charge_pc"])["charge_pc"] == pytest.approx(
        simulator.driver.charge * 1e12, rel=1e-12
    )


def test_a_failed_run_restores_the_config_and_keeps_the_outputs(tmp_path):
    model = short_pwfa_model(tmp_path / "run")
    model.set({})
    simulator = model.simulator
    config = simulator.config()
    charge = model.get(["charge_pc"])["charge_pc"]
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)  # fbpic cannot write its diagnostics here
    if os.access(locked, os.W_OK):
        pytest.skip("a read-only directory is writable here (running as root)")
    simulator.working_directory = locked

    try:
        with pytest.raises(OSError):
            model.set({"driver_density": 1.0e25})
    finally:
        locked.chmod(0o700)

    assert simulator.config() == config
    assert model.get(["charge_pc"])["charge_pc"] == charge


def test_a_refused_value_undoes_the_values_applied_before_it(model, simulator):
    config = simulator.config()

    with pytest.raises(ValueError, match="downramp_length"):
        model.set({"laser_energy": 6.0, "downramp_length": -1.0})

    assert simulator.config() == config


def test_read_only_variable_cannot_be_set(model):
    with pytest.raises(ReadOnlyError):
        model.set({"charge_pc": 1.0})


def test_set_unknown_variable_raises(model):
    with pytest.raises(ValueError):
        model.set({"not_a_real_variable": 1.0})


def test_final_particles_property_delegates_to_simulator(model, simulator):
    simulator.final_particles = "placeholder"
    assert model.final_particles == "placeholder"


def test_an_empty_set_runs_the_simulator(tmp_path):
    model = short_pwfa_model(tmp_path)

    model.set({})

    assert model.simulator.final_particles is not None


def test_a_set_with_an_unchanged_config_runs_again(tmp_path):
    # inversion_fbpic skips a run whose config hash matches the last one in the directory, by
    # default; a set() must run all the same
    model = short_lwfa_model(tmp_path)
    simulator = model.simulator
    model.set({})
    last_dump = sorted(simulator.diagnostics_directory(tmp_path).glob("data*.h5"))[-1]
    written = last_dump.stat().st_mtime_ns

    model.set({})

    assert last_dump.stat().st_mtime_ns > written


def test_an_action_of_another_kind_of_simulator_is_refused(simulator):
    from lume_fbpic.actions import PlasmaFieldAction

    with pytest.raises(TypeError, match="needs a PWFASimulator"):
        LUMEFBPICModel(
            simulator,
            [PlasmaFieldAction(name="plasma_density", field_name="nominal_density")],
        )
