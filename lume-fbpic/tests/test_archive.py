"""Tests for the archive bundle: final particles flag, action definitions, and the action
input values at execution / output values after it."""

from __future__ import annotations

import importlib.metadata
import math
import warnings

import h5py
import numpy
import pytest

from archive_helpers import (
    CustomLaserAction,
    load_simulator,
    recorded_values,
    simulator_model,
)
from downramp_actions import make_actions
from reference_descriptor import reference_descriptor
from short_runs import short_pwfa_model
from lume_fbpic import archive, descriptor
from lume_fbpic.actions import LaserFieldAction, make_descriptor_actions
from lume_fbpic.model import LUMEFBPICModel


def _archive_with_outputs(model, path, outputs):
    """Archive `model` as a run whose recorded `outputs` (by action name) it never held."""
    model.simulator.recorded_outputs = dict(outputs)
    try:
        archive.save(model, path)
    finally:
        model.simulator.recorded_outputs = {}


@pytest.fixture()
def full_model(simulator) -> LUMEFBPICModel:
    """A dummy-run model with the default actions plus the 33 descriptor actions."""
    actions = [*make_actions(simulator), *make_descriptor_actions()]
    return LUMEFBPICModel(simulator, actions, dummy_run=True)


def test_archive_does_not_store_particles_by_default(
    simulator, particle_group, tmp_path
):
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 1.0}

    archive.save(simulator_model(simulator), tmp_path / "a.h5")

    with h5py.File(tmp_path / "a.h5") as f:
        assert "final_particles" not in f
        assert "stats" not in f
    restored = load_simulator(tmp_path / "a.h5")
    assert restored.final_particles is None
    assert restored.stats == {}


def test_archive_stores_final_particles_as_a_particle_group(
    simulator, particle_group, tmp_path
):
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 1.5, "energy_mean_mev": 2.5}

    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", save_final_particles=True
    )

    restored = load_simulator(tmp_path / "a.h5")
    assert len(restored.final_particles) == len(particle_group)
    numpy.testing.assert_allclose(restored.final_particles.pz, particle_group.pz)
    numpy.testing.assert_allclose(
        restored.final_particles.weight, particle_group.weight
    )
    assert restored.stats == {"charge_pc": 1.5, "energy_mean_mev": 2.5}


def test_archive_with_particles_opens_directly_as_a_particle_file(
    simulator, particle_group, tmp_path
):
    from beamphysics import ParticleGroup

    simulator.final_particles = particle_group
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", save_final_particles=True
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a non-compliant openPMD layout warns
        opened = ParticleGroup(h5=str(tmp_path / "a.h5"))

    assert len(opened) == len(particle_group)
    numpy.testing.assert_allclose(opened.weight, particle_group.weight)


def test_saving_particles_before_any_run_is_an_error(simulator, tmp_path):
    with pytest.raises(ValueError, match="final particles"):
        archive.save(
            simulator_model(simulator), tmp_path / "a.h5", save_final_particles=True
        )


def test_model_archive_stores_the_action_definitions_in_order(full_model, tmp_path):
    archive.save(full_model, tmp_path / "m.h5")

    restored = archive.load(tmp_path / "m.h5", dummy_run=True)

    assert list(restored.supported_variables) == list(full_model.supported_variables)
    assert [type(a) for a in restored.supported_variables.values()] == [
        type(a) for a in full_model.supported_variables.values()
    ]
    assert [a.model_dump() for a in restored.supported_variables.values()] == [
        a.model_dump() for a in full_model.supported_variables.values()
    ]


def test_restored_model_sees_the_archived_config(full_model, tmp_path):
    full_model.set({"laser_energy": 6.0})
    archive.save(full_model, tmp_path / "m.h5")

    restored = archive.load(tmp_path / "m.h5", dummy_run=True)

    assert restored.get(["laser_energy"])["laser_energy"] == 6.0


def test_restored_model_returns_outputs_when_particles_were_saved(
    full_model, simulator, particle_group, tmp_path
):
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 7.0, "energy_mean_mev": 8.0, "energy_std_mev": 0.5}
    names = ["charge_pc", "descriptor_mean_uz", "descriptor_total_beam_charge_pc"]
    expected = full_model.get(names)
    archive.save(full_model, tmp_path / "m.h5", save_final_particles=True)

    restored = archive.load(tmp_path / "m.h5", dummy_run=True)

    got = restored.get(names)
    for name in names:
        assert got[name] == pytest.approx(expected[name])


# beamphysics (0.16.1 on) reads particles from a group of a file, with this warning; before
# 0.16.1 it could not, the reason for the minimum version
@pytest.mark.filterwarnings("ignore:The openPMD root is the group:FutureWarning")
def test_an_archive_in_a_group_of_a_file_loads_its_final_particles(
    full_model, particle_group, tmp_path
):
    simulator = full_model.simulator
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 7.0, "energy_mean_mev": 8.0, "energy_std_mev": 0.5}

    with h5py.File(tmp_path / "f.h5", "w") as f:
        archive.save(full_model, f.create_group("sub"), save_final_particles=True)
    with h5py.File(tmp_path / "f.h5", "r") as f:
        restored = archive.load(f["sub"], dummy_run=True)

    numpy.testing.assert_array_equal(
        restored.simulator.final_particles.z, particle_group.z
    )
    numpy.testing.assert_array_equal(
        restored.simulator.final_particles.weight, particle_group.weight
    )
    assert restored.simulator.stats == simulator.stats


def test_archive_stores_the_current_input_and_output_values(tmp_path):
    model = short_pwfa_model(tmp_path / "run")
    model.set({"driver_density": 6.0e24})  # runs
    stats = dict(model.simulator.stats)
    model.dummy_run = True
    model.set(
        {"driver_density": 9.0e24}
    )  # changes the config after the run, without running
    archive.save(model, tmp_path / "m.h5")

    values = recorded_values(tmp_path / "m.h5")

    assert values["charge_pc"] == stats["charge_pc"]
    assert values["energy_mean_mev"] == stats["energy_mean_mev"]
    assert (
        "driver_density" not in values
    )  # the inputs are in the config, not stored as values
    # the config holds the current input, not the 6e24 that ran
    assert (
        archive.load(tmp_path / "m.h5", dummy_run=True).get(["driver_density"])[
            "driver_density"
        ]
        == 9.0e24
    )


def test_model_that_never_ran_stores_its_inputs_in_the_config_and_nan_outputs(
    full_model, tmp_path
):
    full_model.set({"laser_energy": 4.0})
    archive.save(full_model, tmp_path / "m.h5")

    values = recorded_values(tmp_path / "m.h5")

    assert (
        archive.load(tmp_path / "m.h5", dummy_run=True).get(["laser_energy"])[
            "laser_energy"
        ]
        == 4.0
    )
    assert math.isnan(values["charge_pc"])
    assert "final_particles" not in values  # no scalar value for particles


def test_the_default_name_depends_on_the_actions_as_well_as_the_config(
    simulator, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    actions = make_actions(simulator)

    same = [
        archive.save(LUMEFBPICModel(simulator, make_actions(simulator), dummy_run=True))
        for _ in range(2)
    ]
    fewer = archive.save(LUMEFBPICModel(simulator, actions[:-1], dummy_run=True))
    renamed = archive.save(
        LUMEFBPICModel(
            simulator,
            [*actions[:-1], actions[-1].model_copy(update={"name": "renamed"})],
            dummy_run=True,
        )
    )

    assert same[0] == same[1]
    assert len({same[0], fewer, renamed}) == 3
    assert all(name.startswith("lume_fbpic_") for name in (fewer, renamed))
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        {same[0], fewer, renamed}
    )


def test_load_rejects_a_file_without_actions(simulator, tmp_path):
    archive.save(simulator_model(simulator), tmp_path / "s.h5")
    with h5py.File(tmp_path / "s.h5", "a") as f:
        del f["actions"]

    with pytest.raises(ValueError, match="no actions"):
        archive.load(tmp_path / "s.h5")


def test_load_rejects_another_format_version(full_model, tmp_path):
    archive.save(full_model, tmp_path / "m.h5")
    with h5py.File(tmp_path / "m.h5", "a") as f:
        assert f.attrs["format_version"] == archive._FORMAT_VERSION
        f.attrs["format_version"] = 1

    with pytest.raises(ValueError, match="format version 1"):
        archive.load(tmp_path / "m.h5")


def test_an_action_stores_the_module_name_and_package_version_of_its_class(
    full_model, tmp_path
):
    archive.save(full_model, tmp_path / "m.h5")

    with h5py.File(tmp_path / "m.h5") as f:
        entry = f["actions/0000"]
        assert entry.attrs["module"] == "lume_fbpic.actions"
        first = next(iter(full_model.supported_variables.values()))
        assert entry.attrs["class"] == type(first).__qualname__
        assert entry.attrs["version"] == importlib.metadata.version("lume-fbpic")


def test_a_custom_action_class_round_trips(simulator, tmp_path):
    model = LUMEFBPICModel(
        simulator,
        [CustomLaserAction(name="mine", field_name="waist", unit="m")],
        dummy_run=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        archive.save(model, tmp_path / "m.h5")
        restored = archive.load(tmp_path / "m.h5")

    action = restored.supported_variables["mine"]
    assert type(action) is CustomLaserAction
    assert restored.get(["mine"]) == model.get(["mine"])


def test_another_package_version_is_not_a_warning(simulator, tmp_path):
    model = LUMEFBPICModel(
        simulator,
        [CustomLaserAction(name="mine", field_name="waist", unit="m")],
        dummy_run=True,
    )
    archive.save(model, tmp_path / "m.h5")
    with h5py.File(tmp_path / "m.h5", "a") as f:
        f["actions/0000"].attrs["version"] = "0.0.0"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "mine" in archive.load(tmp_path / "m.h5").supported_variables


@pytest.mark.parametrize(
    "attribute, value",
    [("module", "no_such_package.actions"), ("class", "NoSuchAction")],
)
def test_an_action_whose_class_is_gone_is_left_out_with_a_warning(
    full_model, tmp_path, attribute, value
):
    archive.save(full_model, tmp_path / "m.h5")
    with h5py.File(tmp_path / "m.h5", "a") as f:
        f["actions/0000"].attrs[attribute] = value
        name = str(f["actions/0000"].attrs["name"])

    with pytest.warns(UserWarning, match=f"action {name!r} .*cannot be rebuilt"):
        restored = archive.load(tmp_path / "m.h5")

    assert name not in restored.supported_variables
    assert len(restored.supported_variables) == len(full_model.supported_variables) - 1


def test_the_warning_names_the_archived_version(full_model, tmp_path):
    archive.save(full_model, tmp_path / "m.h5")
    with h5py.File(tmp_path / "m.h5", "a") as f:
        f["actions/0000"].attrs["class"] = "NoSuchAction"
        f["actions/0000"].attrs["version"] = "9.9"

    with pytest.warns(UserWarning, match=r"archived with version 9\.9"):
        archive.load(tmp_path / "m.h5")


def test_an_action_that_no_longer_accepts_its_parameters_is_left_out(
    full_model, tmp_path
):
    archive.save(full_model, tmp_path / "m.h5")
    with h5py.File(tmp_path / "m.h5", "a") as f:
        f["actions/0000/parameters"].attrs["read_only"] = "maybe"  # not a bool
        name = str(f["actions/0000"].attrs["name"])

    with pytest.warns(UserWarning, match="cannot be rebuilt"):
        restored = archive.load(tmp_path / "m.h5")

    assert name not in restored.supported_variables


def test_strict_load_raises_for_an_action_that_cannot_be_rebuilt(full_model, tmp_path):
    archive.save(full_model, tmp_path / "m.h5")
    with h5py.File(tmp_path / "m.h5", "a") as f:
        f["actions/0000"].attrs["class"] = "NoSuchAction"

    with pytest.raises(ValueError, match="NoSuchAction"):
        archive.load(tmp_path / "m.h5", strict=True)


def test_an_action_class_defined_in_a_function_is_saved_with_a_warning(
    simulator, tmp_path
):
    class LocalLaserAction(LaserFieldAction):
        pass

    model = LUMEFBPICModel(
        simulator,
        [LocalLaserAction(name="mine", field_name="waist", unit="m")],
        dummy_run=True,
    )

    with pytest.warns(
        UserWarning, match="LocalLaserAction of action 'mine' will not load"
    ):
        archive.save(model, tmp_path / "m.h5")
    with pytest.warns(UserWarning, match="action 'mine' .*cannot be rebuilt"):
        restored = archive.load(tmp_path / "m.h5")
    assert "mine" not in restored.supported_variables


def test_action_parameters_are_stored_as_attributes_not_json(full_model, tmp_path):
    archive.save(full_model, tmp_path / "m.h5")

    with h5py.File(tmp_path / "m.h5") as f:
        entry = f["actions/0000"]
        parameters = entry["parameters"].attrs
        assert entry.attrs["name"] == next(iter(full_model.supported_variables))
        assert parameters["name"] == entry.attrs["name"]
        assert not parameters["read_only"]  # h5py returns numpy.bool_
        assert "variable_class" not in parameters and "default_value" not in parameters
        assert "parameters" not in entry.attrs  # no JSON blob
        assert len(f["actions"]) == len(full_model.supported_variables)


def test_action_parameters_round_trip_with_their_types(
    simulator, particle_group, tmp_path
):
    from lume_fbpic.actions import make_descriptor_actions

    model = LUMEFBPICModel(
        simulator,
        [*make_actions(simulator), *make_descriptor_actions(uz_min=12.5)],
        dummy_run=True,
    )
    archive.save(model, tmp_path / "m.h5")

    restored = archive.load(tmp_path / "m.h5", dummy_run=True)

    assert [a.model_dump() for a in restored.supported_variables.values()] == [
        a.model_dump() for a in model.supported_variables.values()
    ]
    descriptor = restored.supported_variables["descriptor_mean_uz"]
    assert descriptor.uz_min == 12.5 and isinstance(descriptor.longitudinal_bins, int)


def test_an_action_with_an_unstorable_parameter_cannot_be_archived(simulator, tmp_path):
    class ListAction(LaserFieldAction):
        tags: list[str] = []

    model = LUMEFBPICModel(
        simulator,
        [ListAction(name="x", field_name="waist", unit="m", tags=["a"])],
        dummy_run=True,
    )

    with pytest.raises(TypeError, match="tags"):
        archive.save(model, tmp_path / "m.h5")


def test_a_failed_archive_leaves_no_file_and_an_existing_one_as_it_was(
    simulator, tmp_path, mocker
):
    class ListAction(LaserFieldAction):
        tags: list[str] = []

    bad = LUMEFBPICModel(
        simulator,
        [ListAction(name="x", field_name="waist", unit="m", tags=["a"])],
        dummy_run=True,
    )
    path = tmp_path / "m.h5"

    with pytest.raises(TypeError):
        archive.save(bad, path)
    assert not path.exists()  # nothing was written

    path.write_bytes(b"an earlier archive")
    with pytest.raises(TypeError):
        archive.save(bad, path)
    assert path.read_bytes() == b"an earlier archive"

    good = LUMEFBPICModel(simulator, make_actions(simulator), dummy_run=True)
    mocker.patch(
        "lume_fbpic.archive._write_actions", side_effect=RuntimeError("midway")
    )
    with pytest.raises(
        RuntimeError, match="midway"
    ):  # an error after the config is written
        archive.save(good, path)
    assert path.read_bytes() == b"an earlier archive"
    assert [p.name for p in tmp_path.iterdir()] == ["m.h5"]  # no partial file either


def test_the_simulators_recorded_outputs_are_archived(full_model, tmp_path):
    full_model.set({"laser_energy": 4.0})

    _archive_with_outputs(
        full_model,
        tmp_path / "m.h5",
        {"descriptor_mean_uz": 171.6, "descriptor_total_beam_charge_pc": 510.0},
    )

    values = recorded_values(tmp_path / "m.h5")
    assert values["descriptor_mean_uz"] == 171.6
    assert values["descriptor_total_beam_charge_pc"] == 510.0
    assert math.isnan(values["descriptor_cov_uz_uz"])  # not recorded


@pytest.fixture()
def recorded_model(full_model, tmp_path) -> LUMEFBPICModel:
    """A model loaded from an archive that has recorded outputs but no particles."""
    _archive_with_outputs(
        full_model,
        tmp_path / "m.h5",
        {"descriptor_mean_uz": 171.6, "descriptor_total_beam_charge_pc": 510.0},
    )
    return archive.load(tmp_path / "m.h5", dummy_run=True)


def test_loaded_model_serves_the_recorded_outputs_when_there_are_no_particles(
    recorded_model,
):
    got = recorded_model.get(["descriptor_mean_uz", "descriptor_total_beam_charge_pc"])

    assert got == {
        "descriptor_mean_uz": 171.6,
        "descriptor_total_beam_charge_pc": 510.0,
    }
    assert math.isnan(
        recorded_model.get(["descriptor_cov_uz_uz"])["descriptor_cov_uz_uz"]
    )


def test_recorded_outputs_stay_when_an_input_is_set(recorded_model):
    recorded_model.set({"laser_energy": 3.0})

    assert recorded_model.get(["laser_energy"])["laser_energy"] == 3.0
    assert recorded_model.get(["descriptor_mean_uz"])["descriptor_mean_uz"] == 171.6


def test_live_values_win_over_recorded_ones(recorded_model, particle_group):
    recorded_model.simulator.final_particles = particle_group

    live = recorded_model.get(["descriptor_mean_uz"])["descriptor_mean_uz"]

    assert (
        live
        == reference_descriptor(
            particle_group,
            uz_min=descriptor.UZ_MIN,
            central_fraction=descriptor.CENTRAL_FRACTION,
        )["mean_uz"]
    )


def test_recorded_outputs_survive_archiving_again(recorded_model, tmp_path):
    archive.save(recorded_model, tmp_path / "again.h5")

    assert recorded_values(tmp_path / "again.h5")["descriptor_mean_uz"] == 171.6


def test_reset_restores_the_initial_config_without_running(recorded_model, tmp_path):
    recorded_model.simulator.working_directory = tmp_path / "work"
    energy = recorded_model.get(["laser_energy"])["laser_energy"]
    recorded_model.set({"laser_energy": 9.0})
    assert recorded_model.get(["laser_energy"])["laser_energy"] == 9.0

    recorded_model.reset()  # ActionModel.reset() would fail here: default_value is None

    assert recorded_model.get(["laser_energy"])["laser_energy"] == energy
    assert not (tmp_path / "work").exists()  # nothing ran


def test_reset_brings_back_the_recorded_outputs(recorded_model):
    recorded_model.set({"laser_energy": 9.0})
    recorded_model.simulator.recorded_outputs = {"descriptor_mean_uz": 1.0}
    assert recorded_model.get(["descriptor_mean_uz"])["descriptor_mean_uz"] == 1.0

    recorded_model.reset()

    assert recorded_model.get(["descriptor_mean_uz"])["descriptor_mean_uz"] == 171.6


def test_load_reads_only_the_simulators_config_groups(full_model, tmp_path):
    archive.save(full_model, tmp_path / "m.h5")
    with h5py.File(tmp_path / "m.h5", "a") as f:
        f.create_group("not_config")

    restored = archive.load(tmp_path / "m.h5", dummy_run=True)

    assert list(restored.simulator.config()) == list(full_model.simulator.config())


def test_a_config_named_like_an_archive_group_cannot_be_archived(
    full_model, tmp_path, mocker
):
    simulator = full_model.simulator
    mocker.patch.object(simulator, "config", return_value={"stats": simulator.laser})

    with pytest.raises(ValueError, match="name of an archive group"):
        archive.save(full_model, tmp_path / "m.h5")


def test_the_model_archives_itself_and_is_rebuilt_from_its_archive(
    full_model, tmp_path
):
    path = full_model.archive(tmp_path / "m.h5")

    restored = LUMEFBPICModel.from_archive(path, dummy_run=True)

    assert list(restored.supported_variables) == list(full_model.supported_variables)
    assert restored.get(["laser_energy"]) == full_model.get(["laser_energy"])
