"""Tests for LWFASimulator's config/lifecycle/archive paths -- no real fbpic run."""

from __future__ import annotations

import json

import attrs

import h5py
import numpy
import pytest

from inversion_fbpic.lib.datapoint import Parameters
from archive_helpers import load_simulator, simulator_model
from lume_fbpic import archive
from lume_fbpic.actions import make_descriptor_actions
from lume_fbpic.lwfa_simulator import LWFASimulator
from lume_fbpic.model import LUMEFBPICModel
from short_runs import short_lwfa_simulator, short_pwfa_simulator


def test_configure_accepts_complete_config(simulator):
    simulator.configure()  # does not raise


def test_configure_rejects_missing_densities(hyparams, laser, tmp_path):
    simulator = LWFASimulator(
        hyparams,
        laser,
        [],
        target_species="electrons_flattop",
        working_directory=tmp_path,
    )
    with pytest.raises(ValueError):
        simulator.configure()


def test_run_checks_a_config_that_changed_after_it_was_configured(simulator):
    simulator.configure()
    simulator.densities = []

    with pytest.raises(ValueError, match="densities is required"):
        simulator.run()

    assert not simulator.diagnostics_directory(simulator.working_directory).exists()


def test_run_with_an_incomplete_config_raises_and_runs_nothing(simulator):
    simulator.densities = []

    with pytest.raises(ValueError, match="densities is required"):
        simulator.run()

    assert not simulator.diagnostics_directory(simulator.working_directory).exists()


def test_reset_restores_the_starting_state_without_rerunning(simulator):
    config = simulator.config()
    simulator.configure()
    simulator.stats = {"charge_pc": 1.0}
    simulator.final_particles = object()
    simulator.laser = None

    simulator.reset()

    assert simulator.stats == {}
    assert simulator.final_particles is None
    assert simulator.laser is config["laser"]
    assert simulator.densities == config["densities"]


def test_reset_restores_the_archived_state_of_a_loaded_simulator(
    simulator, particle_group, tmp_path
):
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 7.0}
    archive.save(
        simulator_model(simulator), tmp_path / "a.h5", save_final_particles=True
    )
    restored = load_simulator(tmp_path / "a.h5")
    laser = restored.laser

    restored.laser = None
    restored.stats = {}
    restored.final_particles = None
    restored.reset()

    assert restored.laser is laser
    assert restored.stats == {"charge_pc": 7.0}
    assert len(restored.final_particles) == len(particle_group)


def test_an_archive_with_particles_keeps_its_own_stats_not_the_recorded_outputs(
    simulator, particle_group, tmp_path
):
    simulator.final_particles = particle_group
    simulator.stats = {"charge_pc": 7.0}
    model = LUMEFBPICModel(simulator, make_descriptor_actions(), dummy_run=True)
    archive.save(model, tmp_path / "a.h5", save_final_particles=True)

    assert load_simulator(tmp_path / "a.h5").stats == {"charge_pc": 7.0}


def test_reset_needs_a_starting_state(simulator):
    simulator._initial_state = None

    with pytest.raises(RuntimeError, match="starting state"):
        simulator.reset()


def test_the_default_archive_name_comes_from_the_config(
    simulator, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    written = archive.save(simulator_model(simulator))
    simulator.hyparams = attrs.evolve(simulator.hyparams, nz=simulator.hyparams.nz + 1)

    assert written.startswith("lume_fbpic_") and written.endswith(".h5")
    assert (tmp_path / written).is_file()
    assert archive.save(simulator_model(simulator)) != written


def test_parameters_go_into_the_archive_and_not_into_the_fingerprint(
    simulator, tmp_path
):
    fingerprint = archive.fingerprint(simulator)
    simulator.parameters = Parameters(data={"run": "sim_0003", "inputs": {"a": 1.5}})

    archive.save(simulator_model(simulator), tmp_path / "a.h5")
    loaded = load_simulator(tmp_path / "a.h5")
    archive.save(simulator_model(loaded), tmp_path / "b.h5")
    again = load_simulator(tmp_path / "b.h5")

    assert archive.fingerprint(simulator) == fingerprint
    assert again.parameters.data == {"run": "sim_0003", "inputs": {"a": 1.5}}
    again.reset()
    assert again.parameters.data["run"] == "sim_0003"


def test_an_archive_without_parameters_loads_with_none(simulator, tmp_path):
    archive.save(simulator_model(simulator), tmp_path / "a.h5")

    assert load_simulator(tmp_path / "a.h5").parameters is None


def test_the_fingerprint_refuses_a_value_json_does_not_know():
    with pytest.raises(TypeError, match="no fingerprint for object"):
        json.dumps(object(), default=archive._json_value)


def test_the_fingerprint_sees_all_of_a_large_array():
    one, other = numpy.zeros(5000), numpy.zeros(5000)
    other[2500] = 1.0

    assert json.dumps(one, default=archive._json_value) != json.dumps(
        other, default=archive._json_value
    )


def test_the_fingerprint_depends_on_the_target_species(simulator):
    other = LWFASimulator(
        simulator.hyparams,
        simulator.laser,
        simulator.densities,
        target_species="electrons_downramp",
    )

    assert simulator.target_species != other.target_species
    assert archive.fingerprint(simulator) != archive.fingerprint(other)


def test_fingerprint_is_stable_for_the_same_config(simulator):
    assert archive.fingerprint(simulator) == archive.fingerprint(simulator)


def test_archive_round_trips_config_only(simulator, tmp_path):
    simulator.configure()
    archive_path = tmp_path / "archive.h5"
    archive.save(simulator_model(simulator), archive_path)

    restored = load_simulator(archive_path, working_directory=tmp_path)

    assert restored.hyparams.to_dict() == simulator.hyparams.to_dict()
    assert restored.laser.to_dict() == simulator.laser.to_dict()
    assert len(restored.densities) == len(simulator.densities)
    for original, loaded in zip(simulator.densities, restored.densities):
        assert loaded.to_dict() == original.to_dict()
    assert restored.final_particles is None


def test_archive_stores_target_species_and_load_restores_it(simulator, tmp_path):
    archive.save(simulator_model(simulator), tmp_path / "a.h5")

    restored = load_simulator(tmp_path / "a.h5", working_directory=tmp_path)

    assert restored.target_species == simulator.target_species
    assert restored.hyparams.to_dict() == simulator.hyparams.to_dict()
    assert restored.laser.to_dict() == simulator.laser.to_dict()
    assert [d.to_dict() for d in restored.densities] == [
        d.to_dict() for d in simulator.densities
    ]
    assert restored.working_directory == tmp_path


def test_load_results_reads_diagnostics_under_the_given_directory(tmp_path):
    ran = short_pwfa_simulator(tmp_path / "ran")
    ran.run()
    other = short_pwfa_simulator(tmp_path / "other")

    other.load_results(tmp_path / "ran")

    numpy.testing.assert_array_equal(other.final_particles.z, ran.final_particles.z)
    assert other.stats == ran.stats
    assert not (tmp_path / "other").exists()  # nothing ran there


def test_archive_with_a_lume_fbpic_profile_loads_in_a_fresh_process(
    simulator, tmp_path
):
    """A batch worker is a new process: `lume_fbpic`'s own profiles must register on import, and a
    bare-electron profile (`species=None`, stored with `ionization=1`) must load back.
    """
    import subprocess
    import sys

    from lume_fbpic.density_profiles import LinearRampFlattop

    profile = LinearRampFlattop(
        nominal_density=1.0e24,
        species=None,
        ionization=-1,
        p_nz=1,
        p_nr=1,
        p_nt=1,
        ramp_start=1.0e-5,
        ramp_length=1.0e-5,
        elec_name="electrons",
    )
    ramp = LWFASimulator(
        simulator.hyparams,
        simulator.laser,
        [profile],
        target_species="electrons",
        working_directory=tmp_path,
    )
    archive.save(simulator_model(ramp), tmp_path / "ramp.h5")

    code = (
        "from lume_fbpic import archive; "
        f"s = archive.load({str(tmp_path / 'ramp.h5')!r}).simulator; "
        "print(type(s.densities[0]).__name__, s.densities[0].species, s.densities[0].ionization)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-3:] == ["LinearRampFlattop", "None", "1"]


def test_an_archive_records_its_kind_and_load_builds_the_matching_class(
    simulator, tmp_path
):
    archive.save(simulator_model(simulator), tmp_path / "a.h5")

    with h5py.File(tmp_path / "a.h5") as f:
        assert f.attrs["config_kind"] == "lwfa"
    assert type(load_simulator(tmp_path / "a.h5")) is LWFASimulator


def test_loading_an_archive_of_an_unknown_kind_raises(simulator, tmp_path):
    archive.save(simulator_model(simulator), tmp_path / "a.h5")
    with h5py.File(tmp_path / "a.h5", "a") as f:
        f.attrs["config_kind"] = "nope"

    with pytest.raises(ValueError, match="unknown kind"):
        archive.load(tmp_path / "a.h5")


def test_config_and_set_config_round_trip(simulator):
    config = simulator.config()
    original = simulator.laser
    simulator.laser = None

    simulator.set_config(config)

    assert simulator.laser is original
    assert config["densities"] is not simulator.densities


def test_a_real_lwfa_run_writes_its_dumps_and_reads_the_last_one_back(tmp_path):
    simulator = short_lwfa_simulator(tmp_path)

    simulator.run()

    dumps = sorted(simulator.diagnostics_directory(tmp_path).glob("data*.h5"))
    assert len(dumps) == simulator.hyparams.number_dumps
    # the run lasts until the window has passed the plasma, and nothing is trapped in 40 steps
    assert len(simulator.final_particles) == 0
    assert set(simulator.stats.values()) == {0.0}


def test_results_read_back_replace_the_recorded_outputs(tmp_path):
    simulator = short_pwfa_simulator(tmp_path)
    simulator.recorded_outputs = {"descriptor_mean_uz": 171.6}

    simulator.run()

    assert simulator.recorded_outputs == {}
    assert simulator.stats["charge_pc"] == pytest.approx(
        simulator.driver.charge * 1e12, rel=1e-12
    )
