"""Tests for FBPICSimulator's config/lifecycle/archive paths, and a few short real fbpic runs
(`short_runs`)."""

from __future__ import annotations

import json

import attrs

import h5py
import numpy
import pytest

from inversion_fbpic.lib.datapoint import Parameters
from inversion_fbpic.lib.density_profiles import ExpressionDensityProfile
from inversion_fbpic.lib.external_field import (
    ExpressionExternalField,
    UniformExternalField,
)
from inversion_fbpic.lib.math_expression import MathExpression
from scipy.constants import e
from archive_helpers import load_simulator, simulator_model
from lume_fbpic import archive
from lume_fbpic.actions import make_descriptor_actions
from lume_fbpic.simulator import FBPICSimulator
from lume_fbpic.model import LUMEFBPICModel
from short_runs import (
    short_beam_simulator,
    short_lwfa_simulator,
    short_pwfa_simulator,
    uniform_beam_charge,
)


def test_configure_accepts_complete_config(simulator):
    simulator.configure()  # does not raise


def test_configure_rejects_missing_densities(hyparams, laser, tmp_path):
    simulator = FBPICSimulator(
        hyparams,
        lasers=[laser],
        densities=[],
        target_species="electrons_flattop",
        working_directory=tmp_path,
    )
    with pytest.raises(ValueError):
        simulator.configure()


def test_run_checks_a_config_that_changed_after_it_was_configured(simulator):
    simulator.configure()
    simulator.densities = []

    with pytest.raises(ValueError, match="a density profile"):
        simulator.run()

    assert not simulator.diagnostics_directory(simulator.working_directory).exists()


def test_run_with_an_incomplete_config_raises_and_runs_nothing(simulator):
    simulator.densities = []

    with pytest.raises(ValueError, match="a density profile"):
        simulator.run()

    assert not simulator.diagnostics_directory(simulator.working_directory).exists()


def test_reset_restores_the_starting_state_without_rerunning(simulator):
    config = simulator.config()
    simulator.configure()
    simulator.stats = {"charge_pc": 1.0}
    simulator.final_particles = object()
    simulator.lasers = []

    simulator.reset()

    assert simulator.stats == {}
    assert simulator.final_particles is None
    assert simulator.lasers == config["lasers"]
    assert simulator.lasers[0] is config["lasers"][0]
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
    laser = restored.lasers[0]

    restored.lasers = []
    restored.stats = {}
    restored.final_particles = None
    restored.reset()

    assert restored.lasers[0] is laser
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
    other = FBPICSimulator(
        simulator.hyparams,
        lasers=simulator.lasers,
        densities=simulator.densities,
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
    assert restored.lasers[0].to_dict() == simulator.lasers[0].to_dict()
    assert len(restored.densities) == len(simulator.densities)
    for original, loaded in zip(simulator.densities, restored.densities):
        assert loaded.to_dict() == original.to_dict()
    assert restored.final_particles is None


def test_archive_stores_target_species_and_load_restores_it(simulator, tmp_path):
    archive.save(simulator_model(simulator), tmp_path / "a.h5")

    restored = load_simulator(tmp_path / "a.h5", working_directory=tmp_path)

    assert restored.target_species == simulator.target_species
    assert restored.hyparams.to_dict() == simulator.hyparams.to_dict()
    assert restored.lasers[0].to_dict() == simulator.lasers[0].to_dict()
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
    # reading the results of a run made elsewhere removes none of its dumps
    assert list(ran.diagnostics_directory(tmp_path / "ran").glob("data*.h5"))


def test_a_run_does_not_read_the_dumps_of_a_longer_earlier_run(tmp_path):
    short_pwfa_simulator(tmp_path / "run", n_steps=6).run()
    shorter = short_pwfa_simulator(tmp_path / "run", n_steps=3)
    fresh = short_pwfa_simulator(tmp_path / "fresh", n_steps=3)

    shorter.run()
    fresh.run()

    # without removing the earlier dumps, the 6-step run's last dump would be read, 3 um further
    # on; the two 3-step runs agree to rounding
    numpy.testing.assert_allclose(
        shorter.final_particles.z, fresh.final_particles.z, rtol=1e-12
    )
    assert shorter.stats == pytest.approx(fresh.stats, rel=1e-9)


def test_a_run_removes_only_the_dumps_of_the_previous_run(tmp_path):
    simulator = short_pwfa_simulator(tmp_path)
    dumps = simulator.diagnostics_directory(tmp_path)
    dumps.mkdir(parents=True)
    stale = dumps / "data00000099.h5"
    stale.write_bytes(b"an earlier run's dump")
    other = dumps / "notes.txt"
    other.write_text("kept")
    beside = dumps.parent / "lasy_laser_00000.h5"  # where a LASY laser file is written
    beside.write_bytes(b"kept")

    simulator.run()

    assert not stale.exists()
    assert other.read_text() == "kept" and beside.read_bytes() == b"kept"
    assert sorted(path.name for path in dumps.glob("data*.h5")) == [
        "data00000000.h5",
        "data00000002.h5",
        "data00000003.h5",
    ]


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
    ramp = FBPICSimulator(
        simulator.hyparams,
        lasers=simulator.lasers,
        densities=[profile],
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
        assert f.attrs["config_kind"] == "fbpic"
    assert type(load_simulator(tmp_path / "a.h5")) is FBPICSimulator


def test_loading_an_archive_of_an_unknown_kind_raises(simulator, tmp_path):
    archive.save(simulator_model(simulator), tmp_path / "a.h5")
    with h5py.File(tmp_path / "a.h5", "a") as f:
        f.attrs["config_kind"] = "nope"

    with pytest.raises(ValueError, match="unknown kind"):
        archive.load(tmp_path / "a.h5")


def test_config_and_set_config_round_trip(simulator):
    config = simulator.config()
    original = simulator.lasers[0]
    simulator.lasers = []

    simulator.set_config(config)

    assert simulator.lasers[0] is original
    assert config["lasers"] is not simulator.lasers
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
        uniform_beam_charge(simulator.beams[0]) * 1e12, rel=1e-12
    )


def test_a_beam_with_no_laser_and_no_plasma_runs_and_is_read_back(tmp_path):
    simulator = short_beam_simulator(tmp_path)

    simulator.run()

    beam = simulator.beams[0]
    assert len(simulator.final_particles) == beam.n_macroparticles
    assert simulator.final_particles.charge == pytest.approx(
        beam.n_physical_particles * e, rel=1e-9
    )


def test_an_external_field_on_the_beam_deflects_it(tmp_path):
    field = 1.0e9  # [V/m]
    plain = short_beam_simulator(tmp_path / "plain")
    pushed = short_beam_simulator(
        tmp_path / "pushed",
        external_fields=[
            UniformExternalField(field_functions={"Ex": field}, species="beam")
        ],
    )

    plain.run()
    pushed.run()

    # POSIT: an electron (charge -e) at about c gains px = -e E t, t the time the run lasts:
    # the window travels `simulation_length` plus its own length (`right_buffer`'s default)
    h = plain.hyparams
    expected = -field * (h.simulation_length + (h.zmax - h.zmin))  # [eV/c]
    kick = numpy.mean(pushed.final_particles.px) - numpy.mean(plain.final_particles.px)
    assert kick == pytest.approx(expected, rel=0.2)


def test_an_archive_round_trips_beams_external_fields_and_expressions(tmp_path):
    density = ExpressionDensityProfile(
        nominal_density=1.0e24,
        function=MathExpression(expression="1 + z", variables=("z",)),
        z_min=0.0,
        z_max=1.0e-5,
        p_nz=1,
        p_nr=1,
        p_nt=1,
        species=None,
        ionization=-1,
        elec_name="plasma",
    )
    field = ExpressionExternalField(
        field_functions={"Bx": MathExpression(expression="2.0 * y", variables=("y",))},
        species="beam",
    )
    simulator = short_beam_simulator(tmp_path, external_fields=[field])
    simulator.densities = [density]
    archive.save(simulator_model(simulator), tmp_path / "a.h5")

    loaded = load_simulator(tmp_path / "a.h5", working_directory=tmp_path)

    for name in ("lasers", "densities", "beams", "external_fields"):
        assert [x.to_dict() for x in getattr(loaded, name)] == [
            x.to_dict() for x in getattr(simulator, name)
        ], name
    assert archive.fingerprint(loaded) == archive.fingerprint(simulator)


def test_configure_rejects_a_config_with_no_laser_and_no_beam(simulator):
    simulator.lasers = []

    with pytest.raises(ValueError, match="a laser or a particle beam"):
        simulator.configure()


def test_configure_rejects_a_target_species_the_config_does_not_write(tmp_path):
    simulator = short_beam_simulator(tmp_path)
    simulator.target_species = "electrons"

    with pytest.raises(ValueError, match="'electrons' is not one of \\['beam'\\]"):
        simulator.configure()


def test_a_beam_run_without_a_simulation_length_needs_a_plasma(tmp_path):
    simulator = short_beam_simulator(tmp_path)
    simulator.hyparams = attrs.evolve(simulator.hyparams, simulation_length=None)

    with pytest.raises(ValueError, match="simulation_length"):
        simulator.configure()


def test_a_list_left_out_of_a_config_is_set_as_empty(tmp_path):
    simulator = short_beam_simulator(tmp_path)
    config = simulator.config()
    del config["external_fields"]

    simulator.set_config(config)

    assert simulator.external_fields == []


def test_an_antenna_laser_with_no_v_antenna_runs_at_rest_and_keeps_its_none(tmp_path):
    from lume_fbpic.utils import evolve_laser

    simulator = short_lwfa_simulator(tmp_path)
    simulator.lasers = [
        evolve_laser(simulator.lasers[0], method="antenna", z0_antenna=-5.0e-6)
    ]
    assert simulator.lasers[0].v_antenna is None

    simulation = simulator.run_simulation()  # fbpic fails on a v_antenna of None

    assert [a.vz.mean() for a in simulation.simulation.laser_antennas] == [0.0]
    assert simulator.lasers[0].v_antenna is None  # the config is left as it was
    archive.save(simulator_model(simulator), tmp_path / "a.h5")
    assert load_simulator(tmp_path / "a.h5").lasers[0].v_antenna is None


def test_a_laser_that_is_not_an_antenna_or_has_a_v_antenna_is_run_as_it_is(simulator):
    from lume_fbpic.simulator import _antenna_at_rest
    from lume_fbpic.utils import evolve_laser

    direct = simulator.lasers[0]
    moving = evolve_laser(direct, method="antenna", z0_antenna=0.0, v_antenna=1.0e3)

    assert _antenna_at_rest(direct) is direct
    assert _antenna_at_rest(moving) is moving
