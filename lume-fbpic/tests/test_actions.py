"""Tests for lume_fbpic action/variable classes, through the model's `get()` and `set()`."""

from __future__ import annotations

import math

import attrs
import pytest

from conftest import LASER_KWARGS
from downramp_actions import make_actions
from inversion_fbpic.lib.laser import GaussianLaserPulse
from lume_fbpic.actions import (
    DopantFractionAction,
    HyperparameterFieldAction,
    LaserFieldAction,
)
from lume_fbpic.density_profiles import atom_density, nominal_density
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
    assert _get(model, "laser_energy") == model.simulator.lasers[0].energy


def test_a_laser_field_with_an_index_sets_that_laser_only(simulator):
    first = simulator.lasers[0]
    simulator.lasers = [first, GaussianLaserPulse(energy=1.0, **LASER_KWARGS)]
    model = LUMEFBPICModel(
        simulator,
        [LaserFieldAction(name="second_energy", field_name="energy", laser_index=1)],
        dummy_run=True,
    )

    model.set({"second_energy": 2.5})

    assert simulator.lasers[1].energy == 2.5
    assert simulator.lasers[0] is first


def test_setting_a_laser_field_replaces_the_laser_and_leaves_the_old_one(model):
    original = model.simulator.lasers[0]

    model.set({"laser_energy": 7.5})

    assert model.simulator.lasers[0].energy == 7.5
    assert _get(model, "laser_energy") == 7.5
    assert original.energy == 5.0


def test_setting_another_laser_field_keeps_the_energy_and_derives_a0_again(model):
    # Every laser holds both energy and a0, one given and one derived; the derived one must be
    # dropped before the laser is rebuilt, or the rebuild sees both.
    model.set({"laser_focal_position": 20.0e-6})

    expected = GaussianLaserPulse(
        energy=5.0, **{**LASER_KWARGS, "focal_position": 20.0e-6}
    )
    assert model.simulator.lasers[0].focal_position == 20.0e-6
    assert model.simulator.lasers[0].energy == 5.0
    assert model.simulator.lasers[0].a0 == expected.a0


def test_after_a0_is_set_another_laser_field_keeps_a0_and_derives_the_energy(model):
    model.set({"laser_a0": 3.0})
    model.set({"laser_focal_position": 20.0e-6})

    expected = GaussianLaserPulse(a0=3.0, **{**LASER_KWARGS, "focal_position": 20.0e-6})
    assert model.simulator.lasers[0].a0 == 3.0
    assert model.simulator.lasers[0].focal_position == 20.0e-6
    assert model.simulator.lasers[0].energy == expected.energy


def test_a_hyperparameter_field_reads_and_sets_the_hyperparameters(model):
    assert _get(model, "nz") == model.simulator.hyparams.nz

    model.set({"nz": 16})

    assert model.simulator.hyparams.nz == 16


def test_a_density_field_sets_only_the_profile_it_indexes(model):
    other = model.simulator.densities[0]

    model.set({"downramp_length": 0.25e-3})

    assert model.simulator.densities[1].downramp_length == 0.25e-3
    assert model.simulator.densities[0] == other


def _gas_model(simulator, *, he_atoms, n_atoms, **fraction_options):
    """A model of `simulator` with a He host and an N dopant at the given atom densities [m^-3],
    and a dopant fraction action."""
    he, n = simulator.densities
    simulator.densities = [
        attrs.evolve(
            he,
            species="He",
            ionization=0,
            nominal_density=nominal_density("He", he_atoms),
        ),
        attrs.evolve(
            n, species="N", ionization=0, nominal_density=nominal_density("N", n_atoms)
        ),
    ]
    action = DopantFractionAction(
        name="dopant_fraction", host_index=0, dopant_index=1, **fraction_options
    )
    return LUMEFBPICModel(simulator, [action], dummy_run=True)


def test_a_dopant_fraction_is_of_atoms_by_default(simulator):
    model = _gas_model(simulator, he_atoms=9.0e23, n_atoms=1.0e23)
    assert _get(model, "dopant_fraction") == pytest.approx(0.1)

    model.set({"dopant_fraction": 0.25})

    he, n = simulator.densities
    # the total atom density stays 1e24
    assert atom_density(n) == pytest.approx(0.25e24)
    assert atom_density(he) == pytest.approx(0.75e24)


def test_a_dopant_fraction_of_diatomic_molecules(simulator):
    # N2 in He at a neutral density of 1e24 molecules per m^3, 3% N2: 0.97e24 He atoms and
    # 2 * 0.03e24 N atoms, as the ionization-injection template seeds them
    model = _gas_model(
        simulator, he_atoms=0.97e24, n_atoms=0.06e24, dopant_atoms_per_molecule=2
    )
    assert _get(model, "dopant_fraction") == pytest.approx(0.03)

    model.set({"dopant_fraction": 0.05})

    he, n = simulator.densities
    # the neutral density stays 1e24
    assert atom_density(n) == pytest.approx(2 * 0.05e24)
    assert atom_density(he) == pytest.approx(0.95e24)


def test_a_molecule_has_at_least_one_atom():
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        DopantFractionAction(
            name="f", host_index=0, dopant_index=1, dopant_atoms_per_molecule=0
        )


def test_a_stat_reads_the_simulators_stats(model):
    model.simulator.stats = {"charge_pc": 12.3}

    assert _get(model, "charge_pc") == 12.3


def test_a_stat_is_nan_before_any_run(model):
    assert math.isnan(_get(model, "energy_mean_mev"))


def test_the_final_particles_are_the_simulators(model, particle_group):
    model.simulator.final_particles = particle_group

    assert _get(model, "final_particles") is particle_group


def test_the_beam_actions_of_a_gaussian_beam(tmp_path):
    from lume_fbpic.actions import make_beam_actions
    from short_runs import short_beam_simulator

    simulator = short_beam_simulator(tmp_path)
    model = LUMEFBPICModel(simulator, make_beam_actions(simulator), dummy_run=True)

    assert sorted(model.supported_variables) == [
        "beam_charge",
        "beam_emittance",
        "beam_gamma",
        "beam_gamma_spread",
        "beam_position",
        "beam_sigma_r",
        "beam_sigma_z",
    ]
    assert model.supported_variables["beam_sigma_r"].unit == "m"
    assert model.supported_variables["beam_charge"].unit == "C"
    assert _get(model, "beam_charge") == pytest.approx(1.0e6 * 1.602176634e-19)


def test_every_beam_field_an_action_names_is_a_field_of_its_beam():
    # `_BEAM_FIELDS` names inversion_fbpic's fields by string: a renamed field would otherwise
    # fail only on an action's first get(), and only one class of beam is run in the tests
    from inversion_fbpic.lib.particle_beam import FileParticleBeam, GaussianParticleBeam
    from lume_fbpic.actions import _BEAM_FIELDS

    for cls, fields in _BEAM_FIELDS.items():
        assert set(fields.values()) <= set(attrs.fields_dict(cls)), cls.__name__
    # the beams with a charge action (`BeamTotalChargeAction`), as its docstring lists them
    assert {
        c for c in _BEAM_FIELDS if "n_physical_particles" in attrs.fields_dict(c)
    } == {FileParticleBeam, GaussianParticleBeam}


def test_setting_a_beam_field_replaces_that_beam_only(tmp_path):
    from lume_fbpic.actions import make_beam_actions
    from short_runs import short_beam_simulator

    simulator = short_beam_simulator(tmp_path)
    other = attrs.evolve(simulator.beams[0], particle_name="other")
    simulator.beams = [other, simulator.beams[0]]
    model = LUMEFBPICModel(
        simulator, make_beam_actions(simulator, beam_index=1), dummy_run=True
    )

    model.set({"beam_gamma": 250.0, "beam_charge": 2.0e-13})

    assert simulator.beams[1].gamma0 == 250.0
    assert simulator.beams[1].n_physical_particles * 1.602176634e-19 == pytest.approx(
        2.0e-13
    )
    assert simulator.beams[0] is other


def test_beam_actions_need_a_prefix_for_a_beam_with_no_particle_name(tmp_path):
    from lume_fbpic.actions import make_beam_actions
    from short_runs import short_beam_simulator

    simulator = short_beam_simulator(tmp_path, particle_name=None)

    with pytest.raises(ValueError, match="give a prefix"):
        make_beam_actions(simulator)
    assert make_beam_actions(simulator, prefix="b")[0].name.startswith("b_")


def test_an_external_field_action_sets_one_component(tmp_path):
    from inversion_fbpic.lib.external_field import UniformExternalField
    from lume_fbpic.actions import ExternalFieldAction
    from short_runs import short_beam_simulator

    simulator = short_beam_simulator(
        tmp_path,
        external_fields=[UniformExternalField(field_functions={"Ex": 1.0, "Bz": 2.0})],
    )
    model = LUMEFBPICModel(
        simulator,
        [
            ExternalFieldAction(name="bz", field_index=0, field_type="Bz"),
            ExternalFieldAction(name="ey", field_index=0, field_type="Ey"),
        ],
        dummy_run=True,
    )

    model.set({"bz": 0.5})

    assert simulator.external_fields[0].field_functions == {"Ex": 1.0, "Bz": 0.5}
    assert _get(model, "ey") == 0.0
    assert model.supported_variables["bz"].unit == "T"
    assert model.supported_variables["ey"].unit == "V/m"


def test_an_external_field_action_refuses_an_expression_field(tmp_path):
    from inversion_fbpic.lib.external_field import ExpressionExternalField
    from inversion_fbpic.lib.math_expression import MathExpression
    from lume_fbpic.actions import ExternalFieldAction
    from short_runs import short_beam_simulator

    simulator = short_beam_simulator(
        tmp_path,
        external_fields=[
            ExpressionExternalField(
                field_functions={"Bx": MathExpression(expression="y", variables=("y",))}
            )
        ],
    )

    model = LUMEFBPICModel(
        simulator,
        [ExternalFieldAction(name="bx", field_index=0, field_type="Bx")],
        dummy_run=True,
    )

    with pytest.raises(TypeError, match="needs a UniformExternalField"):
        _get(model, "bx")


def test_beam_and_external_field_actions_come_back_from_an_archive(tmp_path):
    from inversion_fbpic.lib.external_field import UniformExternalField
    from lume_fbpic import archive
    from lume_fbpic.actions import ExternalFieldAction, make_beam_actions
    from short_runs import short_beam_simulator

    simulator = short_beam_simulator(
        tmp_path, external_fields=[UniformExternalField(field_functions={"Bz": 2.0})]
    )
    actions = [
        *make_beam_actions(simulator),
        ExternalFieldAction(name="bz", field_index=0, field_type="Bz"),
    ]
    model = LUMEFBPICModel(simulator, actions, dummy_run=True)
    model.archive(tmp_path / "a.h5")

    loaded = archive.load(tmp_path / "a.h5", strict=True, dummy_run=True)

    assert sorted(loaded.supported_variables) == sorted(model.supported_variables)
    loaded.set({"bz": 3.0, "beam_charge": 1.0e-13})
    assert loaded.simulator.external_fields[0].field_functions == {"Bz": 3.0}
    assert _get(loaded, "beam_charge") == pytest.approx(1.0e-13)
