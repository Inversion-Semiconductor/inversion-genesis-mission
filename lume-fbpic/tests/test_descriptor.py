"""Tests for the moment-descriptor actions (`MomentDescriptorAction`, `make_descriptor_actions`)."""

from __future__ import annotations

import collections
import math

import pytest

from bunches import lab_snapshot
from downramp_actions import make_actions
from reference_descriptor import reference_descriptor
from lume_fbpic import descriptor
from lume_fbpic.actions import MomentDescriptorAction, make_descriptor_actions
from lume_fbpic.model import LUMEFBPICModel


@pytest.fixture()
def descriptor_model(simulator) -> LUMEFBPICModel:
    actions = [*make_actions(simulator), *make_descriptor_actions()]
    return LUMEFBPICModel(simulator, actions, dummy_run=True)


def test_actions_are_the_33_read_only_features_in_library_order():
    actions = make_descriptor_actions()

    keys = list(
        reference_descriptor(
            lab_snapshot(),
            uz_min=descriptor.UZ_MIN,
            central_fraction=descriptor.CENTRAL_FRACTION,
        )
    )
    assert [a.name for a in actions] == [f"descriptor_{k}" for k in keys]
    assert len(actions) == 33
    assert all(a.read_only for a in actions)
    assert actions[-1].unit == "pC"


def test_new_final_particles_give_new_values(descriptor_model, simulator):
    # the descriptor is cached: a stale cache would serve the first bunch's values
    names = ["descriptor_mean_uz", "descriptor_cov_uz_uz"]
    simulator.final_particles = lab_snapshot(seed=1)
    first = descriptor_model.get(names)

    pg = lab_snapshot(seed=2)
    simulator.final_particles = pg
    second = descriptor_model.get(names)

    expected = reference_descriptor(
        pg, uz_min=descriptor.UZ_MIN, central_fraction=descriptor.CENTRAL_FRACTION
    )
    assert second == {n: expected[n.removeprefix("descriptor_")] for n in names}
    assert second != first


def test_two_selections_in_one_model_each_get_their_own_values(simulator):
    # the cache is per selection: a shared one would give both the first one's values
    model = LUMEFBPICModel(
        simulator,
        [
            *make_descriptor_actions(),
            *make_descriptor_actions(prefix="all_", uz_min=None, central_fraction=None),
        ],
        dummy_run=True,
    )
    pg = lab_snapshot()
    simulator.final_particles = pg

    got = model.get(["descriptor_mean_uz", "all_mean_uz"])

    assert (
        got["descriptor_mean_uz"]
        == reference_descriptor(
            pg, uz_min=descriptor.UZ_MIN, central_fraction=descriptor.CENTRAL_FRACTION
        )["mean_uz"]
    )
    assert (
        got["all_mean_uz"]
        == reference_descriptor(pg, uz_min=None, central_fraction=None)["mean_uz"]
    )
    assert got["descriptor_mean_uz"] != got["all_mean_uz"]


def test_each_descriptor_action_has_the_unit_of_its_feature():
    units = {action.feature: action.unit for action in make_descriptor_actions()}

    assert units["mean_ux"] == units["mean_uz"] == "1"  # normalized momentum
    assert units["cov_x_x"] == units["cov_z_z"] == "m^2"
    assert units["cov_x_ux"] == units["cov_y_uz"] == units["cov_z_uz"] == "m"
    assert units["cov_ux_ux"] == units["cov_uy_uz"] == "1"
    assert units["longitudinal_mean_uz_00"] == units["longitudinal_rms_uz_03"] == "1"
    assert units["total_beam_charge_pc"] == "pC"
    assert collections.Counter(units.values()) == {"1": 17, "m": 9, "m^2": 6, "pC": 1}


def test_a_descriptor_action_made_without_a_unit_gets_its_features(simulator):
    action = MomentDescriptorAction(name="size", feature="cov_x_x", read_only=True)

    model = LUMEFBPICModel(simulator, [action], dummy_run=True)

    assert model.supported_variables["size"].unit == "m^2"
    assert action.unit is None  # the action given is left as it is


def test_selection_parameters_change_the_result(simulator):
    pg = lab_snapshot()
    simulator.final_particles = pg
    name = "descriptor_total_beam_charge_pc"

    def charge(**selection):
        model = LUMEFBPICModel(
            simulator, make_descriptor_actions(**selection), dummy_run=True
        )
        return model.get([name])[name]

    everything = charge(uz_min=None, central_fraction=None)
    assert everything == pytest.approx(pg.charge * 1e12, rel=1e-12)  # rounding
    assert (
        charge(uz_min=170.0, central_fraction=0.5)
        == reference_descriptor(pg, uz_min=170.0, central_fraction=0.5)[
            "total_beam_charge_pc"
        ]
    )


def test_values_match_the_library_descriptor(descriptor_model, simulator):
    pg = lab_snapshot()
    simulator.final_particles = pg
    expected = reference_descriptor(
        pg, uz_min=descriptor.UZ_MIN, central_fraction=descriptor.CENTRAL_FRACTION
    )

    got = descriptor_model.get([f"descriptor_{key}" for key in expected])

    assert got == {f"descriptor_{key}": value for key, value in expected.items()}


def test_values_are_nan_before_a_run(descriptor_model):
    value = descriptor_model.get(["descriptor_mean_uz"])["descriptor_mean_uz"]

    assert math.isnan(value)


def test_values_are_nan_when_the_selection_leaves_too_few_particles(
    descriptor_model, simulator
):
    simulator.final_particles = lab_snapshot(
        uz_mean=2.0, uz_std=0.5
    )  # none reach uz >= 30

    value = descriptor_model.get(["descriptor_mean_uz"])["descriptor_mean_uz"]

    assert math.isnan(value)


def test_final_particles_can_be_read_before_any_result_exists(descriptor_model):
    """`lume-pva` reads every variable each cycle; a missing result must not make `get()` fail."""
    from lume_fbpic.actions import FinalParticlesAction

    model = LUMEFBPICModel(
        descriptor_model.simulator,
        [FinalParticlesAction(name="final_particles", read_only=True)],
        dummy_run=True,
    )

    assert model.get(["final_particles"])["final_particles"] is None
