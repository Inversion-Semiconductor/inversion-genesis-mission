"""Tests for `lume_fbpic_twin.twin_stage.TwinStage` against the real HTU twin.

Need `htu` (geecs-lume-twin) importable, for example with `PYTHONPATH` pointing at its checkout;
they are skipped otherwise. The twin builds slowly (about a second), so one is shared per module.
"""

from __future__ import annotations

import numpy
import pytest

pytest.importorskip("htu")
pytest.importorskip("lume_pva")
pytest.importorskip("lume_fbpic_twin")

from lume.exceptions import ReadOnlyError  # noqa: E402
from lume.staged_model import StagedModel  # noqa: E402

from htu.model import build_htu_model  # noqa: E402
from bunches import lab_snapshot  # noqa: E402
from downramp_actions import make_actions  # noqa: E402
from reference_descriptor import reference_descriptor  # noqa: E402
from lume_fbpic import archive, descriptor  # noqa: E402
from lume_fbpic.model import LUMEFBPICModel  # noqa: E402
from lume_fbpic_twin.bunch_frame import (  # noqa: E402
    bunch_frame_particles,
    to_bunch_frame,
)
from lume_fbpic_twin.twin_stage import (  # noqa: E402
    TwinStage,
    beam_moments,
    binned_screen_geometries,
    build_chain,
)

from beamphysics import ParticleGroup


@pytest.fixture(scope="module")
def twin_model():
    return build_htu_model()


@pytest.fixture()
def stage(twin_model):
    stage = TwinStage(twin_model)
    yield stage
    stage.reset()  # the twin model is shared: put it back for the next test


def _selected_charge_pc(particles) -> float:
    """The charge [pC] of `particles` after the stage's default selection."""
    return beam_moments(
        bunch_frame_particles(
            particles,
            uz_min=descriptor.UZ_MIN,
            central_fraction=descriptor.CENTRAL_FRACTION,
        )
    )["charge_pc"]


def _get(stage, name):
    return float(stage.get([name])[name])


def test_a_lab_snapshot_is_selected_and_injected(stage):
    pg = lab_snapshot()
    expected = bunch_frame_particles(
        pg, uz_min=descriptor.UZ_MIN, central_fraction=descriptor.CENTRAL_FRACTION
    )

    stage.initial_particles = pg

    assert stage.external_beam is True
    assert len(stage.initial_particles) == len(expected)
    assert _get(stage, "Source_Charge_pC") == pytest.approx(expected.charge * 1e12)
    assert _get(stage, "Source_NumParticles") == len(expected)


def test_source_pvs_read_the_injected_bunchs_moments(stage):
    pg = lab_snapshot(uz_mean=150.0)

    stage.initial_particles = pg

    # the twin holds its source parameters as float32
    moments = beam_moments(
        bunch_frame_particles(
            pg, uz_min=descriptor.UZ_MIN, central_fraction=descriptor.CENTRAL_FRACTION
        )
    )
    assert _get(stage, "Source_Energy_MeV") == pytest.approx(
        moments["energy_mev"], rel=1e-7
    )
    assert _get(stage, "Source_EnergySpread_pct") == pytest.approx(
        moments["energy_spread_pct"], rel=1e-7
    )


def test_screen_images_carry_the_injected_charge(stage):
    pg = lab_snapshot()

    stage.initial_particles = pg
    image = numpy.asarray(stage.get(["TCPhosphor_image"])["TCPhosphor_image"]).copy()
    stage.initial_particles = ParticleGroup(data={**pg.data, "weight": 2 * pg.weight})
    doubled = numpy.asarray(stage.get(["TCPhosphor_image"])["TCPhosphor_image"])

    # some charge reaches the screen, and every pixel scales with the bunch's charge
    assert image.sum() > 0
    numpy.testing.assert_array_equal(doubled, 2 * image)


def test_particles_that_already_have_arrival_times_are_used_as_they_are(stage):
    bunch = bunch_frame_particles(lab_snapshot(), uz_min=30.0, central_fraction=0.9)

    stage.initial_particles = bunch

    assert len(stage.initial_particles) == len(bunch)  # no second selection


def test_a_failed_injection_leaves_the_twin_untouched(stage):
    before = (_get(stage, "Source_Energy_MeV"), _get(stage, "Source_Charge_pC"))

    with pytest.raises(ValueError):
        stage.initial_particles = lab_snapshot(
            uz_mean=2.0, uz_std=0.2
        )  # nothing passes uz >= 30

    assert (_get(stage, "Source_Energy_MeV"), _get(stage, "Source_Charge_pC")) == before
    assert stage.external_beam is False


def test_an_unlocked_source_put_replaces_the_injected_beam(twin_model):
    stage = TwinStage(twin_model, lock_source=False)
    try:
        stage.initial_particles = lab_snapshot()
        assert stage.external_beam is True

        stage.set({"Source_Energy_MeV": 90.0})

        assert stage.external_beam is False
        assert _get(stage, "Source_Energy_MeV") == 90.0
    finally:
        stage.reset()


def test_a_locked_source_cannot_be_set_and_the_injected_beam_stays(stage):
    stage.initial_particles = lab_snapshot()
    charge = _get(stage, "Source_Charge_pC")
    assert charge == pytest.approx(
        stage.initial_particles.charge * 1e12
    )  # it reads the bunch

    with pytest.raises(ReadOnlyError):
        stage.set({"Source_Charge_pC": 5.0})
    stage.set({"EMQ1H_Current": 0.71})  # the other twin variables stay writable

    assert stage.external_beam is True
    assert _get(stage, "Source_Charge_pC") == charge
    assert _get(stage, "EMQ1H_Current") == pytest.approx(0.71, rel=1e-7)  # float32


def test_the_source_can_be_left_writable(twin_model):
    stage = TwinStage(twin_model, lock_source=False)

    assert not stage.supported_variables["Source_Energy_MeV"].read_only


def test_reset_restores_the_original_source_beam_and_forgets_the_drift(stage):
    original = (_get(stage, "Source_Energy_MeV"), _get(stage, "Source_Charge_pC"))
    stage.plasma_exit_z = 0.0029
    stage.initial_particles = lab_snapshot()
    assert stage.drift_length is not None

    stage.reset()

    assert stage.external_beam is False
    assert stage.initial_particles is None
    assert stage.drift_length is None
    assert (
        _get(stage, "Source_Energy_MeV"),
        _get(stage, "Source_Charge_pC"),
    ) == pytest.approx(original)


def test_wrapped_variables_are_the_twins_apart_from_the_locked_source(
    stage, twin_model
):
    twin = twin_model.supported_variables
    assert list(stage.supported_variables) == list(twin)
    sources = [name for name in stage.supported_variables if name.startswith("Source_")]
    assert len(sources) == 14
    assert all(stage.supported_variables[name].read_only for name in sources)
    for name, variable in stage.supported_variables.items():
        if name.startswith("Source_"):
            assert (
                variable.model_copy(update={"read_only": twin[name].read_only})
                == twin[name]
            )
        else:
            assert (
                variable is twin[name]
            )  # the rest is the twin's own, writable as it is


def test_the_stage_follows_an_lpa_model_in_a_staged_chain(stage, simulator):
    simulator.final_particles = lab_snapshot()
    lpa = LUMEFBPICModel(simulator, make_actions(simulator), dummy_run=True)
    chain = StagedModel([lpa, stage])

    with pytest.raises(ReadOnlyError):  # the LPA is the source: a source put is refused
        chain.set({"Source_Energy_MeV": 50.0})
    assert chain.supported_variables["Source_Energy_MeV"].read_only

    chain.set(
        {"EMQ1H_Current": 0.7}
    )  # only a twin variable: the LPA bunch is still passed on

    assert stage.external_beam is True
    assert _get(stage, "EMQ1H_Current") == pytest.approx(0.7, rel=1e-7)  # float32
    assert _get(stage, "Source_Charge_pC") == pytest.approx(
        _selected_charge_pc(simulator.final_particles), rel=1e-7
    )


def test_an_empty_set_of_the_chain_passes_the_lpa_bunch_on(stage, simulator):
    simulator.final_particles = lab_snapshot()
    lpa = LUMEFBPICModel(simulator, make_actions(simulator), dummy_run=True)
    chain = StagedModel([lpa, stage])

    chain.set({})

    assert stage.external_beam is True
    assert _get(stage, "Source_Charge_pC") == pytest.approx(
        _selected_charge_pc(simulator.final_particles), rel=1e-7
    )


def test_without_a_plasma_exit_plane_the_bunch_stays_at_its_mean_z(stage):
    pg = lab_snapshot()
    expected = bunch_frame_particles(
        pg, uz_min=descriptor.UZ_MIN, central_fraction=descriptor.CENTRAL_FRACTION
    )

    stage.initial_particles = pg

    assert stage.drift_length is None
    injected = stage.initial_particles
    for name in ("x", "z"):
        assert numpy.array_equal(
            numpy.asarray(getattr(injected, name)),
            numpy.asarray(getattr(expected, name)),
        ), name
    # t relative to the mean arrival, for Cheetah
    assert numpy.array_equal(
        numpy.asarray(injected.t), numpy.asarray(expected.t - expected.avg("t"))
    )


def test_the_bunch_is_drifted_to_the_plasma_exit_plane(twin_model):
    stage = TwinStage(
        twin_model, plasma_exit_z=0.0029
    )  # the snapshot's mean z is 0.003
    try:
        pg = lab_snapshot()
        expected_selection = bunch_frame_particles(
            pg, uz_min=descriptor.UZ_MIN, central_fraction=descriptor.CENTRAL_FRACTION
        )

        stage.initial_particles = pg

        # the bunch's charge-weighted mean z, which is its plane to rounding
        assert stage.drift_length == pytest.approx(
            0.0029 - expected_selection.z[0], rel=1e-12
        )
        assert stage.drift_length < 0  # the bunch was past the exit: moved back
        injected = stage.initial_particles
        expected = to_bunch_frame(expected_selection, 0.0029)
        assert numpy.all(numpy.asarray(injected.z) == 0.0029)
        for name in ("x", "y"):
            assert numpy.array_equal(
                numpy.asarray(getattr(injected, name)),
                numpy.asarray(getattr(expected, name)),
            ), name
        assert numpy.array_equal(
            numpy.asarray(injected.t), numpy.asarray(expected.t - expected.avg("t"))
        )
        moments = beam_moments(injected)  # the source PVs read the bunch at the plane
        # the twin holds its source parameters as float32
        assert _get(stage, "Source_BetaX_mm") == pytest.approx(
            moments["beta_x_mm"], rel=1e-7
        )
        assert _get(stage, "Source_X_um") == pytest.approx(moments["x_um"], rel=1e-7)
    finally:
        stage.reset()


def test_a_bunch_that_has_not_reached_the_plane_is_drifted_with_a_warning(twin_model):
    stage = TwinStage(
        twin_model, plasma_exit_z=0.0031
    )  # beyond the snapshot's z of 0.003
    try:
        pg = lab_snapshot()
        selection = bunch_frame_particles(
            pg, uz_min=descriptor.UZ_MIN, central_fraction=descriptor.CENTRAL_FRACTION
        )

        with pytest.warns(UserWarning, match="short of the plasma exit"):
            stage.initial_particles = pg

        assert stage.drift_length == pytest.approx(0.0031 - selection.z[0], rel=1e-12)
    finally:
        stage.reset()


def test_the_served_chain_exposes_the_twin_controls_and_keeps_the_source_read_only(
    simulator, twin_model
):
    from lume_pva.runner import PutMode

    from lume_fbpic_twin.serve import build_config

    simulator.final_particles = lab_snapshot()
    chain = build_chain(
        LUMEFBPICModel(simulator, make_actions(simulator), dummy_run=True), twin_model
    )
    stage = chain.lume_model_instances[1]
    try:
        config = build_config(
            chain,
            prefix="HTU:SIM:",
            serve_always=set(stage.supported_variables),
            wait_for_puts=True,
        )
        modes = {
            name: str(entry["mode"]) for name, entry in config["variables"].items()
        }

        assert modes["EMQ1H_Current"] == "rw"  # a twin control, served read-write
        assert modes["Source_Energy_MeV"] == "ro"  # the LPA is the source
        assert modes["charge_pc"] == "ro"  # an LPA output
        assert modes["laser_energy"] == "ro"  # an LPA input is served read-only
        assert config["put_mode"] == PutMode.Complete
        assert config["prefix"] == "HTU:SIM:"
    finally:
        stage.reset()


def test_tracking_after_an_injection_is_deferred_until_the_twin_is_read(
    stage, twin_model, mocker
):
    track = mocker.spy(twin_model.simulator, "track")

    stage.initial_particles = lab_snapshot()
    assert track.call_count == 0

    _get(stage, "Source_Charge_pC")
    assert track.call_count == 1
    _get(stage, "Source_Energy_MeV")
    assert track.call_count == 1  # tracked once, not on every read


def test_a_set_after_an_injection_tracks_once(stage, twin_model, mocker):
    track = mocker.spy(twin_model.simulator, "track")

    stage.initial_particles = lab_snapshot()
    stage.set({"EMQ1H_Current": 0.7})

    assert (
        track.call_count == 1
    )  # the twin's own re-track; not a second one for the injection
    assert stage.external_beam is True


def test_screens_are_current_when_read_right_after_an_injection(stage, twin_model):
    stage.initial_particles = lab_snapshot()

    image = numpy.asarray(stage.get(["TCPhosphor_image"])["TCPhosphor_image"]).copy()

    # the first read tracked the injected bunch: tracking it again changes nothing
    twin_model.simulator.track()
    twin_model.update_state()
    numpy.testing.assert_array_equal(
        numpy.asarray(stage.get(["TCPhosphor_image"])["TCPhosphor_image"]), image
    )


def test_switching_the_lpa_run_changes_the_twins_source(
    simulator, twin_model, tmp_path
):
    from downramp_actions import make_actions
    from lume_fbpic.actions import make_descriptor_actions
    from lume_fbpic_twin.archive_selector import (
        ArchiveSelector,
        particles_from_descriptor,
    )

    models = {}
    expected = {}
    for name, uz, charge_pc in (("low", 100.0, 200.0), ("high", 200.0, 400.0)):
        pg = lab_snapshot(uz_mean=uz)
        pg = ParticleGroup(
            data={**pg.data, "weight": numpy.full(len(pg), charge_pc * 1e-12 / len(pg))}
        )
        descriptor = reference_descriptor(pg, uz_min=None, central_fraction=None)
        # what the twin is fed: the selector's synthetic bunch of the recorded descriptor
        expected[name] = (
            descriptor,
            beam_moments(particles_from_descriptor(descriptor, n_particles=4000)),
        )
        source = LUMEFBPICModel(
            simulator,
            [*make_actions(simulator), *make_descriptor_actions()],
            dummy_run=True,
        )
        source.simulator.recorded_outputs = {
            f"descriptor_{k}": v for k, v in descriptor.items()
        }
        archive.save(source, tmp_path / f"{name}.h5")
        models[name] = archive.load(tmp_path / f"{name}.h5", dummy_run=True)
    chain = build_chain(
        ArchiveSelector(models, synthetic_bunch_particles=4000), twin_model
    )
    stage = chain.lume_model_instances[1]

    def assert_source_is(name):
        descriptor, moments = expected[name]
        assert chain.get(["LPA_Archive"])["LPA_Archive"] == name
        assert len(stage.initial_particles) == 4000  # the selected bunch: no second cut
        # the twin holds its source parameters as float32
        assert _get(stage, "Source_Charge_pC") == pytest.approx(
            moments["charge_pc"], rel=1e-7
        )
        assert _get(stage, "Source_Energy_MeV") == pytest.approx(
            moments["energy_mev"], rel=1e-7
        )
        assert (
            chain.get(["descriptor_total_beam_charge_pc"])[
                "descriptor_total_beam_charge_pc"
            ]
            == descriptor["total_beam_charge_pc"]
        )
        assert stage.external_beam is True

    try:
        chain.set({"EMQ1H_Current": 0.7})
        assert_source_is("low")

        chain.set({"LPA_Archive": "high"})  # only the selector: the twin must follow

        assert_source_is("high")
    finally:
        stage.reset()


def test_binned_geometries_keep_the_field_of_view_and_coarsen_the_pixels():
    from htu.lattice import DEFAULT_SCREEN_GEOMETRY_OVERRIDES, SCREEN_NAMES
    from htu.screens import DEFAULT_GEOMETRY

    geometries = binned_screen_geometries(4)

    assert set(geometries) == set(SCREEN_NAMES)
    for name, geometry in geometries.items():
        base = DEFAULT_SCREEN_GEOMETRY_OVERRIDES.get(name, DEFAULT_GEOMETRY)
        assert (geometry.size_x, geometry.size_y) == (
            base.size_x // 4,
            base.size_y // 4,
        )
        assert geometry.calibration_m == pytest.approx(base.calibration_m * 4)
        assert geometry.size_x * geometry.calibration_m == pytest.approx(
            base.size_x * base.calibration_m
        )
    assert geometries["ChicaneSlit"].calibration_m == pytest.approx(
        80e-6
    )  # its own 20 um x 4


def test_a_binning_that_is_not_a_positive_divisor_is_refused():
    for factor in (0, -2, 1.5, 3):
        with pytest.raises(ValueError):
            binned_screen_geometries(factor)


def test_a_given_model_cannot_be_binned(twin_model):
    with pytest.raises(ValueError, match="builds the twin model"):
        TwinStage(twin_model, screen_binning=4)


def test_a_binned_stage_serves_smaller_images_with_the_same_charge(stage):
    binned = TwinStage(screen_binning=4)
    try:
        pg = lab_snapshot()
        stage.initial_particles = pg
        binned.initial_particles = pg

        full = numpy.asarray(stage.get(["TCPhosphor_image"])["TCPhosphor_image"])
        coarse = numpy.asarray(binned.get(["TCPhosphor_image"])["TCPhosphor_image"])

        assert full.shape == (1024, 1024) and coarse.shape == (256, 256)
        assert coarse.sum() == pytest.approx(
            full.sum(), rel=1e-3
        )  # same charge in view
        occupied = lambda image: numpy.count_nonzero(image)  # noqa: E731
        assert occupied(coarse) < occupied(full)  # fewer, fuller pixels
        assert coarse.max() > full.max()
    finally:
        binned.reset()
