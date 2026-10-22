"""Tests for `lume_fbpic_twin.bunch_frame`: the bunch frame and its ballistic drift. They need
neither `htu`, `lume-pva` nor torch, so they run wherever `lume_fbpic_twin` is installed; the
comparison with Cheetah's drift is skipped without torch and Cheetah."""

from __future__ import annotations

import numpy
import pytest
from scipy.constants import c, e

from bunches import lab_snapshot
from inversion_fbpic.utils import distributions
from reference_descriptor import phase_space

pytest.importorskip("lume_fbpic_twin")

from lume_fbpic_twin.bunch_frame import (  # noqa: E402
    bunch_frame_particles,
    to_bunch_frame,
)


@pytest.mark.parametrize("length", [0.052, -0.00088])
def test_a_drift_matches_cheetahs_drift_kick_drift(length):
    torch = pytest.importorskip("torch")
    cheetah = pytest.importorskip("cheetah")
    pg = bunch_frame_particles(lab_snapshot(), uz_min=30.0, central_fraction=None)
    energy = float(numpy.average(pg.energy, weights=pg.weight))
    beam = cheetah.ParticleBeam.from_openpmd_particlegroup(
        pg, energy=torch.tensor(energy, dtype=torch.float64), dtype=torch.float64
    )
    out = cheetah.Drift(
        length=torch.tensor(length, dtype=torch.float64),
        tracking_method="drift_kick_drift",
    ).track(beam)

    drifted = to_bunch_frame(pg, pg.z[0] + length)

    numpy.testing.assert_allclose(out.x.numpy(), numpy.asarray(drifted.x), atol=1e-15)
    numpy.testing.assert_allclose(out.y.numpy(), numpy.asarray(drifted.y), atol=1e-15)
    # Cheetah's tau is relative to its reference particle, the bunch frame's t absolute: compare
    # each about its mean
    cheetah_dtau = out.tau.numpy() - beam.tau.numpy()
    cheetah_dtau -= numpy.average(cheetah_dtau, weights=pg.weight)
    ours_dtau = c * (numpy.asarray(drifted.t) - numpy.asarray(pg.t))
    ours_dtau -= numpy.average(ours_dtau, weights=pg.weight)
    numpy.testing.assert_allclose(cheetah_dtau, ours_dtau, atol=1e-14)


def test_a_drift_adds_each_particles_time_of_flight():
    pg = bunch_frame_particles(lab_snapshot(), uz_min=30.0, central_fraction=None)
    length = 0.05

    out = to_bunch_frame(pg, pg.z[0] + length)

    # time of flight L / v_z = L E / (pz c); angle px / pz
    flight = length * numpy.asarray(pg.energy) / numpy.asarray(pg.pz) / c
    numpy.testing.assert_allclose(
        numpy.asarray(out.t) - numpy.asarray(pg.t), flight, rtol=1e-12
    )
    numpy.testing.assert_allclose(
        numpy.asarray(out.x),
        numpy.asarray(pg.x) + length * numpy.asarray(pg.px) / numpy.asarray(pg.pz),
        rtol=1e-12,
        atol=1e-18,
    )
    assert numpy.all(numpy.asarray(out.z) == pg.z[0] + length)


def test_a_bunch_on_the_plane_already_is_unchanged():
    pg = bunch_frame_particles(lab_snapshot(), uz_min=30.0, central_fraction=None)

    out = to_bunch_frame(pg, pg.z[0])

    for name in ("x", "y", "z", "t", "px", "py", "pz", "weight", "status"):
        assert numpy.array_equal(
            numpy.asarray(getattr(out, name)), numpy.asarray(getattr(pg, name))
        ), name


def test_drifting_forward_then_back_restores_the_bunch():
    pg = bunch_frame_particles(lab_snapshot(), uz_min=30.0, central_fraction=None)

    out = to_bunch_frame(to_bunch_frame(pg, pg.z[0] + 0.05), pg.z[0])

    numpy.testing.assert_allclose(numpy.asarray(out.x), numpy.asarray(pg.x), atol=1e-15)
    numpy.testing.assert_allclose(numpy.asarray(out.t), numpy.asarray(pg.t), atol=1e-24)
    assert numpy.array_equal(numpy.asarray(out.z), numpy.asarray(pg.z))


def test_a_drift_leaves_momenta_charge_and_status_alone_and_the_input_unchanged():
    pg = bunch_frame_particles(lab_snapshot(), uz_min=30.0, central_fraction=None)
    before = pg.copy()

    out = to_bunch_frame(pg, pg.z[0] + 0.01)

    for name in ("px", "py", "pz", "weight", "status"):
        assert numpy.array_equal(
            numpy.asarray(getattr(out, name)), numpy.asarray(getattr(pg, name))
        )
    for name in ("x", "y", "z", "t"):
        assert numpy.array_equal(
            numpy.asarray(getattr(pg, name)), numpy.asarray(getattr(before, name))
        )


def test_a_particle_that_does_not_move_forward_is_refused():
    with pytest.raises(ValueError, match="pz <= 0"):
        to_bunch_frame(lab_snapshot())  # it has particles with uz < 0


def test_the_bunch_frame_puts_a_snapshot_on_its_mean_z_with_arrival_times():
    pg = lab_snapshot()
    pg = pg[numpy.asarray(pg.pz) > 0]

    out = bunch_frame_particles(pg, uz_min=None, central_fraction=None)

    # each particle drifts from its z to the mean, (z_mean - z) E / (pz c) in time from the
    # snapshot's t = 0: one ahead of the mean arrives first, at a negative t
    z, energy, pz = (numpy.asarray(getattr(pg, n)) for n in ("z", "energy", "pz"))
    z_mean = numpy.average(z, weights=pg.weight)
    assert numpy.all(numpy.asarray(out.z) == z_mean)
    numpy.testing.assert_allclose(
        numpy.asarray(out.t), (z_mean - z) * energy / pz / c, rtol=1e-9, atol=1e-27
    )
    numpy.testing.assert_allclose(
        numpy.asarray(out.x),
        numpy.asarray(pg.x) + (z_mean - z) * numpy.asarray(pg.px) / pz,
        rtol=1e-12,
        atol=1e-21,
    )
    for name in ("px", "py", "pz", "weight"):
        assert numpy.array_equal(
            numpy.asarray(getattr(out, name)), numpy.asarray(getattr(pg, name))
        ), name
    assert numpy.all(numpy.asarray(out.status) == 1)
    assert out.species == "electron"


def test_selection_matches_the_descriptor_library():
    pg = lab_snapshot()
    expected, expected_w = distributions.crop_central_particles(
        *distributions.select_by_uz(
            phase_space(pg), numpy.asarray(pg.weight) / e, uz_min=120.0
        ),
        central_fraction=0.9,
    )

    out = bunch_frame_particles(pg, uz_min=120.0, central_fraction=0.9)

    # the momenta are the selection's; the positions are moved onto the bunch's plane
    momenta = [distributions.COORD_NAMES.index(n) for n in ("ux", "uy", "uz")]
    numpy.testing.assert_array_equal(phase_space(out)[:, momenta], expected[:, momenta])
    numpy.testing.assert_array_equal(numpy.asarray(out.weight) / e, expected_w)


def test_selection_that_leaves_too_few_particles_is_an_error():
    with pytest.raises(ValueError, match="fewer than two"):
        bunch_frame_particles(lab_snapshot(), uz_min=1.0e6)


def test_central_fraction_must_be_a_fraction():
    with pytest.raises(ValueError, match="central_fraction"):
        bunch_frame_particles(lab_snapshot(), central_fraction=1.5)
