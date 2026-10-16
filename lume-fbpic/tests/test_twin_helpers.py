"""Tests for the HTU-twin helpers in `lume_fbpic_twin.bunch_frame` and `lume_fbpic_twin.twin_stage`: the
bunch frame and its ballistic drift, bunch moments and the plasma-exit plane. They need neither `htu` nor `lume-pva`, but `twin_stage` imports torch,
so they are skipped without it."""

from __future__ import annotations

import numpy
import pytest
from scipy.constants import c, e
from lume_fbpic.utils import ELECTRON_MC2_EV

from bunches import lab_snapshot
from inversion_fbpic.utils import distributions
from reference_descriptor import phase_space

pytest.importorskip("torch")
pytest.importorskip("lume_fbpic_twin")

from lume_fbpic_twin.bunch_frame import (  # noqa: E402
    bunch_frame_particles,
    to_bunch_frame,
)
from lume_fbpic_twin.twin_stage import beam_moments, plasma_exit_z  # noqa: E402

from beamphysics import ParticleGroup  # noqa: E402


def _twiss_bunch(
    n=200_000,
    energy_mev=100.0,
    spread=0.05,
    beta=5.0e-3,
    alpha=0.4,
    emit=2.0e-9,
    charge_pc=30.0,
):
    """A Gaussian bunch with the given Twiss in both planes (geometric emittance `emit`)."""
    rng = numpy.random.default_rng(1)
    energy = energy_mev * 1.0e6 * (1.0 + spread * rng.normal(size=n))
    p0c = numpy.sqrt((energy_mev * 1.0e6) ** 2 - ELECTRON_MC2_EV**2)
    data = {
        "species": "electron",
        "t": numpy.zeros(n),
        "status": numpy.ones(n, dtype=int),
    }
    for plane in ("x", "y"):
        a, b = rng.normal(size=n), rng.normal(size=n)
        data[plane] = numpy.sqrt(emit * beta) * a
        data[f"p{plane}"] = numpy.sqrt(emit / beta) * (b - alpha * a) * p0c
    data["z"] = numpy.full(n, 0.003) + 4.0e-6 * rng.normal(size=n)
    data["pz"] = numpy.sqrt(
        energy**2 - ELECTRON_MC2_EV**2 - data["px"] ** 2 - data["py"] ** 2
    )
    data["weight"] = numpy.full(n, charge_pc * 1.0e-12 / n)
    return ParticleGroup(data=data)


def test_moments_are_the_bunchs_statistics():
    pg = _twiss_bunch(n=20_000, beta=5.0e-3, alpha=0.4, emit=2.0e-9)

    moments = beam_moments(pg)

    # Energy, spread, charge and centroids are beamphysics's own statistics. Twiss values are the
    # charge-weighted covariance of (x, px / p0c), not beamphysics's dispersion-corrected
    # `twiss()`; a bunch made with alpha = 0.4 is converging, alpha > 0 in the MAD convention.
    energy = pg["mean_energy"]
    p0c = numpy.sqrt(energy**2 - ELECTRON_MC2_EV**2)
    expected = {
        "energy_mev": energy / 1e6,
        "energy_spread_pct": 100.0 * pg["sigma_energy"] / energy,
        "charge_pc": pg.charge * 1e12,
        "num_particles": 20_000,
    }
    for plane in ("x", "y"):
        position, angle = getattr(pg, plane), getattr(pg, f"p{plane}") / p0c
        covariance = numpy.cov(position, angle, aweights=pg.weight, bias=True)
        emittance = numpy.sqrt(numpy.linalg.det(covariance))
        expected |= {
            f"beta_{plane}_mm": covariance[0, 0] / emittance * 1e3,
            f"alpha_{plane}": -covariance[0, 1] / emittance,
            f"norm_emit_{plane}_um": emittance * p0c / ELECTRON_MC2_EV * 1e6,
            f"{plane}_um": pg[f"mean_{plane}"] * 1e6,
            f"{plane}p_mrad": pg[f"mean_p{plane}"] / p0c * 1e3,
        }
        assert moments[f"alpha_{plane}"] > 0
    assert moments == pytest.approx(expected, rel=1e-9)


def test_moments_report_centroid_and_pointing():
    pg = _twiss_bunch(n=20_000)
    p0c = numpy.sqrt(pg["mean_energy"] ** 2 - ELECTRON_MC2_EV**2)
    shifted = ParticleGroup(
        data={
            **pg.data,
            "x": numpy.asarray(pg.x) + 50.0e-6,
            "px": numpy.asarray(pg.px) + 1.0e-3 * p0c,
        }
    )

    moments, before = beam_moments(shifted), beam_moments(pg)

    assert moments["x_um"] - before["x_um"] == pytest.approx(50.0, rel=1e-9)
    # the kick raises the energy too, so the angle is over the shifted bunch's p0c
    shifted_p0c = numpy.sqrt(shifted["mean_energy"] ** 2 - ELECTRON_MC2_EV**2)
    assert moments["xp_mrad"] == pytest.approx(
        (pg["mean_px"] + 1.0e-3 * p0c) / shifted_p0c * 1e3, rel=1e-9
    )
    assert moments["y_um"] == before["y_um"]


def test_a_zero_emittance_plane_has_zero_twiss():
    pg = _twiss_bunch(n=100, emit=0.0)

    moments = beam_moments(pg)

    assert moments["beta_x_mm"] == 0.0 and moments["alpha_x"] == 0.0
    assert moments["norm_emit_x_um"] == 0.0


def test_moments_need_at_least_two_charged_particles():
    one = ParticleGroup(
        data={
            "x": numpy.zeros(1),
            "y": numpy.zeros(1),
            "z": numpy.zeros(1),
            "px": numpy.zeros(1),
            "py": numpy.zeros(1),
            "pz": numpy.full(1, 1e8),
            "t": numpy.zeros(1),
            "status": numpy.ones(1, dtype=int),
            "weight": numpy.ones(1) * 1e-15,
            "species": "electron",
        }
    )
    with pytest.raises(ValueError, match="at least two"):
        beam_moments(one)


@pytest.mark.parametrize("length", [0.052, -0.00088])
def test_a_drift_matches_cheetahs_drift_kick_drift(length):
    import torch

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


def test_plasma_exit_is_the_largest_profile_end():
    class Profile:
        def __init__(self, extent):
            self._extent = extent

        def get_z_extent(self):
            return self._extent

    assert plasma_exit_z([Profile((0.0, 1.0e-3)), Profile((0.5e-3, 5.4e-3))]) == 5.4e-3


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
