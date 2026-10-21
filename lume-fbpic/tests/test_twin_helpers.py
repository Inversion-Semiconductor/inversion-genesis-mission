"""Tests for the HTU-twin helpers in `lume_fbpic_twin.twin_stage`: bunch moments and the
plasma-exit plane. They need neither `htu` nor `lume-pva`, but `twin_stage` imports torch, so they
are skipped without it. The bunch-frame tests are in `test_bunch_frame.py`."""

from __future__ import annotations

import numpy
import pytest
from lume_fbpic.utils import ELECTRON_MC2_EV

pytest.importorskip("torch")
pytest.importorskip("lume_fbpic_twin")

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


def test_plasma_exit_is_the_largest_profile_end():
    class Profile:
        def __init__(self, extent):
            self._extent = extent

        def get_z_extent(self):
            return self._extent

    assert plasma_exit_z([Profile((0.0, 1.0e-3)), Profile((0.5e-3, 5.4e-3))]) == 5.4e-3
