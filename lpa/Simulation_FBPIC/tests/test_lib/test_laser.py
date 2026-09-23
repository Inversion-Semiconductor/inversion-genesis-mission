"""
Tests for laser profile physics: extents, energy/a0 round-trips,
build_laser_profile polarization variants, and YAML round-trips.

Energy/a0 XOR validation tests live in ``test_serializable_config.py``.

Run from Simulation_FBPIC::

    pytest tests/test_lib/test_laser.py -v
"""

from __future__ import annotations

from math import pi

import numpy as np
import pytest
from scipy.constants import c, epsilon_0

LASER_BASE_KWARGS: dict = {
    "z0": -3.0e-5,
    "wavelength": 8.0e-7,
    "tau_fwhm": 3.8e-14,
    "cep": 0.0,
    "waist": 2.8e-5,
    "focal_position": 3.0e-3,
    "polarization": 0.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_laser(**overrides):
    from inversion_fbpic.lib.laser import GaussianLaserPulse

    kw = {**LASER_BASE_KWARGS, **overrides}
    if "energy" not in kw and "a0" not in kw:
        kw["energy"] = 5.0
    return GaussianLaserPulse(**kw)


def _integrate_laser_energy_from_fields(
    pulse,
    *,
    n_r: int = 128,
    n_z: int = 256,
    n_t_optical: int = 32,
    num_sigma: float = 6.0,
) -> float:
    """Integrate cycle-averaged EM energy density from ``E_field`` profiles.

    At each spatial sample point the squared field magnitude is averaged over
    one optical cycle, then integrated over cylindrical volume using
    ``epsilon_0 * <E^2>`` (total electric plus magnetic energy in vacuum).
    """
    from inversion_fbpic.lib.laser import _ensure_profile_list

    profiles = _ensure_profile_list(pulse.build_laser_profile())
    z_min, z_max = pulse.get_z_extent(num_sigma=num_sigma)
    r_max = pulse.get_r_extent((z_min, z_max), num_sigma=num_sigma)

    r = np.linspace(0.0, r_max, n_r)
    z = np.linspace(z_min, z_max, n_z)
    dr = float(np.diff(r).mean())
    dz = float(np.diff(z).mean())
    optical_period = pulse.wavelength / c
    t_samples = np.linspace(0.0, optical_period, n_t_optical, endpoint=False)

    r_grid, z_grid = np.meshgrid(r, z, indexing="ij")
    x_grid = r_grid
    y_grid = np.zeros_like(r_grid)

    e2_sum = np.zeros_like(r_grid)
    for ti in t_samples:
        ex = np.zeros_like(r_grid)
        ey = np.zeros_like(r_grid)
        for profile in profiles:
            ex_p, ey_p = profile.E_field(x=x_grid, y=y_grid, z=z_grid, t=ti)
            ex += ex_p
            ey += ey_p
        e2_sum += ex**2 + ey**2
    e2_avg = e2_sum / n_t_optical

    volume_element = 2.0 * pi * r_grid * dr * dz
    volume_element[0, :] = pi * dr**2 / 4.0 * dz

    return float(epsilon_0 * np.sum(e2_avg * volume_element))


# Well-behaved laser parameters: long pulse and wide waist vs wavelength.
_ENERGY_INTEGRATION_KWARGS: dict = {
    "z0": 0.0,
    "wavelength": 8.0e-7,
    "tau_fwhm": 200.0e-15,
    "cep": 0.0,
    "waist": 50.0e-6,
    "focal_position": 0.0,
}

# ===================================================================
# Energy / a0 round-trip consistency
# ===================================================================


class TestEnergyA0:
    def test_energy_to_a0_round_trip(self) -> None:
        pulse = _make_laser(energy=5.0)
        from inversion_fbpic.lib.laser import GaussianLaserPulse

        pulse2 = GaussianLaserPulse(a0=pulse.a0, **LASER_BASE_KWARGS)
        assert pulse2.energy == pytest.approx(5.0, rel=1e-6)

    def test_a0_to_energy_round_trip(self) -> None:
        pulse = _make_laser(a0=2.0)
        from inversion_fbpic.lib.laser import GaussianLaserPulse

        pulse2 = GaussianLaserPulse(energy=pulse.energy, **LASER_BASE_KWARGS)
        assert pulse2.a0 == pytest.approx(2.0, rel=1e-6)

    def test_negative_energy_raises(self) -> None:
        from inversion_fbpic.lib.laser import GaussianLaserPulse

        with pytest.raises(ValueError, match="energy must be > 0"):
            GaussianLaserPulse(energy=-1.0, **LASER_BASE_KWARGS)

    def test_negative_a0_raises(self) -> None:
        from inversion_fbpic.lib.laser import GaussianLaserPulse

        with pytest.raises(ValueError, match="a0 must be > 0"):
            GaussianLaserPulse(a0=-1.0, **LASER_BASE_KWARGS)


# ===================================================================
# Field-intensity energy integration
# ===================================================================


class TestEnergyFromFieldIntegration:
    @pytest.mark.parametrize(
        ("polarization", "label"),
        [
            (0.0, "linear"),
            (pi / 4, "linear_45deg"),
            ("left", "left_circular"),
            ("right", "right_circular"),
            ([pi / 4, pi / 2], "elliptical"),
        ],
    )
    def test_energy_from_field_integration(self, polarization, label) -> None:
        target_energy = 5.0
        pulse = _make_laser(
            energy=target_energy,
            polarization=polarization,
            **_ENERGY_INTEGRATION_KWARGS,
        )
        integrated_energy = _integrate_laser_energy_from_fields(pulse)
        assert integrated_energy == pytest.approx(target_energy, rel=2e-2), label


# ===================================================================
# Longitudinal extent
# ===================================================================


class TestGetZExtent:
    def test_symmetric_about_z0(self) -> None:
        pulse = _make_laser(energy=5.0)
        z_min, z_max = pulse.get_z_extent()
        center = (z_min + z_max) / 2.0
        assert center == pytest.approx(pulse.z0, rel=1e-6)

    def test_wider_with_more_sigma(self) -> None:
        pulse = _make_laser(energy=5.0)
        extent_3 = pulse.get_z_extent(num_sigma=3.0)
        extent_5 = pulse.get_z_extent(num_sigma=5.0)
        width_3 = extent_3[1] - extent_3[0]
        width_5 = extent_5[1] - extent_5[0]
        assert width_5 > width_3

    def test_scales_with_tau_fwhm(self) -> None:
        pulse_short = _make_laser(energy=5.0, tau_fwhm=3.0e-14)
        pulse_long = _make_laser(energy=5.0, tau_fwhm=6.0e-14)
        width_short = pulse_short.get_z_extent()[1] - pulse_short.get_z_extent()[0]
        width_long = pulse_long.get_z_extent()[1] - pulse_long.get_z_extent()[0]
        assert width_long > width_short


# ===================================================================
# Radial extent
# ===================================================================


class TestGetRExtent:
    def test_positive(self) -> None:
        pulse = _make_laser(energy=5.0)
        sim_extent = (-1e-4, 0.0)
        r_ext = pulse.get_r_extent(sim_extent)
        assert r_ext > 0.0

    def test_wider_when_focal_far_from_box(self) -> None:
        pulse_near = _make_laser(energy=5.0, focal_position=0.0)
        pulse_far = _make_laser(energy=5.0, focal_position=0.01)
        sim_extent = (-1e-4, 0.0)
        r_near = pulse_near.get_r_extent(sim_extent)
        r_far = pulse_far.get_r_extent(sim_extent)
        assert r_far > r_near

    def test_rayleigh_scaling(self) -> None:
        pulse = _make_laser(energy=5.0)
        rayleigh = pi * pulse.waist**2 / pulse.wavelength
        assert rayleigh > 0.0
        r_ext = pulse.get_r_extent((-1e-4, 0.0))
        assert r_ext > pulse.waist / 2.0


# ===================================================================
# build_laser_profile — polarization variants
# ===================================================================


class TestBuildLaserProfile:
    def test_linear_returns_single(self) -> None:
        pulse = _make_laser(energy=5.0, polarization=0.0)
        profile = pulse.build_laser_profile()
        from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser

        assert isinstance(profile, GaussianLaser)

    def test_linear_45_returns_single(self) -> None:
        pulse = _make_laser(energy=5.0, polarization=pi / 4)
        profile = pulse.build_laser_profile()
        from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser

        assert isinstance(profile, GaussianLaser)

    def test_left_circular_returns_two_profiles(self) -> None:
        pulse = _make_laser(energy=5.0, polarization="left")
        profiles = pulse.build_laser_profile()
        assert isinstance(profiles, list)
        assert len(profiles) == 2
        total_E0_sq = sum(p.E0x**2 + p.E0y**2 for p in profiles)
        assert total_E0_sq > 0

    def test_right_circular_returns_two_profiles(self) -> None:
        pulse = _make_laser(energy=5.0, polarization="right")
        profiles = pulse.build_laser_profile()
        assert isinstance(profiles, list)
        assert len(profiles) == 2
        total_E0_sq = sum(p.E0x**2 + p.E0y**2 for p in profiles)
        assert total_E0_sq > 0

    def test_elliptical_returns_two_profiles(self) -> None:
        pulse = _make_laser(energy=5.0, polarization=[pi / 4, pi / 2])
        profiles = pulse.build_laser_profile()
        assert isinstance(profiles, list)
        assert len(profiles) == 2

    def test_profile_has_correct_waist(self) -> None:
        pulse = _make_laser(energy=5.0, polarization=0.0)
        profile = pulse.build_laser_profile()
        assert profile.transverse_profile.w0 == pytest.approx(pulse.waist)


# ===================================================================
# YAML round-trip
# ===================================================================


class TestLaserYamlRoundTrip:
    def test_energy_based_round_trip(self) -> None:
        pulse = _make_laser(energy=5.0)
        yaml_str = pulse.to_yaml(comments=False)

        from inversion_fbpic.lib.laser import GaussianLaserPulse

        reloaded = GaussianLaserPulse.from_yaml(yaml_str)
        assert reloaded.energy == pytest.approx(pulse.energy, rel=1e-6)
        assert reloaded.a0 == pytest.approx(pulse.a0, rel=1e-6)

    def test_a0_based_round_trip(self) -> None:
        pulse = _make_laser(a0=2.0)
        yaml_str = pulse.to_yaml(comments=False)

        from inversion_fbpic.lib.laser import GaussianLaserPulse

        reloaded = GaussianLaserPulse.from_yaml(yaml_str)
        assert reloaded.a0 == pytest.approx(pulse.a0, rel=1e-6)
        assert reloaded.energy == pytest.approx(pulse.energy, rel=1e-6)

    def test_circular_polarization_round_trip(self) -> None:
        pulse = _make_laser(energy=5.0, polarization="left")
        yaml_str = pulse.to_yaml(comments=False)

        from inversion_fbpic.lib.laser import GaussianLaserPulse

        reloaded = GaussianLaserPulse.from_yaml(yaml_str)
        assert reloaded.polarization == "left"

    def test_elliptical_polarization_round_trip(self) -> None:
        pulse = _make_laser(energy=5.0, polarization=[pi / 4, pi / 2])
        yaml_str = pulse.to_yaml(comments=False)

        from inversion_fbpic.lib.laser import GaussianLaserPulse

        reloaded = GaussianLaserPulse.from_yaml(yaml_str)
        assert reloaded.polarization == pytest.approx([pi / 4, pi / 2])


# ===================================================================
# LasyLaserPulse stub
# ===================================================================


class TestLasyLaserPulse:
    def test_registered_in_concrete_registry(self) -> None:
        from inversion_fbpic.lib.laser import _LaserPulse

        assert "lasy" in _LaserPulse._CONCRETE_REGISTRY

    def test_init_raises(self) -> None:
        from inversion_fbpic.lib.laser import LasyLaserPulse

        with pytest.raises((NotImplementedError, TypeError)):
            LasyLaserPulse(lasy_profile=None)


# ===================================================================
# _format_polarization helper
# ===================================================================


class TestFormatPolarization:
    def test_linear(self) -> None:
        from inversion_fbpic.lib.laser import _format_polarization

        result = _format_polarization(0.0)
        assert "linear" in result

    def test_circular(self) -> None:
        from inversion_fbpic.lib.laser import _format_polarization

        assert "circular" in _format_polarization("left")
        assert "circular" in _format_polarization("right")

    def test_elliptical(self) -> None:
        from inversion_fbpic.lib.laser import _format_polarization

        result = _format_polarization([pi / 4, pi / 2])
        assert "elliptical" in result
