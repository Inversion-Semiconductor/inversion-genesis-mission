"""Tests for scalar-defined synthetic spectral laser pulses."""

from __future__ import annotations

import numpy as np
import pytest

from inversion_fbpic.utils.laser import (
    AnalyticSpectralLongitudinalProfile,
    HighOrderLasyLaser,
)


def test_spectral_profile_is_normalized_and_zero_padded() -> None:
    profile = AnalyticSpectralLongitudinalProfile(
        wavelength=800e-9,
        bandwidth_fwhm=1.0e14,
        time_half_width=100e-15,
        npoints=1024,
    )

    intensity = np.abs(profile.evaluate(profile.time_axis)) ** 2

    assert intensity.max() == pytest.approx(1.0)
    assert profile.evaluate(np.array([200e-15]))[0] == 0.0j


def test_gdd_broadens_the_synthesized_temporal_pulse() -> None:
    transform_limited = AnalyticSpectralLongitudinalProfile(
        wavelength=800e-9,
        bandwidth_fwhm=1.0e14,
        time_half_width=200e-15,
        npoints=4096,
    )
    chirped = AnalyticSpectralLongitudinalProfile(
        wavelength=800e-9,
        bandwidth_fwhm=1.0e14,
        time_half_width=200e-15,
        npoints=4096,
        gdd=2.0e-27,
    )

    time_squared = transform_limited.time_axis**2
    transform_limited_width = np.sqrt(
        np.average(time_squared, weights=np.abs(transform_limited.temporal_field) ** 2)
    )
    chirped_width = np.sqrt(
        np.average(time_squared, weights=np.abs(chirped.temporal_field) ** 2)
    )

    assert chirped_width > transform_limited_width


def test_auto_bandwidth_matches_transform_limited_gaussian() -> None:
    laser = HighOrderLasyLaser.__new__(HighOrderLasyLaser)
    laser.physical_parameters = {
        "laser_spectral_bandwidth_rad_s": "auto",
        "laser_pulse_duration_fwhm_s": 30e-15,
    }

    bandwidth = laser._resolve_spectral_bandwidth()

    assert bandwidth == pytest.approx(4.0 * np.log(2.0) / 30e-15)


def test_peak_delay_expands_temporal_grid() -> None:
    laser = HighOrderLasyLaser.__new__(HighOrderLasyLaser)
    laser.hyperparameters = {"peak_delay_from_file_start_s": 200e-15}

    time_half_width = laser._time_half_width_for_peak_delay(90e-15)

    assert time_half_width == pytest.approx(145e-15)


@pytest.mark.parametrize(
    ("duration", "relative_gdd", "relative_tod", "expected_gdd", "expected_tod"),
    [
        (30e-15, 1.0, -1.0, 5e-28, -1e-41),
        (50e-15, -0.5, 0.5, -0.5 * 5e-28 * (50 / 30) ** 2, 0.5e-41 * (50 / 30) ** 3),
    ],
)
def test_relative_spectral_phase_scales_with_pulse_duration(
    duration: float,
    relative_gdd: float,
    relative_tod: float,
    expected_gdd: float,
    expected_tod: float,
) -> None:
    laser = HighOrderLasyLaser.__new__(HighOrderLasyLaser)
    laser.physical_parameters = {
        "laser_pulse_duration_fwhm_s": duration,
        "laser_relative_gdd_s2": relative_gdd,
        "laser_relative_tod_s3": relative_tod,
        "laser_gdd_s2": 0.0,
        "laser_tod_s3": 0.0,
    }

    laser._resolve_relative_spectral_phase()

    assert laser.physical_parameters["laser_gdd_s2"] == pytest.approx(expected_gdd)
    assert laser.physical_parameters["laser_tod_s3"] == pytest.approx(expected_tod)