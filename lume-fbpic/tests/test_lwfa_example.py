"""The LWFA example, `docs/examples/lwfa.py`."""

from __future__ import annotations


def test_the_laser_duration_reproduces_the_originals_tau_exactly():
    from inversion_fbpic.utils.simulation_setup_tools import (
        calculate_laser_tau_from_fwhm_intensity,
    )
    import lwfa

    assert calculate_laser_tau_from_fwhm_intensity(lwfa.TAU_FWHM) == lwfa.UPSTREAM_TAU
