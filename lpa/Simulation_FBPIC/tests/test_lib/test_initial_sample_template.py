"""Tests for the initial ionization-injection simulation template."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def template_module():
    path = (
        Path(__file__).parents[2]
        / "runs/initial_sample/ionization_injection_template.py"
    )
    spec = importlib.util.spec_from_file_location("initial_sample_template", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_simulation_length_preserves_configured_density_bounds(template_module):
    cls = template_module.IonizationInjectionSimulation
    physical = {key: 0.0 for key in cls._PHYSICAL_PARAMETER_KEYS}
    physical.update(
        peak_neutral_density_cm3=1e18,
        nitrogen_dopant_fraction=0.03,
        density_fwhm_left_m=1e-3,
        density_fwhm_right_m=2e-3,
        density_left_ramp_edge_percent=0.5,
        density_right_ramp_edge_percent=0.5,
        density_center_location_m=2e-3,
        laser_energy_J=1.0,
        laser_wavelength_m=800e-9,
        laser_pulse_duration_fwhm_s=30e-15,
        laser_spot_size_m=20e-6,
        laser_focal_position_m=1e-3,
        laser_super_gaussian_order=2.0,
        laser_spectral_bandwidth_rad_s="auto",
        laser_cep_phase_rad=0.0,
    )

    simulation = cls(
        physical,
        {"simulation_length": 4e-3, "p_zmin": 0.5e-3, "p_zmax": 3.5e-3},
    ).build_simulation()

    assert simulation.hyparams.simulation_length == 4e-3
    assert [(density.p_zmin, density.p_zmax) for density in simulation.densities] == [
        (0.5e-3, 3.5e-3),
        (0.5e-3, 3.5e-3),
    ]
