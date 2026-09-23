"""Shared fixtures for tests/test_lib/."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def yaml_out(tmp_path: Path) -> Path:
    return tmp_path / "yaml_out"


# ---------------------------------------------------------------------------
# Minimal constructor kwargs for density profiles
# ---------------------------------------------------------------------------

_COMMON_DENSITY_KWARGS: dict = {
    "nominal_density": 1.0e24,
    "p_nz": 1,
    "p_nr": 1,
    "p_nt": 1,
}

EXAMPLE_DENSITY_KWARGS: dict = {
    **_COMMON_DENSITY_KWARGS,
    "length": 1.0e-6,
    "start_position": 0.0,
}

ASYMMETRIC_SINE_KWARGS: dict = {
    **_COMMON_DENSITY_KWARGS,
    "peak_z0": 2.5e-3,
    "upramp_length": 2.5e-3,
    "downramp_length": 1.5e-3,
}

SMOOTH_SINE_FLATTOP_KWARGS: dict = {
    **_COMMON_DENSITY_KWARGS,
    "flattop_width": 1.0e-3,
    "upramp_length": 0.5e-3,
    "downramp_length": 0.5e-3,
    "offset_length": 0.0,
}

GAUSSIAN_PLUS_TRIANGLE_KWARGS: dict = {
    **_COMMON_DENSITY_KWARGS,
    "gauss_sigma": 3.0e-6,
    "gauss_z0": 10.0e-6,
    "tri_z0": 15.0e-6,
    "tri_left_width": 5.0e-6,
    "tri_right_width": 5.0e-6,
    "tri_height": 1.0,
}

GENERALIZED_GAUSSIAN_PLUS_TRIANGLE_KWARGS: dict = {
    **_COMMON_DENSITY_KWARGS,
    "gauss_peak": 1.0,
    "gauss_alpha": 3.0e-6,
    "gauss_beta": 2.0,
    "gauss_z0": 10.0e-6,
    "tri_z0": 15.0e-6,
    "tri_left_width": 5.0e-6,
    "tri_right_width": 5.0e-6,
    "tri_height": 1.0,
}

# ---------------------------------------------------------------------------
# Minimal constructor kwargs for lasers
# ---------------------------------------------------------------------------

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
# Minimal hyperparameters
# ---------------------------------------------------------------------------

HYPERPARAMETERS_KWARGS: dict = {
    "zmin": -1.0e-5,
    "zmax": 0.0,
    "rmax": 1.0e-5,
    "nz": 8,
    "nr": 8,
    "nm": 1,
    "use_mpi": False,
    "number_dumps": 2,
}


@pytest.fixture()
def minimal_hyperparameters():
    from inversion_fbpic.lib.simulation import SimulationHyperparameters

    return SimulationHyperparameters(**HYPERPARAMETERS_KWARGS)


@pytest.fixture()
def minimal_density():
    from inversion_fbpic.lib.density_profiles import ExampleDensityProfile

    return ExampleDensityProfile(**EXAMPLE_DENSITY_KWARGS)


@pytest.fixture()
def minimal_laser():
    from inversion_fbpic.lib.laser import GaussianLaserPulse

    return GaussianLaserPulse(energy=5.0, **LASER_BASE_KWARGS)


@pytest.fixture()
def minimal_simulation_elements(
    minimal_hyperparameters, minimal_density, minimal_laser
):
    """Return a Simulation assembled from minimal in-memory objects."""
    from inversion_fbpic.lib.simulation import Simulation

    return Simulation(
        elements=[minimal_hyperparameters, minimal_density, minimal_laser]
    )


@pytest.fixture(autouse=True)
def _matplotlib_agg():
    """Force the non-interactive Agg backend so plot tests never pop a window."""
    import matplotlib

    matplotlib.use("Agg")
