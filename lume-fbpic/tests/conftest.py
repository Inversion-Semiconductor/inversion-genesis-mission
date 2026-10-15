"""Shared fixtures for lume_fbpic tests.

Minimal fixtures (small grid) -- mirrors the style of
Simulation_FBPIC/tests/test_lib/conftest.py's minimal-kwargs fixtures. The `model` fixture uses `dummy_run=True`, so
`set()` never runs a real simulation, and these tests never pay real fbpic execution cost
unless they explicitly call `simulator.run()`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# fbpic's numba kernels use every core by default, and threaded charge and current deposition adds
# up in an order that depends on the thread count; one thread makes the runs repeat on any
# machine. The environment covers a numba not imported yet, `set_num_threads()` one that another
# package's tests already imported (the root `pytest.ini` collects every package).
os.environ["NUMBA_NUM_THREADS"] = "1"

import numba  # noqa: E402
import pytest  # noqa: E402

numba.set_num_threads(1)

# The example scripts (docs/examples), the twin scripts (docs/htu-twin/twin) and the display
# generators and their widgets (docs/htu-twin/phoebus) are scripts, not package modules. The twin
# modules are the package `lume_fbpic_twin` (docs/htu-twin).
_DOCS = Path(__file__).resolve().parents[1] / "docs"
for _directory in (
    _DOCS / "examples",
    _DOCS / "htu-twin" / "phoebus",
    _DOCS / "htu-twin" / "twin",
):
    sys.path.insert(0, str(_directory))

from inversion_fbpic.lib.simulation import SimulationHyperparameters  # noqa: E402
from inversion_fbpic.lib.laser import GaussianLaserPulse  # noqa: E402
from inversion_fbpic.lib.density_profiles import SmoothSineFlattop  # noqa: E402

from downramp_actions import make_actions  # noqa: E402
from lume_fbpic.model import LUMEFBPICModel  # noqa: E402
from lume_fbpic.lwfa_simulator import LWFASimulator  # noqa: E402

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

LASER_KWARGS: dict = {
    "z0": -3.0e-5,
    "wavelength": 8.0e-7,
    "tau_fwhm": 3.8e-14,
    "cep": 0.0,
    "waist": 2.8e-5,
    "focal_position": 3.0e-3,
    "polarization": 0.0,
}

FLATTOP_KWARGS: dict = {
    "nominal_density": 1.0e24,
    "p_nz": 1,
    "p_nr": 1,
    "p_nt": 1,
    "flattop_width": 1.0e-3,
    "upramp_length": 0.5e-3,
    "downramp_length": 0.5e-3,
    "elec_name": "electrons_flattop",
}

DOWNRAMP_KWARGS: dict = {
    "nominal_density": 0.3e24,
    "p_nz": 1,
    "p_nr": 1,
    "p_nt": 1,
    "flattop_width": 1.0e-3,
    "upramp_length": 0.5e-3,
    "downramp_length": 0.5e-3,
    "elec_name": "electrons_downramp",
}


@pytest.fixture()
def hyparams() -> SimulationHyperparameters:
    return SimulationHyperparameters(**HYPERPARAMETERS_KWARGS)


@pytest.fixture()
def laser() -> GaussianLaserPulse:
    return GaussianLaserPulse(energy=5.0, **LASER_KWARGS)


@pytest.fixture()
def densities() -> list[SmoothSineFlattop]:
    return [SmoothSineFlattop(**FLATTOP_KWARGS), SmoothSineFlattop(**DOWNRAMP_KWARGS)]


@pytest.fixture()
def simulator(hyparams, laser, densities, tmp_path) -> LWFASimulator:
    return LWFASimulator(
        hyparams,
        laser,
        densities,
        target_species="electrons_flattop",
        working_directory=tmp_path,
    )


@pytest.fixture()
def model(simulator) -> LUMEFBPICModel:
    return LUMEFBPICModel(simulator, make_actions(simulator), dummy_run=True)


@pytest.fixture()
def particle_group():
    """A small synthetic electron bunch (px/py/pz in eV/c, weights in coulombs)."""
    from bunches import lab_snapshot

    return lab_snapshot(400, seed=3)
