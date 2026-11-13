"""The boosted-frame LWFA example with an injected bunch, `docs/examples/lwfa_injected_beam.py`."""

from __future__ import annotations

import math

import pytest
from scipy.constants import c, e

from lume_fbpic.utils import ELECTRON_MC2_EV
from short_runs import on_cpu


def test_the_example_is_the_originals_setup():
    import lwfa_injected_beam as example

    model = example.build_model()
    simulator = model.simulator
    h = simulator.hyparams

    assert h.gamma_boost == 10.0 and h.nz == 600 and h.nr == 75 and h.nm == 2
    # the original's time step and moving window
    assert h.dt == pytest.approx(min(h.rmax / (2 * 10.0 * h.nr), 30.0e-6 / 600) / c)
    assert h.beta_window == pytest.approx(1 - 0.5 * 3.0e24 / 1.75e27)
    assert simulator.lasers[0].method == "antenna"
    assert simulator.densities[0].species == "H"
    assert simulator.target_species == "bunch"
    assert {"laser_a0", "plasma_density", "bunch_gamma", "bunch_radius"} <= set(
        model.supported_variables
    )


def test_a_short_boosted_run_accelerates_the_bunch_and_reads_it_back(tmp_path):
    import lwfa_injected_beam as example

    # nz 300 resolves the 0.8 um wavelength with 8 cells (150 cells, 4, do not accelerate the bunch)
    model = on_cpu(example.build_model(str(tmp_path), nz=300, nr=20, plateau=2.0e-5))
    simulator = model.simulator

    model.set({})

    dumps = sorted(simulator.diagnostics_directory(tmp_path).glob("data*.h5"))
    assert len(dumps) == example.NUMBER_DUMPS  # back-transformed to the lab frame
    bunch = simulator.beams[0]
    start = (bunch.gamma0 - 1) * ELECTRON_MC2_EV / 1e6
    # the first half of the plasma accelerates the bunch (see the module docstring)
    assert model.get(["energy_mean_mev"])["energy_mean_mev"] > start + 10.0
    # POSIT: fbpic loads the bunch in the whole radial cells whose centres are inside p_rmax, so
    # its radius is a whole number of cells: here one, 7.5 um, of the 10 um asked for
    dr = simulator.hyparams.rmax / simulator.hyparams.nr
    radius = math.ceil(bunch.p_rmax / dr - 0.5) * dr
    cylinder = e * bunch.n * math.pi * radius**2 * (bunch.p_zmax - bunch.p_zmin)
    assert model.get(["charge_pc"])["charge_pc"] == pytest.approx(
        cylinder * 1e12, rel=0.01
    )
