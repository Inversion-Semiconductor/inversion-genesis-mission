# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Laser tangent probes 1 and 3 (inversion plan Section 4.8, Stage A).

1. `laser_initial_fields(sim, GaussianLaser(theta))` equals the six arrays
   that `add_laser_direct` writes into zeroed fields, within 1e-14 relative.
3. The E_L tangent of the spectral E and B after `begin` equals
   s0 / (2 E_L) within 1e-14 relative. The scale of each difference is the
   largest entry of that field (E or B) over all modes and components: for
   the x-polarized laser, some arrays (mode 0, mode 1 Ep) hold rounding
   noise only (1e-13 against 1e4 V/m), and their own maximum is no scale.

Probe 2 (z_f) belongs to Stage B.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

INVERSION = Path(__file__).resolve().parents[1] / "inversion"
sys.path.insert(0, str(INVERSION))

FIELDS = ("Er", "Et", "Ez", "Br", "Bt", "Bz")
SPECT = ("Ep", "Em", "Ez", "Bp", "Bm", "Bz")


def _relative(a, b):
    scale = float(np.abs(b).max())
    return float(np.abs(a - b).max()) / scale if scale else float(np.abs(a).max())


@pytest.mark.parametrize("size", ["twin_xs", "twin_s"])
def test_probe1_initial_fields(size):
    from builders import twin_builder

    from fbpic.autodiff.laser import laser_initial_fields

    build = twin_builder(size)
    theta = np.array(build.config["theta"]["nominal"])
    sim, _, profile = build(theta)
    fields = laser_initial_fields(sim, profile)
    worst = 0.0
    for m, mode in enumerate(fields):
        for name in FIELDS:
            reference = np.asarray(getattr(sim.fld.interp[m], name))
            error = _relative(mode[name], reference)
            worst = max(worst, error)
            assert error <= 1e-14, (size, m, name, error)
    print(f"probe 1 {size}: worst relative difference {worst:.2e}")


@pytest.mark.parametrize("size", ["twin_xs", "twin_s"])
def test_probe3_energy_seed(size):
    from builders import twin_builder

    from fbpic.autodiff.driver import resolve_seeds, seed_tangents
    from fbpic.autodiff.stepper import begin

    build = twin_builder(size)
    theta = np.array(build.config["theta"]["nominal"])
    sim, species, profile = build(theta)
    seeds = resolve_seeds(sim, profile, theta, ("E_L",))
    state = begin(sim, species)
    tangents = seed_tangents(state, seeds)
    scales = {
        kind: max(
            float(cupy.abs(getattr(grid, name)).max())
            for grid in sim.fld.spect
            for name in SPECT
            if name.startswith(kind)
        )
        / (2.0 * theta[0])
        for kind in "EB"
    }
    worst = 0.0
    for m, grid in enumerate(sim.fld.spect):
        for name in SPECT:
            expected = cupy.asnumpy(getattr(grid, name)) / (2.0 * theta[0])
            actual = cupy.asnumpy(tangents.fields.spect[m][name][0])
            error = float(np.abs(actual - expected).max()) / scales[name[0]]
            worst = max(worst, error)
            assert error <= 1e-14, (size, m, name, error)
        np.testing.assert_array_equal(
            cupy.asnumpy(tangents.fields.spect[m]["rho_prev"][0]), 0
        )
    for arrays in tangents.particles.values():
        for value in arrays.values():
            assert not bool(cupy.any(value))
    print(f"probe 3 {size}: worst relative difference {worst:.2e}")
