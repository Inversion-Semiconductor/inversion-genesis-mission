# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Check fused particle kernels and GPU ionization through complete steps."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, e, m_e, m_p

from fbpic.utils.cuda import cuda_installed

pytestmark = pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")

FIELD_NAMES = ("Er", "Et", "Ez", "Br", "Bt", "Bz")


def _host(array):
    """Return a NumPy copy of a NumPy or CuPy array."""
    return array.get() if hasattr(array, "get") else np.asarray(array)


def _density(z, r):
    return np.where(z > 5.0e-6, 1.0, 0.0)


def _lwfa(fuse_particle_kernels, n_modes=5):
    from fbpic.lpa_utils.laser import add_laser_pulse
    from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser
    from fbpic.main import Simulation
    from fbpic.utils.random_seed import set_random_seed

    set_random_seed(11)
    sim = Simulation(
        96,
        20.0e-6,
        32,
        12.0e-6,
        n_modes,
        20.0e-6 / (96 * c),
        zmin=-10.0e-6,
        n_order=-1,
        use_cuda=True,
        boundaries={"z": "open", "r": "reflective"},
        particle_shape="linear",
        verbose_level=0,
        fuse_particle_kernels=fuse_particle_kernels,
    )
    electrons = sim.add_new_species(
        q=-e,
        m=m_e,
        n=4.0e24,
        dens_func=_density,
        p_zmin=0.0,
        p_zmax=20.0e-6,
        p_rmax=10.0e-6,
        p_nz=1,
        p_nr=1,
        p_nt=8,
    )
    ions = sim.add_new_species(
        q=2 * e,
        m=14.0 * m_p,
        n=4.0e22,
        dens_func=_density,
        p_zmin=0.0,
        p_zmax=20.0e-6,
        p_rmax=10.0e-6,
        p_nz=1,
        p_nr=1,
        p_nt=8,
    )
    electrons.track(sim.comm)
    ions.track(sim.comm)
    ions.make_ionizable(
        "N", target_species=electrons, level_start=2, rng_seed=5, rng_stream=1
    )
    # An a0 = 4 pulse with an off-axis waist excites every azimuthal mode.
    add_laser_pulse(sim, GaussianLaser(4.0, 4.0e-6, 10.0e-15, 0.0, theta_pol=0.3))
    sim.set_moving_window(v=c)
    return sim, ions


def test_fused_and_per_mode_simulations_agree():
    states = []
    for fuse in (True, False):
        sim, ions = _lwfa(fuse)
        sim.step(40, show_progress=False)
        fields = [
            _host(getattr(grid, name))
            for grid in sim.fld.interp
            for name in FIELD_NAMES
        ]
        ids = _host(ions.tracker.id)
        levels = _host(ions.ionizer.ionization_level)[np.argsort(ids)]
        states.append((fields, levels, ions.Ntot, sim.ptcl[0].Ntot))
    (fused, fused_levels, *fused_counts), (per_mode, levels, *counts) = states
    # Fused kernels add the same fp64 terms in another order (atomics and
    # mode grouping); over 40 steps this stays near fp64 rounding noise.
    for a, b in zip(fused, per_mode):
        scale = max(float(np.abs(b).max()), np.finfo(np.float64).tiny)
        assert float(np.abs(a - b).max()) <= 1e-9 * scale
    assert fused_counts == counts
    np.testing.assert_array_equal(fused_levels, levels)
    assert levels.max() > 2, "The test must exercise ionization."


def _gpu_ions(count=500):
    from fbpic.particles import Particles

    def species(q, m, n_particles):
        return Particles(
            q,
            m,
            1e24,
            n_particles,
            0,
            10e-6,
            1,
            0,
            2e-6,
            2,
            1e-16,
            continuous_injection=False,
            use_cuda=True,
            grid_shape=(16, 4),
        )

    ions = species(e, 4 * m_p, count)
    electrons = species(-e, m_e, 0)
    ions.track(SimpleNamespace(rank=0, size=1))
    ions.make_ionizable("He", electrons, level_start=1, rng_seed=7, rng_stream=2)
    ions.send_particles_to_gpu()
    electrons.send_particles_to_gpu()
    return ions


def test_gpu_ionization_decisions_follow_ids():
    reference = None
    for reverse in (False, True):
        ions = _gpu_ions()
        ions.Ex[:] = 4e11
        if reverse:
            ions.tracker.id = ions.tracker.id[::-1].copy()
        ions.ionizer.handle_ionization(ions, iteration=23)
        ids = ions.tracker.id.get()
        levels = ions.ionizer.ionization_level.get()[np.argsort(ids)]
        if reference is None:
            reference = levels.copy()
            assert np.any(levels == 1) and np.any(levels == 2)
        else:
            np.testing.assert_array_equal(levels, reference)
