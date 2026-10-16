# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Verify ID-based ionization draws and particle-ID lifecycle behavior."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import e, m_e, m_p

from fbpic.particles import Particles
from fbpic.particles.elementary_process.ionization.random import (
    generate_ionization_draws_cpu,
    philox4x32,
)
from fbpic.particles.tracking.tracking import ParticleTracker
from fbpic.utils.cuda import cuda_installed

# Independent known-answer vectors from Random123, tests/kat_vectors.
# https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
VECTORS = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    (
        (0xFFFFFFFF,) * 4,
        (0xFFFFFFFF,) * 2,
        (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
    ),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]


@pytest.mark.parametrize("counter,key,expected", VECTORS)
def test_philox_known_answers(counter, key, expected):
    actual = philox4x32(*(np.uint32(word) for word in counter + key))
    np.testing.assert_array_equal(actual, expected)


def _draws(ids, levels, iteration=17, seed=0xFEDCBA9876543210, stream=9):
    draws = np.empty(ids.size, dtype=np.float32)
    generate_ionization_draws_cpu(
        ids, levels, np.uint32(iteration), np.uint64(seed), np.uint32(stream), draws
    )
    return draws


def _particles(q=e, m=4 * m_p, count=31):
    return Particles(
        q,
        m,
        1e24,
        count,
        0,
        10e-6,
        1,
        0,
        2e-6,
        2,
        1e-16,
        continuous_injection=False,
    )


def test_draws_survive_permutation_and_batch_changes():
    ids = np.array([0, 1, 2**32 - 1, 2**32, 2**63, 2**64 - 1], dtype=np.uint64)
    levels = np.arange(ids.size, dtype=np.uint64)
    original = _draws(ids, levels)
    permutation = np.array([4, 1, 5, 0, 3, 2])
    np.testing.assert_array_equal(
        _draws(ids[permutation], levels[permutation]), original[permutation]
    )
    chunked = np.concatenate([_draws(ids[:3], levels[:3]), _draws(ids[3:], levels[3:])])
    np.testing.assert_array_equal(chunked, original)
    assert original.dtype == np.float32
    assert np.all((original >= 0) & (original < 1))
    for changed in (
        _draws(ids, levels, iteration=18),
        _draws(ids, levels, stream=10),
        _draws(ids, levels, seed=0xFEDCBA9876543211),
        _draws(ids, levels + np.uint64(1)),
    ):
        assert not np.array_equal(changed, original)


def test_draws_have_uniform_statistics():
    ids = np.arange(100_000, dtype=np.uint64)
    draws = _draws(ids, np.zeros(ids.size, dtype=np.uint64))
    # Broad checks detect broken streams without requiring a particular sequence.
    # For N = 1e5 uniform draws the bounds are about 5.5, 12.7, and 4.0 standard
    # deviations of the mean, the variance, and the fraction below 0.2.
    assert abs(float(draws.mean()) - 0.5) < 0.005
    assert abs(float(draws.var()) - 1 / 12) < 0.003
    assert abs(float(np.mean(draws < 0.2)) - 0.2) < 0.005


def test_restored_ids_advance_without_float_rounding():
    comm = SimpleNamespace(rank=1, size=3, mpi_comm=None)
    tracker = ParticleTracker(comm.size, comm.rank, 2)
    saved_ids = np.array([2**60 + 2, 2**60 + 5], dtype=np.uint64)
    tracker.overwrite_ids(saved_ids, comm)
    new_ids = tracker.generate_new_ids(3)
    assert int(new_ids[0]) > int(saved_ids.max())
    assert int(new_ids[0]) % comm.size == comm.rank
    assert np.all(np.diff(new_ids) == comm.size)


def test_tracking_reload_preserves_buffer_count():
    ions = _particles()
    comm = SimpleNamespace(rank=0, size=1)
    ions.track(comm)
    integer_count = ions.n_integer_quantities
    ions.track(comm)
    assert ions.n_integer_quantities == integer_count


@pytest.mark.parametrize(
    "seed,stream,error",
    [
        (-1, 0, "rng_seed"),
        (2**64, 0, "rng_seed"),
        (0, None, "rng_stream"),
        (0, -1, "rng_stream"),
        (0, 2**24, "rng_stream"),
        (None, 0, "rng_seed"),
    ],
)
def test_invalid_rng_configuration(seed, stream, error):
    ions = _particles()
    electrons = _particles(-e, m_e, 0)
    ions.track(SimpleNamespace(rank=0, size=1))
    with pytest.raises(ValueError, match=error):
        ions.make_ionizable("He", electrons, rng_seed=seed, rng_stream=stream)


def test_id_rng_requires_tracking():
    ions = _particles()
    electrons = _particles(-e, m_e, 0)
    with pytest.raises(ValueError, match="track"):
        ions.make_ionizable("He", electrons, rng_seed=0, rng_stream=0)


def test_legacy_cpu_sequence_is_unchanged():
    ions = _particles()
    electrons = _particles(-e, m_e, 0)
    ions.make_ionizable("He", electrons)
    np.random.seed(45)
    expected = np.random.rand(ions.Ntot)
    np.random.seed(45)
    actual = ions.ionizer._draw_random_numbers(ions, None)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("iteration", [None, -1, 2**32])
def test_invalid_iteration(iteration):
    ions = _particles()
    electrons = _particles(-e, m_e, 0)
    ions.track(SimpleNamespace(rank=0, size=1))
    ions.make_ionizable("He", electrons, rng_seed=0, rng_stream=0)
    with pytest.raises(ValueError, match="iteration"):
        ions.ionizer._draw_random_numbers(ions, iteration)


def test_ionization_decisions_follow_ids():
    reference = None
    for reverse in (False, True):
        ions = _particles(count=500)
        electrons = _particles(-e, m_e, 0)
        ions.track(SimpleNamespace(rank=0, size=1))
        ions.make_ionizable("He", electrons, level_start=1, rng_seed=7, rng_stream=2)
        ions.Ex[:] = 4e11
        if reverse:
            ions.tracker.id = ions.tracker.id[::-1].copy()
        ions.ionizer.handle_ionization(ions, iteration=23)
        levels = ions.ionizer.ionization_level[np.argsort(ions.tracker.id)]
        if reference is None:
            reference = levels.copy()
            assert np.any(levels == 1) and np.any(levels == 2)
        else:
            np.testing.assert_array_equal(levels, reference)


def _ionization_simulation(boundaries=None):
    from fbpic.main import Simulation

    sim = Simulation(
        64,
        8e-6,
        4,
        4e-6,
        1,
        1e-16,
        use_cuda=False,
        verbose_level=0,
        boundaries=boundaries or {"z": "periodic", "r": "reflective"},
        n_guard=16,
        n_damp={"z": 8, "r": 0},
        n_order=16,
    )
    ions = sim.add_new_species(
        e,
        4 * m_p,
        n=1e18,
        p_nz=1,
        p_nr=1,
        p_nt=2,
        p_zmin=7e-6,
        p_zmax=100e-6,
        p_rmax=4e-6,
    )
    electrons = sim.add_new_species(-e, m_e)
    ions.track(sim.comm)
    ions.make_ionizable("He", electrons, rng_seed=13, rng_stream=0)
    return sim, ions


def test_simulation_passes_global_iteration_across_step_calls():
    sim, ions = _ionization_simulation()
    iterations = []
    original = ions.ionizer._draw_random_numbers

    def observe(species, iteration):
        iterations.append(iteration)
        return original(species, iteration)

    ions.ionizer._draw_random_numbers = observe
    sim.step(2, show_progress=False)
    sim.step(2, show_progress=False)
    assert iterations == [0, 1, 2, 3]


def test_checkpoint_restores_rng_identity_and_iteration(tmp_path):
    from fbpic.openpmd_diag import restart_from_checkpoint, set_periodic_checkpoint

    sim, ions = _ionization_simulation()
    checkpoint = str(tmp_path / "checkpoints")
    set_periodic_checkpoint(sim, 2, checkpoint_dir=checkpoint)
    sim.step(2, show_progress=False)
    expected = ions.ionizer._draw_random_numbers(ions, sim.iteration)
    restored, restored_ions = _ionization_simulation()
    integer_count = restored_ions.n_integer_quantities
    restart_from_checkpoint(restored, iteration=2, checkpoint_dir=checkpoint)
    np.testing.assert_array_equal(restored_ions.tracker.id, ions.tracker.id)
    assert restored_ions.n_integer_quantities == integer_count
    actual = restored_ions.ionizer._draw_random_numbers(
        restored_ions, restored.iteration
    )
    np.testing.assert_array_equal(actual, expected)
    new_ids = restored_ions.tracker.generate_new_ids(3)
    assert int(new_ids[0]) > int(restored_ions.tracker.id.max())


def test_continuous_injection_assigns_unique_persistent_ids():
    from scipy.constants import c

    sim, ions = _ionization_simulation({"z": "open", "r": "reflective"})
    initial_max = int(ions.tracker.id.max())
    sim.set_moving_window(v=c)
    sim.step(20, show_progress=False)
    ids = ions.tracker.id
    assert ids.size == ions.Ntot
    assert np.unique(ids).size == ids.size
    assert int(ids.max()) > initial_max
    draws = ions.ionizer._draw_random_numbers(ions, sim.iteration)
    assert draws.dtype == np.float32 and draws.size == ids.size


def test_adk_ionization_regression_with_id_rng(monkeypatch, tmp_path):
    """Run the upstream ADK ionization test with ID-based draws enabled."""
    from test_ionization import run_simulation

    from fbpic.utils.mpi import comm

    original = Particles.make_ionizable

    def enable_id_rng(species, *args, **kwargs):
        species.track(comm)
        kwargs.update(rng_seed=0, rng_stream=0)
        return original(species, *args, **kwargs)

    monkeypatch.setattr(Particles, "make_ionizable", enable_id_rng)
    monkeypatch.chdir(tmp_path)
    run_simulation(gamma_boost=1.0, use_separate_electron_species=False)


@pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")
def test_gpu_raw_philox_known_answers():
    import cupy as cp

    counters = cp.asarray([case[0] for case in VECTORS], dtype=cp.uint32)
    keys = cp.asarray([case[1] for case in VECTORS], dtype=cp.uint32)
    words = cp.empty((len(VECTORS), 4), dtype=cp.uint32)
    _raw_philox_kernel[1, 128](counters, keys, words)
    # This diagnostic transfer occurs only in tests.
    np.testing.assert_array_equal(words.get(), [case[2] for case in VECTORS])


@pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")
@pytest.mark.parametrize("block_size", [64, 256])
def test_gpu_draws_match_cpu_after_permutation(block_size):
    import cupy as cp

    from fbpic.particles.elementary_process.ionization.random import (
        generate_ionization_draws_gpu,
    )

    ids = np.arange(1009, dtype=np.uint64) + np.uint64(2**63)
    ids[-1] = np.uint64(2**64 - 1)
    levels = np.arange(ids.size, dtype=np.uint64) % np.uint64(8)
    permutation = np.random.default_rng(8).permutation(ids.size)
    expected = _draws(ids, levels, iteration=2**32 - 1, stream=2**24 - 1)
    draws = cp.empty(ids.size, dtype=cp.float32)
    generate_ionization_draws_gpu[
        (ids.size + block_size - 1) // block_size, block_size
    ](
        cp.asarray(ids[permutation]),
        cp.asarray(levels[permutation]),
        np.uint32(2**32 - 1),
        np.uint64(0xFEDCBA9876543210),
        np.uint32(2**24 - 1),
        draws,
    )
    np.testing.assert_array_equal(draws.get(), expected[permutation])


if cuda_installed:
    import warp as wp

    from fbpic.particles.elementary_process.ionization.random import philox4x32_gpu
    from fbpic.utils.warp import warp_kernel

    @warp_kernel
    def _raw_philox_kernel(
        counters: wp.array2d(dtype=wp.uint32),
        keys: wp.array2d(dtype=wp.uint32),
        words: wp.array2d(dtype=wp.uint32),
    ):
        index = wp.tid()
        if index < counters.shape[0]:
            a, b, c, d = philox4x32_gpu(
                counters[index, 0],
                counters[index, 1],
                counters[index, 2],
                counters[index, 3],
                keys[index, 0],
                keys[index, 1],
            )
            words[index, 0] = a
            words[index, 1] = b
            words[index, 2] = c
            words[index, 3] = d
