# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Check that the exchange and injection fast paths are bitwise identical.

- `generate_evenly_spaced` against a frozen copy of the previous version.
- The GPU open-boundary rebuild against the previous remove/generate/add
  sequence, including the charge that is deposited after the exchange.
- The antenna velocity update without GPU readback against the previous one.
"""

import warnings

import numpy as np
import pytest
from scipy.constants import c, e, m_e, m_p

from fbpic.particles.injection import generate_evenly_spaced
from fbpic.particles.injection.continuous_injection import (
    _check_dens_func_arguments,
    unalign_angles,
)
from fbpic.utils.cuda import cuda_installed

if cuda_installed:
    import cupy


def _legacy_generate_evenly_spaced(
    Npz,
    zmin,
    zmax,
    Npr,
    rmin,
    rmax,
    Nptheta,
    n,
    dens_func,
    ux_m,
    uy_m,
    uz_m,
    ux_th,
    uy_th,
    uz_th,
):
    """Frozen copy of `generate_evenly_spaced` before the (z, r) fast path."""
    if Npz * Npr * Nptheta > 0:
        dz = (zmax - zmin) * 1.0 / Npz
        z_reg = zmin + dz * (np.arange(Npz) + 0.5)
        dr = (rmax - rmin) * 1.0 / Npr
        r_reg = rmin + dr * (np.arange(Npr) + 0.5)
        dtheta = 2 * np.pi / Nptheta
        theta_reg = dtheta * np.arange(Nptheta)
        zp, rp, thetap = np.meshgrid(z_reg, r_reg, theta_reg, copy=True, indexing="ij")
        unalign_angles(thetap, Npz, Npr, method="random")
        r = rp.flatten()
        x = r * np.cos(thetap.flatten())
        y = r * np.sin(thetap.flatten())
        z = zp.flatten()
        w = n * r * dtheta * dr * dz
        if dens_func is not None:
            args = _check_dens_func_arguments(dens_func)
            if args == ["x", "y", "z"]:
                w *= dens_func(x=x, y=y, z=z)
            elif args == ["z", "r"]:
                w *= dens_func(z=z, r=r)
        selected = w > 0
        if np.any(w < 0):
            warnings.warn("negative densities")
        Ntot = int(selected.sum())
        x = x[selected]
        y = y[selected]
        z = z[selected]
        w = w[selected]
        uz = uz_m * np.ones(Ntot) + uz_th * np.random.normal(size=Ntot)
        ux = ux_m * np.ones(Ntot) + ux_th * np.random.normal(size=Ntot)
        uy = uy_m * np.ones(Ntot) + uy_th * np.random.normal(size=Ntot)
        inv_gamma = 1.0 / np.sqrt(1 + ux**2 + uy**2 + uz**2)
        return (Ntot, x, y, z, ux, uy, uz, inv_gamma, w)
    empty = np.empty(0)
    return (0,) + (empty,) * 8


def _dens_plateau(z, r):
    # Ramp, plateau and radial channel, with an exponential and a power
    n = np.where(z < 20e-6, np.clip(z / 20e-6, 0, None), 1.0)
    n = n * (1.0 + 0.3 * (r / 40e-6) ** 2) * np.exp(-((r / 90e-6) ** 8))
    return np.where(z > 1.5e-3, 0.0, n)


def _dens_zero(z, r):
    return np.zeros_like(z)


def _dens_negative(z, r):
    return np.where(r < 30e-6, 1.0, -0.5)


def _dens_xyz(x, y, z):
    return np.exp(-((x - 10e-6) ** 2 + y**2) / (40e-6) ** 2) * (z > 1.0e-3)


class _CallableDensity:
    def __call__(self, z, r):
        return np.cos(r / 50e-6) ** 2


@pytest.mark.parametrize(
    "dens_func",
    [_dens_plateau, _dens_zero, _dens_negative, None, _dens_xyz, _CallableDensity()],
)
@pytest.mark.parametrize("u_th", [0.0, 0.01])
@pytest.mark.parametrize(
    "Npz,zmin,Npr,Nptheta",
    [(62, 1.0e-3, 270, 15), (7, 1.49e-3, 33, 4), (0, 1.0e-3, 10, 4), (5, 0.0, 9, 1)],
)
def test_generate_evenly_spaced_matches_legacy(
    dens_func, u_th, Npz, zmin, Npr, Nptheta
):
    args = (
        Npz,
        zmin,
        zmin + Npz * 0.33e-6,
        Npr,
        0.0,
        90e-6,
        Nptheta,
        1e24,
        dens_func,
        0.01,
        -0.02,
        -1.118,
        u_th,
        2 * u_th,
        u_th,
    )
    results = []
    for function in (_legacy_generate_evenly_spaced, generate_evenly_spaced):
        np.random.seed(7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = function(*args)
        state = np.random.get_state()
        results.append((out, state))
    (legacy, legacy_state), (new, new_state) = results
    assert type(new[0]) is int and new[0] == legacy[0]
    for a, b in zip(legacy[1:], new[1:]):
        assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
    np.testing.assert_array_equal(legacy_state[1], new_state[1])
    assert legacy_state[2:] == new_state[2:]


def test_negative_density_warns():
    with pytest.warns(UserWarning, match="negative densities"):
        generate_evenly_spaced(
            3, 0.0, 1e-6, 4, 0.0, 1e-4, 2, 1e24, _dens_negative, 0, 0, 0, 0, 0, 0
        )


# GPU open-boundary rebuild
# -------------------------

gpu = pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")


def _density_zr(z, r):
    return 1.0 + 0.5 * np.sin(z / 3.0e-6)


def _density_xyz(x, y, z):
    return 1.0 + 0.3 * x / 10.0e-6


def _beam_density(z, r):
    return np.where((z > 12.0e-6) & (z < 16.0e-6) & (r < 4.0e-6), 1.0, 0.0)


def _moving_window_simulation():
    from fbpic.main import Simulation
    from fbpic.utils.random_seed import set_random_seed

    set_random_seed(3)
    sim = Simulation(
        64,
        20.0e-6,
        16,
        10.0e-6,
        2,
        20.0e-6 / (64 * c),
        zmin=-10.0e-6,
        n_order=-1,
        use_cuda=True,
        boundaries={"z": "open", "r": "reflective"},
        particle_shape="linear",
        verbose_level=0,
    )
    electrons = sim.add_new_species(
        q=-e,
        m=m_e,
        n=4.0e24,
        dens_func=_density_zr,
        p_zmin=-10.0e-6,
        p_zmax=20.0e-6,
        p_rmax=8.0e-6,
        p_nz=1,
        p_nr=1,
        p_nt=4,
        ux_th=0.01,
        uz_th=0.01,
    )
    ions = sim.add_new_species(
        q=e,
        m=14.0 * m_p,
        n=4.0e22,
        dens_func=_density_xyz,
        p_zmin=-10.0e-6,
        p_zmax=20.0e-6,
        p_rmax=8.0e-6,
        p_nz=1,
        p_nr=1,
        p_nt=4,
    )
    # A species without removal or injection at the exchange: a beam in the
    # right part of the box, at rest, with no continuous injection
    beam = sim.add_new_species(
        q=-e,
        m=m_e,
        n=1.0e24,
        dens_func=_beam_density,
        p_zmin=12.0e-6,
        p_zmax=16.0e-6,
        p_rmax=4.0e-6,
        p_nz=1,
        p_nr=1,
        p_nt=4,
        continuous_injection=False,
    )
    electrons.track(sim.comm)
    ions.track(sim.comm)
    # Random spins: the injection draws them from np.random
    electrons.activate_spin_tracking(sz_m=0.3, spin_distr="rand")
    ions.make_ionizable(
        "N", target_species=electrons, level_start=2, rng_seed=5, rng_stream=1
    )
    sim.set_moving_window(v=c)
    return sim, (electrons, ions, beam)


# Exchanges before the tested one. Particles leave the box through the left
# boundary from the 7th exchange on, so the tested exchange removes particles.
N_EXCHANGES = 8


def _state_objects(species):
    return {
        "species": species,
        "tracker": species.tracker,
        "ionizer": species.ionizer,
        "injector": species.injector,
        "spin_tracker": species.spin_tracker,
    }


def _snapshot(sim):
    """Copy the device arrays and scalars of all species, and the RNG state."""
    state = {"rng": np.random.get_state(), "species": []}
    for species in sim.ptcl:
        saved = {}
        for label, obj in _state_objects(species).items():
            if obj is None:
                continue
            for key, value in vars(obj).items():
                if isinstance(value, cupy.ndarray):
                    saved[label, key] = value.copy()
                elif value is None or isinstance(value, (bool, int, float, np.generic)):
                    saved[label, key] = value
        state["species"].append(saved)
    return state


def _restore(sim, state):
    np.random.set_state(state["rng"])
    for species, saved in zip(sim.ptcl, state["species"]):
        objects = _state_objects(species)
        for (label, key), value in saved.items():
            if isinstance(value, cupy.ndarray):
                value = value.copy()
            setattr(objects[label], key, value)


def _legacy_exchange(comm, species, fld, time):
    """The per-species exchange before the GPU rebuild and the hoisting.

    Returns the number of removed particles.
    """
    from fbpic.boundaries.particle_buffer_handling import (
        add_buffers_to_particles,
        remove_outside_particles,
    )

    Ntot = species.Ntot
    float_l, float_r, uint_l, uint_r = remove_outside_particles(
        species, fld, comm.n_guard, comm.left_proc, comm.right_proc
    )
    n_removed = Ntot - species.Ntot
    n_float, n_int = float_l.shape[0], uint_l.shape[0]
    float_recv_l = np.zeros((n_float, 0), dtype=np.float64)
    float_recv_r = np.zeros((n_float, 0), dtype=np.float64)
    uint_recv_l = np.zeros((n_int, 0), dtype=np.uint64)
    uint_recv_r = np.zeros((n_int, 0), dtype=np.uint64)
    if species.continuous_injection:
        float_recv_r, uint_recv_r = species.generate_continuously_injected_particles(
            time
        )
    add_buffers_to_particles(
        species, float_recv_l, float_recv_r, uint_recv_l, uint_recv_r
    )
    return n_removed


def _exchange_result(sim):
    """Return the particle data, counters, RNG state and deposited charge."""
    sim.deposit("rho_prev", update_spectral=False)
    result = {"rng": np.random.get_state()[1].copy()}
    for i, species in enumerate(sim.ptcl):
        for name in ("x", "y", "z", "ux", "uy", "uz", "inv_gamma", "w"):
            result[i, name] = getattr(species, name).get()
        result[i, "Ntot"] = species.Ntot
        if species.tracker is not None:
            result[i, "id"] = species.tracker.id.get()
            result[i, "next_id"] = species.tracker.next_attributed_id
        if species.ionizer is not None:
            result[i, "level"] = species.ionizer.ionization_level.get()
            result[i, "w_times_level"] = species.ionizer.w_times_level.get()
        if species.spin_tracker is not None:
            for name in ("sx", "sy", "sz"):
                result[i, name] = getattr(species.spin_tracker, name).get()
        result[i, "prefix_sum"] = species.prefix_sum.get()
        for name in ("Ex", "sorting_buffer"):
            assert getattr(species, name).shape == (species.Ntot,)
    for m, grid in enumerate(sim.fld.interp):
        result["rho", m] = grid.rho.get()
    return result


def _assert_bitwise_equal(legacy, new):
    """Compare all entries bitwise, except the deposited charge.

    The charge deposition uses atomic additions, so two depositions of the
    same particles can differ by rounding. The particle arrays and the
    prefix sums, which determine the deposition, are compared bitwise.
    """
    assert legacy.keys() == new.keys()
    for key in legacy:
        a, b = legacy[key], new[key]
        if key[0] == "rho":
            scale = float(np.abs(a).max())
            np.testing.assert_allclose(b, a, rtol=0, atol=1e-13 * scale)
        elif isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape, key
            np.testing.assert_array_equal(
                a.view(np.uint8), b.view(np.uint8), err_msg=str(key)
            )
        else:
            assert a == b, key


@gpu
def test_open_boundary_rebuild_matches_legacy_sequence(monkeypatch):
    import fbpic.boundaries.boundary_communicator as communicator
    from fbpic.utils.cuda import send_data_to_gpu

    sim, (electrons, ions, beam) = _moving_window_simulation()
    period = sim.comm.exchange_period
    # Stop just before an exchange iteration. The last sort (for rho_next)
    # happened before the window moved, so prefix_sum_shift is not zero.
    sim.step(N_EXCHANGES * period, show_progress=False)
    assert sim.iteration % period == 0
    # `step` copies the data back to the CPU at the end: copy it to the GPU
    # again, as at the beginning of `step`
    send_data_to_gpu(sim)
    assert beam.sorted and beam.prefix_sum_shift != 0
    assert electrons.injector.nz_inject > 0 and ions.injector.nz_inject > 0
    # Exercise both ionization levels in the stay range
    levels = ions.ionizer.ionization_level
    levels[::3] = 4
    ions.ionizer.w_times_level[:] = ions.w * levels

    state = _snapshot(sim)
    counts = [species.Ntot for species in sim.ptcl]

    # Previous sequence: per species, remove, then generate, then add
    n_removed = [
        _legacy_exchange(sim.comm, species, sim.fld, sim.time) for species in sim.ptcl
    ]
    legacy = _exchange_result(sim)
    # The electrons and ions lose particles at the left boundary and gain
    # injected particles at the right boundary
    assert n_removed[0] > 0 and n_removed[1] > 0
    # The beam has neither removal nor injection
    assert beam.Ntot == counts[2]
    assert electrons.Ntot != counts[0] and ions.Ntot != counts[1]

    # New path, from the same state. Record the calls of the GPU rebuild,
    # so that a fallback to the previous path cannot pass unnoticed.
    _restore(sim, state)
    rebuilt = []
    rebuild = communicator.rebuild_open_boundary_gpu

    def spy(species, *args, **kwargs):
        rebuilt.append(species)
        return rebuild(species, *args, **kwargs)

    monkeypatch.setattr(communicator, "rebuild_open_boundary_gpu", spy)
    sim.comm.exchange_all_particles(sim.ptcl, sim.fld, sim.time)
    new = _exchange_result(sim)

    assert rebuilt == list(sim.ptcl)
    assert electrons.injection_staging is not None
    assert ions.injection_staging is not None
    assert beam.injection_staging is None
    _assert_bitwise_equal(legacy, new)
    assert np.unique(new[1, "level"]).size == 2
    assert (0, "sx") in new
    assert np.abs(new["rho", 0]).max() > 0


def _float_quantities(species):
    """Return the float quantities of all particles of `species` on the host,
    in the order of the rows of the particle buffers."""
    rows = [
        getattr(species, name).get()
        for name in ("x", "y", "z", "ux", "uy", "uz", "inv_gamma", "w")
    ]
    if species.ionizer is not None:
        rows.append(species.ionizer.w_times_level.get())
    if species.spin_tracker is not None:
        rows += [getattr(species.spin_tracker, n).get() for n in ("sx", "sy", "sz")]
    return np.array(rows)


def _injected_uints(species, n):
    """Return the integer buffer of `n` injected particles, as
    `generate_continuously_injected_particles` makes it."""
    rows = []
    if species.tracker is not None:
        rows.append(species.tracker.generate_new_ids(n))
    if species.ionizer is not None:
        rows.append(np.full(n, species.ionizer.level_start, dtype=np.uint64))
    return np.array(rows, dtype=np.uint64).reshape(-1, n)


@gpu
def test_open_boundary_rebuild_keeps_count():
    """Inject as many particles as are removed: Ntot does not change, so the
    rebuild replaces the particle arrays but keeps the auxiliary arrays."""
    from fbpic.boundaries.particle_buffer_handling import (
        add_buffers_to_particles,
        rebuild_open_boundary_gpu,
        remove_outside_particles,
    )
    from fbpic.utils.cuda import PinnedStagingBuffer, send_data_to_gpu

    sim, (electrons, ions, beam) = _moving_window_simulation()
    period = sim.comm.exchange_period
    sim.step(N_EXCHANGES * period, show_progress=False)
    send_data_to_gpu(sim)
    comm = sim.comm
    state = _snapshot(sim)
    counts = [species.Ntot for species in sim.ptcl]
    # Inject copies of the last particles, which are inside the box
    floats = [_float_quantities(species) for species in (electrons, ions)]

    # Previous sequence, with the injection of the removed number of particles
    n_removed = []
    for species in (electrons, ions):
        Ntot = species.Ntot
        float_l, _, uint_l, _ = remove_outside_particles(
            species, sim.fld, comm.n_guard, comm.left_proc, comm.right_proc
        )
        n_removed.append(Ntot - species.Ntot)
        add_buffers_to_particles(
            species,
            np.zeros((float_l.shape[0], 0)),
            floats[len(n_removed) - 1][:, -n_removed[-1] :],
            np.zeros((uint_l.shape[0], 0), dtype=np.uint64),
            _injected_uints(species, n_removed[-1]),
        )
    legacy = _exchange_result(sim)
    assert min(n_removed) > 0
    assert [species.Ntot for species in sim.ptcl] == counts

    # GPU rebuild, from the same state
    _restore(sim, state)
    for i, species in enumerate((electrons, ions)):
        auxiliary = species.Ex, species.Bz, species.sorting_buffer
        old_x = species.x
        float_r = floats[i][:, -n_removed[i] :]
        species.injection_staging = PinnedStagingBuffer()
        staged = species.injection_staging.array(float_r.shape)
        staged[:] = float_r
        rebuild_open_boundary_gpu(species, sim.fld, comm.n_guard, staged)
        assert species.Ntot == counts[i]
        assert species.x is not old_x
        kept = (species.Ex, species.Bz, species.sorting_buffer)
        assert all(a is b for a, b in zip(kept, auxiliary))
    new = _exchange_result(sim)

    _assert_bitwise_equal(legacy, new)


@gpu
def test_open_boundary_rebuild_runs_in_simulation():
    sim, (electrons, ions, beam) = _moving_window_simulation()
    sim.step(4 * sim.comm.exchange_period + 1, show_progress=False)
    for species in (electrons, ions):
        ids = species.tracker.id  # `step` copies the data back to the CPU
        assert ids.size == species.Ntot
        assert np.unique(ids).size == ids.size
    assert ions.injection_staging is not None


@gpu
def test_pinned_staging_buffer_grows_and_waits():
    from fbpic.utils.cuda import PinnedStagingBuffer

    staging = PinnedStagingBuffer()
    small = staging.array((3, 10))
    small[:] = np.arange(30).reshape(3, 10)
    # Queue long work before the copy, on the same stream, so that the copy
    # is still pending when the host writes into the buffer again
    stream = cupy.cuda.get_current_stream()
    a = cupy.ones((4000, 4000))
    for _ in range(3):
        a = a @ a / 4000.0
    d_small = staging.to_device(small)
    assert not staging._event.done, "the queued work is too short"
    capacity = staging._capacity
    assert capacity == int(30 * 8 * 1.25)
    # `array` must wait for the copy before it returns the same memory
    large = staging.array((3, 12))
    assert staging._event is None
    assert stream.done
    assert staging._capacity == capacity
    large[:] = 1.0
    d_large = staging.to_device(large)
    np.testing.assert_array_equal(d_small.get(), np.arange(30).reshape(3, 10))
    np.testing.assert_array_equal(d_large.get(), np.ones((3, 12)))
    staging.array((3, 100))
    assert staging._capacity == int(300 * 8 * 1.25)
    assert staging.array((3, 0)).shape == (3, 0)


# Antenna without readback
# ------------------------


def _laser_profile():
    from fbpic.lpa_utils.laser.laser_profiles import LaserProfile

    class CpuLaser(LaserProfile):
        """A profile that is evaluated with NumPy only."""

        def __init__(self):
            LaserProfile.__init__(self, propagation_direction=1, gpu_capable=False)

        def E_field(self, x, y, z, t):
            assert isinstance(x, np.ndarray) and isinstance(z, np.ndarray)
            envelope = 1.0e12 * np.exp(-(x**2 + y**2) / (4.0e-6) ** 2)
            phase = 2.0e6 * (z - c * t) + 0.3 * x / 1.0e-6
            return envelope * np.cos(phase), 0.5 * envelope * np.sin(phase)

    return CpuLaser()


def _legacy_update_v(antenna, t, dt):
    """Frozen copy of `LaserAntenna.update_v` (GPU, no boost) with readback."""
    x = antenna.d_baseline_x + antenna.vx * 0.5 * dt
    y = antenna.d_baseline_y + antenna.vy * 0.5 * dt
    z = antenna.d_baseline_z + antenna.d_vz * 0.5 * dt
    x, y, z = x.get(), y.get(), z.get()
    Ex, Ey = antenna.laser_profile.E_field(x, y, z, t)
    antenna.vx.set(antenna.mobility_coef * Ex)
    antenna.vy.set(antenna.mobility_coef * Ey)


@gpu
@pytest.mark.parametrize("v_antenna", [0.0, 0.3 * c])
def test_antenna_update_v_without_readback(monkeypatch, v_antenna):
    from fbpic.lpa_utils.laser.antenna_injection import LaserAntenna

    def antenna():
        return LaserAntenna(
            _laser_profile(), 2.0e-6, v_antenna, 0.5e-6, 24, 2, None, use_cuda=True
        )

    legacy, new = antenna(), antenna()
    dt = 1.0e-16

    def no_readback(self, *args, **kwargs):
        raise AssertionError("update_v copied an array from the GPU")

    for step in range(50):
        t = step * dt
        _legacy_update_v(legacy, t + 0.5 * dt, dt)
        with monkeypatch.context() as patch:
            patch.setattr(cupy.ndarray, "get", no_readback)
            if step == 0:
                with pytest.raises(AssertionError, match="copied an array"):
                    new.vx.get()
            new.update_v(t + 0.5 * dt, dt)
        for item in (legacy, new):
            item.push_x(0.5 * dt)
            item.push_x(0.5 * dt)

    for name in ("vx", "vy", "excursion_x", "excursion_y", "d_baseline_z"):
        np.testing.assert_array_equal(
            getattr(legacy, name).get().view(np.uint64),
            getattr(new, name).get().view(np.uint64),
            err_msg=name,
        )
    np.testing.assert_array_equal(new.h_vx, legacy.vx.get())
    np.testing.assert_array_equal(new.baseline_z, new.d_baseline_z.get())
    assert np.abs(new.vx.get()).max() > 0
