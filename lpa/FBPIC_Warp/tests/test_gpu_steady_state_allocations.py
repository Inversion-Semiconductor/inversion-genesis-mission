# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Check that the steps between two exchanges allocate no device memory.

The GPU particle arrays are views of arrays with a capacity
(`fbpic.particles.utilities.capacity`), so that ionization grows the electron
species without allocation, and the sort and ionization workspaces persist
between steps. An allocator hook records each CuPy allocation, and the Warp
allocation tracker counts the native Warp allocations (the CUB sort storage).
"""

import ctypes
import math
import traceback
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, e, m_e, m_p

from fbpic.utils.cuda import cuda_installed

pytestmark = pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")

MIB = 1 << 20
PARTICLE_DATA = ("x", "y", "z", "ux", "uy", "uz", "w", "inv_gamma")
PARTICLE_FIELDS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")


@pytest.fixture
def async_runtime():
    import fbpic.utils.warp as warp_utils

    warp_utils.get_runtime().use()
    if warp_utils.allocator_mode() != "async":
        pytest.skip("The allocator hook needs the CUDA mempool allocator.")
    return warp_utils


class AllocationRecorder:
    """Record the device allocations of a simulation, by phase.

    The phase is "exchange" from the particle exchange to the first field
    gather of the iteration (this includes the release of unused memory and
    the reservation of the step workspaces), and "step" from the gather to
    the next exchange.
    """

    def __init__(self, monkeypatch, warp_utils, sim):
        import fbpic.main as fbpic_main
        from fbpic.boundaries.boundary_communicator import BoundaryCommunicator
        from fbpic.particles import Particles

        self.sim = sim
        self.phase = "setup"
        self.events = []  # (iteration, phase, bytes, site)
        self.native = {"exchange": 0, "step": 0, "setup": 0, "end": 0}
        self.ntot = {}  # iteration -> Ntot of each species at the gather
        self.checks = []
        core = _warp_core()
        self._core = core
        self._native_count = self._read_native()

        allocate = warp_utils._allocate_async

        def traced(size):
            frames = [
                f"{frame.filename.rsplit('/fbpic/', 1)[-1]}:{frame.lineno}"
                for frame in traceback.extract_stack()[:-1]
                if "/fbpic/" in frame.filename
                and not frame.filename.endswith("utils/warp.py")
            ]
            site = " <- ".join(frames[::-1][:3])
            self.events.append((sim.iteration, self.phase, int(size), site))
            return allocate(size)

        monkeypatch.setattr(warp_utils, "_allocate_async", traced)

        exchange = BoundaryCommunicator.exchange_all_particles

        def traced_exchange(comm, *args, **kwargs):
            self._switch("exchange")
            return exchange(comm, *args, **kwargs)

        monkeypatch.setattr(
            BoundaryCommunicator, "exchange_all_particles", traced_exchange
        )

        gather = Particles.gather

        def traced_gather(species, *args, **kwargs):
            if self.phase != "step":
                self._switch("step")
            if sim.iteration not in self.ntot:
                self.ntot[sim.iteration] = [s.Ntot for s in sim.ptcl]
                self.checks.append(_check_particle_arrays(sim))
            return gather(species, *args, **kwargs)

        monkeypatch.setattr(Particles, "gather", traced_gather)

        receive = fbpic_main.receive_data_from_gpu

        def traced_receive(simulation):
            self._switch("end")
            return receive(simulation)

        monkeypatch.setattr(fbpic_main, "receive_data_from_gpu", traced_receive)

    def _read_native(self):
        if self._core is None:
            return 0
        return int(self._core.wp_alloc_tracker_get_total_alloc_count())

    def _switch(self, phase):
        count = self._read_native()
        self.native[self.phase] += count - self._native_count
        self._native_count = count
        self.phase = phase

    def clear(self):
        self.events.clear()
        self.native = dict.fromkeys(self.native, 0)
        self.ntot.clear()
        self.checks.clear()

    def step_allocations(self):
        return [event for event in self.events if event[1] == "step"]


def _warp_core():
    """Enable the Warp allocation tracker, if this Warp build has one."""
    import warp as wp

    wp.init()
    core = wp._src.context.runtime.core
    if not hasattr(core, "wp_alloc_tracker_enable"):
        return None
    core.wp_alloc_tracker_get_total_alloc_count.restype = ctypes.c_size_t
    core.wp_alloc_tracker_enable(1)
    return core


def _check_particle_arrays(sim):
    """Return the problems of the particle arrays of all species: each array
    must have Ntot elements and the capacity of its species, and no two
    arrays of a species may share memory."""
    import cupy

    from fbpic.particles.utilities.capacity import capacity

    problems = []
    for i, species in enumerate(sim.ptcl):
        pointers = {}
        for obj, name, _, _ in species.particle_arrays():
            array = getattr(obj, name, None)
            if not isinstance(array, cupy.ndarray):
                continue
            if array.shape != (species.Ntot,):
                problems.append((sim.iteration, i, name, "shape", array.shape))
            if capacity(array) != species.particle_capacity:
                problems.append((sim.iteration, i, name, "capacity", capacity(array)))
            pointer = array.data.ptr
            if pointer in pointers:
                problems.append((sim.iteration, i, name, "shares", pointers[pointer]))
            pointers[pointer] = name
    return problems


def _ionizing_simulation(per_level, electron_headroom=8.0):
    """A moving window through a nitrogen plasma (N2+), in a constant Ex field
    that ionizes N2+ at a few percent per step. The continuous injection
    brings new ions at each exchange, so the electrons grow at each step.
    A large `electron_headroom` holds the growth of a whole exchange period
    from the start; None keeps the default (the headroom then follows the
    measured growth)."""
    from fbpic.lpa_utils.external_fields import ExternalField
    from fbpic.main import Simulation
    from fbpic.utils.random_seed import set_random_seed

    set_random_seed(11)
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
    if per_level:
        targets = {level: sim.add_new_species(q=-e, m=m_e) for level in (2, 3)}
        level_max = 4
    else:
        targets = sim.add_new_species(q=-e, m=m_e)
        level_max = 3
    ions = sim.add_new_species(
        q=e,
        m=14.0 * m_p,
        n=1.0e24,
        p_zmin=-10.0e-6,
        p_zmax=20.0e-6,
        p_rmax=8.0e-6,
        p_nz=2,
        p_nr=2,
        p_nt=4,
    )
    ions.track(sim.comm)
    ions.make_ionizable(
        "N",
        target_species=targets,
        level_start=2,
        level_max=level_max,
        rng_seed=7,
        rng_stream=3,
    )
    electrons = list(targets.values()) if per_level else [targets]
    if electron_headroom is not None:
        for species in electrons:
            species.capacity_headroom = electron_headroom

    def constant(F, x, y, z, t, amplitude, length_scale):
        return F + amplitude

    sim.external_fields = [ExternalField(constant, "Ex", 1.3e11, 0.0, species=ions)]
    sim.set_moving_window(v=c)
    return sim, ions, electrons


@pytest.mark.parametrize(
    "per_level, electron_headroom",
    [(False, 8.0), (True, 8.0), (False, None)],
    ids=["single-level", "per-level", "measured-growth"],
)
def test_ionizing_steps_do_not_allocate(
    async_runtime, monkeypatch, per_level, electron_headroom
):
    """With a large fixed headroom, and with the default headroom that
    follows the measured growth (after one period of measurement)."""
    sim, ions, electrons = _ionizing_simulation(per_level, electron_headroom)
    period = sim.comm.exchange_period
    recorder = AllocationRecorder(monkeypatch, async_runtime, sim)
    # Warm up: the first ionization steps outgrow the initial capacity
    sim.step(2 * period, show_progress=False)
    recorder.clear()
    sim.step(3 * period, show_progress=False)

    # The electrons grow within the steps, so the check is not vacuous
    growth = [
        recorder.ntot[it + 1][sim.ptcl.index(electrons[0])]
        - recorder.ntot[it][sim.ptcl.index(electrons[0])]
        for it in sorted(recorder.ntot)
        if it + 1 in recorder.ntot and (it + 1) % period != 0
    ]
    assert sum(g > 0 for g in growth) >= period, growth
    steps = recorder.step_allocations()
    sites = sorted({(event[2], event[3]) for event in steps})
    assert not steps, f"{len(steps)} allocations within steps: {sites}"
    if recorder._core is not None:
        assert recorder.native["step"] == 0
    problems = [problem for check in recorder.checks for problem in check]
    assert not problems, problems[:10]


def test_exchange_rebuild_reuses_sorting_buffers(async_runtime, monkeypatch):
    """When the capacity target of a species does not change, the
    open-boundary rebuild writes into the sorting buffers and allocates
    nothing (except the device copy of the injected particles)."""
    import fbpic.boundaries.boundary_communicator as communicator
    from fbpic.utils.cuda import send_data_to_gpu

    sim, ions, electrons = _ionizing_simulation(False)
    sim.step(2 * sim.comm.exchange_period, show_progress=False)
    # The ions grow at each exchange: a large headroom keeps their target
    ions.capacity_headroom = 1.0
    # `step` copies the data back to the CPU at the end
    send_data_to_gpu(sim)
    for species in sim.ptcl:
        species.reserve_step_workspaces()
    capacities = [species.particle_capacity for species in sim.ptcl]
    before = [[a.data.ptr for a in _arrays(species)] for species in sim.ptcl]
    sites = []
    allocate = async_runtime._allocate_async

    def traced(size):
        stack = traceback.extract_stack()[:-1]
        sites.append(
            [
                f"{f.filename.rsplit('/fbpic/', 1)[-1]}:{f.lineno}"
                for f in stack[-6:]
                if "/fbpic/" in f.filename
            ]
        )
        return allocate(size)

    rebuilt = []
    rebuild = communicator.rebuild_open_boundary_gpu

    def spy(species, *args, **kwargs):
        rebuilt.append(species)
        return rebuild(species, *args, **kwargs)

    monkeypatch.setattr(communicator, "rebuild_open_boundary_gpu", spy)
    monkeypatch.setattr(async_runtime, "_allocate_async", traced)
    counts = [species.Ntot for species in sim.ptcl]
    sim.comm.exchange_all_particles(sim.ptcl, sim.fld, sim.time)
    monkeypatch.setattr(async_runtime, "_allocate_async", allocate)

    assert rebuilt == list(sim.ptcl)
    assert ions.Ntot != counts[sim.ptcl.index(ions)]
    assert [species.particle_capacity for species in sim.ptcl] == capacities
    # The device copy of the injected particles (PinnedStagingBuffer) is the
    # only allocation of the rebuild
    staging = [site for site in sites if any("utils/cuda.py" in f for f in site)]
    assert len(staging) == 1
    rebuild_sites = [
        site
        for site in sites
        if any("particle_buffer_handling" in f for f in site) and site not in staging
    ]
    assert not rebuild_sites, [site[-4:] for site in rebuild_sites]
    # The particle arrays rotated through the same memory
    after = [[a.data.ptr for a in _arrays(species)] for species in sim.ptcl]
    assert [set(p) for p in after] == [set(p) for p in before]


def _gpu_species(count, seed, n_int=0):
    import cupy

    from fbpic.particles import Particles

    rng = np.random.default_rng(seed)
    species = Particles.__new__(Particles)
    species.use_cuda = True
    species.q = -e
    species.Ntot = count
    species.tracker = species.ionizer = species.spin_tracker = None
    species.n_integer_quantities = n_int
    for name in (*PARTICLE_DATA, *PARTICLE_FIELDS):
        setattr(species, name, cupy.asarray(rng.normal(size=count)))
    species.sorting_buffer = cupy.empty(count)
    if n_int:
        species.tracker = SimpleNamespace(
            id=cupy.asarray(rng.permutation(count).astype(np.uint64))
        )
        species.int_sorting_buffer = cupy.empty(count, dtype=np.uint64)
    return species


def _host_arrays(species):
    arrays = {}
    for obj, name, kind, _ in species.particle_arrays():
        if kind != "scratch":
            arrays[name] = getattr(obj, name).get()
    return arrays


def _append(species, old_count, values):
    """Write the new particles, as the ionization kernels do."""
    import cupy

    for obj, name, kind, _ in species.particle_arrays():
        if kind != "scratch":
            array = getattr(obj, name)
            array[old_count:] = cupy.asarray(values[name][old_count:])


def test_growth_in_place_matches_reallocation(async_runtime):
    """Growth within the capacity (views, no copy) and growth beyond it
    (new arrays and copy) give the same particle data, bit for bit."""
    import cupy

    from fbpic.particles.elementary_process.cuda_numba_utils import (
        reallocate_and_copy_old,
    )
    from fbpic.particles.utilities.capacity import capacity, provision_particle_arrays

    counts = [5000, 5300, 5301, 5900, 7000, 40_000]
    in_place = _gpu_species(counts[0], seed=1, n_int=1)
    copied = _gpu_species(counts[0], seed=1, n_int=1)
    in_place.capacity_headroom = 0.25
    provision_particle_arrays(in_place)
    rng = np.random.default_rng(2)
    for old_count, new_count in zip(counts[:-1], counts[1:]):
        values = {
            name: rng.normal(size=new_count)
            for name in (*PARTICLE_DATA, *PARTICLE_FIELDS)
        }
        values["id"] = rng.permutation(new_count).astype(np.uint64)
        pointers = [a.data.ptr for a in _arrays(in_place)]
        old_cap = in_place.particle_capacity
        # The arrays of `copied` have no capacity: the growth reallocates
        # them and copies the old particles (the previous behaviour)
        for obj, name, _, _ in copied.particle_arrays():
            setattr(obj, name, getattr(obj, name).copy())
        copied.particle_capacity = 0
        for species in (in_place, copied):
            reallocate_and_copy_old(species, True, old_count, new_count)
            _append(species, old_count, values)
        assert in_place.Ntot == copied.Ntot == new_count
        if new_count <= old_cap:
            assert [a.data.ptr for a in _arrays(in_place)] == pointers
        else:
            # Every array moves to the new capacity (checked below)
            assert in_place.particle_capacity > old_cap
        for array in _arrays(in_place):
            assert array.shape == (new_count,)
            assert capacity(array) == in_place.particle_capacity
        a, b = _host_arrays(in_place), _host_arrays(copied)
        assert a.keys() == b.keys()
        for name in a:
            np.testing.assert_array_equal(
                a[name].view(np.uint64), b[name].view(np.uint64), err_msg=name
            )
    assert in_place.particle_capacity > counts[-2]
    cupy.cuda.get_current_stream().synchronize()


def _arrays(species):
    return [getattr(obj, name) for obj, name, _, _ in species.particle_arrays()]


def test_capacity_refuses_foreign_views(async_runtime):
    import cupy

    from fbpic.particles.utilities.capacity import (
        capacity,
        empty_particle_array,
        fit_particle_array,
    )

    block = cupy.zeros(1000)
    prefix = block[:100]
    rows = cupy.zeros((3, 100))
    assert capacity(prefix) == 100
    assert capacity(rows[1]) == 100
    owned = empty_particle_array(100, np.float64, 1000)
    assert capacity(owned) == 1000
    assert capacity(owned[::2]) == 50
    assert capacity(owned[10:20]) == 10
    # A foreign prefix is never extended over its neighbours
    grown = fit_particle_array(prefix, 200, 400, keep=100)
    assert grown.data.ptr != block.data.ptr and capacity(grown) == 400
    # An owned view is extended in place
    assert fit_particle_array(owned, 900, 1000).data.ptr == owned.data.ptr
    # With `exact`, a base of another capacity is replaced
    assert fit_particle_array(owned, 50, 512, exact=True).data.ptr != owned.data.ptr


def test_headroom_follows_measured_growth():
    """The capacity target holds GROWTH_PERIODS periods of the measured
    growth, all calls of one exchange agree, and a species that stops
    growing returns to the default headroom (bounded by SHRINK_FACTOR)."""
    from fbpic.particles.utilities import capacity as cap_module
    from fbpic.particles.utilities.capacity import (
        DEFAULT_HEADROOM,
        GROWTH_PERIODS,
        SHRINK_FACTOR,
        capacity_for,
        reserve_particle_capacity,
        start_growth_period,
    )

    species = SimpleNamespace(capacity_headroom=DEFAULT_HEADROOM)
    n = 1_000_000
    assert reserve_particle_capacity(species, n, rebalance=True) == capacity_for(
        n, DEFAULT_HEADROOM
    )
    start_growth_period(species)
    # In-step overflow: the new capacity holds the growth since the exchange
    # GROWTH_PERIODS more times
    species.grown_since_exchange = 400_000
    n += 400_000
    grown = reserve_particle_capacity(species, n)
    assert grown >= n + GROWTH_PERIODS * 400_000
    # At the exchange, every call uses the growth of the finished period
    for _ in range(3):
        assert reserve_particle_capacity(species, n, rebalance=True) == grown
    start_growth_period(species)
    # One period at the same rate fits; the next exchange keeps the target
    species.grown_since_exchange = 400_000
    n += 400_000
    assert reserve_particle_capacity(species, n) == grown
    assert reserve_particle_capacity(species, n, rebalance=True) == grown
    start_growth_period(species)
    # The growth stops: the target returns to the default headroom
    stopped = reserve_particle_capacity(species, n, rebalance=True)
    assert stopped == capacity_for(n, DEFAULT_HEADROOM) < grown
    # The net growth of an exchange (injection) also counts
    species.Ntot = n
    start_growth_period(species)
    assert species.count_at_exchange == n
    injected = n + n // 10
    assert reserve_particle_capacity(species, injected, rebalance=True) == (
        capacity_for(injected, GROWTH_PERIODS * (n // 10) / injected)
    )
    # Bound after any exchange: c <= SHRINK_FACTOR * capacity_for(n, h)
    rng = np.random.default_rng(3)
    for _ in range(2000):
        n = int(np.clip(n * rng.uniform(0.8, 1.25), 1, 10**8))
        species.grown_since_exchange = int(rng.integers(0, n // 4 + 1))
        c = reserve_particle_capacity(species, n, rebalance=True)
        h = cap_module.growth_headroom(species, n)
        assert max(n * (1 + h / 2), 1024) <= c <= SHRINK_FACTOR * capacity_for(n, h)
        assert c < max(SHRINK_FACTOR * 17 / 16 * (1 + h) * n + 1, 1024 * 1.25)
        species.Ntot = n
        start_growth_period(species)


def test_size_classes():
    from fbpic.particles.utilities.capacity import capacity_for, size_class

    previous = 0
    for n in [0, 1, 1023, 1024, 1025, 5000, 65_537, 10**6, 21_064_049, 2**31]:
        cap = size_class(n)
        assert cap >= max(n, 1024) and cap >= previous
        # At most 5 significant bits: steps of at most 1/16
        assert cap <= max(1024, math.ceil(n * 17 / 16))
        assert (
            cap >> max(cap.bit_length() - 5, 0) << max(cap.bit_length() - 5, 0) == cap
        )
        previous = cap
    assert capacity_for(1000, 0.25) == size_class(1250)
    # Close counts share a class, so their arrays have the same size
    assert size_class(21_000_000) == size_class(21_064_049)
