# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Check fusion dispatch, FP64 parity, and particle-buffer lifecycle behavior.

Device-to-host copies belong to verification only, outside simulation hot paths.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import e

from fbpic.fields import Fields
from fbpic.fields.interpolation_grid import InterpolationGrid
from fbpic.particles import Particles
from fbpic.utils.cuda import cuda_installed
from fbpic.utils.threading import nthreads

pytestmark = pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")

PARTICLE_FIELDS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
GRID_FIELDS = ("Er", "Et", "Ez", "Br", "Bt", "Bz")
PARTICLE_DATA = ("x", "y", "z", "ux", "uy", "uz", "inv_gamma", "w")
NZ, NR = 13, 7
DZ, DR = 0.03, 0.04
ZMIN, RMAX = -0.2, NR * DR


@pytest.fixture
def runtime():
    from fbpic.utils.warp import get_runtime

    runtime = get_runtime()
    runtime.use()
    yield runtime
    runtime.synchronize()


def _particle_data(count, seed=31):
    rng = np.random.default_rng(seed)
    radius = rng.uniform(0, 1.05 * RMAX, count)
    theta = rng.uniform(-np.pi, np.pi, count)
    z = rng.uniform(ZMIN, ZMIN + NZ * DZ, count)
    if count >= 17:
        radius[:17] = DR * np.array(
            [
                0,
                1e-12,
                0.1,
                0.499,
                0.5,
                0.501,
                1,
                2.5,
                3,
                5.5,
                6.499,
                6.5,
                6.9,
                7,
                7.01,
                7.2,
                1.2,
            ]
        )
        z[:6] = ZMIN + DZ * np.array([-0.25, 0, 0.5, 12.5, 13, 13.25])
    momenta = rng.normal(size=(3, count))
    return {
        "x": radius * np.cos(theta),
        "y": radius * np.sin(theta),
        "z": z,
        "ux": momenta[0],
        "uy": momenta[1],
        "uz": momenta[2],
        "inv_gamma": 1 / np.sqrt(1 + np.sum(momenta**2, axis=0)),
        "w": rng.uniform(0.2, 2, count),
    }


def _species(data, shape, use_cuda, charge=-e, effective_weights=False):
    # Explicit fixtures avoid particle-loading noise while exercising dispatch.
    species = Particles.__new__(Particles)
    species.Ntot = data["x"].size
    species.q = charge
    species.particle_shape = shape
    species.use_cuda = use_cuda
    species.fuse_deposition = species.fuse_gather = True
    species.sorted = False
    species.keep_fields_sorted = True
    species.spin_tracker = species.tracker = None
    species.ionizer = None
    species.n_integer_quantities = 0
    species.deposit_tpb = 32
    species.gather_tpb = 128 if shape == "linear" else 256
    convert = np.copy
    if use_cuda:
        import cupy as cp

        convert = cp.asarray
        species.cell_idx = cp.empty(species.Ntot, dtype=np.int32)
        species.sorted_idx = cp.empty(species.Ntot, dtype=np.intp)
        species.prefix_sum = cp.empty(NZ * (NR + 1), dtype=np.int32)
        species.sorting_buffer = cp.empty(species.Ntot, dtype=np.float64)
        species.prefix_sum_shift = 0
    for name in PARTICLE_DATA:
        setattr(species, name, convert(data[name]))
    for index, name in enumerate(PARTICLE_FIELDS):
        setattr(species, name, convert(np.full(species.Ntot, 17.0 + index)))
    if effective_weights:
        levels = (np.arange(species.Ntot) % 4).astype(np.uint64)
        species.ionizer = SimpleNamespace(
            w_times_level=convert(data["w"] * levels),
            ionization_level=convert(levels),
        )
        species.n_integer_quantities = 1
        if use_cuda:
            species.int_sorting_buffer = cp.empty(species.Ntot, dtype=np.uint64)
    return species


def _fields(n_modes, use_cuda, active_mode=None):
    fields = Fields.__new__(Fields)
    fields.Nm, fields.Nz, fields.Nr = n_modes, NZ, NR
    fields.use_cuda = use_cuda
    fields.interp = []
    rng = np.random.default_rng(47)
    for mode in range(n_modes):
        grid = InterpolationGrid(
            NZ, NR, mode, ZMIN, ZMIN + NZ * DZ, RMAX, use_cuda=False
        )
        for name in GRID_FIELDS:
            values = rng.normal(size=(NZ, NR)) + 1j * rng.normal(size=(NZ, NR))
            if active_mode is not None and mode != active_mode:
                values.fill(0)
            getattr(grid, name)[:] = values
        if use_cuda:
            import cupy as cp

            grid.use_cuda = True
            grid.d_ruyten_linear_coef = cp.asarray(grid.ruyten_linear_coef)
            grid.d_ruyten_cubic_coef = cp.asarray(grid.ruyten_cubic_coef)
            grid.send_fields_to_gpu()
        fields.interp.append(grid)
    if not use_cuda:
        for name in ("rho", "Jr", "Jt", "Jz"):
            setattr(
                fields,
                f"{name}_global",
                np.zeros((nthreads, n_modes, NZ + 4, NR + 4), dtype=np.complex128),
            )
    return fields


def _assert_scaled_close(observed, expected, tolerance=2e-12):
    # Fused and unfused kernels add the same fp64 terms in a different order
    # (atomics, mode grouping); 2e-12 of the field scale bounds that reordering.
    scale = max(float(np.abs(expected).max(initial=0)), np.finfo(np.float64).tiny)
    assert np.isfinite(observed).all()
    np.testing.assert_allclose(
        observed, expected, rtol=tolerance, atol=tolerance * scale
    )


def _expected_deposit_launches(n_modes, shape, fieldtype, fusion):
    if fusion and shape == "linear":
        # Charge fuses groups of five modes; current fuses groups of two.
        group = 5 if fieldtype == "rho" else 2
        return math.ceil(n_modes / group)
    return 1 if n_modes == 2 else n_modes


def _expected_gather_launches(n_modes, shape, count, fusion):
    if fusion and shape == "linear":
        # One launch per group of five modes; none without particles.
        return 0 if count == 0 else math.ceil(n_modes / 5)
    return 1 if n_modes == 2 else n_modes + 1


# The linear cases cover one and several mode groups, including a last group
# of one mode (11 modes), populated cells, partial particle blocks, neutral
# effective weights, and empty buffers. Cubic shapes use the per-mode kernels.
DEPOSITION_CASES = [
    (5, "linear", 0, False),
    (5, "linear", 17, False),
    (5, "linear", 257, True),
    (5, "cubic", 17, True),
    (1, "linear", 17, False),
    (2, "linear", 17, False),
    (3, "linear", 17, False),
    (6, "linear", 17, False),
    (11, "linear", 257, True),
    (6, "cubic", 17, False),
]


@pytest.mark.parametrize("fieldtype", ["rho", "J"])
@pytest.mark.parametrize("n_modes,shape,count,effective_weights", DEPOSITION_CASES)
def test_deposition_fusion_parity(
    runtime, fieldtype, n_modes, shape, count, effective_weights
):
    import cupy as cp

    data = _particle_data(count)
    charge = e if effective_weights else -e
    cpu = _species(data, shape, False, charge, effective_weights)
    cpu_fields = _fields(n_modes, False)
    cpu.deposit(cpu_fields, fieldtype)
    cpu_fields.sum_reduce_deposition_array(fieldtype)
    components = ("rho",) if fieldtype == "rho" else ("Jr", "Jt", "Jz")
    results = []
    for fusion in (False, True):
        gpu = _species(data, shape, True, charge, effective_weights)
        gpu.fuse_deposition = fusion
        gpu_fields = _fields(n_modes, True)
        # Run real sorting separately to isolate the deposition launch count.
        gpu.sort_particles(gpu_fields)
        gpu.sorted = True
        before = runtime.launch_count
        gpu.deposit(gpu_fields, fieldtype)
        assert runtime.launch_count - before == _expected_deposit_launches(
            n_modes, shape, fieldtype, fusion
        )
        runtime.synchronize()
        observed = []
        for gpu_grid, cpu_grid in zip(gpu_fields.interp, cpu_fields.interp):
            for name in components:
                value = cp.asnumpy(getattr(gpu_grid, name))
                _assert_scaled_close(value, getattr(cpu_grid, name))
                observed.append(value)
        results.append(observed)
    for unfused, fused in zip(*results):
        _assert_scaled_close(fused, unfused)
    if fieldtype == "rho":
        weight = cpu.ionizer.w_times_level if effective_weights else data["w"]
        total_charge = charge * np.sum(weight)
        np.testing.assert_allclose(
            np.sum(results[1][0].real),
            total_charge,
            rtol=2e-12,
            atol=2e-12 * abs(total_charge),
        )


# Lane cases: empty and sparse cells, cells with fewer particles than lanes,
# dense cells (2000 particles, about 20 per cell) where every lane loops several
# times, a last current group of one mode (5 and 11 modes), and effective weights.
LANE_CASES = [
    (5, 0, False),
    (5, 17, False),
    (5, 257, True),
    (5, 2000, False),
    (1, 17, False),
    (2, 2000, True),
    (11, 257, True),
]


@pytest.mark.parametrize("lanes", [2, 4, 8, 32])
@pytest.mark.parametrize("fieldtype", ["rho", "J"])
@pytest.mark.parametrize("n_modes,count,effective_weights", LANE_CASES)
def test_lane_deposition_parity(
    runtime, launches, lanes, fieldtype, n_modes, count, effective_weights
):
    import cupy as cp

    data = _particle_data(count)
    charge = e if effective_weights else -e
    cpu = _species(data, "linear", False, charge, effective_weights)
    cpu_fields = _fields(n_modes, False)
    cpu.deposit(cpu_fields, fieldtype)
    cpu_fields.sum_reduce_deposition_array(fieldtype)
    components = ("rho",) if fieldtype == "rho" else ("Jr", "Jt", "Jz")
    results = []
    for lane_count in (1, lanes):
        gpu = _species(data, "linear", True, charge, effective_weights)
        gpu.deposit_lanes = {fieldtype: lane_count}
        gpu_fields = _fields(n_modes, True)
        gpu.sort_particles(gpu_fields)
        gpu.sorted = True
        before = runtime.launch_count
        launches.clear()
        gpu.deposit(gpu_fields, fieldtype)
        # Lanes change the threads per launch, not the number of launches.
        assert runtime.launch_count - before == _expected_deposit_launches(
            n_modes, "linear", fieldtype, True
        )
        # One lane uses make_kernel; more lanes use make_lane_kernel.
        suffix = f"_lanes{lane_count}"
        assert all(key.endswith(suffix) == (lane_count > 1) for key, _ in launches)
        runtime.synchronize()
        observed = []
        for gpu_grid, cpu_grid in zip(gpu_fields.interp, cpu_fields.interp):
            for name in components:
                value = cp.asnumpy(getattr(gpu_grid, name))
                _assert_scaled_close(value, getattr(cpu_grid, name))
                observed.append(value)
        results.append(observed)
    for one_lane, many_lanes in zip(*results):
        _assert_scaled_close(many_lanes, one_lane)


@pytest.fixture
def launches(monkeypatch):
    """Record (kernel key, threads per block) of every fused deposition launch."""
    from fbpic.particles.deposition import cuda_methods_fused

    recorded = []
    launch = cuda_methods_fused.launch_kernel

    def record(kernel, dimensions, args, block_dim):
        recorded.append((kernel.key, block_dim))
        launch(kernel, dimensions, args, block_dim)

    monkeypatch.setattr(cuda_methods_fused, "launch_kernel", record)
    return recorded


# Five current modes end with the one-mode group (4); six charge modes end with
# the one-mode group (5). Only those groups take the separate block size.
SINGLE_MODE_CASES = [
    ("J", 5, ["m0_n2", "m2_n2", "m4_n1"]),
    ("rho", 6, ["m0_n5", "m5_n1"]),
]


@pytest.mark.parametrize("lanes", [1, 4])
@pytest.mark.parametrize("fieldtype,n_modes,groups", SINGLE_MODE_CASES)
def test_single_mode_block_size(runtime, launches, lanes, fieldtype, n_modes, groups):
    import cupy as cp

    data = _particle_data(257)
    cpu = _species(data, "linear", False)
    cpu_fields = _fields(n_modes, False)
    cpu.deposit(cpu_fields, fieldtype)
    cpu_fields.sum_reduce_deposition_array(fieldtype)
    components = ("rho",) if fieldtype == "rho" else ("Jr", "Jt", "Jz")
    suffix = "" if lanes == 1 else f"_lanes{lanes}"
    for single_mode_tpb in (None, 16):
        gpu = _species(data, "linear", True)
        gpu.deposit_lanes = {fieldtype: lanes}
        gpu.deposit_tpb_single_mode = single_mode_tpb
        gpu_fields = _fields(n_modes, True)
        gpu.sort_particles(gpu_fields)
        gpu.sorted = True
        launches.clear()
        gpu.deposit(gpu_fields, fieldtype)
        runtime.synchronize()
        expected = [
            (
                f"fused_linear_{fieldtype}_{group}{suffix}",
                16 if single_mode_tpb and group.endswith("_n1") else 32,
            )
            for group in groups
        ]
        assert launches == expected
        for gpu_grid, cpu_grid in zip(gpu_fields.interp, cpu_fields.interp):
            for name in components:
                _assert_scaled_close(
                    cp.asnumpy(getattr(gpu_grid, name)), getattr(cpu_grid, name)
                )


@pytest.fixture
def launch_shapes(monkeypatch):
    """Record (kernel key, dimensions, threads per block) of every fused launch."""
    from fbpic.particles.deposition import cuda_methods_fused

    recorded = []
    launch = cuda_methods_fused.launch_kernel

    def record(kernel, dimensions, args, block_dim):
        recorded.append((kernel.key, tuple(dimensions), block_dim))
        launch(kernel, dimensions, args, block_dim)

    monkeypatch.setattr(cuda_methods_fused, "launch_kernel", record)
    return recorded


# Per-group lane mappings: the A100 and Ada candidates for five current modes,
# a last one-mode group after five two-mode groups (11 modes), a mapping
# without a one-mode entry when no such group exists (4 modes), one lane next
# to several lanes, and charge with a last one-mode group (6 modes). Each
# expected group is (group, lanes).
PER_GROUP_LANE_CASES = [
    ("J", 5, {2: 2, 1: 4}, [("m0_n2", 2), ("m2_n2", 2), ("m4_n1", 4)]),
    ("J", 5, {2: 4, 1: 8}, [("m0_n2", 4), ("m2_n2", 4), ("m4_n1", 8)]),
    (
        "J",
        11,
        {2: 2, 1: 4},
        [(f"m{first}_n2", 2) for first in range(0, 10, 2)] + [("m10_n1", 4)],
    ),
    ("J", 4, {2: 4}, [("m0_n2", 4), ("m2_n2", 4)]),
    ("J", 5, {2: 1, 1: 4}, [("m0_n2", 1), ("m2_n2", 1), ("m4_n1", 4)]),
    ("rho", 6, {5: 4, 1: 2}, [("m0_n5", 4), ("m5_n1", 2)]),
]


@pytest.mark.parametrize("count,effective_weights", [(257, True), (2000, False)])
@pytest.mark.parametrize("fieldtype,n_modes,setting,groups", PER_GROUP_LANE_CASES)
def test_per_group_lane_deposition(
    runtime,
    launch_shapes,
    fieldtype,
    n_modes,
    setting,
    groups,
    count,
    effective_weights,
):
    import cupy as cp

    data = _particle_data(count)
    charge = e if effective_weights else -e
    cpu = _species(data, "linear", False, charge, effective_weights)
    cpu_fields = _fields(n_modes, False)
    cpu.deposit(cpu_fields, fieldtype)
    cpu_fields.sum_reduce_deposition_array(fieldtype)
    components = ("rho",) if fieldtype == "rho" else ("Jr", "Jt", "Jz")
    results = []
    for lane_setting in (1, setting):
        gpu = _species(data, "linear", True, charge, effective_weights)
        gpu.deposit_lanes = {fieldtype: lane_setting}
        gpu_fields = _fields(n_modes, True)
        gpu.sort_particles(gpu_fields)
        gpu.sorted = True
        launch_shapes.clear()
        gpu.deposit(gpu_fields, fieldtype)
        runtime.synchronize()
        if lane_setting == setting:
            # Each group selects its own kernel and launches lanes threads per
            # cell. A grid sized for another group's lanes would still pass the
            # parity checks, because surplus threads exit at the cell bound.
            cells = gpu.prefix_sum.shape[0]
            expected = [
                (
                    f"fused_linear_{fieldtype}_{group}"
                    + ("" if lanes == 1 else f"_lanes{lanes}"),
                    (cells * lanes,),
                    32,
                )
                for group, lanes in groups
            ]
            assert launch_shapes == expected
        observed = []
        for gpu_grid, cpu_grid in zip(gpu_fields.interp, cpu_fields.interp):
            for name in components:
                value = cp.asnumpy(getattr(gpu_grid, name))
                _assert_scaled_close(value, getattr(cpu_grid, name))
                observed.append(value)
        results.append(observed)
    for one_lane, per_group in zip(*results):
        _assert_scaled_close(per_group, one_lane)


GROUP_LANES_ACCEPTED = [
    (1, 2, 2, 1),
    (4, 1, 2, 4),
    (np.int64(8), 2, 2, 8),
    ({2: 2, 1: 4}, 2, 2, 2),
    ({2: 2, 1: 4}, 1, 2, 4),
    ({5: 4, 1: 1}, 1, 5, 1),
]


@pytest.mark.parametrize("setting,mode_count,group_size,lanes", GROUP_LANES_ACCEPTED)
def test_group_lanes_resolves_settings(setting, mode_count, group_size, lanes):
    from fbpic.particles.deposition.cuda_methods_fused import group_lanes

    resolved = group_lanes(setting, mode_count, group_size)
    assert resolved == lanes
    assert type(resolved) is int


# Each case is (setting, mode count, group size, exception). Current groups
# have at most two modes and charge groups at most five.
GROUP_LANES_REJECTED = [
    (0, 2, 2, ValueError),
    (3, 2, 2, ValueError),
    (64, 2, 2, ValueError),
    (True, 2, 2, TypeError),
    (4.0, 2, 2, TypeError),
    ("4", 2, 2, TypeError),
    ({2: 3, 1: 4}, 2, 2, ValueError),
    ({2: 64}, 2, 2, ValueError),
    ({"2": 2, 1: 4}, 1, 2, TypeError),
    ({True: 2}, 1, 2, TypeError),
    ({2: True}, 2, 2, TypeError),
    ({2: np.int64(4)}, 2, 2, TypeError),
    ({np.int64(2): 4}, 2, 2, TypeError),
    ({3: 4, 2: 4}, 2, 2, ValueError),
    ({0: 4, 2: 4}, 2, 2, ValueError),
    ({2: 4}, 1, 2, KeyError),
    ({5: 4}, 1, 5, KeyError),
]


@pytest.mark.parametrize(
    "setting,mode_count,group_size,exception", GROUP_LANES_REJECTED
)
def test_group_lanes_rejects_bad_settings(setting, mode_count, group_size, exception):
    from fbpic.particles.deposition.cuda_methods_fused import group_lanes

    with pytest.raises(exception):
        group_lanes(setting, mode_count, group_size)


# A mapping without the one-mode group of five current modes, and a key that
# no current group can have. Both must fail before the first group launches.
@pytest.mark.parametrize(
    "setting,exception", [({2: 4}, KeyError), ({3: 4, 2: 4, 1: 4}, ValueError)]
)
def test_bad_lane_setting_fails_before_launch(
    runtime, launch_shapes, setting, exception
):
    data = _particle_data(257)
    gpu = _species(data, "linear", True)
    gpu.deposit_lanes = {"J": setting}
    gpu_fields = _fields(5, True)
    gpu.sort_particles(gpu_fields)
    gpu.sorted = True
    launch_shapes.clear()
    before = runtime.launch_count
    with pytest.raises(exception):
        gpu.deposit(gpu_fields, "J")
    assert launch_shapes == []
    assert runtime.launch_count == before


@pytest.mark.parametrize("lanes", [0, 1, 3, 64])
def test_lane_kernel_rejects_unsupported_lanes(lanes):
    from fbpic.particles.deposition.cuda_methods_fused import make_lane_kernel

    with pytest.raises(ValueError, match="lanes"):
        make_lane_kernel(True, 0, 2, lanes)


# The last column scales rmax_gather; 0.5 cuts the gather inside the grid.
GATHER_CASES = [
    (5, "linear", 0, 1.0),
    (5, "linear", 17, 1.0),
    (5, "linear", 257, 1.0),
    (5, "linear", 257, 0.5),
    (5, "cubic", 17, 1.0),
    (1, "linear", 17, 1.0),
    (2, "linear", 17, 1.0),
    (3, "linear", 17, 1.0),
    (6, "linear", 17, 1.0),
    (11, "linear", 257, 0.5),
    (6, "cubic", 17, 1.0),
]


@pytest.mark.parametrize("n_modes,shape,count,rmax_fraction", GATHER_CASES)
def test_gather_fusion_parity(runtime, n_modes, shape, count, rmax_fraction):
    import cupy as cp

    data = _particle_data(count)
    rmax_gather = rmax_fraction * RMAX
    comm = SimpleNamespace(get_rmax=lambda with_damp: rmax_gather)
    cpu = _species(data, shape, False)
    cpu_fields = _fields(n_modes, False)
    cpu.gather(cpu_fields.interp, comm)
    outputs = []
    for fusion in (False, True):
        gpu = _species(data, shape, True)
        gpu.fuse_gather = fusion
        gpu_fields = _fields(n_modes, True)
        before = runtime.launch_count
        gpu.gather(gpu_fields.interp, comm)
        assert runtime.launch_count - before == _expected_gather_launches(
            n_modes, shape, count, fusion
        )
        runtime.synchronize()
        observed = [cp.asnumpy(getattr(gpu, name)) for name in PARTICLE_FIELDS]
        for value, name in zip(observed, PARTICLE_FIELDS):
            _assert_scaled_close(value, getattr(cpu, name))
            radius = np.sqrt(data["x"] * data["x"] + data["y"] * data["y"])
            # Threshold parity is checked above. Exact cutoff classification
            # can differ by one ulp between hypot, CPU sqrt, and GPU sqrt.
            outside = radius > rmax_gather * (1 + 8 * np.finfo(np.float64).eps)
            np.testing.assert_array_equal(value[outside], np.zeros(outside.sum()))
        for name in ("x", "y", "z"):
            np.testing.assert_array_equal(cp.asnumpy(getattr(gpu, name)), data[name])
        assert gpu.keep_fields_sorted and not gpu.sorted
        outputs.append(observed)
    for unfused, fused in zip(*outputs):
        # Gather has no atomics, so only the mode-sum order differs.
        _assert_scaled_close(fused, unfused, tolerance=2e-13)


@pytest.mark.parametrize("active_mode", range(5))
def test_fused_gather_each_mode(runtime, active_mode):
    import cupy as cp

    data = _particle_data(17)
    comm = SimpleNamespace(get_rmax=lambda with_damp: RMAX)
    cpu = _species(data, "linear", False)
    gpu = _species(data, "linear", True)
    cpu.gather(_fields(5, False, active_mode).interp, comm)
    gpu.gather(_fields(5, True, active_mode).interp, comm)
    runtime.synchronize()
    for name in PARTICLE_FIELDS:
        _assert_scaled_close(cp.asnumpy(getattr(gpu, name)), getattr(cpu, name))


def test_neutral_species_skip_deposition_and_gather(runtime):
    import cupy as cp

    species = _species(_particle_data(17), "linear", True, charge=0.0)
    fields = _fields(5, True)
    comm = SimpleNamespace(get_rmax=lambda with_damp: RMAX)
    before = runtime.launch_count
    species.deposit(fields, "rho")
    species.deposit(fields, "J")
    species.gather(fields.interp, comm)
    assert runtime.launch_count == before
    assert not species.sorted
    runtime.synchronize()
    for index, name in enumerate(PARTICLE_FIELDS):
        np.testing.assert_array_equal(cp.asnumpy(getattr(species, name)), 17 + index)
    for grid in fields.interp:
        for name in ("rho", "Jr", "Jt", "Jz"):
            np.testing.assert_array_equal(cp.asnumpy(getattr(grid, name)), 0)


def test_gather_graph_after_sorting_and_particle_reallocation(runtime):
    import cupy as cp

    from fbpic.particles.elementary_process.cuda_numba_utils import (
        reallocate_and_copy_old,
    )
    from fbpic.utils.warp import CudaGraphSections

    species = _species(_particle_data(17), "linear", True)
    fields = _fields(5, True)
    comm = SimpleNamespace(get_rmax=lambda with_damp: RMAX)
    graphs = CudaGraphSections()

    def check_graph_result():
        with graphs.section("gather"):
            species.gather(fields.interp, comm)
        runtime.synchronize()
        # CPU reference follows the current particle order and population.
        data = {name: cp.asnumpy(getattr(species, name)) for name in PARTICLE_DATA}
        cpu = _species(data, "linear", False)
        cpu.gather(_fields(5, False).interp, comm)
        eager = _species(data, "linear", True)
        eager.gather(fields.interp, comm)
        runtime.synchronize()
        for name in PARTICLE_FIELDS:
            result = cp.asnumpy(getattr(species, name))
            _assert_scaled_close(result, getattr(cpu, name))
            np.testing.assert_array_equal(result, cp.asnumpy(getattr(eager, name)))

    def check_deposition_result():
        data = {name: cp.asnumpy(getattr(species, name)) for name in PARTICLE_DATA}
        cpu = _species(data, "linear", False)
        cpu_fields = _fields(5, False)
        for fieldtype, components in (("rho", ("rho",)), ("J", ("Jr", "Jt", "Jz"))):
            fields.erase(fieldtype)
            cpu.deposit(cpu_fields, fieldtype)
            cpu_fields.sum_reduce_deposition_array(fieldtype)
            species.deposit(fields, fieldtype)
            runtime.synchronize()
            for gpu_grid, cpu_grid in zip(fields.interp, cpu_fields.interp):
                for name in components:
                    _assert_scaled_close(
                        cp.asnumpy(getattr(gpu_grid, name)), getattr(cpu_grid, name)
                    )

    check_graph_result()
    check_graph_result()
    assert (graphs.capture_count, graphs.replay_count) == (1, 2)
    # Actual sorting rotates particle-buffer owners and changes the graph key.
    species.sort_particles(fields)
    species.sorted = True
    check_graph_result()
    assert (graphs.capture_count, graphs.replay_count) == (2, 3)
    check_deposition_result()
    old_count = species.Ntot
    new_data = _particle_data(23, seed=113)
    reallocate_and_copy_old(species, True, old_count, 23)
    for name in PARTICLE_DATA:
        getattr(species, name)[old_count:] = cp.asarray(new_data[name][old_count:])
    for name in PARTICLE_FIELDS:
        getattr(species, name)[old_count:] = 123.0
    species.sorted = False
    species.sort_particles(fields)
    species.sorted = True
    check_graph_result()
    check_graph_result()
    assert (graphs.capture_count, graphs.replay_count) == (3, 5)
    check_deposition_result()
    assert species.Ntot == 23
    assert species.sorted and species.keep_fields_sorted
