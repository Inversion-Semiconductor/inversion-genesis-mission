# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Check the GPU particle sort against a host stable argsort, bit for bit.

The sort must reproduce the stable permutation of the original thrust argsort:
the in-cell order sets the deposition summation order and the exchange slices.
Device-to-host copies belong to verification only, outside simulation hot paths.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from fbpic.particles import Particles
from fbpic.utils.cuda import cuda_installed

pytestmark = pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")

PARTICLE_DATA = ("x", "y", "z", "ux", "uy", "uz", "w", "inv_gamma")
PARTICLE_FIELDS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
NZ, NR = 64, 32
DZ, DR = 0.03, 0.04
ZMIN = -0.2


@pytest.fixture
def runtime():
    from fbpic.utils.warp import get_runtime

    runtime = get_runtime()
    runtime.use()
    yield runtime
    runtime.synchronize()


def _grid():
    return SimpleNamespace(
        invdz=1 / DZ, zmin=ZMIN, Nz=NZ, invdr=1 / DR, rmin=0.0, Nr=NR
    )


def _host_data(count, n_int, seed):
    rng = np.random.default_rng(seed)
    # z extends past both ends (periodic wrap of iz) and r past rmax (ir == Nr).
    z = rng.uniform(ZMIN - 0.9 * DZ, ZMIN + (NZ + 0.4) * DZ, count)
    radius = rng.uniform(0, (NR + 2) * DR, count)
    theta = rng.uniform(-np.pi, np.pi, count)
    if count >= 8:
        # Cell centres: below zmin, above zmax, and beyond rmax.
        z[:4] = ZMIN + DZ * np.array([-0.75, -0.6, NZ + 0.25, NZ + 0.3])
        radius[4:8] = DR * np.array([NR + 0.75, NR + 1.5, 0.25, 0.0])
    if count >= 64:
        # Exact duplicates of earlier particles share their cell.
        z[-16:], radius[-16:], theta[-16:] = z[:16], radius[:16], theta[:16]
    data = {
        "x": radius * np.cos(theta),
        "y": radius * np.sin(theta),
        "z": z,
    }
    for name in ("ux", "uy", "uz", "w", "inv_gamma", *PARTICLE_FIELDS):
        data[name] = rng.normal(size=count)
    if count >= 3:
        # Negative zero and NaN payloads: only a bit copy preserves them.
        data["ux"][:3] = np.array(
            [0x8000000000000000, 0x7FF0000000000001, 0x7FF8000000000123],
            dtype=np.uint64,
        ).view(np.float64)
    levels = rng.integers(0, 8, count).astype(np.uint64)
    if n_int >= 1:
        data["ionization_level"] = levels
        data["w_times_level"] = data["w"] * levels.astype(np.float64)
    if n_int == 2:
        data["id"] = (rng.permutation(count) + 2**40).astype(np.uint64)
    return data


def _host_cell_idx(data):
    radius = np.sqrt(data["x"] * data["x"] + data["y"] * data["y"])
    r_cell = (1 / DR) * (radius - 0.0) - 0.5
    z_cell = (1 / DZ) * (data["z"] - ZMIN) - 0.5
    ir = np.minimum(np.ceil(r_cell).astype(np.int32), NR)
    iz = np.ceil(z_cell).astype(np.int32)
    iz = np.where(iz < 0, iz + NZ, np.where(iz > NZ - 1, iz - NZ, iz))
    return (ir + iz * (NR + 1)).astype(np.int32)


def _set_gpu_data(species, data):
    """Load host data, as the reallocation paths do after a change of Ntot."""
    import cupy as cp

    count = data["x"].size
    species.Ntot = count
    for name in (*PARTICLE_DATA, *PARTICLE_FIELDS):
        setattr(species, name, cp.asarray(data[name]))
    species.sorting_buffer = cp.empty(count, dtype=np.float64)
    species.ionizer = species.tracker = None
    species.n_integer_quantities = 0
    if "ionization_level" in data:
        species.ionizer = SimpleNamespace(
            w_times_level=cp.asarray(data["w_times_level"]),
            ionization_level=cp.asarray(data["ionization_level"]),
        )
        species.n_integer_quantities += 1
    if "id" in data:
        species.tracker = SimpleNamespace(id=cp.asarray(data["id"]))
        species.n_integer_quantities += 1
    if species.n_integer_quantities > 0:
        species.int_sorting_buffer = cp.empty(count, dtype=np.uint64)


def _species(data, keep_fields_sorted):
    import cupy as cp

    species = Particles.__new__(Particles)
    species.use_cuda = True
    species.keep_fields_sorted = keep_fields_sorted
    species.spin_tracker = None
    species.prefix_sum = cp.empty(NZ * (NR + 1), dtype=np.int32)
    species.prefix_sum_shift = 0
    species.sorted = False
    _set_gpu_data(species, data)
    return species


def _gpu_arrays(species):
    arrays = {name: getattr(species, name) for name in PARTICLE_DATA}
    arrays.update({name: getattr(species, name) for name in PARTICLE_FIELDS})
    if species.ionizer is not None:
        arrays["w_times_level"] = species.ionizer.w_times_level
        arrays["ionization_level"] = species.ionizer.ionization_level
    if species.tracker is not None:
        arrays["id"] = species.tracker.id
    return arrays


def _bits(array):
    array = np.asarray(array)
    return array.view(np.uint64) if array.dtype == np.float64 else array


def _permute_launches(count):
    # Chunks of 4 arrays, then a remainder of 3 = 2 + 1, 2, or 1.
    return count // 4 + (0, 1, 1, 2)[count % 4]


def _sort_and_check(runtime, species, data, keep_fields_sorted):
    import cupy as cp

    species.keep_fields_sorted = keep_fields_sorted
    # The sort block is released at the end of the sort: copy it before.
    captured = {}
    rearrange = Particles.rearrange_particle_arrays.__get__(species)

    def capture():
        captured["sorted_idx"] = species.sorted_idx.copy()
        captured["cell_idx"] = species.cell_idx.copy()
        rearrange()

    species.rearrange_particle_arrays = capture
    before = runtime.launch_count
    species.sort_particles(SimpleNamespace(interp=[_grid()]))
    launches = runtime.launch_count - before
    runtime.synchronize()
    assert species.cell_idx is None and species.sorted_idx is None
    if species.Ntot > 0:
        n_float = len(PARTICLE_DATA) + ("w_times_level" in data)
        n_float += len(PARTICLE_FIELDS) if keep_fields_sorted else 0
        n_int = ("ionization_level" in data) + ("id" in data)
        # Cell index, prefix-sum prefill and prefix sum, then the permutes.
        expected = 3 + _permute_launches(n_float) + _permute_launches(n_int)
        assert launches == expected

    keys = _host_cell_idx(data)
    perm = np.argsort(keys, kind="stable")
    sorted_keys = keys[perm]
    assert captured["sorted_idx"].dtype == np.int32
    assert captured["cell_idx"].dtype == np.int32
    np.testing.assert_array_equal(cp.asnumpy(captured["sorted_idx"]), perm)
    np.testing.assert_array_equal(cp.asnumpy(captured["cell_idx"]), sorted_keys)
    # prefix_sum[c] counts the particles in cells 0 to c (inclusive).
    expected_prefix = np.searchsorted(
        sorted_keys, np.arange(NZ * (NR + 1)), side="right"
    ).astype(np.int32)
    np.testing.assert_array_equal(cp.asnumpy(species.prefix_sum), expected_prefix)

    arrays = _gpu_arrays(species)
    checked = [name for name in arrays if name not in PARTICLE_FIELDS]
    if keep_fields_sorted:
        checked += list(PARTICLE_FIELDS)
    for name in checked:
        observed = cp.asnumpy(arrays[name])
        np.testing.assert_array_equal(_bits(observed), _bits(data[name][perm]), name)
    for name, array in arrays.items():
        # Unsorted field arrays are scratch, but keep their shape and dtype.
        assert array.shape == (species.Ntot,), name
        assert array.dtype == data[name].dtype, name
    # The host data now follows the sorted order (for repeated sorts).
    return {name: values[perm] for name, values in data.items()}


def _pointers(species):
    """Return the GPU allocations that a sort rotates between arrays."""
    pointers = [array.data.ptr for array in _gpu_arrays(species).values()]
    pointers.append(species.sorting_buffer.data.ptr)
    if species.n_integer_quantities > 0:
        pointers.append(species.int_sorting_buffer.data.ptr)
    return pointers


@pytest.mark.parametrize("keep_fields_sorted", [False, True])
@pytest.mark.parametrize("n_int", [0, 1, 2])
@pytest.mark.parametrize("count", [0, 1, 7, 1_000_003])
def test_sort_matches_host_reference(runtime, count, n_int, keep_fields_sorted):
    data = _host_data(count, n_int, seed=count + 11 * n_int)
    species = _species(data, keep_fields_sorted)
    before = _pointers(species)
    data = _sort_and_check(runtime, species, data, keep_fields_sorted)
    # A second sort of sorted particles is the identity permutation.
    _sort_and_check(runtime, species, data, keep_fields_sorted)
    after = _pointers(species)
    if not keep_fields_sorted:
        # Dead field arrays serve as the float buffers: no allocation.
        assert getattr(species, "_sort_slot_float1", None) is None
    if count > 0:
        # The sort swaps buffers but never loses or duplicates an allocation.
        assert len(set(after)) == len(after)
        assert set(before) <= set(after) | set(_extra_slot_pointers(species))


def _extra_slot_pointers(species):
    return [
        getattr(species, name).data.ptr
        for name in dir(species)
        if name.startswith("_sort_slot") and getattr(species, name) is not None
    ]


@pytest.mark.parametrize("n_int", [1, 2])
def test_sort_after_change_of_ntot(runtime, n_int):
    # Growth, a large growth, a shrink, an empty species, and a small one.
    for step, count in enumerate([40_000, 48_000, 120_000, 3_001, 0, 5]):
        data = _host_data(count, n_int, seed=100 + step)
        if step == 0:
            species = _species(data, keep_fields_sorted=False)
        else:
            _set_gpu_data(species, data)
        _sort_and_check(runtime, species, data, keep_fields_sorted=step % 2 == 1)


def test_sort_rejects_int32_overflow():
    from fbpic.particles.utilities.cuda_sorting import allocate_sort_scratch

    species = SimpleNamespace(Ntot=2**31)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        allocate_sort_scratch(species)
