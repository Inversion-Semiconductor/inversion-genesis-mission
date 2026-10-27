# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Give the per-particle GPU arrays a capacity, so that size changes reuse memory.

Each per-particle CuPy array of a species is a prefix view ``base[:Ntot]`` of a
base array that this module allocates. Kernels see only the view, so their
``shape[0]`` is ``Ntot`` as before. When the species grows within the capacity,
the arrays become longer views of the same bases: there is no allocation and
no copy.

All arrays of a species have the same capacity, ``species.particle_capacity``.
A sort and an exchange swap the bases between the arrays of a species (see
``Particles._permute_arrays``), so one array with less capacity would make the
next growth allocate. The exchange restores this invariant for every array
(`provision_particle_arrays`).

Capacities are size classes (16 classes per octave), so that repeated
allocations request the same sizes and the device pool can reuse freed blocks.

The headroom of a species follows its measured growth. The species counts
the particles that it gains within the steps (ionization, Compton
scattering) in ``species.grown_since_exchange``, and keeps its count after
the last exchange in ``species.count_at_exchange``. The relative headroom is

    h = max(species.capacity_headroom, GROWTH_PERIODS * g / n),

where ``n`` is the number of particles and ``g`` the larger of the growth
within the steps and the net growth ``n - count_at_exchange`` (which also
includes the injection of the exchange). At an exchange, ``g`` is the growth
of the last exchange period; within a step, the growth since the exchange.
The capacity target is ``capacity_for(n, h)``. A species that stops growing
returns to ``capacity_headroom`` (1/32 by default).

Memory bound: after an exchange, the capacity ``c`` of each species is
renewed unless ``n * (1 + h/2) <= c <= SHRINK_FACTOR * capacity_for(n, h)``.
`size_class` rounds up by less than 1/16, so

    c < SHRINK_FACTOR * (17/16) * (1 + h) * n    (or the minimum, 1024).

A species without growth within the steps (``h = 1/32``) holds less than
1.37 n elements in each particle array. A growing species holds less than
1.33 (1 + h) n, which is the growth of `GROWTH_PERIODS` exchange periods at
the measured rate, plus at most 33%. Within a step, the particle count only
grows, so the ratio only decreases until an overflow; an overflow sets
``c < (17/16) * (1 + h) * n``, and the old arrays stay alive until the
copy is queued. The shared workspaces add the sort block (16 bytes times
the largest capacity) and the ionization arrays (`reserve_workspace`); they
only grow until the end of ``Simulation.step``.

Only bases that this module allocated grow in place. A prefix view of another
block (for example a row of a 2D array) reports its own size as its capacity,
so it is never extended over the memory of a neighbour.

The module also keeps the shared step workspaces (`reserve_workspace`): device
scratch that one step reuses, such as the particle sort block and the
ionization arrays. One CUDA stream orders all their users, so the species can
share them. An out-of-memory retry drops them (see
``fbpic.utils.warp.register_out_of_memory_callback``).
"""
import weakref

import numpy as np

from fbpic.utils.cuda import cuda_installed

if cuda_installed:
    import cupy

    from fbpic.utils.warp import register_out_of_memory_callback

# Default (minimum) relative headroom of the particle arrays of a species
DEFAULT_HEADROOM = 1.0 / 32
# The headroom holds the measured growth within the steps of this many
# exchange periods
GROWTH_PERIODS = 2.5
# At an exchange, a capacity above SHRINK_FACTOR times the target is released
SHRINK_FACTOR = 1.25
_MIN_CAPACITY = 1024
_CLASS_BITS = 4

# Bases allocated by this module, by id (weak references)
_bases = weakref.WeakValueDictionary()
# Shared step workspaces, by name
_workspaces = {}


def size_class(n):
    """Return the smallest size class (5 significant bits) of at least `n`."""
    n = max(int(n), _MIN_CAPACITY)
    shift = max(n.bit_length() - (_CLASS_BITS + 1), 0)
    return -(-n >> shift) << shift


def capacity_for(n, headroom):
    """Return the capacity of `n` particles with a relative `headroom`."""
    n = int(n)
    return size_class(n + int(np.ceil(n * headroom)))


def _owned_base(array):
    """Return the base of `array` if this module allocated it, else None."""
    if not isinstance(array, cupy.ndarray) or array.ndim != 1:
        return None
    if not array.flags.c_contiguous:
        return None
    base = array if array.base is None else array.base
    if _bases.get(id(base)) is not base:
        return None
    if base.dtype != array.dtype or base.data.ptr != array.data.ptr:
        return None
    return base


def capacity(array):
    """Return the number of elements that `array` can grow to in place."""
    base = _owned_base(array)
    return array.size if base is None else base.size


def empty_particle_array(n, dtype, cap):
    """Allocate a base of `cap` elements and return its first `n` elements."""
    base = cupy.empty(max(int(cap), int(n)), dtype=dtype)
    _bases[id(base)] = base
    return base[: int(n)]


def fit_particle_array(array, n, cap, keep=0, exact=False, dtype=None):
    """Return an `n`-element particle array whose first `keep` elements are
    those of `array`.

    The result is a view of the base of `array` when that base has at least
    `n` elements (and exactly `cap` elements when `exact` is True). Otherwise
    the result is a new array of capacity `cap`, and the copy of `keep`
    elements is queued on the current stream.

    Args:
        array: A 1D CuPy array, or None (then `dtype` is required).
        n: Number of elements of the result.
        cap: Capacity of a new base (at least `n`).
        keep: Number of leading elements to preserve (at most `array.size`).
        exact: Also reallocate when the capacity of `array` is not `cap`.
        dtype: Element type of a new array; the default is ``array.dtype``.

    Returns:
        A CuPy array of shape ``(n,)``. `array` must not be used afterwards
        if the result is a view of its base.
    """
    n = int(n)
    base = _owned_base(array)
    if base is not None and n <= base.size and (not exact or base.size == cap):
        return array if array.size == n else base[:n]
    if base is None and not exact and isinstance(array, cupy.ndarray):
        if array.shape == (n,):
            return array
    new = empty_particle_array(n, array.dtype if dtype is None else dtype, cap)
    if keep > 0:
        _copy_prefix(keep, array, new)
    return new


def _copy_prefix(count, source, target):
    """Copy the first `count` elements of `source` into `target` (GPU)."""
    # Late import: cuda_numba_utils imports this module
    from fbpic.particles.elementary_process.cuda_numba_utils import (
        copy_particle_data_cuda,
    )
    from fbpic.utils.cuda import cuda_tpb_bpg_1d

    grid, block = cuda_tpb_bpg_1d(count)
    copy_particle_data_cuda[grid, block](count, source, target)


def to_device_particle_array(array, cap):
    """Return a device copy of the particle array `array`, with capacity `cap`.

    Args:
        array: A 1D NumPy or CuPy array. A CuPy array whose base already has
            capacity `cap` is returned as it is.
        cap: Capacity of the device array.

    Returns:
        A CuPy array with the shape, dtype and values of `array`.
    """
    if isinstance(array, cupy.ndarray):
        return fit_particle_array(array, array.size, cap, keep=array.size, exact=True)
    array = np.asarray(array)
    new = empty_particle_array(array.size, array.dtype, cap)
    if array.size > 0:
        new.set(np.ascontiguousarray(array))
    return new


def growth_headroom(species, n):
    """Return the relative headroom of `species` for `n` particles.

    The headroom is the larger of ``species.capacity_headroom`` and
    `GROWTH_PERIODS` times the growth since the last exchange, relative to
    `n`. The growth is the larger of the growth within the steps
    (``species.grown_since_exchange``) and the net growth
    ``n - species.count_at_exchange``.
    """
    headroom = species.capacity_headroom
    grown = int(getattr(species, "grown_since_exchange", 0))
    count = getattr(species, "count_at_exchange", None)
    if count is not None:
        grown = max(grown, int(n) - int(count))
    if grown > 0 and n > 0:
        headroom = max(headroom, GROWTH_PERIODS * grown / n)
    return headroom


def reserve_particle_capacity(species, n, rebalance=False):
    """Return the capacity target of `species` for `n` particles.

    Within a step (`rebalance` False), the target only grows, when `n` exceeds
    it. At an exchange (`rebalance` True), the target is also renewed when its
    headroom falls below half of the `growth_headroom` or when it exceeds
    `SHRINK_FACTOR` times the renewed target (or is below the minimum
    capacity). This hysteresis keeps the target, and so the bases, unchanged
    over several exchanges. The calls of one exchange use the same growth,
    so that they agree on the target (`start_growth_period` resets it after
    the exchange).

    Args:
        species: A Particles object. Sets ``species.particle_capacity``.
        n: Number of particles.
        rebalance: Whether the caller is an exchange.

    Returns:
        The capacity target, at least `n`.
    """
    n = int(n)
    headroom = growth_headroom(species, n)
    cap = int(getattr(species, "particle_capacity", 0))
    if rebalance:
        target = capacity_for(n, headroom)
        if (
            cap < max(n + n * headroom / 2, _MIN_CAPACITY)
            or cap > SHRINK_FACTOR * target
        ):
            cap = target
    elif cap < n:
        cap = capacity_for(n, headroom)
    species.particle_capacity = cap
    return cap


def start_growth_period(species):
    """Start the measurement of the growth of `species` (at the end of an
    exchange, after the last `reserve_particle_capacity` of the exchange):
    reset ``species.grown_since_exchange`` and store ``species.Ntot`` in
    ``species.count_at_exchange``."""
    species.grown_since_exchange = 0
    species.count_at_exchange = getattr(species, "Ntot", None)


def grow_particle_arrays(species, old_Ntot, new_Ntot):
    """Resize all particle arrays of `species` from `old_Ntot` to `new_Ntot`
    particles, within a step.

    The particle data and the fields of the first `old_Ntot` particles are
    kept; the other elements are undefined. When `new_Ntot` fits in the
    capacity of the species, no array is allocated. Otherwise all arrays move
    to new bases with a larger capacity (see `growth_headroom`).

    Args:
        species: A Particles object with its data on the GPU. Sets
            ``species.Ntot`` and adds the growth to
            ``species.grown_since_exchange``.
        old_Ntot, new_Ntot: Number of particles before and after.
    """
    species.grown_since_exchange = int(
        getattr(species, "grown_since_exchange", 0)
    ) + max(int(new_Ntot) - int(old_Ntot), 0)
    cap = reserve_particle_capacity(species, new_Ntot)
    for obj, name, kind, dtype in species.particle_arrays():
        array = getattr(obj, name, None)
        if array is None and kind != "scratch":
            continue
        if array is not None and not isinstance(array, cupy.ndarray):
            continue
        keep = 0 if kind == "scratch" else old_Ntot
        setattr(
            obj, name, fit_particle_array(array, new_Ntot, cap, keep=keep, dtype=dtype)
        )
    species.Ntot = new_Ntot


def provision_particle_arrays(species, keep_fields=True):
    """Give every particle array of `species` exactly the capacity target, for
    ``species.Ntot`` particles (at an exchange, where allocation is allowed).

    Args:
        species: A Particles object with its data on the GPU.
        keep_fields: Whether to keep the fields on the particles of a
            reallocated field array. The particle data is always kept; the
            scratch arrays never.
    """
    n = int(species.Ntot)
    cap = reserve_particle_capacity(species, n, rebalance=True)
    for obj, name, kind, dtype in species.particle_arrays():
        array = getattr(obj, name, None)
        if array is None and kind != "scratch":
            continue
        if array is not None and not isinstance(array, cupy.ndarray):
            continue
        keep = 0
        if array is not None and (kind == "data" or (kind == "field" and keep_fields)):
            keep = min(n, array.size)
        setattr(
            obj,
            name,
            fit_particle_array(array, n, cap, keep=keep, exact=True, dtype=dtype),
        )


def reserve_workspace(name, count, dtype):
    """Return the shared workspace `name`, with at least `count` elements.

    Args:
        name: Workspace name.
        count: Number of elements needed.
        dtype: Element type.

    Returns:
        ``(buffer, grown)``: the 1D CuPy buffer, and whether this call
        allocated it. The content is undefined.
    """
    buffer = _workspaces.get(name)
    if buffer is not None and buffer.size >= count and buffer.dtype == dtype:
        return buffer, False
    # Release the old buffer before the new one is allocated
    _workspaces.pop(name, None)
    buffer = None
    buffer = cupy.empty(size_class(count), dtype=dtype)
    _workspaces[name] = buffer
    return buffer, True


def workspace_view(name, shape, dtype):
    """Return a C-contiguous view of `shape` from the shared workspace `name`."""
    count = int(np.prod(shape))
    buffer, _ = reserve_workspace(name, count, dtype)
    return buffer[:count].reshape(shape)


def release_workspaces():
    """Drop the shared workspaces (their memory returns to the pool once the
    views that the caller still holds are gone)."""
    _workspaces.clear()


if cuda_installed:
    # An allocation that runs out of memory drops the workspaces before its
    # retry; the next `reserve_workspace` allocates them again.
    register_out_of_memory_callback(release_workspaces)
