# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This files contains cuda methods that are used in the boosted-frame
diagnostics
"""
import numpy as np
from typing import Any
import warp as wp
from scipy.constants import c
from fbpic.utils.cuda import cupy, cuda_tpb_bpg_1d, pinned_array
from fbpic.utils.warp import get_runtime, warp_kernel


def extract_slice_from_gpu(pref_sum_curr, N_area, species):
    """
    Extract the particles which have which have index between pref_sum_curr
    and pref_sum_curr + N_area, and return them in dictionaries.

    Parameters
    ----------
    pref_sum_curr: int
        The starting index needed for the extraction process
    N_area: int
        The number of particles to extract.
    species: an fbpic Species object
        The species from to extract data

    Returns
    -------
    particle_data : A dictionary of 1D float arrays (that are on the CPU)
        A dictionary that contains the particle data of
        the simulation (with normalized weigths), including optional
        integer arrays (e.g. "id", "charge")
    """
    # Call kernel that extracts particles from GPU
    dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d(N_area)
    # - General particle quantities
    part_data = cupy.empty((8, N_area), dtype=np.float64)
    extract_particles_from_gpu[dim_grid_1d, dim_block_1d](
        pref_sum_curr,
        species.x,
        species.y,
        species.z,
        species.ux,
        species.uy,
        species.uz,
        species.w,
        species.inv_gamma,
        part_data,
    )
    # - Optional particle arrays
    if species.tracker is not None:
        selected_particle_id = cupy.empty((N_area,), dtype=np.uint64)
        extract_array_from_gpu[dim_grid_1d, dim_block_1d](
            pref_sum_curr, species.tracker.id, selected_particle_id
        )
    if species.ionizer is not None:
        selected_particle_charge = cupy.empty((N_area,), dtype=np.uint64)
        extract_array_from_gpu[dim_grid_1d, dim_block_1d](
            pref_sum_curr, species.ionizer.ionization_level, selected_particle_charge
        )
        selected_particle_weight = cupy.empty((N_area,), dtype=np.float64)
        extract_array_from_gpu[dim_grid_1d, dim_block_1d](
            pref_sum_curr, species.ionizer.w_times_level, selected_particle_weight
        )

    # Copy GPU arrays to the host
    part_data = part_data.get()
    particle_data = {
        "x": part_data[0],
        "y": part_data[1],
        "z": part_data[2],
        "ux": part_data[3],
        "uy": part_data[4],
        "uz": part_data[5],
        "w": part_data[6],
        "inv_gamma": part_data[7],
    }
    if species.tracker is not None:
        particle_data["id"] = selected_particle_id.get()
    if species.ionizer is not None:
        particle_data["charge"] = selected_particle_charge.get()
        # Replace particle weight
        particle_data["w"] = selected_particle_weight.get()

    # Return the data as dictionary
    return particle_data


@warp_kernel
def extract_particles_from_gpu(
    part_idx_start: int,
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    z: wp.array(dtype=wp.float64),
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    w: wp.array(dtype=wp.float64),
    inv_gamma: wp.array(dtype=wp.float64),
    selected: wp.array2d(dtype=wp.float64),
):
    """
    Extract a selection of particles from the GPU and
    store them in a 2D array (8, N_part) in the following
    order: x, y, z, ux, uy, uz, w, inv_gamma.
    Selection goes from starting index (part_idx_start)
    to (part_idx_start + N_part-1), where N_part is derived
    from the shape of the 2D array (selected).

    Parameters
    ----------
    part_idx_start : int
        The starting index needed for the extraction process.
        ( minimum particle index to be extracted )

    x, y, z, ux, uy, uz, w, inv_gamma : 1D arrays of floats
        The GPU particle arrays for a given species.

    selected : 2D array of floats
        An empty GPU array to store the particles
        that are extracted.
    """
    i = wp.tid()
    N_part = selected.shape[1]

    if i < N_part:
        ptcl_idx = part_idx_start + i
        selected[0, i] = x[ptcl_idx]
        selected[1, i] = y[ptcl_idx]
        selected[2, i] = z[ptcl_idx]
        selected[3, i] = ux[ptcl_idx]
        selected[4, i] = uy[ptcl_idx]
        selected[5, i] = uz[ptcl_idx]
        selected[6, i] = w[ptcl_idx]
        selected[7, i] = inv_gamma[ptcl_idx]


@warp_kernel
def extract_array_from_gpu(
    part_idx_start: int,
    array: wp.array(dtype=Any),
    selected: wp.array(dtype=Any),
):
    """
    Extract a selection of particles from the GPU and
    store them in a 1D array (N_part,)
    Selection goes from starting index (part_idx_start)
    to (part_idx_start + N_part-1), where N_part is derived
    from the shape of the array `selected`.

    Parameters
    ----------
    part_idx_start : int
        The starting index needed for the extraction process.
        ( minimum particle index to be extracted )

    array : 1D arrays of ints or floats
        The GPU particle arrays for a given species. (e.g. particle id)

    selected : 1D array of ints or floats
        An empty GPU array to store the particles that are extracted.
    """
    i = wp.tid()
    N_part = selected.shape[0]

    if i < N_part:
        selected[i] = array[part_idx_start + i]


# -----------------------------------------------------------------------------
# Batched back-transformed particle diagnostics
# -----------------------------------------------------------------------------
# A launch of `catch_crossing_particles` receives up to BT_MAX_RANGES particle
# ranges by value (one per active snapshot), so it needs no host-to-device copy.
BT_MAX_RANGES = 32
bt_range_ints = wp.types.vector(length=BT_MAX_RANGES, dtype=wp.int32)
bt_range_floats = wp.types.vector(length=BT_MAX_RANGES, dtype=wp.float64)
_BT_MAX_RANGES = wp.constant(BT_MAX_RANGES)
# Columns of a staging row: x, y, z, ux, uy, uz, w, inv_gamma, sort key
BT_ROW_WIDTH = 9
# Sort key of a staged particle: entry * BT_KEY_SCALE + particle index.
# (Exact in float64 for particle indices below 2**31 and entries below 2**21.)
BT_KEY_SCALE = 2**31
_BT_KEY_SCALE = wp.constant(wp.float64(BT_KEY_SCALE))
_INT32_MAX = 2**31 - 1
_C_LIGHT = wp.constant(wp.float64(c))


@warp_kernel
def extract_column_ends(
    prefix_sum: wp.array(dtype=wp.int32),
    Nr: int,
    column_ends: wp.array(dtype=wp.int32),
):
    """
    Copy the last prefix-sum value of each z column into a contiguous array:
    column_ends[iz] = prefix_sum[(iz + 1) * (Nr + 1) - 1]
    (i.e. the number of particles in the cells with z index <= iz)
    """
    iz = wp.tid()
    if iz < column_ends.shape[0]:
        column_ends[iz] = prefix_sum[(iz + 1) * (Nr + 1) - 1]


class ColumnEndCopy:
    """Copy the column ends of a species' prefix sum to the host after each sort.

    An instance is registered in ``species.sort_listeners``. After each sort,
    it launches `extract_column_ends` and an asynchronous copy of the
    ``Nz`` int32 values into page-locked memory, on the shared stream.
    The diagnostics then read the prefix sum at the end of any z column
    without a blocking device-to-host transfer at diagnostic time.
    (``prefix_sum`` is only written in ``Particles.sort_particles``.)
    """

    def __init__(self):
        self.device = None
        self.host = None
        self.event = None
        # True when the host copy corresponds to the current prefix sum
        self.valid = False

    def __call__(self, species):
        runtime = get_runtime()
        if runtime.recorded_launches is not None:
            # A CUDA graph section would defer the kernel, but not the copy
            self.valid = False
            return
        Nz, Nr = species.grid_shape
        if self.device is None:
            self.device = cupy.empty(Nz, dtype=np.int32)
            self.host = pinned_array(Nz, np.int32)
            self.event = cupy.cuda.Event(disable_timing=True)
        dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d(Nz)
        extract_column_ends[dim_grid_1d, dim_block_1d](
            species.prefix_sum, Nr, self.device
        )
        cupy.cuda.runtime.memcpyAsync(
            self.host.ctypes.data,
            self.device.data.ptr,
            self.device.nbytes,
            cupy.cuda.runtime.memcpyDeviceToHost,
            runtime.cupy_stream.ptr,
        )
        self.event.record(runtime.cupy_stream)
        self.valid = True

    def read(self, species):
        """
        Return the column ends of the current prefix sum of `species`
        (a host array of shape (Nz,), valid until the next sort).
        Waits for the copy of the last sort, which normally finished long ago.
        """
        if not self.valid:
            self(species)
        self.event.synchronize()
        return self.host


def column_end_copy(species):
    """
    Return the `ColumnEndCopy` listener of `species`, and register
    a new one if needed.
    """
    if not hasattr(species, "sort_listeners"):
        species.sort_listeners = []
    for listener in species.sort_listeners:
        if isinstance(listener, ColumnEndCopy):
            return listener
    listener = ColumnEndCopy()
    species.sort_listeners.append(listener)
    return listener


@warp_kernel
def catch_crossing_particles(
    n_ranges: int,
    n_area: int,
    range_start: bt_range_ints,
    range_offset: bt_range_ints,
    range_entry: bt_range_ints,
    z_boost: bt_range_floats,
    prev_z_boost: bt_range_floats,
    dt: wp.float64,
    tol_z: wp.float64,
    has_lower: int,
    uz_lower: wp.float64,
    has_upper: int,
    uz_upper: wp.float64,
    gamma_boost: wp.float64,
    beta_gamma_boost: wp.float64,
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    z: wp.array(dtype=wp.float64),
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    w: wp.array(dtype=wp.float64),
    inv_gamma: wp.array(dtype=wp.float64),
    copy_id: int,
    ids: wp.array(dtype=wp.uint64),
    copy_level: int,
    levels: wp.array(dtype=wp.uint64),
    count: wp.array(dtype=wp.int32),
    rows: wp.array2d(dtype=wp.float64),
    int_rows: wp.array2d(dtype=wp.uint64),
):
    """
    Append to `rows` the particles of the given ranges that may have crossed
    the output plane of their snapshot during the last step.

    Thread i handles element i of the concatenated ranges; range k starts at
    particle range_start[k] and at element range_offset[k] (padding entries
    have range_offset = INT32_MAX). The test is a superset of the host tests
    in `get_particle_slice` and (for "uz") `apply_selection`: only the
    quantities that the host recomputes (previous z, lab-frame uz) get a
    tolerance against rounding differences. The host repeats the exact tests.

    `count` (initialized to 0) receives the number of appended particles.
    Row r+1 of `rows` holds appended particle r: x, y, z, ux, uy, uz, w,
    inv_gamma and the key entry * 2**31 + particle index, which restores the
    particle order on the host. `int_rows` holds the id and ionization level.
    """
    i = wp.tid()
    if i >= n_area:
        return
    # Find the range of this element
    k = int(0)
    for j in range(1, _BT_MAX_RANGES):
        if j < n_ranges and range_offset[j] <= i:
            k = j
    p = range_start[k] + (i - range_offset[k])

    # Crossing of the output plane during the last step
    z_curr = z[p]
    shift = uz[p] * inv_gamma[p] * _C_LIGHT * dt
    z_prev = z_curr - shift
    tol = tol_z + wp.float64(1.0e-12) * (wp.abs(z_curr) + wp.abs(shift))
    zb = z_boost[k]
    zb_prev = prev_z_boost[k]
    keep = (z_curr >= zb and z_prev <= zb_prev + tol) or (
        z_curr <= zb and z_prev >= zb_prev - tol
    )
    # Selection on the lab-frame longitudinal momentum
    if keep and (has_lower != 0 or has_upper != 0):
        u_boost = gamma_boost * uz[p]
        u_frame = (wp.float64(1.0) / inv_gamma[p]) * beta_gamma_boost
        uz_lab = u_boost + u_frame
        tol_u = wp.float64(1.0e-9) * (wp.abs(u_boost) + wp.abs(u_frame))
        if has_lower != 0:
            if not (uz_lab > uz_lower - tol_u):
                keep = False
        if has_upper != 0:
            if not (uz_lab < uz_upper + tol_u):
                keep = False
    if not keep:
        return

    slot = wp.atomic_add(count, 0, 1)
    if slot >= rows.shape[0] - 1:
        return
    r = slot + 1
    rows[r, 0] = x[p]
    rows[r, 1] = y[p]
    rows[r, 2] = z_curr
    rows[r, 3] = ux[p]
    rows[r, 4] = uy[p]
    rows[r, 5] = uz[p]
    rows[r, 6] = w[p]
    rows[r, 7] = inv_gamma[p]
    rows[r, 8] = wp.float64(range_entry[k]) * _BT_KEY_SCALE + wp.float64(p)
    if copy_id != 0:
        int_rows[r, 0] = ids[p]
    if copy_level != 0:
        int_rows[r, 1] = levels[p]


class CrossingStaging:
    """Persistent staging buffers of one species for the batched GPU path.

    `enqueue` launches `catch_crossing_particles` and an asynchronous copy of
    the first rows to page-locked memory; it does not wait for the GPU.
    `collect` waits for that copy, copies any remaining rows (rare), and
    returns the particles in the order of the particle arrays.
    `collect` must run before the next `enqueue`.
    The device buffers only grow, so steady-state steps allocate nothing.
    """

    def __init__(self, copy_floor=256, adaptive_copy=True):
        # Minimum number of rows of the asynchronous copy
        self.copy_floor = copy_floor
        # Whether the number of copied rows follows the recent counts
        self.adaptive_copy = adaptive_copy
        self.hint = copy_floor
        # Number of `collect` calls that needed a blocking copy
        self.fallbacks = 0
        self.capacity = 0
        self.host_capacity = -1
        self.rows = None
        self.int_rows = None
        self.count = None
        self.host_rows = None
        self.host_int_rows = None
        self.event = None
        self.dummy = None
        self.copied = None
        self.with_ints = False

    def _reserve(self, n_area, with_ints):
        if self.event is None:
            self.event = cupy.cuda.Event(disable_timing=True)
            self.dummy = cupy.zeros(1, dtype=np.uint64)
        if n_area > self.capacity:
            self.capacity = max(n_area, (3 * self.capacity) // 2, 1024)
            self.rows = cupy.empty((self.capacity + 1, BT_ROW_WIDTH))
            # The counter shares the first (header) row of `rows`,
            # so that a single copy brings back the count and the rows
            self.count = self.rows[0, :1].view(np.int32)
            self.int_rows = None
        if with_ints and self.int_rows is None:
            self.int_rows = cupy.empty((self.capacity + 1, 2), dtype=np.uint64)

    def _reserve_host(self, n_rows):
        if n_rows > self.host_capacity:
            self.host_capacity = max(n_rows, 2 * self.host_capacity)
            self.host_rows = pinned_array((self.host_capacity + 1, BT_ROW_WIDTH))
            self.host_int_rows = pinned_array((self.host_capacity + 1, 2), np.uint64)

    def enqueue(
        self,
        species,
        ranges,
        z_boost,
        prev_z_boost,
        dt,
        tol_z,
        uz_bounds,
        gamma_boost,
        beta_gamma_boost,
    ):
        """
        Launch the extraction for the particle ranges of one step.

        Parameters
        ----------
        species: an fbpic Particles object (on the GPU)
        ranges: list of (entry, start, n) tuples, with n > 0
            Entry index (used to split the result), first particle index
            and number of particles of each range
        z_boost, prev_z_boost: lists of floats
            Current and previous output plane of each entry
        dt: float
            Time step of the simulation
        tol_z: float
            Tolerance on the previous particle position
        uz_bounds: tuple (lower, upper) of floats or None
            Selection bounds on the lab-frame "uz"
        gamma_boost, beta_gamma_boost: floats
        """
        if self.copied is not None:
            # The host would read rows that the new launch overwrites
            raise RuntimeError("Call `collect` before the next `enqueue`.")
        runtime = get_runtime()
        stream = runtime.cupy_stream
        with_id = species.tracker is not None
        with_level = species.ionizer is not None
        self.with_ints = with_id or with_level
        n_area = sum(n for _, _, n in ranges)
        self._reserve(n_area, self.with_ints)
        cupy.cuda.runtime.memsetAsync(self.rows.data.ptr, 0, 4, stream.ptr)

        w = species.ionizer.w_times_level if with_level else species.w
        ids = species.tracker.id if with_id else self.dummy
        levels = species.ionizer.ionization_level if with_level else self.dummy
        int_rows = self.int_rows if self.with_ints else self.dummy.reshape(1, 1)
        lower, upper = uz_bounds
        for first in range(0, len(ranges), BT_MAX_RANGES):
            chunk = ranges[first : first + BT_MAX_RANGES]
            pad = BT_MAX_RANGES - len(chunk)
            offsets = []
            n_chunk = 0
            for _, _, n in chunk:
                offsets.append(n_chunk)
                n_chunk += n
            entries = [entry for entry, _, _ in chunk]
            dim_grid_1d, dim_block_1d = cuda_tpb_bpg_1d(n_chunk)
            catch_crossing_particles[dim_grid_1d, dim_block_1d](
                len(chunk),
                n_chunk,
                bt_range_ints(*([start for _, start, _ in chunk] + [0] * pad)),
                bt_range_ints(*(offsets + [_INT32_MAX] * pad)),
                bt_range_ints(*(entries + [0] * pad)),
                bt_range_floats(*([z_boost[e] for e in entries] + [0.0] * pad)),
                bt_range_floats(*([prev_z_boost[e] for e in entries] + [0.0] * pad)),
                dt,
                tol_z,
                int(lower is not None),
                0.0 if lower is None else float(lower),
                int(upper is not None),
                0.0 if upper is None else float(upper),
                gamma_boost,
                beta_gamma_boost,
                species.x,
                species.y,
                species.z,
                species.ux,
                species.uy,
                species.uz,
                w,
                species.inv_gamma,
                int(with_id),
                ids,
                int(with_level),
                levels,
                self.count,
                self.rows,
                int_rows,
            )

        # Copy the header row and the first rows to the host, asynchronously
        n_copy = min(n_area, max(self.copy_floor, self.hint))
        self._reserve_host(n_copy)
        cupy.cuda.runtime.memcpyAsync(
            self.host_rows.ctypes.data,
            self.rows.data.ptr,
            (n_copy + 1) * BT_ROW_WIDTH * 8,
            cupy.cuda.runtime.memcpyDeviceToHost,
            stream.ptr,
        )
        if self.with_ints:
            cupy.cuda.runtime.memcpyAsync(
                self.host_int_rows.ctypes.data,
                self.int_rows.data.ptr,
                (n_copy + 1) * 2 * 8,
                cupy.cuda.runtime.memcpyDeviceToHost,
                stream.ptr,
            )
        self.event.record(stream)
        self.copied = n_copy

    def collect(self):
        """
        Wait for the last `enqueue` and return its particles.

        Returns
        -------
        rows: 2darray of floats, shape (n, BT_ROW_WIDTH)
        int_rows: 2darray of uint64, shape (n, 2), or None
        entry: 1darray of int64, shape (n,)
            The entry index of each row. The rows are sorted by entry,
            then by particle index (i.e. in the order of the particle arrays).
        """
        self.event.synchronize()
        n = int(self.host_rows[0].view(np.int32)[0])
        n_host = min(n, self.copied)
        rows = self.host_rows[1 : n_host + 1]
        int_rows = self.host_int_rows[1 : n_host + 1] if self.with_ints else None
        if n > self.copied:
            # More particles than copied rows: blocking copy of the rest
            # (the device buffers are intact until the next `enqueue`)
            self.fallbacks += 1
            stream = get_runtime().cupy_stream
            rest = np.empty((n - n_host, BT_ROW_WIDTH))
            cupy.cuda.runtime.memcpyAsync(
                rest.ctypes.data,
                self.rows[n_host + 1 :].data.ptr,
                rest.nbytes,
                cupy.cuda.runtime.memcpyDeviceToHost,
                stream.ptr,
            )
            if self.with_ints:
                int_rest = np.empty((n - n_host, 2), dtype=np.uint64)
                cupy.cuda.runtime.memcpyAsync(
                    int_rest.ctypes.data,
                    self.int_rows[n_host + 1 :].data.ptr,
                    int_rest.nbytes,
                    cupy.cuda.runtime.memcpyDeviceToHost,
                    stream.ptr,
                )
            stream.synchronize()
            rows = np.concatenate([rows, rest])
            if self.with_ints:
                int_rows = np.concatenate([int_rows, int_rest])
        if self.adaptive_copy:
            self.hint = max(self.copy_floor, 2 * n, self.hint // 2)
        self.copied = None

        # Restore the particle order (the appends are unordered);
        # fancy indexing also copies the rows out of the staging buffers
        keys = rows[:, BT_ROW_WIDTH - 1].astype(np.int64)
        order = np.argsort(keys)
        rows = rows[order]
        if int_rows is not None:
            int_rows = int_rows[order]
        return rows, int_rows, keys[order] // BT_KEY_SCALE
