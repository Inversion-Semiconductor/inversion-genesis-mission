# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Share CuPy storage and one CUDA stream with Warp kernels."""

import functools
import gc
import inspect
import math
import os
import time
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    import cupy


@wp.func
def cmul(a: wp.vec2d, b: wp.vec2d):
    return wp.vec2d(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def mul_i(a: wp.vec2d):
    return wp.vec2d(-a[1], a[0])


# wp.pow with a float64 exponent compiles to the full ::pow(double, double).
@wp.func
def square(a: wp.float64):
    return a * a


@wp.func
def cube(a: wp.float64):
    return a * a * a


class CudaRuntime:
    """Own the shared CUDA stream for one device.

    CuPy owns allocations and the stream. Warp borrows both resources.
    The runtime remains alive for the Python process.
    """

    def __init__(self, ordinal: int) -> None:
        import cupy

        self.ordinal = ordinal
        self.device = wp.get_device(f"cuda:{ordinal}")
        with cupy.cuda.Device(ordinal):
            self.cupy_stream = cupy.cuda.Stream(non_blocking=False)
            _install_allocator(ordinal)
        self.stream = wp.Stream(self.device, cuda_stream=self.cupy_stream.ptr)
        self.launch_count = 0
        self.recorded_launches = None

    def use(self) -> None:
        """Select the device and shared stream in CuPy and Warp."""
        import cupy

        cupy.cuda.Device(self.ordinal).use()
        self.cupy_stream.use()
        wp.set_device(self.device)
        wp.set_stream(self.stream)

    def synchronize(self) -> None:
        """Wait for work on the shared stream."""
        self.cupy_stream.synchronize()


_runtimes: dict[int, CudaRuntime] = {}

# Device allocator
# ----------------
# Particle arrays change size at every exchange, so the legacy CuPy pool
# cannot reuse their freed blocks. It had to return them with cudaFree, which
# drains the device. The stream-ordered CUDA mempool reuses freed ranges and
# frees on the shared stream, without a device synchronization.

_GIB = 1 << 30
# The slack is the reserved but unused memory of the mempool. Trim it when it
# exceeds the cap max(_TRIM_MIN_SLACK_BYTES, _TRIM_SLACK_FRACTION * total
# device memory), and keep half of the cap for the next exchanges. Also trim
# when less than _TRIM_FREE_FRACTION of the device memory is free and at least
# _TRIM_MIN_RELEASE_BYTES of slack can be released.
_TRIM_MIN_SLACK_BYTES = 2 * _GIB
_TRIM_SLACK_FRACTION = 0.1
_TRIM_FREE_FRACTION = 0.1
_TRIM_MIN_RELEASE_BYTES = 256 << 20
_CUDA_ERROR_MEMORY_ALLOCATION = 2  # cudaErrorMemoryAllocation
_RELEASE_NEVER = 2**64 - 1

_allocator_state = {
    "mode": None,
    "releases": 0,
    "trims": 0,
    "oom_retries": 0,
    # Slack that the last trim could not release (partly used chunks).
    "trim_residual": 0,
    "history": [],
    "trim_log": [],
}


def _select_allocator_mode(ordinal):
    import cupy
    from fbpic.utils.mpi import gpudirect_enabled

    mode = os.environ.get("FBPIC_GPU_ALLOCATOR", "async")
    if mode not in ("async", "pool"):
        raise ValueError(
            f"FBPIC_GPU_ALLOCATOR must be 'async' or 'pool', not {mode!r}."
        )
    runtime = cupy.cuda.runtime
    # Default-mempool memory cannot be exported through CUDA IPC, which
    # CUDA-aware MPI can use for transfers between ranks on one node.
    if mode == "async" and (
        gpudirect_enabled
        or not runtime.deviceGetAttribute(
            runtime.cudaDevAttrMemoryPoolsSupported, ordinal
        )
    ):
        mode = "pool"
    return mode


def _install_allocator(ordinal):
    """Select the CuPy allocator once per process; configure each device pool."""
    import cupy

    if _allocator_state["mode"] is None:
        _allocator_state["mode"] = _select_allocator_mode(ordinal)
        if _allocator_state["mode"] == "async":
            # cupy.cuda.MemoryAsyncPool would create a CUDA context on every
            # visible device (about 0.4 GiB each), so allocate directly.
            cupy.cuda.set_allocator(_malloc_async)
    if _allocator_state["mode"] == "async":
        runtime = cupy.cuda.runtime
        # Keep freed memory in the pool across synchronizations.
        runtime.memPoolSetAttribute(
            runtime.deviceGetMemPool(ordinal),
            runtime.cudaMemPoolAttrReleaseThreshold,
            _RELEASE_NEVER,
        )


def _allocate_async(size):
    import cupy

    memory = cupy.cuda.memory.MemoryAsync(size, cupy.cuda.get_current_stream())
    return cupy.cuda.memory.MemoryPointer(memory, 0)


def _is_out_of_memory(error):
    import cupy

    if isinstance(error, cupy.cuda.memory.OutOfMemoryError):
        return True
    return (
        isinstance(error, cupy.cuda.runtime.CUDARuntimeError)
        and error.status == _CUDA_ERROR_MEMORY_ALLOCATION
    )


def _malloc_async(size):
    """Allocate on the current stream; release cached memory and retry once."""
    import cupy

    try:
        return _allocate_async(size)
    except (MemoryError, RuntimeError) as error:
        if not _is_out_of_memory(error):
            raise
    _allocator_state["oom_retries"] += 1
    gc.collect()
    _trim_device_memory()
    try:
        return _allocate_async(size)
    except (MemoryError, RuntimeError) as error:
        if not _is_out_of_memory(error):
            raise
        used = _mempool_attribute("UsedMemCurrent")
        raise cupy.cuda.memory.OutOfMemoryError(size, used, 0) from error


def _mempool_attribute(name):
    import cupy

    runtime = cupy.cuda.runtime
    pool = runtime.deviceGetMemPool(runtime.getDevice())
    return int(
        runtime.memPoolGetAttribute(pool, getattr(runtime, f"cudaMemPoolAttr{name}"))
    )


def _trim_device_memory(keep_bytes=0):
    """Return unused cached memory of the current device to CUDA.

    The mempool keeps at least ``keep_bytes`` reserved.
    """
    import cupy

    runtime = cupy.cuda.runtime
    # Pending cudaFreeAsync calls release memory only once they execute.
    runtime.deviceSynchronize()
    cupy.get_default_memory_pool().free_all_blocks()
    runtime.memPoolTrimTo(runtime.deviceGetMemPool(runtime.getDevice()), keep_bytes)
    _allocator_state["trims"] += 1


def allocator_mode() -> str | None:
    """Return the CuPy allocator of this process.

    Returns:
        ``"async"`` for the stream-ordered CUDA mempool, ``"pool"`` for the
        legacy CuPy pool, or None before the first ``CudaRuntime`` exists.
        ``FBPIC_GPU_ALLOCATOR=pool`` selects the legacy pool. GPUDirect MPI
        and devices without mempool support also select it.
    """
    return _allocator_state["mode"]


def release_unused_device_memory() -> bool:
    """Return cached device memory to CUDA when the cache is large.

    With the legacy pool, always free all unused CuPy blocks; this calls
    cudaFree and synchronizes the device. With the CUDA mempool, read the
    pool counters (no synchronization) and trim in two cases:

    - The slack (reserved minus used memory) grew by more than the cap
      max(2 GiB, 10% of the device) since the last trim. The trim keeps
      used memory plus half of the cap reserved.
    - Less than 10% of the device memory is free, and at least 256 MiB of
      slack can be released. The trim keeps only used memory.

    Slack that a trim cannot release (partly used chunks) does not count
    towards the next trim, so fragmentation cannot cause a trim at each call.
    A trim synchronizes the device first.

    Returns:
        True if cached memory was released.
    """
    import cupy

    _allocator_state["releases"] += 1
    if _allocator_state["mode"] != "async":
        cupy.get_default_memory_pool().free_all_blocks()
        return True
    reserved = _mempool_attribute("ReservedMemCurrent")
    used = _mempool_attribute("UsedMemCurrent")
    free, total = cupy.cuda.runtime.memGetInfo()
    _allocator_state["history"].append((reserved, used, free))
    slack_cap = int(max(_TRIM_MIN_SLACK_BYTES, _TRIM_SLACK_FRACTION * total))
    releasable = reserved - used - _allocator_state["trim_residual"]
    if free < _TRIM_FREE_FRACTION * total:
        if releasable < _TRIM_MIN_RELEASE_BYTES:
            return False
        keep_slack = 0
    elif releasable > slack_cap:
        keep_slack = slack_cap // 2
    else:
        return False
    start = time.perf_counter()
    _trim_device_memory(used + keep_slack)
    seconds = time.perf_counter() - start
    reserved_after = _mempool_attribute("ReservedMemCurrent")
    used_after = _mempool_attribute("UsedMemCurrent")
    _allocator_state["trim_residual"] = max(0, reserved_after - used_after - keep_slack)
    _allocator_state["trim_log"].append(
        (len(_allocator_state["history"]) - 1, reserved, reserved_after, seconds)
    )
    return True


def device_memory_stats() -> dict:
    """Report the allocator mode and the memory counters of the current device.

    Returns:
        A dict with ``mode``, ``releases`` (calls of
        ``release_unused_device_memory``), ``trims`` and ``oom_retries``.
        ``history`` lists ``(reserved, used, free)`` bytes at each release
        call in async mode. ``trim_log`` lists ``(history index, reserved
        before, reserved after, host seconds)`` for each trim at a release
        call. In async mode, the dict also has the current and peak reserved
        and used bytes of the CUDA mempool.
    """
    stats = {
        key: _allocator_state[key]
        for key in ("mode", "releases", "trims", "oom_retries")
    }
    stats["history"] = [list(entry) for entry in _allocator_state["history"]]
    stats["trim_log"] = [list(entry) for entry in _allocator_state["trim_log"]]
    if _allocator_state["mode"] == "async":
        for name in (
            "ReservedMemCurrent",
            "ReservedMemHigh",
            "UsedMemCurrent",
            "UsedMemHigh",
        ):
            stats[name] = _mempool_attribute(name)
    return stats


def get_runtime(ordinal: int | None = None) -> CudaRuntime:
    """Return the persistent runtime for the selected CuPy device.

    Args:
        ordinal: Visible CUDA device index. The current device is the default.

    Returns:
        The runtime that owns the shared, non-null CUDA stream.
    """
    import cupy

    if ordinal is None:
        ordinal = cupy.cuda.runtime.getDevice()
    if ordinal not in _runtimes:
        _runtimes[ordinal] = CudaRuntime(ordinal)
        _runtimes[ordinal].use()
    return _runtimes[ordinal]


_array_dtypes = {
    np.dtype(np.float64): wp.float64,
    np.dtype(np.float32): wp.float32,
    np.dtype(np.int32): wp.int32,
    np.dtype(np.int64): wp.int64,
    np.dtype(np.uint32): wp.uint32,
    np.dtype(np.uint64): wp.uint64,
    np.dtype(np.int16): wp.int16,
    np.dtype(np.uint16): wp.uint16,
    np.dtype(np.uint8): wp.uint8,
    np.dtype(np.int8): wp.int8,
    np.dtype(np.bool_): wp.bool,
    np.dtype(np.complex128): wp.vec2d,
    np.dtype(np.complex64): wp.vec2f,
}


def as_warp_array(array: "cupy.ndarray", dtype: Any = None) -> wp.array:
    """Borrow a CuPy array without copying or changing its dimensions.

    Complex elements become vectors containing real and imaginary parts.
    The returned view retains the CuPy owner, shape, strides, and device.
    A new view uses the current allocation after particle buffer swaps.

    Args:
        array: CuPy array with a supported numerical dtype.
        dtype: Expected Warp element type, or ``None`` to infer the type.

    Returns:
        A mutable Warp view that shares the original GPU allocation.

    Raises:
        TypeError: The array dtype is unsupported or differs from ``dtype``.
    """
    actual_dtype = _array_dtypes.get(array.dtype)
    if actual_dtype is None:
        raise TypeError(f"Unsupported Warp view dtype: {array.dtype}")
    if dtype is not None and dtype is not Any and dtype != actual_dtype:
        raise TypeError(f"Expected {dtype}, received CuPy dtype {array.dtype}")
    view = wp.array(
        ptr=array.data.ptr,
        shape=array.shape,
        strides=array.strides,
        dtype=actual_dtype,
        device=f"cuda:{array.device.id}",
        copy=False,
    )
    view._cupy_owner = array
    return view


class WarpKernel:
    """Retain FBPIC's kernel launch syntax with typed Warp kernels."""

    def __init__(self, function: Callable[..., Any], module: Any = None) -> None:
        self.kernel = wp.kernel(function, enable_backward=False, module=module)
        self.signature = inspect.signature(function)
        self.annotations = tuple(
            parameter.annotation for parameter in self.signature.parameters.values()
        )
        functools.update_wrapper(self, function)

    def __getitem__(self, launch_shape: tuple) -> Callable[..., None]:
        blocks, threads = launch_shape
        blocks = (blocks,) if isinstance(blocks, (int, np.integer)) else tuple(blocks)
        threads = (
            (threads,) if isinstance(threads, (int, np.integer)) else tuple(threads)
        )
        if len(blocks) != len(threads):
            raise ValueError("Block and thread dimensions must match.")
        dimensions = tuple(int(b) * int(t) for b, t in zip(blocks, threads))
        block_dim = math.prod(threads)

        def launch(*args):
            import cupy

            if len(args) != len(self.annotations):
                raise TypeError(
                    f"{self.__name__} expects {len(self.annotations)} arguments; "
                    f"received {len(args)}."
                )
            converted = []
            for value, annotation in zip(args, self.annotations):
                if isinstance(value, cupy.ndarray):
                    # launch_kernel checks that the arrays share the device.
                    converted.append(
                        as_warp_array(value, getattr(annotation, "dtype", None))
                    )
                elif isinstance(value, np.generic):
                    converted.append(value.item())
                else:
                    converted.append(value)
            launch_kernel(self.kernel, dimensions, converted, block_dim)

        return launch


def warp_kernel(function: Callable[..., Any], module: Any = None) -> WarpKernel:
    """Compile a forward Warp kernel with the existing FBPIC launch syntax."""
    return WarpKernel(function, module=module)


def launch_kernel(
    kernel: wp.Kernel, dimensions: tuple[int, ...], args: list, block_dim: int
) -> None:
    """Launch a Warp kernel on the shared stream, or record it in a graph section.

    Args:
        kernel: Compiled Warp kernel.
        dimensions: Launch dimensions (threads, not blocks). A zero-size launch
            is skipped.
        args: Kernel arguments: Warp arrays, Warp structs, or scalars. Arrays,
            including arrays inside structs, must be on the current device.
        block_dim: Threads per block.

    Raises:
        ValueError: An array argument is on another CUDA device.
    """
    if math.prod(dimensions) == 0:
        return
    runtime = get_runtime()
    for arg in args:
        for array in _arrays_in(arg):
            if array.device.ordinal != runtime.ordinal:
                raise ValueError("Kernel arrays must share the selected CUDA device.")
    launch = (kernel, tuple(dimensions), args, block_dim)
    if runtime.recorded_launches is not None:
        runtime.recorded_launches.append(launch)
    else:
        _launch(runtime, launch)


def _struct_fields(arg):
    """Return the field values of a Warp struct instance, or None."""
    struct = getattr(arg, "_cls", None)
    if struct is None or not hasattr(struct, "vars"):
        return None
    return [getattr(arg, name) for name in struct.vars]


def _arrays_in(arg):
    """Yield the Warp arrays in an argument, including nested struct fields."""
    if isinstance(arg, wp.array):
        yield arg
        return
    fields = _struct_fields(arg)
    if fields is not None:
        for field in fields:
            yield from _arrays_in(field)


def _launch(runtime, launch, record_cmd=False):
    kernel, dimensions, args, block_dim = launch
    command = wp.launch(
        kernel,
        dim=dimensions,
        inputs=args,
        stream=runtime.stream,
        block_dim=block_dim,
        record_cmd=record_cmd,
    )
    if not record_cmd:
        runtime.launch_count += 1
    return command


def _arg_key(arg):
    if isinstance(arg, wp.array):
        return (arg.ptr, arg.shape, arg.strides, arg.dtype, arg.device.ordinal)
    if isinstance(arg, float):
        # Preserve signed zero and NaN payloads when checking scalar changes.
        return (float, np.float64(arg).tobytes())
    fields = _struct_fields(arg)
    if fields is not None:
        # Key structs by contents: callers build a new instance for each launch.
        return (arg._cls, tuple(_arg_key(field) for field in fields))
    return (type(arg), arg)


def _launch_key(launch):
    kernel, dimensions, args, block_dim = launch
    return (kernel, dimensions, block_dim, *(_arg_key(arg) for arg in args))


class CudaGraphSections:
    """Cache graphs for sections containing only Warp kernel launches.

    Host orchestration runs on each call. Kernels execute once at section exit.
    Changed arguments invalidate captures, including scalar values and pointers.
    Dynamic CuPy operations, diagnostics, and MPI must remain outside sections.
    """

    def __init__(self) -> None:
        self.cache = {}
        self.capture_count = 0
        self.replay_count = 0

    def section(self, name: str) -> "_CudaGraphSection":
        """Record a named section and replay its current GPU operations."""
        return _CudaGraphSection(self, name)

    def clear(self) -> None:
        """Release graphs and references to their device arrays."""
        self.cache.clear()

    def _execute(self, name, runtime, launches):
        if not launches:
            self.cache.pop(name, None)
            return
        key = tuple(_launch_key(launch) for launch in launches)
        cached = self.cache.get(name)
        if cached is None or cached[0] != key:
            # Recording compiles kernels and resolves generic argument types.
            # It does not advance particles, fields, or random number states.
            commands = [
                _launch(runtime, launch, record_cmd=True) for launch in launches
            ]
            # A CuPy array collected here would call cudaFreeAsync on the
            # capturing stream and put that free into the graph.
            gc_enabled = gc.isenabled()
            gc.disable()
            try:
                with wp.ScopedCapture(
                    stream=runtime.stream, force_module_load=False
                ) as capture:
                    for command in commands:
                        command.launch(stream=runtime.stream)
            finally:
                if gc_enabled:
                    gc.enable()
            # Retain array owners for the complete lifetime of the graph.
            cached = (key, capture.graph, launches)
            self.cache[name] = cached
            self.capture_count += 1
        wp.capture_launch(cached[1], stream=runtime.stream)
        runtime.launch_count += len(launches)
        self.replay_count += 1


class _CudaGraphSection:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name

    def __enter__(self):
        self.runtime = get_runtime()
        if self.runtime.recorded_launches is not None:
            raise RuntimeError("CUDA graph sections cannot be nested.")
        self.runtime.use()
        self.runtime.recorded_launches = []
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        launches = self.runtime.recorded_launches
        self.runtime.recorded_launches = None
        if exc_type is None:
            self.owner._execute(self.name, self.runtime, launches)
