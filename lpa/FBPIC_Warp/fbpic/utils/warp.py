# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Share CuPy storage and one CUDA stream with Warp kernels."""

import functools
import inspect
import math
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
            with wp.ScopedCapture(
                stream=runtime.stream, force_module_load=False
            ) as capture:
                for command in commands:
                    command.launch(stream=runtime.stream)
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
