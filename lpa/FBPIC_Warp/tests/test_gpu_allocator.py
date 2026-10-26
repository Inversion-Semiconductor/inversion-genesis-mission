# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Check the CuPy device allocator and the exchange-time memory release."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from fbpic.utils.cuda import cuda_installed


pytestmark = pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")

MIB = 1 << 20


@pytest.fixture
def async_runtime():
    import fbpic.utils.warp as warp_utils

    warp_utils.get_runtime().use()
    if warp_utils.allocator_mode() != "async":
        pytest.skip("The CUDA mempool allocator is not active.")
    return warp_utils


def _counter(warp_utils, name):
    return warp_utils._mempool_attribute(name)


def test_large_and_small_allocations_use_device_mempool(async_runtime):
    import cupy

    legacy_used = cupy.get_default_memory_pool().used_bytes()
    used = _counter(async_runtime, "UsedMemCurrent")
    small = cupy.empty(4096, dtype=cupy.uint8)
    assert _counter(async_runtime, "UsedMemCurrent") >= used + small.nbytes
    large = cupy.empty(64 * MIB, dtype=cupy.uint8)
    assert (
        _counter(async_runtime, "UsedMemCurrent") >= used + small.nbytes + large.nbytes
    )
    assert cupy.get_default_memory_pool().used_bytes() == legacy_used
    del small, large


def test_out_of_memory_retries_once_after_trim(async_runtime, monkeypatch):
    import cupy

    allocate = async_runtime._allocate_async
    calls = []

    def fail_once(size):
        calls.append(size)
        if len(calls) == 1:
            raise cupy.cuda.runtime.CUDARuntimeError(2)
        return allocate(size)

    monkeypatch.setattr(async_runtime, "_allocate_async", fail_once)
    state = async_runtime._allocator_state
    retries, trims = state["oom_retries"], state["trims"]
    array = cupy.arange(1000, dtype=cupy.float64)
    assert calls == [array.nbytes, array.nbytes]
    assert float(array.sum()) == 499500.0
    assert state["oom_retries"] == retries + 1
    assert state["trims"] == trims + 1


def test_out_of_memory_raises_after_retry(async_runtime, monkeypatch):
    import cupy

    def fail(size):
        raise cupy.cuda.runtime.CUDARuntimeError(2)

    monkeypatch.setattr(async_runtime, "_allocate_async", fail)
    with pytest.raises(cupy.cuda.memory.OutOfMemoryError):
        cupy.empty(1000, dtype=cupy.float64)


def test_other_allocation_errors_are_not_retried(async_runtime, monkeypatch):
    import cupy

    calls = []

    def fail(size):
        calls.append(size)
        raise cupy.cuda.runtime.CUDARuntimeError(1)  # cudaErrorInvalidValue

    monkeypatch.setattr(async_runtime, "_allocate_async", fail)
    with pytest.raises(cupy.cuda.runtime.CUDARuntimeError):
        cupy.empty(1000, dtype=cupy.float64)
    assert len(calls) == 1


def test_release_keeps_cache_with_ample_memory(async_runtime):
    import cupy

    cupy.empty(64 * MIB, dtype=cupy.uint8)
    cupy.cuda.runtime.deviceSynchronize()
    free, total = cupy.cuda.runtime.memGetInfo()
    if free < 0.2 * total:
        pytest.skip("The device has too little free memory for this check.")
    trims = async_runtime._allocator_state["trims"]
    assert async_runtime.release_unused_device_memory() is False
    assert async_runtime._allocator_state["trims"] == trims


def test_release_trims_when_unused_memory_exceeds_cap(async_runtime, monkeypatch):
    import cupy

    monkeypatch.setattr(async_runtime, "_TRIM_MIN_SLACK_BYTES", 32 * MIB)
    monkeypatch.setattr(async_runtime, "_TRIM_SLACK_FRACTION", 0.0)
    cupy.empty(256 * MIB, dtype=cupy.uint8)
    cupy.cuda.runtime.deviceSynchronize()
    reserved = _counter(async_runtime, "ReservedMemCurrent")
    used = _counter(async_runtime, "UsedMemCurrent")
    assert reserved - used >= 256 * MIB
    assert async_runtime.release_unused_device_memory() is True
    assert _counter(async_runtime, "ReservedMemCurrent") <= reserved - 256 * MIB
    assert async_runtime.release_unused_device_memory() is False


def test_release_trims_when_free_memory_is_low(async_runtime, monkeypatch):
    import cupy

    free, total = cupy.cuda.runtime.memGetInfo()
    monkeypatch.setattr(
        cupy.cuda.runtime, "memGetInfo", lambda: (int(0.05 * total), total)
    )
    trims = async_runtime._allocator_state["trims"]
    assert async_runtime.release_unused_device_memory() is True
    assert async_runtime._allocator_state["trims"] == trims + 1


def test_pool_mode_keeps_legacy_allocator():
    # The allocator is selected once per process, so check it in a new one.
    script = textwrap.dedent(
        """
        import cupy
        import fbpic.utils.warp as warp_utils

        warp_utils.get_runtime().use()
        assert warp_utils.allocator_mode() == "pool"
        pool = cupy.get_default_memory_pool()
        array = cupy.empty(64 << 20, dtype=cupy.uint8)
        assert pool.used_bytes() >= array.nbytes
        del array
        assert pool.total_bytes() >= 64 << 20
        assert warp_utils.release_unused_device_memory() is True
        assert pool.total_bytes() == 0
        """
    )
    package_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, FBPIC_GPU_ALLOCATOR="pool")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(package_root), env.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=package_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
