# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Check the CuPy device allocator and the exchange-time memory release."""

import gc
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import warp as wp

from fbpic.utils.cuda import cuda_installed


pytestmark = pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")

MIB = 1 << 20
TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TESTS_DIR.parent


@wp.kernel
def _increment(values: wp.array(dtype=wp.float64)):
    index = wp.tid()
    values[index] = values[index] + wp.float64(1.0)


def _async_unavailable_reason(warp_utils):
    """Return why this process may legitimately use the legacy pool."""
    import cupy

    from fbpic.utils.mpi import gpudirect_enabled

    if os.environ.get("FBPIC_GPU_ALLOCATOR", "async") != "async":
        return "FBPIC_GPU_ALLOCATOR selects another allocator."
    if gpudirect_enabled:
        return "GPUDirect MPI selects the legacy pool."
    runtime = cupy.cuda.runtime
    ordinal = warp_utils.get_runtime().ordinal
    if not runtime.deviceGetAttribute(runtime.cudaDevAttrMemoryPoolsSupported, ordinal):
        return "The device does not support CUDA mempools."
    return None


@pytest.fixture
def async_runtime(monkeypatch):
    import fbpic.utils.warp as warp_utils

    warp_utils.get_runtime().use()
    if warp_utils.allocator_mode() != "async":
        reason = _async_unavailable_reason(warp_utils)
        if reason is None:
            pytest.fail("The default allocator is not the CUDA mempool.")
        pytest.skip(reason)
    # Earlier trims must not change the trim decisions of a test.
    monkeypatch.setitem(warp_utils._allocator_state, "trim_residual", 0)
    return warp_utils


def _run_python(script, **env_overrides):
    """Run a script in a new process, which selects its allocator again."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("FBPIC_GPU_ALLOCATOR", "FBPIC_ENABLE_GPUDIRECT")
    }
    env.update(env_overrides)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(PACKAGE_ROOT), str(TESTS_DIR), env.get("PYTHONPATH")])
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _small_simulation():
    from scipy.constants import c, e, m_e

    from fbpic.main import Simulation

    sim = Simulation(
        32,
        10.0e-6,
        16,
        8.0e-6,
        2,
        10.0e-6 / (32 * c),
        zmin=0.0,
        n_order=-1,
        use_cuda=True,
        boundaries={"z": "open", "r": "reflective"},
        verbose_level=0,
    )
    sim.add_new_species(
        q=-e,
        m=m_e,
        n=1.0e24,
        p_zmin=1.0e-6,
        p_zmax=9.0e-6,
        p_rmax=6.0e-6,
        p_nz=1,
        p_nr=1,
        p_nt=4,
    )
    return sim


def _exchange_count(sim, n_steps):
    """Count the exchange iterations of ``Simulation.step(n_steps)``."""
    period = sim.comm.exchange_period
    return sum(
        (sim.iteration + i_step) % period == 0 or i_step == 0
        for i_step in range(n_steps)
    )


def test_default_allocator_is_cuda_mempool():
    import fbpic.utils.warp as warp_utils

    warp_utils.get_runtime().use()
    reason = _async_unavailable_reason(warp_utils)
    if reason is not None:
        pytest.skip(reason)
    assert warp_utils.allocator_mode() == "async"


def test_allocator_mode_selection(monkeypatch):
    import cupy

    import fbpic.utils.mpi as mpi_utils
    import fbpic.utils.warp as warp_utils

    ordinal = warp_utils.get_runtime().ordinal
    runtime = cupy.cuda.runtime
    supported = runtime.deviceGetAttribute(
        runtime.cudaDevAttrMemoryPoolsSupported, ordinal
    )
    monkeypatch.delenv("FBPIC_GPU_ALLOCATOR", raising=False)
    monkeypatch.setattr(mpi_utils, "gpudirect_enabled", False)
    expected = "async" if supported else "pool"
    assert warp_utils._select_allocator_mode(ordinal) == expected
    monkeypatch.setenv("FBPIC_GPU_ALLOCATOR", "pool")
    assert warp_utils._select_allocator_mode(ordinal) == "pool"
    monkeypatch.setenv("FBPIC_GPU_ALLOCATOR", "bogus")
    with pytest.raises(ValueError, match="FBPIC_GPU_ALLOCATOR"):
        warp_utils._select_allocator_mode(ordinal)
    monkeypatch.setenv("FBPIC_GPU_ALLOCATOR", "async")
    monkeypatch.setattr(mpi_utils, "gpudirect_enabled", True)
    assert warp_utils._select_allocator_mode(ordinal) == "pool"
    monkeypatch.setattr(mpi_utils, "gpudirect_enabled", False)
    monkeypatch.setattr(runtime, "deviceGetAttribute", lambda attribute, device: 0)
    assert warp_utils._select_allocator_mode(ordinal) == "pool"


def test_invalid_allocator_name_fails_at_runtime_creation():
    result = _run_python(
        """
        import fbpic.utils.warp as warp_utils

        try:
            warp_utils.get_runtime()
        except ValueError as error:
            assert "FBPIC_GPU_ALLOCATOR" in str(error)
        else:
            raise AssertionError("An invalid allocator name was accepted.")
        """,
        FBPIC_GPU_ALLOCATOR="bogus",
    )
    assert result.returncode == 0, result.stderr


def test_gpudirect_selects_legacy_pool():
    pytest.importorskip("mpi4py")
    result = _run_python(
        """
        import fbpic.utils.warp as warp_utils
        from fbpic.utils.mpi import gpudirect_enabled

        assert gpudirect_enabled
        warp_utils.get_runtime().use()
        assert warp_utils.allocator_mode() == "pool"
        """,
        FBPIC_ENABLE_GPUDIRECT="1",
    )
    assert result.returncode == 0, result.stderr


def _counter(warp_utils, name):
    return warp_utils._mempool_attribute(name)


def _empty_mempool(warp_utils):
    """Release earlier slack; skip if live arrays would change the threshold."""
    gc.collect()
    warp_utils._trim_device_memory()
    if _counter(warp_utils, "UsedMemCurrent") > 64 * MIB:
        pytest.skip("Other live arrays would hide the used-memory threshold.")


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


def test_out_of_memory_retry_drops_step_workspaces(async_runtime, monkeypatch):
    import cupy
    import numpy as np

    from fbpic.particles.utilities import capacity

    unused, _ = capacity.reserve_workspace("test_unused", 1 << 20, np.int32)
    del unused
    in_use = capacity.workspace_view("test_in_use", (4096,), np.float64)
    in_use[:] = 2.0
    allocate = async_runtime._allocate_async
    calls = []

    def fail_once(size):
        calls.append(size)
        if len(calls) == 1:
            raise cupy.cuda.runtime.CUDARuntimeError(2)
        return allocate(size)

    monkeypatch.setattr(async_runtime, "_allocate_async", fail_once)
    cupy.empty(1000, dtype=cupy.float64)
    assert len(calls) == 2
    # The retry dropped the workspaces; a view in use keeps its memory
    assert "test_unused" not in capacity._workspaces
    assert "test_in_use" not in capacity._workspaces
    assert float(in_use.sum()) == 2.0 * 4096
    monkeypatch.setattr(async_runtime, "_allocate_async", allocate)
    _, grown = capacity.reserve_workspace("test_unused", 1 << 20, np.int32)
    assert grown
    capacity.release_workspaces()


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
    _empty_mempool(async_runtime)
    cupy.empty(256 * MIB, dtype=cupy.uint8)
    cupy.cuda.runtime.deviceSynchronize()
    reserved = _counter(async_runtime, "ReservedMemCurrent")
    used = _counter(async_runtime, "UsedMemCurrent")
    assert reserved - used >= 256 * MIB
    assert async_runtime.release_unused_device_memory() is True
    # The trim keeps at least half of the threshold (at most 32 MiB here),
    # rounded up to whole pool chunks (32 MiB on an RTX 6000 Ada).
    reserved_after = _counter(async_runtime, "ReservedMemCurrent")
    assert reserved_after < reserved
    assert reserved_after - _counter(async_runtime, "UsedMemCurrent") <= 64 * MIB
    assert async_runtime.release_unused_device_memory() is False


def test_release_keeps_slack_below_used_memory(async_runtime, monkeypatch):
    import cupy

    monkeypatch.setattr(async_runtime, "_TRIM_MIN_SLACK_BYTES", 32 * MIB)
    monkeypatch.setattr(async_runtime, "_TRIM_SLACK_FRACTION", 0.0)
    _empty_mempool(async_runtime)
    live = cupy.empty(512 * MIB, dtype=cupy.uint8)
    cupy.empty(256 * MIB, dtype=cupy.uint8)
    cupy.cuda.runtime.deviceSynchronize()
    trims = async_runtime._allocator_state["trims"]
    # Like the particle arrays replaced at an exchange, this slack is
    # smaller than the live data, so it is kept for reuse.
    assert async_runtime.release_unused_device_memory() is False
    del live
    cupy.cuda.runtime.deviceSynchronize()
    assert async_runtime.release_unused_device_memory() is True
    assert async_runtime._allocator_state["trims"] == trims + 1


def test_release_trims_when_free_memory_is_low(async_runtime, monkeypatch):
    import cupy

    cupy.empty(512 * MIB, dtype=cupy.uint8)
    cupy.cuda.runtime.deviceSynchronize()
    free, total = cupy.cuda.runtime.memGetInfo()
    monkeypatch.setattr(
        cupy.cuda.runtime, "memGetInfo", lambda: (int(0.05 * total), total)
    )
    trims = async_runtime._allocator_state["trims"]
    assert async_runtime.release_unused_device_memory() is True
    assert async_runtime._allocator_state["trims"] == trims + 1
    # Nothing is left to release, so the next call does not synchronize again.
    assert async_runtime.release_unused_device_memory() is False
    assert async_runtime._allocator_state["trims"] == trims + 1


def test_release_keeps_half_cap_and_ignores_unreleasable_slack(
    async_runtime, monkeypatch
):
    import cupy

    cap = 256 * MIB
    monkeypatch.setattr(async_runtime, "_TRIM_MIN_SLACK_BYTES", cap)
    monkeypatch.setattr(async_runtime, "_TRIM_SLACK_FRACTION", 0.0)
    _empty_mempool(async_runtime)
    cupy.empty(1024 * MIB, dtype=cupy.uint8)
    cupy.cuda.runtime.deviceSynchronize()
    log = async_runtime._allocator_state["trim_log"]
    count = len(log)
    assert async_runtime.release_unused_device_memory() is True
    assert len(log) == count + 1
    _, before, after, seconds = log[-1]
    assert before - after >= 1024 * MIB - cap
    assert seconds >= 0
    reserved = _counter(async_runtime, "ReservedMemCurrent")
    used = _counter(async_runtime, "UsedMemCurrent")
    assert reserved - used >= cap // 2
    # Pretend that the kept slack could not be released: no trim follows
    # until the slack grows by more than the cap again.
    monkeypatch.setitem(async_runtime._allocator_state, "trim_residual", 2 * cap)
    cupy.empty(cap, dtype=cupy.uint8)
    cupy.cuda.runtime.deviceSynchronize()
    assert async_runtime.release_unused_device_memory() is False


def test_simulation_step_releases_memory_at_each_exchange(async_runtime):
    sim = _small_simulation()
    state = async_runtime._allocator_state
    for n_steps in (1, sim.comm.exchange_period):
        expected = _exchange_count(sim, n_steps)
        releases, history = state["releases"], len(state["history"])
        sim.step(n_steps, show_progress=False)
        assert state["releases"] == releases + expected
        assert len(state["history"]) == history + expected


@pytest.mark.parametrize("gc_enabled", [True, False])
def test_graph_capture_pauses_garbage_collection(monkeypatch, gc_enabled):
    import cupy
    import warp as wp

    import fbpic.utils.warp as warp_utils

    runtime = warp_utils.get_runtime()
    runtime.use()
    states = []

    class RecordingCapture(wp.ScopedCapture):
        def __enter__(self):
            states.append(gc.isenabled())
            return super().__enter__()

    monkeypatch.setattr(wp, "ScopedCapture", RecordingCapture)
    values = cupy.zeros(64, dtype=cupy.float64)
    graphs = warp_utils.CudaGraphSections()
    if not gc_enabled:
        gc.disable()
    try:
        with graphs.section("increment"):
            warp_utils.launch_kernel(
                _increment, (values.size,), [warp_utils.as_warp_array(values)], 64
            )
        assert gc.isenabled() is gc_enabled
    finally:
        gc.enable()
    runtime.synchronize()
    assert states == [False]
    assert graphs.capture_count == 1
    assert float(values.sum()) == values.size


def test_pool_mode_keeps_legacy_allocator():
    result = _run_python(
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
        """,
        FBPIC_GPU_ALLOCATOR="pool",
    )
    assert result.returncode == 0, result.stderr


def test_pool_mode_simulation_frees_legacy_pool_at_exchange():
    result = _run_python(
        """
        import cupy
        import fbpic.utils.warp as warp_utils
        from test_gpu_allocator import _small_simulation

        sim = _small_simulation()
        assert warp_utils.allocator_mode() == "pool"
        pool = cupy.get_default_memory_pool()
        calls = []
        release = warp_utils.release_unused_device_memory

        def checked_release():
            assert release() is True
            calls.append(pool.total_bytes() - pool.used_bytes())

        import fbpic.main

        fbpic.main.release_unused_device_memory = checked_release
        sim.step(1, show_progress=False)
        assert calls == [0], calls
        assert warp_utils._allocator_state["releases"] == 1
        assert pool.total_bytes() > 0
        """,
        FBPIC_GPU_ALLOCATOR="pool",
    )
    assert result.returncode == 0, result.stderr
