# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Verify the opt-in GPU Hankel precision and its stream contract."""

import numpy as np
import pytest

from fbpic.fields import Fields
from fbpic.fields.spectral_transform import SpectralTransformer
from fbpic.fields.spectral_transform.hankel import _CUBLAS_PEDANTIC_MATH, DHT
from fbpic.main import Simulation
from fbpic.utils.cuda import cuda_installed


@pytest.mark.parametrize("precision", ["float16", "single", None])
@pytest.mark.parametrize("constructor", ["simulation", "fields", "transformer", "dht"])
def test_invalid_hankel_precision(precision, constructor):
    factories = {
        "simulation": lambda: Simulation(
            32, 20e-6, 8, 10e-6, 1, 1e-16, hankel_precision=precision
        ),
        "fields": lambda: Fields(
            32, 20e-6, 8, 10e-6, 1, 1e-16, hankel_precision=precision
        ),
        "transformer": lambda: SpectralTransformer(
            32, 8, 0, 10e-6, hankel_precision=precision
        ),
        "dht": lambda: DHT(0, 0, 8, 32, 10e-6, hankel_precision=precision),
    }
    with pytest.raises(ValueError, match="hankel_precision"):
        factories[constructor]()


@pytest.mark.parametrize("constructor", ["simulation", "fields", "transformer", "dht"])
def test_float32_requires_gpu(constructor):
    factories = {
        "simulation": lambda: Simulation(
            32, 20e-6, 8, 10e-6, 1, 1e-16, hankel_precision="float32"
        ),
        "fields": lambda: Fields(
            32, 20e-6, 8, 10e-6, 1, 1e-16, hankel_precision="float32"
        ),
        "transformer": lambda: SpectralTransformer(
            32, 8, 0, 10e-6, hankel_precision="float32"
        ),
        "dht": lambda: DHT(0, 0, 8, 32, 10e-6, hankel_precision="float32"),
    }
    with pytest.raises(ValueError, match="(?i)(GPU|CUDA)"):
        factories[constructor]()


def _assert_fp32_close(actual, expected):
    # float32 GEMM rounding grows with the Nr-term dot products (eps32 = 6e-8).
    # 3e-6 of the field scale bounds it with margin for Nr <= 300.
    scale = max(float(np.abs(expected).max()), np.finfo(np.float64).tiny)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-6 * scale)


@pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")
@pytest.mark.parametrize("nz,nr", [(32, 17), (33, 20)])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_float32_scalar_vector_transforms(nz, nr, mode):
    import cupy as cp

    from fbpic.utils.warp import get_runtime

    get_runtime().use()
    rng = np.random.default_rng(19)
    fields = [
        rng.standard_normal((nz, nr)) + 1j * rng.standard_normal((nz, nr))
        for _ in range(2)
    ]
    cpu = SpectralTransformer(nz, nr, mode, 50e-6)
    gpu = SpectralTransformer(
        nz, nr, mode, 50e-6, use_cuda=True, hankel_precision="float32"
    )
    assert gpu.spect_buffer_r.dtype == np.complex128
    for dht in (gpu.dht0, gpu.dhtp, gpu.dhtm):
        assert dht.M.dtype == dht.invM.dtype == np.float64
        assert dht.d_in.dtype == dht.d_out.dtype == np.float32
        assert dht.d_M.dtype == dht.d_invM.dtype == np.float32
    expected = [np.empty_like(fields[0]) for _ in range(2)]
    actual = [cp.empty(fields[0].shape, dtype=cp.complex128) for _ in range(2)]
    inputs = [cp.asarray(field) for field in fields]
    cpu.interp2spect_scal(fields[0], expected[0])
    gpu.interp2spect_scal(inputs[0], actual[0])
    _assert_fp32_close(actual[0].get(), expected[0])
    cpu.spect2interp_scal(expected[0], expected[1])
    gpu.spect2interp_scal(actual[0], actual[1])
    _assert_fp32_close(actual[1].get(), expected[1])
    cpu.interp2spect_vect(*fields, *expected)
    gpu.interp2spect_vect(*inputs, *actual)
    for result, reference in zip(actual, expected):
        _assert_fp32_close(result.get(), reference)
    cpu.spect2interp_vect(*expected, *fields)
    gpu.spect2interp_vect(*actual, *inputs)
    for result, reference in zip(inputs, fields):
        _assert_fp32_close(result.get(), reference)


@pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")
@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize("nr,mode,order", [(17, 0, 0), (300, 4, 3)])
def test_float32_dht_matches_sgemm_in_place_on_foreign_stream(
    direction, nr, mode, order
):
    import cupy as cp
    from cupy.cuda import cublas

    from fbpic.utils.warp import get_runtime

    runtime = get_runtime()
    runtime.use()
    nz = 9
    dht = DHT(order, mode, nr, nz, 200e-6, use_cuda=True, hankel_precision="float32")
    assert cublas.getMathMode(dht.blas) == _CUBLAS_PEDANTIC_MATH
    rng = np.random.default_rng(23)
    source = cp.asarray(rng.normal(size=(nz, nr)) + 1j * rng.normal(size=(nz, nr)))
    packed = cp.empty((2 * nz, nr), dtype=cp.float32)
    cp.copyto(packed[:nz], source.real)
    cp.copyto(packed[nz:], source.imag)
    transformed = cp.empty_like(packed)
    matrix = dht.d_M if direction == "forward" else dht.d_invM
    alpha, beta = np.array(1, dtype=np.float32), np.array(0, dtype=np.float32)
    reference_handle = cublas.create()
    try:
        cublas.setStream(reference_handle, runtime.cupy_stream.ptr)
        cublas.setMathMode(reference_handle, _CUBLAS_PEDANTIC_MATH)
        cublas.sgemm(
            reference_handle,
            0,
            0,
            nr,
            2 * nz,
            nr,
            alpha.ctypes.data,
            matrix.data.ptr,
            nr,
            packed.data.ptr,
            nr,
            beta.ctypes.data,
            transformed.data.ptr,
            nr,
        )
        reference = cp.empty_like(source)
        cp.copyto(reference.real, transformed[:nz])
        cp.copyto(reference.imag, transformed[nz:])
        runtime.synchronize()
        cp.cuda.Stream(non_blocking=True).use()
        operation = dht.transform if direction == "forward" else dht.inverse_transform
        operation(source, source)
        assert cp.cuda.get_current_stream().ptr == runtime.cupy_stream.ptr
        runtime.synchronize()
        _assert_fp32_close(source.get(), reference.get())
    finally:
        runtime.synchronize()
        cublas.destroy(reference_handle)


@pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")
def test_float32_eager_and_graph_simulations_agree():
    from scipy.constants import c

    from fbpic.utils.random_seed import set_random_seed

    states = []
    for graphs in (False, True):
        set_random_seed(7)
        sim = Simulation(
            64,
            20e-6,
            17,
            30e-6,
            3,
            5e-16,
            n_e=1e20,
            p_nz=1,
            p_nr=1,
            p_nt=12,
            initialize_ions=True,
            use_cuda=True,
            n_order=16,
            n_guard=16,
            use_cuda_graphs=graphs,
            hankel_precision="float32",
            verbose_level=0,
        )
        z = sim.fld.interp[0].z[:, None]
        sim.fld.interp[0].Ez[:] = 1e3 * np.cos(2 * np.pi * z / 20e-6)
        sim.step(12, show_progress=False)
        states.append(
            [
                getattr(grid, component).copy()
                for grid in sim.fld.interp
                for component in ("Er", "Et", "Ez", "Br", "Bt", "Bz")
            ]
        )
        if graphs:
            assert sim._cuda_graphs.capture_count > 0
            assert sim._cuda_graphs.replay_count > sim._cuda_graphs.capture_count
    # Compare electromagnetic energy scales, including components near zero.
    difference_squared = reference_squared = 0.0
    for index, (eager, graph) in enumerate(zip(*states)):
        weight = 1.0 if index % 6 < 3 else c**2
        difference_squared += weight * np.sum(np.abs(graph - eager) ** 2)
        reference_squared += weight * np.sum(np.abs(eager) ** 2)
    # Eager and graph runs execute the same kernels; only the order of atomic
    # additions differs, so the energy-weighted difference stays near fp64 noise.
    assert np.sqrt(difference_squared / reference_squared) < 1e-10


@pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_cublas_handles_use_the_shared_stream(precision):
    from cupy.cuda import cublas

    from fbpic.utils.warp import get_runtime

    runtime = get_runtime()
    runtime.use()
    dht = DHT(0, 0, 17, 9, 200e-6, use_cuda=True, hankel_precision=precision)
    # Copy kernels run on the shared stream, so the GEMM must be ordered on it.
    assert cublas.getStream(dht.blas) == runtime.cupy_stream.ptr
