# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""Check the CPU and GPU transforms from the original manual comparison."""

import numpy as np
import pytest

from fbpic.fields.spectral_transform import SpectralTransformer
from fbpic.utils.cuda import cuda_installed


pytestmark = pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")


def _assert_transform_equal(actual, expected):
    # This new tolerance checks double-precision CPU/GPU roundoff agreement.
    # Scale each output separately because spectral and spatial units differ.
    tolerance = max(1.0e-12 * np.abs(expected).max(), np.finfo(np.float64).tiny)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("Nz, Nr", [(32, 17), (33, 20)])
@pytest.mark.parametrize("m", [0, 1, 2])
def test_cuda_transform(Nz, Nr, m):
    """Compare both transform directions for scalar and transverse vector fields."""
    import cupy
    from fbpic.utils.warp import get_runtime

    runtime = get_runtime()
    runtime.use()
    initial_launch_count = runtime.launch_count
    rmax = 50.0e-6
    rng = np.random.default_rng(0)
    interp_field_r = rng.random((Nz, Nr)) + 1.0j * rng.random((Nz, Nr))
    interp_field_t = rng.random((Nz, Nr)) + 1.0j * rng.random((Nz, Nr))
    d_interp_field_r = cupy.asarray(interp_field_r)
    d_interp_field_t = cupy.asarray(interp_field_t)

    spect_field_p = np.empty_like(interp_field_r)
    spect_field_m = np.empty_like(interp_field_t)
    back_field_r = np.empty_like(interp_field_r)
    back_field_t = np.empty_like(interp_field_t)
    d_spect_field_p = cupy.empty_like(d_interp_field_r)
    d_spect_field_m = cupy.empty_like(d_interp_field_t)
    d_back_field_r = cupy.empty_like(d_interp_field_r)
    d_back_field_t = cupy.empty_like(d_interp_field_t)

    trans_cpu = SpectralTransformer(Nz, Nr, m, rmax)
    trans_gpu = SpectralTransformer(Nz, Nr, m, rmax, use_cuda=True)
    assert trans_gpu.use_cuda

    trans_cpu.interp2spect_scal(interp_field_r, spect_field_p)
    trans_cpu.spect2interp_scal(spect_field_p, back_field_r)
    trans_gpu.interp2spect_scal(d_interp_field_r, d_spect_field_p)
    trans_gpu.spect2interp_scal(d_spect_field_p, d_back_field_r)
    _assert_transform_equal(d_spect_field_p.get(), spect_field_p)
    _assert_transform_equal(d_back_field_r.get(), back_field_r)

    trans_cpu.interp2spect_vect(
        interp_field_r, interp_field_t, spect_field_p, spect_field_m
    )
    trans_cpu.spect2interp_vect(
        spect_field_p, spect_field_m, back_field_r, back_field_t
    )
    trans_gpu.interp2spect_vect(
        d_interp_field_r, d_interp_field_t, d_spect_field_p, d_spect_field_m
    )
    trans_gpu.spect2interp_vect(
        d_spect_field_p, d_spect_field_m, d_back_field_r, d_back_field_t
    )
    _assert_transform_equal(d_spect_field_p.get(), spect_field_p)
    _assert_transform_equal(d_spect_field_m.get(), spect_field_m)
    _assert_transform_equal(d_back_field_r.get(), back_field_r)
    _assert_transform_equal(d_back_field_t.get(), back_field_t)
    assert runtime.launch_count > initial_launch_count


@pytest.mark.parametrize(
    "Nz, Nr, sliced",
    [(1, 1, False), (33, 17, False), (1821, 301, False), (33, 17, True)],
)
def test_cuda_copy_1d_2d_bitwise(Nz, Nr, sliced):
    """Check that the z-r transposing copies are exact and invert each other."""
    import cupy
    from fbpic.fields.spectral_transform.cuda_methods import (
        cuda_copy_1d_2d_launch,
        cuda_copy_1d_to_2d,
        cuda_copy_2d_to_1d,
    )
    from fbpic.utils.warp import get_runtime

    get_runtime().use()
    rng = np.random.default_rng(1)
    shape = (2 * Nz + 3, 3 * Nr + 2) if sliced else (Nz, Nr)
    parent = cupy.asarray(rng.random(shape) + 1.0j * rng.random(shape))
    # A sliced view has row and column strides that differ from (Nr*16, 16).
    array_2d = parent[1 : 2 * Nz + 1 : 2, 2 : 3 * Nr + 2 : 3] if sliced else parent
    assert array_2d.shape == (Nz, Nr)
    assert array_2d.flags.c_contiguous != sliced
    launch = cuda_copy_1d_2d_launch(Nz, Nr)

    # Fill the outputs with NaN so that a skipped element cannot pass.
    array_1d = cupy.full(Nz * Nr, np.nan, dtype=np.complex128)
    cuda_copy_2d_to_1d[launch](array_2d, array_1d)
    assert cupy.array_equal(array_1d, cupy.ascontiguousarray(array_2d.T).ravel())

    back_parent = cupy.full_like(parent, np.nan)
    back_2d = (
        back_parent[1 : 2 * Nz + 1 : 2, 2 : 3 * Nr + 2 : 3] if sliced else back_parent
    )
    cuda_copy_1d_to_2d[launch](array_1d, back_2d)
    assert cupy.array_equal(back_2d, array_2d)
    if sliced:
        # The inverse copy must not write outside the strided view.
        untouched = cupy.ones(shape, dtype=bool)
        untouched[1 : 2 * Nz + 1 : 2, 2 : 3 * Nr + 2 : 3] = False
        assert bool(cupy.isnan(back_parent[untouched]).all())
