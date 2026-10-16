"""Check borrowed gradient views and composition on the shared GPU stream."""

import numpy as np
import pytest
import warp as wp

from fbpic.autodiff.arrays import DifferentiableArrays
from fbpic.autodiff.fields import filter_scalar
from fbpic.utils.cuda import cuda_installed

pytestmark = pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable.")


def test_exact_aliases_share_gradients_and_retain_primal_owner():
    import cupy as cp

    from fbpic.utils.warp import get_runtime

    get_runtime().use()
    data = cp.arange(12, dtype=cp.float64).reshape(3, 4)
    registry = DifferentiableArrays()
    first = registry.view(data)
    assert registry.view(data) is first
    assert registry.view(data.view()) is first
    assert first._cupy_owner is data
    first.grad.fill_(3.0)
    get_runtime().synchronize()
    np.testing.assert_array_equal(first.grad.numpy(), 3)
    registry.zero_gradients()
    get_runtime().synchronize()
    np.testing.assert_array_equal(first.grad.numpy(), 0)
    np.testing.assert_array_equal(first.numpy(), np.arange(12).reshape(3, 4))


def test_gradient_policy_is_explicit():
    import cupy as cp

    from fbpic.utils.warp import get_runtime

    get_runtime().use()
    registry = DifferentiableArrays()
    integers = cp.arange(7, dtype=cp.int32)
    with pytest.raises(TypeError, match="float64 or complex128"):
        registry.view(integers)
    assert registry.view(integers, requires_grad=False).grad is None
    with pytest.raises(TypeError, match="float64 or complex128"):
        registry.view(cp.arange(7, dtype=cp.float32))
    values = cp.arange(7, dtype=cp.float64)
    registry.view(values, requires_grad=False)
    with pytest.raises(ValueError, match="different gradient policy"):
        registry.view(values)


def test_nonexact_overlapping_gradient_views_are_rejected():
    import cupy as cp

    from fbpic.utils.warp import get_runtime

    get_runtime().use()
    registry = DifferentiableArrays()
    values = cp.arange(12, dtype=cp.float64)
    registry.view(values)
    with pytest.raises(ValueError, match="nonoverlapping storage"):
        registry.view(values[1:])
    with pytest.raises(ValueError, match="nonoverlapping storage"):
        registry.view(values.reshape(3, 4))
    # Disjoint intervals within one owner remain valid independent inputs.
    registry = DifferentiableArrays()
    first = registry.view(values[:6])
    second = registry.view(values[6:])
    assert first.grad is not second.grad


def test_complex_strided_view_and_two_filters_compose():
    import cupy as cp

    from fbpic.utils.warp import get_runtime

    runtime = get_runtime()
    runtime.use()
    rng = np.random.default_rng(14)
    host = rng.normal(size=(3, 8)) + 1j * rng.normal(size=(3, 8))
    data = cp.asarray(host)[:, ::2]
    registry = DifferentiableArrays()
    inputs = registry.view(data)
    fz = wp.array(np.array([0.2, 0.4, 0.8]), dtype=wp.float64, device="cuda:0")
    fr = wp.array(np.array([0.1, 0.3, 0.6, 0.9]), dtype=wp.float64, device="cuda:0")
    middle = wp.empty((3, 4), dtype=wp.vec2d, device="cuda:0", requires_grad=True)
    outputs = wp.empty_like(middle, requires_grad=True)
    cotangent = wp.array(np.ones((3, 4, 2)), dtype=wp.vec2d, device="cuda:0")
    tape = wp.Tape()
    with tape:
        wp.launch(filter_scalar, dim=(3, 4), inputs=[inputs, 3, 4, fz, fr, middle])
        wp.launch(filter_scalar, dim=(3, 4), inputs=[middle, 3, 4, fz, fr, outputs])
    tape.backward(grads={outputs: cotangent})
    runtime.synchronize()
    factor = np.array([0.2, 0.4, 0.8])[:, None] * np.array([0.1, 0.3, 0.6, 0.9])[None]
    expected = np.stack((factor**2, factor**2), axis=-1)
    np.testing.assert_allclose(inputs.grad.numpy(), expected, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(cp.asnumpy(data), host[:, ::2])
