"""Verify particle primal parity and scaled FP64 reverse derivatives."""

import os

import numpy as np
import pytest
import warp as wp
from scipy.constants import e, m_e, m_p

from fbpic.autodiff.particles import (
    get_gather_kernel,
    push_ion_momenta,
    push_momenta,
    push_positions,
)
from fbpic.autodiff.verification import check_vjp
from fbpic.utils.cuda import cuda_installed

MOMENTA = ("ux", "uy", "uz", "inv_gamma")
ELECTRIC = ("Ex", "Ey", "Ez")
MAGNETIC = ("Bx", "By", "Bz")
POSITIONS = ("x", "y", "z")
DT = 1e-15


@pytest.fixture(scope="module")
def device():
    wp.init()
    selected = os.environ.get(
        "FBPIC_AUTODIFF_DEVICE", "cuda:0" if cuda_installed else "cpu"
    )
    wp.set_device(selected)
    return selected


def _momenta_fixture(ions=False):
    rng = np.random.default_rng(91)
    momentum = rng.normal(scale=0.7, size=(3, 7))
    arrays = {name: value.copy() for name, value in zip(MOMENTA[:3], momentum)}
    arrays["inv_gamma"] = 1 / np.sqrt(1 + np.sum(momentum**2, axis=0))
    electric_scale, magnetic_scale = (1e11, 300.0) if ions else (1e9, 3.0)
    for name in ELECTRIC:
        arrays[name] = rng.normal(size=7) * electric_scale
    for name in MAGNETIC:
        arrays[name] = rng.normal(size=7) * magnetic_scale
    if ions:
        arrays["level"] = np.array([0, 1, 2, 3, 0, 1, 7], dtype=np.uint64)
    return arrays


def _position_fixture():
    arrays = _momenta_fixture()
    rng = np.random.default_rng(102)
    for name in POSITIONS:
        arrays[name] = rng.normal(size=7) * 1e-6
    return arrays


def _device_arrays(arrays, device, requires_grad=False):
    result = {}
    for name, value in arrays.items():
        dtype = wp.uint64 if value.dtype == np.uint64 else wp.float64
        if np.iscomplexobj(value):
            value = np.stack((value.real, value.imag), axis=-1)
            dtype = wp.vec2d
        result[name] = wp.array(
            value,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad and dtype != wp.uint64,
        )
    return result


def _outputs(names, count, device):
    return {
        name: wp.zeros(count, dtype=wp.float64, device=device, requires_grad=True)
        for name in names
    }


def _launch_momenta(values, device, ions=False):
    result = _outputs(MOMENTA, values["ux"].shape[0], device)
    arguments = [values[name] for name in MOMENTA + ELECTRIC + MAGNETIC]
    if ions:
        arguments += [4 * m_p, DT, values["level"]]
        kernel = push_ion_momenta
    else:
        arguments += [-e, m_e, DT]
        kernel = push_momenta
    wp.launch(
        kernel,
        dim=values["ux"].shape[0],
        inputs=arguments,
        outputs=list(result.values()),
        device=device,
    )
    return result


def _launch_positions(values, device):
    result = _outputs(POSITIONS, values["x"].shape[0], device)
    arguments = [values[name] for name in POSITIONS + MOMENTA]
    arguments += [DT, 1.0, -0.7, 0.2]
    wp.launch(
        push_positions,
        dim=values["x"].shape[0],
        inputs=arguments,
        outputs=list(result.values()),
        device=device,
    )
    return result


@pytest.mark.parametrize("ions", [False, True])
def test_momenta_forward_parity(device, ions):
    from fbpic.particles.push.cuda_methods import push_p_gpu, push_p_ioniz_gpu

    host = _momenta_fixture(ions)
    values = _device_arrays(host, device)
    result = _launch_momenta(values, device, ions)
    production = _device_arrays(host, device)
    count = host["ux"].size
    arguments = [production[name] for name in MOMENTA + ELECTRIC + MAGNETIC]
    if ions:
        arguments += [4 * m_p, count, DT, production["level"]]
        kernel = push_p_ioniz_gpu.kernel
    else:
        arguments += [-e, m_e, count, DT]
        kernel = push_p_gpu.kernel
    wp.launch(kernel, dim=count, inputs=arguments, device=device)
    wp.synchronize_device(device)
    for name in MOMENTA:
        np.testing.assert_allclose(
            result[name].numpy(), production[name].numpy(), rtol=2e-14, atol=2e-15
        )
    for name, value in values.items():
        np.testing.assert_array_equal(value.numpy(), host[name])


def test_positions_forward_parity(device):
    from fbpic.particles.push.cuda_methods import push_x_gpu

    host = _position_fixture()
    values = _device_arrays(host, device)
    result = _launch_positions(values, device)
    production = _device_arrays(host, device)
    wp.launch(
        push_x_gpu.kernel,
        dim=7,
        inputs=[production[name] for name in POSITIONS + MOMENTA]
        + [DT, 1.0, -0.7, 0.2],
        device=device,
    )
    wp.synchronize_device(device)
    for name in POSITIONS:
        np.testing.assert_allclose(
            result[name].numpy(), production[name].numpy(), rtol=2e-14, atol=1e-20
        )
        np.testing.assert_array_equal(values[name].numpy(), host[name])


@pytest.mark.parametrize("ions", [False, True])
@pytest.mark.parametrize(
    "group", [MOMENTA, ELECTRIC, MAGNETIC], ids=["state", "electric", "magnetic"]
)
def test_momenta_vjp(device, ions, group):
    arrays = _momenta_fixture(ions)
    scales = {name: 1.0 for name in MOMENTA}
    scales.update({name: 1e11 if ions else 1e9 for name in ELECTRIC})
    scales.update({name: 300.0 if ions else 3.0 for name in MAGNETIC})
    report = check_vjp(
        lambda values: _launch_momenta(values, device, ions),
        arrays,
        group,
        device=device,
        scales=scales,
        seed=13,
    )
    assert len(report["directions"]) == 3


@pytest.mark.parametrize("positions_only", [False, True])
def test_positions_vjp(device, positions_only):
    names = POSITIONS if positions_only else POSITIONS + MOMENTA
    scales = {name: 1e-6 for name in POSITIONS}
    scales.update({name: 1.0 for name in MOMENTA})
    check_vjp(
        lambda values: _launch_positions(values, device),
        _position_fixture(),
        names,
        device=device,
        scales=scales,
        seed=14,
        linear=positions_only,
    )


def test_neutral_ion_gradients(device):
    host = _momenta_fixture(ions=True)
    values = _device_arrays(host, device, requires_grad=True)
    with wp.Tape() as tape:
        result = _launch_momenta(values, device, ions=True)
    tape.backward(
        grads={
            output: wp.ones(7, dtype=wp.float64, device=device)
            for output in result.values()
        }
    )
    wp.synchronize_device(device)
    neutral = host["level"] == 0
    for name in MOMENTA:
        np.testing.assert_array_equal(
            result[name].numpy()[neutral], host[name][neutral]
        )
        np.testing.assert_array_equal(values[name].grad.numpy()[neutral], 1)
    for name in ELECTRIC + MAGNETIC:
        np.testing.assert_array_equal(values[name].grad.numpy()[neutral], 0)


def _gather_fixture():
    rng = np.random.default_rng(211)
    dz, dr = 2e-6, 3e-6
    radius = dr * np.array([1.17, 2.28, 3.36, 4.41, 2.64])
    theta = np.array([0.22, 1.04, -1.73, 2.45, -0.53])
    arrays = {
        "x": radius * np.cos(theta),
        "y": radius * np.sin(theta),
        "z": -5e-6 + dz * np.array([1.17, 2.28, 3.36, 4.41, 5.22]),
    }
    for mode in range(5):
        for field in ("Er", "Et", "Ez", "Br", "Bt", "Bz"):
            scale = 1e9 if field.startswith("E") else 3.0
            arrays[f"{field}_{mode}"] = scale * (
                rng.normal(size=(8, 6)) + 1j * rng.normal(size=(8, 6))
            )
    return arrays


def _launch_gather(values, device, kernel=None):
    count = values["x"].shape[0]
    result = _outputs(ELECTRIC + MAGNETIC, count, device)
    grids = [
        values[f"{field}_{mode}"]
        for mode in range(5)
        for field in ("Er", "Et", "Ez", "Br", "Bt", "Bz")
    ]
    wp.launch(
        get_gather_kernel() if kernel is None else kernel,
        dim=count,
        inputs=[values[name] for name in POSITIONS]
        + [18e-6, 1 / 2e-6, -5e-6, 8, 1 / 3e-6, 0.0, 6]
        + grids,
        outputs=list(result.values()),
        device=device,
    )
    return result


def test_gather_forward_parity_and_module_options(device):
    from fbpic.particles.gathering.cuda_methods_fused import (
        gather_field_gpu_linear_five_modes,
    )

    host = _gather_fixture()
    values = _device_arrays(host, device, requires_grad=True)
    original = gather_field_gpu_linear_five_modes.kernel
    result = _launch_gather(values, device)
    reference = _launch_gather(values, device, original)
    wp.synchronize_device(device)
    for name in ELECTRIC + MAGNETIC:
        np.testing.assert_array_equal(result[name].numpy(), reference[name].numpy())
    assert not original.module.options["enable_backward"]
    assert get_gather_kernel().module.options["enable_backward"]
    assert get_gather_kernel() is get_gather_kernel()


@pytest.mark.parametrize("group", ["positions", "electric", "magnetic"])
def test_gather_vjp(device, group):
    arrays = _gather_fixture()
    if group == "positions":
        names = POSITIONS
    else:
        names = tuple(
            name
            for name in arrays
            if name.startswith("E" if group == "electric" else "B")
        )
    scales = {name: 3e-6 for name in POSITIONS}
    scales.update(
        {
            name: 1e9 if name.startswith("E") else 3.0
            for name in arrays
            if name not in POSITIONS
        }
    )
    check_vjp(
        lambda values: _launch_gather(values, device),
        arrays,
        names,
        device=device,
        scales=scales,
        seed=18,
        linear=group != "positions",
    )


def test_momentum_position_composition_vjp(device):
    def forward(values):
        momenta = _launch_momenta(values, device)
        return _launch_positions({**values, **momenta}, device)

    scales = {name: 1e-6 for name in POSITIONS}
    scales.update({name: 1.0 for name in MOMENTA})
    check_vjp(
        forward,
        _position_fixture(),
        POSITIONS + MOMENTA,
        device=device,
        scales=scales,
        seed=21,
    )
