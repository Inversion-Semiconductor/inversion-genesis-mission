# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Verify field primal parity, complex adjoints, and fixed payload permutations."""

import os
from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp
from scipy.constants import c

from fbpic.autodiff import fields as ad
from fbpic.autodiff.verification import check_vjp
from fbpic.boundaries import cuda_methods as boundaries
from fbpic.boundaries import moving_window
from fbpic.fields import cuda_methods as production
from fbpic.fields.spectral_transform import cuda_methods as transforms
from fbpic.particles.utilities.cuda_sorting import write_sorting_buffer
from fbpic.utils.cuda import cuda_installed

SHAPE = (7, 5)
EB = ("Ep", "Em", "Ez", "Bp", "Bm", "Bz")
CURRENT = ("Jp", "Jm", "Jz")
RHO = ("rho_prev", "rho_next")
CASE_NAMES = (
    "copy_2dc_to_2dr",
    "copy_2dr_to_2dc",
    "copy_2d_to_1d",
    "copy_1d_to_2d",
    "rt_to_pm",
    "pm_to_rt",
    "permute_float64",
    "permute_vec2d",
    "erase_scalar",
    "erase_vector",
    "divide_scalar",
    "divide_vector",
    "filter_scalar",
    "filter_vector",
    "push_rho",
    "correct_standard",
    "correct_comoving",
    "push_standard_true",
    "push_standard_divergence",
    "push_comoving_true",
    "push_comoving_divergence",
    "damp_left",
    "damp_right",
    "shift_positive",
    "shift_positive_four",
    "shift_negative",
    "shift_negative_five",
    "shift_zero",
)


@pytest.fixture(scope="module")
def device():
    wp.init()
    selected = os.environ.get(
        "FBPIC_AUTODIFF_DEVICE", "cuda:0" if cuda_installed else "cpu"
    )
    wp.set_device(selected)
    return selected


@dataclass
class _Case:
    kernel: wp.Kernel
    production_kernel: wp.Kernel | None
    arrays: dict
    arguments: list
    differentiable: tuple
    outputs: tuple
    templates: dict
    dimensions: tuple | int = SHAPE
    inplace: bool = True
    production_arguments: list | None = None


def _complex(rng, shape=SHAPE):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _case(name):
    rng = np.random.default_rng(602)
    if name.startswith("copy_"):
        copies = {
            "copy_2dc_to_2dr": (
                transforms.cuda_copy_2dC_to_2dR.kernel,
                _complex(rng),
                np.zeros((2 * SHAPE[0], SHAPE[1])),
            ),
            "copy_2dr_to_2dc": (
                transforms.cuda_copy_2dR_to_2dC.kernel,
                rng.normal(size=(2 * SHAPE[0], SHAPE[1])),
                np.zeros(SHAPE, dtype=np.complex128),
            ),
            "copy_2d_to_1d": (
                transforms.cuda_copy_2d_to_1d.kernel,
                _complex(rng),
                np.zeros(np.prod(SHAPE), dtype=np.complex128),
            ),
            "copy_1d_to_2d": (
                transforms.cuda_copy_1d_to_2d.kernel,
                _complex(rng, (np.prod(SHAPE),)),
                np.zeros(SHAPE, dtype=np.complex128),
            ),
        }
        original, source, output = copies[name]
        return _Case(
            getattr(ad, name),
            original,
            {"source": source},
            ["source"],
            ("source",),
            ("result",),
            {"result": output},
            inplace=False,
        )
    if name in ("rt_to_pm", "pm_to_rt"):
        original = getattr(transforms, f"cuda_{name}").kernel
        arrays = {"component0": _complex(rng), "component1": _complex(rng)}
        return _Case(
            getattr(ad, name),
            original,
            arrays,
            list(arrays),
            tuple(arrays),
            ("result0", "result1"),
            {"result0": arrays["component0"], "result1": arrays["component1"]},
            inplace=False,
        )
    if name.startswith("permute_"):
        count = 19
        source = (
            _complex(rng, (count,))
            if name.endswith("vec2d")
            else rng.normal(size=count)
        )
        arrays = {
            "source": source,
            "permutation": rng.permutation(count).astype(np.int64),
        }
        return _Case(
            getattr(ad, name),
            write_sorting_buffer.kernel,
            arrays,
            ["source", "permutation"],
            ("source",),
            ("result",),
            {"result": source},
            dimensions=count,
            inplace=False,
            production_arguments=["permutation", "source"],
        )
    if name.startswith("erase_"):
        vector = name.endswith("vector")
        fields = CURRENT if vector else ("field",)
        arrays = {field: _complex(rng) for field in fields}
        return _Case(
            ad.erase_vector if vector else ad.erase_scalar,
            (
                production.cuda_erase_vector.kernel
                if vector
                else production.cuda_erase_scalar.kernel
            ),
            arrays,
            list(fields),
            fields,
            fields,
            arrays.copy(),
        )
    if name in ("divide_scalar", "divide_vector", "filter_scalar", "filter_vector"):
        vector = name.endswith("vector")
        fields = CURRENT if vector else ("field",)
        arrays = {field: _complex(rng) for field in fields}
        if name.startswith("divide"):
            arrays["inverse_volume"] = rng.uniform(0.1, 3, SHAPE[1])
            arguments = list(fields) + ["inverse_volume"]
            kernel = (
                ad.divide_vector_by_volume if vector else ad.divide_scalar_by_volume
            )
            original = (
                production.cuda_divide_vector_by_volume.kernel
                if vector
                else production.cuda_divide_scalar_by_volume.kernel
            )
        else:
            arrays["filter_z"] = rng.uniform(0, 1, SHAPE[0])
            arrays["filter_r"] = rng.uniform(0, 1, SHAPE[1])
            # Include exact zero coefficients and unit coefficients.
            arrays["filter_z"][0], arrays["filter_r"][-1] = 0, 1
            arguments = list(fields) + list(SHAPE) + ["filter_z", "filter_r"]
            kernel = ad.filter_vector if vector else ad.filter_scalar
            original = (
                production.cuda_filter_vector.kernel
                if vector
                else production.cuda_filter_scalar.kernel
            )
        return _Case(
            kernel,
            original,
            arrays,
            arguments,
            fields,
            fields,
            {field: arrays[field] for field in fields},
        )
    if name == "push_rho":
        arrays = {field: _complex(rng) for field in RHO}
        return _Case(
            ad.push_rho,
            production.cuda_push_rho.kernel,
            arrays,
            list(RHO) + list(SHAPE),
            RHO,
            RHO,
            arrays.copy(),
        )
    if name.startswith("correct_"):
        fields = RHO + CURRENT
        arrays = {field: _complex(rng) for field in fields}
        arrays["kz"] = rng.normal(size=SHAPE)
        arrays["kr"] = rng.uniform(0.3, 1.7, SHAPE)
        arrays["kz"][0], arrays["kr"][0, 0] = 0, 0
        k2 = arrays["kz"] ** 2 + arrays["kr"] ** 2
        arrays["inverse_k2"] = np.divide(1, k2, out=np.zeros_like(k2), where=k2 != 0)
        arguments = list(fields) + ["kz", "kr", "inverse_k2"]
        comoving = name.endswith("comoving")
        if comoving:
            for coefficient in ("correction_coef", "t_eb", "t_cc"):
                arrays[coefficient] = _complex(rng)
            arguments += ["correction_coef", "t_eb", "t_cc"]
        arguments += [0.7] + list(SHAPE)
        kernel = (
            ad.correct_currents_curlfree_comoving
            if comoving
            else ad.correct_currents_curlfree_standard
        )
        original = (
            production.cuda_correct_currents_curlfree_comoving.kernel
            if comoving
            else production.cuda_correct_currents_curlfree_standard.kernel
        )
        return _Case(
            kernel,
            original,
            arrays,
            arguments,
            fields,
            CURRENT,
            {field: arrays[field] for field in CURRENT},
        )
    if name.startswith("push_"):
        comoving = name.startswith("push_comoving")
        fields = EB + CURRENT + RHO
        arrays = {field: _complex(rng) for field in fields}
        coefficients = ("rho_prev_coef", "rho_next_coef", "j_coef")
        for coefficient in coefficients:
            arrays[coefficient] = _complex(rng) if comoving else rng.normal(size=SHAPE)
        arrays["cosine"] = rng.uniform(0.1, 1, SHAPE)
        arrays["sine_over_omega"] = rng.normal(size=SHAPE) / c**2
        arrays["kr"] = rng.normal(size=SHAPE)
        arrays["kz"] = rng.normal(size=SHAPE)
        arrays["kr"][0, 0], arrays["kz"][0, 0] = 0, 0
        arguments = list(fields) + list(coefficients) + ["cosine", "sine_over_omega"]
        if comoving:
            for coefficient in ("t_eb", "t_cc", "t_rho"):
                arrays[coefficient] = _complex(rng)
            arguments += ["t_eb", "t_cc", "t_rho"]
        arguments += ["kr", "kz", 0.13]
        if comoving:
            arguments += [0.37]
        arguments += [name.endswith("true")] + list(SHAPE)
        return _Case(
            ad.push_eb_comoving if comoving else ad.push_eb_standard,
            (
                production.cuda_push_eb_comoving.kernel
                if comoving
                else production.cuda_push_eb_standard.kernel
            ),
            arrays,
            arguments,
            fields,
            EB,
            {field: arrays[field] for field in EB},
        )
    if name.startswith("damp_"):
        arrays = {field: _complex(rng) for field in EB}
        arrays["damping"] = np.array([0, 0.2, 1.0])
        left = name.endswith("left")
        return _Case(
            ad.damp_eb_left if left else ad.damp_eb_right,
            (
                boundaries.cuda_damp_EB_left.kernel
                if left
                else boundaries.cuda_damp_EB_right.kernel
            ),
            arrays,
            list(EB) + ["damping", 3],
            EB,
            EB,
            {field: arrays[field] for field in EB},
        )
    if name.startswith("shift_"):
        arrays = {
            "field": _complex(rng),
            "shift_factor": np.exp(1j * rng.normal(size=SHAPE[0])),
        }
        n_move = {
            "shift_positive": 3,
            "shift_positive_four": 4,
            "shift_negative": -2,
            "shift_negative_five": -5,
            "shift_zero": 0,
        }[name]
        original = (
            moving_window.shift_spect_array_gpu.kernel if cuda_installed else None
        )
        return _Case(
            ad.shift_spectral,
            original,
            arrays,
            ["field", "shift_factor", n_move],
            ("field",),
            ("field",),
            {"field": arrays["field"]},
        )
    raise AssertionError(name)


def _device_array(value, device, requires_grad=False):
    value = np.asarray(value)
    if value.dtype == np.complex128:
        data, dtype = np.stack((value.real, value.imag), axis=-1), wp.vec2d
    else:
        data = value
        dtype = wp.int64 if value.dtype == np.int64 else wp.float64
    return wp.array(data, dtype=dtype, device=device, requires_grad=requires_grad)


def _values(case, device, requires_grad=False):
    return {
        name: _device_array(
            value, device, requires_grad and name in case.differentiable
        )
        for name, value in case.arrays.items()
    }


def _arguments(specification, values):
    return [
        values[value] if isinstance(value, str) else value for value in specification
    ]


def _forward(case, values, device):
    outputs = {
        name: wp.zeros_like(
            _device_array(case.templates[name], device), requires_grad=True
        )
        for name in case.outputs
    }
    wp.launch(
        case.kernel,
        dim=case.dimensions,
        inputs=_arguments(case.arguments, values),
        outputs=list(outputs.values()),
        device=device,
    )
    return outputs


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_field_forward_parity(device, case_name):
    case = _case(case_name)
    values = _values(case, device)
    actual = _forward(case, values, device)
    original_values = _values(case, device)
    if case.inplace:
        expected = {name: original_values[name] for name in case.outputs}
        original_outputs = []
    else:
        expected = {
            name: wp.zeros_like(_device_array(case.templates[name], device))
            for name in case.outputs
        }
        original_outputs = list(expected.values())
    if case.production_kernel is None:
        # The CPU reference is available without importing a CUDA-only wrapper.
        reference = case.arrays["field"].copy()
        moving_window.shift_spect_array_cpu(
            reference, case.arrays["shift_factor"], case.arguments[-1]
        )
        expected = {"field": _device_array(reference, device)}
    else:
        wp.launch(
            case.production_kernel,
            dim=case.dimensions,
            inputs=_arguments(
                case.production_arguments or case.arguments, original_values
            ),
            outputs=original_outputs,
            device=device,
        )
    wp.synchronize_device(device)
    for name in case.outputs:
        np.testing.assert_allclose(
            actual[name].numpy(), expected[name].numpy(), rtol=2e-14, atol=2e-14
        )
    for name, value in values.items():
        np.testing.assert_array_equal(
            value.numpy(), _device_array(case.arrays[name], device).numpy()
        )


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_field_vjp(device, case_name):
    case = _case(case_name)
    report = check_vjp(
        lambda values: _forward(case, values, device),
        case.arrays,
        case.differentiable,
        device=device,
        seed=23,
        linear=True,
    )
    assert len(report["directions"]) == 3


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_field_linear_adjoint_identity(device, case_name):
    case = _case(case_name)
    values = _values(case, device, requires_grad=True)
    rng = np.random.default_rng(631)
    with wp.Tape() as tape:
        outputs = _forward(case, values, device)
    wp.synchronize_device(device)
    cotangents = {
        name: rng.normal(size=output.numpy().shape) for name, output in outputs.items()
    }
    tape.backward(
        grads={
            output: wp.array(cotangents[name], dtype=output.dtype, device=device)
            for name, output in outputs.items()
        }
    )
    wp.synchronize_device(device)
    for active_input in case.differentiable:
        gradient = values[active_input].grad.numpy()
        for _ in range(3):
            direction = rng.normal(size=case.arrays[active_input].shape)
            if np.iscomplexobj(case.arrays[active_input]):
                direction = direction + 1j * rng.normal(size=direction.shape)
            tangent_host = {
                name: (
                    direction
                    if name == active_input
                    else (
                        np.zeros_like(value) if name in case.differentiable else value
                    )
                )
                for name, value in case.arrays.items()
            }
            tangent_values = {
                name: _device_array(value, device)
                for name, value in tangent_host.items()
            }
            action = _forward(case, tangent_values, device)
            wp.synchronize_device(device)
            left = sum(
                np.sum(action[name].numpy() * cotangents[name]) for name in outputs
            )
            packed_direction = _device_array(direction, device).numpy()
            right = np.sum(packed_direction * gradient)
            normalization = max(
                sum(
                    np.linalg.norm(action[name].numpy())
                    * np.linalg.norm(cotangents[name])
                    for name in outputs
                ),
                np.linalg.norm(packed_direction) * np.linalg.norm(gradient),
                np.finfo(np.float64).tiny,
            )
            assert abs(left - right) / normalization <= 1e-10, (
                case_name,
                active_input,
                left,
                right,
            )
