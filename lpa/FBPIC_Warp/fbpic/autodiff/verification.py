# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Check FP64 operator VJPs with scaled directional finite differences.

Host transfers and synchronization are confined to this verification utility.
"""

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

_DTYPES = {
    np.dtype(np.float64): wp.float64,
    np.dtype(np.complex128): wp.vec2d,
    np.dtype(np.int32): wp.int32,
    np.dtype(np.int64): wp.int64,
    np.dtype(np.uint32): wp.uint32,
    np.dtype(np.uint64): wp.uint64,
}


def _array(value, device, requires_grad=False):
    value = np.asarray(value)
    data = value
    if np.iscomplexobj(value):
        data = np.stack((value.real, value.imag), axis=-1)
    return wp.array(
        data, dtype=_DTYPES[value.dtype], device=device, requires_grad=requires_grad
    )


def _real_components(value):
    if np.iscomplexobj(value):
        return np.stack((value.real, value.imag), axis=-1)
    return value


def check_vjp(
    forward: Callable[[Mapping[str, wp.array]], Mapping[str, wp.array]],
    arrays: Mapping[str, np.ndarray],
    differentiable_names: Sequence[str],
    *,
    device: str = "cuda:0",
    scales: Mapping[str, float] | None = None,
    seed: int = 0,
    linear: bool = False,
    epsilon_sizes: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compare reverse derivatives with three independent input directions.

    If FBPIC_AUTODIFF_REPORT_DIR is set, write per-case JSON verification records.

    Args:
        forward: Launch an operator with persistent input views and return named
            output arrays with gradients enabled. Inputs must remain unchanged.
        arrays: Host primal fixtures, including fixed integer and coefficient data.
        differentiable_names: Continuous array inputs to perturb and differentiate.
        device: Device for operator execution. Only the selected device is used.
        scales: Optional characteristic input scales in each input's units.
        seed: Deterministic perturbation and output-cotangent seed.
        linear: Require 1e-10 instead of 1e-6 normalized directional agreement.
        epsilon_sizes: Dimensionless central-difference sizes. Linear checks
            default to 1 through 1e-5, avoiding small-step cancellation.

    Returns:
        Per-direction errors and the finite-difference sizes that passed.

    Raises:
        AssertionError: Gradients, primal preservation, or directional checks fail.
        TypeError: A differentiated input lacks FP64 continuous storage.
    """
    if epsilon_sizes is None:
        epsilon_sizes = (
            (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
            if linear
            else (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
        )
    host = {name: np.array(value, copy=True) for name, value in arrays.items()}
    names = tuple(differentiable_names)
    for name in names:
        if host[name].dtype not in (np.float64, np.complex128):
            raise TypeError(f"A continuous FP64 input is required: {name}")
    values = {
        name: _array(value, device, name in names) for name, value in host.items()
    }
    rng = np.random.default_rng(seed)
    tape = wp.Tape()
    with tape:
        outputs = dict(forward(values))
    wp.synchronize_device(device)
    for name, value in values.items():
        np.testing.assert_array_equal(value.numpy(), _real_components(host[name]))
    cotangents = {}
    for name, output in outputs.items():
        data = output.numpy()
        scale = float(np.max(np.abs(data), initial=0)) or 1.0
        cotangents[name] = rng.normal(size=data.shape) / (scale * max(data.size, 1))
    output_grads = {
        output: wp.array(cotangents[name], dtype=output.dtype, device=device)
        for name, output in outputs.items()
    }
    tape.backward(grads=output_grads)
    wp.synchronize_device(device)
    gradients = {name: values[name].grad.numpy() for name in names}
    for gradient in gradients.values():
        assert np.isfinite(
            gradient
        ).all(), "The reverse pass returned nonfinite gradients."

    def evaluate(perturbed):
        inputs = {
            name: _array(value, device, name in names)
            for name, value in perturbed.items()
        }
        result = forward(inputs)
        wp.synchronize_device(device)
        return float(
            sum(np.sum(result[name].numpy() * cotangents[name]) for name in outputs)
        )

    checks = []
    tolerance = 1e-10 if linear else 1e-6
    for direction_index in range(3):
        directions = {}
        for name in names:
            shape = host[name].shape
            direction = rng.normal(size=shape)
            if np.iscomplexobj(host[name]):
                direction = direction + 1j * rng.normal(size=shape)
            scale = (scales or {}).get(
                name, float(np.max(np.abs(host[name]), initial=0)) or 1.0
            )
            directions[name] = direction * scale / max(np.linalg.norm(direction), 1)
        contributions = [
            float(np.sum(gradients[name] * _real_components(directions[name])))
            for name in names
        ]
        analytical = sum(contributions)
        errors = []
        for epsilon in epsilon_sizes:
            plus = {
                name: value + epsilon * directions[name] if name in names else value
                for name, value in host.items()
            }
            minus = {
                name: value - epsilon * directions[name] if name in names else value
                for name, value in host.items()
            }
            numerical = (evaluate(plus) - evaluate(minus)) / (2 * epsilon)
            normalization = max(
                abs(analytical), abs(numerical), sum(map(abs, contributions)), 1e-12
            )
            errors.append(
                {
                    "epsilon": epsilon,
                    "finite_difference": numerical,
                    "absolute_error": abs(analytical - numerical),
                    "normalization": normalization,
                    "normalized_error": abs(analytical - numerical) / normalization,
                }
            )
        stable_interval = any(
            first["normalized_error"] <= tolerance
            and second["normalized_error"] <= tolerance
            for first, second in pairwise(errors)
        )
        assert stable_interval, (
            f"Direction {direction_index} failed VJP agreement: "
            f"analytical={analytical}, errors={errors}"
        )
        checks.append({"analytical": analytical, "samples": errors})
    # Repeated reverse calls must start from zero gradients and reproduce the VJP.
    tape.zero()
    tape.backward(grads=output_grads)
    wp.synchronize_device(device)
    for name, expected in gradients.items():
        np.testing.assert_allclose(
            values[name].grad.numpy(), expected, rtol=1e-12, atol=0
        )
    report = {
        "device": device,
        "seed": seed,
        "differentiable_inputs": names,
        "tolerance": tolerance,
        "directions": checks,
        "normalization": "max(abs(reverse), abs(central), sum(abs(input-block contractions)), 1e-12)",
    }
    output_directory = os.environ.get("FBPIC_AUTODIFF_REPORT_DIR")
    if output_directory:
        report["primal_inputs"] = {
            name: {
                "shape": value.shape,
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(value).tobytes()
                ).hexdigest(),
            }
            for name, value in host.items()
        }
        report["operators"] = [
            (
                {
                    "kernel": launch[0].key,
                    "scalars": [
                        value.item() if isinstance(value, np.generic) else value
                        for value in launch[3]
                        if isinstance(value, (int, float, bool, np.generic))
                    ],
                }
                if not callable(launch)
                else {"custom_backward": launch.__qualname__}
            )
            for launch in tape.launches
        ]
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(report, indent=2, allow_nan=False)
        name = hashlib.sha256(encoded.encode()).hexdigest()[:20]
        (directory / f"vjp_{name}.json").write_text(encoded + "\n")
    return report
