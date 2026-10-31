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


def _host_output(value):
    """Host real-pair copy of a Warp, CuPy, or NumPy output array."""
    if isinstance(value, wp.array):
        value = value.numpy()
    elif not isinstance(value, np.ndarray):
        import cupy

        value = cupy.asnumpy(value)
    return np.asarray(_real_components(np.asarray(value)), dtype=np.float64)


def check_jvp(
    forward: Callable[[Mapping[str, wp.array]], Mapping[str, Any]],
    tangent: Callable[
        [Mapping[str, np.ndarray], Mapping[str, np.ndarray]], Mapping[str, Any]
    ],
    arrays: Mapping[str, np.ndarray],
    differentiable_names: Sequence[str],
    *,
    device: str = "cuda:0",
    scales: Mapping[str, float] | None = None,
    seed: int = 0,
    linear: bool = False,
    epsilon_sizes: Sequence[float] | None = None,
    direction_masks: Mapping[str, np.ndarray] | None = None,
    directions_count: int = 3,
) -> dict[str, Any]:
    """Compare forward derivatives with central differences in vector form.

    Inversion plan Section 6.1 (L1). For each of `directions_count` random
    input directions d, the error at one epsilon is the largest over the
    outputs o of

        |J d - FD(d)|_inf / max(|J d|_inf, |FD(d)|_inf, |sum_b |J_b d_b||_inf, 1e-12),

    where b runs over the input blocks (one per differentiable name), all
    norms are taken over the entries of output o, and complex outputs count
    as real pairs. A direction passes when two adjacent epsilon sizes both
    meet the tolerance (the pair rule of `check_vjp`).

    Args:
        forward: Launch the primal operator on Warp views of the inputs (as in
            `check_vjp`) and return named outputs (Warp, CuPy, or NumPy).
        tangent: tangent(host_arrays, directions) -> named output tangents.
            `directions` holds one host array per differentiable name, with
            the input's dtype and shape; it must not be modified.
        arrays: Host primal fixtures, including fixed integer and coefficient data.
        differentiable_names: Continuous array inputs to perturb (the blocks).
        device: Device for the forward operator.
        scales: Characteristic input scales; default max |input|.
        seed: Seed of the random directions.
        linear: Require 1e-10 instead of 1e-6.
        epsilon_sizes: Dimensionless central-difference sizes; the defaults
            match `check_vjp`.
        direction_masks: Optional boolean masks per name, broadcastable to
            the input shape. Entries where the mask is False get no
            perturbation (plan Section 6.3, rule 6).
        directions_count: Number of random directions.

    Returns:
        Per-direction samples with the error of each epsilon.

    Raises:
        AssertionError: A direction fails the pair rule, or a tangent is not finite.
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
    rng = np.random.default_rng(seed)
    tolerance = 1e-10 if linear else 1e-6

    def evaluate(perturbed):
        inputs = {name: _array(value, device) for name, value in perturbed.items()}
        result = forward(inputs)
        wp.synchronize_device(device)
        return {name: _host_output(value) for name, value in result.items()}

    def tangents(directions):
        result = tangent(host, directions)
        return {name: _host_output(value) for name, value in result.items()}

    checks = []
    for direction_index in range(directions_count):
        directions = {}
        for name in names:
            shape = host[name].shape
            direction = rng.normal(size=shape)
            if np.iscomplexobj(host[name]):
                direction = direction + 1j * rng.normal(size=shape)
            if direction_masks is not None and name in direction_masks:
                direction = direction * np.broadcast_to(direction_masks[name], shape)
            scale = (scales or {}).get(
                name, float(np.max(np.abs(host[name]), initial=0)) or 1.0
            )
            norm = np.linalg.norm(direction)
            directions[name] = direction * (scale / norm if norm > 0 else 0.0)
        full = tangents(directions)
        for name, value in full.items():
            assert np.isfinite(value).all(), f"Nonfinite tangent output: {name}"
        blocks = {name: np.zeros_like(value) for name, value in full.items()}
        for name in names:
            single = {
                other: (directions[other] if other == name else 0 * directions[other])
                for other in names
            }
            for output, value in tangents(single).items():
                blocks[output] += np.abs(value)
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
            upper, lower = evaluate(plus), evaluate(minus)
            per_output = {}
            for output, analytical in full.items():
                numerical = (upper[output] - lower[output]) / (2 * epsilon)
                normalization = max(
                    float(np.max(np.abs(analytical), initial=0)),
                    float(np.max(np.abs(numerical), initial=0)),
                    float(np.max(blocks[output], initial=0)),
                    1e-12,
                )
                per_output[output] = (
                    float(np.max(np.abs(analytical - numerical), initial=0))
                    / normalization
                )
            worst = max(per_output, key=per_output.get) if per_output else None
            errors.append(
                {
                    "epsilon": epsilon,
                    "normalized_error": per_output[worst] if worst else 0.0,
                    "worst_output": worst,
                    "per_output": per_output,
                }
            )
        stable_interval = any(
            first["normalized_error"] <= tolerance
            and second["normalized_error"] <= tolerance
            for first, second in pairwise(errors)
        )
        assert stable_interval, (
            f"Direction {direction_index} failed JVP agreement: "
            f"errors={[(e['epsilon'], e['normalized_error'], e['worst_output']) for e in errors]}"
        )
        checks.append({"samples": errors})
    return {
        "device": device,
        "seed": seed,
        "differentiable_inputs": names,
        "tolerance": tolerance,
        "directions": checks,
        "best_error": [
            min(e["normalized_error"] for e in c["samples"]) for c in checks
        ],
    }


def dot_product_error(
    tangent_outputs: Mapping[str, Any],
    cotangents: Mapping[str, Any],
    directions: Mapping[str, Any],
    adjoint_inputs: Mapping[str, Any],
    input_scales: Mapping[str, float] | None = None,
    output_scales: Mapping[str, float] | None = None,
) -> float:
    """Return the normalized dot-product error of a JVP against a VJP.

    Inversion plan Section 4.7:
    e = |<J u, v> - <u, J^T v>| / (|J u| |v| + |u| |J^T v|).
    The norms are taken in scaled variables (inputs divided by their scale,
    outputs divided by theirs), so that blocks with different units weigh
    equally. The inner products do not depend on the scaling. Complex
    arrays count as real pairs (the paired-real inner product).

    Args:
        tangent_outputs: J u by output name.
        cotangents: v by output name.
        directions: u by input name.
        adjoint_inputs: J^T v by input name.
        input_scales, output_scales: Scale per name; default 1.
    """
    input_scales = input_scales or {}
    output_scales = output_scales or {}
    lhs = rhs = 0.0
    ju = v = u = jtv = 0.0
    for name, value in tangent_outputs.items():
        a, b = _host_output(value), _host_output(cotangents[name])
        s = output_scales.get(name, 1.0)
        lhs += float(np.sum(a * b))
        ju += float(np.sum((a / s) ** 2))
        v += float(np.sum((b * s) ** 2))
    for name, value in directions.items():
        a, b = _host_output(value), _host_output(adjoint_inputs[name])
        s = input_scales.get(name, 1.0)
        rhs += float(np.sum(a * b))
        u += float(np.sum((a / s) ** 2))
        jtv += float(np.sum((b * s) ** 2))
    denominator = np.sqrt(ju * v) + np.sqrt(u * jtv)
    return abs(lhs - rhs) / denominator if denominator > 0 else abs(lhs - rhs)
