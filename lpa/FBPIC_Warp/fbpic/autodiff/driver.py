# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Forward-mode driver: `jvp` and `value_and_grad` (inversion plan Section 4.10).

One pass of `step` carries the primal state and K tangents together, one per
parameter: K = 1 (E_L) in Stage A. Only the current state, the current
tangents, and the per-step stash live in memory. The gradient is the
paired-real inner product of the objective's state gradient with the final
tangents.

The driver does not build simulations itself. A `build(theta)` callable
returns `(sim, species, profile)`: the production `Simulation` built and not
stepped, the species by name, and the laser profile. The research code in
`inversion/` provides builders for the twins and for L4w.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cupy
import numpy as np

from fbpic.autodiff.field_tangents import STATE_SPECT_NAMES
from fbpic.autodiff.laser import amplitude_seed, energy_seed, laser_initial_fields
from fbpic.autodiff.state import TANGENT_PARTICLE_NAMES, TangentState
from fbpic.autodiff.stepper import begin, step


@dataclass
class JVPResult:
    """Final state, tangents, decisions, and run information of `jvp`."""

    state: object
    tangents: TangentState
    decisions: list
    info: dict = field(default_factory=dict)


@dataclass
class GradResult:
    """Result of `value_and_grad`.

    Attributes:
        J: Objective value.
        grad: dJ/dtheta, shape (K,), in the units of theta.
        grad_hat: dJ/dtheta_hat = (theta_hi - theta_lo) * grad, or None
            without bounds.
        terms: Per-term values and gradients when the objective returns terms.
        info: Memory peaks, timings, and tangent norms.
        residuals: r_w, shape (R,), when the objective returns residuals.
        jacobian: dr_w/dtheta, shape (R, K), with residuals.
        jacobian_hat: dr_w/dtheta_hat, shape (R, K), with residuals and bounds.
    """

    J: float
    grad: np.ndarray
    grad_hat: np.ndarray | None
    terms: dict
    info: dict
    residuals: np.ndarray | None = None
    jacobian: np.ndarray | None = None
    jacobian_hat: np.ndarray | None = None


def resolve_seeds(sim, profile, theta, seeds) -> list:
    """Initial interp-field tangents, one list of per-mode dicts per seed.

    Args:
        seeds: Names or callables. "E_L": F0 / (2 E_L), with E_L = theta[0].
            "a0": F0 / a0, with a0 = profile.a0 (L4w). A callable
            seed(sim, profile, theta) returns the per-mode dicts itself.
    """
    initial = None
    out = []
    for seed in seeds:
        if callable(seed):
            out.append(seed(sim, profile, theta))
            continue
        if initial is None:
            initial = laser_initial_fields(sim, profile)
        if seed == "E_L":
            out.append(energy_seed(initial, float(theta[0])))
        elif seed == "a0":
            out.append(amplitude_seed(initial, float(profile.a0)))
        else:
            raise ValueError(f"Unknown seed: {seed}")
    return out


def seed_tangents(state, seeds) -> TangentState:
    """Zero particle tangents and the seeded field tangents after `begin`.

    The interp E and B tangents take the seeds; then the field operations of
    the `begin` prologue (open-z damping D, interp2spect T) run on them, so
    that the spectral tangents are T D t_j (plan Section 4.8). The `rho_prev`
    tangent is zero because the particle tangents are zero.
    """
    sim = state.sim
    tangents = TangentState.zeros(state, len(seeds))
    for k, seed in enumerate(seeds):
        for m, mode in enumerate(seed):
            for name, value in mode.items():
                tangents.fields.interp[m][name][k] = cupy.asarray(value)

    def prologue():
        sim.comm.exchange_fields(sim.fld.interp, "E", "replace")
        sim.comm.exchange_fields(sim.fld.interp, "B", "replace")
        sim.comm.damp_EB_open_boundary(sim.fld.interp)
        sim.fld.interp2spect("E")
        sim.fld.interp2spect("B")

    tangents.fields.apply(sim.fld, prologue)
    return tangents


def tangent_norms(tangents: TangentState) -> list[dict]:
    """Infinity norms of each tangent: spectral fields and particle arrays."""
    out = []
    for k in range(tangents.K):
        norms = {}
        for name in STATE_SPECT_NAMES:
            norms[f"spect.{name}"] = max(
                float(cupy.abs(mode[name][k]).max()) for mode in tangents.fields.spect
            )
        for comp in TANGENT_PARTICLE_NAMES:
            values = [
                float(cupy.abs(arrays[comp][k]).max())
                for arrays in tangents.particles.values()
                if arrays[comp].shape[1] > 0
            ]
            norms[f"particles.{comp}"] = max(values, default=0.0)
        out.append(norms)
    return out


def _memory():
    free, total = cupy.cuda.runtime.memGetInfo()
    pool = cupy.get_default_memory_pool()
    return {"device_used": int(total - free), "cupy_pool_used": int(pool.used_bytes())}


def jvp(
    theta,
    build,
    seeds=("E_L",),
    n_steps: int = 1,
    *,
    callback=None,
    norm_every: int = 0,
    begin_kwargs: dict | None = None,
) -> JVPResult:
    """Run n_steps steps with K tangents and return the final state.

    Args:
        theta: Parameter vector; theta[0] is E_L in joules.
        build: build(theta) -> (sim, species, profile).
        seeds: One seed per tangent (`resolve_seeds`).
        n_steps: Number of steps after `begin`.
        callback: Optional callback(state, tangents, decisions) after each
            step, for horizon sweeps.
        norm_every: Record the tangent norms every this many steps (0: never).
        begin_kwargs: Keyword arguments of `begin` (ionization options).

    Returns:
        `JVPResult` with the final state, the tangents, the decisions of every
        step, and timings and memory in `info`.
    """
    sim, species, profile = build(theta)
    resolved = resolve_seeds(sim, profile, theta, seeds)
    t0 = time.perf_counter()
    state = begin(sim, species, **(begin_kwargs or {}))
    tangents = seed_tangents(state, resolved)
    cupy.cuda.get_current_stream().synchronize()
    t1 = time.perf_counter()
    decisions, norms = [], []
    peak = _memory()
    for n in range(n_steps):
        decisions.append(step(state, tangents)[2])
        if callback is not None:
            callback(state, tangents, decisions[-1])
        if norm_every and (n + 1) % norm_every == 0:
            norms.append({"step": n + 1, "norms": tangent_norms(tangents)})
            current = _memory()
            peak = {key: max(peak[key], current[key]) for key in peak}
    cupy.cuda.get_current_stream().synchronize()
    t2 = time.perf_counter()
    info = {
        "K": tangents.K,
        "n_steps": n_steps,
        "time_begin_s": t1 - t0,
        "time_steps_s": t2 - t1,
        "time_per_step_s": (t2 - t1) / max(n_steps, 1),
        "memory_peak_sampled": peak,
        "tangent_norms": norms,
    }
    return JVPResult(state, tangents, decisions, info)


def contract(gradient: dict, tangents: TangentState, sim) -> np.ndarray:
    """Paired-real inner product <g_s, sdot[k]> for every tangent k.

    Args:
        gradient: {"fields": [per-mode dict name -> (Nz, Nr) complex],
            "particles": {species: {component: (N,) float}}}. A complex
            field gradient g holds dJ/dRe + i dJ/dIm, so that
            dJ = sum Re(conj(g) sdot).
    """
    out = np.zeros(tangents.K)
    for k in range(tangents.K):
        total = 0.0
        for m, mode in enumerate(gradient.get("fields", [])):
            for name, g in mode.items():
                total += float(
                    cupy.real(cupy.vdot(g, tangents.fields.spect[m][name][k]))
                )
        for name, comps in gradient.get("particles", {}).items():
            for comp, g in comps.items():
                total += float(cupy.dot(g, tangents.particles[name][comp][k]))
        out[k] = total
    return out


def value_and_grad(
    theta,
    objective,
    build,
    n_steps: int,
    seeds=("E_L",),
    bounds=None,
    **jvp_kwargs,
) -> GradResult:
    """Objective value and its gradient in one forward pass with K tangents.

    Args:
        theta: Parameter vector.
        objective: objective(state) -> (J, g_s), or a dict of terms
            {name: (J_term, g_term)}, whose values and gradients add up, or
            a residual value with `residuals` (R,) and `contract(tangents)`
            -> dr/dtheta (R, K), such as `loss.ResidualObjective`. Then
            J = |r|^2, grad = 2 J_r^T r, and the result holds the Jacobian.
        build: build(theta) -> (sim, species, profile).
        n_steps: Horizon N_t.
        seeds: One seed per tangent; the gradient has one entry per seed.
        bounds: (lo, hi) of the seeded parameters, for grad_hat.
        jvp_kwargs: Further arguments of `jvp`.
    """
    result = jvp(theta, build, seeds, n_steps, **jvp_kwargs)
    value = objective(result.state)
    scale = None
    if bounds is not None:
        lo, hi = (np.asarray(b, float)[: result.tangents.K] for b in bounds)
        scale = hi - lo
    if hasattr(value, "residuals"):
        r = np.asarray(value.residuals, float)
        jac = value.contract(result.tangents)
        grad = 2.0 * jac.T @ r
        result.info["decisions"] = result.decisions
        return GradResult(
            J=float(r @ r),
            grad=grad,
            grad_hat=None if scale is None else scale * grad,
            terms={},
            info=result.info,
            residuals=r,
            jacobian=jac,
            jacobian_hat=None if scale is None else jac * scale[None, :],
        )
    terms = value if isinstance(value, dict) else {"total": value}
    out_terms = {}
    J, grad = 0.0, np.zeros(result.tangents.K)
    for name, (J_term, g_term) in terms.items():
        d = contract(g_term, result.tangents, result.state.sim)
        out_terms[name] = {"J": float(J_term), "grad": d}
        J += float(J_term)
        grad += d
    grad_hat = None
    if scale is not None:
        grad_hat = scale * grad
        for term in out_terms.values():
            term["grad_hat"] = scale * term["grad"]
    result.info["decisions"] = result.decisions
    return GradResult(J, grad, grad_hat, out_terms, result.info)
