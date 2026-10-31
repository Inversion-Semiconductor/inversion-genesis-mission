# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Expected-value (EV) ionization of the nitrogen K shell (plan Section 5).

Each N ion carries the population fractions P5, P6, P7 and a pending birth
weight W. For each step, with the ADK exponent

    x_l = a_l E^q_l exp(c_l / E) / gamma,   p_l = -expm1(-x_l),

the populations move with the old values (one level per step, as production):

    D56 = P5 p5,  D67 = P6 p6,
    P5' = P5 - D56,  P6' = P6 + D56 - D67,  P7' = P7 + D67,
    W'  = W + w (D56 + D67).

An ion with W' > epsilon w emits one electron of weight W' into the target
species in the same step, and resets W to 0. The ion is pushed and deposited
with the charge e (w Zbar - W), so pending births are neutral.

Only the forward model is implemented here (Phase P3). The kernel
`ev_ionize` keeps `enable_backward=True`, so that Phase P6 can use its
Warp-generated VJP as the reference for the hand-written tangent.
"""

from __future__ import annotations

from types import SimpleNamespace

import cupy
import numpy as np
import warp as wp
from scipy.constants import c, e

from fbpic.autodiff.state import EVArrays, StepState
from fbpic.particles.elementary_process.cuda_numba_utils import (
    generate_new_ids,
    reallocate_and_copy_old,
)
from fbpic.particles.elementary_process.ionization.ionizer import Ionizer
from fbpic.particles.elementary_process.ionization.warp_inline_functions import (
    get_E_amplitude,
)
from fbpic.utils.warp import as_warp_array, get_runtime

LIGHT_SPEED = wp.constant(wp.float64(c))
LEVEL_START = 5


def adk_coefficients(element: str, dt: float, levels=(5, 6)) -> dict:
    """ADK prefactor, power, and exponent prefactor of the given transitions,
    from the production formula (`Ionizer.initialize_ADK_parameters`)."""
    holder = SimpleNamespace(level_max=None)
    Ionizer.initialize_ADK_parameters(holder, element, dt)
    levels = list(levels)
    return {
        "adk_prefactor": np.asarray(holder.adk_prefactor)[levels].astype(float),
        "adk_power": np.asarray(holder.adk_power)[levels].astype(float),
        "adk_exp_prefactor": np.asarray(holder.adk_exp_prefactor)[levels].astype(float),
    }


# Device functions
# ----------------


@wp.func
def expm1_accurate(x: wp.float64):
    """exp(x) - 1 to a few ulp (Kahan): (u - 1) x / log(u), u = exp(x)."""
    u = wp.exp(x)
    if u == wp.float64(1.0):
        return x
    um1 = u - wp.float64(1.0)
    if um1 == wp.float64(-1.0):
        return wp.float64(-1.0)
    return um1 * x / wp.log(u)


@wp.func
def adk_exponent(
    E: wp.float64,
    gamma: wp.float64,
    prefactor: wp.float64,
    power: wp.float64,
    exp_prefactor: wp.float64,
):
    """x = prefactor E^power exp(exp_prefactor / E) / gamma; 0 at E = 0."""
    if E == wp.float64(0.0):
        return wp.float64(0.0)
    return prefactor * wp.pow(E, power) * wp.exp(exp_prefactor / E) / gamma


# Kernels
# -------


@wp.kernel(enable_backward=True, module="unique")
def ev_ionize(
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    Ex: wp.array(dtype=wp.float64),
    Ey: wp.array(dtype=wp.float64),
    Ez: wp.array(dtype=wp.float64),
    Bx: wp.array(dtype=wp.float64),
    By: wp.array(dtype=wp.float64),
    Bz: wp.array(dtype=wp.float64),
    w: wp.array(dtype=wp.float64),
    pop5: wp.array(dtype=wp.float64),
    pop6: wp.array(dtype=wp.float64),
    pop7: wp.array(dtype=wp.float64),
    w_pending: wp.array(dtype=wp.float64),
    prefactor5: wp.float64,
    power5: wp.float64,
    exp_prefactor5: wp.float64,
    prefactor6: wp.float64,
    power6: wp.float64,
    exp_prefactor6: wp.float64,
    epsilon: wp.float64,
    use_mask: wp.int32,
    emit_mask: wp.array(dtype=wp.int32),
    pop5_out: wp.array(dtype=wp.float64),
    pop6_out: wp.array(dtype=wp.float64),
    pop7_out: wp.array(dtype=wp.float64),
    w_pending_out: wp.array(dtype=wp.float64),
    birth_w: wp.array(dtype=wp.float64),
    emit: wp.array(dtype=wp.int32),
):
    """One EV step per ion (Section 5.1), out of place.

    Writes the new populations and pending weight, the weight `birth_w` of the
    electron that the ion emits in this step (0 if none), and `emit` (0 or 1).
    With use_mask != 0, `emit_mask` replaces the threshold test (frozen
    decisions, Section 6.3).
    """
    i = wp.tid()
    if i < ux.shape[0]:
        E, gamma = get_E_amplitude(
            ux[i],
            uy[i],
            uz[i],
            Ex[i],
            Ey[i],
            Ez[i],
            LIGHT_SPEED * Bx[i],
            LIGHT_SPEED * By[i],
            LIGHT_SPEED * Bz[i],
        )
        x5 = adk_exponent(E, gamma, prefactor5, power5, exp_prefactor5)
        x6 = adk_exponent(E, gamma, prefactor6, power6, exp_prefactor6)
        p5 = -expm1_accurate(-x5)
        p6 = -expm1_accurate(-x6)
        d56 = pop5[i] * p5
        d67 = pop6[i] * p6
        pop5_out[i] = pop5[i] - d56
        pop6_out[i] = pop6[i] + d56 - d67
        pop7_out[i] = pop7[i] + d67
        pending = w_pending[i] + w[i] * (d56 + d67)
        fire = pending > epsilon * w[i]
        if use_mask != 0:
            fire = emit_mask[i] != 0
        if fire:
            birth_w[i] = pending
            w_pending_out[i] = wp.float64(0.0)
            emit[i] = 1
        else:
            birth_w[i] = wp.float64(0.0)
            w_pending_out[i] = pending
            emit[i] = 0


@wp.kernel(enable_backward=False, module="unique")
def copy_ev_electrons(
    emit: wp.array(dtype=wp.int32),
    slot: wp.array(dtype=wp.int64),
    elec_old_Ntot: wp.int64,
    birth_w: wp.array(dtype=wp.float64),
    ion_x: wp.array(dtype=wp.float64),
    ion_y: wp.array(dtype=wp.float64),
    ion_z: wp.array(dtype=wp.float64),
    ion_ux: wp.array(dtype=wp.float64),
    ion_uy: wp.array(dtype=wp.float64),
    ion_uz: wp.array(dtype=wp.float64),
    ion_inv_gamma: wp.array(dtype=wp.float64),
    elec_x: wp.array(dtype=wp.float64),
    elec_y: wp.array(dtype=wp.float64),
    elec_z: wp.array(dtype=wp.float64),
    elec_ux: wp.array(dtype=wp.float64),
    elec_uy: wp.array(dtype=wp.float64),
    elec_uz: wp.array(dtype=wp.float64),
    elec_inv_gamma: wp.array(dtype=wp.float64),
    elec_w: wp.array(dtype=wp.float64),
):
    """Create the emitted electrons in parent order (Section 5.2).

    `slot` is the exclusive prefix sum of `emit`, so the electron of ion i
    goes to elec_old_Ntot + slot[i]. No atomic counter is used.
    """
    i = wp.tid()
    # Nested on purpose: Warp may evaluate both operands of `and`.
    if i < emit.shape[0]:  # noqa: SIM102
        if emit[i] != 0:
            j = elec_old_Ntot + slot[i]
            elec_x[j] = ion_x[i]
            elec_y[j] = ion_y[i]
            elec_z[j] = ion_z[i]
            elec_ux[j] = ion_ux[i]
            elec_uy[j] = ion_uy[i]
            elec_uz[j] = ion_uz[i]
            elec_inv_gamma[j] = ion_inv_gamma[i]
            elec_w[j] = birth_w[i]


def _launch(kernel, n, inputs):
    """Launch a 1D kernel of n threads on the shared runtime stream."""
    if n == 0:
        return
    runtime = get_runtime()
    wp.launch(
        kernel,
        dim=n,
        inputs=[as_warp_array(a) if isinstance(a, cupy.ndarray) else a for a in inputs],
        device=runtime.device,
        stream=runtime.stream,
    )


# State management
# ----------------


def attach(state: StepState, epsilon: float = 1e-4, K_e: int = 1, immobile=False):
    """Create the EV arrays of the ion species (P = (1, 0, 0), W = 0).

    Args:
        state: State whose `ev_species` holds fixed-charge N5+ ions.
        epsilon: Emission threshold relative to the ion weight (D12).
        K_e: Emission period; only 1 is supported (Section 5.2).
        immobile: Freeze the ion positions and momenta (Phase P3 step 8).
    """
    if K_e != 1:
        raise ValueError("Only K_e = 1 is supported (Section 5.2).")
    ion = state.species[state.ev_species]
    if abs(ion.q - LEVEL_START * e) > 1e-6 * e:
        raise ValueError("The EV ions must be built as fixed-charge N5+.")
    n = int(ion.Ntot)
    state.ev = EVArrays(
        pop5=cupy.ones(n),
        pop6=cupy.zeros(n),
        pop7=cupy.zeros(n),
        w_pending=cupy.zeros(n),
    )
    params = dict(adk_coefficients("N", state.sim.dt))
    params.update(epsilon=float(epsilon), K_e=K_e, immobile=bool(immobile))
    state.ev_params = params
    state.ledger = {
        "emitted": cupy.zeros((), dtype=np.float64),
        "removed_elec_w": 0.0,
        "removed_ion_ionized": 0.0,
        "removed_ion_pending": 0.0,
        "emitted_first_id": [],
    }
    state.emit_masks = {}


def snapshot(state: StepState) -> dict:
    """Copy the ion IDs and EV arrays, and the target electron IDs and weights,
    before a production exchange."""
    ion = state.species[state.ev_species]
    elec = state.species[state.ev_target]
    n, ne = int(ion.Ntot), int(elec.Ntot)
    return {
        "ids": ion.tracker.id[:n].copy(),
        "w": ion.w[:n].copy(),
        "arrays": [a[:n].copy() for a in state.ev.arrays()],
        "next_id": int(ion.tracker.next_attributed_id),
        "elec_ids": elec.tracker.id[:ne].copy(),
        "elec_w": elec.w[:ne].copy(),
    }


def _match(old_ids, new_ids):
    """Index into old_ids of each new ID, and whether it was found."""
    if old_ids.size == 0:
        return cupy.zeros(new_ids.size, np.int64), cupy.zeros(new_ids.size, bool)
    order = cupy.argsort(old_ids)
    sorted_ids = old_ids[order]
    pos = cupy.searchsorted(sorted_ids, new_ids)
    pos = cupy.minimum(pos, old_ids.size - 1)
    found = sorted_ids[pos] == new_ids
    return order[pos], found


def realign(state: StepState, snap: dict):
    """Realign the EV arrays with the ions after a production exchange.

    Kept ions take their old values by tracker ID. Injected ions start at
    P = (1, 0, 0), W = 0. Updates the removal ledgers (Phase P3 step 4).
    """
    ion = state.species[state.ev_species]
    elec = state.species[state.ev_target]
    n = int(ion.Ntot)
    new_ids = ion.tracker.id[:n]
    idx, found = _match(snap["ids"], new_ids)
    if not bool(cupy.all(found | (new_ids >= snap["next_id"]))):
        raise RuntimeError("An exchange created ions with old tracker IDs.")
    init = (1.0, 0.0, 0.0, 0.0)
    new_arrays = []
    for old, value in zip(snap["arrays"], init):
        if old.size:
            new_arrays.append(cupy.where(found, old[idx], value))
        else:
            new_arrays.append(cupy.full(n, value))
    state.ev = EVArrays(*new_arrays)
    # Removal ledgers: ions and electrons whose IDs disappeared
    kept_ion = cupy.zeros(snap["ids"].size, bool)
    kept_ion[idx[found]] = True
    removed = ~kept_ion
    p5, p6, p7, pending = snap["arrays"]
    zbar = 5.0 * p5 + 6.0 * p6 + 7.0 * p7
    ledger = state.ledger
    ledger["removed_ion_ionized"] += float((snap["w"] * (zbar - 5.0))[removed].sum())
    ledger["removed_ion_pending"] += float(pending[removed].sum())
    ne = int(elec.Ntot)
    eidx, efound = _match(snap["elec_ids"], elec.tracker.id[:ne])
    kept_elec = cupy.zeros(snap["elec_ids"].size, bool)
    kept_elec[eidx[efound]] = True
    ledger["removed_elec_w"] += float(snap["elec_w"][~kept_elec].sum())


def charge_weights(state: StepState):
    """w_tl = w Zbar - W of the ions (Section 4.2, rule 2)."""
    ion = state.species[state.ev_species]
    n = int(ion.Ntot)
    ev = state.ev
    zbar = 5.0 * ev.pop5 + 6.0 * ev.pop6 + 7.0 * ev.pop7
    return ion.w[:n] * zbar - ev.w_pending


def set_deposit_charge(state: StepState):
    """Deposit the ions with charge e and weight w_tl until `restore_charge`."""
    ion = state.species[state.ev_species]
    state._ev_saved = (ion.w, ion.q)
    ion.w = charge_weights(state)
    ion.q = e


def restore_charge(state: StepState):
    saved = getattr(state, "_ev_saved", None)
    if saved is not None:
        ion = state.species[state.ev_species]
        ion.w, ion.q = saved
        state._ev_saved = None


def push_ion_momenta(state: StepState):
    """Vay push of the ions with the charge e w_tl / w (Section 4.4)."""
    from fbpic.autodiff.particles import push_ion_momenta_ev

    ion = state.species[state.ev_species]
    if state.ev_params["immobile"]:
        return
    n = int(ion.Ntot)
    if n == 0:
        return
    out = [cupy.empty(n) for _ in range(4)]
    _launch(
        push_ion_momenta_ev,
        n,
        [
            ion.ux[:n],
            ion.uy[:n],
            ion.uz[:n],
            ion.inv_gamma[:n],
            ion.Ex[:n],
            ion.Ey[:n],
            ion.Ez[:n],
            ion.Bx[:n],
            ion.By[:n],
            ion.Bz[:n],
            ion.w[:n],
            charge_weights(state),
            float(ion.m),
            float(state.sim.dt),
            *out,
        ],
    )
    for name, array in zip(("ux", "uy", "uz", "inv_gamma"), out):
        getattr(ion, name)[:n] = array


def ionize_and_emit(state: StepState):
    """EV update and emission for one step (Sections 5.1 and 5.2).

    Uses `state.emit_masks[iteration]` (device array of parent tracker IDs)
    instead of the threshold when it exists.

    Returns:
        Device array of the tracker IDs of the emitting ions, in slot order.
    """
    ion = state.species[state.ev_species]
    elec = state.species[state.ev_target]
    n = int(ion.Ntot)
    params = state.ev_params
    if n == 0:
        return cupy.zeros(0, dtype=np.uint64)
    ids = ion.tracker.id[:n]
    mask_ids = state.emit_masks.get(state.iteration)
    use_mask = 0 if mask_ids is None else 1
    emit_mask = (
        cupy.zeros(1, np.int32)
        if mask_ids is None
        else cupy.isin(ids, mask_ids).astype(np.int32)
    )
    ev = state.ev
    outs = [cupy.empty(n) for _ in range(5)]
    emit = cupy.empty(n, dtype=np.int32)
    a, q, cexp = (
        params["adk_prefactor"],
        params["adk_power"],
        params["adk_exp_prefactor"],
    )
    _launch(
        ev_ionize,
        n,
        [
            ion.ux[:n],
            ion.uy[:n],
            ion.uz[:n],
            ion.Ex[:n],
            ion.Ey[:n],
            ion.Ez[:n],
            ion.Bx[:n],
            ion.By[:n],
            ion.Bz[:n],
            ion.w[:n],
            ev.pop5,
            ev.pop6,
            ev.pop7,
            ev.w_pending,
            float(a[0]),
            float(q[0]),
            float(cexp[0]),
            float(a[1]),
            float(q[1]),
            float(cexp[1]),
            params["epsilon"],
            use_mask,
            emit_mask,
            *outs[:4],
            outs[4],
            emit,
        ],
    )
    state.ev = EVArrays(*outs[:4])
    birth_w = outs[4]
    state.ledger["emitted"] += birth_w.sum()
    slot = cupy.cumsum(emit, dtype=np.int64) - emit
    n_new = int(slot[-1]) + int(emit[-1])
    parent_ids = ids[emit.astype(bool)]
    if n_new > 0:
        old = int(elec.Ntot)
        reallocate_and_copy_old(elec, True, old, old + n_new)
        _launch(
            copy_ev_electrons,
            n,
            [
                emit,
                slot,
                old,
                birth_w,
                ion.x[:n],
                ion.y[:n],
                ion.z[:n],
                ion.ux[:n],
                ion.uy[:n],
                ion.uz[:n],
                ion.inv_gamma[:n],
                elec.x,
                elec.y,
                elec.z,
                elec.ux,
                elec.uy,
                elec.uz,
                elec.inv_gamma,
                elec.w,
            ],
        )
        first_id = int(elec.tracker.next_attributed_id)
        generate_new_ids(elec, old, old + n_new)
        elec.sorted = False
        state.ledger["emitted_first_id"].append((state.iteration, first_id, parent_ids))
    return parent_ids
