# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Forward-mode tangent (JVP) kernels of the particle operations.

Inversion plan Section 4.7. Warp 1.18 generates reverse-mode code only, so the
tangents of the linear gather, the Vay momentum push, the position push, and
the fused linear deposition are written by hand. Each kernel differentiates
the production kernel body line by line, with every discrete choice (cell
indices, axis branch, `rmax_gather` branch, radial clamp, guard-cell fold)
taken from the primal and frozen.

Layout. Particle tangents are (K, N) FP64 arrays: row k is the tangent of
parameter k. Field tangents are (K, Nz, Nr) complex128 arrays, one per mode
and component. The mode count (at most 5) and K are compile-time constants
of the gather and deposition kernels; the push kernels loop over K at run
time.

Inverse gamma. `inv_gamma` is an independent state variable (plan Section
4.2, rule 1): `vay_tangent` writes the tangent of the new value, and
`position_tangent` and `deposit_tangent` read that stored tangent.

The host helpers take CuPy arrays and launch on the shared runtime stream.
"""

from __future__ import annotations

from functools import cache

import cupy
import warp as wp
from scipy.constants import c

from fbpic.particles.deposition.cuda_methods_fused import ModeBetas, mode_betas
from fbpic.particles.deposition.cuda_particle_shapes import Sr_linear, Sz_linear
from fbpic.particles.gathering.cuda_methods_fused import (
    FIELD_NAMES,
    GroupFields,
    group_fields,
    select_mode,
)
from fbpic.utils.warp import as_warp_array, cmul, get_runtime

LIGHT_SPEED = wp.constant(wp.float64(c))
MAX_MODES = 5


def _launch(kernel, dim, inputs, block_dim=256):
    """Launch a kernel on the shared runtime stream; skip empty launches."""
    if dim == 0:
        return
    runtime = get_runtime()
    wp.launch(
        kernel,
        dim=dim,
        inputs=[as_warp_array(a) if isinstance(a, cupy.ndarray) else a for a in inputs],
        device=runtime.device,
        stream=runtime.stream,
        block_dim=block_dim,
    )


# Field tangent structs
# ---------------------


@wp.struct
class ModeFieldTangents:
    Er: wp.array3d(dtype=wp.vec2d)
    Et: wp.array3d(dtype=wp.vec2d)
    Ez: wp.array3d(dtype=wp.vec2d)
    Br: wp.array3d(dtype=wp.vec2d)
    Bt: wp.array3d(dtype=wp.vec2d)
    Bz: wp.array3d(dtype=wp.vec2d)


@wp.struct
class GroupFieldTangents:
    mode0: ModeFieldTangents
    mode1: ModeFieldTangents
    mode2: ModeFieldTangents
    mode3: ModeFieldTangents
    mode4: ModeFieldTangents


@wp.struct
class OutputTangents:
    component0: wp.array3d(dtype=wp.vec2d)
    component1: wp.array3d(dtype=wp.vec2d)
    component2: wp.array3d(dtype=wp.vec2d)
    component3: wp.array3d(dtype=wp.vec2d)
    component4: wp.array3d(dtype=wp.vec2d)
    component5: wp.array3d(dtype=wp.vec2d)
    component6: wp.array3d(dtype=wp.vec2d)
    component7: wp.array3d(dtype=wp.vec2d)
    component8: wp.array3d(dtype=wp.vec2d)
    component9: wp.array3d(dtype=wp.vec2d)
    component10: wp.array3d(dtype=wp.vec2d)
    component11: wp.array3d(dtype=wp.vec2d)
    component12: wp.array3d(dtype=wp.vec2d)
    component13: wp.array3d(dtype=wp.vec2d)
    component14: wp.array3d(dtype=wp.vec2d)


def group_field_tangents(modes) -> GroupFieldTangents:
    """Borrow the (K, Nz, Nr) E and B tangents of up to five modes.

    Args:
        modes: One mapping per mode from the names Er, Et, Ez, Br, Bt, Bz to
            complex128 CuPy arrays of shape (K, Nz, Nr). Unused slots repeat
            the last mode; the kernel never reads them.
    """
    fields = GroupFieldTangents()
    for slot in range(MAX_MODES):
        mode = modes[min(slot, len(modes) - 1)]
        tangent = ModeFieldTangents()
        for name in FIELD_NAMES:
            setattr(tangent, name, as_warp_array(mode[name], wp.vec2d))
        setattr(fields, f"mode{slot}", tangent)
    return fields


def output_tangents(arrays) -> OutputTangents:
    """Borrow (K, Nz, Nr) complex128 output tangents, field-major order."""
    buffers = OutputTangents()
    for index in range(15):
        setattr(
            buffers,
            f"component{index}",
            as_warp_array(arrays[min(index, len(arrays) - 1)], wp.vec2d),
        )
    return buffers


@wp.func
def select_tangent_mode(fields: GroupFieldTangents, local_mode: int):
    # local_mode is a compile-time constant at each call site (unrolled loop).
    selected = fields.mode0
    if local_mode == 1:
        selected = fields.mode1
    elif local_mode == 2:
        selected = fields.mode2
    elif local_mode == 3:
        selected = fields.mode3
    elif local_mode == 4:
        selected = fields.mode4
    return selected


@wp.func
def atomic_tangent(
    outputs: OutputTangents, component: int, k: int, iz: int, ir: int, value: wp.vec2d
):
    if component == 0:
        wp.atomic_add(outputs.component0, k, iz, ir, value)
    elif component == 1:
        wp.atomic_add(outputs.component1, k, iz, ir, value)
    elif component == 2:
        wp.atomic_add(outputs.component2, k, iz, ir, value)
    elif component == 3:
        wp.atomic_add(outputs.component3, k, iz, ir, value)
    elif component == 4:
        wp.atomic_add(outputs.component4, k, iz, ir, value)
    elif component == 5:
        wp.atomic_add(outputs.component5, k, iz, ir, value)
    elif component == 6:
        wp.atomic_add(outputs.component6, k, iz, ir, value)
    elif component == 7:
        wp.atomic_add(outputs.component7, k, iz, ir, value)
    elif component == 8:
        wp.atomic_add(outputs.component8, k, iz, ir, value)
    elif component == 9:
        wp.atomic_add(outputs.component9, k, iz, ir, value)
    elif component == 10:
        wp.atomic_add(outputs.component10, k, iz, ir, value)
    elif component == 11:
        wp.atomic_add(outputs.component11, k, iz, ir, value)
    elif component == 12:
        wp.atomic_add(outputs.component12, k, iz, ir, value)
    elif component == 13:
        wp.atomic_add(outputs.component13, k, iz, ir, value)
    else:
        wp.atomic_add(outputs.component14, k, iz, ir, value)


@wp.func
def beta_of(betas: ModeBetas, mode: int, radial_bin: int):
    value = wp.float64(0.0)
    if mode == 0:
        value = betas.mode0[radial_bin]
    elif mode == 1:
        value = betas.mode1[radial_bin]
    elif mode == 2:
        value = betas.mode2[radial_bin]
    elif mode == 3:
        value = betas.mode3[radial_bin]
    else:
        value = betas.mode4[radial_bin]
    return value


# Gather
# ------


@wp.func
def stencil(
    grid: wp.array2d(dtype=wp.vec2d),
    iz_lower: int,
    iz_upper: int,
    ir_lower: int,
    ir_upper: int,
    S_ll: wp.float64,
    S_lu: wp.float64,
    S_ul: wp.float64,
    S_uu: wp.float64,
    S_lg: wp.float64,
    S_ug: wp.float64,
    guard_sign: wp.float64,
):
    """Linear gather sum of one mode grid (`add_linear_gather_for_mode`)."""
    value = S_ll * grid[iz_lower, ir_lower]
    value += S_lu * grid[iz_lower, ir_upper]
    value += S_ul * grid[iz_upper, ir_lower]
    value += S_uu * grid[iz_upper, ir_upper]
    if ir_lower == 0 and ir_upper == 0:
        value += guard_sign * S_lg * grid[iz_lower, 0]
        value += guard_sign * S_ug * grid[iz_upper, 0]
    return value


@wp.func
def stencil_k(
    grid: wp.array3d(dtype=wp.vec2d),
    k: int,
    iz_lower: int,
    iz_upper: int,
    ir_lower: int,
    ir_upper: int,
    S_ll: wp.float64,
    S_lu: wp.float64,
    S_ul: wp.float64,
    S_uu: wp.float64,
    S_lg: wp.float64,
    S_ug: wp.float64,
    guard_sign: wp.float64,
):
    """`stencil` applied to tangent k of a (K, Nz, Nr) grid."""
    value = S_ll * grid[k, iz_lower, ir_lower]
    value += S_lu * grid[k, iz_lower, ir_upper]
    value += S_ul * grid[k, iz_upper, ir_lower]
    value += S_uu * grid[k, iz_upper, ir_upper]
    if ir_lower == 0 and ir_upper == 0:
        value += guard_sign * S_lg * grid[k, iz_lower, 0]
        value += guard_sign * S_ug * grid[k, iz_upper, 0]
    return value


@cache
def make_gather_tangent(mode_count: int, K: int):
    """Return the tangent of the fused linear gather (modes 0 to mode_count-1).

    The primal is `gathering/cuda_methods_fused.py`. The tangent is
    G' = sum_c S_c F'_c + sum_c S'_c F_c per mode, with S' from the position
    tangents through r, z (frozen cells), the phase tangent of exp(-i m phi),
    and the tangent of the Cartesian rotation by cos and sin.
    """
    if not 1 <= mode_count <= MAX_MODES or K < 1:
        raise ValueError("The gather tangent needs 1-5 modes and K >= 1.")
    vec_k = wp.types.vector(length=K, dtype=wp.float64)

    @wp.kernel(enable_backward=False, module="unique")
    def gather_tangent(
        x: wp.array(dtype=wp.float64),
        y: wp.array(dtype=wp.float64),
        z: wp.array(dtype=wp.float64),
        dx: wp.array2d(dtype=wp.float64),
        dy: wp.array2d(dtype=wp.float64),
        dz: wp.array2d(dtype=wp.float64),
        rmax_gather: wp.float64,
        invdz: wp.float64,
        zmin: wp.float64,
        Nz: wp.int32,
        invdr: wp.float64,
        rmin: wp.float64,
        Nr: wp.int32,
        fields: GroupFields,
        tangents: GroupFieldTangents,
        dEx: wp.array2d(dtype=wp.float64),
        dEy: wp.array2d(dtype=wp.float64),
        dEz: wp.array2d(dtype=wp.float64),
        dBx: wp.array2d(dtype=wp.float64),
        dBy: wp.array2d(dtype=wp.float64),
        dBz: wp.array2d(dtype=wp.float64),
    ):
        i = wp.tid()
        if i >= x.shape[0]:
            return
        xj = x[i]
        yj = y[i]
        rj = wp.sqrt(xj * xj + yj * yj)
        cos = wp.float64(1.0)
        sin = wp.float64(0.0)
        dcos = vec_k()
        dsin = vec_k()
        dr_cell = vec_k()
        dz_cell = vec_k()
        for k in range(wp.static(K)):
            dz_cell[k] = invdz * dz[k, i]
        if rj != wp.float64(0.0):
            invr = wp.float64(1.0) / rj
            cos = xj * invr
            sin = yj * invr
            for k in range(wp.static(K)):
                drj = (xj * dx[k, i] + yj * dy[k, i]) * invr
                dinvr = -drj * invr * invr
                dcos[k] = dx[k, i] * invr + xj * dinvr
                dsin[k] = dy[k, i] * invr + yj * dinvr
                dr_cell[k] = invdr * drj
        # At r = 0 the production branch holds: cos = 1, sin = 0, and the
        # radius and angle tangents are zero (plan Section 4.7).
        r_cell = invdr * (rj - rmin) - wp.float64(0.5)
        z_cell = invdz * (z[i] - zmin) - wp.float64(0.5)

        dex = vec_k()
        dey = vec_k()
        dez = vec_k()
        dbx = vec_k()
        dby = vec_k()
        dbz = vec_k()
        if rj < rmax_gather:
            ir_lower = wp.int32(wp.floor(r_cell))
            ir_upper = ir_lower + 1
            iz_lower = wp.int32(wp.floor(z_cell))
            iz_upper = iz_lower + 1
            Sr_lower = wp.float64(ir_upper) - r_cell
            Sr_upper = r_cell - wp.float64(ir_lower)
            Sz_lower = wp.float64(iz_upper) - z_cell
            Sz_upper = z_cell - wp.float64(iz_lower)
            Sr_guard = wp.float64(0.0)
            # Tangents of the linear weights: d(Sr_lower) = -d(r_cell), ...
            dSr_lower = -dr_cell
            dSr_upper = dr_cell
            dSr_guard = vec_k()
            if ir_lower < 0:
                Sr_guard = Sr_lower
                dSr_guard = dSr_lower
                Sr_lower = wp.float64(0.0)
                dSr_lower = vec_k()
                ir_lower = wp.int32(0)
            if ir_lower > Nr - 1:
                ir_lower = Nr - 1
            if ir_upper > Nr - 1:
                ir_upper = Nr - 1
            if iz_lower < 0:
                iz_lower += Nz
            if iz_upper < 0:
                iz_upper += Nz
            if iz_lower > Nz - 1:
                iz_lower -= Nz
            if iz_upper > Nz - 1:
                iz_upper -= Nz
            S_ll = Sz_lower * Sr_lower
            S_lu = Sz_lower * Sr_upper
            S_ul = Sz_upper * Sr_lower
            S_uu = Sz_upper * Sr_upper
            S_lg = Sz_lower * Sr_guard
            S_ug = Sz_upper * Sr_guard

            # exp(-i m phi) and its tangent, advanced mode by mode
            phase_step = wp.vec2d(cos, -sin)
            phase = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
            phase_re = vec_k()
            phase_im = vec_k()
            for local_mode in range(wp.static(mode_count)):
                mode = select_mode(fields, local_mode)
                tmode = select_tangent_mode(tangents, local_mode)
                flip = wp.float64(1 - 2 * (local_mode % 2))
                factor = wp.float64(2.0)
                if local_mode == 0:
                    factor = wp.float64(1.0)
                # Primal per-mode sums (complex) of the six components
                er_m = stencil(
                    mode.Er, iz_lower, iz_upper, ir_lower, ir_upper,
                    S_ll, S_lu, S_ul, S_uu, S_lg, S_ug, -flip,
                )  # fmt: skip
                et_m = stencil(
                    mode.Et, iz_lower, iz_upper, ir_lower, ir_upper,
                    S_ll, S_lu, S_ul, S_uu, S_lg, S_ug, -flip,
                )  # fmt: skip
                ez_m = stencil(
                    mode.Ez, iz_lower, iz_upper, ir_lower, ir_upper,
                    S_ll, S_lu, S_ul, S_uu, S_lg, S_ug, flip,
                )  # fmt: skip
                br_m = stencil(
                    mode.Br, iz_lower, iz_upper, ir_lower, ir_upper,
                    S_ll, S_lu, S_ul, S_uu, S_lg, S_ug, -flip,
                )  # fmt: skip
                bt_m = stencil(
                    mode.Bt, iz_lower, iz_upper, ir_lower, ir_upper,
                    S_ll, S_lu, S_ul, S_uu, S_lg, S_ug, -flip,
                )  # fmt: skip
                bz_m = stencil(
                    mode.Bz, iz_lower, iz_upper, ir_lower, ir_upper,
                    S_ll, S_lu, S_ul, S_uu, S_lg, S_ug, flip,
                )  # fmt: skip
                Er = factor * cmul(er_m, phase)[0]
                Et = factor * cmul(et_m, phase)[0]
                Br = factor * cmul(br_m, phase)[0]
                Bt = factor * cmul(bt_m, phase)[0]
                for k in range(wp.static(K)):
                    dzc = dz_cell[k]
                    # d(Sz_lower) = -d(z_cell), d(Sz_upper) = +d(z_cell)
                    dS_ll = -dzc * Sr_lower + Sz_lower * dSr_lower[k]
                    dS_lu = -dzc * Sr_upper + Sz_lower * dSr_upper[k]
                    dS_ul = dzc * Sr_lower + Sz_upper * dSr_lower[k]
                    dS_uu = dzc * Sr_upper + Sz_upper * dSr_upper[k]
                    dS_lg = -dzc * Sr_guard + Sz_lower * dSr_guard[k]
                    dS_ug = dzc * Sr_guard + Sz_upper * dSr_guard[k]
                    dphase = wp.vec2d(phase_re[k], phase_im[k])
                    der_m = stencil(
                        mode.Er, iz_lower, iz_upper, ir_lower, ir_upper,
                        dS_ll, dS_lu, dS_ul, dS_uu, dS_lg, dS_ug, -flip,
                    ) + stencil_k(
                        tmode.Er, k, iz_lower, iz_upper, ir_lower, ir_upper,
                        S_ll, S_lu, S_ul, S_uu, S_lg, S_ug, -flip,
                    )  # fmt: skip
                    det_m = stencil(
                        mode.Et, iz_lower, iz_upper, ir_lower, ir_upper,
                        dS_ll, dS_lu, dS_ul, dS_uu, dS_lg, dS_ug, -flip,
                    ) + stencil_k(
                        tmode.Et, k, iz_lower, iz_upper, ir_lower, ir_upper,
                        S_ll, S_lu, S_ul, S_uu, S_lg, S_ug, -flip,
                    )  # fmt: skip
                    dez_m = stencil(
                        mode.Ez, iz_lower, iz_upper, ir_lower, ir_upper,
                        dS_ll, dS_lu, dS_ul, dS_uu, dS_lg, dS_ug, flip,
                    ) + stencil_k(
                        tmode.Ez, k, iz_lower, iz_upper, ir_lower, ir_upper,
                        S_ll, S_lu, S_ul, S_uu, S_lg, S_ug, flip,
                    )  # fmt: skip
                    dbr_m = stencil(
                        mode.Br, iz_lower, iz_upper, ir_lower, ir_upper,
                        dS_ll, dS_lu, dS_ul, dS_uu, dS_lg, dS_ug, -flip,
                    ) + stencil_k(
                        tmode.Br, k, iz_lower, iz_upper, ir_lower, ir_upper,
                        S_ll, S_lu, S_ul, S_uu, S_lg, S_ug, -flip,
                    )  # fmt: skip
                    dbt_m = stencil(
                        mode.Bt, iz_lower, iz_upper, ir_lower, ir_upper,
                        dS_ll, dS_lu, dS_ul, dS_uu, dS_lg, dS_ug, -flip,
                    ) + stencil_k(
                        tmode.Bt, k, iz_lower, iz_upper, ir_lower, ir_upper,
                        S_ll, S_lu, S_ul, S_uu, S_lg, S_ug, -flip,
                    )  # fmt: skip
                    dbz_m = stencil(
                        mode.Bz, iz_lower, iz_upper, ir_lower, ir_upper,
                        dS_ll, dS_lu, dS_ul, dS_uu, dS_lg, dS_ug, flip,
                    ) + stencil_k(
                        tmode.Bz, k, iz_lower, iz_upper, ir_lower, ir_upper,
                        S_ll, S_lu, S_ul, S_uu, S_lg, S_ug, flip,
                    )  # fmt: skip
                    dEr = factor * (cmul(der_m, phase)[0] + cmul(er_m, dphase)[0])
                    dEt = factor * (cmul(det_m, phase)[0] + cmul(et_m, dphase)[0])
                    dEzm = factor * (cmul(dez_m, phase)[0] + cmul(ez_m, dphase)[0])
                    dBr = factor * (cmul(dbr_m, phase)[0] + cmul(br_m, dphase)[0])
                    dBt = factor * (cmul(dbt_m, phase)[0] + cmul(bt_m, dphase)[0])
                    dBzm = factor * (cmul(dbz_m, phase)[0] + cmul(bz_m, dphase)[0])
                    # ex += cos Fr - sin Ft; ey += sin Fr + cos Ft
                    dex[k] += dcos[k] * Er + cos * dEr - dsin[k] * Et - sin * dEt
                    dey[k] += dsin[k] * Er + sin * dEr + dcos[k] * Et + cos * dEt
                    dez[k] += dEzm
                    dbx[k] += dcos[k] * Br + cos * dBr - dsin[k] * Bt - sin * dBt
                    dby[k] += dsin[k] * Br + sin * dBr + dcos[k] * Bt + cos * dBt
                    dbz[k] += dBzm
                    # d(phase * step) = dphase * step + phase * dstep
                    dstep = wp.vec2d(dcos[k], -dsin[k])
                    next_dphase = cmul(dphase, phase_step) + cmul(phase, dstep)
                    phase_re[k] = next_dphase[0]
                    phase_im[k] = next_dphase[1]
                phase = cmul(phase, phase_step)
        for k in range(wp.static(K)):
            dEx[k, i] = dex[k]
            dEy[k, i] = dey[k]
            dEz[k, i] = dez[k]
            dBx[k, i] = dbx[k]
            dBy[k, i] = dby[k]
            dBz[k, i] = dbz[k]

    gather_tangent.key = f"gather_tangent_n{mode_count}_K{K}"
    return gather_tangent


def gather_tangent(
    x,
    y,
    z,
    tangents_xyz,
    grids,
    field_tangents,
    rmax_gather: float,
    outputs,
    block_dim: int = 256,
):
    """Write the tangents of the gathered E and B at the particles.

    Args:
        x, y, z: Primal positions (CuPy float64, length N), in meters.
        tangents_xyz: Position tangents (dx, dy, dz), each (K, N).
        grids: Primal interpolation grids of every mode (at most 5).
        field_tangents: One mapping per mode from Er..Bz to (K, Nz, Nr)
            complex128 tangent arrays.
        rmax_gather: Particles at this radius or beyond gather zero.
        outputs: (dEx, dEy, dEz, dBx, dBy, dBz), each (K, N); overwritten.
    """
    n = int(x.shape[0])
    K = int(tangents_xyz[0].shape[0])
    grid0 = grids[0]
    kernel = make_gather_tangent(len(grids), K)
    _launch(
        kernel,
        n,
        [x, y, z, *tangents_xyz]
        + [
            float(rmax_gather),
            float(grid0.invdz),
            float(grid0.zmin),
            int(grid0.Nz),
            float(grid0.invdr),
            float(grid0.rmin),
            int(grid0.Nr),
            group_fields(grids),
            group_field_tangents(field_tangents),
        ]
        + list(outputs),
        block_dim,
    )


# Momentum push
# -------------


@wp.func
def vay_jvp(
    ux: wp.float64,
    uy: wp.float64,
    uz: wp.float64,
    inv_gamma: wp.float64,
    Ex: wp.float64,
    Ey: wp.float64,
    Ez: wp.float64,
    Bx: wp.float64,
    By: wp.float64,
    Bz: wp.float64,
    dux: wp.float64,
    duy: wp.float64,
    duz: wp.float64,
    dinv_gamma: wp.float64,
    dEx: wp.float64,
    dEy: wp.float64,
    dEz: wp.float64,
    dBx: wp.float64,
    dBy: wp.float64,
    dBz: wp.float64,
    econst: wp.float64,
    bconst: wp.float64,
):
    """Tangent of `push_p_vay`, line by line; returns (dux, duy, duz, dig)."""
    taux = bconst * Bx
    tauy = bconst * By
    tauz = bconst * Bz
    dtaux = bconst * dBx
    dtauy = bconst * dBy
    dtauz = bconst * dBz
    tau2 = taux * taux + tauy * tauy + tauz * tauz
    dtau2 = wp.float64(2.0) * (taux * dtaux + tauy * dtauy + tauz * dtauz)

    uxp = ux + econst * Ex + inv_gamma * (uy * tauz - uz * tauy)
    uyp = uy + econst * Ey + inv_gamma * (uz * taux - ux * tauz)
    uzp = uz + econst * Ez + inv_gamma * (ux * tauy - uy * taux)
    duxp = (
        dux
        + econst * dEx
        + dinv_gamma * (uy * tauz - uz * tauy)
        + inv_gamma * (duy * tauz + uy * dtauz - duz * tauy - uz * dtauy)
    )
    duyp = (
        duy
        + econst * dEy
        + dinv_gamma * (uz * taux - ux * tauz)
        + inv_gamma * (duz * taux + uz * dtaux - dux * tauz - ux * dtauz)
    )
    duzp = (
        duz
        + econst * dEz
        + dinv_gamma * (ux * tauy - uy * taux)
        + inv_gamma * (dux * tauy + ux * dtauy - duy * taux - uy * dtaux)
    )
    sigma = wp.float64(1) + uxp * uxp + uyp * uyp + uzp * uzp - tau2
    dsigma = wp.float64(2.0) * (uxp * duxp + uyp * duyp + uzp * duzp) - dtau2
    utau = uxp * taux + uyp * tauy + uzp * tauz
    dutau = (
        duxp * taux
        + uxp * dtaux
        + duyp * tauy
        + uyp * dtauy
        + duzp * tauz
        + uzp * dtauz
    )

    root = wp.sqrt(sigma * sigma + wp.float64(4) * (tau2 + utau * utau))
    droot = (
        sigma * dsigma + wp.float64(2.0) * (dtau2 + wp.float64(2.0) * utau * dutau)
    ) / root
    denominator = sigma + root
    inv_gamma_f = wp.sqrt(wp.float64(2.0) / denominator)
    # d sqrt(2/A) = -sqrt(2/A) dA / (2A)
    dinv_gamma_f = -inv_gamma_f * (dsigma + droot) / (wp.float64(2.0) * denominator)

    tx = inv_gamma_f * taux
    ty = inv_gamma_f * tauy
    tz = inv_gamma_f * tauz
    ut = inv_gamma_f * utau
    dtx = dinv_gamma_f * taux + inv_gamma_f * dtaux
    dty = dinv_gamma_f * tauy + inv_gamma_f * dtauy
    dtz = dinv_gamma_f * tauz + inv_gamma_f * dtauz
    dut = dinv_gamma_f * utau + inv_gamma_f * dutau
    s = wp.float64(1.0) / (wp.float64(1) + tau2 * inv_gamma_f * inv_gamma_f)
    ds = (
        -s
        * s
        * (
            dtau2 * inv_gamma_f * inv_gamma_f
            + wp.float64(2.0) * tau2 * inv_gamma_f * dinv_gamma_f
        )
    )

    ax = uxp + tx * ut + uyp * tz - uzp * ty
    ay = uyp + ty * ut + uzp * tx - uxp * tz
    az = uzp + tz * ut + uxp * ty - uyp * tx
    dax = duxp + dtx * ut + tx * dut + duyp * tz + uyp * dtz - duzp * ty - uzp * dty
    day = duyp + dty * ut + ty * dut + duzp * tx + uzp * dtx - duxp * tz - uxp * dtz
    daz = duzp + dtz * ut + tz * dut + duxp * ty + uxp * dty - duyp * tx - uyp * dtx
    return (ds * ax + s * dax, ds * ay + s * day, ds * az + s * daz, dinv_gamma_f)


@wp.kernel(enable_backward=False, module="unique")
def vay_tangent_kernel(
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    inv_gamma: wp.array(dtype=wp.float64),
    Ex: wp.array(dtype=wp.float64),
    Ey: wp.array(dtype=wp.float64),
    Ez: wp.array(dtype=wp.float64),
    Bx: wp.array(dtype=wp.float64),
    By: wp.array(dtype=wp.float64),
    Bz: wp.array(dtype=wp.float64),
    q: wp.float64,
    m: wp.float64,
    dt: wp.float64,
    dux: wp.array2d(dtype=wp.float64),
    duy: wp.array2d(dtype=wp.float64),
    duz: wp.array2d(dtype=wp.float64),
    dinv_gamma: wp.array2d(dtype=wp.float64),
    dEx: wp.array2d(dtype=wp.float64),
    dEy: wp.array2d(dtype=wp.float64),
    dEz: wp.array2d(dtype=wp.float64),
    dBx: wp.array2d(dtype=wp.float64),
    dBy: wp.array2d(dtype=wp.float64),
    dBz: wp.array2d(dtype=wp.float64),
):
    """Update the momentum and inverse-gamma tangents in place.

    ux to inv_gamma are the momenta before the push (the stash). The
    coefficients match `push_p_gpu`.
    """
    i = wp.tid()
    if i >= ux.shape[0]:
        return
    econst = q * dt / (m * LIGHT_SPEED)
    bconst = wp.float64(0.5) * q * dt / m
    for k in range(dux.shape[0]):
        a, b, cc, d = vay_jvp(
            ux[i], uy[i], uz[i], inv_gamma[i],
            Ex[i], Ey[i], Ez[i], Bx[i], By[i], Bz[i],
            dux[k, i], duy[k, i], duz[k, i], dinv_gamma[k, i],
            dEx[k, i], dEy[k, i], dEz[k, i], dBx[k, i], dBy[k, i], dBz[k, i],
            econst, bconst,
        )  # fmt: skip
        dux[k, i] = a
        duy[k, i] = b
        duz[k, i] = cc
        dinv_gamma[k, i] = d


def vay_tangent(stash, gathered, q, m, dt, tangents, field_tangents):
    """Apply the Vay push tangent to K momentum tangents in place.

    Args:
        stash: (ux, uy, uz, inv_gamma) before the push, each length N.
        gathered: (Ex, Ey, Ez, Bx, By, Bz) primal gathered fields.
        q, m, dt: Species charge (C), mass (kg), and time step (s).
        tangents: (dux, duy, duz, dinv_gamma), each (K, N); updated in place.
        field_tangents: (dEx, ..., dBz), each (K, N).
    """
    n = int(stash[0].shape[0])
    _launch(
        vay_tangent_kernel,
        n,
        [*stash, *gathered, float(q), float(m), float(dt), *tangents, *field_tangents],
    )


# Position push
# -------------


@wp.kernel(enable_backward=False, module="unique")
def position_tangent_kernel(
    ux: wp.array(dtype=wp.float64),
    uy: wp.array(dtype=wp.float64),
    uz: wp.array(dtype=wp.float64),
    inv_gamma: wp.array(dtype=wp.float64),
    dux: wp.array2d(dtype=wp.float64),
    duy: wp.array2d(dtype=wp.float64),
    duz: wp.array2d(dtype=wp.float64),
    dinv_gamma: wp.array2d(dtype=wp.float64),
    dt: wp.float64,
    x_push: wp.float64,
    y_push: wp.float64,
    z_push: wp.float64,
    dx: wp.array2d(dtype=wp.float64),
    dy: wp.array2d(dtype=wp.float64),
    dz: wp.array2d(dtype=wp.float64),
):
    """Tangent of `push_x_gpu`: dx += c dt x_push (dux g + ux dg), in place."""
    i = wp.tid()
    if i >= ux.shape[0]:
        return
    cdt = LIGHT_SPEED * dt
    g = inv_gamma[i]
    for k in range(dx.shape[0]):
        dg = dinv_gamma[k, i]
        dx[k, i] += cdt * x_push * (dg * ux[i] + g * dux[k, i])
        dy[k, i] += cdt * y_push * (dg * uy[i] + g * duy[k, i])
        dz[k, i] += cdt * z_push * (dg * uz[i] + g * duz[k, i])


def position_tangent(momenta, momentum_tangents, dt, position_tangents, push=(1, 1, 1)):
    """Apply the position push tangent to K position tangents in place.

    Args:
        momenta: (ux, uy, uz, inv_gamma) used by the primal push, length N.
        momentum_tangents: (dux, duy, duz, dinv_gamma), each (K, N).
        dt: Push interval (s).
        position_tangents: (dx, dy, dz), each (K, N); updated in place.
        push: (x_push, y_push, z_push) of `push_x`.
    """
    n = int(momenta[0].shape[0])
    _launch(
        position_tangent_kernel,
        n,
        [
            *momenta,
            *momentum_tangents,
            float(dt),
            *map(float, push),
            *position_tangents,
        ],
    )


# Deposition
# ----------


@wp.func
def Sr_linear_slope(
    cell_position: wp.float64, index: wp.int32, flip: wp.float64, beta_n: wp.float64
):
    """d Sr_linear / d cell_position, with the cell and the fold frozen."""
    ir = wp.int32(wp.ceil(cell_position)) - 1
    u = cell_position - wp.float64(ir)
    ds = wp.float64(-1.0) + beta_n * (wp.float64(1.0) - wp.float64(2.0) * u)
    if index == 1:
        ds = -ds
    if index + ir < 0:
        ds *= flip
    return ds


@cache
def make_deposit_tangent(current: bool, mode_count: int, K: int):
    """Return the tangent of the fused linear deposition (one cell per thread).

    The primal is `deposition/cuda_methods_fused.py::make_kernel`. Each
    thread walks the particles of one sorted cell in reverse order, adds the
    tangent contributions of all modes, components, and tangents into a
    local matrix, and adds the matrix to the K tangent grids with atomics.
    Differentiated terms: r, cos, sin, the phase exp(+i m phi), v_r and
    v_theta (with their rotation terms), z, the Ruyten-corrected radial
    shape, the axial shape, the weight, and inv_gamma (stored tangent).
    """
    if not 1 <= mode_count <= MAX_MODES or K < 1:
        raise ValueError("The deposit tangent needs 1-5 modes and K >= 1.")
    field_count = 3 if current else 1
    component_count = mode_count * field_count
    values_type = wp.types.matrix(shape=(4 * component_count * K, 2), dtype=wp.float64)
    vec_k = wp.types.vector(length=K, dtype=wp.float64)

    @wp.kernel(enable_backward=False, module="unique")
    def deposit_tangent(
        x: wp.array(dtype=wp.float64),
        y: wp.array(dtype=wp.float64),
        z: wp.array(dtype=wp.float64),
        w: wp.array(dtype=wp.float64),
        q: wp.float64,
        ux: wp.array(dtype=wp.float64),
        uy: wp.array(dtype=wp.float64),
        uz: wp.array(dtype=wp.float64),
        inv_gamma: wp.array(dtype=wp.float64),
        dx: wp.array2d(dtype=wp.float64),
        dy: wp.array2d(dtype=wp.float64),
        dz: wp.array2d(dtype=wp.float64),
        dw: wp.array2d(dtype=wp.float64),
        dux: wp.array2d(dtype=wp.float64),
        duy: wp.array2d(dtype=wp.float64),
        duz: wp.array2d(dtype=wp.float64),
        dinv_gamma: wp.array2d(dtype=wp.float64),
        invdz: wp.float64,
        zmin: wp.float64,
        nz: int,
        invdr: wp.float64,
        rmin: wp.float64,
        nr: int,
        outputs: OutputTangents,
        prefix: wp.array(dtype=wp.int32),
        betas: ModeBetas,
    ):
        cell = wp.tid()
        if cell >= prefix.shape[0]:
            return
        end = prefix[cell]
        start = 0
        if cell > 0:
            start = prefix[cell - 1]
        if end <= start:
            return
        values = values_type()
        for offset in range(end - start):
            p = end - 1 - offset
            px = x[p]
            py = y[p]
            radius = wp.sqrt(px * px + py * py)
            cosine = wp.float64(1.0)
            sine = wp.float64(0.0)
            dcos = vec_k()
            dsin = vec_k()
            drad = vec_k()
            if radius != wp.float64(0.0):
                inverse_radius = wp.float64(1.0) / radius
                cosine = px * inverse_radius
                sine = py * inverse_radius
                for k in range(wp.static(K)):
                    dr_k = (px * dx[k, p] + py * dy[k, p]) * inverse_radius
                    dinv = -dr_k * inverse_radius * inverse_radius
                    dcos[k] = dx[k, p] * inverse_radius + px * dinv
                    dsin[k] = dy[k, p] * inverse_radius + py * dinv
                    drad[k] = dr_k
            radial_position = invdr * (radius - rmin) - wp.float64(0.5)
            axial_position = invdz * (z[p] - zmin) - wp.float64(0.5)
            beta_index = min(int(wp.ceil(radial_position)), nr)
            weight = q * w[p]
            axial0 = Sz_linear(axial_position, 0)
            axial1 = Sz_linear(axial_position, 1)
            velocity_r = wp.float64(0.0)
            velocity_t = wp.float64(0.0)
            velocity_z = wp.float64(0.0)
            current_weight = wp.float64(0.0)
            if wp.static(current):
                velocity_r = cosine * ux[p] + sine * uy[p]
                velocity_t = cosine * uy[p] - sine * ux[p]
                velocity_z = uz[p]
                current_weight = weight * LIGHT_SPEED * inv_gamma[p]
            phase_step = wp.vec2d(cosine, sine)
            phase = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
            phase_re = vec_k()
            phase_im = vec_k()
            for local_mode in range(wp.static(mode_count)):
                coefficient = beta_of(betas, local_mode, beta_index)
                parity = wp.float64(1 - 2 * (local_mode % 2))
                for field in range(wp.static(field_count)):
                    flip = parity
                    velocity = velocity_z
                    if wp.static(current):
                        if field == 0:
                            velocity = velocity_r
                        elif field == 1:
                            velocity = velocity_t
                        if field < 2:
                            flip = -flip
                    radial0 = Sr_linear(radial_position, 0, flip, coefficient)
                    radial1 = Sr_linear(radial_position, 1, flip, coefficient)
                    slope0 = Sr_linear_slope(radial_position, 0, flip, coefficient)
                    slope1 = Sr_linear_slope(radial_position, 1, flip, coefficient)
                    component = field * wp.static(mode_count) + local_mode
                    for k in range(wp.static(K)):
                        dphase = wp.vec2d(phase_re[k], phase_im[k])
                        dscale = wp.vec2d()
                        scale = wp.vec2d()
                        if wp.static(current):
                            # d(cw v phase) with cw = q w c g
                            dcw = (
                                q * LIGHT_SPEED * (dw[k, p] * inv_gamma[p] + w[p] * dinv_gamma[k, p])
                            )  # fmt: skip
                            dvel = duz[k, p]
                            if field == 0:
                                dvel = (
                                    dcos[k] * ux[p] + cosine * dux[k, p]
                                    + dsin[k] * uy[p] + sine * duy[k, p]
                                )  # fmt: skip
                            elif field == 1:
                                dvel = (
                                    dcos[k] * uy[p] + cosine * duy[k, p]
                                    - dsin[k] * ux[p] - sine * dux[k, p]
                                )  # fmt: skip
                            scale = current_weight * velocity * phase
                            dscale = (
                                (dcw * velocity + current_weight * dvel) * phase
                                + current_weight * velocity * dphase
                            )  # fmt: skip
                        else:
                            scale = weight * phase
                            dscale = q * dw[k, p] * phase + weight * dphase
                        drp = invdr * drad[k]
                        dzp = invdz * dz[k, p]
                        dr0 = slope0 * drp
                        dr1 = slope1 * drp
                        # d(Sr Sz scale) = (dSr Sz + Sr dSz) scale + Sr Sz dscale
                        base = 4 * (component * wp.static(K) + k)
                        values[base] += (
                            (dr0 * axial0 - radial0 * dzp) * scale
                            + radial0 * axial0 * dscale
                        )  # fmt: skip
                        values[base + 1] += (
                            (dr1 * axial0 - radial1 * dzp) * scale
                            + radial1 * axial0 * dscale
                        )  # fmt: skip
                        values[base + 2] += (
                            (dr0 * axial1 + radial0 * dzp) * scale
                            + radial0 * axial1 * dscale
                        )  # fmt: skip
                        values[base + 3] += (
                            (dr1 * axial1 + radial1 * dzp) * scale
                            + radial1 * axial1 * dscale
                        )  # fmt: skip
                for k in range(wp.static(K)):
                    # phase_{m+1} = phase_m * step (exp(+i m phi) in deposition)
                    dphase = wp.vec2d(phase_re[k], phase_im[k])
                    dstep = wp.vec2d(dcos[k], dsin[k])
                    next_dphase = cmul(dphase, phase_step) + cmul(phase, dstep)
                    phase_re[k] = next_dphase[0]
                    phase_im[k] = next_dphase[1]
                phase = cmul(phase, phase_step)
        upper_z = cell // (nr + 1)
        upper_r = cell % (nr + 1)
        for component in range(wp.static(component_count)):
            for k in range(wp.static(K)):
                base = 4 * (component * wp.static(K) + k)
                for dzi in range(2):
                    physical_z = upper_z - 1 + dzi
                    if physical_z < 0:
                        physical_z += nz
                    for dri in range(2):
                        physical_r = min(upper_r - 1 + dri, nr - 1)
                        if physical_r < 0:
                            physical_r = -(1 + physical_r)
                        atomic_tangent(
                            outputs,
                            component,
                            k,
                            physical_z,
                            physical_r,
                            wp.vec2d(
                                values[base + 2 * dzi + dri, 0],
                                values[base + 2 * dzi + dri, 1],
                            ),
                        )

    deposit_tangent.key = (
        f"deposit_tangent_{'J' if current else 'rho'}_n{mode_count}_K{K}"
    )
    return deposit_tangent


def deposit_tangent(
    primal, tangents, q, grids, prefix, outputs, current: bool, block_dim: int = 64
):
    """Add the tangents of a sorted linear deposit into (K, Nz, Nr) grids.

    Args:
        primal: (x, y, z, w, ux, uy, uz, inv_gamma) of the sorted particles,
            each length N. w is the effective weight (w_times_level for EV).
        tangents: (dx, dy, dz, dw, dux, duy, duz, dinv_gamma), each (K, N),
            in the same sorted order.
        q: Species charge (C).
        grids: Primal interpolation grids of every mode (at most 5); give
            the geometry and the Ruyten coefficients.
        prefix: Inclusive prefix sum of the sorted cell keys (int32).
        outputs: Tangent grids (K, Nz, Nr) in field-major, mode-minor order:
            rho for each mode, or Jr, Jt, Jz for each mode. Added to.
        current: Deposit J instead of rho.
    """
    K = int(tangents[0].shape[0])
    grid0 = grids[0]
    kernel = make_deposit_tangent(current, len(grids), K)
    _launch(
        kernel,
        int(prefix.shape[0]),
        [*primal[:4], float(q), *primal[4:], *tangents]
        + [
            float(grid0.invdz),
            float(grid0.zmin),
            int(grid0.Nz),
            float(grid0.invdr),
            float(grid0.rmin),
            int(grid0.Nr),
            output_tangents(outputs),
            prefix,
            mode_betas([g.d_ruyten_linear_coef for g in grids]),
        ],
        block_dim,
    )
