# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Differentiate FP64 field payloads with fixed grid coefficients.

These eager Warp kernels preserve their input buffers. Callers supply distinct
gradient-enabled output arrays and retain every input until backward completes.
Complex128 fields use ``wp.vec2d`` elements with real and imaginary components.
Grid coefficients, indices, timesteps, and event choices remain fixed.
"""

import warp as wp
from scipy.constants import c, epsilon_0, mu_0

from fbpic.fields.spectral_transform import cuda_methods as _layout
from fbpic.utils.warp import cmul, mul_i

_C2 = wp.constant(wp.float64(c**2))
_EPSILON0 = wp.constant(wp.float64(epsilon_0))
_MU0 = wp.constant(wp.float64(mu_0))

# These production formulas already have separate inputs and outputs. Unique
# modules keep generated backward code independent of forward-only wrappers.
copy_2dc_to_2dr = wp.kernel(
    _layout.cuda_copy_2dC_to_2dR.kernel.func, enable_backward=True, module="unique"
)
copy_2dr_to_2dc = wp.kernel(
    _layout.cuda_copy_2dR_to_2dC.kernel.func, enable_backward=True, module="unique"
)
copy_2d_to_1d = wp.kernel(
    _layout.cuda_copy_2d_to_1d.kernel.func, enable_backward=True, module="unique"
)
copy_1d_to_2d = wp.kernel(
    _layout.cuda_copy_1d_to_2d.kernel.func, enable_backward=True, module="unique"
)
rt_to_pm = wp.kernel(
    _layout.cuda_rt_to_pm.kernel.func, enable_backward=True, module="unique"
)
pm_to_rt = wp.kernel(
    _layout.cuda_pm_to_rt.kernel.func, enable_backward=True, module="unique"
)


def copy_1d_2d_dim(shape):
    """Return the 1D launch dimension of `copy_2d_to_1d` and `copy_1d_to_2d`.

    These clones keep the production warp micro-tiles, so each thread maps its
    linear index to one (iz, ir) cell. A 2D launch over `shape` copies only the
    ir = 0 column.

    Args:
        shape: The (Nz, Nr) shape of the 2D array.

    Returns:
        The thread count to pass as ``dim`` to ``wp.launch``.
    """
    blocks, threads = _layout.cuda_copy_1d_2d_launch(*shape)
    return blocks * threads


@wp.kernel(enable_backward=True, module="unique")
def permute_float64(
    source: wp.array(dtype=wp.float64),
    permutation: wp.array(dtype=wp.int32),
    result: wp.array(dtype=wp.float64),
):
    """Copy scalar payloads through a fixed, valid particle permutation."""
    index = wp.tid()
    if index < result.shape[0]:
        result[index] = source[permutation[index]]


@wp.kernel(enable_backward=True, module="unique")
def permute_vec2d(
    source: wp.array(dtype=wp.vec2d),
    permutation: wp.array(dtype=wp.int32),
    result: wp.array(dtype=wp.vec2d),
):
    """Copy complex payloads through a fixed, valid particle permutation."""
    index = wp.tid()
    if index < result.shape[0]:
        result[index] = source[permutation[index]]


@wp.kernel(enable_backward=True, module="unique")
def erase_scalar(
    field: wp.array2d(dtype=wp.vec2d),
    result: wp.array2d(dtype=wp.vec2d),
):
    """Reset a separate complex field buffer with zero derivative."""
    iz, ir = wp.tid()
    if iz < field.shape[0] and ir < field.shape[1]:
        result[iz, ir] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))


@wp.kernel(enable_backward=True, module="unique")
def erase_vector(
    field_r: wp.array2d(dtype=wp.vec2d),
    field_t: wp.array2d(dtype=wp.vec2d),
    field_z: wp.array2d(dtype=wp.vec2d),
    result_r: wp.array2d(dtype=wp.vec2d),
    result_t: wp.array2d(dtype=wp.vec2d),
    result_z: wp.array2d(dtype=wp.vec2d),
):
    """Reset three separate complex field buffers with zero derivatives."""
    iz, ir = wp.tid()
    if iz < field_r.shape[0] and ir < field_r.shape[1]:
        result_r[iz, ir] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        result_t[iz, ir] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        result_z[iz, ir] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))


@wp.kernel(enable_backward=True, module="unique")
def divide_scalar_by_volume(
    field: wp.array2d(dtype=wp.vec2d),
    inverse_volume: wp.array(dtype=wp.float64),
    result: wp.array2d(dtype=wp.vec2d),
):
    """Normalize a complex field with fixed radial inverse cell volumes."""
    iz, ir = wp.tid()
    if iz < field.shape[0] and ir < field.shape[1]:
        result[iz, ir] = field[iz, ir] * inverse_volume[ir]


@wp.kernel(enable_backward=True, module="unique")
def divide_vector_by_volume(
    field_r: wp.array2d(dtype=wp.vec2d),
    field_t: wp.array2d(dtype=wp.vec2d),
    field_z: wp.array2d(dtype=wp.vec2d),
    inverse_volume: wp.array(dtype=wp.float64),
    result_r: wp.array2d(dtype=wp.vec2d),
    result_t: wp.array2d(dtype=wp.vec2d),
    result_z: wp.array2d(dtype=wp.vec2d),
):
    """Normalize three complex components without overwriting their inputs."""
    iz, ir = wp.tid()
    if iz < field_r.shape[0] and ir < field_r.shape[1]:
        result_r[iz, ir] = field_r[iz, ir] * inverse_volume[ir]
        result_t[iz, ir] = field_t[iz, ir] * inverse_volume[ir]
        result_z[iz, ir] = field_z[iz, ir] * inverse_volume[ir]


@wp.kernel(enable_backward=True, module="unique")
def filter_scalar(
    field: wp.array2d(dtype=wp.vec2d),
    nz: int,
    nr: int,
    filter_z: wp.array(dtype=wp.float64),
    filter_r: wp.array(dtype=wp.float64),
    result: wp.array2d(dtype=wp.vec2d),
):
    """Apply fixed axial and radial spectral filter coefficients."""
    iz, ir = wp.tid()
    if iz < nz and ir < nr:
        result[iz, ir] = filter_z[iz] * filter_r[ir] * field[iz, ir]


@wp.kernel(enable_backward=True, module="unique")
def filter_vector(
    field_r: wp.array2d(dtype=wp.vec2d),
    field_t: wp.array2d(dtype=wp.vec2d),
    field_z: wp.array2d(dtype=wp.vec2d),
    nz: int,
    nr: int,
    filter_z: wp.array(dtype=wp.float64),
    filter_r: wp.array(dtype=wp.float64),
    result_r: wp.array2d(dtype=wp.vec2d),
    result_t: wp.array2d(dtype=wp.vec2d),
    result_z: wp.array2d(dtype=wp.vec2d),
):
    """Apply fixed spectral filters to three complex field components."""
    iz, ir = wp.tid()
    if iz < nz and ir < nr:
        result_r[iz, ir] = filter_z[iz] * filter_r[ir] * field_r[iz, ir]
        result_t[iz, ir] = filter_z[iz] * filter_r[ir] * field_t[iz, ir]
        result_z[iz, ir] = filter_z[iz] * filter_r[ir] * field_z[iz, ir]


@wp.kernel(enable_backward=True, module="unique")
def push_rho(
    rho_prev: wp.array2d(dtype=wp.vec2d),
    rho_next: wp.array2d(dtype=wp.vec2d),
    nz: int,
    nr: int,
    result_prev: wp.array2d(dtype=wp.vec2d),
    result_next: wp.array2d(dtype=wp.vec2d),
):
    """Copy the next charge density and reset its separate output buffer."""
    iz, ir = wp.tid()
    if iz < nz and ir < nr:
        result_prev[iz, ir] = rho_next[iz, ir]
        result_next[iz, ir] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))


@wp.kernel(enable_backward=True, module="unique")
def correct_currents_curlfree_standard(
    rho_prev: wp.array2d(dtype=wp.vec2d),
    rho_next: wp.array2d(dtype=wp.vec2d),
    jp: wp.array2d(dtype=wp.vec2d),
    jm: wp.array2d(dtype=wp.vec2d),
    jz: wp.array2d(dtype=wp.vec2d),
    kz: wp.array2d(dtype=wp.float64),
    kr: wp.array2d(dtype=wp.float64),
    inverse_k2: wp.array2d(dtype=wp.float64),
    inverse_dt: wp.float64,
    nz: int,
    nr: int,
    result_p: wp.array2d(dtype=wp.vec2d),
    result_m: wp.array2d(dtype=wp.vec2d),
    result_z: wp.array2d(dtype=wp.vec2d),
):
    """Correct continuous currents with the standard curl-free projection."""
    iz, ir = wp.tid()
    if iz < nz and ir < nr:
        correction = -inverse_k2[iz, ir] * (
            (rho_next[iz, ir] - rho_prev[iz, ir]) * inverse_dt
            + kz[iz, ir] * mul_i(jz[iz, ir])
            + kr[iz, ir] * (jp[iz, ir] - jm[iz, ir])
        )
        result_p[iz, ir] = jp[iz, ir] + wp.float64(0.5) * kr[iz, ir] * correction
        result_m[iz, ir] = jm[iz, ir] - wp.float64(0.5) * kr[iz, ir] * correction
        result_z[iz, ir] = jz[iz, ir] - kz[iz, ir] * mul_i(correction)


@wp.kernel(enable_backward=True, module="unique")
def correct_currents_curlfree_comoving(
    rho_prev: wp.array2d(dtype=wp.vec2d),
    rho_next: wp.array2d(dtype=wp.vec2d),
    jp: wp.array2d(dtype=wp.vec2d),
    jm: wp.array2d(dtype=wp.vec2d),
    jz: wp.array2d(dtype=wp.vec2d),
    kz: wp.array2d(dtype=wp.float64),
    kr: wp.array2d(dtype=wp.float64),
    inverse_k2: wp.array2d(dtype=wp.float64),
    correction_coef: wp.array2d(dtype=wp.vec2d),
    t_eb: wp.array2d(dtype=wp.vec2d),
    t_cc: wp.array2d(dtype=wp.vec2d),
    inverse_dt: wp.float64,
    nz: int,
    nr: int,
    result_p: wp.array2d(dtype=wp.vec2d),
    result_m: wp.array2d(dtype=wp.vec2d),
    result_z: wp.array2d(dtype=wp.vec2d),
):
    """Correct continuous currents with fixed comoving PSATD coefficients."""
    iz, ir = wp.tid()
    if iz < nz and ir < nr:
        correction = -inverse_k2[iz, ir] * (
            cmul(
                cmul(t_cc[iz, ir], correction_coef[iz, ir]),
                rho_next[iz, ir] - cmul(rho_prev[iz, ir], t_eb[iz, ir]),
            )
            + kz[iz, ir] * mul_i(jz[iz, ir])
            + kr[iz, ir] * (jp[iz, ir] - jm[iz, ir])
        )
        result_p[iz, ir] = jp[iz, ir] + wp.float64(0.5) * kr[iz, ir] * correction
        result_m[iz, ir] = jm[iz, ir] - wp.float64(0.5) * kr[iz, ir] * correction
        result_z[iz, ir] = jz[iz, ir] - kz[iz, ir] * mul_i(correction)


@wp.kernel(enable_backward=True, module="unique")
def push_eb_standard(
    ep: wp.array2d(dtype=wp.vec2d),
    em: wp.array2d(dtype=wp.vec2d),
    ez: wp.array2d(dtype=wp.vec2d),
    bp: wp.array2d(dtype=wp.vec2d),
    bm: wp.array2d(dtype=wp.vec2d),
    bz: wp.array2d(dtype=wp.vec2d),
    jp: wp.array2d(dtype=wp.vec2d),
    jm: wp.array2d(dtype=wp.vec2d),
    jz: wp.array2d(dtype=wp.vec2d),
    rho_prev: wp.array2d(dtype=wp.vec2d),
    rho_next: wp.array2d(dtype=wp.vec2d),
    rho_prev_coef: wp.array2d(dtype=wp.float64),
    rho_next_coef: wp.array2d(dtype=wp.float64),
    j_coef: wp.array2d(dtype=wp.float64),
    cosine: wp.array2d(dtype=wp.float64),
    sine_over_omega: wp.array2d(dtype=wp.float64),
    kr: wp.array2d(dtype=wp.float64),
    kz: wp.array2d(dtype=wp.float64),
    dt: wp.float64,
    use_true_rho: bool,
    nz: int,
    nr: int,
    result_ep: wp.array2d(dtype=wp.vec2d),
    result_em: wp.array2d(dtype=wp.vec2d),
    result_ez: wp.array2d(dtype=wp.vec2d),
    result_bp: wp.array2d(dtype=wp.vec2d),
    result_bm: wp.array2d(dtype=wp.vec2d),
    result_bz: wp.array2d(dtype=wp.vec2d),
):
    """Advance E and B with fixed standard PSATD coefficients."""
    iz, ir = wp.tid()
    if iz < nz and ir < nr:
        if use_true_rho:
            rho_diff = (
                rho_next_coef[iz, ir] * rho_next[iz, ir]
                - rho_prev_coef[iz, ir] * rho_prev[iz, ir]
            )
        else:
            div_e = kr[iz, ir] * (ep[iz, ir] - em[iz, ir]) + kz[iz, ir] * mul_i(
                ez[iz, ir]
            )
            div_j = kr[iz, ir] * (jp[iz, ir] - jm[iz, ir]) + kz[iz, ir] * mul_i(
                jz[iz, ir]
            )
            rho_diff = (
                rho_next_coef[iz, ir] - rho_prev_coef[iz, ir]
            ) * _EPSILON0 * div_e - rho_next_coef[iz, ir] * dt * div_j
        result_ep[iz, ir] = (
            cosine[iz, ir] * ep[iz, ir]
            + wp.float64(0.5) * kr[iz, ir] * rho_diff
            + _C2
            * sine_over_omega[iz, ir]
            * (
                -wp.float64(0.5) * kr[iz, ir] * mul_i(bz[iz, ir])
                + kz[iz, ir] * bp[iz, ir]
                - _MU0 * jp[iz, ir]
            )
        )
        result_em[iz, ir] = (
            cosine[iz, ir] * em[iz, ir]
            - wp.float64(0.5) * kr[iz, ir] * rho_diff
            + _C2
            * sine_over_omega[iz, ir]
            * (
                -wp.float64(0.5) * kr[iz, ir] * mul_i(bz[iz, ir])
                - kz[iz, ir] * bm[iz, ir]
                - _MU0 * jm[iz, ir]
            )
        )
        result_ez[iz, ir] = (
            cosine[iz, ir] * ez[iz, ir]
            - kz[iz, ir] * mul_i(rho_diff)
            + _C2
            * sine_over_omega[iz, ir]
            * (
                kr[iz, ir] * mul_i(bp[iz, ir])
                + kr[iz, ir] * mul_i(bm[iz, ir])
                - _MU0 * jz[iz, ir]
            )
        )
        result_bp[iz, ir] = (
            cosine[iz, ir] * bp[iz, ir]
            - sine_over_omega[iz, ir]
            * (
                -wp.float64(0.5) * kr[iz, ir] * mul_i(ez[iz, ir])
                + kz[iz, ir] * ep[iz, ir]
            )
            + j_coef[iz, ir]
            * (
                -wp.float64(0.5) * kr[iz, ir] * mul_i(jz[iz, ir])
                + kz[iz, ir] * jp[iz, ir]
            )
        )
        result_bm[iz, ir] = (
            cosine[iz, ir] * bm[iz, ir]
            - sine_over_omega[iz, ir]
            * (
                -wp.float64(0.5) * kr[iz, ir] * mul_i(ez[iz, ir])
                - kz[iz, ir] * em[iz, ir]
            )
            + j_coef[iz, ir]
            * (
                -wp.float64(0.5) * kr[iz, ir] * mul_i(jz[iz, ir])
                - kz[iz, ir] * jm[iz, ir]
            )
        )
        result_bz[iz, ir] = (
            cosine[iz, ir] * bz[iz, ir]
            - sine_over_omega[iz, ir]
            * (kr[iz, ir] * mul_i(ep[iz, ir]) + kr[iz, ir] * mul_i(em[iz, ir]))
            + j_coef[iz, ir]
            * (kr[iz, ir] * mul_i(jp[iz, ir]) + kr[iz, ir] * mul_i(jm[iz, ir]))
        )


@wp.kernel(enable_backward=True, module="unique")
def push_eb_comoving(
    ep: wp.array2d(dtype=wp.vec2d),
    em: wp.array2d(dtype=wp.vec2d),
    ez: wp.array2d(dtype=wp.vec2d),
    bp: wp.array2d(dtype=wp.vec2d),
    bm: wp.array2d(dtype=wp.vec2d),
    bz: wp.array2d(dtype=wp.vec2d),
    jp: wp.array2d(dtype=wp.vec2d),
    jm: wp.array2d(dtype=wp.vec2d),
    jz: wp.array2d(dtype=wp.vec2d),
    rho_prev: wp.array2d(dtype=wp.vec2d),
    rho_next: wp.array2d(dtype=wp.vec2d),
    rho_prev_coef: wp.array2d(dtype=wp.vec2d),
    rho_next_coef: wp.array2d(dtype=wp.vec2d),
    j_coef: wp.array2d(dtype=wp.vec2d),
    cosine: wp.array2d(dtype=wp.float64),
    sine_over_omega: wp.array2d(dtype=wp.float64),
    t_eb: wp.array2d(dtype=wp.vec2d),
    t_cc: wp.array2d(dtype=wp.vec2d),
    t_rho: wp.array2d(dtype=wp.vec2d),
    kr: wp.array2d(dtype=wp.float64),
    kz: wp.array2d(dtype=wp.float64),
    dt: wp.float64,
    velocity: wp.float64,
    use_true_rho: bool,
    nz: int,
    nr: int,
    result_ep: wp.array2d(dtype=wp.vec2d),
    result_em: wp.array2d(dtype=wp.vec2d),
    result_ez: wp.array2d(dtype=wp.vec2d),
    result_bp: wp.array2d(dtype=wp.vec2d),
    result_bm: wp.array2d(dtype=wp.vec2d),
    result_bz: wp.array2d(dtype=wp.vec2d),
):
    """Advance E and B with fixed comoving or Galilean PSATD coefficients."""
    iz, ir = wp.tid()
    if iz < nz and ir < nr:
        if use_true_rho:
            rho_diff = cmul(rho_next_coef[iz, ir], rho_next[iz, ir]) - cmul(
                rho_prev_coef[iz, ir], rho_prev[iz, ir]
            )
        else:
            div_e = kr[iz, ir] * (ep[iz, ir] - em[iz, ir]) + kz[iz, ir] * mul_i(
                ez[iz, ir]
            )
            div_j = kr[iz, ir] * (jp[iz, ir] - jm[iz, ir]) + kz[iz, ir] * mul_i(
                jz[iz, ir]
            )
            rho_diff = cmul(
                (cmul(t_eb[iz, ir], rho_next_coef[iz, ir]) - rho_prev_coef[iz, ir])
                * _EPSILON0,
                div_e,
            ) + cmul(cmul(t_rho[iz, ir], rho_next_coef[iz, ir]), div_j)
        result_ep[iz, ir] = (
            cmul(t_eb[iz, ir] * cosine[iz, ir], ep[iz, ir])
            + wp.float64(0.5) * kr[iz, ir] * rho_diff
            + cmul(mul_i(j_coef[iz, ir]) * kz[iz, ir] * velocity, jp[iz, ir])
            + cmul(
                _C2 * t_eb[iz, ir] * sine_over_omega[iz, ir],
                -wp.float64(0.5) * kr[iz, ir] * mul_i(bz[iz, ir])
                + kz[iz, ir] * bp[iz, ir]
                - cmul(_MU0 * t_cc[iz, ir], jp[iz, ir]),
            )
        )
        result_em[iz, ir] = (
            cmul(t_eb[iz, ir] * cosine[iz, ir], em[iz, ir])
            - wp.float64(0.5) * kr[iz, ir] * rho_diff
            + cmul(mul_i(j_coef[iz, ir]) * kz[iz, ir] * velocity, jm[iz, ir])
            + cmul(
                _C2 * t_eb[iz, ir] * sine_over_omega[iz, ir],
                -wp.float64(0.5) * kr[iz, ir] * mul_i(bz[iz, ir])
                - kz[iz, ir] * bm[iz, ir]
                - cmul(_MU0 * t_cc[iz, ir], jm[iz, ir]),
            )
        )
        result_ez[iz, ir] = (
            cmul(t_eb[iz, ir] * cosine[iz, ir], ez[iz, ir])
            - kz[iz, ir] * mul_i(rho_diff)
            + cmul(mul_i(j_coef[iz, ir]) * kz[iz, ir] * velocity, jz[iz, ir])
            + cmul(
                _C2 * t_eb[iz, ir] * sine_over_omega[iz, ir],
                kr[iz, ir] * mul_i(bp[iz, ir])
                + kr[iz, ir] * mul_i(bm[iz, ir])
                - cmul(_MU0 * t_cc[iz, ir], jz[iz, ir]),
            )
        )
        result_bp[iz, ir] = (
            cmul(t_eb[iz, ir] * cosine[iz, ir], bp[iz, ir])
            - cmul(
                t_eb[iz, ir] * sine_over_omega[iz, ir],
                -wp.float64(0.5) * kr[iz, ir] * mul_i(ez[iz, ir])
                + kz[iz, ir] * ep[iz, ir],
            )
            + cmul(
                j_coef[iz, ir],
                -wp.float64(0.5) * kr[iz, ir] * mul_i(jz[iz, ir])
                + kz[iz, ir] * jp[iz, ir],
            )
        )
        result_bm[iz, ir] = (
            cmul(t_eb[iz, ir] * cosine[iz, ir], bm[iz, ir])
            - cmul(
                t_eb[iz, ir] * sine_over_omega[iz, ir],
                -wp.float64(0.5) * kr[iz, ir] * mul_i(ez[iz, ir])
                - kz[iz, ir] * em[iz, ir],
            )
            + cmul(
                j_coef[iz, ir],
                -wp.float64(0.5) * kr[iz, ir] * mul_i(jz[iz, ir])
                - kz[iz, ir] * jm[iz, ir],
            )
        )
        result_bz[iz, ir] = (
            cmul(t_eb[iz, ir] * cosine[iz, ir], bz[iz, ir])
            - cmul(
                t_eb[iz, ir] * sine_over_omega[iz, ir],
                kr[iz, ir] * mul_i(ep[iz, ir]) + kr[iz, ir] * mul_i(em[iz, ir]),
            )
            + cmul(
                j_coef[iz, ir],
                kr[iz, ir] * mul_i(jp[iz, ir]) + kr[iz, ir] * mul_i(jm[iz, ir]),
            )
        )


@wp.kernel(enable_backward=True, module="unique")
def damp_eb_left(
    er: wp.array2d(dtype=wp.vec2d),
    et: wp.array2d(dtype=wp.vec2d),
    ez: wp.array2d(dtype=wp.vec2d),
    br: wp.array2d(dtype=wp.vec2d),
    bt: wp.array2d(dtype=wp.vec2d),
    bz: wp.array2d(dtype=wp.vec2d),
    damping: wp.array(dtype=wp.float64),
    nd: int,
    result_er: wp.array2d(dtype=wp.vec2d),
    result_et: wp.array2d(dtype=wp.vec2d),
    result_ez: wp.array2d(dtype=wp.vec2d),
    result_br: wp.array2d(dtype=wp.vec2d),
    result_bt: wp.array2d(dtype=wp.vec2d),
    result_bz: wp.array2d(dtype=wp.vec2d),
):
    """Copy all fields and apply fixed damping to the left boundary cells."""
    iz, ir = wp.tid()
    if iz < er.shape[0] and ir < er.shape[1]:
        factor = wp.float64(1.0)
        if iz < nd:
            factor = damping[iz]
        result_er[iz, ir] = er[iz, ir] * factor
        result_et[iz, ir] = et[iz, ir] * factor
        result_ez[iz, ir] = ez[iz, ir] * factor
        result_br[iz, ir] = br[iz, ir] * factor
        result_bt[iz, ir] = bt[iz, ir] * factor
        result_bz[iz, ir] = bz[iz, ir] * factor


@wp.kernel(enable_backward=True, module="unique")
def damp_eb_right(
    er: wp.array2d(dtype=wp.vec2d),
    et: wp.array2d(dtype=wp.vec2d),
    ez: wp.array2d(dtype=wp.vec2d),
    br: wp.array2d(dtype=wp.vec2d),
    bt: wp.array2d(dtype=wp.vec2d),
    bz: wp.array2d(dtype=wp.vec2d),
    damping: wp.array(dtype=wp.float64),
    nd: int,
    result_er: wp.array2d(dtype=wp.vec2d),
    result_et: wp.array2d(dtype=wp.vec2d),
    result_ez: wp.array2d(dtype=wp.vec2d),
    result_br: wp.array2d(dtype=wp.vec2d),
    result_bt: wp.array2d(dtype=wp.vec2d),
    result_bz: wp.array2d(dtype=wp.vec2d),
):
    """Copy all fields and apply mirrored damping to the right boundary cells."""
    iz, ir = wp.tid()
    if iz < er.shape[0] and ir < er.shape[1]:
        factor = wp.float64(1.0)
        mirrored_index = er.shape[0] - iz - 1
        if mirrored_index < nd:
            factor = damping[mirrored_index]
        result_er[iz, ir] = er[iz, ir] * factor
        result_et[iz, ir] = et[iz, ir] * factor
        result_ez[iz, ir] = ez[iz, ir] * factor
        result_br[iz, ir] = br[iz, ir] * factor
        result_bt[iz, ir] = bt[iz, ir] * factor
        result_bz[iz, ir] = bz[iz, ir] * factor


@wp.func
def _fixed_shift_power(shift_factor: wp.vec2d, n_move: int):
    # A function boundary replays this fixed power calculation during backward.
    # A mutable local accumulator inside the kernel loses its final value.
    power = wp.vec2d(wp.float64(1.0), wp.float64(0.0))
    for _ in range(abs(n_move)):
        power = cmul(power, shift_factor)
    if n_move < 0:
        power = wp.vec2d(power[0], -power[1])
    return power


@wp.kernel(enable_backward=True, module="unique")
def shift_spectral(
    field: wp.array2d(dtype=wp.vec2d),
    shift_factor: wp.array(dtype=wp.vec2d),
    n_move: int,
    result: wp.array2d(dtype=wp.vec2d),
):
    """Shift a field through fixed integer displacement and spectral factors."""
    iz, ir = wp.tid()
    if iz < field.shape[0] and ir < field.shape[1]:
        power = _fixed_shift_power(shift_factor[iz], n_move)
        result[iz, ir] = cmul(field[iz, ir], power)
