"""Beam observable H and the minimal loss (plan Sections 2.3 and 2.4).

Works on NumPy or CuPy arrays (the array module of the inputs).
The phase-space coordinates are xi = (x, ux, y, uy, z - z_win, uz).
`beam_statistics` returns the 6 first moments <xi_k> and the 6 raw diagonal
second moments <xi_k^2>, all weighted by w * sigma with the Q_f floor.
The loss compares the central second moments <xi_k^2> - <xi_k>^2, so that
each second-moment term measures a spread and does not count the mean again.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import e

XI_NAMES = ("x", "ux", "y", "uy", "zeta", "uz")
MOMENT_NAMES = tuple(f"<{n}>" for n in XI_NAMES) + tuple(f"<{n}^2>" for n in XI_NAMES)
CENTRAL_NAMES = tuple(f"<{n}>" for n in XI_NAMES) + tuple(f"var({n})" for n in XI_NAMES)
# <y> and <uy> are left out of the loss: by symmetry they carry no
# information on theta (P1 sensitivity near 0), and they jitter by several
# model-error units under small changes of theta (Section 12.5).
LOSS_MEANS = (0, 1, 4, 5)
RESIDUAL_NAMES = (
    ("log(Q+Q_f)",)
    + tuple(f"<{XI_NAMES[k]}>" for k in LOSS_MEANS)
    + tuple(f"log(sigma_{n})" for n in XI_NAMES[:5])
    + ("log(sigma_uz/<uz>)",)
)
FLOOR_FRACTION = 1e-3


def _xp(a):
    try:
        import cupy

        return cupy.get_array_module(a)
    except ImportError:  # pragma: no cover
        return np


def smoothstep5(t, xp=np):
    """Quintic smoothstep 6t^5 - 15t^4 + 10t^3, clamped to [0, 1]."""
    t = xp.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def selection(uz, u_c, delta_u):
    """sigma_i = S5((uz - u_c + 2 delta_u) / (4 delta_u)) (Section 2.3)."""
    xp = _xp(uz)
    return smoothstep5((uz - u_c + 2.0 * delta_u) / (4.0 * delta_u), xp)


def phase_space(arrays: dict, z_win: float):
    """Stack xi = (x, ux, y, uy, z - z_win, uz) as a (6, N) array."""
    xp = _xp(arrays["uz"])
    return xp.stack(
        [
            arrays["x"],
            arrays["ux"],
            arrays["y"],
            arrays["uy"],
            arrays["z"] - z_win,
            arrays["uz"],
        ]
    )


def beam_statistics(arrays: dict, z_win: float, u_c: float, delta_u: float, Q_f: float):
    """Selected charge [C] and the 12 moments.

    Args:
        arrays: Particle arrays x, y, z, ux, uy, uz, w of `elec_from_N`.
        z_win: Left edge of the moving window [m] at the observation step.
        u_c, delta_u: Selection center and width (Section 3.4).
        Q_f: Charge floor [C] of the moment denominator.

    Returns:
        A dict with `Q` (float), `moments` (12 floats, host), `n_selected`
        (macroparticles with sigma > 0.5), and `sum_wsigma`.
    """
    xp = _xp(arrays["uz"])
    sigma = selection(arrays["uz"], u_c, delta_u)
    ws = arrays["w"] * sigma
    xi = phase_space(arrays, z_win)
    sum_ws = float(ws.sum())
    denom = sum_ws + Q_f / e
    first = (xi * ws).sum(axis=1) / denom
    second = (xi * xi * ws).sum(axis=1) / denom
    moments = xp.concatenate([first, second])
    if xp is not np:
        moments = moments.get()
    return {
        "Q": e * sum_ws,
        "moments": np.asarray(moments, float),
        "n_selected": int((sigma > 0.5).sum()),
        "sum_wsigma": sum_ws,
    }


def moment_scales(arrays: dict, z_win: float, u_c: float, delta_u: float):
    """Scales s_k from a target beam: the weighted standard deviation for
    first moments and its square for second moments (Section 2.4)."""
    xp = _xp(arrays["uz"])
    ws = arrays["w"] * selection(arrays["uz"], u_c, delta_u)
    xi = phase_space(arrays, z_win)
    mean = (xi * ws).sum(axis=1) / ws.sum()
    std = xp.sqrt((((xi - mean[:, None]) ** 2) * ws).sum(axis=1) / ws.sum())
    if xp is not np:
        std = std.get()
    std = np.asarray(std, float)
    return np.concatenate([std, std**2])


def observable_vector(stats: dict, Q_f: float) -> np.ndarray:
    """v = (log(Q + Q_f), m_1, ..., m_12) of plan Phase P1 step 7."""
    return np.concatenate([[np.log(stats["Q"] + Q_f)], stats["moments"]])


def central_moments(moments) -> np.ndarray:
    """(<xi>, <xi^2>) -> (<xi>, <xi^2> - <xi>^2) for the 12 moments."""
    m = np.asarray(moments, float)
    return np.concatenate([m[:6], m[6:] - m[:6] ** 2])


def residuals(stats: dict, target: dict, alpha_Q=1.0, alpha_M=None) -> np.ndarray:
    """The 13 weighted residuals r with minimal loss J = sum(r^2).

    r_0 = sqrt(alpha_Q) (log(Q + Q_f) - log(Q_y + Q_f)), and
    r_k = sqrt(alpha_M) (c_k - c_{k,y}) / s_k for the 6 first moments and the
    6 central second moments c_k. s_k is the target standard deviation for a
    first moment and its square, the target variance, for a second moment.

    Args:
        stats: `beam_statistics` output (raw moments) with the target's Q_f.
        target: Dict with `Q`, `moments` (raw), `s_k`, `Q_f`.
        alpha_Q, alpha_M: Term weights; alpha_M defaults to 1/K, K = 12.
    """
    K = len(target["moments"])
    alpha_M = 1.0 / K if alpha_M is None else alpha_M
    Q_f = target["Q_f"]
    charge = np.log(stats["Q"] + Q_f) - np.log(target["Q"] + Q_f)
    diff = central_moments(stats["moments"]) - central_moments(target["moments"])
    return np.concatenate(
        [[np.sqrt(alpha_Q) * charge], np.sqrt(alpha_M) * diff / target["s_k"]]
    )


def minimal_loss(stats: dict, target: dict, alpha_Q=1.0, alpha_M=None) -> float:
    """Charge and moment terms of the Section 2.4 loss, alpha_D = 0.

    J = sum of the squared `residuals`, with central second moments.

    Args:
        stats: Output of `beam_statistics` with the target's Q_f.
        target: Dict with `Q`, `moments`, `s_k`, `Q_f`.
    """
    return float(np.sum(residuals(stats, target, alpha_Q, alpha_M) ** 2))


def loss_observables(stats: dict, Q_f: float, floors: dict) -> np.ndarray:
    """The 11 loss observables v of plan Section 2.4.

    v = (log(Q + Q_f), the first moments of x, ux, zeta, and uz,
    log(sigma_k + f_k) for x, ux, y,
    uy, zeta, and log((sigma_uz + f_uz) / (<uz> + f_mu))). The spreads come
    from the central moments, sigma_k^2 = <xi_k^2> - <xi_k>^2 >= 0 (this holds
    with the Q_f floor in the denominators). The floors keep v finite when the
    selected beam vanishes.

    Args:
        stats: `beam_statistics` output (raw moments) with the same Q_f.
        Q_f: Charge floor [C].
        floors: Dict with `sigma` (6 floors f_k) and `mean_uz` (f_mu).
    """
    m = np.asarray(stats["moments"], float)
    mean = m[:6]
    sigma = np.sqrt(np.maximum(m[6:] - mean**2, 0.0))
    f = np.asarray(floors["sigma"], float)
    log_sigma = np.log(sigma[:5] + f[:5])
    rel_spread = np.log((sigma[5] + f[5]) / (mean[5] + floors["mean_uz"]))
    return np.concatenate(
        [[np.log(stats["Q"] + Q_f)], mean[list(LOSS_MEANS)], log_sigma, [rel_spread]]
    )


def loss_floors(target: dict) -> dict:
    """Floors f_k = 1e-3 times the target spread, and 1e-3 times <uz>_y."""
    m = np.asarray(target["moments"], float)
    sigma = np.sqrt(np.maximum(m[6:] - m[:6] ** 2, 0.0))
    return {"sigma": FLOOR_FRACTION * sigma, "mean_uz": FLOOR_FRACTION * abs(m[5])}


def whitened_residuals(stats: dict, target: dict, whitening: dict) -> np.ndarray:
    """r_w = L^-1 (v - v_y) with Sigma = L L^T (plan Section 2.4).

    Args:
        stats: `beam_statistics` output with the target's Q_f.
        target: T0 target dict (raw moments, `Q`, `Q_f`).
        whitening: Dict with the lower Cholesky factor `L` and `floors`.
    """
    from scipy.linalg import solve_triangular

    floors = {
        "sigma": whitening["floor_sigma"],
        "mean_uz": float(whitening["floor_mean_uz"]),
    }
    v = loss_observables(stats, target["Q_f"], floors)
    v_y = loss_observables(target, target["Q_f"], floors)
    return solve_triangular(whitening["L"], v - v_y, lower=True)


def whitened_loss(stats: dict, target: dict, whitening: dict) -> float:
    """J = |r_w|^2, the loss of plan Section 2.4 with alpha_D = 0."""
    r = whitened_residuals(stats, target, whitening)
    return float(r @ r)


def minimal_loss_raw(stats: dict, target: dict, alpha_Q=1.0, alpha_M=None) -> float:
    """The first P3 loss, with raw second moments (kept for the record)."""
    K = len(target["moments"])
    alpha_M = 1.0 / K if alpha_M is None else alpha_M
    Q_f = target["Q_f"]
    charge = (np.log(stats["Q"] + Q_f) - np.log(target["Q"] + Q_f)) ** 2
    moments = np.sum((stats["moments"] - target["moments"]) ** 2 / target["s_k"] ** 2)
    return float(alpha_Q * charge + alpha_M * moments)
