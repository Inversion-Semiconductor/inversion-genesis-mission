"""Beam observable H and the minimal loss (plan Sections 2.3 and 2.4).

Works on NumPy or CuPy arrays (the array module of the inputs).
The phase-space coordinates are xi = (x, ux, y, uy, z - z_win, uz).
The moments are the 6 first moments <xi_k> and the 6 raw diagonal second
moments <xi_k^2>, all weighted by w * sigma with the Q_f floor.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import e

XI_NAMES = ("x", "ux", "y", "uy", "zeta", "uz")
MOMENT_NAMES = tuple(f"<{n}>" for n in XI_NAMES) + tuple(f"<{n}^2>" for n in XI_NAMES)


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


def minimal_loss(stats: dict, target: dict, alpha_Q=1.0, alpha_M=None) -> float:
    """Charge and moment terms of the Section 2.4 loss, alpha_D = 0.

    Args:
        stats: Output of `beam_statistics` with the target's Q_f.
        target: Dict with `Q`, `moments`, `s_k`, `Q_f`.
    """
    K = len(target["moments"])
    alpha_M = 1.0 / K if alpha_M is None else alpha_M
    Q_f = target["Q_f"]
    charge = (np.log(stats["Q"] + Q_f) - np.log(target["Q"] + Q_f)) ** 2
    moments = np.sum((stats["moments"] - target["moments"]) ** 2 / target["s_k"] ** 2)
    return float(alpha_Q * charge + alpha_M * moments)
