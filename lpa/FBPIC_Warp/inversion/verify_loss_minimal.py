"""Phase P3 step 7: the minimal loss is finite on empty and nearly empty beams.

Runs on the host; no GPU needed. Uses a synthetic target with the Twin-S
selection, so it does not depend on a simulation.

    python verify_loss_minimal.py
"""

from __future__ import annotations

import numpy as np
from observable import beam_statistics, central_moments, minimal_loss
from twin_config import load_config


def synthetic_beam(rng, n, uz_center):
    return {
        "x": rng.normal(0, 2e-6, n),
        "y": rng.normal(0, 2e-6, n),
        "z": 78e-6 + rng.normal(0, 0.5e-6, n),
        "ux": rng.normal(0, 1.0, n),
        "uy": rng.normal(0, 1.0, n),
        "uz": rng.normal(uz_center, 2.0, n),
        "w": rng.uniform(1e3, 2e3, n),
    }


def main():
    config = load_config("twin_s")
    sel = config["selection"]
    u_c, du = sel["u_c"], sel["delta_u"]
    z_win = 70e-6
    rng = np.random.default_rng(0)
    beam = synthetic_beam(rng, 5000, 11.0)
    st = beam_statistics(beam, z_win, u_c, du, 0.0)
    Q_f = 1e-3 * st["Q"]
    st = beam_statistics(beam, z_win, u_c, du, Q_f)
    target = {
        "Q": st["Q"],
        "Q_f": Q_f,
        "moments": st["moments"],
    }
    target["s_k"] = np.abs(st["moments"]) + 1e-6
    assert minimal_loss(st, target) == 0.0

    # Empty: every sigma is 0 (all uz below u_c - 2 delta_u)
    empty = synthetic_beam(rng, 100, 0.0)
    empty["uz"] = np.full(100, u_c - 3 * du)
    se = beam_statistics(empty, z_win, u_c, du, Q_f)
    loss_empty = minimal_loss(se, target)
    K = len(target["moments"])
    expected = (np.log(Q_f) - np.log(target["Q"] + Q_f)) ** 2 + np.sum(
        central_moments(target["moments"]) ** 2 / target["s_k"] ** 2
    ) / K
    assert se["Q"] == 0.0 and np.all(se["moments"] == 0.0)
    assert np.isfinite(loss_empty)
    assert abs(loss_empty - expected) <= 1e-12 * expected

    # Nearly empty: one particle with sigma = 1e-6
    near = synthetic_beam(rng, 1, 0.0)
    # Solve S5(t) = 1e-6 for t; S5 ~ 10 t^3 near 0
    t = (1e-7) ** (1 / 3)
    for _ in range(50):
        f = t**3 * (t * (6 * t - 15) + 10) - 1e-6
        df = 30 * t**2 * (t - 1) ** 2
        t -= f / df
    near["uz"] = np.array([u_c - 2 * du + 4 * du * t])
    sn = beam_statistics(near, z_win, u_c, du, Q_f)
    loss_near = minimal_loss(sn, target)
    assert np.isfinite(loss_near) and np.all(np.isfinite(sn["moments"]))
    sigma = sn["sum_wsigma"] / near["w"][0]
    print(
        {
            "loss_empty": loss_empty,
            "expected_empty": expected,
            "loss_nearly_empty": loss_near,
            "sigma_nearly_empty": sigma,
        }
    )
    assert abs(sigma - 1e-6) < 1e-12
    print("PASS")


if __name__ == "__main__":
    main()
