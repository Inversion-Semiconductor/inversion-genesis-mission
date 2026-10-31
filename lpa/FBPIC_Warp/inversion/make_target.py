"""Write a twin target (plan Section 3.4, item 5).

Usage (from `lpa/FBPIC_Warp/inversion`, one GPU):

    python make_target.py twin_s T0
    python make_target.py twin_s T0p

Variant T0 runs the EV model with the inversion settings (K_e, epsilon), a
different NumPy injection seed, and p_nz = 2, at theta*. Variant T0p (T0'
in the plan) uses exactly the inversion settings (config seed, p_nz = 1), so
J(theta*) is at round-off level: it separates convergence from model bias. The target holds the
selected charge, the 12 moments, the frozen standardization of xi, and the
scales s_k. `rff_mean` stays empty until the MMD term exists (Phase P7).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from observable import beam_statistics, moment_scales, phase_space, selection
from twin_config import config_sha256, load_config, theta_from_hat

TARGET_DIR = Path(__file__).resolve().parent / "targets"
VARIANT_SEEDS = {"T0": 7, "T1": 8, "T2": 9, "T0p": None}
VARIANT_P_NZ = 2
P_NZ = {"T0p": 1}


def target_path(size, variant, n_t):
    return TARGET_DIR / f"{size}_{variant}_Nt{n_t}.npz"


def git_hash() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
        check=True,
    )
    return out.stdout.strip()


def target_from_beam(elec, z_win, config, theta_star, variant, seed, p_nz=VARIANT_P_NZ):
    """Target dict from the elec_from_N arrays of a truth run."""
    sel = config["selection"]
    stats0 = beam_statistics(elec, z_win, sel["u_c"], sel["delta_u"], 0.0)
    Q_f = 1e-3 * stats0["Q"]
    stats = beam_statistics(elec, z_win, sel["u_c"], sel["delta_u"], Q_f)
    ws = elec["w"] * selection(elec["uz"], sel["u_c"], sel["delta_u"])
    xi = phase_space(elec, z_win)
    mean = (xi * ws).sum(axis=1) / ws.sum()
    std = np.sqrt((((xi - mean[:, None]) ** 2) * ws).sum(axis=1) / ws.sum())
    return {
        "Q": stats["Q"],
        "Q_f": Q_f,
        "moments": stats["moments"],
        "rff_mean": np.zeros(0),
        "standardization": np.stack([mean, std]),
        "s_k": moment_scales(elec, z_win, sel["u_c"], sel["delta_u"]),
        "theta_star": np.asarray(theta_star, float),
        "seed": -1 if seed is None else seed,
        "p_nz": p_nz,
        "N_t": config["num_steps"],
        "variant": variant,
        "config_sha256": config_sha256(config),
        "git_hash": git_hash(),
        "n_selected": stats["n_selected"],
    }


def load_target(size, variant, config):
    """Load a target; refuse it if config_sha256 or N_t does not match."""
    path = target_path(size, variant, config["num_steps"])
    data = dict(np.load(path, allow_pickle=False))
    target = {k: (v.item() if v.ndim == 0 else v) for k, v in data.items()}
    if target["config_sha256"] != config_sha256(config):
        raise ValueError(f"{path.name}: config_sha256 does not match the config.")
    if int(target["N_t"]) != int(config["num_steps"]):
        raise ValueError(f"{path.name}: N_t does not match the config.")
    return target


def main(size, variant):
    from forward import run_ev

    if variant not in ("T0", "T0p"):
        raise NotImplementedError("Phase P3 writes T0 and T0p; T1 and T2 follow in P9.")
    config = load_config(size)
    theta_star = theta_from_hat(config, config["theta_star_hat"])
    seed = VARIANT_SEEDS[variant]
    p_nz = P_NZ.get(variant, VARIANT_P_NZ)
    run = run_ev(theta_star, config, seed=seed, p_nz=p_nz)
    target = target_from_beam(
        run["elec"], run["z_win"], config, theta_star, variant, seed, p_nz
    )
    TARGET_DIR.mkdir(exist_ok=True)
    path = target_path(size, variant, config["num_steps"])
    np.savez(path, **target)
    print(path.name, {k: target[k] for k in ("Q", "Q_f", "n_selected", "seed", "p_nz")})
    print("moments", target["moments"])
    print("s_k", target["s_k"])
    print("wall_s", run["wall_s"], "births", run["births"], "memory", run["memory"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
