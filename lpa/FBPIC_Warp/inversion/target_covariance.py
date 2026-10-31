"""Residual weights (whitening) for the Section 2.4 loss (Phase P3).

Sigma = diag(Delta^2), a model-error term: the mean over the nominal theta
and the 4 corners of (v_truth - v_model)^2, with v_truth from the truth
settings (p_nz = 2, NumPy seed 7) and v_model from the inversion settings
(the cached `fd_recovery.py` runs, p_nz = 1, seed 0). The points exclude
theta*, so the weights carry no knowledge of the truth. One point is not
enough: at the nominal point alone, Delta of the relative energy spread is
70 times smaller than the model error at theta*.

Usage (from `lpa/FBPIC_Warp/inversion`, one GPU; the runs can go in
parallel processes):

    python target_covariance.py run nominal      # also corner00, corner10,
                                                 # corner01, corner11
    python target_covariance.py build
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from make_target import VARIANT_P_NZ, VARIANT_SEEDS, load_target, target_path
from observable import RESIDUAL_NAMES, beam_statistics, loss_floors, loss_observables
from twin_config import config_sha256, load_config, theta_from_hat

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SIZE, VARIANT = "twin_s", "T0"
MODEL_ERROR_POINTS = {
    "nominal": (0.5, 0.5),
    "corner00": (0.0, 0.0),
    "corner10": (1.0, 0.0),
    "corner01": (0.0, 1.0),
    "corner11": (1.0, 1.0),
}


def _particles_path(name):
    return RUNS / f"p3_{SIZE}_{name}_particles.npz"


def stage_run(name):
    from forward import run_ev

    config = load_config(SIZE)
    hat = MODEL_ERROR_POINTS[name]
    run = run_ev(
        theta_from_hat(config, hat),
        config,
        seed=VARIANT_SEEDS[VARIANT],
        p_nz=VARIANT_P_NZ,
    )
    sel = config["selection"]
    keep = run["elec"]["uz"] > sel["u_c"] - 2 * sel["delta_u"]
    RUNS.mkdir(exist_ok=True)
    np.savez(
        _particles_path(name),
        z_win=run["z_win"],
        **{k: v[keep] for k, v in run["elec"].items()},
    )
    print(name, "kept", int(keep.sum()), "wall", run["wall_s"])


def _load(path):
    data = dict(np.load(path))
    z_win = float(data.pop("z_win"))
    return data, z_win


def stage_build():
    from fd_recovery import evaluate_batch

    config = load_config(SIZE)
    target = load_target(SIZE, VARIANT, config)
    sel = config["selection"]
    Q_f = target["Q_f"]
    floors = loss_floors(target)

    def v_of(arrays, z_win):
        st = beam_statistics(arrays, z_win, sel["u_c"], sel["delta_u"], Q_f)
        return loss_observables(st, Q_f, floors)

    names = list(MODEL_ERROR_POINTS)
    model = evaluate_batch([MODEL_ERROR_POINTS[n] for n in names])
    deltas = []
    for name, run in zip(names, model):
        stats = {"Q": run["Q_pC"] * 1e-12, "moments": np.asarray(run["moments"])}
        deltas.append(
            v_of(*_load(_particles_path(name))) - loss_observables(stats, Q_f, floors)
        )
    deltas = np.array(deltas)
    delta_rms = np.sqrt(np.mean(deltas**2, axis=0))
    out = {
        "L": np.diag(delta_rms),
        "Sigma": np.diag(delta_rms**2),
        "Delta_model": delta_rms,
        "Delta_points": deltas,
        "Delta_point_names": np.array(names),
        "floor_sigma": floors["sigma"],
        "floor_mean_uz": floors["mean_uz"],
        "v_target": loss_observables(target, Q_f, floors),
        "config_sha256": config_sha256(config),
        "N_t": config["num_steps"],
    }
    path = target_path(SIZE, VARIANT, config["num_steps"]).with_name(
        f"{SIZE}_{VARIANT}_Nt{config['num_steps']}_whitening.npz"
    )
    np.savez(path, **out)
    print(path.name)
    print(
        f"{'observable':20s} {'rms Delta':>11s} " + " ".join(f"{n:>10s}" for n in names)
    )
    for k, n in enumerate(RESIDUAL_NAMES):
        print(
            f"{n:20s} {delta_rms[k]:11.3e} "
            + " ".join(f"{d:+10.2e}" for d in deltas[:, k])
        )


def load_whitening(config):
    """Load the whitening file; refuse a config or N_t mismatch."""
    path = target_path(SIZE, VARIANT, config["num_steps"]).with_name(
        f"{SIZE}_{VARIANT}_Nt{config['num_steps']}_whitening.npz"
    )
    data = dict(np.load(path))
    if str(data["config_sha256"]) != config_sha256(config):
        raise ValueError(f"{path.name}: config_sha256 does not match the config.")
    if int(data["N_t"]) != int(config["num_steps"]):
        raise ValueError(f"{path.name}: N_t does not match the config.")
    return data


if __name__ == "__main__":
    if sys.argv[1] == "run":
        stage_run(sys.argv[2])
    else:
        stage_build()
