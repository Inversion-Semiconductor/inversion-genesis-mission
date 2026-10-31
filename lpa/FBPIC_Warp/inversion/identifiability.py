"""Phase P1 steps 6 and 7: selection, trapping check, and identifiability.

Usage (from `lpa/FBPIC_Warp/inversion`, one GPU):

    python identifiability.py selection     # nominal run, sets u_c and delta_u
    python identifiability.py trapping      # nominal, 4 corners, 5 starts
    python identifiability.py sensitivity   # theta* +- 0.05 e_i, 4 seeds; T2
    python identifiability.py report        # G, cond(G), sqrt(diag G^-1)

Each forward run is cached in `runs/p1/` (not committed). The `selection`
stage writes `u_c`, `delta_u`, `theta_star_hat`, and the optimizer starts
into `configs/twin_s.json`.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
from forward import run_production
from observable import (
    MOMENT_NAMES,
    beam_statistics,
    moment_scales,
    observable_vector,
)
from twin_config import CONFIG_DIR, config_sha256, load_config, theta_from_hat

SIZE = "twin_s"
RUN_DIR = Path(__file__).resolve().parent / "runs" / "p1"
RESULT_FILE = Path(__file__).resolve().parent / "results" / "p1_twin_s.json"
H_IDENT = 0.05
IDENT_SEEDS = (101, 102, 103, 104)
T2_NUMPY_SEED = 7
T2_RNG_SEED = 999
THETA_STAR_HAT = (0.6, 0.4)
LHS_SEED = 2026


def _run(name, theta_hat, config, **kwargs):
    """Cached production run at normalized parameters theta_hat."""
    path = RUN_DIR / f"{name}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return data["result"].item()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    theta = theta_from_hat(config, theta_hat)
    result = run_production(theta, config, **kwargs)
    result["theta_hat"] = np.asarray(theta_hat, float)
    np.savez(path, result=np.array(result, dtype=object))
    print(
        f"{name}: theta_hat={np.round(theta_hat, 4)} wall={result['wall_s']:.1f}s "
        f"n_elec={result['elec']['uz'].size}",
        flush=True,
    )
    gc.collect()
    return result


def choose_selection(uz, w, bin_width=0.25, smooth=5):
    """u_c at the histogram minimum between the low-energy peak and the
    trapped peak; delta_u from the 99th and 1st weighted percentiles."""
    edges = np.arange(min(uz.min(), 0.0), uz.max() + bin_width, bin_width)
    hist, edges = np.histogram(uz, bins=edges, weights=w)
    kernel = np.ones(smooth) / smooth
    hs = np.convolve(hist, kernel, mode="same")
    centers = 0.5 * (edges[1:] + edges[:-1])
    low = np.argmax(np.where(centers < 3.0, hs, -1))
    trapped = np.argmax(np.where(centers > centers[low] + 3.0, hs, -1))
    between = slice(low, trapped + 1)
    u_c = float(centers[between][np.argmin(hs[between])])

    def wpercentile(values, weights, q):
        order = np.argsort(values)
        cw = np.cumsum(weights[order])
        return float(values[order][np.searchsorted(cw, q * cw[-1])])

    below, above = uz < u_c, uz >= u_c
    u_lo = wpercentile(uz[below], w[below], 0.99)
    u_hi = wpercentile(uz[above], w[above], 0.01)
    delta_u = min(u_c - u_lo, u_hi - u_c) / 4.0
    return {
        "u_c": u_c,
        "delta_u": delta_u,
        "u_lo": u_lo,
        "u_hi": u_hi,
        "u_low_peak": float(centers[low]),
        "u_trapped_peak": float(centers[trapped]),
    }


def optimizer_starts(theta_star_hat, n=5, seed=LHS_SEED, min_dist=0.25):
    """Seeded Latin hypercube in [0.1, 0.9]^2, each point at least
    `min_dist` from theta_star_hat in the infinity norm. Redraw the whole
    design with the next seed until every point qualifies."""
    from scipy.stats import qmc

    for s in range(seed, seed + 1000):
        pts = 0.1 + 0.8 * qmc.LatinHypercube(d=2, seed=s).random(n)
        dist = np.abs(pts - np.asarray(theta_star_hat)).max(axis=1)
        if np.all(dist >= min_dist):
            return pts, s
    raise RuntimeError("No admissible Latin hypercube design found.")


def _save_config(config):
    with open(CONFIG_DIR / f"{SIZE}.json", "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def _load_results():
    if RESULT_FILE.exists():
        return json.loads(RESULT_FILE.read_text())
    return {}


def _save_results(results):
    RESULT_FILE.parent.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps(results, indent=2) + "\n")


def stage_selection():
    config = load_config(SIZE)
    nominal_hat = (0.5, 0.5)
    res = _run("nominal", nominal_hat, config)
    sel = choose_selection(res["elec"]["uz"], res["elec"]["w"])
    starts, lhs_seed = optimizer_starts(THETA_STAR_HAT)
    config["selection"] = {"u_c": sel["u_c"], "delta_u": sel["delta_u"]}
    config["theta_star_hat"] = list(THETA_STAR_HAT)
    config["optimizer_starts_hat"] = starts.tolist()
    config["seeds"]["lhs"] = lhs_seed
    _save_config(config)
    results = _load_results()
    results["selection"] = sel
    results["optimizer_starts"] = {"lhs_seed": lhs_seed, "points": starts.tolist()}
    results["nominal_run"] = {
        "wall_s": res["wall_s"],
        "memory": res["memory"],
        "counts": res["counts"],
        "z_win": res["z_win"],
    }
    _save_results(results)
    print(json.dumps(sel, indent=2))


def _stats(res, config, Q_f=0.0):
    sel = config["selection"]
    return beam_statistics(res["elec"], res["z_win"], sel["u_c"], sel["delta_u"], Q_f)


def stage_trapping():
    config = load_config(SIZE)
    nominal = _stats(_run("nominal", (0.5, 0.5), config), config)
    points = {"nominal": (0.5, 0.5)}
    for i, corner in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        points[f"corner{i}"] = corner
    for i, start in enumerate(config["optimizer_starts_hat"]):
        points[f"start{i}"] = tuple(start)
    table = {}
    for name, hat in points.items():
        st = _stats(_run(name, hat, config), config)
        passed = st["n_selected"] >= 1000 and st["Q"] >= 0.1 * nominal["Q"]
        table[name] = {
            "theta_hat": list(hat),
            "n_selected": st["n_selected"],
            "Q_pC": st["Q"] * 1e12,
            "Q_over_nominal": st["Q"] / nominal["Q"],
            "pass": bool(passed),
        }
        print(name, table[name], flush=True)
    results = _load_results()
    results["trapping"] = table
    _save_results(results)


def stage_sensitivity():
    config = load_config(SIZE)
    star = np.asarray(config["theta_star_hat"])
    for seed in IDENT_SEEDS:
        _run(f"star_s{seed}", star, config, rng_seed=seed)
        for i in range(2):
            for sign, tag in ((+1, "p"), (-1, "m")):
                hat = star.copy()
                hat[i] += sign * H_IDENT
                _run(f"d{i}{tag}_s{seed}", hat, config, rng_seed=seed)
    _run("t2_truth", star, config, seed=T2_NUMPY_SEED, rng_seed=T2_RNG_SEED, p_nz=2)


def stage_report():
    config = load_config(SIZE)
    star = np.asarray(config["theta_star_hat"])
    sel = config["selection"]
    star_runs = [_run(f"star_s{s}", star, config, rng_seed=s) for s in IDENT_SEEDS]
    Q_y = np.mean([_stats(r, config)["Q"] for r in star_runs])
    Q_f = 1e-3 * Q_y
    s_k = np.mean(
        [
            moment_scales(r["elec"], r["z_win"], sel["u_c"], sel["delta_u"])
            for r in star_runs
        ],
        axis=0,
    )
    scales = np.concatenate([[1.0], s_k])  # log Q has unit scale

    def v_of(res):
        return observable_vector(_stats(res, config, Q_f), Q_f)

    v_star = np.array([v_of(r) for r in star_runs])
    S_seeds = np.zeros((len(IDENT_SEEDS), v_star.shape[1], 2))
    for k, seed in enumerate(IDENT_SEEDS):
        for i in range(2):
            hp, hm = star.copy(), star.copy()
            hp[i] += H_IDENT
            hm[i] -= H_IDENT
            vp = v_of(_run(f"d{i}p_s{seed}", hp, config, rng_seed=seed))
            vm = v_of(_run(f"d{i}m_s{seed}", hm, config, rng_seed=seed))
            S_seeds[k, :, i] = (vp - vm) / (2 * H_IDENT)
    S = S_seeds.mean(axis=0)
    S_se = S_seeds.std(axis=0, ddof=1) / np.sqrt(len(IDENT_SEEDS))
    v_model = v_star.mean(axis=0)
    seed_var = v_star.var(axis=0, ddof=1)
    t2 = _run(
        "t2_truth", star, config, seed=T2_NUMPY_SEED, rng_seed=T2_RNG_SEED, p_nz=2
    )
    v_truth = v_of(t2)
    sigma2 = np.maximum(seed_var + (v_truth - v_model) ** 2, (0.02 * scales) ** 2)
    G = S.T @ (S / sigma2[:, None])
    evals, evecs = np.linalg.eigh(G)
    Ginv = np.linalg.inv(G)
    report = {
        "Q_y_pC": Q_y * 1e12,
        "names": ["logQ", *MOMENT_NAMES],
        "S": S.tolist(),
        "S_se": S_se.tolist(),
        "v_model": v_model.tolist(),
        "v_truth_T2": v_truth.tolist(),
        "seed_var": seed_var.tolist(),
        "sigma2": sigma2.tolist(),
        "scales": scales.tolist(),
        "G": G.tolist(),
        "eigenvalues": evals.tolist(),
        "cond_G": float(evals[-1] / evals[0]),
        "weakest_eigenvector": evecs[:, 0].tolist(),
        "sqrt_diag_Ginv": np.sqrt(np.diag(Ginv)).tolist(),
        "gate_pass": bool(
            evals[-1] / evals[0] <= 1e4 and np.all(np.sqrt(np.diag(Ginv)) <= 0.02)
        ),
        "config_sha256": config_sha256(config),
    }
    results = _load_results()
    results["identifiability"] = report
    _save_results(results)
    for key in (
        "eigenvalues",
        "cond_G",
        "weakest_eigenvector",
        "sqrt_diag_Ginv",
        "gate_pass",
    ):
        print(key, report[key])
    print("per-observable S / sqrt(sigma2):")
    for n, srow, s2 in zip(report["names"], S, sigma2):
        print(f"  {n:8s} {srow[0] / np.sqrt(s2):9.3f} {srow[1] / np.sqrt(s2):9.3f}")


if __name__ == "__main__":
    stages = {
        "selection": stage_selection,
        "trapping": stage_trapping,
        "sensitivity": stage_sensitivity,
        "report": stage_report,
    }
    for arg in sys.argv[1:]:
        stages[arg]()
