"""Phase P3 steps 10 and 11: loss slices and the FD recovery gate.

Usage (from `lpa/FBPIC_Warp/inversion`, one GPU):

    python fd_recovery.py noise      # base theta* three times: sigma_J
    python fd_recovery.py slices     # 13 points over theta*_i +- 0.3, each i
    python fd_recovery.py recover    # L-BFGS-B from the farthest start
    python fd_recovery.py lsq        # trust-region least squares, 5 starts

The recovery gate of P3 steps 11 to 14 (whitened loss of Section 2.4):

    python fd_recovery.py edge                  # z_f edge test (step 12)
    python fd_recovery.py --target T0p design   # 5 x 5 design and basins
    python fd_recovery.py --target T0p fine     # slices at theta* +- 0.05
    python fd_recovery.py --target T0p dogbox   # solves from the design
    python fd_recovery.py --target T0p farthest # farthest optimizer start
    python fd_recovery.py --target T0 --starts 1 dogbox   # bias report
    python fd_recovery.py --target T0p one_E_L one_z_f   # 1-D recoveries

`--target` selects the target of the gate stages (default T0); T0p is the
control target with the inversion settings. The results go under keys with
the target name. `--starts` sets the number of design starts (default 3).

Every loss evaluation is one EV forward run of Twin-S (p_nz = 1, inversion
seed) against the T0 target, with the minimal loss (alpha_D = 0). The runs of
one batch execute as parallel processes on the same GPU (`ev_point.py`), and
each result is cached in `runs/p3/` by its normalized parameters.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from twin_config import load_config

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "runs" / "p3"
RESULT_FILE = HERE / "results" / "p3_fd_recovery.json"
SIZE, VARIANT = "twin_s", "T0"
H_FD = 0.02
MAX_PARALLEL = 5
LOG = []


def _key(theta_hat, tag=""):
    return f"p{theta_hat[0]:.6f}_{theta_hat[1]:.6f}{tag}"


def evaluate_batch(points, tag="", tags=None):
    """Losses at the normalized points; uncached points run in parallel.

    `tags` gives one cache tag per point (repeated runs of one point);
    otherwise every point uses `tag`. A cached result must carry the
    provenance of the current config, target, and EV code.
    """
    from ev_point import provenance

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tags = [tag] * len(points) if tags is None else list(tags)
    jobs = list(dict.fromkeys((tuple(p), t) for p, t in zip(points, tags)))
    todo = [(p, t) for p, t in jobs if not (RUN_DIR / f"{_key(p, t)}.json").exists()]
    for start in range(0, len(todo), MAX_PARALLEL):
        chunk = todo[start : start + MAX_PARALLEL]
        procs = []
        for p, t in chunk:
            out = RUN_DIR / f"{_key(p, t)}.json"
            # Kept open while the child runs; closed after wait().
            log = open(RUN_DIR / f"{_key(p, t)}.log", "w")  # noqa: SIM115
            cmd = [
                sys.executable,
                str(HERE / "ev_point.py"),
                str(p[0]),
                str(p[1]),
                str(out),
            ]
            procs.append(
                (subprocess.Popen(cmd, stdout=log, stderr=log, env=os.environ), log)
            )
        for proc, log in procs:
            proc.wait()
            log.close()
            if proc.returncode != 0:
                raise RuntimeError(f"ev_point.py failed; see {log.name}")
    current = provenance()
    results = []
    for p, t in zip(points, tags):
        result = json.loads((RUN_DIR / f"{_key(p, t)}.json").read_text())
        cached = result.get("provenance")
        if cached is not None and cached != current:
            raise RuntimeError(f"Stale cache entry {_key(p, t)}: provenance differs.")
        results.append(result)
    return results


def _save(section, value):
    data = json.loads(RESULT_FILE.read_text()) if RESULT_FILE.exists() else {}
    data[section] = value
    RESULT_FILE.parent.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps(data, indent=2) + "\n")


def stage_noise(config):
    star = tuple(config["theta_star_hat"])
    # The untagged run is also the slice and recovery cache entry of theta*.
    res = evaluate_batch([star] * 3, tags=["", "_rep1", "_rep2"])
    J = np.array([r["J"] for r in res])
    out = {
        "J": J.tolist(),
        "sigma_J": float(J.std(ddof=1)),
        "Q_pC": [r["Q_pC"] for r in res],
        "wall_s": [r["wall_s"] for r in res],
    }
    _save("noise", out)
    print(json.dumps(out, indent=2))


def stage_slices(config):
    star = np.asarray(config["theta_star_hat"])
    offsets = np.linspace(-0.3, 0.3, 13)
    points = []
    for i in range(2):
        for d in offsets:
            p = star.copy()
            p[i] += d
            points.append(tuple(np.round(p, 6)))
    t0 = time.perf_counter()
    res = evaluate_batch(points)
    out = {"offsets": offsets.tolist(), "wall_s": time.perf_counter() - t0}
    for i, name in enumerate(("E_L", "z_f")):
        rows = res[13 * i : 13 * (i + 1)]
        out[name] = {
            "theta_hat": [r["theta_hat"] for r in rows],
            "J": [r["J"] for r in rows],
            "Q_pC": [r["Q_pC"] for r in rows],
            "n_selected": [r["n_selected"] for r in rows],
        }
        print(name)
        for d, r in zip(offsets, rows):
            print(
                f"  {d:+.2f} J={r['J']:.4e} Q={r['Q_pC']:.3f} pC n_sel={r['n_selected']}"
            )
    _save("slices", out)


def stage_recover(config):
    from scipy.optimize import minimize

    star = np.asarray(config["theta_star_hat"])
    starts = np.asarray(config["optimizer_starts_hat"])
    far = int(np.argmax(np.abs(starts - star).max(axis=1)))
    x0 = starts[far]
    history = []
    counter = {"runs": 0}
    t0 = time.perf_counter()

    def value_and_fd_grad(x):
        x = np.round(np.asarray(x, float), 6)
        pts = [tuple(x)]
        for i in range(2):
            for s in (+1, -1):
                p = x.copy()
                p[i] += s * H_FD
                pts.append(tuple(np.round(p, 6)))
        res = evaluate_batch(pts)
        counter["runs"] += 5
        J = res[0]["J"]
        g = np.array(
            [(res[1 + 2 * i]["J"] - res[2 + 2 * i]["J"]) / (2 * H_FD) for i in range(2)]
        )
        history.append(
            {
                "theta_hat": x.tolist(),
                "J": J,
                "grad_hat": g.tolist(),
                "Q_pC": res[0]["Q_pC"],
                "elapsed_s": time.perf_counter() - t0,
            }
        )
        print(f"eval {len(history)}: x={x} J={J:.5e} g={g}", flush=True)
        return J, g

    res = minimize(
        value_and_fd_grad,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        options={"maxiter": 60, "ftol": 1e-12, "gtol": 1e-6},
    )
    out = {
        "start_index": far,
        "start": x0.tolist(),
        "theta_star_hat": star.tolist(),
        "lbfgsb": {
            "x": res.x.tolist(),
            "fun": float(res.fun),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "message": str(res.message),
            "success": bool(res.success),
        },
    }
    final = res.x
    if "ABNORMAL" in str(res.message).upper():
        n_before = len(history)

        def value_only(x):
            # Powell needs values only: one forward run per point.
            x = np.round(np.asarray(x, float), 6)
            r = evaluate_batch([tuple(x)])[0]
            counter["runs"] += 1
            history.append(
                {
                    "theta_hat": x.tolist(),
                    "J": r["J"],
                    "grad_hat": None,
                    "Q_pC": r["Q_pC"],
                    "elapsed_s": time.perf_counter() - t0,
                    "method": "Powell",
                }
            )
            print(f"powell eval: x={x} J={r['J']:.5e}", flush=True)
            return r["J"]

        pw = minimize(
            value_only,
            res.x,
            method="Powell",
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            options={"xtol": 1e-3, "ftol": 1e-10, "maxfev": 60},
        )
        out["powell"] = {
            "x": pw.x.tolist(),
            "fun": float(pw.fun),
            "nfev": int(pw.nfev),
            "message": str(pw.message),
            "first_eval": n_before + 1,
        }
        final = pw.x
    err = np.abs(np.asarray(final) - star)
    out.update(
        {
            "final": np.asarray(final).tolist(),
            "error": err.tolist(),
            "gate_pass": bool(np.all(err <= 0.02)),
            "forward_runs_requested": counter["runs"],
            "wall_s": time.perf_counter() - t0,
            "history": history,
        }
    )
    _save("recovery", out)
    print(json.dumps({k: v for k, v in out.items() if k != "history"}, indent=2))


def _residuals(result, target):
    """The 13 residuals of a cached run, recomputed with the current loss."""
    from observable import residuals

    stats = {"Q": result["Q_pC"] * 1e-12, "moments": np.asarray(result["moments"])}
    return residuals(stats, target)


def solve_lsq(config, target, x0, label):
    """Bounded trust-region least squares (`least_squares`, method trf) on
    the 13 residuals, with a central-FD Jacobian at h = H_FD (5 runs per
    Jacobian, the centre shared with the residual call)."""
    from scipy.optimize import least_squares

    history = []
    t0 = time.perf_counter()

    def fun(x):
        x = np.round(np.asarray(x, float), 6)
        r = _residuals(evaluate_batch([tuple(x)])[0], target)
        history.append(
            {
                "kind": "residual",
                "theta_hat": x.tolist(),
                "J": float(r @ r),
                "elapsed_s": time.perf_counter() - t0,
            }
        )
        print(f"[{label}] f x={x} J={r @ r:.5e}", flush=True)
        return r

    def jac(x):
        x = np.round(np.asarray(x, float), 6)
        pts = [tuple(x)]
        for i in range(2):
            for sign in (+1, -1):
                p = x.copy()
                p[i] += sign * H_FD
                pts.append(tuple(np.round(p, 6)))
        res = [_residuals(r, target) for r in evaluate_batch(pts)]
        jacobian = np.stack(
            [(res[1 + 2 * i] - res[2 + 2 * i]) / (2 * H_FD) for i in range(2)], axis=1
        )
        grad = 2 * jacobian.T @ res[0]
        history.append(
            {
                "kind": "jacobian",
                "theta_hat": x.tolist(),
                "J": float(res[0] @ res[0]),
                "grad_hat": grad.tolist(),
                "elapsed_s": time.perf_counter() - t0,
            }
        )
        print(f"[{label}] J x={x} grad={grad}", flush=True)
        return jacobian

    res = least_squares(
        fun,
        np.asarray(x0, float),
        jac=jac,
        bounds=([0.0, 0.0], [1.0, 1.0]),
        method="trf",
        x_scale=1.0,
        loss="linear",
        xtol=2e-3,
        ftol=1e-4,
        gtol=1e-8,
        max_nfev=20,
    )
    star = np.asarray(config["theta_star_hat"])
    err = np.abs(res.x - star)
    return {
        "start": list(map(float, x0)),
        "final": res.x.tolist(),
        "J_final": float(2 * res.cost),
        "error": err.tolist(),
        "pass": bool(np.all(err <= 0.02)),
        "nfev": int(res.nfev),
        "njev": int(res.njev),
        "status": int(res.status),
        "message": str(res.message),
        "wall_s": time.perf_counter() - t0,
        "history": history,
    }


def stage_lsq(config):
    """Trust-region least squares from every optimizer start, farthest first."""
    from make_target import load_target

    target = load_target(SIZE, VARIANT, config)
    star = np.asarray(config["theta_star_hat"])
    starts = np.asarray(config["optimizer_starts_hat"])
    order = np.argsort(-np.abs(starts - star).max(axis=1))
    data = json.loads(RESULT_FILE.read_text()) if RESULT_FILE.exists() else {}
    out = data.get("recovery_lsq", {})
    out["settings"] = {
        "method": "trf",
        "bounds": [0, 1],
        "x_scale": 1.0,
        "xtol": 2e-3,
        "ftol": 1e-4,
        "gtol": 1e-8,
        "max_nfev": 20,
        "fd_step": H_FD,
        "loss": "central second moments",
    }
    out["farthest_start_index"] = int(order[0])
    for k in order:
        key = f"start{k}"
        if key in out.get("runs", {}):
            continue
        result = solve_lsq(config, target, starts[k], key)
        out.setdefault("runs", {})[key] = result
        _save("recovery_lsq", out)
        print(
            f"{key}: final={np.round(result['final'], 4)} err={np.round(result['error'], 4)} "
            f"J={result['J_final']:.4e} pass={result['pass']} nfev={result['nfev']} "
            f"njev={result['njev']} wall={result['wall_s']:.0f}s",
            flush=True,
        )
    out["gate_pass_farthest"] = out["runs"][f"start{order[0]}"]["pass"]
    out["all_pass"] = all(r["pass"] for r in out["runs"].values())
    _save("recovery_lsq", out)


# ---------------------------------------------------------------------------
# Recovery gate with the whitened loss (P3 steps 11 to 14, Section 2.5).

# FD steps per coordinate (E_L, z_f): fixed, above the width of the jumps
# from emission thresholds and cell crossings (Section 2.5).
H_GATE = (0.02, 0.05)
GATE = {"target": "T0", "starts": 3}
FINE_OFFSETS = (-0.05, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.05)
EDGE_POINT = (0.603082, 0.0)
DESIGN_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)
DOGBOX = {"xtol": 2e-3, "ftol": 1e-4, "gtol": 1e-8, "max_nfev": 20}


def _gate_inputs(config):
    from make_target import load_target
    from target_covariance import load_whitening

    return load_target(SIZE, GATE["target"], config), load_whitening(config)


def _gkey(name):
    return f"{name}_{GATE['target']}"


def _stats_of(result, target=None):
    """Q and raw moments of a cached run, for the charge floor of `target`.

    `ev_point.py` forms the moments with the Q_f of T0. Each moment is
    sum / (Q/e + Q_f/e), so a different target floor is an exact rescale.
    """
    from make_target import load_target

    Q = result["Q_pC"] * 1e-12
    moments = np.asarray(result["moments"])
    if target is not None:
        Q_f_run = load_target(SIZE, VARIANT, load_config(SIZE))["Q_f"]
        moments = moments * (Q + Q_f_run) / (Q + target["Q_f"])
    return {"Q": Q, "moments": moments}


def _wres(result, target, whitening):
    """Whitened residuals r_w of a cached run (Section 2.4)."""
    from observable import whitened_residuals

    return whitened_residuals(_stats_of(result, target), target, whitening)


def _pt(x):
    return tuple(float(v) for v in np.round(np.asarray(x, float), 6))


def _stencil_points(x, h):
    """Stencil per coordinate: central inside, one-sided within h of a bound."""
    pts, kinds = [], []
    for i in range(2):
        if x[i] - h[i] < 0.0:
            kind, offsets = "forward", (h[i], 2 * h[i])
        elif x[i] + h[i] > 1.0:
            kind, offsets = "backward", (-h[i], -2 * h[i])
        else:
            kind, offsets = "central", (h[i], -h[i])
        for d in offsets:
            p = np.array(x, float)
            p[i] += d
            pts.append(_pt(p))
        kinds.append(kind)
    return pts, kinds


def _fd_column(kind, r0, r1, r2, h):
    if kind == "central":
        return (r1 - r2) / (2 * h)
    if kind == "forward":
        return (-3 * r0 + 4 * r1 - r2) / (2 * h)
    return (3 * r0 - 4 * r1 + r2) / (2 * h)


def stage_edge(config):
    """One-sided z_f derivative at the L-BFGS-B stop, old and new loss."""
    from observable import minimal_loss_raw

    target, white = _gate_inputs(config)
    x = np.asarray(EDGE_POINT)
    out = {"point": list(EDGE_POINT)}
    for h in (0.02, 0.1):
        pts = [_pt(x), _pt(x + [0, h]), _pt(x + [0, 2 * h])]
        res = evaluate_batch(pts)
        old = [minimal_loss_raw(_stats_of(r), target) for r in res]
        new = [float(_wres(r, target, white) @ _wres(r, target, white)) for r in res]
        out[f"h{h}"] = {
            "points": [list(p) for p in pts],
            "J_old": old,
            "J_new": new,
            "dJdz_old": _fd_column("forward", *old, h),
            "dJdz_new": _fd_column("forward", *new, h),
        }
        print(
            f"h={h}: dJ/dz_f old={out[f'h{h}']['dJdz_old']:+.4e} "
            f"new={out[f'h{h}']['dJdz_new']:+.4e}",
            flush=True,
        )
    _save("gate_edge", out)


def _design(config):
    target, white = _gate_inputs(config)
    pts = [(a, b) for a in DESIGN_LEVELS for b in DESIGN_LEVELS]
    J = np.array(
        [float(np.sum(_wres(r, target, white) ** 2)) for r in evaluate_batch(pts)]
    )
    return pts, J.reshape(len(DESIGN_LEVELS), len(DESIGN_LEVELS))


def stage_design(config):
    """5 x 5 design; grid local minima mark the basins (step 12)."""
    target, white = _gate_inputs(config)
    star = tuple(config["theta_star_hat"])
    r_star = _wres(evaluate_batch([star])[0], target, white)
    J_star = float(r_star @ r_star)
    _, J = _design(config)
    n = len(DESIGN_LEVELS)
    minima = []
    for a in range(n):
        for b in range(n):
            nbrs = [
                J[a + da, b + db]
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= a + da < n and 0 <= b + db < n
            ]
            if J[a, b] <= min(nbrs):
                minima.append(
                    {
                        "theta_hat": [DESIGN_LEVELS[a], DESIGN_LEVELS[b]],
                        "J": float(J[a, b]),
                    }
                )
    minima.sort(key=lambda m: m["J"])
    best = np.asarray(minima[0]["theta_hat"])
    in_basin = bool(np.abs(best - np.asarray(star)).max() <= 0.2)
    others = minima[1:]
    well_above = all(m["J"] >= max(2 * J_star, J_star + 1.0) for m in others)
    out = {
        "levels": list(DESIGN_LEVELS),
        "J": J.tolist(),
        "J_star": J_star,
        "local_minima": minima,
        "criterion": "best minimum within one grid step of theta*; every other "
        "grid minimum has J >= max(2 J*, J* + 1)",
        "best_in_star_basin": in_basin,
        "others_well_above": well_above,
        "gate_a_pass": in_basin and well_above,
    }
    out["target"] = GATE["target"]
    _save(_gkey("gate_design"), out)
    print("J(theta*) =", J_star)
    print("rows E_L =", DESIGN_LEVELS, "; columns z_f =", DESIGN_LEVELS)
    for a, row in zip(DESIGN_LEVELS, J):
        print(f"  {a:.1f} " + " ".join(f"{v:10.3e}" for v in row))
    print("local minima", minima)
    print("gate (a) pass", out["gate_a_pass"])


def stage_fine(config):
    """Fine slices at theta* +- 0.05: is the minimum of J at theta*?"""
    target, white = _gate_inputs(config)
    star = np.asarray(config["theta_star_hat"])
    out = {"offsets": list(FINE_OFFSETS), "target": GATE["target"]}
    for i, coord in enumerate(("E_L", "z_f")):
        pts = []
        for d in FINE_OFFSETS:
            p = star.copy()
            p[i] += d
            pts.append(_pt(p))
        J = [float(np.sum(_wres(r, target, white) ** 2)) for r in evaluate_batch(pts)]
        k = int(np.argmin(J))
        out[coord] = {"J": J, "argmin_offset": FINE_OFFSETS[k]}
        print(coord, " ".join(f"{d:+.2f}:{j:.3e}" for d, j in zip(FINE_OFFSETS, J)))
        print(coord, "minimum at offset", FINE_OFFSETS[k], flush=True)
    _save(_gkey("gate_fine"), out)


def solve_dogbox(config, x0, label, h, free=(0, 1)):
    """Bounded trust-region least squares (dogbox) on r_w with the batched
    FD Jacobian; the centre run is the cached residual call.

    `free` lists the optimized coordinates; the others stay at theta*.
    x0 is the full 2-vector.
    """
    from scipy.optimize import least_squares

    target, white = _gate_inputs(config)
    star = np.asarray(config["theta_star_hat"], float)
    free = list(free)
    history = []
    t0 = time.perf_counter()

    def full(y):
        x = star.copy()
        x[free] = y
        return x

    def fun(y):
        x = full(y)
        r = _wres(evaluate_batch([_pt(x)])[0], target, white)
        history.append(
            {
                "kind": "residual",
                "theta_hat": list(_pt(x)),
                "J": float(r @ r),
                "elapsed_s": time.perf_counter() - t0,
            }
        )
        print(f"[{label}] f x={np.round(x, 6)} J={r @ r:.5e}", flush=True)
        return r

    def jac(y):
        x = np.asarray(_pt(full(y)))
        pts, kinds = _stencil_points(x, h)
        pts = [q for i in free for q in pts[2 * i : 2 * i + 2]]
        kinds = [kinds[i] for i in free]
        res = evaluate_batch([_pt(x), *pts])
        r0 = _wres(res[0], target, white)
        cols = []
        for k, i in enumerate(free):
            r1 = _wres(res[1 + 2 * k], target, white)
            r2 = _wres(res[2 + 2 * k], target, white)
            cols.append(_fd_column(kinds[k], r0, r1, r2, h[i]))
        jacobian = np.stack(cols, axis=1)
        grad = 2 * jacobian.T @ r0
        history.append(
            {
                "kind": "jacobian",
                "theta_hat": x.tolist(),
                "stencil": kinds,
                "J": float(r0 @ r0),
                "grad_hat": grad.tolist(),
                "elapsed_s": time.perf_counter() - t0,
            }
        )
        print(f"[{label}] J x={x} grad={grad} stencil={kinds}", flush=True)
        return jacobian

    res = least_squares(
        fun,
        np.asarray(x0, float)[free],
        jac=jac,
        bounds=([0.0] * len(free), [1.0] * len(free)),
        method="dogbox",
        x_scale="jac",
        loss="linear",
        **DOGBOX,
    )
    final = full(res.x)
    err = np.abs(final - star)
    G = res.jac.T @ res.jac
    return {
        "free": free,
        "start": list(map(float, x0)),
        "final": final.tolist(),
        "J_final": float(2 * res.cost),
        "error": err.tolist(),
        "pass": bool(np.all(err <= 0.02)),
        "G_at_solution": G.tolist(),
        "Ginv_at_solution": np.linalg.inv(G).tolist(),
        "nfev": int(res.nfev),
        "njev": int(res.njev),
        "status": int(res.status),
        "message": str(res.message),
        "wall_s": time.perf_counter() - t0,
        "history": history,
    }


ONE_D_LEVELS = tuple(round(0.05 + 0.1 * k, 2) for k in range(10))


def stage_one(config, coord):
    """One-parameter recovery: the other parameter stays at theta*.

    A 10-point design at 0.05, 0.15, ..., 0.95 (theta* is not on it), then
    dogbox from the 2 best design points at least 0.2 apart, and from the
    optimizer start farthest from theta* in this coordinate (report only).
    """
    i = {"E_L": 0, "z_f": 1}[coord]
    target, white = _gate_inputs(config)
    star = np.asarray(config["theta_star_hat"], float)

    def point(v):
        x = star.copy()
        x[i] = v
        return x

    res = evaluate_batch([_pt(point(v)) for v in ONE_D_LEVELS])
    J = np.array([float(np.sum(_wres(r, target, white) ** 2)) for r in res])
    print(
        coord, "design", " ".join(f"{v:.2f}:{j:.1f}" for v, j in zip(ONE_D_LEVELS, J))
    )
    starts = []
    for k in np.argsort(J):
        v = ONE_D_LEVELS[k]
        if all(abs(v - u) >= 0.2 for u in starts):
            starts.append(v)
        if len(starts) == 2:
            break
    far = max(
        np.asarray(config["optimizer_starts_hat"])[:, i], key=lambda v: abs(v - star[i])
    )
    key = _gkey(f"gate_one_{coord}")
    data = json.loads(RESULT_FILE.read_text())
    out = data.get(key, {"runs": {}})
    out.update(
        {
            "design_levels": list(ONE_D_LEVELS),
            "design_J": J.tolist(),
            "settings": _settings(H_GATE),
            "target": GATE["target"],
        }
    )
    for tag, v in [
        *(("design", u) for u in starts[: GATE["starts"]]),
        ("farthest", far),
    ]:
        name = f"{tag}_{v:.3f}"
        if name in out["runs"]:
            continue
        out["runs"][name] = solve_dogbox(
            config, point(v), f"{coord}:{name}", H_GATE, free=(i,)
        )
        _save(key, out)
        r = out["runs"][name]
        print(
            f"{coord}:{name}: final={r['final'][i]:.4f} err={r['error'][i]:.4f} "
            f"J={r['J_final']:.4e} nfev={r['nfev']} njev={r['njev']} wall={r['wall_s']:.0f}s",
            flush=True,
        )
    design_runs = [r for n, r in out["runs"].items() if n.startswith("design")]
    best = min(design_runs, key=lambda r: r["J_final"])
    out["best"] = {k: best[k] for k in ("start", "final", "J_final", "error", "pass")}
    out["pass"] = best["pass"]
    _save(key, out)


def _settings(h):
    return {
        "method": "dogbox",
        "bounds": [0, 1],
        "x_scale": "jac",
        **DOGBOX,
        "fd_step": list(h),
        "loss": "whitened, Section 2.4",
    }


def stage_dogbox(config):
    """Local solves from the 2 or 3 best well-separated design points."""
    data = json.loads(RESULT_FILE.read_text())
    J = np.asarray(data[_gkey("gate_design")]["J"])
    order = np.argsort(J, axis=None)
    starts = []
    for flat in order:
        a, b = np.unravel_index(flat, J.shape)
        p = np.array([DESIGN_LEVELS[a], DESIGN_LEVELS[b]])
        if all(np.abs(p - q).max() >= 0.4 for q in starts):
            starts.append(p)
        if len(starts) == GATE["starts"]:
            break
    h = H_GATE
    out = data.get(_gkey("gate_dogbox"), {"runs": {}})
    out["target"] = GATE["target"]
    out["settings"] = _settings(h)
    for p in starts:
        key = f"design_{p[0]:.1f}_{p[1]:.1f}"
        if key in out["runs"]:
            continue
        out["runs"][key] = solve_dogbox(config, p, key, h)
        _save(_gkey("gate_dogbox"), out)
        r = out["runs"][key]
        print(
            f"{key}: final={np.round(r['final'], 4)} err={np.round(r['error'], 4)} "
            f"J={r['J_final']:.4e} pass={r['pass']} nfev={r['nfev']} "
            f"njev={r['njev']} wall={r['wall_s']:.0f}s",
            flush=True,
        )
    best = min(out["runs"].values(), key=lambda r: r["J_final"])
    out["best"] = {k: best[k] for k in ("start", "final", "J_final", "error", "pass")}
    out["gate_b_pass"] = best["pass"]
    _save(_gkey("gate_dogbox"), out)


def stage_farthest(config):
    """Dogbox from the optimizer start farthest from theta* (report only)."""
    star = np.asarray(config["theta_star_hat"])
    starts = np.asarray(config["optimizer_starts_hat"])
    far = int(np.argmax(np.abs(starts - star).max(axis=1)))
    h = H_GATE
    out = solve_dogbox(config, starts[far], f"start{far}", h)
    out["settings"] = _settings(h)
    out["start_index"] = far
    out["target"] = GATE["target"]
    _save(_gkey("gate_farthest"), out)
    print(
        f"start{far}: final={np.round(out['final'], 4)} err={np.round(out['error'], 4)} "
        f"J={out['J_final']:.4e} nfev={out['nfev']} njev={out['njev']}"
    )


if __name__ == "__main__":
    cfg = load_config(SIZE)
    args = sys.argv[1:]
    for flag, key, cast in (("--target", "target", str), ("--starts", "starts", int)):
        if flag in args:
            k = args.index(flag)
            GATE[key] = cast(args[k + 1])
            del args[k : k + 2]
    for stage in args:
        stages = {
            "noise": stage_noise,
            "slices": stage_slices,
            "recover": stage_recover,
            "lsq": stage_lsq,
            "edge": stage_edge,
            "design": stage_design,
            "fine": stage_fine,
            "dogbox": stage_dogbox,
            "farthest": stage_farthest,
            "one_E_L": lambda c: stage_one(c, "E_L"),
            "one_z_f": lambda c: stage_one(c, "z_f"),
        }
        stages[stage](cfg)
