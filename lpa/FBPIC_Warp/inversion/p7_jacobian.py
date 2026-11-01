"""Phase P7 steps 3, 5, and 6 on Twin-S: cost, Jacobian usability, epsilon.

Usage (from lpa/FBPIC_Warp, CUDA_VISIBLE_DEVICES set; each stage launches
its runs as child processes, one at a time, and caches every result in
`inversion/runs/p7_<code_sha256[:12]>/`, so that a stage resumes):

    python inversion/p7_jacobian.py cost        # step 3
    python inversion/p7_jacobian.py usability   # step 5, about 7 to 9 h
    python inversion/p7_jacobian.py epsilon     # step 6, reduced Twin-S
    python inversion/p7_jacobian.py report      # step 5 analysis only

Step 3 runs a primal EV run and a `value_and_grad` run (residual objective,
T0, K = 1) twice, interleaved, each in its own process, and records the
wall time, the CUDA mempool high-water marks, and the host peak RSS.

Step 5 (plan Phase P7 step 5): `value_and_grad` at every point of three
41-point slices over +-0.05 in theta_hat_E (z_f at z_f*): around the T0
solution (0.588), the false minimum (0.545), and the far field (0.75).
Each point gives r_w and the AD column dr_w/dtheta_hat_E. At the three
centers, the AD run's emission decisions are kept, and central FD runs at
small h follow in the same process: frozen (emissions of the AD run reused
as `emit_mask`) and free (thresholds active). Frozen minus AD measures code
errors; free minus frozen measures the birth-time term (Section 6.3 rule 4).

Analysis: the smoothed column of each residual is the derivative of a
least-squares polynomial (cubic) fitted to that residual over its slice;
central FD from the slice at h = 0.005, 0.01, 0.02 (and 0.05 at the center).
Reported: the column error |J_AD - J_sm| / |J_sm| per point (median, 90th
percentile), the cosine between J_AD and J_sm, and the dogbox fixed-point
shift delta = |(J_AD - J_sm)^T r_w| / |J_AD^T J_sm| at the T0 solution.

Step 6 (Section 5.5 item 4): on the reduced Twin-S (nitrogen in z_c +- 2
um), `value_and_grad` at epsilon = 1e-4, 1e-5, 1e-6, and 0, at the T0
solution, and free central FD at epsilon = 0 (where frozen equals free).
The residual columns are compared with the epsilon = 0 column.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from twin_config import config_sha256, load_config, theta_from_hat  # noqa: E402

SIZE = "twin_s"
TARGET = "T0"
CENTERS = {"solution": 0.588, "false_minimum": 0.545, "far_field": 0.75}
SLICE_HALF_WIDTH = 0.05
SLICE_POINTS = 41
FROZEN_H = (1e-2, 1e-3, 1e-4, 1e-5)
FREE_H = (1e-3, 1e-4, 1e-5)
SLICE_FD_H = (0.005, 0.01, 0.02, 0.05)
EPSILONS = (1e-4, 1e-5, 1e-6, 0.0)
EPSILON_FD_H = (1e-2, 1e-3)
RESULTS = HERE / "results"


def code_dir() -> Path:
    """Cache directory keyed by the simulation and differentiation code."""
    import hashlib

    files = [
        HERE / "forward.py",
        HERE / "twin_config.py",
        HERE / "residual_objective.py",
    ] + sorted((HERE.parent / "fbpic" / "autodiff").glob("*.py"))
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()
    path = HERE / "runs" / f"p7_{digest[:12]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _key(x: float) -> str:
    return f"{x:.6f}"


def _config(size: str):
    if size == "twin_s_reduced":
        from ev_runs import reduced_config

        return reduced_config()
    return load_config(size)


def _begin_kwargs(config, epsilon):
    return {
        "ionization": "ev",
        "ev_species": "nitrogen",
        "ev_target": "elec_from_N",
        "ev_options": {
            "epsilon": config["ev"]["epsilon"] if epsilon is None else epsilon,
            "K_e": config["ev"]["K_e"],
        },
    }


def _theta(config, x):
    star = np.asarray(config["theta_star_hat"], float)
    return theta_from_hat(config, np.array([x, star[1]]))


def _bounds(config):
    return (np.asarray(config["theta"]["lo"]), np.asarray(config["theta"]["hi"]))


def _memory():
    from forward import device_memory

    out = {k: int(v) for k, v in device_memory().items()}
    out["host_peak_rss"] = (
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    )
    return out


# Child-process workers
# ---------------------


def worker_grad(size, x, epsilon, out, keep_decisions=False, contributions=False):
    """value_and_grad at theta_hat_E = x; writes r_w, the AD column, timings.

    Same computation as `driver.value_and_grad` with a residual objective;
    the final state stays available for the contribution diagnostic.
    """
    import cupy

    from builders import twin_builder
    from residual_objective import residual_objective

    from fbpic.autodiff.driver import jvp

    config = _config(size)
    objective = residual_objective(load_config(SIZE), TARGET)
    build = twin_builder(config, ionization="ev")
    lo, hi = _bounds(config)
    t0 = time.perf_counter()
    result = jvp(
        _theta(config, x),
        build,
        ("E_L",),
        config["num_steps"],
        begin_kwargs=_begin_kwargs(config, epsilon),
        norm_every=100,
    )
    value = objective(result.state)
    jac_hat = value.contract(result.tangents)[:, 0] * (hi[0] - lo[0])
    cupy.cuda.runtime.deviceSynchronize()
    wall = time.perf_counter() - t0
    r = np.asarray(value.residuals, float)
    decisions = result.decisions
    births = sum(int(d.emitted_parent_ids.size) for d in decisions)
    record = {
        "size": size,
        "x": x,
        "epsilon": epsilon,
        "r_w": r.tolist(),
        "J": float(r @ r),
        "jac_hat": jac_hat.tolist(),
        "grad_hat": float(2 * jac_hat @ r),
        "births": births,
        "wall_s": wall,
        "time_steps_s": result.info["time_steps_s"],
        "tangent_norms": result.info["tangent_norms"],
        "memory": _memory(),
    }
    if contributions:
        record["contributions"] = particle_contributions(
            value, result.tangents, hi[0] - lo[0]
        )
    if keep_decisions:
        return record, decisions
    Path(out).write_text(json.dumps(record, indent=1) + "\n")
    return record, None


def particle_contributions(value, tangents, scale):
    """How many particles carry each AD residual column (diagnostic).

    The column is a sum over particles: dr_j = sum_i c_ji with
    c_ji = sum_b (dr_j/dbase_b) dbase_b,i. For each residual, reports the
    column, sum |c_ji|, the share of sum |c_ji| held by the top 1, 10, 100,
    and 1000 particles, and the particles needed for 50% and 90%.
    """
    import cupy

    a = value._arrays
    t = tangents.particles[value._species]
    sigma, w, xi = a["sigma"], a["w"], a["xi"]
    ws = w * sigma
    dws = sigma * t["w"][0] + w * value._sigma_slope * t["uz"][0]
    names = ("x", "ux", "y", "uy", "z", "uz")
    per_base = [dws]
    for k, name in enumerate(names):
        per_base.append(dws * xi[k] + ws * t[name][0])
    for k, name in enumerate(names):
        per_base.append(dws * xi[k] * xi[k] + 2.0 * ws * xi[k] * t[name][0])
    out = []
    for j in range(value.dr_dbase.shape[0]):
        c = sum(float(value.dr_dbase[j, b]) * per_base[b] for b in range(13)) * scale
        mag = cupy.sort(cupy.abs(c))[::-1]
        total = float(mag.sum())
        cum = cupy.cumsum(mag) / max(total, 1e-300)
        top = int(cupy.argmax(cupy.abs(c)))
        out.append(
            {
                "column": float(c.sum()),
                "sum_abs": total,
                "top_share": {
                    str(k): float(cum[min(k, cum.size) - 1]) for k in (1, 10, 100, 1000)
                },
                "n_for_50": int(cupy.searchsorted(cum, 0.5)) + 1,
                "n_for_90": int(cupy.searchsorted(cum, 0.9)) + 1,
                "top_particle": {
                    "sigma": float(sigma[top]),
                    "uz": float(xi[5][top]),
                    "zeta": float(xi[4][top]),
                    "dx": float(t["x"][0][top]) * scale,
                    "dux": float(t["ux"][0][top]) * scale,
                    "duz": float(t["uz"][0][top]) * scale,
                    "dw": float(t["w"][0][top]) * scale,
                },
            }
        )
    n_sel = int(cupy.count_nonzero(sigma > 0))
    return {"selected_with_sigma_gt_0": n_sel, "residuals": out}


def primal_residuals(size, x, epsilon, emit_masks=None):
    """r_w of a plain EV run (thresholds active, or frozen by emit_masks)."""
    from builders import twin_builder
    from residual_objective import residual_objective

    from fbpic.autodiff.stepper import begin, step

    config = _config(size)
    objective = residual_objective(load_config(SIZE), TARGET)
    sim, species, _ = twin_builder(config, ionization="ev")(_theta(config, x))
    state = begin(sim, species, **_begin_kwargs(config, epsilon))
    if emit_masks is not None:
        state.emit_masks = emit_masks
    for _ in range(config["num_steps"]):
        step(state)
    return objective(state).residuals


def worker_center(x, out):
    """AD at a slice center, then frozen and free central FD at small h."""
    import cupy

    record, decisions = worker_grad(
        SIZE, x, None, out, keep_decisions=True, contributions=True
    )
    masks = {
        d.iteration: (
            cupy.zeros(0, np.uint64)
            if d.emitted_parent_ids is None
            else d.emitted_parent_ids
        )
        for d in decisions
    }
    del decisions
    fd = {"frozen": {}, "free": {}}
    for mode, hs in (("frozen", FROZEN_H), ("free", FREE_H)):
        for h in hs:
            m = masks if mode == "frozen" else None
            plus = primal_residuals(SIZE, x + h, None, m)
            minus = primal_residuals(SIZE, x - h, None, m)
            fd[mode][str(h)] = {
                "column": ((plus - minus) / (2 * h)).tolist(),
                "J_plus": float(plus @ plus),
                "J_minus": float(minus @ minus),
            }
            print(f"center {x}: {mode} h={h:.0e} done", flush=True)
    record["fd"] = fd
    Path(out).write_text(json.dumps(record, indent=1) + "\n")


def worker_primal_cost(out):
    from forward import run_ev

    config = load_config(SIZE)
    run = run_ev(_theta(config, CENTERS["solution"]), config)
    record = {"wall_s": run["wall_s"], "births": run["births"], "memory": _memory()}
    Path(out).write_text(json.dumps(record, indent=1) + "\n")


def worker_fd_point(size, x, epsilon, out):
    r = primal_residuals(size, x, epsilon)
    Path(out).write_text(json.dumps({"x": x, "r_w": r.tolist()}) + "\n")


# Stages
# ------


def _child(args, out: Path, log: Path):
    if out.exists():
        return json.loads(out.read_text())
    cmd = [sys.executable, str(Path(__file__).resolve()), "_worker", *map(str, args)]
    cmd.append(str(out))
    t0 = time.perf_counter()
    with open(log, "w") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=handle, env=os.environ)
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed: {' '.join(cmd)}; see {log}")
    print(f"{out.name}: {time.perf_counter() - t0:.0f} s", flush=True)
    return json.loads(out.read_text())


def stage_cost():
    d = code_dir()
    rows = []
    for rep in range(2):
        primal = _child(
            ["primal_cost"], d / f"cost_primal_{rep}.json", d / f"cost_primal_{rep}.log"
        )
        grad = _child(
            ["grad", SIZE, CENTERS["solution"], "None"],
            d / f"cost_grad_{rep}.json",
            d / f"cost_grad_{rep}.log",
        )
        rows.append({"primal": primal, "grad": grad})
    best_primal = min(r["primal"]["wall_s"] for r in rows)
    best_grad = min(r["grad"]["time_steps_s"] for r in rows)
    out = {
        "rows": [
            {
                "primal_wall_s": r["primal"]["wall_s"],
                "grad_steps_s": r["grad"]["time_steps_s"],
                "grad_wall_s": r["grad"]["wall_s"],
                "primal_memory": r["primal"]["memory"],
                "grad_memory": r["grad"]["memory"],
                "births": r["grad"]["births"],
            }
            for r in rows
        ],
        "run_equivalents": best_grad / best_primal,
        "J_at_solution": rows[0]["grad"]["J"],
    }
    _save("cost", out)
    print(json.dumps(out, indent=1))


def _slice(center):
    return np.round(
        center + np.linspace(-SLICE_HALF_WIDTH, SLICE_HALF_WIDTH, SLICE_POINTS), 6
    )


def stage_usability():
    d = code_dir()
    for name, c in CENTERS.items():
        _child(["center", c], d / f"center_{_key(c)}.json", d / f"center_{_key(c)}.log")
    for name, c in CENTERS.items():
        for x in _slice(c):
            if abs(x - c) < 1e-9:
                continue
            _child(
                ["grad", SIZE, x, "None"],
                d / f"grad_{_key(x)}.json",
                d / f"grad_{_key(x)}.log",
            )
    stage_report()


def _point_path(d, x, c):
    name = f"center_{_key(x)}.json" if abs(x - c) < 1e-9 else f"grad_{_key(x)}.json"
    return d / name


def _center_summary(center):
    """AD column, frozen and free FD columns, and contributions at a center."""
    jac0 = np.asarray(center["jac_hat"])
    out = {"r_w": center["r_w"], "J": center["J"], "AD": center["jac_hat"], "fd": {}}
    for mode in ("frozen", "free"):
        for h, v in center["fd"][mode].items():
            col = np.asarray(v["column"])
            out["fd"][f"{mode} {h}"] = {
                "column": v["column"],
                "relative_to_AD": float(
                    np.linalg.norm(col - jac0) / np.linalg.norm(jac0)
                ),
            }
    out["contributions"] = center.get("contributions")
    return out


def _load_slice(c):
    d = code_dir()
    xs, r, jac = [], [], []
    for x in _slice(c):
        path = _point_path(d, x, c)
        rec = json.loads(path.read_text())
        xs.append(x)
        r.append(rec["r_w"])
        jac.append(rec["jac_hat"])
    return np.array(xs), np.array(r), np.array(jac)


def smoothed_columns(xs, r, degree=3):
    """Derivative of a least-squares polynomial fit of each residual."""
    t = xs - xs.mean()
    out = np.empty_like(r)
    for j in range(r.shape[1]):
        coef = np.polyfit(t, r[:, j], degree)
        out[:, j] = np.polyval(np.polyder(coef), t)
    return out


def _column_errors(jac_ad, jac_sm):
    err = np.linalg.norm(jac_ad - jac_sm, axis=1) / np.linalg.norm(jac_sm, axis=1)
    cos = np.sum(jac_ad * jac_sm, axis=1) / (
        np.linalg.norm(jac_ad, axis=1) * np.linalg.norm(jac_sm, axis=1)
    )
    return err, cos


def _delta(jac_a, jac_b, r):
    """Dogbox fixed-point shift |(J_a - J_b)^T r| / |J_a^T J_b| (1 parameter)."""
    return float(abs((jac_a - jac_b) @ r) / abs(jac_a @ jac_b))


def stage_report():
    d = code_dir()
    report = {"slices": {}}
    for name, c in CENTERS.items():
        missing = [x for x in _slice(c) if not _point_path(d, x, c).exists()]
        if missing:
            center = json.loads((d / f"center_{_key(c)}.json").read_text())
            report["slices"][name] = {
                "center": c,
                "complete": False,
                "points_done": SLICE_POINTS - len(missing),
                "center_only": _center_summary(center),
            }
            print(f"{name}: {SLICE_POINTS - len(missing)} of {SLICE_POINTS} points")
            continue
        xs, r, jac = _load_slice(c)
        sm = smoothed_columns(xs, r)
        err, cos = _column_errors(jac, sm)
        i0 = int(np.argmin(np.abs(xs - c)))
        spacing = xs[1] - xs[0]
        fd_slice = {}
        for h in SLICE_FD_H:
            k = int(round(h / spacing))
            if i0 - k >= 0 and i0 + k < xs.size:
                fd_slice[str(h)] = (
                    (r[i0 + k] - r[i0 - k]) / (2 * (xs[i0 + k] - xs[i0]))
                ).tolist()
        center = json.loads((d / f"center_{_key(c)}.json").read_text())
        jac0, sm0, r0 = jac[i0], sm[i0], r[i0]
        fd_rows = {}
        for mode in ("frozen", "free"):
            for h, v in center["fd"][mode].items():
                col = np.asarray(v["column"])
                fd_rows[f"{mode} {h}"] = {
                    "relative_to_AD": float(
                        np.linalg.norm(col - jac0) / np.linalg.norm(jac0)
                    ),
                    "relative_to_smoothed": float(
                        np.linalg.norm(col - sm0) / np.linalg.norm(sm0)
                    ),
                }
        for h, col in fd_slice.items():
            col = np.asarray(col)
            fd_rows[f"slice free {h}"] = {
                "relative_to_AD": float(
                    np.linalg.norm(col - jac0) / np.linalg.norm(jac0)
                ),
                "relative_to_smoothed": float(
                    np.linalg.norm(col - sm0) / np.linalg.norm(sm0)
                ),
            }
        J = np.sum(r * r, axis=1)
        report["slices"][name] = {
            "center": c,
            "complete": True,
            "center_only": _center_summary(center),
            "x": xs.tolist(),
            "J": J.tolist(),
            "column_error": {
                "median": float(np.median(err)),
                "p90": float(np.percentile(err, 90)),
                "max": float(err.max()),
                "per_point": err.tolist(),
            },
            "cosine": {
                "median": float(np.median(cos)),
                "p10": float(np.percentile(cos, 10)),
                "min": float(cos.min()),
                "per_point": cos.tolist(),
            },
            "center_columns": {"AD": jac0.tolist(), "smoothed": sm0.tolist()},
            "delta_AD_vs_smoothed": _delta(jac0, sm0, r0),
            "fd_at_center": fd_rows,
            "fd_columns_at_center": {
                **{
                    f"{m} {h}": center["fd"][m][h]["column"]
                    for m in ("frozen", "free")
                    for h in center["fd"][m]
                },
                **{f"slice free {h}": v for h, v in fd_slice.items()},
            },
            "per_residual_error_at_center": (
                np.abs(jac0 - sm0) / np.maximum(np.abs(sm0), 1e-12)
            ).tolist(),
            "grad_AD_center": float(2 * jac0 @ r0),
            "grad_smoothed_center": float(2 * sm0 @ r0),
        }
        s = report["slices"][name]
        print(
            f"{name} ({c}): column error median {s['column_error']['median']:.3f} "
            f"p90 {s['column_error']['p90']:.3f}; cosine median {s['cosine']['median']:.4f} "
            f"min {s['cosine']['min']:.4f}; delta {s['delta_AD_vs_smoothed']:.2e}"
        )
        for k, v in fd_rows.items():
            print(
                f"   {k:18s} vs AD {v['relative_to_AD']:.2e}  vs smoothed {v['relative_to_smoothed']:.2e}"
            )
    sol = report["slices"]["solution"]
    if not sol["complete"]:
        _save("usability", report)
        return
    xs = np.array(sol["x"])
    J = np.array(sol["J"])
    i_min = int(np.argmin(J))
    report["solution_slice_min"] = {"x": float(xs[i_min]), "J": float(J[i_min])}
    _save("usability", report)


def stage_epsilon():
    d = code_dir()
    x = CENTERS["solution"]
    rows = {}
    for eps in EPSILONS:
        rows[str(eps)] = _child(
            ["grad", "twin_s_reduced", x, eps],
            d / f"eps_grad_{eps:g}.json",
            d / f"eps_grad_{eps:g}.log",
        )
    fd = {}
    for h in EPSILON_FD_H:
        plus = _child(
            ["fd", "twin_s_reduced", x + h, 0.0],
            d / f"eps0_fd_{_key(x + h)}.json",
            d / f"eps0_fd_{_key(x + h)}.log",
        )
        minus = _child(
            ["fd", "twin_s_reduced", x - h, 0.0],
            d / f"eps0_fd_{_key(x - h)}.json",
            d / f"eps0_fd_{_key(x - h)}.log",
        )
        fd[str(h)] = (
            (np.asarray(plus["r_w"]) - np.asarray(minus["r_w"])) / (2 * h)
        ).tolist()
    ref = np.asarray(rows["0.0"]["jac_hat"])
    r_ref = np.asarray(rows["0.0"]["r_w"])
    out = {"x": x, "columns": {}, "fd_epsilon0": fd, "summary": {}}
    for eps, rec in rows.items():
        col = np.asarray(rec["jac_hat"])
        out["columns"][eps] = col.tolist()
        g = 2 * col @ np.asarray(rec["r_w"])
        g_ref = 2 * ref @ r_ref
        out["summary"][eps] = {
            "column_relative_to_eps0": float(
                np.linalg.norm(col - ref) / np.linalg.norm(ref)
            ),
            "cosine_to_eps0": float(
                col @ ref / (np.linalg.norm(col) * np.linalg.norm(ref))
            ),
            "grad_hat": float(g),
            "grad_relative_to_eps0": float(abs(g - g_ref) / abs(g_ref)),
            "births": rec["births"],
            "J": rec["J"],
            "wall_s": rec["wall_s"],
            "memory": rec["memory"],
        }
        print(f"eps={eps}: {out['summary'][eps]}")
    for h, col in fd.items():
        col = np.asarray(col)
        out["summary"][f"fd eps0 h={h}"] = {
            "column_relative_to_AD_eps0": float(
                np.linalg.norm(col - ref) / np.linalg.norm(ref)
            ),
        }
        print(f"FD eps=0 h={h}: vs AD eps=0 {out['summary'][f'fd eps0 h={h}']}")
    _save("epsilon", out)


def _save(section, value):
    path = RESULTS / "p7_jacobian.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[section] = value
    data["config_sha256"] = config_sha256(load_config(SIZE))
    data["code_dir"] = code_dir().name
    path.write_text(json.dumps(data, indent=1) + "\n")


def _worker(args):
    kind = args[0]
    if kind == "grad":
        eps = None if args[3] == "None" else float(args[3])
        worker_grad(args[1], float(args[2]), eps, args[4])
    elif kind == "center":
        worker_center(float(args[1]), args[2])
    elif kind == "primal_cost":
        worker_primal_cost(args[1])
    elif kind == "fd":
        worker_fd_point(args[1], float(args[2]), float(args[3]), args[4])
    else:
        raise ValueError(kind)


if __name__ == "__main__":
    if sys.argv[1] == "_worker":
        _worker(sys.argv[2:])
    else:
        {
            "cost": stage_cost,
            "usability": stage_usability,
            "epsilon": stage_epsilon,
            "report": stage_report,
        }[sys.argv[1]]()
