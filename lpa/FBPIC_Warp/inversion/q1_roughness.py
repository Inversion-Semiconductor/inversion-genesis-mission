"""Track R note Q1: does the Twin-S roughness fall with the particle count?

Usage (from lpa/FBPIC_Warp, CUDA_VISIBLE_DEVICES set; each run is a child
process, cached in `inversion/runs/q1/`, so that a stage resumes):

    python inversion/q1_roughness.py scan 1     # particle multiplier m = 1
    python inversion/q1_roughness.py scan 4     # m = 4 (p_nr 4, p_nt 16)
    python inversion/q1_roughness.py seeds 1    # injection-seed spread, m = 1
    python inversion/q1_roughness.py seeds 4
    python inversion/q1_roughness.py report

Every run is a free EV run (thresholds active, as an optimizer sees it) at
theta_hat_E = x, z_f at z_f*, and records the whitened T0 residuals r_w at
the snapshot steps SNAPS. The steps before 2000 bracket the plasma exit
(laser at z_hi at step 1520, the bunch at about step 1760); there the
residuals are offsets from the step-2000 target, so only their differences
in theta are meaningful.

`scan` runs the center X0 and X0 +- h for h in H. The roughness amplitude is
A_m(h) = median_j h |D_j(h) - D_j(H[0])|, with D_j(h) the central secant of
residual j. Particle noise predicts A_4 / A_1 = 0.5; continuum roughness
predicts about 1 (note Section 7.1, Q1).

`seeds` repeats X0 and X0 +- H[0] with other injection seeds, so that the
seed spread of the h = H[0] secant measures the particle noise directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from twin_config import load_config, theta_from_hat  # noqa: E402

SIZE = "twin_s"
TARGET = "T0"
X0 = 0.588
H = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
SNAPS = (1200, 1400, 1520, 1640, 1760, 1880, 2000)
EXTRA_SEEDS = (1, 2, 3)
MULTIPLIERS = {1: (2, 8), 4: (4, 16)}
# Convergence check only (stage `points`): label -> (p_nr, p_nt).
EXTRA_RESOLUTIONS = {"2r": (4, 8), "2t": (2, 16), "8": (8, 16), "16r": (16, 8)}
RUNS = HERE / "runs" / "q1"
RESULTS = HERE / "results"


def _config(m):
    config = load_config(SIZE)
    p_nr, p_nt = {**{str(k): v for k, v in MULTIPLIERS.items()}, **EXTRA_RESOLUTIONS}[
        str(m)
    ]
    config["plasma"]["p_nr"], config["plasma"]["p_nt"] = p_nr, p_nt
    return config


def worker_point(m, seed, x, out):
    import cupy

    from builders import twin_builder
    from residual_objective import residual_objective

    from fbpic.autodiff.stepper import begin, step

    config = _config(m)
    objective = residual_objective(load_config(SIZE), TARGET)
    star = np.asarray(config["theta_star_hat"], float)
    theta = theta_from_hat(config, np.array([x, star[1]]))
    kwargs = {} if seed is None else {"seed": seed}
    sim, species, _ = twin_builder(config, ionization="ev", **kwargs)(theta)
    state = begin(
        sim,
        species,
        ionization="ev",
        ev_species="nitrogen",
        ev_target="elec_from_N",
        ev_options={"epsilon": config["ev"]["epsilon"], "K_e": config["ev"]["K_e"]},
    )
    t0 = time.perf_counter()
    snaps = {}
    for it in range(1, config["num_steps"] + 1):
        step(state)
        if it in SNAPS:
            value = objective(state)
            n_sel = int(cupy.count_nonzero(value._arrays["sigma"] > 0.5))
            snaps[str(it)] = {
                "r_w": np.asarray(value.residuals, float).tolist(),
                "n_selected": n_sel,
                "n_elec_from_N": int(state.species["elec_from_N"].Ntot),
            }
    cupy.cuda.runtime.deviceSynchronize()
    record = {
        "m": m,
        "seed": seed,
        "x": x,
        "snaps": snaps,
        "wall_s": time.perf_counter() - t0,
        "pool_bytes_end": int(cupy.get_default_memory_pool().total_bytes()),
    }
    Path(out).write_text(json.dumps(record, indent=1) + "\n")


def _path(m, seed, x):
    tag = "default" if seed is None else f"s{seed}"
    return RUNS / f"m{m}_{tag}_{x:.6f}.json"


def _child(m, seed, x):
    out = _path(m, seed, x)
    if out.exists():
        return json.loads(out.read_text())
    RUNS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "_worker", str(m)]
    cmd += [str(seed), repr(x), str(out)]
    t0 = time.perf_counter()
    with open(out.with_suffix(".log"), "w") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=handle, env=os.environ)
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed: {' '.join(cmd)}")
    print(f"{out.name}: {time.perf_counter() - t0:.0f} s", flush=True)
    return json.loads(out.read_text())


def stage_scan(m):
    _child(m, None, X0)
    for h in H:
        _child(m, None, X0 + h)
        _child(m, None, X0 - h)


def stage_seeds(m):
    for seed in EXTRA_SEEDS:
        for x in (X0, X0 + H[0], X0 - H[0]):
            _child(m, seed, x)


def _r(m, seed, x, snap):
    path = _path(m, seed, x)
    if not path.exists():
        return None
    return np.asarray(json.loads(path.read_text())["snaps"][str(snap)]["r_w"])


def _secant(m, seed, h, snap):
    plus, minus = _r(m, seed, X0 + h, snap), _r(m, seed, X0 - h, snap)
    if plus is None or minus is None:
        return None
    return (plus - minus) / (2 * h)


def stage_report():
    out = {"X0": X0, "H": H, "snaps": SNAPS, "multipliers": MULTIPLIERS}
    for snap in SNAPS:
        row = {}
        amp = {}
        for m in MULTIPLIERS:
            ref = _secant(m, None, H[0], snap)
            if ref is None:
                continue
            a = {}
            for h in H[1:]:
                d = _secant(m, None, h, snap)
                if d is not None:
                    a[h] = float(np.median(h * np.abs(d - ref)))
            amp[m] = a
            seeds = [ref] + [_secant(m, s, H[0], snap) for s in EXTRA_SEEDS]
            seeds = [s for s in seeds if s is not None]
            row[f"m{m}"] = {
                "A": {str(h): v for h, v in a.items()},
                "secant_h0": ref.tolist(),
                "n_seeds": len(seeds),
            }
            if len(seeds) > 1:
                s = np.array(seeds)
                spread = np.linalg.norm(s.std(0, ddof=1)) / np.linalg.norm(s.mean(0))
                row[f"m{m}"]["seed_spread_rel"] = float(spread)
                row[f"m{m}"]["seed_mean"] = s.mean(0).tolist()
        if 1 in amp and 4 in amp:
            row["A4_over_A1"] = {
                str(h): amp[4][h] / amp[1][h]
                for h in amp[1]
                if h in amp[4] and amp[1][h] > 0
            }
        out[str(snap)] = row
        ratio = row.get("A4_over_A1", {})
        spreads = {k: v.get("seed_spread_rel") for k, v in row.items() if k[0] == "m"}
        print(f"step {snap}: A4/A1 {ratio}  seed spread {spreads}")
    path = RESULTS / "q1_roughness.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    if sys.argv[1] == "_worker":
        m, seed, x, out = sys.argv[2:6]
        worker_point(m, None if seed == "None" else int(seed), float(x), out)
    elif sys.argv[1] == "report":
        stage_report()
    elif sys.argv[1] == "points":
        for x in (X0, X0 + H[0], X0 - H[0]):
            _child(sys.argv[2], None, x)
    else:
        {"scan": stage_scan, "seeds": stage_seeds}[sys.argv[1]](int(sys.argv[2]))
