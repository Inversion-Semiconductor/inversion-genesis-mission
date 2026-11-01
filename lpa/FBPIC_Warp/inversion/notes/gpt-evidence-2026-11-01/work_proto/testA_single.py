"""Test A3: single GPT projection at step t_r on the standard map (blob f0), then 2 exact steps.
At this short horizon the Lagrangian estimator is accurate, so the paired difference
(GPT - Lagrangian) on the same particles isolates the error of one projection (truncation + sampling).
Same particles, N = 1e6, 4 seeds. Compare with R0 E2c (R-v weight conversion), which used the same setup.
"""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from testA_stdmap import init, step_tan, contract, proj, K0

N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 1_000_000
out = {}
for tr in range(1, 9):
    for kind, par in (("four", 2), ("four", 4), ("four", 8), ("bsp", 8), ("bsp", 16), ("bsp", 32), ("bsp", 64)):
        lag, rel = [], []
        for s in range(4):
            rng = np.random.default_rng(700 + s)
            x, p = init(N, rng, "blob"); w = np.full(N, 1.0 / N)
            dx, dp = np.zeros(N), np.zeros(N)
            for n in range(tr):
                x, p, dx, dp = step_tan(x, p, dx, dp, K0)
            qx, qp = proj(kind, par, x, p, dx, dp, w)
            xa, pa, dxa, dpa = x.copy(), p.copy(), dx.copy(), dp.copy()
            for n in range(2):
                xa, pa, dxa, dpa = step_tan(xa, pa, dxa, dpa, K0)
                x, p, qx, qp = step_tan(x, p, qx, qp, K0)
            L = contract(xa, pa, dxa, dpa, w); G = contract(x, p, qx, qp, w)
            lag.append(L); rel.append((G - L) / abs(L))
        key = f"tr{tr}_{kind}{par}"
        out[key] = dict(lag=float(np.mean(lag)), lag_sd=float(np.std(lag)), rel=float(np.mean(rel)), rel_sd=float(np.std(rel)))
        print(f"t_r={tr} {kind}{par:<3d} Lagrangian {np.mean(lag):+.4f} (sd {np.std(lag):.4f})  GPT rel. change {np.mean(rel):+.3f} (sd {np.std(rel):.3f})", flush=True)
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "out/sm_single.json", "w"), indent=1)
