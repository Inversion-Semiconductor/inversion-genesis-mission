"""E1b: two-stream (bimodal) f and a nonlinear xi(u); same converters as E1.

f = 0.5 N(u; -1 + 0.3 cos x, s^2) + 0.5 N(u; +1 + 0.3 cos x, s^2).
xi_u = tanh(u / 0.3) (1 + 0.5 cos x): the two streams move in opposite directions.
Extra observable 'sel_u' = S5((u - 0.5)/0.4) * u, like the beam selection of the loss.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import e1_cold_conversion as e1

def sample(n, sigma, rng):
    x = rng.uniform(0, e1.TWO_PI, n)
    sgn = np.where(rng.uniform(size=n) < 0.5, -1.0, 1.0)
    u = sgn + 0.3 * np.cos(x) + sigma * rng.standard_normal(n)
    w = np.full(n, 1.0 / n)
    return x, u, w, np.zeros(n), np.tanh(u / 0.3) * (1 + 0.5 * np.cos(x))

def s5(t):
    t = np.clip(t, 0, 1); return t**3 * (10 - 15 * t + 6 * t * t)
def ds5(t):
    m = (t > 0) & (t < 1); tt = np.clip(t, 0, 1); return np.where(m, 30 * tt**2 * (1 - tt) ** 2, 0.0)

def lag(x, u, w, xi_x, xi_u):
    o = e1.observables_lagr(x, u, w, xi_x, xi_u)
    t = (u - 0.5) / 0.4
    o["sel_u"] = np.sum(w * xi_u * (ds5(t) / 0.4 * u + s5(t)))
    return o
def wt(x, u, dw):
    o = e1.observables_weight(x, u, dw)
    o["sel_u"] = np.sum(dw * s5((u - 0.5) / 0.4) * u)
    return o

rng = np.random.default_rng(7)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
rows = []
for sigma in (0.3, 0.1, 0.03):
    ref = lag(*sample(4_000_000, sigma, rng))
    est = {k: [] for k in ("L", "A_du0.05", "B_p2", "B_p4", "B_p6")}
    for s in range(8):
        x, u, w, xx, xu = sample(n, sigma, rng)
        est["L"].append(lag(x, u, w, xx, xu))
        est["A_du0.05"].append(wt(x, u, e1.convert_mesh(x, u, w, xx, xu, 0.05)[0]))
        for p in (2, 4, 6):
            est[f"B_p{p}"].append(wt(x, u, e1.convert_moment(x, u, w, xx, xu, p)[0]))
    print(f"sigma={sigma}")
    row = {"sigma": sigma, "ref": ref, "est": {}}
    for obs in ref:
        line = f"  {obs:8s} ref {ref[obs]:+.4f} |"
        for k, lst in est.items():
            v = np.array([d[obs] for d in lst]); row["est"].setdefault(k, {})[obs] = [float(v.mean()), float(v.std(ddof=1))]
            line += f" {k} {v.mean():+.4f}+-{v.std(ddof=1):.4f} |"
        print(line)
    rows.append(row)
json.dump({"N": n, "rows": rows}, open(f"e1b_results_N{n}.json", "w"), indent=1)
