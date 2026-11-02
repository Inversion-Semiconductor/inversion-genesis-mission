"""LR variance from a resolved fast-oscillating force (laser-quiver analog), non-interacting particles.

dv = theta A exp(-x^2/2) g(t) cos(w0 t) dt + sigma dW,  dx = v dt,  g(t) = exp(-((t - 15)/5)^2), theta0 = 1.
Particles start at rest, x ~ N(1, 0.3^2). The net effect is the ponderomotive kick ~ theta^2 A^2 / w0^2.
LR weight increment per step: A exp(-x^2/2) g cos(w0 t) sqrt(dt) xi / sigma (does not shrink with w0).
Observable: <v> at t = 30. Estimators: LR (baseline), pathwise tangent, CRN central FD (h = 0.05).
Usage: python -I quiver_lr.py <sigma> <out.json>
"""
import json
import sys
from multiprocessing import Pool

import numpy as np

A = 1.0; DT = 0.005; TEND = 30.0


def job(a):
    w0, sigma, seed, N = a
    rng = np.random.default_rng(seed)
    x0 = 1.0 + 0.3 * rng.standard_normal(N)
    nt = int(TEND / DT)
    res = {}
    for th, tag in ((1.0, "b"), (1.05, "p"), (0.95, "m")):
        r2 = np.random.default_rng(seed + 10 ** 6)
        x = x0.copy(); v = np.zeros(N); om = np.zeros(N); dx = np.zeros(N); dv = np.zeros(N)
        for n in range(nt):
            t = n * DT
            prof = A * np.exp(-x * x / 2) * np.exp(-((t - 15) / 5) ** 2) * np.cos(w0 * t)
            xi = r2.standard_normal(N)
            if tag == "b":
                om += prof * np.sqrt(DT) * xi / sigma
                dv = dv + (prof + th * prof * (-x) * dx) * DT
                dx = dx + dv * DT
            v = v + th * prof * DT + sigma * np.sqrt(DT) * xi
            x = x + v * DT
        res[tag] = float(v.mean())
        if tag == "b":
            res["lr"] = float(np.mean((v - v.mean()) * om)); res["tan"] = float(dv.mean())
            res["omrms"] = float(np.sqrt(np.mean(om ** 2)))
    res["fd"] = (res["p"] - res["m"]) / 0.1
    return w0, res


if __name__ == "__main__":
    sigma = float(sys.argv[1])
    cases = [(w0, sigma, 100 + s, 100000) for w0 in (2.0, 4.0, 8.0, 16.0) for s in range(6)]
    with Pool(24) as p:
        out = p.map(job, cases)
    summ = {}
    for w0 in (2.0, 4.0, 8.0, 16.0):
        rows = [r for w, r in out if w == w0]
        d = {}
        for k in ("lr", "tan", "fd", "omrms"):
            a = np.array([r[k] for r in rows])
            d[k] = (float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size)))
        summ[f"{w0:g}"] = d
        print(f"w0 {w0:4g}: tan {d['tan'][0]:+.3e} +- {d['tan'][1]:.1e}  fd {d['fd'][0]:+.3e} +- {d['fd'][1]:.1e}  "
              f"lr {d['lr'][0]:+.3e} +- {d['lr'][1]:.1e}  omrms {d['omrms'][0]:.2f}  "
              f"LR rel SE (6x1e5 particles) {d['lr'][1] / abs(d['tan'][0]):.2f}")
    json.dump(dict(sigma=sigma, summary=summ), open(sys.argv[2], "w"), indent=1)
