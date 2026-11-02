"""Deterministic (sigma = 0) two-stream: FD of J(theta) at several h, quiet start and 4 jittered loadings.

The quiet-start model has one realization. Jittered loadings (positions shifted by U(-0.5, 0.5) x spacing x 1e-3)
show how much a tiny change of the initial state moves the FD secant. Usage: python -I es_det.py <N> <out.json>
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import es1d_lr as M

HS = (0.3, 0.1, 0.03, 0.01, 0.003)


def job(a):
    th, N, jit = a
    if jit >= 0:
        rng = np.random.default_rng(500 + jit)
        x0, v0, W = M.init(N)
        dxj = (rng.random(N) - 0.5) * (M.L / (N // 2)) * 1e-3
        orig = M.init
        M.init = lambda n: ((x0 + dxj) % M.L, v0.copy(), W)
        r = M.run(th, 0.0, N, 0, do_mf=False, do_tan=(th == M.TH0))
        M.init = orig
    else:
        r = M.run(th, 0.0, N, 0, do_mf=False, do_tan=(th == M.TH0))
    return a, r


if __name__ == "__main__":
    N = int(float(sys.argv[1]))
    cases = [(M.TH0 + s * h, N, j) for j in (-1, 0, 1, 2, 3) for h in HS for s in (1, -1)]
    cases += [(M.TH0, N, j) for j in (-1, 0, 1, 2, 3)]
    with Pool(30) as p:
        res = p.map(job, cases)
    R = {(round(a[0], 6), a[2]): r for a, r in res}
    out = {}
    for j in (-1, 0, 1, 2, 3):
        for tk in R[(M.TH0, j)]:
            o = {"tan": R[(M.TH0, j)][tk]["tan"], "J": R[(M.TH0, j)][tk]["J"]}
            for h in HS:
                o[f"fd{h:g}"] = ((np.array(R[(round(M.TH0 + h, 6), j)][tk]["J"]) -
                                  np.array(R[(round(M.TH0 - h, 6), j)][tk]["J"])) / (2 * h)).tolist()
            out[f"j{j}_{tk}"] = o
            print(f"load {j:+d} {tk}: tan {np.round(o['tan'], 4)} " +
                  " ".join(f"fd{h:g} {np.round(o[f'fd{h:g}'], 4)}" for h in HS))
    json.dump(out, open(sys.argv[2], "w"), indent=1)
