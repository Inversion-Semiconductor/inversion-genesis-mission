"""E2b: calibrate the E2 conversion where the Lagrangian estimator is still exact and low-variance.

For n_f in {2, 4, 6, 8, 10, 12}: Lagrangian (no reset) at N = 4e6 is the reference.
R-v: N = 1e6, M in {32, 64, 128}, T in {1, 2} (resets before n_f), 4 seeds.
Also reports the instantaneous conversion check at the first reset: sum dw phi vs sum w xi.grad phi.
Also reports a filament width proxy: the fraction of mesh cells needed to hold 90% of the particles.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import e2_standard_map_reset as e2

K = 1.5
rng = np.random.default_rng(99)
out = []
for nf in (2, 4, 6, 8, 10, 12):
    ref, _, _, _ = e2.rv_estimate(4_000_000, K, nf, 0, 64, rng)
    row = {"n_f": nf, "lagrangian_ref": ref}
    line = f"n_f={nf:2d} Lagr(4e6) {ref:+.4f} |"
    for M in (32, 64, 128):
        for T in (1, 2):
            v = [e2.rv_estimate(1_000_000, K, nf, T, M, rng)[0] for _ in range(4)]
            row[f"M{M}_T{T}"] = [float(np.mean(v)), float(np.std(v, ddof=1))]
            line += f" M{M} T{T} {np.mean(v):+.4f}+-{np.std(v, ddof=1):.4f} |"
    print(line, flush=True)
    out.append(row)
# occupancy proxy of f at each step (M = 64 and 256)
x, p = e2.init(1_000_000, rng)
occ = []
for k in range(1, 26):
    x, p = e2.step(x, p, K)
    r = {"step": k}
    for M in (64, 256):
        H, _, _ = np.histogram2d(x, p, bins=M, range=[[0, 2*np.pi], [0, 2*np.pi]])
        c = np.sort(H.ravel())[::-1]; cum = np.cumsum(c) / c.sum()
        r[f"cells90_M{M}"] = int(np.searchsorted(cum, 0.9) + 1)
    occ.append(r)
print("occupancy (cells holding 90% of particles):", [(r['step'], r['cells90_M64'], r['cells90_M256']) for r in occ])
json.dump({"K": K, "rows": out, "occupancy": occ}, open("e2b_results.json", "w"), indent=1)
