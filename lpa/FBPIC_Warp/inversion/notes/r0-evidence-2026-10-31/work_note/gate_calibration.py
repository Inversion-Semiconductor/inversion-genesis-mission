"""Calibrate the guarded usability gate on the P7 solution slice.

Usage: python3 -I gate_calibration.py <cache_dir>
Reference J_ref(c) = derivative of a per-residual cubic fit over the 41-point
slice (scale 0.05), evaluated at each center c. Candidates per center:
central FD at h, forward FD at h, and the AD column. Reports per-point
kappa = |1 - cos/rho| and cos, as median, 90th percentile, and worst.
"""
import json, os, sys
import numpy as np
cache = sys.argv[1]
pts, ad = {}, {}
for f in os.listdir(cache):
    if (f.startswith("grad_") or f.startswith("center_0.588")) and f.endswith(".json"):
        d = json.load(open(os.path.join(cache, f)))
        x = d["x"][0] if isinstance(d["x"], list) else d["x"]
        x = round(float(x), 5)
        pts[x] = np.array(d["r_w"], float)
        if "jac_hat" in d:
            ad[x] = np.array(d["jac_hat"], float).reshape(11, -1)[:, 0]
xs = np.array(sorted(k for k in pts if k >= 0.5379))
R = np.array([pts[x] for x in xs])
co = [np.polyfit(xs - 0.588, R[:, j], 3) for j in range(11)]
Jref = lambda c: np.array([np.polyval(np.polyder(cj), c - 0.588) for cj in co])

def metrics(JA, Jr):
    cos = JA @ Jr / np.linalg.norm(JA) / np.linalg.norm(Jr)
    rho = np.linalg.norm(JA) / np.linalg.norm(Jr)
    return abs(1 - cos / rho), cos

def report(name, cols):
    k = np.array([m[0] for m in cols]); c = np.array([m[1] for m in cols])
    print(f"{name:24s} n={len(k):2d} kappa med {np.median(k):6.3f} p90 {np.percentile(k,90):6.3f} max {k.max():8.3f} | cos med {np.median(c):6.3f} min {c.min():6.3f}")

cen = [i for i in range(len(xs)) if abs(xs[i] - 0.588) <= 0.0201]
for k in (1, 2, 4, 8):
    h = 0.0025 * k
    report(f"central h={h:.4f}", [metrics((R[i+k]-R[i-k])/(2*h), Jref(xs[i])) for i in cen])
    report(f"forward h={h:.4f}", [metrics((R[i+k]-R[i])/h, Jref(xs[i])) for i in cen])
adc = [metrics(ad[x], Jref(x)) for x in xs if x in ad and abs(x - 0.588) <= 0.0201]
report("AD", adc)
rng = np.random.default_rng(0)
nul = []
for _ in range(20000):
    v = rng.standard_normal(11); v /= np.linalg.norm(v)
    v *= np.linalg.norm(Jref(0.588)) * 10 ** rng.uniform(-2, 3)
    nul.append(metrics(v, Jref(0.588)))
nul = np.array(nul)
print("null: P(kappa<=0.5 and cos>=0.9) =", np.mean((nul[:, 0] <= 0.5) & (nul[:, 1] >= 0.9)))
