"""Separate curvature from roughness in the P7 solution-slice secants.

Usage: python3 -I detrended_spread.py <cache_dir> <p7_jacobian.json>
For each h, compare the central secant at each center c (|c-0.588|<=0.02)
with the derivative of a per-residual polynomial fit (degree 3 and 5) at c.
"""
import json, os, sys
import numpy as np
cache, p7 = sys.argv[1], sys.argv[2]
pts = {}
for f in os.listdir(cache):
    if (f.startswith("grad_") or f.startswith("center_0.588")) and f.endswith(".json"):
        d = json.load(open(os.path.join(cache, f)))
        x = d["x"][0] if isinstance(d["x"], list) else d["x"]
        pts[round(float(x), 5)] = np.array(d["r_w"], float)
xs = np.array(sorted(k for k in pts if k >= 0.5379))
R = np.array([pts[x] for x in xs])
Jsm = np.array(json.load(open(p7))["usability"]["slices"]["solution"]["center_columns"]["smoothed"], float)
n = np.linalg.norm(Jsm)
print("points", len(xs))
for deg in (3, 5):
    co = [np.polyfit(xs - 0.588, R[:, j], deg) for j in range(R.shape[1])]
    def dfit(c):
        return np.array([np.polyval(np.polyder(cj), c - 0.588) for cj in co])
    def fit(c):
        return np.array([np.polyval(cj, c - 0.588) for cj in co])
    res = R - np.array([fit(x) for x in xs])
    print(f"deg {deg}: fit residual rms per residual (whitened units):", np.round(res.std(0), 3))
    cen = [i for i in range(len(xs)) if abs(xs[i] - 0.588) <= 0.0201]
    Dfit = np.array([dfit(xs[i]) for i in cen])
    sp_fit = np.mean(np.linalg.norm(Dfit - Dfit.mean(0), axis=1)) / n
    print(f"  spread of fitted derivative itself over centers: {sp_fit:.3f}")
    for k in (1, 2, 4, 8):
        h = 0.0025 * k
        Ds = np.array([(R[i + k] - R[i - k]) / (xs[i + k] - xs[i - k]) for i in cen])
        # secant of the fit (curvature part) at the same h
        Dsf = np.array([(fit(xs[i + k]) - fit(xs[i - k])) / (xs[i + k] - xs[i - k]) for i in cen])
        rough = np.mean(np.linalg.norm(Ds - Dsf, axis=1)) / n
        print(f"  h={h:.4f}: mean ||D_c - Dfit_c(h)||/||J_sm|| = {rough:.3f}  (roughness only)")
