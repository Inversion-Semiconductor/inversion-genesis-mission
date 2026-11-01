"""Spread of central secants across neighbouring centers on the P7 solution slice.

Usage: python3 -I slice_secant_spread.py <cache_dir> <p7_jacobian.json>
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
print("points", len(xs), "spacing", np.diff(xs)[:3])
print("For each h: secants at every center c with c+-h inside [0.538,0.638], |c-0.588|<=0.02")
print(" h      | n centers | mean ||D_c - D_mean||/||J_sm|| | min cos(D_c,J_sm) | median cos | ||D_mean - J_sm||/||J_sm||")
for k in (1, 2, 3, 4, 6, 8):
    h = 0.0025 * k
    Ds = []
    for i in range(k, len(xs) - k):
        if abs(xs[i] - 0.588) <= 0.0201:
            Ds.append((R[i + k] - R[i - k]) / (xs[i + k] - xs[i - k]))
    Ds = np.array(Ds)
    Dm = Ds.mean(0)
    spread = np.mean(np.linalg.norm(Ds - Dm, axis=1)) / n
    cs = Ds @ Jsm / np.linalg.norm(Ds, axis=1) / n
    print(f" {h:.4f} | {len(Ds):3d} | {spread:6.3f} | {cs.min():6.3f} | {np.median(cs):6.3f} | {np.linalg.norm(Dm-Jsm)/n:6.3f}")
