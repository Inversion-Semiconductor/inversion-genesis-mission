"""Guarded-gate metrics (sec_tests 6.3) of one-sided FD columns on the P7 solution slice.

Usage: python3 -I onesided_gate.py <cache_dir> <p7_jacobian.json>
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
xs = np.array(sorted(pts)); R = np.array([pts[x] for x in xs])
sol = json.load(open(p7))["usability"]["slices"]["solution"]
Jsm = np.array(sol["center_columns"]["smoothed"], float)
i0 = int(np.argmin(abs(xs - 0.588)))
rw = R[i0]
def metrics(D):
    rho = np.linalg.norm(D) / np.linalg.norm(Jsm)
    cs = D @ Jsm / np.linalg.norm(D) / np.linalg.norm(Jsm)
    kap = abs(1 - cs / rho)
    delta = abs((D - Jsm) @ rw) / abs(D @ Jsm)
    return cs, rho, kap, delta
print(" stencil              | cos  | rho  | kappa | delta")
for k in (2, 4, 8):
    h = 0.0025 * k
    for name, D in (("forward", (R[i0 + k] - R[i0]) / (xs[i0 + k] - xs[i0])),
                    ("backward", (R[i0] - R[i0 - k]) / (xs[i0] - xs[i0 - k])),
                    ("central", (R[i0 + k] - R[i0 - k]) / (xs[i0 + k] - xs[i0 - k]))):
        cs, rho, kap, de = metrics(D)
        print(f" {name:8s} h={h:.4f}   | {cs:5.3f} | {rho:4.2f} | {kap:5.3f} | {de:.1e}")
