"""Scale dependence of FD secants on the cached P7 T0 solution slice (read only).

Usage: python3 -I slice_scale.py <cache_dir> <p7_jacobian.json>
"""
import json, os, sys
import numpy as np

cache, p7 = sys.argv[1], sys.argv[2]
pts = {}
for f in os.listdir(cache):
    if f.startswith("grad_") and f.endswith(".json"):
        d = json.load(open(os.path.join(cache, f)))
        pts[round(float(d["x"][0] if isinstance(d["x"], list) else d["x"]), 5)] = d
c = json.load(open(os.path.join(cache, "center_0.588000.json")))
print("center keys", list(c.keys())[:20])
xs = np.array(sorted(pts))
print("n slice pts", len(xs), xs.min(), xs.max())
R = np.array([np.array(pts[x]["r_w"], float) for x in xs])
A = np.array([np.array(pts[x]["jac_hat"], float).reshape(-1) for x in xs])
# add center if it has r_w
x0 = 0.588
if "r_w" in c:
    xs = np.append(xs, x0); R = np.vstack([R, np.array(c["r_w"], float)])
    o = np.argsort(xs); xs, R = xs[o], R[o]
sol = json.load(open(p7))["usability"]["slices"]["solution"]
Jsm = np.array(sol["center_columns"]["smoothed"], float)
Jad = np.array(sol["center_columns"]["AD"], float)
i0 = int(np.argmin(abs(xs - x0)))
print("x0 in slice:", xs[i0])
def cosine(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
print("\nCentral secants at x0 = 0.588 from slice points (r_w units per unit theta_hat)")
print(" h      | ||D_h|| | rel diff to J_sm | cos to J_sm | rel diff to D_0.02 | kappa(K=1) vs J_sm")
D02 = None
rows = []
for k in range(1, 21):
    h = 0.0025 * k
    ip, im = i0 + k, i0 - k
    if im < 0 or ip >= len(xs):
        break
    D = (R[ip] - R[im]) / (xs[ip] - xs[im])
    rows.append((h, D))
    if abs(h - 0.02) < 1e-9:
        D02 = D
for h, D in rows:
    rho = np.linalg.norm(D) / np.linalg.norm(Jsm)
    cs = cosine(D, Jsm)
    kap = abs(1 - cs / rho)
    rd02 = np.linalg.norm(D - D02) / np.linalg.norm(D02)
    print(f" {h:.4f} | {np.linalg.norm(D):7.1f} | {np.linalg.norm(D - Jsm)/np.linalg.norm(Jsm):6.3f} | {cs:6.3f} | {rd02:6.3f} | {kap:5.3f}")
# roughness: second differences along the slice, relative to slope scale
d2 = R[2:] - 2 * R[1:-1] + R[:-2]
hstep = 0.0025
print("\nrms second difference per residual (r_w units):", np.round(np.sqrt((d2**2).mean(0)), 4))
print("rms |r(x+h)-r(x)| at h=0.0025:", np.round(np.sqrt(((R[1:]-R[:-1])**2).mean(0)), 4))
print("|J_sm| * h per residual at h=0.0025:", np.round(abs(Jsm) * hstep, 4))
# quadratic+cubic fit residual noise
xx = xs - x0
V = np.vander(xx, 4)
coef, *_ = np.linalg.lstsq(V, R, rcond=None)
res = R - V @ coef
print("cubic-fit residual rms per residual:", np.round(res.std(0), 4))
noise = res.std(0)
# predicted FD noise at step h if roughness is white with that rms: sqrt(2)*sigma/(2h)
for h in (0.0025, 0.005, 0.01, 0.02):
    e = np.linalg.norm(np.sqrt(2) * noise / (2 * h)) / np.linalg.norm(Jsm)
    print(f" white-roughness FD column error at h={h}: {e:.3f}")
# mean AD column over the slice vs global secant
Dglob = (R[-1] - R[0]) / (xs[-1] - xs[0])
Amean = A.mean(0)
Ase = A.std(0) / np.sqrt(len(A))
print("\nmean of AD columns over slice, norm:", np.linalg.norm(Amean), " SE norm:", np.linalg.norm(Ase))
print("global secant over slice, norm:", np.linalg.norm(Dglob), " cos(meanAD, secant):", cosine(Amean, Dglob))
print("J_AD at center norm", np.linalg.norm(Jad), "cos(J_AD,J_sm)", cosine(Jad, Jsm))
