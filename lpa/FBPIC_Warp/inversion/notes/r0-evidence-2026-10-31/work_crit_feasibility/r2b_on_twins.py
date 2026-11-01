"""Apply the R2 test (b) regularity rule to the Twin-S P7 solution slice.

Rule (sec_tests 5.3 b): D_0.01, D_0.02, D_0.04 agree within 2 combined SE + 10%
for at least 5 of 6 observables. Twin-S has one deterministic loading, so SE is
proxied by the spread of the secant over 17 neighbouring centers (|c-0.588|<=0.02).
Usage: python3 -I r2b_on_twins.py <cache_dir>
"""
import json, os, sys
import numpy as np
cache = sys.argv[1]
pts = {}
for f in os.listdir(cache):
    if (f.startswith("grad_") or f.startswith("center_0.588")) and f.endswith(".json"):
        d = json.load(open(os.path.join(cache, f)))
        x = d["x"][0] if isinstance(d["x"], list) else d["x"]
        pts[round(float(x), 5)] = np.array(d["r_w"], float)
xs = np.array(sorted(pts)); R = np.array([pts[x] for x in xs])
i0 = int(np.argmin(abs(xs - 0.588)))
names = ["logQ", "<x>", "<ux>", "<zeta>", "<uz>", "log sx", "log sux", "log sy", "log suy", "log szeta", "log spread"]
def D(i, k):
    return (R[i + k] - R[i - k]) / (xs[i + k] - xs[i - k])
out = {}
for h, k in ((0.01, 4), (0.02, 8), (0.04, 16)):
    out[h] = D(i0, k)
# local spread proxy for SE: std over centers within +-0.01 (k=4 only fits fully)
sd = {}
for h, k in ((0.01, 4), (0.02, 8)):
    Ds = [D(i, k) for i in range(k, len(xs) - k) if abs(xs[i] - 0.588) <= 0.0101 and xs[i - k] >= 0.4949]
    sd[h] = np.std(Ds, axis=0)
print(" residual    | D_0.01 | D_0.02 | D_0.04 | max rel diff | spread(0.01) | pass(10%+2sd)")
npass = 0
for j, nm in enumerate(names):
    a, b, c = out[0.01][j], out[0.02][j], out[0.04][j]
    rel = max(abs(a - b), abs(b - c), abs(a - c)) / max(abs(a), abs(b), abs(c))
    tol = 0.10 * max(abs(a), abs(b), abs(c)) + 2 * np.sqrt(2) * sd[0.01][j]
    ok = max(abs(a - b), abs(b - c), abs(a - c)) <= tol
    npass += ok
    print(f" {nm:11s} | {a:7.1f} | {b:7.1f} | {c:7.1f} | {rel:5.2f} | {sd[0.01][j]:6.1f} | {ok}")
print("observables passing:", npass, "of 11")
print("column rel diff D_0.01 vs D_0.04:", np.linalg.norm(out[0.01]-out[0.04])/np.linalg.norm(out[0.02]))
