"""Evaluate the D32 fixed-point shift and proposed guards on the P7 columns (read-only)."""
import json, sys
import numpy as np
d = json.load(open(sys.argv[1]))["usability"]["slices"]

def metrics(JA, JR, r):
    JA, JR, r = map(np.asarray, (JA, JR, r))
    delta = abs((JA - JR) @ r) / abs(JA @ JR)
    cos = JA @ JR / np.linalg.norm(JA) / np.linalg.norm(JR)
    ratio = np.linalg.norm(JA) / np.linalg.norm(JR)
    ecol = np.linalg.norm(JA - JR) / np.linalg.norm(JR)
    kappa = abs(1 - (JA @ JR) / (JA @ JA))       # GN contraction, K = 1
    step_ratio = ((JA @ r) / (JA @ JA)) / ((JR @ r) / (JR @ JR))
    return dict(delta=delta, cos=cos, norm_ratio=ratio, col_err=ecol, kappa=kappa, step_ratio=step_ratio,
                r_over_JR=np.linalg.norm(r) / np.linalg.norm(JR))

s = d["solution"]; c = s["center_only"]; r = c["r_w"]
JR = s["center_columns"]["smoothed"]
print("solution 0.588, reference = smoothed column")
for name, JA in [("AD", c["AD"])] + [(k, v) for k, v in s["fd_columns_at_center"].items()]:
    m = metrics(JA, JR, r); print(f"  {name:18s}", " ".join(f"{k}={v:.3g}" for k, v in m.items()))
fm = d["false_minimum"]["center_only"]
JRf = fm["fd"]["frozen 0.01"]["column"]
print("false minimum 0.545, reference = frozen FD h=0.01 (smoothed not complete)")
for name in ["AD"]:
    m = metrics(fm["AD"], JRf, fm["r_w"]); print(f"  {name:18s}", " ".join(f"{k}={v:.3g}" for k, v in m.items()))
m = metrics(fm["fd"]["frozen 0.001"]["column"], JRf, fm["r_w"]); print("  FD 1e-3 vs FD 1e-2", " ".join(f"{k}={v:.3g}" for k, v in m.items()))
# random-direction column of AD magnitude: how often does delta pass?
rng = np.random.default_rng(0); passes = 0; n = 20000
for _ in range(n):
    JA = rng.standard_normal(11) * 5e4
    if metrics(JA, JR, r)["delta"] <= 0.005: passes += 1
print(f"random 11-vector columns passing delta<=0.005 at the solution: {passes/n:.3f}")

# null test: random columns, random direction and log-uniform magnitude
n = 200000; passd = passg = passboth = 0
JRn = np.asarray(JR); rr = np.asarray(r)
for _ in range(n):
    JA = rng.standard_normal(11); JA *= np.linalg.norm(JRn) * 10 ** rng.uniform(-2, 3) / np.linalg.norm(JA)
    m = metrics(JA, JRn, rr)
    a = m["delta"] <= 0.005; b = (m["kappa"] <= 0.5) and (m["cos"] >= 0.9)
    passd += a; passg += b; passboth += a and b
print(f"null random columns: delta-only pass {passd/n:.4f}; kappa<=0.5 & cos>=0.9 pass {passg/n:.2e}; all {passboth/n:.2e}")
