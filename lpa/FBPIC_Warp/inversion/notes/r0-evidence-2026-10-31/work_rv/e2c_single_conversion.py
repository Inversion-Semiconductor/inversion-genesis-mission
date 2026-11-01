"""E2c: error of ONE conversion at step t_r, on the standard map, versus how much f has deformed.

Particles start as in E2. Steps 1..t_r: exact Lagrangian tangent. At t_r: convert xi -> dw, set xi = 0.
Steps t_r+1..t_r+2: exact Lagrangian tangent again. Observable at n_f = t_r + 2.
Reference: Lagrangian (no conversion) on the same particles (exact derivative of the sample),
so the difference is the conversion error alone (paired, no sampling noise in the comparison).
Converters: (A) periodic CIC mesh M x M (E2); (B) global polynomial basis in the UNWRAPPED
coordinates (x - xbar, p - pbar)/scale up to degree d, Galerkin: G c = b, dw = w psi.c.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import e2_standard_map_reset as e2

def step_unwrapped(x, p, dx, dp, K):
    s, c = np.sin(x), np.cos(x)
    dp = dp + s + K * c * dx
    p = p + K * s
    dx = dx + dp
    x = x + p
    return x, p, dx, dp

def monomials(a, b, d):
    cols, dcols_a, dcols_b = [], [], []
    for i in range(d + 1):
        for j in range(d + 1 - i):
            cols.append(a**i * b**j)
            dcols_a.append(i * a ** max(i - 1, 0) * b**j if i else np.zeros_like(a))
            dcols_b.append(j * a**i * b ** max(j - 1, 0) if j else np.zeros_like(a))
    return np.stack(cols, 1), np.stack(dcols_a, 1), np.stack(dcols_b, 1)

def convert_poly(x, p, w, dx, dp, d):
    xb, pb = np.sum(w * x), np.sum(w * p)
    sx, sp = np.sqrt(np.sum(w * (x - xb) ** 2)), np.sqrt(np.sum(w * (p - pb) ** 2))
    a, b = (x - xb) / sx, (p - pb) / sp
    Psi, Da, Db = monomials(a, b, d)
    G = Psi.T @ (w[:, None] * Psi)
    rhs = (w * dx / sx) @ Da + (w * dp / sp) @ Db
    c = np.linalg.lstsq(G, rhs, rcond=1e-12)[0]
    return w * (Psi @ c)

K = 1.5
N = 1_000_000
rng = np.random.default_rng(5)
rows = []
for t_r in (1, 2, 3, 4, 5, 6, 8):
    x, p = e2.init(N, rng)
    x = np.where(x > np.pi + 2.0, x - 2 * np.pi, x)  # unwrap the initial blob around (2, 1)
    p = np.where(p > np.pi + 1.0, p - 2 * np.pi, p)
    w = np.full(N, 1.0 / N)
    dx, dp = np.zeros(N), np.zeros(N)
    for _ in range(t_r):
        x, p, dx, dp = step_unwrapped(x, p, dx, dp, K)
    convs = {}
    for M in (64, 128, 256):
        convs[f"mesh{M}"] = e2.convert(x % (2*np.pi), p % (2*np.pi), w, dx, dp, M)[0]
    for d in (2, 4, 6, 8):
        convs[f"poly{d}"] = convert_poly(x, p, w, dx, dp, d)
    # transport two more steps: Lagrangian reference keeps (dx, dp); converted runs restart from 0
    xr, pr, dxr, dpr = x.copy(), p.copy(), dx.copy(), dp.copy()
    x0, p0, dx0, dp0 = x.copy(), p.copy(), np.zeros(N), np.zeros(N)
    for _ in range(2):
        xr, pr, dxr, dpr = step_unwrapped(xr, pr, dxr, dpr, K)
        x0, p0, dx0, dp0 = step_unwrapped(x0, p0, dx0, dp0, K)
    gx, gp = e2.grad_phi(xr, pr)
    ref = np.sum(w * (gx * dxr + gp * dpr))
    lag_after = np.sum(w * (gx * dx0 + gp * dp0))
    ph = e2.phi(xr, pr)
    row = {"t_r": t_r, "ref": float(ref)}
    # deformation proxy: aspect ratio of the covariance ellipse of the blob (unwrapped)
    C = np.cov(np.stack([x, p]))
    ev = np.linalg.eigvalsh(C)
    row["aspect"] = float(np.sqrt(ev[-1] / ev[0]))
    row["sigma_min"] = float(np.sqrt(ev[0]))
    line = f"t_r={t_r} aspect {row['aspect']:7.1f} sigma_min {row['sigma_min']:.3f} ref {ref:+.4f} |"
    for k, dw in convs.items():
        val = float(np.sum(dw * ph) + lag_after)
        row[k] = val
        line += f" {k} {(val - ref) / abs(ref):+.3f} |"
    print(line, flush=True)
    rows.append(row)
json.dump({"K": K, "N": N, "rows": rows, "note": "entries are absolute values; printed numbers are relative errors"}, open("e2c_results.json", "w"), indent=1)
