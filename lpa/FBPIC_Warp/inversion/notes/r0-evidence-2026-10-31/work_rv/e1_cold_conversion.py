"""E1: bias and variance of the dipole-to-weight conversion for a warm-to-cold beam, 1D-1V.

f(x,u) = (1/2pi) N(u; U(x), sigma^2) on periodic x in [0, 2pi), U(x) = 0.5 cos x.
Displacement field: xi_u = 1 + 0.5 cos x, xi_x = 0 (a smooth momentum kick).
Converters:
  A  paper-style collocated mesh conversion on a 2D (x,u) CIC mesh: P_g deposit,
     dF_g = -div_h P_g, h_g = dF_g / nhat_g, dw_i = sum_g h_g S_g(z_i)  (Chung eqs. 41-42 pattern)
  B  moment-constrained Galerkin conversion: dw_i = w_i sum_{g,a} c_{g,a} S_g(x_i) u_i^a,
     with G c = b, G = sum w psi psi^T, b = sum w xi . grad psi  (degree p in u)
Estimators of d<phi>/dtheta for phi in {u, u^2, cos(x + u t) for t in {0, 5, 20}}:
  L  Lagrangian: sum w xi . grad phi   (exact derivative of the sample)
  W  weight form: sum dw phi
Reference: L with N_ref = 4e6 (smooth xi, low variance).
"""

import json
import sys

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

TWO_PI = 2 * np.pi
NX = 32
TIMES = (0.0, 5.0, 20.0)


def sample(n, sigma, rng):
    x = rng.uniform(0, TWO_PI, n)
    u = 0.5 * np.cos(x) + sigma * rng.standard_normal(n)
    w = np.full(n, 1.0 / n)
    xi_x = np.zeros(n)
    xi_u = 1.0 + 0.5 * np.cos(x)
    return x, u, w, xi_x, xi_u


def observables_lagr(x, u, w, xi_x, xi_u):
    out = {"u": np.sum(w * xi_u), "u2": np.sum(w * 2 * u * xi_u)}
    for t in TIMES:
        # phi = cos(x + u t): grad = -sin(.) (1, t)
        s = -np.sin(x + u * t)
        out[f"cos_t{t:g}"] = np.sum(w * s * (xi_x + t * xi_u))
    return out


def observables_weight(x, u, dw):
    out = {"u": np.sum(dw * u), "u2": np.sum(dw * u * u)}
    for t in TIMES:
        out[f"cos_t{t:g}"] = np.sum(dw * np.cos(x + u * t))
    return out


def cic_x(x):
    dx = TWO_PI / NX
    s = x / dx
    i0 = np.floor(s).astype(np.int64)
    f = s - i0
    return i0 % NX, (i0 + 1) % NX, 1 - f, f, dx


def convert_mesh(x, u, w, xi_x, xi_u, du):
    """Converter A on a CIC (x,u) mesh with spacing (2pi/NX, du)."""
    ix0, ix1, fx0, fx1, dx = cic_x(x)
    ulo = u.min() - 3 * du
    nu = int(np.ceil((u.max() + 3 * du - ulo) / du)) + 1
    su = (u - ulo) / du
    iu0 = np.floor(su).astype(np.int64)
    fu1 = su - iu0
    fu0 = 1 - fu1
    iu1 = iu0 + 1
    shape = (NX, nu)
    combos = [(ix0, iu0, fx0 * fu0), (ix1, iu0, fx1 * fu0), (ix0, iu1, fx0 * fu1), (ix1, iu1, fx1 * fu1)]

    def deposit(vals):
        g = np.zeros(NX * nu)
        for a, b, s in combos:
            g += np.bincount(a * nu + b, weights=s * vals, minlength=NX * nu)
        return g.reshape(shape)

    nhat = deposit(np.ones_like(x))
    px = deposit(w * xi_x)
    pu = deposit(w * xi_u)
    dF = -(np.roll(px, -1, 0) - np.roll(px, 1, 0)) / (2 * dx)
    dF[:, 1:-1] -= (pu[:, 2:] - pu[:, :-2]) / (2 * du)
    # P_g is a weight sum, so dF_g = -div_h P_g is the weight change per mesh cell
    # (delta f_g * dx * du).  h_g = dF_g / nhat_g is the weight change per particle (Chung eq. 42).
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(nhat > 1e-12, dF / nhat, 0.0)
    lost = np.sum(np.abs(dF[nhat <= 1e-12])) / max(np.sum(np.abs(dF)), 1e-300)
    hf = h.ravel()
    dw = np.zeros_like(x)
    for a, b, s in combos:
        dw += s * hf[a * nu + b]
    return dw, lost


def convert_moment(x, u, w, xi_x, xi_u, degree, reg=1e-10):
    """Converter B: Galerkin projection on S_g(x) * ((u - ubar)/s)^a, a = 0..degree."""
    ix0, ix1, fx0, fx1, dx = cic_x(x)
    ubar, us = np.sum(w * u), max(np.sqrt(np.sum(w * (u - np.sum(w * u)) ** 2)), 1e-3)
    v = (u - ubar) / us
    na = degree + 1
    npar = len(x)
    rows, cols, vals = [], [], []
    # psi matrix (N x NX*na), CSR: each particle touches 2 x-nodes * na powers
    pw = np.stack([v**a for a in range(na)], 1)
    dpw = np.stack([a * v ** max(a - 1, 0) / us if a > 0 else np.zeros_like(v) for a in range(na)], 1)
    for ixk, fk, slope in ((ix0, fx0, -1.0 / dx), (ix1, fx1, 1.0 / dx)):
        for a in range(na):
            rows.append(np.arange(npar))
            cols.append(ixk * na + a)
            vals.append(fk * pw[:, a])
    Psi = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(npar, NX * na))
    # b = sum w (xi_x d_x psi + xi_u d_u psi)
    b = np.zeros(NX * na)
    for ixk, fk, slope in ((ix0, fx0, -1.0 / dx), (ix1, fx1, 1.0 / dx)):
        for a in range(na):
            contrib = w * (xi_x * slope * pw[:, a] + xi_u * fk * dpw[:, a])
            b += np.bincount(ixk * na + a, weights=contrib, minlength=NX * na)
    G = (Psi.T @ sps.diags(w) @ Psi).tocsc()
    scale = G.diagonal().max()
    G = G + reg * scale * sps.identity(G.shape[0], format="csc")
    c = spla.spsolve(G, b)
    dw = w * (Psi @ c)
    cond_proxy = None
    return dw, cond_proxy


def run(sigmas, n, seeds, n_ref):
    rng = np.random.default_rng(12345)
    out = {"N": n, "seeds": seeds, "N_ref": n_ref, "rows": []}
    for sigma in sigmas:
        xr = sample(n_ref, sigma, rng)
        ref = observables_lagr(*xr)
        est = {"L": [], "A_du0.05": [], "A_du0.2": [], "B_p1": [], "B_p2": []}
        lost = {"A_du0.05": [], "A_du0.2": []}
        for s in range(seeds):
            x, u, w, xi_x, xi_u = sample(n, sigma, rng)
            est["L"].append(observables_lagr(x, u, w, xi_x, xi_u))
            for du in (0.05, 0.2):
                dw, lo = convert_mesh(x, u, w, xi_x, xi_u, du)
                est[f"A_du{du:g}"].append(observables_weight(x, u, dw))
                lost[f"A_du{du:g}"].append(lo)
            for p in (1, 2):
                dw, _ = convert_moment(x, u, w, xi_x, xi_u, p)
                est[f"B_p{p}"].append(observables_weight(x, u, dw))
        row = {"sigma": sigma, "ref": ref, "est": {}}
        for k, lst in est.items():
            row["est"][k] = {
                obs: {
                    "mean": float(np.mean([d[obs] for d in lst])),
                    "std": float(np.std([d[obs] for d in lst], ddof=1)),
                }
                for obs in ref
            }
        row["lost_fraction"] = {k: float(np.mean(v)) for k, v in lost.items()}
        out["rows"].append(row)
        print(f"sigma={sigma:g}", flush=True)
        for obs in ref:
            line = f"  {obs:9s} ref {ref[obs]:+.4f} |"
            for k in est:
                m, sd = row["est"][k][obs]["mean"], row["est"][k][obs]["std"]
                line += f" {k} {m:+.4f}+-{sd:.4f} |"
            print(line)
        print("  lost dF fraction (A):", row["lost_fraction"])
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    res = run([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0], n, seeds=8, n_ref=4_000_000)
    with open(f"e1_results_N{n}.json", "w") as fh:
        json.dump(res, fh, indent=1)
