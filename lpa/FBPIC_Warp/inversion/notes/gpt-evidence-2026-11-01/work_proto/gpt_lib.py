"""Gauge-projected tangent (GPT): weighted least-squares projection of particle tangents
onto gradient fields grad_z psi_k of a finite basis.

project(G-assembly) solves  min_c sum_i w_i |xi_i - sum_k c_k grad psi_k(z_i)|_M^2
with a diagonal metric M (per component scale), and returns the projected tangents.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


# ---------------------------------------------------------------- 1D Fourier on [0,1)
def fourier1d_grads(x, K):
    """Columns: d/dx of sin(2 pi k x)/(2 pi k) and -cos(2 pi k x)/(2 pi k), k=1..K -> cos, sin."""
    k = np.arange(1, K + 1)
    ang = 2 * np.pi * np.outer(x, k)
    return np.hstack([np.cos(ang), np.sin(ang)])


def project_dense(comps, xi_list, w, ridge=1e-12):
    """comps: list of (n x K) arrays, one per phase-space component (gradient components).
    xi_list: list of length-n arrays (same components). Returns projected list and coefs."""
    K = comps[0].shape[1]
    G = np.zeros((K, K)); b = np.zeros(K)
    for B, xi in zip(comps, xi_list):
        Bw = B * w[:, None]
        G += B.T @ Bw
        b += Bw.T @ xi
    G += ridge * np.trace(G) / K * np.eye(K)
    c = np.linalg.solve(G, b)
    return [B @ c for B in comps], c


def project_dense_chunked(make_comps, n, xi_list, w, chunk=200_000, ridge=1e-12):
    """Same as project_dense, but builds the basis in chunks (memory)."""
    G = None; b = None
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        comps = make_comps(sl)
        if G is None:
            K = comps[0].shape[1]; G = np.zeros((K, K)); b = np.zeros(K)
        for B, xi in zip(comps, xi_list):
            Bw = B * w[sl, None]
            G += B.T @ Bw
            b += Bw.T @ xi[sl]
    K = G.shape[0]
    G += ridge * np.trace(G) / K * np.eye(K)
    c = np.linalg.solve(G, b)
    out = [np.empty(n) for _ in xi_list]
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        comps = make_comps(sl)
        for o, B in zip(out, comps):
            o[sl] = B @ c
    return out, c


# ---------------------------------------------------------------- 2D Fourier on torus [0,2pi)^2
def fourier2d_modes(Kf):
    m = [(a, b) for a in range(0, Kf + 1) for b in range(-Kf, Kf + 1) if (a > 0 or b > 0)]
    return np.array(m)


def fourier2d_grads(x, p, modes):
    ph = np.outer(x, modes[:, 0]) + np.outer(p, modes[:, 1])
    c, s = np.cos(ph), np.sin(ph)
    # psi = sin(ph): grad = (a cos, b cos); psi = cos(ph): grad = (-a sin, -b sin)
    a, b = modes[:, 0][None, :], modes[:, 1][None, :]
    gx = np.hstack([a * c, -a * s]); gp = np.hstack([b * c, -b * s])
    return [gx, gp]


# ---------------------------------------------------------------- tensor quadratic B-splines
def _bspline_q(s):
    """Quadratic B-spline weights/derivatives for coordinate s in cell units.
    Returns base index i0 and arrays (3, n) of values and d/ds."""
    i = np.floor(s).astype(np.int64)
    t = s - i
    v = np.stack([0.5 * (1 - t) ** 2, 0.5 + t - t * t, 0.5 * t * t])
    d = np.stack([-(1 - t), 1 - 2 * t, t])
    return i, v, d


def bspline2d_design(x, p, xlo, xhi, Mx, plo, phi, Mp, periodic=False, sx=1.0, sp_=1.0):
    """Sparse design matrices (n x Nb) for d/dx psi and d/dp psi, psi = N_j(x) N_l(p) on
    quadratic B-splines. Non-periodic: node index range is 0..M+1 (M cells plus 2 halo).
    sx, sp_: metric scales (components are divided by them, i.e. normalized coordinates)."""
    hx = (xhi - xlo) / Mx; hp = (phi - plo) / Mp
    ix, vx, dx = _bspline_q((x - xlo) / hx)
    ip, vp, dp = _bspline_q((p - plo) / hp)
    if periodic:
        nx, npn = Mx, Mp
    else:
        nx, npn = Mx + 2, Mp + 2
        ix = np.clip(ix, 0, Mx - 1); ip = np.clip(ip, 0, Mp - 1)
    n = x.size
    rows, cols, gxv, gpv = [], [], [], []
    for a in range(3):
        ja = (ix + a) % nx if periodic else ix + a
        for b_ in range(3):
            lb = (ip + b_) % npn if periodic else ip + b_
            rows.append(np.arange(n)); cols.append(ja * npn + lb)
            gxv.append(dx[a] / hx * vp[b_] * sx); gpv.append(vx[a] * dp[b_] / hp * sp_)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    Nb = nx * npn
    Gx = sp.csr_matrix((np.concatenate(gxv), (rows, cols)), shape=(n, Nb))
    Gp = sp.csr_matrix((np.concatenate(gpv), (rows, cols)), shape=(n, Nb))
    return Gx, Gp


def project_sparse(comps, xi_list, w, ridge=1e-10):
    """comps: sparse (n x Nb) gradient-component matrices; returns projected list, coefs, residual
    fraction (weighted norm of the removed part over the input norm)."""
    W = sp.diags(w)
    G = None; b = None
    for B, xi in zip(comps, xi_list):
        BtW = B.T @ W
        G = BtW @ B if G is None else G + BtW @ B
        b = BtW @ xi if b is None else b + BtW @ xi
    G = G.tocsc()
    dg = G.diagonal()
    active = dg > 1e-14 * max(dg.max(), 1e-300)
    idx = np.where(active)[0]
    Ga = G[idx][:, idx]
    lam = ridge * dg[idx].mean()
    Ga = Ga + lam * sp.identity(idx.size, format="csc")
    c = np.zeros(G.shape[0])
    c[idx] = spl.spsolve(Ga, b[idx])
    out = [B @ c for B in comps]
    num = sum(np.sum(w * (xi - o) ** 2) for xi, o in zip(xi_list, out))
    den = sum(np.sum(w * xi ** 2) for xi in xi_list) + 1e-300
    return out, c, float(num / den)
