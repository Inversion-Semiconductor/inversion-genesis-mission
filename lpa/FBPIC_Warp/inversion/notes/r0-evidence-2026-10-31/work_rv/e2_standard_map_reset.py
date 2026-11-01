"""E2: R-v (periodic projection of Lagrangian tangents into weights) on a chaotic ensemble.

Toy: Chirikov standard map on the torus, p' = p + K sin x, x' = x + p' (mod 2pi).
It is area preserving, so f is transported and weights are constant along orbits,
as for the Vlasov flow in (x, u). Parameter theta = K. No self-consistent field.
f0: wrapped Gaussian blob, x ~ N(2.0, 0.3^2), p ~ N(1.0, 0.3^2).
Observable at step n_f: phi = cos(x) + 0.5 sin(p), averaged with weights 1/N.

Estimators of d<phi>/dK at n_f:
  Lagrangian (exact derivative of the sample): sum w xi . grad phi.
  R-v with reset interval T: every T steps convert xi to dw with the collocated
  CIC phase-space mesh (M x M, periodic): P_g = sum w xi S_g, dF = -div_h P,
  dw_i = sum_g (dF_g / nhat_g) S_g(z_i); then xi = 0.  Final value
  sum dw phi + sum w xi . grad phi (xi accumulated since the last reset).
Reference: central FD with common random numbers, N_ref = 2e7, h in {0.01, 0.03}.
"""

import json
import sys

import numpy as np

TWO_PI = 2 * np.pi


def init(n, rng):
    x = (2.0 + 0.3 * rng.standard_normal(n)) % TWO_PI
    p = (1.0 + 0.3 * rng.standard_normal(n)) % TWO_PI
    return x, p


def phi(x, p):
    return np.cos(x) + 0.5 * np.sin(p)


def grad_phi(x, p):
    return -np.sin(x), 0.5 * np.cos(p)


def step(x, p, K):
    p = (p + K * np.sin(x)) % TWO_PI
    x = (x + p) % TWO_PI
    return x, p


def step_tan(x, p, dx, dp, K):
    s, c = np.sin(x), np.cos(x)
    dp = dp + s + K * c * dx
    p = (p + K * s) % TWO_PI
    dx = dx + dp
    x = (x + p) % TWO_PI
    return x, p, dx, dp


def convert(x, p, w, dx, dp, M):
    h = TWO_PI / M
    sx, sp = x / h, p / h
    ix, ip = np.floor(sx).astype(np.int64), np.floor(sp).astype(np.int64)
    fx, fp = sx - ix, sp - ip
    ix0, ix1, ip0, ip1 = ix % M, (ix + 1) % M, ip % M, (ip + 1) % M
    combos = [
        (ix0, ip0, (1 - fx) * (1 - fp)),
        (ix1, ip0, fx * (1 - fp)),
        (ix0, ip1, (1 - fx) * fp),
        (ix1, ip1, fx * fp),
    ]

    def dep(v):
        g = np.zeros(M * M)
        for a, b, s in combos:
            g += np.bincount(a * M + b, weights=s * v, minlength=M * M)
        return g.reshape(M, M)

    nhat = dep(np.ones_like(x))
    Px, Pp = dep(w * dx), dep(w * dp)
    dF = -(np.roll(Px, -1, 0) - np.roll(Px, 1, 0)) / (2 * h) - (np.roll(Pp, -1, 1) - np.roll(Pp, 1, 1)) / (2 * h)
    # dF_g: weight change per cell, times 1/h^2 * h^2 -> cell volume cancels (see e1)
    dF = dF * 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        hg = np.where(nhat > 0, dF / nhat, 0.0).ravel()
    lost = float(np.abs(dF[nhat == 0]).sum() / max(np.abs(dF).sum(), 1e-300))
    dw = np.zeros_like(x)
    for a, b, s in combos:
        dw += s * hg[a * M + b]
    return dw, lost


def rv_estimate(n, K, nf, T, M, rng):
    x, p = init(n, rng)
    w = np.full(n, 1.0 / n)
    dx, dp, dw = np.zeros(n), np.zeros(n), np.zeros(n)
    lost = []
    for k in range(1, nf + 1):
        x, p, dx, dp = step_tan(x, p, dx, dp, K)
        if T and k % T == 0 and k < nf:
            d, lo = convert(x, p, w, dx, dp, M)
            dw += d
            dx[:] = 0
            dp[:] = 0
            lost.append(lo)
    gx, gp = grad_phi(x, p)
    lag = np.sum(w * (gx * dx + gp * dp))
    return float(np.sum(dw * phi(x, p)) + lag), float(np.mean(lost)) if lost else 0.0, dx, dp


def fd_reference(n, K, nf, h, rng, chunk=5_000_000):
    acc = {+1: 0.0, -1: 0.0}
    sq = 0.0
    done = 0
    while done < n:
        m = min(chunk, n - done)
        x0, p0 = init(m, rng)
        vals = {}
        for sgn in (+1, -1):
            x, p = x0.copy(), p0.copy()
            for _ in range(nf):
                x, p = step(x, p, K + sgn * h)
            vals[sgn] = phi(x, p)
            acc[sgn] += vals[sgn].sum()
        d = (vals[1] - vals[-1]) / (2 * h)
        sq += np.sum(d * d)
        done += m
    mean = (acc[1] - acc[-1]) / (2 * h) / n
    se = np.sqrt(max(sq / n - mean * mean, 0) / n)
    return float(mean), float(se)


def lyapunov(K, nf, rng, n=20000):
    x, p = init(n, rng)
    dx, dp = np.zeros(n), np.zeros(n)
    # homogeneous tangent of a random direction: remove the K source by differencing
    ex, ep = rng.standard_normal(n), rng.standard_normal(n)
    out = []
    for k in range(1, nf + 1):
        s, c = np.sin(x), np.cos(x)
        ep = ep + K * c * ex
        ex = ex + ep
        p = (p + K * s) % TWO_PI
        x = (x + p) % TWO_PI
        out.append(float(np.median(np.log(np.hypot(ex, ep)))))
    return out


if __name__ == "__main__":
    K = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    rng = np.random.default_rng(2026)
    res = {"K": K, "n_f": nf}
    lg = lyapunov(K, nf, rng)
    res["median_log_tangent"] = lg
    lam = (lg[-1] - lg[nf // 2]) / (nf - 1 - nf // 2)
    res["lambda_median"] = lam
    print(f"K={K} n_f={nf}: median Lyapunov rate {lam:.3f} per step; e^(lambda n_f) = {np.exp(lam * nf):.3g}")
    ref = {}
    for h in (0.01, 0.03):
        ref[h] = fd_reference(20_000_000, K, nf, h, rng)
        print(f"  FD reference h={h}: {ref[h][0]:+.5f} +- {ref[h][1]:.5f}")
    res["fd"] = {str(h): v for h, v in ref.items()}
    rows = []
    for n in (200_000, 1_000_000):
        for M in (32, 64, 128):
            for T in (1, 2, 3, 5, 8, 12, 0):
                if M != 64 and T not in (1, 3, 8):
                    continue
                vals, losts = [], []
                for s in range(6):
                    v, lo, dx, dp = rv_estimate(n, K, nf, T, M, rng)
                    vals.append(v)
                    losts.append(lo)
                vals = np.array(vals)
                row = {
                    "N": n, "M": M, "T": T, "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                    "lost": float(np.mean(losts)),
                }
                rows.append(row)
                tag = "Lagrangian (no reset)" if T == 0 else f"T={T}"
                print(f"  N={n:8d} M={M:3d} {tag:22s} mean {row['mean']:+.5f} std {row['std']:.5f} lost {row['lost']:.2e}", flush=True)
    res["rows"] = rows
    with open(f"e2_results_K{K}_nf{nf}.json", "w") as fh:
        json.dump(res, fh, indent=1)
