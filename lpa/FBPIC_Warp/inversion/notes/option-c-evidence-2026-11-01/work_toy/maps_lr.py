"""Option C (likelihood-ratio / Girsanov weights) on noisy chaotic maps.

Doubling map (case G of testA_doubling.py), additive noise on x:
    x' = T_theta(x) + sigma xi  (mod 1),  T = 2x + a sin(2 pi x)/(2 pi) + theta s(x),
    s(x) = (0.5 + sin(2 pi x + 0.7))/(2 pi),  theta0 = 0,  f0 = 1 + 0.5 cos(2 pi x - 1),
    psi = cos(2 pi x) + 0.5 sin(4 pi x).
    LR increment: omega += s(x) xi / sigma.
    Reference: noisy transfer operator (Fourier quadrature) with exact tangent.

Standard map, noise on the kick only:
    p' = p + K sin x + sigma xi,  x' = x + p'  (torus), theta = K, K0 = 1.5,
    blob f0: x ~ N(2, 0.3^2), p ~ N(1, 0.3^2); phi = cos x + 0.5 sin p.
    LR increment: omega += sin(x) xi / sigma.
    Reference: CRN central FD of the noisy model over a large particle ensemble.

Usage:
    python -I maps_lr.py dbl  <sigma> <N_per_batch> <nbatch> <out.json>
    python -I maps_lr.py std  <sigma> <N_per_batch> <nbatch> <out.json>
    python -I maps_lr.py pf   <out.json>            (doubling-map transfer-operator references)
sigma = 0 runs the deterministic model (LR skipped).
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

TP = 2 * np.pi
A_PERT = 0.3
K0 = 1.5
NREC = (5, 10, 20, 40, 80)
WINS = (5, 10, 20)
HS = (1e-1, 3e-2, 1e-2, 1e-3)


# ------------------------------------------------------------------ doubling map
def Tmap(x, th):
    y = 2 * x + A_PERT * np.sin(TP * x) / TP + th * (0.5 + np.sin(TP * x + 0.7)) / TP
    d = 2 + A_PERT * np.cos(TP * x) + th * np.cos(TP * x + 0.7)
    s = (0.5 + np.sin(TP * x + 0.7)) / TP
    return y, d, s


def psi(x):
    return np.cos(TP * x) + 0.5 * np.sin(2 * TP * x)


def dpsi(x):
    return -TP * np.sin(TP * x) + 2 * TP * np.cos(2 * TP * x)


def pf_noisy(sigma, nmax, M=4096, K=600):
    """Noisy transfer operator on Fourier coefficients with its exact theta-tangent at theta0 = 0."""
    xg = np.arange(M) / M
    k = np.arange(K + 1)
    G = np.exp(-2 * np.pi ** 2 * sigma ** 2 * k ** 2)
    y, _, s = Tmap(xg, 0.0)
    E = np.exp(-2j * np.pi * np.outer(k, y)) / M          # (K+1, M)
    f = 1 + 0.5 * np.cos(TP * xg - 1); fd = np.zeros(M)
    A, dA = [], []
    for n in range(nmax):
        c = G * (E @ f)
        cd = G * (E @ fd - 2j * np.pi * (E * (k[:, None] * s[None, :])) @ f)
        A.append(c[1].real - 0.5 * c[2].imag); dA.append(cd[1].real - 0.5 * cd[2].imag)
        # back to the grid (real field from one-sided coefficients)
        full = np.zeros(M, complex); full[:K + 1] = c; full[M - K:] = np.conj(c[1:][::-1])
        fullt = np.zeros(M, complex); fullt[:K + 1] = cd; fullt[M - K:] = np.conj(cd[1:][::-1])
        f = np.real(np.fft.ifft(full) * M); fd = np.real(np.fft.ifft(fullt) * M)
    return np.array(A), np.array(dA)


def dbl_batch(args):
    sigma, N, seed = args
    rng = np.random.default_rng(seed)
    x0 = rng.random(N); w = 1 + 0.5 * np.cos(TP * x0 - 1); w /= w.sum()
    nmax = max(NREC)
    xi_all = rng.standard_normal((nmax, N)) if sigma > 0 else None
    out = {}
    # base path with LR and Lagrangian tangent
    x = x0.copy(); om = np.zeros(N); tan = np.zeros(N); hist = {0: om.copy()}
    for n in range(1, nmax + 1):
        y, d, s = Tmap(x, 0.0)
        tan = d * tan + s
        if sigma > 0:
            om = om + s * xi_all[n - 1] / sigma
            y = y + sigma * xi_all[n - 1]
        x = y % 1.0
        if n in NREC or any(n + W in NREC for W in WINS):
            hist[n] = om.copy()
        if n in NREC:
            g = psi(x); gb = np.sum(w * g)
            out[f"J_n{n}"] = float(gb)
            out[f"tan_n{n}"] = float(np.sum(w * dpsi(x) * tan))
            out[f"tanrms_n{n}"] = float(np.sqrt(np.sum(w * tan ** 2)))
            if sigma > 0:
                out[f"lr_n{n}"] = float(np.sum(w * (g - gb) * om))
                out[f"lr0_n{n}"] = float(np.sum(w * g * om))        # no baseline
                out[f"omrms_n{n}"] = float(np.sqrt(np.sum(w * om ** 2)))
                for W in WINS:
                    if n - W >= 0:
                        out[f"lrW{W}_n{n}"] = float(np.sum(w * (g - gb) * (om - hist[n - W])))
    # CRN central FD
    for h in HS:
        xs = {}
        for sg in (1, -1):
            x = x0.copy()
            for n in range(1, nmax + 1):
                y = Tmap(x, sg * h)[0]
                if sigma > 0:
                    y = y + sigma * xi_all[n - 1]
                x = y % 1.0
                if n in NREC:
                    xs[(sg, n)] = np.sum(w * psi(x))
        for n in NREC:
            out[f"fd{h:g}_n{n}"] = float((xs[(1, n)] - xs[(-1, n)]) / (2 * h))
    return out


# ------------------------------------------------------------------ standard map
def std_batch(args):
    sigma, N, seed = args
    rng = np.random.default_rng(seed)
    x0 = (2.0 + 0.3 * rng.standard_normal(N)) % TP; p0 = (1.0 + 0.3 * rng.standard_normal(N)) % TP
    nmax = max(NREC)
    xi_all = rng.standard_normal((nmax, N)) if sigma > 0 else None
    out = {}
    x, p = x0.copy(), p0.copy(); dx, dp = np.zeros(N), np.zeros(N); om = np.zeros(N); hist = {0: om.copy()}
    for n in range(1, nmax + 1):
        sx, cx = np.sin(x), np.cos(x)
        dp = dp + sx + K0 * cx * dx
        dx = dx + dp
        p = p + K0 * sx
        if sigma > 0:
            om = om + sx * xi_all[n - 1] / sigma
            p = p + sigma * xi_all[n - 1]
        p = p % TP; x = (x + p) % TP
        if n in NREC or any(n + W in NREC for W in WINS):
            hist[n] = om.copy()
        if n in NREC:
            g = np.cos(x) + 0.5 * np.sin(p); gb = g.mean()
            out[f"J_n{n}"] = float(gb)
            out[f"tan_n{n}"] = float(np.mean(-np.sin(x) * dx + 0.5 * np.cos(p) * dp))
            out[f"tanrms_n{n}"] = float(np.sqrt(np.mean(dx * dx + dp * dp)))
            if sigma > 0:
                out[f"lr_n{n}"] = float(np.mean((g - gb) * om))
                out[f"lr0_n{n}"] = float(np.mean(g * om))
                out[f"omrms_n{n}"] = float(np.sqrt(np.mean(om ** 2)))
                for W in WINS:
                    if n - W >= 0:
                        out[f"lrW{W}_n{n}"] = float(np.mean((g - gb) * (om - hist[n - W])))
    for h in HS:
        xs = {}
        for sg in (1, -1):
            x, p = x0.copy(), p0.copy()
            for n in range(1, nmax + 1):
                p = p + (K0 + sg * h) * np.sin(x)
                if sigma > 0:
                    p = p + sigma * xi_all[n - 1]
                p = p % TP; x = (x + p) % TP
                if n in NREC:
                    xs[(sg, n)] = np.mean(np.cos(x) + 0.5 * np.sin(p))
        for n in NREC:
            out[f"fd{h:g}_n{n}"] = float((xs[(1, n)] - xs[(-1, n)]) / (2 * h))
    return out


def summarize(rows):
    keys = rows[0].keys(); S = len(rows)
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1) / np.sqrt(S)))
            for k in keys}


if __name__ == "__main__":
    mode = sys.argv[1]
    t0 = time.time()
    if mode == "pf":
        res = {}
        for sigma in (0.0, 0.003, 0.01, 0.03, 0.1, 0.3):
            A, dA = pf_noisy(sigma, max(NREC))
            A2, dA2 = pf_noisy(sigma, max(NREC), M=8192, K=1200)
            res[f"{sigma:g}"] = {f"n{n}": (float(dA[n - 1]), float(abs(dA[n - 1] - dA2[n - 1])), float(A[n - 1]))
                                 for n in NREC}
            print(sigma, {n: f"{dA[n-1]:+.5f} (grid {abs(dA[n-1]-dA2[n-1]):.1e})" for n in NREC}, flush=True)
        json.dump(res, open(sys.argv[2], "w"), indent=1)
        print(f"time {time.time()-t0:.1f}s")
        sys.exit(0)
    sigma = float(sys.argv[2]); N = int(float(sys.argv[3])); nb = int(sys.argv[4]); outp = sys.argv[5]
    fn = dbl_batch if mode == "dbl" else std_batch
    with Pool(min(nb, 24)) as pool:
        sb = int(sys.argv[6]) if len(sys.argv) > 6 else 7000
        rows = pool.map(fn, [(sigma, N, sb + b) for b in range(nb)])
    summ = summarize(rows)
    json.dump(dict(mode=mode, sigma=sigma, N=N, nbatch=nb, summary=summ, time=time.time() - t0),
              open(outp, "w"), indent=1)
    print(f"{mode} sigma {sigma} N {N}x{nb}  time {time.time()-t0:.1f}s")
    for k in sorted(summ):
        print(f"  {k:16s} {summ[k][0]:+.4e} +- {summ[k][1]:.2e}")
