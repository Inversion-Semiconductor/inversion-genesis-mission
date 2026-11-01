"""Test A1: GPT on perturbed doubling maps of the circle [0,1).

Case U: T(x) = 2x + theta,  f0 uniform. Lebesgue is invariant for every theta -> exact d<psi>/dtheta = 0.
Case G: T(x) = 2x + a sin(2 pi x)/(2 pi) + theta (0.5 + sin(2 pi x + 0.7))/(2 pi), a = 0.3, theta0 = 0,
        f0 = 1 + 0.5 cos(2 pi x - 1). Reference: Perron-Frobenius (PF) transfer operator on a
        trigonometric grid, central FD in theta at h = 1e-5.
Observable psi = cos(2 pi x) + 0.5 sin(4 pi x), A = int psi f_n.

Estimators (per seed): Lagrangian exact tangent; local mean (NGP cells); GPT Fourier K every T steps;
CRN central FD on the same particles at h.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gpt_lib import fourier1d_grads, project_dense

TP = 2 * np.pi
A_PERT = 0.3


def Tmap(x, th, case):
    if case == "U":
        return 2 * x + th, 2.0 + 0 * x, 1.0 + 0 * x
    y = 2 * x + A_PERT * np.sin(TP * x) / TP + th * (0.5 + np.sin(TP * x + 0.7)) / TP
    d = 2 + A_PERT * np.cos(TP * x) + th * np.cos(TP * x + 0.7)
    s = (0.5 + np.sin(TP * x + 0.7)) / TP
    return y, d, s


def f0(x, case):
    return np.ones_like(x) if case == "U" else 1 + 0.5 * np.cos(TP * x - 1)


def psi(x):
    return np.cos(TP * x) + 0.5 * np.sin(2 * TP * x)


def dpsi(x):
    return -TP * np.sin(TP * x) + 0.5 * 2 * TP * np.cos(2 * TP * x)


# ---------------------------------------------------------------- PF reference
def pf_A(th, case, nf, M=512):
    xg = np.arange(M) / M
    rho = f0(xg, case).astype(float)
    k = np.fft.fftfreq(M, 1.0 / M)
    out = []
    for n in range(nf):
        c = np.fft.fft(rho) / M
        new = np.zeros(M)
        for j in (-1, 0, 1, 2):
            lo, hi = np.zeros(M), np.ones(M)
            Flo = Tmap(lo, th, case)[0] - xg - j; Fhi = Tmap(hi, th, case)[0] - xg - j
            ok = (Flo <= 0) & (Fhi > 0)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                F = Tmap(mid, th, case)[0] - xg - j
                lo = np.where(F <= 0, mid, lo); hi = np.where(F > 0, mid, hi)
            xs = 0.5 * (lo + hi)
            r = np.real(np.exp(2j * np.pi * np.outer(xs, k)) @ c)
            new += np.where(ok, r / Tmap(xs, th, case)[1], 0.0)
        rho = new
        out.append(float(np.mean(psi(xg) * rho)))
    return np.array(out)


def pf_reference(case, nfs, h=1e-5):
    nmax = max(nfs)
    ap = pf_A(h, case, nmax); am = pf_A(-h, case, nmax)
    ap2 = pf_A(h, case, nmax, M=1024); am2 = pf_A(-h, case, nmax, M=1024)
    d1 = (ap - am) / (2 * h); d2 = (ap2 - am2) / (2 * h)
    return {n: (float(d2[n - 1]), float(abs(d2[n - 1] - d1[n - 1]))) for n in nfs}


# ---------------------------------------------------------------- particle estimators
def sample(N, rng, case):
    x = rng.random(N)
    w = f0(x, case); w = w / w.sum()
    return x, w


def run_estimators(N, seed, case, nf, Ts, Ks, Mcells, hs):
    rng = np.random.default_rng(seed)
    x0, w = sample(N, rng, case)
    res = {}
    # Lagrangian
    x, xi = x0.copy(), np.zeros(N)
    for n in range(nf):
        y, d, s = Tmap(x, 0.0, case)
        xi = d * xi + s; x = y % 1.0
    res["lag"] = float(np.sum(w * xi * dpsi(x)))
    res["xi_rms"] = float(np.sqrt(np.sum(w * xi ** 2)))
    # local mean (NGP cell average of xi), every step
    for M in Mcells:
        x, xi = x0.copy(), np.zeros(N)
        for n in range(nf):
            y, d, s = Tmap(x, 0.0, case)
            xi = d * xi + s; x = y % 1.0
            c = np.minimum((x * M).astype(np.int64), M - 1)
            num = np.bincount(c, w * xi, M); den = np.bincount(c, w, M)
            xi = (num / np.maximum(den, 1e-300))[c]
        res[f"lmean_M{M}"] = float(np.sum(w * xi * dpsi(x)))
    # GPT
    for K in Ks:
        for T in Ts:
            x, xi = x0.copy(), np.zeros(N)
            rms = []
            for n in range(1, nf + 1):
                y, d, s = Tmap(x, 0.0, case)
                xi = d * xi + s; x = y % 1.0
                if n % T == 0:
                    (xi,), _ = project_dense([fourier1d_grads(x, K)], [xi], w)
                    rms.append(float(np.sqrt(np.sum(w * xi ** 2))))
            res[f"gpt_K{K}_T{T}"] = float(np.sum(w * xi * dpsi(x)))
            res[f"gptrms_K{K}_T{T}"] = max(rms) if rms else 0.0
    # CRN FD
    for h in hs:
        vals = []
        for sg in (1, -1):
            x = x0.copy()
            for n in range(nf):
                x = Tmap(x, sg * h, case)[0] % 1.0
            vals.append(np.sum(w * psi(x)))
        res[f"fd_h{h}"] = float((vals[0] - vals[1]) / (2 * h))
    return res


if __name__ == "__main__":
    case = sys.argv[1]; nf = int(sys.argv[2]); N = int(float(sys.argv[3])); nseed = int(sys.argv[4])
    out = sys.argv[5]
    t0 = time.time()
    ref = pf_reference(case, [nf]) if case == "G" else {nf: (0.0, 0.0)}
    rows = []
    for s in range(nseed):
        Ts = tuple(int(v) for v in (sys.argv[6] if len(sys.argv) > 6 else "1,2,3,5").split(","))
        Ks = tuple(int(v) for v in (sys.argv[7] if len(sys.argv) > 7 else "2,4,8,16").split(","))
        rows.append(run_estimators(N, 1000 + s, case, nf, Ts=Ts, Ks=Ks,
                                   Mcells=(16, 64), hs=(1e-2, 1e-3, 1e-4)))
    keys = rows[0].keys()
    summ = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1) / np.sqrt(nseed)))
            for k in keys}
    json.dump(dict(case=case, nf=nf, N=N, nseed=nseed, ref=ref[nf], summary=summ, rows=rows,
                   time=time.time() - t0), open(out, "w"), indent=1)
    print(f"case {case} nf {nf} N {N} ref {ref[nf][0]:+.5f} (grid diff {ref[nf][1]:.1e})  time {time.time()-t0:.1f}s")
    for k in keys:
        print(f"  {k:18s} mean {summ[k][0]:+.4e}  SE {summ[k][1]:.2e}")
