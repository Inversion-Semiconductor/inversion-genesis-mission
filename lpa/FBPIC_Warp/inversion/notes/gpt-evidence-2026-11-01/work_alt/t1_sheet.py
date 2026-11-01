"""T1: cold beam in two prescribed overlapping waves (1D-1V), symplectic Euler.

    x' = x + DT v,   v' = v + DT F(x', t'; A) [+ sigma sqrt(DT) eps]
    F = -A sin(x) - B sin(K2 x - OM t),  theta = A.

Cold sheet: labels a = x0 in [0, 2 pi), v0 = V0, uniform label density.
Observables (v only): J1 = <sel>, J2 = <sel v>, J3 = <v^2>, sel = S5 window.

Estimators of dJ/dA at T:
  lag   : Lagrangian (pathwise) mean of grad(phi) . xi
  libp  : label-space integration by parts on the sheet (this study):
          xi = tau t + nu n, contribution -phi d_a tau + nu d_n phi,
          d_a tau from second-order tangents t_a = d_a t, xi_a = d_a xi.
  nlr   : noise-injected likelihood ratio, noisy dynamics, exact Gaussian score.
Truth: lattice (midpoint) quadrature of lag with N up to 1e7 labels, which
converges spectrally once the label lattice resolves the stretched sheet.
"""

import json
import sys
import time
from multiprocessing import Pool

import numpy as np

A0, B, K2, OM = 1.0, 0.2, 1.0, 2.2
DT = 0.05
VC, DU = 2.0, 0.1
V0 = 1.0


def S5(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (10 - 15 * t + 6 * t * t)


def dS5(t):
    inside = (t > 0) & (t < 1)
    return np.where(inside, 30 * t * t * (1 - t) ** 2, 0.0)


def phi_all(v):
    s = S5((v - VC + 2 * DU) / (4 * DU))
    return np.stack([s, s * v, v * v])


def dphi_all(v):
    t = (v - VC + 2 * DU) / (4 * DU)
    s = S5(t)
    ds = dS5(t) / (4 * DU)
    return np.stack([ds, ds * v + s, 2 * v])


def run_tangents(a, A, nstep):
    """Exact first- and second-order tangents along the cold sheet."""
    x = a.copy()
    v = np.full_like(a, V0)
    tx, tv = np.ones_like(a), np.zeros_like(a)
    tax, tav = np.zeros_like(a), np.zeros_like(a)
    qx, qv = np.zeros_like(a), np.zeros_like(a)  # xi = dz/dA
    qax, qav = np.zeros_like(a), np.zeros_like(a)  # xi_a = d2z/da dA
    for k in range(nstep):
        t1 = (k + 1) * DT
        x = x + DT * v
        tx = tx + DT * tv
        tax = tax + DT * tav
        qx = qx + DT * qv
        qax = qax + DT * qav
        s1, c1 = np.sin(x), np.cos(x)
        ph = K2 * x - OM * t1
        F = -A * s1 - B * np.sin(ph)
        Fx = -A * c1 - B * K2 * np.cos(ph)
        Fxx = A * s1 + B * K2 * K2 * np.sin(ph)
        FA, FAx = -s1, -c1
        v = v + DT * F
        tav = tav + DT * (Fx * tax + Fxx * tx * tx)
        tv = tv + DT * Fx * tx
        qav = qav + DT * (Fx * qax + Fxx * tx * qx + FAx * tx)
        qv = qv + DT * (Fx * qx + FA)
    return x, v, (tx, tv), (tax, tav), (qx, qv), (qax, qav)


def estimators(a, A, nstep):
    x, v, t, ta, q, qa = run_tangents(a, A, nstep)
    tx, tv = t
    tax, tav = ta
    qx, qv = q
    qax, qav = qa
    phi = phi_all(v)
    dphi = dphi_all(v)
    lag = dphi * qv
    t2 = tx * tx + tv * tv
    tn = np.sqrt(t2)
    nx_, nv_ = -tv / tn, tx / tn
    nu = nx_ * qx + nv_ * qv
    qt = qx * tx + qv * tv
    dtau = ((qax * tx + qav * tv + qx * tax + qv * tav) * t2 - 2 * qt * (tx * tax + tv * tav)) / (t2 * t2)
    libp = -phi * dtau + nu * nv_ * dphi
    return dict(phi=phi, lag=lag, libp=libp, tn=tn, dtau=dtau, nu=nu, qn=np.hypot(qx, qv))


def run_nlr(a, A, nstep, sigma, seed):
    rng = np.random.default_rng(seed)
    x = a.copy()
    v = np.full_like(a, V0)
    wd = np.zeros_like(a)
    sq = np.sqrt(DT)
    for k in range(nstep):
        t1 = (k + 1) * DT
        x = x + DT * v
        s1 = np.sin(x)
        F = -A * s1 - B * np.sin(K2 * x - OM * t1)
        eps = rng.standard_normal(a.size)
        v = v + DT * F + sigma * sq * eps
        wd += eps * (-s1) * sq / sigma
    return phi_all(v), wd


def J_noisy(a, A, nstep, sigma, seed):
    rng = np.random.default_rng(seed)
    x = a.copy()
    v = np.full_like(a, V0)
    sq = np.sqrt(DT)
    for k in range(nstep):
        t1 = (k + 1) * DT
        x = x + DT * v
        F = -A * np.sin(x) - B * np.sin(K2 * x - OM * t1)
        v = v + DT * F + sigma * sq * rng.standard_normal(a.size)
    return phi_all(v).mean(axis=1)


def lattice(n, j, nchunk):
    m = n // nchunk
    i = np.arange(j * m, (j + 1) * m)
    return 2 * np.pi * (i + 0.5) / n


def _truth_chunk(args):
    n, j, nchunk, nstep = args
    e = estimators(lattice(n, j, nchunk), A0, nstep)
    return e["lag"].sum(axis=1), e["libp"].sum(axis=1), e["phi"].sum(axis=1)


def truth(n, nstep, pool, nchunk=64):
    res = pool.map(_truth_chunk, [(n, j, nchunk, nstep) for j in range(nchunk)])
    lag = sum(r[0] for r in res) / n
    lib = sum(r[1] for r in res) / n
    J = sum(r[2] for r in res) / n
    return lag, lib, J


def _J_chunk(args):
    n, j, nchunk, nstep, A = args
    a = lattice(n, j, nchunk)
    x = a.copy()
    v = np.full_like(a, V0)
    for k in range(nstep):
        t1 = (k + 1) * DT
        x = x + DT * v
        v = v + DT * (-A * np.sin(x) - B * np.sin(K2 * x - OM * t1))
    return phi_all(v).sum(axis=1)


def J_lattice(n, nstep, A, pool, nchunk=64):
    res = pool.map(_J_chunk, [(n, j, nchunk, nstep, A) for j in range(nchunk)])
    return sum(res) / n


def _rand_chunk(args):
    n, seed, nstep = args
    rng = np.random.default_rng(seed)
    e = estimators(2 * np.pi * rng.random(n), A0, nstep)
    out = {}
    for k in ("lag", "libp"):
        out[k] = e[k].mean(axis=1)
    out["tn_med"] = float(np.median(e["tn"]))
    out["tn_p99"] = float(np.quantile(e["tn"], 0.99))
    out["qn_med"] = float(np.median(e["qn"]))
    out["qn_p99"] = float(np.quantile(e["qn"], 0.99))
    out["libp_absmax"] = np.abs(e["libp"]).max(axis=1).tolist()
    out["libp_sd"] = e["libp"].std(axis=1).tolist()
    out["lag_sd"] = e["lag"].std(axis=1).tolist()
    return out


def _nlr_chunk(args):
    n, seed, nstep, sigma = args
    rng = np.random.default_rng(seed)
    a = 2 * np.pi * rng.random(n)
    phi, wd = run_nlr(a, A0, nstep, sigma, seed + 7919)
    c = wd * (phi - phi.mean(axis=1, keepdims=True))
    return c.mean(axis=1)


def _fdn_chunk(args):
    n, seed, nstep, sigma, h = args
    rng = np.random.default_rng(seed)
    a = 2 * np.pi * rng.random(n)
    Jp = J_noisy(a, A0 + h, nstep, sigma, seed + 7919)
    Jm = J_noisy(a, A0 - h, nstep, sigma, seed + 7919)
    return (Jp - Jm) / (2 * h)


def ms(rows):
    r = np.array(rows)
    return r.mean(axis=0).tolist(), (r.std(axis=0, ddof=1) / np.sqrt(len(r))).tolist(), r.std(axis=0, ddof=1).tolist()


def main(out):
    res = {}
    with Pool(32) as pool:
        for T in (10.0, 20.0, 30.0):
            nstep = int(round(T / DT))
            r = {}
            t0 = time.time()
            for n in (10**5, 10**6, 10**7):
                lag, lib, J = truth(n, nstep, pool)
                r[f"lattice_{n}"] = dict(lag=lag.tolist(), libp=lib.tolist(), J=J.tolist())
                print(T, n, "lattice lag", lag, "libp", lib, f"{time.time() - t0:.0f}s", flush=True)
            # large-scale secants on the lattice (n = 1e6)
            sec = {}
            for h in (0.1, 0.03, 0.01, 0.003):
                Jp = J_lattice(10**6, nstep, A0 + h, pool)
                Jm = J_lattice(10**6, nstep, A0 - h, pool)
                sec[str(h)] = ((Jp - Jm) / (2 * h)).tolist()
            r["secant_lattice_1e6"] = sec
            print(T, "secants", sec, flush=True)
            # random-label (PIC-like) estimators: 32 seeds of N = 1e4 and 1e5
            for n in (10**4, 10**5):
                outs = pool.map(_rand_chunk, [(n, 1000 + s, nstep) for s in range(32)])
                d = {}
                for k in ("lag", "libp"):
                    m, se, sd = ms([o[k] for o in outs])
                    d[k] = dict(mean=m, se=se, seed_sd=sd)
                for k in ("tn_med", "tn_p99", "qn_med", "qn_p99"):
                    d[k] = float(np.median([o[k] for o in outs]))
                d["libp_absmax"] = np.max([o["libp_absmax"] for o in outs], axis=0).tolist()
                d["libp_particle_sd"] = np.median([o["libp_sd"] for o in outs], axis=0).tolist()
                d["lag_particle_sd"] = np.median([o["lag_sd"] for o in outs], axis=0).tolist()
                r[f"random_{n}"] = d
                print(T, n, "random", json.dumps(d), flush=True)
            # NLR, N = 1e5, 32 seeds, several sigma; noisy-system CRN FD, N = 1e5, 32 seeds
            for sigma in (0.01, 0.03, 0.1, 0.3):
                outs = pool.map(_nlr_chunk, [(10**5, 2000 + s, nstep, sigma) for s in range(32)])
                m, se, sd = ms(outs)
                fd = pool.map(_fdn_chunk, [(10**5, 3000 + s, nstep, sigma, 0.02) for s in range(32)])
                mf, sef, sdf = ms(fd)
                r[f"nlr_{sigma}"] = dict(mean=m, se=se, seed_sd=sd, fd_noisy_mean=mf, fd_noisy_se=sef)
                print(T, "nlr", sigma, m, se, "fd_noisy", mf, sef, flush=True)
            res[str(T)] = r
            with open(out, "w") as fh:
                json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
