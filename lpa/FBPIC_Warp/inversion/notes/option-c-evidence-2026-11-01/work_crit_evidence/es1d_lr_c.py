"""Option C on a 1D-1V electrostatic mean-field system: cold symmetric two-stream with momentum noise.

Units: omega_p = 1 (total electron density 1), q = -1, m = 1, periodic box L = 10 pi (k1 = 0.2).
Two cold beams at v = +-V0 (V0 = 1), quiet start, uniform neutralizing ions. Unstable modes m = 1..4.
Driver (theta enters only here, like the laser): E_ext = theta EPS [sin(2 k1 x) + sin(3 k1 x + 1)] g(t),
g(t) = exp(-((t - 4)/1.5)^2), theta0 = 1.
Push (Euler-Maruyama in v, then drift):
    v' = v + q (E(x) + E_ext(x, t)) dt + sigma sqrt(dt) xi,   x' = x + v' dt.
Field: CIC deposit, spectral Poisson dE/dx = rho, modes m > MCUT removed (same filter for all tangents).

Estimators of d/dtheta E[J], all from ONE noisy run (per seed):
  mf  : mean-field LR. omega_i += q (dE[omega](x_i) + dtheta E_ext) sqrt(dt) xi_i / sigma,
        dE[omega] = field of drho = q sum_i W omega_i S(x - x_i) (field-tangent coupling).
  jnt : joint-measure LR. Omega = sum_i sum_n q dtheta E_ext sqrt(dt) xi_i / sigma (one scalar),
        estimator (J - mean_seeds J) Omega.
  tan : exact pathwise (Lagrangian) tangent with the same noise (CRN), incl. the deposit tangent.
  fd  : CRN central FD of the noisy model at several h (separate runs, same noise).
Observables: K = <v^2> (all), m1 = <v> and s1 = var(v) of beam 1 (initial v = +V0).

Usage:
  python -I es1d_lr.py diag <sigma> <N> <seed> <out.json>             time series for one seed
  python -I es1d_lr.py ens  <sigma> <N> <nseed> <out.json> [hs] [mf_only]
"""
import json
import sys
import time
from multiprocessing import Pool

import os

import numpy as np

L = 10 * np.pi; NG = 128; DX = L / NG; DT = 0.1; V0 = float(os.environ.get('ES_V0', '1.0')); Q = -1.0
CENTER = int(os.environ.get('ES_CENTER', '0'))
MCUT = 24; EPS = float(os.environ.get('ES_EPS', '1e-3')); TH0 = 1.0
TREC = tuple(float(v) for v in os.environ.get('ES_TREC', '10,20,30,40,60').split(','))
k = 2 * np.pi * np.fft.rfftfreq(NG, DX)
FILT = (np.arange(k.size) <= MCUT).astype(float)
IKINV = np.zeros(k.size, complex); IKINV[1:] = 1.0 / (1j * k[1:])


def ext(x, t):
    """dtheta E_ext and d/dx E_ext / theta at (x, t)."""
    g = np.exp(-((t - 4.0) / 1.5) ** 2)
    k1 = 2 * np.pi / L
    e = EPS * g * (np.sin(2 * k1 * x) + np.sin(3 * k1 * x + 1))
    de = EPS * g * (2 * k1 * np.cos(2 * k1 * x) + 3 * k1 * np.cos(3 * k1 * x + 1))
    return e, de


def cic(x):
    s = x / DX
    i = np.floor(s).astype(np.int64); d = s - i
    return i % NG, (i + 1) % NG, d


def solve(rho):
    return np.fft.irfft(np.fft.rfft(rho) * FILT * IKINV, NG)


def deposit(i0, i1, d, val):
    return (np.bincount(i0, val * (1 - d), NG) + np.bincount(i1, val * d, NG)) / DX


def gather(F, i0, i1, d):
    return F[i0] * (1 - d) + F[i1] * d


def init(N):
    nb = N // 2
    x = np.concatenate([(np.arange(nb) + 0.25) * L / nb, (np.arange(nb) + 0.75) * L / nb])
    v = np.concatenate([np.full(nb, V0), np.full(nb, -V0)])
    return x, v, L / N


def observables(v, nb):
    v1 = v[:nb]
    return np.array([np.mean(v * v), v1.mean(), v1.var()])


def lr_obs(v, om, nb):
    """d<g>/dtheta from weights om for the three observables (baseline-subtracted, equal weights)."""
    v1 = v[:nb]; o1 = om[:nb]
    K = np.mean(v * v); m1 = v1.mean(); s1 = v1.var()
    return np.array([np.mean((v * v - K) * om), np.mean((v1 - m1) * o1), np.mean(((v1 - m1) ** 2 - s1) * o1)])


def tan_obs(v, dv, nb):
    v1 = v[:nb]; m1 = v1.mean()
    return np.array([np.mean(2 * v * dv), np.mean(dv[:nb]), np.mean(2 * (v1 - m1) * dv[:nb])])


def run(theta, sigma, N, seed, do_mf=True, do_tan=True, series=False, tmax=None):
    tmax = tmax or max(TREC)
    nt = int(round(tmax / DT)); rec = {int(round(t / DT)): t for t in TREC if t <= tmax}
    x, v, W = init(N); nb = N // 2
    rng = np.random.default_rng(seed)
    om = np.zeros(N); Om = 0.0; lam = 0.0; dx = np.zeros(N); dv = np.zeros(N)
    out = {}; ser = []
    sq = np.sqrt(DT)
    for n in range(nt + 1):
        t = n * DT
        i0, i1, d = cic(x)
        rho = 1.0 + Q * deposit(i0, i1, d, np.full(N, W))
        E = solve(rho)
        Ep = gather(E, i0, i1, d)
        if n in rec or (series and n % 10 == 0):
            o = {"J": observables(v, nb).tolist(), "t": t}
            if do_mf and sigma > 0:
                o["mf"] = lr_obs(v, om, nb).tolist(); o["omrms"] = float(np.sqrt(np.mean(om ** 2)))
                o["Om"] = Om; o["lam"] = lam
            if do_tan:
                o["tan"] = tan_obs(v, dv, nb).tolist()
                o["tanrms"] = float(np.sqrt(np.mean(dx ** 2 + dv ** 2)))
            o["WE"] = float(0.5 * np.sum(E ** 2) * DX)
            if n in rec:
                out[f"t{rec[n]:g}"] = o
            if series:
                ser.append(o)
        if n == nt:
            break
        e, de = ext(x, t)
        xi = rng.standard_normal(N) if sigma > 0 else 0.0
        F = Q * (Ep + theta * e)
        if do_mf and sigma > 0:
            dE = solve(Q * deposit(i0, i1, d, W * om))
            dEp_w = gather(dE, i0, i1, d)
            dF = Q * (dEp_w + e)
            lam = float(np.mean(dEp_w ** 2) / max(np.mean(om ** 2), 1e-300)) / sigma ** 2
            inc = dF * sq * xi / sigma
            if CENTER:
                num = deposit(i0, i1, d, W * inc); den = deposit(i0, i1, d, np.full(N, W))
                inc = inc - gather(num / np.maximum(den, 1e-30), i0, i1, d)
            om = om + inc
            Om += float(np.sum(Q * e * sq * xi)) / sigma
        if do_tan:
            # deposit tangent: d/dx_i of W S(x_g - x_i)/DX -> (-1, +1) W/DX^2 on (i0, i1)
            drho = Q * (np.bincount(i0, -W * dx, NG) + np.bincount(i1, W * dx, NG)) / DX ** 2
            dE = solve(drho)
            dEp = gather(dE, i0, i1, d) + (E[i1] - E[i0]) / DX * dx
            dvn = dv + Q * (dEp + e + theta * de * dx) * DT
            dx = dx + dvn * DT; dv = dvn
        v = v + F * DT + (sigma * sq * xi if sigma > 0 else 0.0)
        x = (x + v * DT) % L
    if series:
        out["series"] = ser
    return out


def ens_job(args):
    sigma, N, seed, hs, mf_only = args
    t0 = time.time()
    base = run(TH0, sigma, N, seed, do_mf=mf_only != "fd_only", do_tan=mf_only not in ("mf_only", "fd_only"))
    res = {"base": base}
    for h in hs:
        rp = run(TH0 + h, sigma, N, seed, do_mf=False, do_tan=False)
        rm = run(TH0 - h, sigma, N, seed, do_mf=False, do_tan=False)
        res[f"fd{h:g}"] = {tk: ((np.array(rp[tk]["J"]) - np.array(rm[tk]["J"])) / (2 * h)).tolist() for tk in rp}
    res["time"] = time.time() - t0
    return res


if __name__ == "__main__":
    mode = sys.argv[1]; sigma = float(sys.argv[2]); N = int(float(sys.argv[3]))
    t0 = time.time()
    if mode == "verify":
        # pathwise tangent vs CRN FD at tiny h, one seed
        seed = int(sys.argv[4])
        b = run(TH0, sigma, N, seed, do_mf=False, do_tan=True, tmax=30)
        for h in (1e-5, 1e-6):
            rp = run(TH0 + h, sigma, N, seed, do_mf=False, do_tan=False, tmax=30)
            rm = run(TH0 - h, sigma, N, seed, do_mf=False, do_tan=False, tmax=30)
            for tk in ("t10", "t20", "t30"):
                fd = (np.array(rp[tk]["J"]) - np.array(rm[tk]["J"])) / (2 * h)
                print(f"h {h:g} {tk} tan {np.array(b[tk]['tan'])} fd {fd}")
        sys.exit(0)
    if mode == "diag":
        seed = int(sys.argv[4])
        r = run(TH0, sigma, N, seed, series=True, tmax=float(sys.argv[6]) if len(sys.argv) > 6 else None)
        json.dump(r, open(sys.argv[5], "w"))
        for o in r["series"][::5]:
            print(f"t {o['t']:5.1f} WE {o['WE']:.3e} K {o['J'][0]:.4f} tanrms {o.get('tanrms',0):.3e} "
                  f"omrms {o.get('omrms',0):.3e} mf {np.round(o.get('mf',[0,0,0]),4)} tan {np.round(o.get('tan',[0,0,0]),4)}")
        print(f"time {time.time()-t0:.1f}s")
    else:
        nseed = int(sys.argv[4]); outp = sys.argv[5]
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", f"SKIP_s{sigma:g}_N{N}")):
            print("skipped by flag file"); sys.exit(0)
        hs = tuple(float(s) for s in sys.argv[6].split(",")) if len(sys.argv) > 6 else (0.3, 0.1, 0.03)
        mf_only = sys.argv[7] if len(sys.argv) > 7 else ""
        with Pool(int(os.environ.get('ES_POOL', '8'))) as pool:
            rows = pool.map(ens_job, [(sigma, N, 100 + s, hs, mf_only) for s in range(nseed)])
        json.dump(dict(sigma=sigma, N=N, nseed=nseed, hs=hs, rows=rows, time=time.time() - t0), open(outp, "w"))
        print(f"ens sigma {sigma} N {N} nseed {nseed} time {time.time()-t0:.1f}s")
