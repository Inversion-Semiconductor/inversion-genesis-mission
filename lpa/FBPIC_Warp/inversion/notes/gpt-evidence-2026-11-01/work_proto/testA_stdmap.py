"""Test A2: GPT on the Chirikov standard map p' = p + K sin x, x' = x + p' (torus [0,2pi)^2), theta = K.

Case blob: f0 = wrapped Gaussian x ~ N(2, 0.3^2), p ~ N(1, 0.3^2) (as R0 E2), observable cos x + 0.5 sin p.
Case unif: f0 uniform on the torus. The map preserves area, so f_n = f0 for every K -> exact derivative 0.

Modes:
  est  <case> <nf> <N> <nseed> <out> [what]   per-seed estimators
  ref  <case> <nf> <N> <out>                   CRN central FD at several h, large N (chunked)
"""
import json, sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gpt_lib import fourier2d_modes, fourier2d_grads, project_dense_chunked, bspline2d_design, project_sparse

TP = 2 * np.pi
K0 = 1.5


def init(n, rng, case):
    if case == "unif":
        return rng.random(n) * TP, rng.random(n) * TP
    return (2.0 + 0.3 * rng.standard_normal(n)) % TP, (1.0 + 0.3 * rng.standard_normal(n)) % TP


def phi(x, p):
    return np.cos(x) + 0.5 * np.sin(p)


def gphi(x, p):
    return -np.sin(x), 0.5 * np.cos(p)


def step(x, p, K):
    p = (p + K * np.sin(x)) % TP
    return (x + p) % TP, p


def step_tan(x, p, dx, dp, K):
    s, c = np.sin(x), np.cos(x)
    dp = dp + s + K * c * dx
    dx = dx + dp
    p = (p + K * s) % TP
    return (x + p) % TP, p, dx, dp


def contract(x, p, dx, dp, w):
    gx, gp = gphi(x, p)
    return float(np.sum(w * (gx * dx + gp * dp)))


def proj(kind, par, x, p, dx, dp, w):
    if kind == "four":
        modes = fourier2d_modes(par)
        (dx, dp), _ = project_dense_chunked(lambda sl: fourier2d_grads(x[sl], p[sl], modes), x.size, [dx, dp], w)
        return dx, dp
    if kind == "bsp":
        Gx, Gp = bspline2d_design(x, p, 0, TP, par, 0, TP, par, periodic=True)
        (dx, dp), _, _ = project_sparse([Gx, Gp], [dx, dp], w)
        return dx, dp
    if kind == "lmean":
        M = par
        c = np.minimum((x / TP * M).astype(np.int64), M - 1) * M + np.minimum((p / TP * M).astype(np.int64), M - 1)
        den = np.maximum(np.bincount(c, w, M * M), 1e-300)
        return (np.bincount(c, w * dx, M * M) / den)[c], (np.bincount(c, w * dp, M * M) / den)[c]
    raise ValueError(kind)


def estimate(N, seed, case, nf, kind, par, T):
    rng = np.random.default_rng(seed)
    x, p = init(N, rng, case); w = np.full(N, 1.0 / N)
    dx, dp = np.zeros(N), np.zeros(N)
    rms = 0.0
    for n in range(1, nf + 1):
        x, p, dx, dp = step_tan(x, p, dx, dp, K0)
        if kind != "lag" and n % T == 0:
            dx, dp = proj(kind, par, x, p, dx, dp, w)
            rms = max(rms, float(np.sqrt(np.sum(w * (dx * dx + dp * dp)))))
    if kind == "lag":
        rms = float(np.sqrt(np.sum(w * (dx * dx + dp * dp))))
    return contract(x, p, dx, dp, w), rms


def crn_fd(N, seed, case, nf, hs):
    rng = np.random.default_rng(seed)
    x0, p0 = init(N, rng, case)
    out = {}
    for h in hs:
        v = []
        for sg in (1, -1):
            x, p = x0.copy(), p0.copy()
            for _ in range(nf):
                x, p = step(x, p, K0 + sg * h)
            v.append(phi(x, p).mean())
        out[h] = float((v[0] - v[1]) / (2 * h))
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ref":
        case, nf, N, out = sys.argv[2], int(sys.argv[3]), int(float(sys.argv[4])), sys.argv[5]
        hs = (0.003, 0.01, 0.02, 0.03, 0.05, 0.08)
        chunks = [crn_fd(2_000_000, 5000 + s, case, nf, hs) for s in range(N // 2_000_000)]
        res = {}
        for h in hs:
            v = np.array([c[h] for c in chunks])
            res[h] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)))
            print(f"ref {case} nf={nf} h={h}: {res[h][0]:+.5f} +- {res[h][1]:.5f}", flush=True)
        H = np.array(hs); D = np.array([res[h][0] for h in hs]); S = np.array([res[h][1] for h in hs])
        fits = {}
        for deg_name, cols in (("quad", 2), ("quart", 3)):
            for hmax in (0.03, 0.05, 0.08):
                m = H <= hmax + 1e-12
                if m.sum() < cols + 1:
                    continue
                Am = np.vander(H[m] ** 2, cols, increasing=True) / S[m, None]
                cf, *_ = np.linalg.lstsq(Am, D[m] / S[m], rcond=None)
                cov = np.linalg.inv(Am.T @ Am)
                fits[f"{deg_name}_hmax{hmax}"] = (float(cf[0]), float(np.sqrt(cov[0, 0])))
                print(f"  fit D(h)=D0+c h^2{'+d h^4' if cols==3 else ''} h<={hmax}: D0 = {cf[0]:+.5f} +- {np.sqrt(cov[0,0]):.5f}")
        json.dump(dict(case=case, nf=nf, N=N, fd={str(h): res[h] for h in hs}, fits=fits), open(out, "w"), indent=1)
    if mode == "est":
        case, nf, N, nseed, out = sys.argv[2], int(sys.argv[3]), int(float(sys.argv[4])), int(sys.argv[5]), sys.argv[6]
        what = sys.argv[7].split(";")  # entries like lag:0:0, four:4:1, bsp:16:2, lmean:32:1, fd
        t0 = time.time(); res = {}
        for item in what:
            if item == "fd":
                vals = [crn_fd(N, 100 + s, case, nf, (0.003, 0.01, 0.03)) for s in range(nseed)]
                for h in (0.003, 0.01, 0.03):
                    v = np.array([r[h] for r in vals])
                    res[f"fd_h{h}"] = dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(nseed)), sd=float(v.std(ddof=1)))
                continue
            kind, par, T = item.split(":"); par, T = int(par), int(T)
            t1 = time.time()
            r = [estimate(N, 100 + s, case, nf, kind, par, T) for s in range(nseed)]
            v = np.array([a for a, b in r]); rm = np.array([b for a, b in r])
            res[item] = dict(mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(nseed)), sd=float(v.std(ddof=1)),
                             rms=float(rm.mean()), sec_per_run=(time.time() - t1) / nseed)
            print(f"{case} nf={nf} N={N} {item:14s} mean {v.mean():+.4f} SE {res[item]['se']:.4f} sd {res[item]['sd']:.4f} rms {rm.mean():.3g}  {res[item]['sec_per_run']:.1f}s/run", flush=True)
        json.dump(dict(case=case, nf=nf, N=N, nseed=nseed, res=res), open(out, "w"), indent=1)
