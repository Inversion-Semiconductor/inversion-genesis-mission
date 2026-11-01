"""D1: gauge-projected tangent (GPT) on the doubling map x -> 2x + theta mod 1.

Uniform f is invariant for every theta, so d<psi>/dtheta = 0 exactly.
Exact particle tangent: xi_n = 2^n - 1 (same for every particle).
Estimators of d<psi>/dtheta at step n for psi = cos(2 pi x) (in span) and
psi = cos(2 pi (K+1) x) (outside the span):
  LAG: sum w xi psi'(x).
  GPT-conf: project xi onto gradients of the periodic Fourier basis
            {sin, cos(2 pi k x), k = 1..K} (conforming, global).
  GPT-broken: project per cell (M cells) onto gradients of polynomials of
            degree <= 2 on that cell (broken basis).
Particles: iid random, or a quiet lattice a_i = (i + 0.5)/N with N odd,
plus jitter of size eps/N.
"""
import numpy as np

rng = np.random.default_rng(7)


def gpt_conf(x, w, xi, K):
    k = np.arange(1, K + 1)
    G = np.concatenate([2 * np.pi * k * np.cos(2 * np.pi * np.outer(x, k)),
                        -2 * np.pi * k * np.sin(2 * np.pi * np.outer(x, k))], axis=1)
    A = G * np.sqrt(w)[:, None]
    c, *_ = np.linalg.lstsq(A, xi * np.sqrt(w), rcond=None)
    return G @ c


def gpt_broken(x, w, xi, M, deg=2):
    out = np.zeros_like(xi)
    cell = np.minimum((x * M).astype(int), M - 1)
    for m in range(M):
        s = cell == m
        if s.sum() == 0:
            continue
        t = x[s] * M - m - 0.5
        G = np.stack([j * t ** (j - 1) * M for j in range(1, deg + 1)], axis=1)
        A = G * np.sqrt(w[s])[:, None]
        c, *_ = np.linalg.lstsq(A, xi[s] * np.sqrt(w[s]), rcond=None)
        out[s] = G @ c
    return out


def run(N, n, mode, K=4, M=8, eps=0.0):
    if mode == "random":
        x = rng.random(N)
    else:
        x = ((np.arange(N) + 0.5 + eps * rng.standard_normal(N)) / N) % 1.0
    xi = np.zeros(N)
    for _ in range(n):
        x = (2 * x + 0.0) % 1.0
        xi = 2 * xi + 1.0
    w = np.full(N, 1.0 / N)
    res = {}
    for name, kk in (("in", 1), ("out", K + 1)):
        dpsi = -2 * np.pi * kk * np.sin(2 * np.pi * kk * x)
        res[f"LAG_{name}"] = np.sum(w * xi * dpsi)
        res[f"CONF_{name}"] = np.sum(w * gpt_conf(x, w, xi, K) * dpsi)
        res[f"BRK_{name}"] = np.sum(w * gpt_broken(x, w, xi, M) * dpsi)
    res["xi"] = xi[0]
    return res


if __name__ == "__main__":
    print("exact d<psi>/dtheta = 0 for every psi")
    for mode, eps in (("random", 0.0), ("lattice", 0.0), ("lattice", 0.3)):
        for N in (2001, 20001):
            for n in (5, 10, 15, 20):
                vals = [run(N, n, mode, eps=eps) for _ in range(5)]
                r = {k: np.sqrt(np.mean([v[k] ** 2 for v in vals])) for k in vals[0]}
                print(f"{mode:7s} eps={eps:.1f} N={N:6d} n={n:2d} xi={r['xi']:.2e} | rms: "
                      f"LAG_in {r['LAG_in']:.2e} CONF_in {r['CONF_in']:.2e} BRK_in {r['BRK_in']:.2e} | "
                      f"LAG_out {r['LAG_out']:.2e} CONF_out {r['CONF_out']:.2e} BRK_out {r['BRK_out']:.2e} | "
                      f"xi/sqrtN {r['xi']/np.sqrt(N):.2e}", flush=True)
