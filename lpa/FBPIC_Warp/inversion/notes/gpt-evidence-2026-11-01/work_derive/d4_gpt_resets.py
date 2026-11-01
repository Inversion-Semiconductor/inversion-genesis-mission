"""D4: discrete GPT with periodic projections on the standard map.

Cases:
  sheet: cold sheet x0 = 2 pi u, p0 = 0.3, K = 2 (continuum gauge is 1-D).
  blob : warm wrapped Gaussian of R0 E2 (x ~ N(2, 0.3^2), p ~ N(1, 0.3^2)),
         K = 1.5 (continuum gauge is infinite-dimensional).
Every T steps (k % T == 0, k < n_f): xi <- P xi, the weighted least-squares
projection onto gradients of a basis:
  conf: periodic Fourier basis on the torus, |kx|, |kp| <= Kf (conforming).
  brk : per-cell polynomials of degree 2 on an M x M mesh (broken).
Estimate: d<psi>/dK = sum w xi . grad psi at n_f, psi = cos x + 0.5 sin p
(psi is in the conf span). Reference: Lagrangian on a 2^22 label lattice for
the sheet (equal to FD there, see d2), and the mean of 8 Lagrangian runs with
N = 1e6 random particles for the blob (its standard error is printed).
"""
import sys
import numpy as np

TWO_PI = 2 * np.pi
rng = np.random.default_rng(11)


def init(case, n, lattice=False):
    if case == "sheet":
        u = (np.arange(n) + 0.5) / n if lattice else rng.random(n)
        return TWO_PI * u, np.full(n, 0.3)
    return (2.0 + 0.3 * rng.standard_normal(n)) % TWO_PI, (1.0 + 0.3 * rng.standard_normal(n)) % TWO_PI


def fourier_grads(x, p, Kf):
    ks = [(a, b) for a in range(0, Kf + 1) for b in range(-Kf, Kf + 1) if (a > 0 or b > 0)]
    kx = np.array([k[0] for k in ks], float)
    kp = np.array([k[1] for k in ks], float)
    ph = np.outer(x, kx) + np.outer(p, kp)
    s, c = np.sin(ph), np.cos(ph)
    # basis cos(ph), sin(ph); gradients
    gx = np.concatenate([-kx * s, kx * c], axis=1)
    gp = np.concatenate([-kp * s, kp * c], axis=1)
    return gx, gp


def proj_conf(x, p, dx, dp, w, Kf):
    gx, gp = fourier_grads(x % TWO_PI, p % TWO_PI, Kf)
    sw = np.sqrt(w)[:, None]
    G = (gx * w[:, None]).T @ gx + (gp * w[:, None]).T @ gp
    b = gx.T @ (w * dx) + gp.T @ (w * dp)
    c = np.linalg.solve(G + 1e-14 * np.trace(G) / len(G) * np.eye(len(G)), b)
    return gx @ c, gp @ c


def proj_brk(x, p, dx, dp, w, M):
    h = TWO_PI / M
    xm, pm = x % TWO_PI, p % TWO_PI
    ix = np.minimum((xm / h).astype(int), M - 1)
    ip = np.minimum((pm / h).astype(int), M - 1)
    cell = ix * M + ip
    order = np.argsort(cell)
    bounds = np.searchsorted(cell[order], np.arange(M * M + 1))
    ox, op = np.zeros_like(dx), np.zeros_like(dp)
    for m in range(M * M):
        idx = order[bounds[m]:bounds[m + 1]]
        if idx.size == 0:
            continue
        t = xm[idx] / h - (m // M) - 0.5
        s = pm[idx] / h - (m % M) - 0.5
        # gradients of t, s, t^2, ts, s^2 (scaled by 1/h)
        one, zero = np.ones_like(t), np.zeros_like(t)
        gx = np.stack([one, zero, 2 * t, s, zero], 1) / h
        gp = np.stack([zero, one, zero, t, 2 * s], 1) / h
        sw = np.sqrt(w[idx])[:, None]
        A = np.concatenate([gx * sw, gp * sw], 0)
        b = np.concatenate([dx[idx] * sw[:, 0], dp[idx] * sw[:, 0]])
        c, *_ = np.linalg.lstsq(A, b, rcond=1e-10)
        ox[idx], op[idx] = gx @ c, gp @ c
    return ox, op


def estimate(case, n, K, nf, T, kind, par, lattice=False):
    x, p = init(case, n, lattice)
    w = np.full(n, 1.0 / n)
    dx, dp = np.zeros(n), np.zeros(n)
    removed = []
    for k in range(1, nf + 1):
        s, c = np.sin(x), np.cos(x)
        dp = dp + s + K * c * dx
        p = p + K * s
        dx = dx + dp
        x = x + p
        if T and k % T == 0 and k < nf:
            nx, np_ = (proj_conf if kind == "conf" else proj_brk)(x, p, dx, dp, w, par)
            before = np.sum(w * (dx * dx + dp * dp))
            removed.append(np.sum(w * ((dx - nx) ** 2 + (dp - np_) ** 2)) / max(before, 1e-300))
            dx, dp = nx, np_
    val = np.sum(w * (-np.sin(x) * dx + 0.5 * np.cos(p) * dp))
    return float(val), float(np.sqrt(np.sum(w * (dx * dx + dp * dp)))), float(np.mean(removed)) if removed else 0.0


if __name__ == "__main__":
    case = sys.argv[1]
    K = 2.0 if case == "sheet" else 1.5
    for nf in [int(a) for a in sys.argv[2].split(",")]:
        if case == "sheet":
            ref, _, _ = estimate(case, 2 ** 22, K, nf, 0, "conf", 0, lattice=True)
        else:
            rr = [estimate(case, 1_000_000, K, nf, 0, "conf", 0)[0] for _ in range(8)]
            ref = np.mean(rr)
            print(f"   reference SE {np.std(rr, ddof=1) / np.sqrt(8):.4f}")
        print(f"== {case} K={K} n_f={nf}: reference {ref:+.4f}", flush=True)
        for n in (20_000,):
            configs = [("lag", 0, "conf", 0), ("conf Kf=4 T=1", 1, "conf", 4), ("conf Kf=8 T=1", 1, "conf", 8),
                       ("conf Kf=8 T=3", 3, "conf", 8), ("brk M=16 T=1", 1, "brk", 16)]
            for name, T, kind, par in configs:
                nseed = 5
                r = [estimate(case, n, K, nf, T, kind, par) for _ in range(nseed)]
                v = np.array([a[0] for a in r])
                print(f"  N={n:7d} {name:18s} mean {v.mean():+.4f} sd {v.std(ddof=1):.4f} "
                      f"err {v.mean()-ref:+.4f} |xi|_end {np.mean([a[1] for a in r]):.2e} "
                      f"removed-frac {np.mean([a[2] for a in r]):.3f}", flush=True)
