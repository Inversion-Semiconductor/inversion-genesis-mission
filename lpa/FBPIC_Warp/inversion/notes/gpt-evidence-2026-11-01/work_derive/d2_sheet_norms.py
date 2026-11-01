"""D2: exact cold sheet under the standard map. Does the L^2(f) gradient
projection (Wasserstein tangent) stop the growth?

Map: p' = p + K sin x, x' = x + p' (torus 2pi x 2pi, unwrapped for tangents).
Sheet: x0 = 2 pi u, p0 = P0, label u in [0,1), label density n0 = 1.
It is a closed curve that winds once in x, so the continuum gauge on it is
one-dimensional: eta_t = C / n_s (n_s = line density per arc length).
Tangents: xi = dz/dK (source dp += sin x); F = dz/du (homogeneous).
Per step, on a fine label lattice (continuum proxy):
  |xi|, normal part xi_n = xi x Fhat, tangential xi_t = xi . Fhat,
  the minimal (Wasserstein) norm |P xi| = sqrt(|xi_n|^2 + |xi_t - C|F||^2),
  the label displacement zeta = xi_t/|F| and the label density change -dzeta/du,
  the stretching |F|.
Observable psi = cos x + 0.5 sin p. d<psi>/dK by:
  Lagrangian on the fine lattice, label integration by parts on the fine
  lattice, and central FD on the fine lattice (h = 1e-7).
"""
import sys
import numpy as np

K = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 14
P0 = 0.3
NL = 2 ** 22


def rms(a):
    return float(np.sqrt(np.mean(a * a)))


def run(Kv, nsteps, u, track=True):
    x = 2 * np.pi * u
    p = np.full_like(u, P0)
    dx, dp = np.zeros_like(u), np.zeros_like(u)
    fx, fp = np.full_like(u, 2 * np.pi), np.zeros_like(u)
    rows = []
    for n in range(1, nsteps + 1):
        s, c = np.sin(x), np.cos(x)
        dp = dp + s + Kv * c * dx
        fp = fp + Kv * c * fx
        p = p + Kv * s
        dx = dx + dp
        fx = fx + fp
        x = x + p
        if track:
            Fn = np.hypot(fx, fp)
            tx, tp = fx / Fn, fp / Fn
            xit = dx * tx + dp * tp
            xin = dx * tp - dp * tx
            C = np.mean(xit * Fn) / np.mean(Fn ** 2)
            pt = xit - C * Fn
            zeta = xit / Fn
            dn = -np.gradient(zeta, u[1] - u[0])
            rows.append(dict(n=n, xi=rms(np.hypot(dx, dp)), xin=rms(xin), xit=rms(xit),
                             wass=float(np.sqrt(np.mean(xin ** 2 + pt ** 2))), gauge=rms(C * Fn),
                             F=rms(Fn), zeta=rms(zeta), dn=rms(dn), xin_F=rms(xin * Fn),
                             maxFdu=float(Fn.max() / NL)))
    return x, p, dx, dp, fx, fp, rows


if __name__ == "__main__":
    u = (np.arange(NL) + 0.5) / NL
    x, p, dx, dp, fx, fp, rows = run(K, NF, u)
    print(f"K={K} P0={P0} NL={NL}")
    print(" n   |xi|       |xi_n|     |xi_t|     |P xi|     |gauge|    |F|        |zeta|     |dn_lab|   |xi_n F|  max|F|du")
    for r in rows:
        print(f"{r['n']:2d} {r['xi']:.3e} {r['xin']:.3e} {r['xit']:.3e} {r['wass']:.3e} {r['gauge']:.3e} "
              f"{r['F']:.3e} {r['zeta']:.3e} {r['dn']:.3e} {r['xin_F']:.3e} {r['maxFdu']:.1e}")
    psi = np.cos(x) + 0.5 * np.sin(p)
    gx, gp = -np.sin(x), 0.5 * np.cos(p)
    lag = np.mean(dx * gx + dp * gp)
    Fn = np.hypot(fx, fp)
    tx, tp = fx / Fn, fp / Fn
    xit, xin = dx * tx + dp * tp, dx * tp - dp * tx
    zeta = xit / Fn
    dn = -np.gradient(zeta, u[1] - u[0])
    ibp = np.mean(dn * psi) + np.mean(xin * (gx * tp - gp * tx))
    h = 1e-7
    xa, pa, *_ = run(K + h, NF, u, track=False)
    xb, pb, *_ = run(K - h, NF, u, track=False)
    fd = (np.mean(np.cos(xa) + 0.5 * np.sin(pa)) - np.mean(np.cos(xb) + 0.5 * np.sin(pb))) / (2 * h)
    print(f"d<psi>/dK at n={NF}: Lagrangian(fine) {lag:+.6f}  label-IBP(fine) {ibp:+.6f}  FD(fine,h=1e-7) {fd:+.6f}")
    print(f"  normal part contributes {np.mean(xin * (gx * tp - gp * tx)):+.6f}")
