"""1D-1V test-particle model: bias and variance of grad_v f regularizations.

Model (normalized): x periodic on [0, 2 pi), dv/dt = F(x, t; A)
    F = -A sin(x) - B sin(k2 x - Om t)
Parameter theta = A. Two waves with overlapping resonances give a chaotic
layer, like trapping near a separatrix. Fields are prescribed (no
self-consistency), so delta F = dF/dA = -sin(x) exactly.

Observables at T (smooth, like the twin selection):
    sel(v) = S5((v - vc + 2 du) / (4 du))
    J1 = <sel>,  J2 = <sel * v>,  J3 = <v^2>
Estimators of dJ/dA:
  ref   : smooth secant, quadratic fit of J(A) over many large ensembles
  lag   : Lagrangian tangent (particle-exact), mean of grad(phi) . xi
  pair  : particle-pdf weak form with symmetric dipole pairs at v +- sigma
  coll  : particle-pdf, collocated, phase-space mesh (Chung 2020 eqs 36, 41, 42)
f0 is either cold (v = v0) or warm (v = v0 + vT * normal).
"""

import json
import sys
import time

import numpy as np

A0, B, K2, OM = 1.0, 0.2, 1.0, 2.2
DT, T = 0.05, 30.0
NSTEP = int(round(T / DT))
VC, DU = 2.0, 0.1
V0 = 1.0


def S5(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (10 - 15 * t + 6 * t * t)


def dS5(t):
    inside = (t > 0) & (t < 1)
    return np.where(inside, 30 * t * t * (1 - t) ** 2, 0.0)


def obs_point(x, v):
    """Per-particle observable values, shape (3, n)."""
    s = S5((v - VC + 2 * DU) / (4 * DU))
    return np.stack([s, s * v, v * v])


def obs_grad_v(x, v):
    t = (v - VC + 2 * DU) / (4 * DU)
    s = S5(t)
    ds = dS5(t) / (4 * DU)
    return np.stack([ds, ds * v + s, 2 * v])


def force(x, t, A):
    return -A * np.sin(x) - B * np.sin(K2 * x - OM * t)


def dforce_dx(x, t, A):
    return -A * np.cos(x) - B * K2 * np.cos(K2 * x - OM * t)


def push(x, v, t0_step, n_steps, A):
    """Leapfrog (kick-drift-kick) from step t0_step for n_steps."""
    for k in range(t0_step, t0_step + n_steps):
        t = k * DT
        v = v + 0.5 * DT * force(x, t, A)
        x = x + DT * v
        v = v + 0.5 * DT * force(x, t + DT, A)
    return x, v


def init(n, rng, vT, stratified=False):
    if stratified:
        x = 2 * np.pi * (np.arange(n) + rng.random()) / n
    else:
        x = 2 * np.pi * rng.random(n)
    v = V0 + vT * rng.standard_normal(n) if vT > 0 else np.full(n, V0)
    return x, v


def J_of_A(A, n, seed, vT):
    rng = np.random.default_rng(seed)
    x, v = init(n, rng, vT, stratified=True)
    x, v = push(x, v, 0, NSTEP, A)
    return obs_point(x, v).mean(axis=1)


def reference(vT, n=200_000, half=0.1, npts=9, seed0=100):
    As = np.linspace(A0 - half, A0 + half, npts)
    Js = np.array([J_of_A(a, n, seed0 + i, vT) for i, a in enumerate(As)])
    slopes = []
    for j in range(3):
        c = np.polyfit(As - A0, Js[:, j], 2)
        slopes.append(c[1])
    # secant FD at two step sizes with common random numbers
    fd = {}
    for h in (0.05, 0.02):
        Jp = J_of_A(A0 + h, n, 7, vT)
        Jm = J_of_A(A0 - h, n, 7, vT)
        fd[h] = ((Jp - Jm) / (2 * h)).tolist()
    return np.array(slopes), Js, fd


def lagrangian(n, seed, vT):
    """Particle-exact tangent: returns per-particle contributions (3, n)."""
    rng = np.random.default_rng(seed)
    x, v = init(n, rng, vT)
    xi_x = np.zeros(n)
    xi_v = np.zeros(n)
    growth = []
    for k in range(NSTEP):
        t = k * DT
        xi_v = xi_v + 0.5 * DT * (dforce_dx(x, t, A0) * xi_x - np.sin(x))
        v = v + 0.5 * DT * force(x, t, A0)
        x = x + DT * v
        xi_x = xi_x + DT * xi_v
        xi_v = xi_v + 0.5 * DT * (dforce_dx(x, t + DT, A0) * xi_x - np.sin(x))
        v = v + 0.5 * DT * force(x, t + DT, A0)
        if (k + 1) % 100 == 0:
            growth.append(float(np.median(np.hypot(xi_x, xi_v))))
    # observables depend on v only
    c = obs_grad_v(x, v) * xi_v
    return c, growth


def pairs(n, seed, vT, sigmas, every=4):
    """Weak-form particle-pdf with dipole pairs. Returns (n_sigma, 3, n)."""
    rng = np.random.default_rng(seed)
    x, v = init(n, rng, vT)
    ns = len(sigmas)
    sig = np.asarray(sigmas)[:, None]
    contrib = np.zeros((ns, 3, n))
    # pair particles: arrays grow; keep (owner, sign, sigma idx, weight)
    px = np.empty(0)
    pv = np.empty(0)
    pw = np.empty((0,))
    pown = np.empty(0, dtype=np.int64)
    psig = np.empty(0, dtype=np.int64)
    for k in range(NSTEP):
        if k % every == 0:
            # source over [t, t + every*DT]: midpoint value at the current state
            dF = -np.sin(x)  # dF/dA at the particle
            wgt = dF * every * DT / (2 * sig)  # (ns, n)
            for sgn in (+1.0, -1.0):
                px = np.concatenate([px, np.tile(x, ns)])
                pv = np.concatenate([pv, (v[None, :] + sgn * sig).ravel()])
                pw = np.concatenate([pw, (sgn * wgt).ravel()])
                pown = np.concatenate([pown, np.tile(np.arange(n), ns)])
                psig = np.concatenate([psig, np.repeat(np.arange(ns), n)])
        t = k * DT
        # push base particles and pair particles together
        v = v + 0.5 * DT * force(x, t, A0)
        x = x + DT * v
        v = v + 0.5 * DT * force(x, t + DT, A0)
        pv = pv + 0.5 * DT * force(px, t, A0)
        px = px + DT * pv
        pv = pv + 0.5 * DT * force(px, t + DT, A0)
    phi = obs_point(px, pv)  # (3, npairs)
    for j in range(3):
        np.add.at(contrib[:, j, :], (psig, pown), pw * phi[j])
    return contrib


def bspline2_deriv(z):
    """Derivative of the quadratic B-spline (support |z| < 3/2), eq. (A.2) shape."""
    a = np.abs(z)
    d = np.where(a <= 0.5, -2 * z, 0.0)
    d = np.where((a > 0.5) & (a <= 1.5), -np.sign(z) * (1.5 - a), d)
    return d


def collocated(n, seed, vT, nx, dv, every=4):
    """Collocated particle-pdf with a phase-space mesh (eqs 36, 41, 42).

    Shapes: x linear hat for H and n-hat; v: d/dv quadratic B-spline for H,
    linear hat for n-hat and the interpolation back (b_{l+1}, l = 0 in v).
    Returns per-particle contributions (3, n) and the lost-source fraction.
    """
    rng = np.random.default_rng(seed)
    x, v = init(n, rng, vT)
    dx = 2 * np.pi / nx
    w = 1.0 / n
    hsum = np.zeros(n)  # integral of h_s dt
    lost = []
    xs, vs = [], []
    for k in range(NSTEP):
        if k % every == 0:
            vmin = v.min() - 3 * dv
            nv = int(np.ceil((v.max() + 3 * dv - vmin) / dv)) + 1
            # x weights (linear, periodic)
            gx = x / dx
            ix0 = np.floor(gx).astype(int)
            fx = gx - ix0
            ixs = [ix0 % nx, (ix0 + 1) % nx]
            wxs = [1 - fx, fx]
            gv = (v - vmin) / dv
            # n-hat (linear in v)
            iv0 = np.floor(gv).astype(int)
            fv = gv - iv0
            ivs_n = [iv0, iv0 + 1]
            wvs_n = [1 - fv, fv]
            nhat = np.zeros((nx, nv))
            for a in range(2):
                for b in range(2):
                    np.add.at(nhat, (ixs[a], ivs_n[b]), wxs[a] * wvs_n[b])
            nhat /= dx * dv
            # H = dF . d_v f on the mesh; dF on mesh nodes = -sin(x_i)
            ivc = np.rint(gv).astype(int)
            H = np.zeros((nx, nv))
            for a in range(2):
                dF_node = -np.sin(ixs[a] * dx)
                for off in (-1, 0, 1):
                    iv = ivc + off
                    # d/dv S(v_i - v_p) = S'(zeta)/dv, zeta = (v_i - v_p)/dv
                    zeta = iv - gv
                    H_add = w * wxs[a] * dF_node * bspline2_deriv(zeta) / dv
                    np.add.at(H, (ixs[a], iv), H_add / (dx * dv))
            has_src = np.abs(H) > 0
            lost_mass = np.abs(H[has_src & (nhat <= 0)]).sum()
            lost.append(float(lost_mass / max(np.abs(H).sum(), 1e-300)))
            h_mesh = np.where(nhat > 0, H / np.where(nhat > 0, nhat, 1.0), 0.0)
            hs = np.zeros(n)
            for a in range(2):
                for b in range(2):
                    hs += h_mesh[ixs[a], ivs_n[b]] * wxs[a] * wvs_n[b]
            hsum += hs * every * DT
        t = k * DT
        v = v + 0.5 * DT * force(x, t, A0)
        x = x + DT * v
        v = v + 0.5 * DT * force(x, t + DT, A0)
    phi = obs_point(x, v)
    c = -hsum[None, :] * phi * n  # per-particle, mean over n gives the sum
    return c, float(np.mean(lost)), float(np.max(lost))


def summarize(c):
    """Mean and standard error over the last axis."""
    n = c.shape[-1]
    return c.mean(axis=-1), c.std(axis=-1, ddof=1) / np.sqrt(n)


def main(out_path):
    res = {"params": dict(A0=A0, B=B, K2=K2, OM=OM, DT=DT, T=T, VC=VC, DU=DU, V0=V0)}
    for label, vT in (("cold", 0.0), ("warm", 0.3)):
        t0 = time.time()
        ref, Js, fd = reference(vT)
        r = {"ref_slope": ref.tolist(), "J_center": Js[len(Js) // 2].tolist(), "fd_secant": fd}
        print(label, "ref", ref, "fd", fd, f"{time.time() - t0:.0f}s", flush=True)
        c, growth = lagrangian(20000, 1, vT)
        m, se = summarize(c)
        r["lagrangian"] = {"mean": m.tolist(), "se": se.tolist(), "median_xi_every_100": growth}
        print(label, "lag", m, se, flush=True)
        sigmas = [1e-6, 0.01, 0.03, 0.1, 0.3]
        cp = pairs(4000, 2, vT, sigmas)
        r["pairs"] = {}
        for i, s in enumerate(sigmas):
            m, se = summarize(cp[i])
            r["pairs"][str(s)] = {"mean": m.tolist(), "se": se.tolist()}
            print(label, "pair", s, m, se, flush=True)
        r["collocated"] = {}
        for nx, dv in ((64, 0.05), (64, 0.1), (32, 0.2)):
            cc, lost_mean, lost_max = collocated(20000, 3, vT, nx, dv)
            m, se = summarize(cc)
            r["collocated"][f"nx{nx}_dv{dv}"] = {
                "mean": m.tolist(), "se": se.tolist(),
                "lost_source_fraction_mean": lost_mean, "lost_max": lost_max,
            }
            print(label, "coll", nx, dv, m, se, lost_mean, flush=True)
        res[label] = r
        print(label, f"done {time.time() - t0:.0f}s", flush=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
