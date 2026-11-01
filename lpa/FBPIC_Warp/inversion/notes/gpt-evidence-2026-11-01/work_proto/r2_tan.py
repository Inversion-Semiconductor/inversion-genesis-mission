"""R2 pilot (copied from r0/work_tests/r2_pilot.py, same physics and numerics) with an exact
forward tangent in theta = a0 and optional gauge-projected tangent (GPT) resets.

1D relativistic electrostatic wake, prescribed ponderomotive driver, EV ionization injection,
c = omega_p = k_p = 1, lab frame. Tangent state: plasma electrons (dx, du), beam (dx, du, dW),
injector remainder d rem, ion background d ion. Births carry dx = du = 0 (ions are fixed).

GPT: every T steps for t >= t_on, per projected species, weighted least squares of the tangent
(dx, du) onto gradients of quadratic tensor B-splines in (zeta = x - x_d, s = asinh(u/u0)),
metric diag(1/Lx^2, 1/Lu^2). dW is not touched.

CLI:
  python r2_tan.py run <a0> <seed|none> <tf> <ppc> <mode> <out.json>
     mode = exact | none | gpt:<T>:<dz>:<Mu>:<species>[:<Lu>]   (species = all | beam)
"""
import sys, json, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gpt_lib import bspline2d_design, project_sparse

DX = 0.025; DT = 0.025
X0, X1 = -6.0, 95.0
NX = int(round((X1 - X0) / DX))
GAM_G = 10.0; BETA_G = np.sqrt(1 - 1 / GAM_G ** 2)
S_L = 1.0
XD0 = -4.0
RAMP = (0.0, 3.0)
DOPE = (8.0, 14.0); NDOPE = 0.03
G0, BION = 10.0, 8.0
EPS = 1e-4
U0 = 0.5  # asinh scale for the u basis coordinate


def dens(x):
    return np.clip((x - RAMP[0]) / (RAMP[1] - RAMP[0]), 0, 1)


def A_of(x, t, a0):
    xi = x - (XD0 + BETA_G * t)
    A = 0.5 * a0 ** 2 * np.exp(-xi ** 2 / (2 * S_L ** 2))
    F = -xi / S_L ** 2 * A
    return A, F, xi


def wts(x):
    s = (x - X0) / DX
    i = np.floor(s).astype(np.int64)
    return i, s - i


def efield(rho):
    seg = 0.5 * (rho[1:] + rho[:-1]) * DX
    return -np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])


def S5(t):
    tr = np.real(t)
    poly = t ** 3 * (10 - 15 * t + 6 * t * t)
    return np.where(tr <= 0, 0.0, np.where(tr >= 1, 1.0, poly))


def observables(ub, Wb, xb, xd, uc=3.0, du=0.3):
    """Same as r2_pilot.observables (uc=3, du=0.3); complex-step safe."""
    sig = S5((ub - uc + 2 * du) / (4 * du))
    w = Wb * sig; S0 = w.sum(); D = S0 + 1e-6
    xi = xb - xd
    mu = (w * ub).sum() / D; mx = (w * xi).sum() / D
    su = np.sqrt((w * ub ** 2).sum() / D - mu ** 2); sx = np.sqrt((w * xi ** 2).sum() / D - mx ** 2)
    return np.array([np.log(S0 + 1e-6), mu, mx, np.log(su + 1e-3), np.log(sx + 1e-3), np.log((su + 1e-3) / (mu + 1e-2))])


def obs_and_tangent(ub, Wb, xb, xd, dub, dWb, dxb):
    h = 1e-30
    v = observables(ub + 1j * h * dub, Wb + 1j * h * dWb, xb + 1j * h * dxb, xd)
    return np.real(v), np.imag(v) / h


def gpt_project(x, u, w, dx, du, xd, dz, Mu, Lx=1.0, Lu=1.0):
    """Project (dx, du) of one species onto gradients of B-splines in (zeta, asinh(u/U0))."""
    if x.size < 10:
        return dx, du, 0.0
    z = x - xd; s = np.arcsinh(u / U0)
    zlo, zhi = z.min() - 1e-6, z.max() + 1e-6
    slo, shi = s.min() - 1e-6, s.max() + 1e-6
    Mx = max(1, int(np.ceil((zhi - zlo) / dz)))
    dsdu = 1.0 / np.sqrt(U0 ** 2 + u ** 2)
    Gx, Gp = bspline2d_design(z, s, zlo, zhi, Mx, slo, shi, Mu, periodic=False, sx=Lx, sp_=Lu * dsdu)
    (px, pu), _, frac = project_sparse([Gx, Gp], [dx / Lx, du / Lu], w)
    return px * Lx, pu * Lu, frac


def run(a0, ppc=20, seed=None, tf=75.0, mode="none", ion_ppc=4, t_on=15.0, log_every=200):
    rng = np.random.default_rng(seed) if seed is not None else None
    tangent = mode != "none"
    gpt = None
    if mode.startswith("gpt"):
        parts = mode.split(":")
        gpt = dict(T=int(parts[1]), dz=float(parts[2]), Mu=int(parts[3]), species=parts[4],
                   Lu=float(parts[5]) if len(parts) > 5 else 1.0)
        if len(parts) > 6:
            t_on = float(parts[6])  # optional start time of the projections
    xs = np.arange(RAMP[0], X1 - 1.0, DX / ppc) + 0.5 * DX / ppc
    if rng is not None:
        xs = xs + (rng.random(xs.size) - 0.5) * DX / ppc
    W = dens(xs) * DX / ppc
    keep = W > 0; xe = xs[keep]; We = W[keep]; ue = np.zeros_like(xe)
    ion = np.zeros(NX + 1)
    i, d = wts(xe); np.add.at(ion, i, We * (1 - d)); np.add.at(ion, i + 1, We * d)
    xi_ = np.arange(DOPE[0], DOPE[1], DX / ion_ppc) + 0.5 * DX / ion_ppc
    if rng is not None:
        xi_ = xi_ + (rng.random(xi_.size) - 0.5) * DX / ion_ppc
    Wi = NDOPE * DX / ion_ppc * np.ones_like(xi_); rem = np.ones_like(xi_)
    xb = np.zeros(0); ub = np.zeros(0); Wb = np.zeros(0)
    # tangents
    dxe = np.zeros_like(xe); due = np.zeros_like(xe)
    dxb = np.zeros(0); dub = np.zeros(0); dWb = np.zeros(0)
    drem = np.zeros_like(xi_); dion = np.zeros(NX + 1)
    nt = int(round(tf / DT)); log = []; gpt_frac = []; tproj = 0.0
    for n in range(nt):
        t = n * DT
        # ---- EV births
        Ai, _, _ = A_of(xi_, t, a0)
        a_loc = np.sqrt(2 * Ai)
        Gam = G0 * np.exp(-BION / np.maximum(a_loc, 1e-12))
        Ex = np.exp(-Gam * DT)
        frac = rem * (1 - Ex)
        m = frac > EPS
        if m.any():
            dW = Wi[m] * frac[m]
            rem[m] -= frac[m]
            xb = np.concatenate([xb, xi_[m]]); ub = np.concatenate([ub, np.zeros(m.sum())])
            Wb = np.concatenate([Wb, dW])
            ii, dd = wts(xi_[m]); np.add.at(ion, ii, dW * (1 - dd)); np.add.at(ion, ii + 1, dW * dd)
            if tangent:
                dGam = Gam[m] * BION / a_loc[m] ** 2 * (a_loc[m] / a0)
                dfrac = drem[m] * (1 - Ex[m]) + rem_before(rem, frac, m) * Ex[m] * DT * dGam
                ddW = Wi[m] * dfrac
                drem[m] -= dfrac
                dxb = np.concatenate([dxb, np.zeros(m.sum())]); dub = np.concatenate([dub, np.zeros(m.sum())])
                dWb = np.concatenate([dWb, ddW])
                np.add.at(dion, ii, ddW * (1 - dd)); np.add.at(dion, ii + 1, ddW * dd)
        # ---- field
        x_all = np.concatenate([xe, xb]); W_all = np.concatenate([We, Wb])
        i, d = wts(x_all)
        ne = np.bincount(i, W_all * (1 - d), NX + 1) + np.bincount(i + 1, W_all * d, NX + 1)
        E = efield((ion - ne) / DX)
        Ep = E[i] * (1 - d) + E[i + 1] * d
        A, F, zz = A_of(x_all, t, a0)
        u = np.concatenate([ue, ub])
        g0 = np.sqrt(1 + u ** 2 + A)
        f0 = -Ep - F / (2 * g0)
        uh = u + 0.5 * DT * f0
        gh = np.sqrt(1 + uh ** 2 + A)
        f1 = -Ep - F / (2 * gh)
        u1 = u + DT * f1
        g1 = np.sqrt(1 + u1 ** 2 + A)
        x1 = x_all + DT * u1 / g1
        if tangent:
            ddx = np.concatenate([dxe, dxb]); ddu = np.concatenate([due, dub])
            dWall = np.concatenate([np.zeros_like(We), dWb])
            dne = (np.bincount(i, dWall * (1 - d) - W_all * ddx / DX, NX + 1)
                   + np.bincount(i + 1, dWall * d + W_all * ddx / DX, NX + 1))
            dE = efield((dion - dne) / DX)
            dEp = dE[i] * (1 - d) + dE[i + 1] * d + (E[i + 1] - E[i]) / DX * ddx
            Ax = F; Aa = 2 * A / a0
            Fx = -A / S_L ** 2 - zz / S_L ** 2 * F; Fa = 2 * F / a0
            dA = Ax * ddx + Aa; dF = Fx * ddx + Fa
            dg0 = (2 * u * ddu + dA) / (2 * g0)
            df0 = -dEp - dF / (2 * g0) + F * dg0 / (2 * g0 ** 2)
            duh = ddu + 0.5 * DT * df0
            dgh = (2 * uh * duh + dA) / (2 * gh)
            df1 = -dEp - dF / (2 * gh) + F * dgh / (2 * gh ** 2)
            du1 = ddu + DT * df1
            dg1 = (2 * u1 * du1 + dA) / (2 * g1)
            dx1 = ddx + DT * (du1 / g1 - u1 * dg1 / g1 ** 2)
        ne_ = xe.size
        xe, xb = x1[:ne_], x1[ne_:]; ue, ub = u1[:ne_], u1[ne_:]
        if tangent:
            dxe, dxb = dx1[:ne_], dx1[ne_:]; due, dub = du1[:ne_], du1[ne_:]
        ke = xe > X0 + 2 * DX; kb = xb > X0 + 2 * DX
        if not ke.all():
            xe, ue, We = xe[ke], ue[ke], We[ke]
            if tangent: dxe, due = dxe[ke], due[ke]
        if not kb.all():
            xb, ub, Wb = xb[kb], ub[kb], Wb[kb]
            if tangent: dxb, dub, dWb = dxb[kb], dub[kb], dWb[kb]
        tn = (n + 1) * DT; xd = XD0 + BETA_G * tn
        if gpt is not None and tn >= t_on and (n + 1) % gpt["T"] == 0:
            t1 = time.time()
            dxb, dub, fb = gpt_project(xb, ub, Wb, dxb, dub, xd, gpt["dz"], gpt["Mu"], Lu=gpt["Lu"])
            fe = np.nan
            if gpt["species"] == "all":
                # project only the perturbed plasma electrons (zeta < 2); the rest have xi = 0 exactly
                sel = (xe - xd) < 2.0
                pxe, pue, fe = gpt_project(xe[sel], ue[sel], We[sel], dxe[sel], due[sel], xd, gpt["dz"], gpt["Mu"], Lu=gpt["Lu"])
                dxe = dxe.copy(); due = due.copy(); dxe[sel] = pxe; due[sel] = pue
            gpt_frac.append((tn, fb, fe)); tproj += time.time() - t1
        if tangent and ((n + 1) % log_every == 0 or n + 1 == nt):
            rec = dict(t=tn)
            for nm, a, b in (("e", dxe, due), ("b", dxb, dub)):
                nrm = np.hypot(a, b)
                rec["med_" + nm] = float(np.median(nrm)) if nrm.size else 0.0
                rec["p90_" + nm] = float(np.percentile(nrm, 90)) if nrm.size else 0.0
                rec["max_" + nm] = float(nrm.max()) if nrm.size else 0.0
            log.append(rec)
    xd = XD0 + BETA_G * tf
    out = dict(obs=observables(ub, Wb, xb, xd).tolist(), nb=int(xb.size), ne=int(xe.size), log=log,
               gpt_frac=gpt_frac, tproj=tproj)
    if tangent:
        v, dv = obs_and_tangent(ub, Wb, xb, xd, dub, dWb, dxb)
        out["dobs"] = dv.tolist()
    return out


def rem_before(rem, frac, m):
    # rem was already decremented by frac[m]; return the value before the decrement
    return rem[m] + frac[m]


if __name__ == "__main__":
    if sys.argv[1] == "run":
        a0 = float(sys.argv[2]); seed = None if sys.argv[3] == "none" else int(sys.argv[3])
        tf = float(sys.argv[4]); ppc = int(sys.argv[5]); mode = sys.argv[6]; outp = sys.argv[7]
        ion_ppc = int(sys.argv[8]) if len(sys.argv) > 8 else 4
        t0 = time.time()
        r = run(a0, ppc=ppc, seed=seed, tf=tf, mode=mode, ion_ppc=ion_ppc)
        r.update(a0=a0, seed=seed, tf=tf, ppc=ppc, mode=mode, ion_ppc=ion_ppc, time=time.time() - t0)
        json.dump(r, open(outp, "w"))
        print(json.dumps(dict(a0=a0, seed=seed, mode=mode, time=r["time"], tproj=r["tproj"], obs=r["obs"], dobs=r.get("dobs"))))
