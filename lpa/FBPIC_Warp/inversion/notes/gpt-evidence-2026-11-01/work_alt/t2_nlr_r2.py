"""T2: noise-injected likelihood ratio (NLR) on the self-consistent R2 pilot.

Model: copy of r0 work_tests/r2_pilot.py (1D relativistic ES wake, prescribed
ponderomotive driver, EV ionization injection, cold plasma), theta = a0.
Adds an exact forward tangent (field tangent from delta rho, Lagrangian
particle tangents, EV birth-weight tangent) and an optional NLR mode:

  u^{n+1} = mu(z^n, theta) + sigma sqrt(DT) eps,  eps ~ N(0, 1) (counter hash)
  dW_i   += W_i eps_i (delta mu_i) / (sigma sqrt(DT)),  delta u^{n+1} = 0

for the species listed in `noisy` ("b" = EV bunch, "e" = plasma electrons).
delta mu includes the field tangent and the position tangent. The position
tangent then only accumulates the explicit ponderomotive term (gamma(A)).

Modes:
  tan  a0 tf sigma noisy seed out.json : value and tangent column
  val  a0 tf sigma noisy seed          : value only (for CRN FD of the noisy system)
"""

import json
import sys

import numpy as np

sys.path.insert(0, "inversion/notes/gpt-evidence-2026-11-01/work_alt")
sys.dont_write_bytecode = True
import r2_pilot_copy as P  # noqa: E402

DX, DT, X0, NX = P.DX, P.DT, P.X0, P.NX
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix(z):
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & M64
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & M64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & M64
    return z ^ (z >> np.uint64(31))


def normals(ids, step, seed):
    """Counter-based N(0,1) per (id, step, seed): common random numbers across theta."""
    with np.errstate(over="ignore"):
        base = (ids.astype(np.uint64) * np.uint64(1000003) + np.uint64(step)) * np.uint64(7919) + np.uint64(seed) * np.uint64(0x632BE59BD9B4E019)
        h1 = splitmix(base & M64)
        h2 = splitmix(h1 ^ np.uint64(0xD1B54A32D192ED03))
    u1 = ((h1 >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0**53
    u2 = ((h2 >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0**53
    return np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)


def deposit(x, W):
    i, d = P.wts(x)
    return np.bincount(i, W * (1 - d), NX + 1) + np.bincount(i + 1, W * d, NX + 1)


def deposit_tan(x, W, dx, dW):
    i, d = P.wts(x)
    a = dW * (1 - d) - W * dx / DX
    b = dW * d + W * dx / DX
    return np.bincount(i, a, NX + 1) + np.bincount(i + 1, b, NX + 1)


def field(rho):
    seg = 0.5 * (rho[1:] + rho[:-1]) * DX
    return -np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])


def run(a0, tf, sigma=0.0, noisy="", seed=0, tangent=True, rec=None):
    xs = np.arange(P.RAMP[0], P.X1 - 1.0, DX / 20) + 0.5 * DX / 20
    W = P.dens(xs) * DX / 20
    keep = W > 0
    xe, We = xs[keep], W[keep]
    ue = np.zeros_like(xe)
    ide = np.arange(xe.size, dtype=np.int64)
    ion = deposit(xe, We)
    xi_ = np.arange(P.DOPE[0], P.DOPE[1], DX / 4) + DX / 8
    Wi = P.NDOPE * DX / 4 * np.ones_like(xi_)
    rem = np.ones_like(xi_)
    nexe = xe.size
    # tangents
    dxe, due, dWe = np.zeros_like(xe), np.zeros_like(xe), np.zeros_like(xe)
    dion = np.zeros(NX + 1)
    drem = np.zeros_like(xi_)
    xb = np.zeros(0); ub = np.zeros(0); Wb = np.zeros(0); idb = np.zeros(0, np.int64)
    dxb = np.zeros(0); dub = np.zeros(0); dWb = np.zeros(0)
    nt = int(round(tf / DT))
    sq = np.sqrt(DT)
    log = []
    for n in range(nt):
        t = n * DT
        Ai, _ = P.A_of(xi_, t, a0)
        a_loc = np.sqrt(2 * Ai)
        Gam = P.G0 * np.exp(-P.BION / np.maximum(a_loc, 1e-12))
        e = np.exp(-Gam * DT)
        frac = rem * (1 - e)
        m = frac > P.EPS
        if m.any():
            dGam = Gam * P.BION / np.maximum(a_loc, 1e-12) ** 2 * (a_loc / a0)
            dfrac = drem * (1 - e) + rem * e * DT * dGam
            dW = Wi[m] * frac[m]
            ddW = Wi[m] * dfrac[m]
            rem[m] -= frac[m]
            drem[m] -= dfrac[m]
            k = m.sum()
            xb = np.concatenate([xb, xi_[m]]); ub = np.concatenate([ub, np.zeros(k)])
            Wb = np.concatenate([Wb, dW]); idb = np.concatenate([idb, nexe + np.nonzero(m)[0] * 10**4 + n])
            dxb = np.concatenate([dxb, np.zeros(k)]); dub = np.concatenate([dub, np.zeros(k)])
            dWb = np.concatenate([dWb, ddW])
            ion += deposit(xi_[m], dW)
            dion += deposit(xi_[m], ddW)
        x = np.concatenate([xe, xb]); u = np.concatenate([ue, ub]); Wt = np.concatenate([We, Wb])
        ne = deposit(x, Wt)
        rho = (ion - ne) / DX
        E = field(rho)
        i, d = P.wts(x)
        Ep = E[i] * (1 - d) + E[i + 1] * d
        A, dA = P.A_of(x, t, a0)
        g0 = np.sqrt(1 + u ** 2 + A)
        f0 = -Ep - dA / (2 * g0)
        uh = u + 0.5 * DT * f0
        gh = np.sqrt(1 + uh ** 2 + A)
        fh = -Ep - dA / (2 * gh)
        mu = u + DT * fh
        if tangent:
            dx_ = np.concatenate([dxe, dxb]); du_ = np.concatenate([due, dub]); dW_ = np.concatenate([dWe, dWb])
            dne = deposit_tan(x, Wt, dx_, dW_)
            dE = field((dion - dne) / DX)
            dEp = dE[i] * (1 - d) + dE[i + 1] * d + (E[i + 1] - E[i]) / DX * dx_
            S2 = P.S_L ** 2
            xi = x - (P.XD0 + P.BETA_G * t)
            dAx = -A / S2 - xi / S2 * dA
            dAA = dA * dx_ + 2 * A / a0
            ddA = dAx * dx_ + 2 * dA / a0
            dg0 = (2 * u * du_ + dAA) / (2 * g0)
            df0 = -dEp - ddA / (2 * g0) + dA * dg0 / (2 * g0 ** 2)
            duh = du_ + 0.5 * DT * df0
            dgh = (2 * uh * duh + dAA) / (2 * gh)
            dfh = -dEp - ddA / (2 * gh) + dA * dgh / (2 * gh ** 2)
            dmu = du_ + DT * dfh
        u1 = mu.copy()
        if sigma > 0 and noisy:
            ne_ = xe.size
            ids = np.concatenate([ide, idb])
            sel = np.zeros(x.size, bool)
            if "e" in noisy:
                sel[:ne_] = True
            if "b" in noisy:
                sel[ne_:] = True
            eps = normals(ids[sel], n, seed)
            u1[sel] += sigma * sq * eps
        g1 = np.sqrt(1 + u1 ** 2 + A)
        x1 = x + DT * u1 / g1
        if tangent:
            du1 = dmu.copy()
            if sigma > 0 and noisy:
                dW_[sel] += Wt[sel] * eps * dmu[sel] / (sigma * sq)
                du1[sel] = 0.0
            dg1 = (2 * u1 * du1 + dAA) / (2 * g1)
            dx1 = dx_ + DT * (du1 / g1 - u1 * dg1 / g1 ** 2)
        ne_ = xe.size
        xe, xb = x1[:ne_], x1[ne_:]; ue, ub = u1[:ne_], u1[ne_:]
        if tangent:
            dxe, dxb = dx1[:ne_], dx1[ne_:]; due, dub = du1[:ne_], du1[ne_:]; dWe, dWb = dW_[:ne_], dW_[ne_:]
        ke = xe > X0 + 2 * DX
        kb = xb > X0 + 2 * DX
        if not ke.all():
            xe, ue, We, ide = xe[ke], ue[ke], We[ke], ide[ke]
            if tangent:
                dxe, due, dWe = dxe[ke], due[ke], dWe[ke]
        if not kb.all():
            xb, ub, Wb, idb = xb[kb], ub[kb], Wb[kb], idb[kb]
            if tangent:
                dxb, dub, dWb = dxb[kb], dub[kb], dWb[kb]
        if tangent and (n + 1) % 200 == 0:
            log.append(dict(t=(n + 1) * DT,
                            e_dx_med=float(np.median(np.abs(dxe))), e_dx_p90=float(np.quantile(np.abs(dxe), 0.9)),
                            b_dx_med=float(np.median(np.abs(dxb))) if dxb.size else 0.0,
                            b_du_med=float(np.median(np.abs(dub))) if dub.size else 0.0,
                            e_dw_rel_med=float(np.median(np.abs(dWe / We))),
                            b_dw_rel_p90=float(np.quantile(np.abs(dWb / Wb), 0.9)) if Wb.size else 0.0,
                            dE_max=float(np.abs(dE).max())))
    xd = P.XD0 + P.BETA_G * tf
    obs = P.observables(dict(ub=ub, Wb=Wb, xb=xb, xd=xd), uc=3.0, du=0.3)
    out = dict(a0=a0, tf=tf, sigma=sigma, noisy=noisy, seed=seed, obs=obs.tolist())
    if tangent:
        h = 1e-6
        op = P.observables(dict(ub=ub + h * dub, Wb=Wb + h * dWb, xb=xb + h * dxb, xd=xd), uc=3.0, du=0.3)
        om = P.observables(dict(ub=ub - h * dub, Wb=Wb - h * dWb, xb=xb - h * dxb, xd=xd), uc=3.0, du=0.3)
        out["dobs"] = ((op - om) / (2 * h)).tolist()
        out["log"] = log
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    a0, tf, sigma, noisy, seed = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], int(sys.argv[6])
    noisy = "" if noisy == "none" else noisy
    if mode == "tan":
        r = run(a0, tf, sigma, noisy, seed, tangent=True)
    else:
        r = run(a0, tf, sigma, noisy, seed, tangent=False)
    print(json.dumps(r))
