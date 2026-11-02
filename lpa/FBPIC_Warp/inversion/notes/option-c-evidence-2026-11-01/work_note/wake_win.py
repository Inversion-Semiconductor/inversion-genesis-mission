"""Option C on the 1D cold wake with trapping (R2 pilot physics, restructured for CRN noise).

Physics and push are those of r0/work_tests/r2_pilot.py and mt/work_proto/r2_tan.py:
1D relativistic electrostatic wake, prescribed ponderomotive driver A = a0^2/2 exp(-xi^2/2),
EV ionization injection, c = omega_p = k_p = 1, lab frame, theta = a0.

Changes (all references use this same code, so they are consistent):
  * Domain front X1 = 80 (driver ends at x = 70.6); plasma electrons ahead of xd + 8 are not
    pushed (they are at rest, E = 0 there, driver force < 1e-13) and are left out of the deposit
    together with their own ion charge.
  * EV births on a theta-independent schedule: every MB steps, every injector ion with
    |x - xd| < 1.5 emits weight rem (1 - exp(-Gamma MB dt)). Particle slots are therefore
    fixed in advance, so noise can be indexed by a stable particle slot (common random numbers).
  * Momentum noise: u_{n+1} = u + dt f1 + sigma sqrt(dt) z, z ~ N(0,1) from
    default_rng([seed, n]) over the active slot range; noise acts only for xi = x - xd < XI_ON
    (behind the driver front) and only on the species in NOISE ("all" or "beam").

Tangent modes (theta = a0):
  primal : no tangent.
  path   : exact pathwise tangent (dx, du, dW) of the noisy model at fixed noise (sigma = 0: the
           exact Lagrangian tangent of the deterministic model).
  lr     : mean-field likelihood-ratio (Girsanov) tangent. Noise species carry du = 0, a weight
           tangent dW with dW += W * df1 * z sqrt(dt) / sigma (df1 = change of the mean force at
           fixed (x, u)), and a non-growing position tangent dx from the explicit dependence of
           the position update on A(x; a0). Non-noise species stay pathwise. Field tangent from
           dion - deposit(dW, -W dx S'). Births enter through dW at birth.
           With NOISE=all also accumulates the joint-measure score S (explicit theta dependence
           only: driver terms and the field of the birth-weight change at fixed positions).

CLI: python -I wake_lr.py <a0> <seed> <sigma> <mode> <noise> <nmult> <out.json> [tf]
     nmult: particle multiplier; plasma ppc = 5 nmult, injector ppc = nmult.
"""
import sys, json, time
import numpy as np
import os
NOFB = os.environ.get("NOFB", "0") == "1"
CENTER = int(os.environ.get("CENTER", "0"))
T_ON = float(os.environ.get("T_ON", "0"))   # synthesis addition: noise only for t >= T_ON
MIXK = os.environ.get("MIXK")              # synthesis addition: mixed estimator, xi_u damped at rate MIXK inside the window
MIXK = None if MIXK is None else float(MIXK)   # 0 plain, 1 local mean, 2 leave-one-out local mean (critic addition)   # diagnostic: weight tangents do not feed the field tangent

DX = 0.025; DT = 0.025
X0, X1 = -6.0, 80.0
NX = int(round((X1 - X0) / DX))
GAM_G = 10.0; BETA_G = np.sqrt(1 - 1 / GAM_G ** 2)
S_L = 1.0
XD0 = -4.0
RAMP = (0.0, 3.0)
DOPE = (8.0, 14.0); NDOPE = 0.03
G0, BION = 10.0, 8.0
MB = 4          # birth every MB steps
XI_B = 1.5      # birth window |x - xd| < XI_B
XI_PUSH = 8.0   # plasma electrons ahead of xd + XI_PUSH are frozen
XI_ON = 6.0     # noise on for xi < XI_ON
REC_T = (15.0, 20.0, 30.0, 45.0, 60.0, 75.0)


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


def dep(i, d, w):
    return np.bincount(i, w * (1 - d), NX + 1) + np.bincount(i + 1, w * d, NX + 1)


def efield(rho):
    seg = 0.5 * (rho[1:] + rho[:-1]) * DX
    return -np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])


def S5(t):
    tr = np.real(t)
    poly = t ** 3 * (10 - 15 * t + 6 * t * t)
    return np.where(tr <= 0, 0.0, np.where(tr >= 1, 1.0, poly))


NAMES = ["logQ", "<u>", "<xi>", "log su", "log sxi", "log su/<u>"]


def observables(ub, Wb, xb, xd, uc=3.0, du=0.3):
    """r2_pilot observables (uc=3, du=0.3); complex-step safe."""
    sig = S5((ub - uc + 2 * du) / (4 * du))
    w = Wb * sig; S0 = w.sum(); D = S0 + 1e-6
    xi = xb - xd
    mu = (w * ub).sum() / D; mx = (w * xi).sum() / D
    su = np.sqrt((w * ub ** 2).sum() / D - mu ** 2); sx = np.sqrt((w * xi ** 2).sum() / D - mx ** 2)
    return np.array([np.log(S0 + 1e-6), mu, mx, np.log(su + 1e-3), np.log(sx + 1e-3), np.log((su + 1e-3) / (mu + 1e-2))])


def obs_tan(ub, Wb, xb, xd, dub, dWb, dxb):
    h = 1e-30
    v = observables(ub + 1j * h * dub, Wb + 1j * h * dWb, xb + 1j * h * dxb, xd)
    return np.real(v), np.imag(v) / h


def run(a0, seed, sigma, mode="primal", noise="all", nmult=1, tf=75.0):
    rng0 = np.random.default_rng([seed, 999999])
    ppc = 5 * nmult; ion_ppc = nmult
    # plasma electrons, stored in descending x so the frozen ones are a prefix
    xs = np.arange(RAMP[0], X1 - 1.0, DX / ppc) + 0.5 * DX / ppc
    xs = xs + (rng0.random(xs.size) - 0.5) * 0.5 * DX / ppc
    W = dens(xs) * DX / ppc
    keep = W > 0; xp0 = xs[keep][::-1].copy(); Wp = W[keep][::-1].copy(); NP = xp0.size
    xp_asc = xp0[::-1].copy()
    # injector ions
    xion = np.arange(DOPE[0], DOPE[1], DX / ion_ppc) + 0.5 * DX / ion_ppc
    xion = xion + (rng0.random(xion.size) - 0.5) * 0.5 * DX / ion_ppc
    Wion = NDOPE * DX / ion_ppc * np.ones_like(xion); rem = np.ones_like(xion); drem = np.zeros_like(xion)
    nt = int(round(tf / DT))
    sched = {}
    for n in range(0, nt, MB):
        xd = XD0 + BETA_G * n * DT
        m = np.nonzero(np.abs(xion - xd) < XI_B)[0]
        if m.size: sched[n] = m
    NB = sum(v.size for v in sched.values()); NT = NP + NB
    x = np.zeros(NT); u = np.zeros(NT); Wt = np.zeros(NT); x[:NP] = xp0; Wt[:NP] = Wp
    isb = np.zeros(NT, bool); isb[NP:] = True
    alive = np.ones(NT, bool)
    tan = mode != "primal"; lr = mode == "lr"
    dx = np.zeros(NT); du = np.zeros(NT); dW = np.zeros(NT); dW0 = np.zeros(NT)
    noise_sp = np.ones(NT, bool) if noise == "all" else isb.copy()
    lr_sp = noise_sp & lr if sigma > 0 else np.zeros(NT, bool)
    joint = lr and noise == "all" and sigma > 0 and os.environ.get("NOJOINT", "0") != "1"
    Sj = 0.0
    ion = np.zeros(NX + 1); dion = np.zeros(NX + 1)
    n_ahead = NP; nb = 0
    tlog = []
    recs = []; t_start = time.time(); drop = 0.0
    for n in range(nt):
        t = n * DT; xd = XD0 + BETA_G * t
        # ---- activate plasma electrons (add their ion charge)
        k = NP - np.searchsorted(xp_asc, xd + XI_PUSH)
        if k < n_ahead:
            ii, dd = wts(x[k:n_ahead]); ion += dep(ii, dd, Wt[k:n_ahead]); n_ahead = k
        # ---- EV births
        if n in sched:
            m = sched[n]
            Ai, _, _ = A_of(xion[m], t, a0)
            a_loc = np.sqrt(2 * Ai)
            Gam = G0 * np.exp(-BION / np.maximum(a_loc, 1e-12))
            Ex = np.exp(-Gam * MB * DT)
            remb = rem[m].copy()
            frac = remb * (1 - Ex)
            rem[m] = remb - frac
            sl = slice(NP + nb, NP + nb + m.size)
            x[sl] = xion[m]; u[sl] = 0.0; Wt[sl] = Wion[m] * frac
            ii, dd = wts(xion[m]); ion += dep(ii, dd, Wion[m] * frac)
            if tan:
                dGam = Gam * BION / a_loc ** 2 * (a_loc / a0)
                dfrac = drem[m] * (1 - Ex) + remb * Ex * MB * DT * dGam
                drem[m] -= dfrac
                dW[sl] = Wion[m] * dfrac; dW0[sl] = dW[sl]
                dion += dep(ii, dd, Wion[m] * dfrac)
            nb += m.size
        a = slice(n_ahead, NP + nb)
        xa = x[a]; ua = u[a]; Wa = Wt[a]
        i, d = wts(xa)
        E = efield((ion - dep(i, d, Wa)) / DX)
        Ep = E[i] * (1 - d) + E[i + 1] * d
        A, F, zz = A_of(xa, t, a0)
        g0 = np.sqrt(1 + ua ** 2 + A)
        f0 = -Ep - F / (2 * g0)
        uh = ua + 0.5 * DT * f0
        gh = np.sqrt(1 + uh ** 2 + A)
        f1 = -Ep - F / (2 * gh)
        sig_i = np.where((zz < XI_ON) & noise_sp[a] & (t >= T_ON - 1e-12), sigma, 0.0)
        if sigma > 0:
            z = np.random.default_rng([seed, n]).standard_normal(xa.size)
            eta = sig_i * np.sqrt(DT) * z
        else:
            z = np.zeros(xa.size); eta = 0.0
        u1 = ua + DT * f1 + eta
        g1 = np.sqrt(1 + u1 ** 2 + A)
        x1 = xa + DT * u1 / g1
        if tan:
            ddx = dx[a]; ddu = du[a]; dWa = dW[a]; lra = lr_sp[a]
            dWf = np.where(lra, 0.0, dWa) if NOFB else dWa
            dE = efield((dion - (np.bincount(i, dWf * (1 - d) - Wa * ddx / DX, NX + 1)
                                 + np.bincount(i + 1, dWf * d + Wa * ddx / DX, NX + 1))) / DX)
            dEp = dE[i] * (1 - d) + dE[i + 1] * d + (E[i + 1] - E[i]) / DX * ddx
            Aa = 2 * A / a0; Fa = 2 * F / a0
            Fx = -A / S_L ** 2 - zz / S_L ** 2 * F
            dA = F * ddx + Aa; dF = Fx * ddx + Fa
            dg0 = (2 * ua * ddu + dA) / (2 * g0)
            df0 = -dEp - dF / (2 * g0) + F * dg0 / (2 * g0 ** 2)
            duh = ddu + 0.5 * DT * df0
            dgh = (2 * uh * duh + dA) / (2 * gh)
            df1 = -dEp - dF / (2 * gh) + F * dgh / (2 * gh ** 2)
            du1 = ddu + DT * df1
            if lr:
                on = lra & (sig_i > 0)
                if MIXK is None:
                    inc = np.where(on, Wa * df1 * z * np.sqrt(DT) / np.where(sig_i > 0, sig_i, 1.0), 0.0)
                else:
                    # mixed identity: xi_u -> (1 - k dt) xi_u, the rest H of the one-step tangent goes into the weight
                    tgt = (1.0 - MIXK * DT) * ddu
                    Hm = du1 - tgt
                    inc = np.where(on, Wa * Hm * z / (np.where(sig_i > 0, sig_i, 1.0) * np.sqrt(DT)), 0.0)
                if CENTER:
                    Won = np.where(on, Wa, 0.0)
                    Cn = dep(i, d, inc); Dn = dep(i, d, Won)
                    w0 = 1 - d; w1 = d
                    if CENTER == 2:
                        C0 = Cn[i] - w0 * inc; D0 = Dn[i] - w0 * Won; C1 = Cn[i + 1] - w1 * inc; D1 = Dn[i + 1] - w1 * Won
                    else:
                        C0 = Cn[i]; D0 = Dn[i]; C1 = Cn[i + 1]; D1 = Dn[i + 1]
                    tiny = 1e-300
                    M0 = np.where(D0 > 1e-14, C0 / np.maximum(D0, tiny), 0.0); M1 = np.where(D1 > 1e-14, C1 / np.maximum(D1, tiny), 0.0)
                    inc = np.where(on, inc - Wa * (w0 * M0 + w1 * M1), 0.0)
                dWa = dWa + inc
                if MIXK is None:
                    off = lra & (sig_i == 0)
                    if off.any(): drop = max(drop, float(np.abs(df1[off]).max()))
                    du1 = np.where(lra, 0.0, du1)
                else:
                    du1 = np.where(on, tgt, du1)
                dW[a] = dWa
                if joint:
                    dEW = efield((dion - dep(i, d, dW0[a])) / DX)
                    dEWp = dEW[i] * (1 - d) + dEW[i + 1] * d
                    dg0e = Aa / (2 * g0)
                    df0e = -dEWp - Fa / (2 * g0) + F * dg0e / (2 * g0 ** 2)
                    dghe = (uh * DT * df0e + Aa) / (2 * gh)
                    df1e = -dEWp - Fa / (2 * gh) + F * dghe / (2 * gh ** 2)
                    Sj += float(np.sum(np.where(sig_i > 0, df1e * z * np.sqrt(DT) / np.where(sig_i > 0, sig_i, 1.0), 0.0)))
            dg1 = (2 * u1 * du1 + dA) / (2 * g1)
            dx[a] = ddx + DT * (du1 / g1 - u1 * dg1 / g1 ** 2)
            du[a] = du1
        x[a] = x1; u[a] = u1
        dead = alive[a] & (x1 < X0 + 2 * DX)
        if dead.any():
            idx = np.nonzero(dead)[0] + n_ahead
            alive[idx] = False
        dm = ~alive[a]
        if dm.any():
            idx = np.nonzero(dm)[0] + n_ahead
            x[idx] = X0 + 2.5 * DX; u[idx] = 0.0; Wt[idx] = 0.0; dx[idx] = 0.0; du[idx] = 0.0; dW[idx] = 0.0; dW0[idx] = 0.0
        tn = (n + 1) * DT
        if tan and (n + 1) % 40 == 0:
            wa_ = Wt[a]; sa = (sig_i > 0) & (wa_ > 0) & lr_sp[a]
            pl = sa & ~isb[a]; bm = sa & isb[a]
            omr = lambda s_: float(np.sqrt(np.sum(wa_[s_] * (dW[a][s_] / wa_[s_]) ** 2) / wa_[s_].sum())) if s_.any() else 0.0
            tlog.append([tn, omr(pl), omr(bm), float(np.abs(dE[i]).max()),
                         float(np.median(np.hypot(dx[a], du[a])[~isb[a]])) if (~isb[a]).any() else 0.0])
        if any(abs(tn - r) < 1e-9 for r in REC_T):
            xdn = XD0 + BETA_G * tn
            b = slice(NP, NP + nb)
            rec = dict(t=tn)
            if tan:
                v, dv = obs_tan(u[b], Wt[b], x[b], xdn, du[b], dW[b], dx[b])
                rec["dobs"] = dv.tolist()
                if joint:
                    _, dvW = obs_tan(u[b], Wt[b], x[b], xdn, 0 * u[b], dW0[b], 0 * u[b])
                    rec["dobs_W"] = dvW.tolist(); rec["Sj"] = Sj
            else:
                v = observables(u[b], Wt[b], x[b], xdn)
            rec["obs"] = v.tolist()
            # plasma local velocity spread in xi bins, xi in [-40, 0]
            xp = x[n_ahead:NP]; up = u[n_ahead:NP]; wp = Wt[n_ahead:NP]
            zp = xp - xdn; sel = (zp > -40) & (zp < 0) & (wp > 0)
            kb = ((zp[sel] + 40) / 0.25).astype(int)
            c0 = np.bincount(kb, wp[sel], 160); c1 = np.bincount(kb, wp[sel] * up[sel], 160)
            c2 = np.bincount(kb, wp[sel] * up[sel] ** 2, 160)
            ok = c0 > 0
            var = np.zeros(160); var[ok] = c2[ok] / c0[ok] - (c1[ok] / c0[ok]) ** 2
            rec["plasma_du_rms"] = float(np.sqrt(np.sum(c0 * var) / np.sum(c0)))
            ub_ = u[b]; trp = ub_ > 3.0
            rec["Q_trap"] = float(Wt[b][trp].sum()); rec["nb_trap"] = int(trp.sum())
            # field and field-tangent samples on xi in [-40, 6], step 0.5
            xs_ = xdn + np.arange(-40.0, 6.01, 0.5)
            ii, dd = wts(xs_)
            # recompute E at end of step positions
            aa = slice(n_ahead, NP + nb)
            i2, d2 = wts(x[aa])
            E2 = efield((ion - dep(i2, d2, Wt[aa])) / DX)
            rec["E_s"] = (E2[ii] * (1 - dd) + E2[ii + 1] * dd).tolist()
            if tan:
                dE2 = efield((dion - (np.bincount(i2, dW[aa] * (1 - d2) - Wt[aa] * dx[aa] / DX, NX + 1)
                                      + np.bincount(i2 + 1, dW[aa] * d2 + Wt[aa] * dx[aa] / DX, NX + 1))) / DX)
                rec["dE_s"] = (dE2[ii] * (1 - dd) + dE2[ii + 1] * dd).tolist()
                wb = Wt[b]; tb = trp & (wb > 0)
                if lr:
                    om = np.where(wb > 0, dW[b] / np.where(wb > 0, wb, 1), 0)
                    rec["om_rms_trap"] = float(np.sqrt(np.sum(wb[tb] * om[tb] ** 2) / max(wb[tb].sum(), 1e-300)))
                    pa = slice(n_ahead, NP); wpp = Wt[pa]; zpp = x[pa] - xdn; s2 = (zpp < XI_ON) & (wpp > 0)
                    omp = dW[pa][s2] / wpp[s2]
                    rec["om_rms_plasma"] = float(np.sqrt(np.mean(omp ** 2))) if omp.size else 0.0
                nrm = np.hypot(dx[b][tb], du[b][tb])
                rec["tan_med_trap"] = float(np.median(nrm)) if nrm.size else 0.0
                rec["tan_p90_trap"] = float(np.percentile(nrm, 90)) if nrm.size else 0.0
                pa = slice(n_ahead, NP); s2 = (x[pa] - xdn) < XI_ON
                nrmp = np.hypot(dx[pa][s2], du[pa][s2])
                rec["tan_med_pl"] = float(np.median(nrmp)) if nrmp.size else 0.0
                rec["tan_p90_pl"] = float(np.percentile(nrmp, 90)) if nrmp.size else 0.0
            recs.append(rec)
    return dict(recs=recs, tlog=tlog, NP=int(NP), NB=int(NB), time=time.time() - t_start, drop=drop)


if __name__ == "__main__":
    a0 = float(sys.argv[1]); seed = int(sys.argv[2]); sigma = float(sys.argv[3]); mode = sys.argv[4]
    noise = sys.argv[5]; nmult = int(sys.argv[6]); outp = sys.argv[7]
    tf = float(sys.argv[8]) if len(sys.argv) > 8 else 75.0
    if os.path.exists(outp):
        sys.exit(0)
    r = run(a0, seed, sigma, mode, noise, nmult, tf)
    r.update(a0=a0, seed=seed, sigma=sigma, mode=mode, noise=noise, nmult=nmult, tf=tf)
    json.dump(r, open(outp, "w"))
    last = r["recs"][-1] if r["recs"] else {}
    print(json.dumps(dict(a0=a0, seed=seed, sigma=sigma, mode=mode, time=round(r["time"], 1), NP=r["NP"], NB=r["NB"],
                          obs=last.get("obs"), dobs=last.get("dobs"))))
