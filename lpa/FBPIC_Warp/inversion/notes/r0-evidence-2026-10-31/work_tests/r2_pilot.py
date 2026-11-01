"""R2 pilot: 1D relativistic electrostatic wake driven by a prescribed ponderomotive
driver, with expected-value (EV) ionization-injection analog. Lab frame, c = omega_p = k_p = 1.

Plasma electrons: cold, regular loading with optional seeded jitter.
Injector ions: emit EV electrons at rest with weight dW = W r (1 - exp(-Gamma(a) dt)).
Parameter theta = a0 (driver amplitude, the E_L analog: E_L ~ a0^2).
"""
import sys, os, json, time
import numpy as np

DX = 0.025; DT = 0.025
X0, X1 = -6.0, 95.0
NX = int(round((X1 - X0) / DX)); xf = X0 + np.arange(NX + 1) * DX  # nodes
GAM_G = 10.0; BETA_G = np.sqrt(1 - 1 / GAM_G ** 2)
S_L = 1.0       # rms length of <a^2>
XD0 = -4.0      # driver center at t=0
RAMP = (0.0, 3.0)
DOPE = (8.0, 14.0); NDOPE = 0.03
G0, BION = 10.0, 8.0
EPS = 1e-4


def dens(x):
    return np.clip((x - RAMP[0]) / (RAMP[1] - RAMP[0]), 0, 1)


def A_of(x, t, a0):
    xi = x - (XD0 + BETA_G * t)
    A = 0.5 * a0 ** 2 * np.exp(-xi ** 2 / (2 * S_L ** 2))
    dA = -xi / S_L ** 2 * A
    return A, dA


def wts(x):
    s = (x - X0) / DX
    i = np.floor(s).astype(np.int64); d = s - i
    return i, d  # linear (CIC) shape onto nodes


def run(a0, ppc=20, seed=None, tf=80.0, rec_every=40, xpos=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    # plasma electrons: regular loading from 0 to X1-1, weight = local density
    xs = np.arange(RAMP[0], X1 - 1.0, DX / ppc) + 0.5 * DX / ppc
    if rng is not None:
        xs = xs + (rng.random(xs.size) - 0.5) * DX / ppc
    W = dens(xs) * DX / ppc
    keep = W > 0; xe = xs[keep]; We = W[keep]; ue = np.zeros_like(xe); ide = np.arange(xe.size)
    # fixed ion background on nodes (plasma)
    ion = np.zeros(NX + 1)
    i, d = wts(xe); np.add.at(ion, i, We * (1 - d)); np.add.at(ion, i + 1, We * d)
    # injector ions
    xi_ = np.arange(DOPE[0], DOPE[1], DX / 4) + DX / 8
    if rng is not None:
        xi_ = xi_ + (rng.random(xi_.size) - 0.5) * DX / 4
    Wi = NDOPE * DX / 4 * np.ones_like(xi_); rem = np.ones_like(xi_)
    xb = np.zeros(0); ub = np.zeros(0); Wb = np.zeros(0); tb = np.zeros(0); idb = np.zeros(0, np.int64); nborn = 0
    nt = int(round(tf / DT)); hist = []
    for n in range(nt):
        t = n * DT
        # EV births at rest (time t)
        Ai, _ = A_of(xi_, t, a0)
        a_loc = np.sqrt(2 * Ai)
        Gam = G0 * np.exp(-BION / np.maximum(a_loc, 1e-12))
        frac = rem * (1 - np.exp(-Gam * DT))
        m = frac > EPS
        if m.any():
            dW = Wi[m] * frac[m]
            rem[m] -= frac[m]
            xb = np.concatenate([xb, xi_[m]]); ub = np.concatenate([ub, np.zeros(m.sum())])
            Wb = np.concatenate([Wb, dW]); tb = np.concatenate([tb, np.full(m.sum(), t)])
            idb = np.concatenate([idb, nborn + np.arange(m.sum())]); nborn += int(m.sum())
            ii, dd = wts(xi_[m]); np.add.at(ion, ii, dW * (1 - dd)); np.add.at(ion, ii + 1, dW * dd)
        # field: dE/dx = rho, E(X0)=0
        x_all = np.concatenate([xe, xb]); W_all = np.concatenate([We, Wb])
        i, d = wts(x_all)
        ne = np.bincount(i, W_all * (1 - d), NX + 1) + np.bincount(i + 1, W_all * d, NX + 1)
        rho = (ion - ne) / DX
        seg = 0.5 * (rho[1:] + rho[:-1]) * DX  # integrate from the unperturbed front, E(X1) = 0
        E = -np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        Ep = E[i] * (1 - d) + E[i + 1] * d
        A, dA = A_of(x_all, t, a0)
        u = np.concatenate([ue, ub])
        # leapfrog-like push with predictor gamma
        g0 = np.sqrt(1 + u ** 2 + A)
        uh = u + 0.5 * DT * (-Ep - dA / (2 * g0))
        gh = np.sqrt(1 + uh ** 2 + A)
        u = u + DT * (-Ep - dA / (2 * gh))
        g1 = np.sqrt(1 + u ** 2 + A)
        x_all = x_all + DT * u / g1
        ne_ = xe.size
        xe, xb = x_all[:ne_], x_all[ne_:]; ue, ub = u[:ne_], u[ne_:]
        ke = xe > X0 + 2 * DX; kb = xb > X0 + 2 * DX  # absorb at the back boundary
        if not ke.all(): xe, ue, We, ide = xe[ke], ue[ke], We[ke], ide[ke]
        if not kb.all(): xb, ub, Wb, tb, idb = xb[kb], ub[kb], Wb[kb], tb[kb], idb[kb]
        if (n + 1) % rec_every == 0:
            hist.append(dict(t=(n + 1) * DT, xe=xe.copy(), ue=ue.copy(), ide=ide.copy(), xb=xb.copy(), ub=ub.copy(),
                             Wb=Wb.copy(), idb=idb.copy(), xd=XD0 + BETA_G * (n + 1) * DT))
    return dict(xe=xe, ue=ue, xb=xb, ub=ub, Wb=Wb, tb=tb, E=E, hist=hist, xd=XD0 + BETA_G * tf, ide=ide, idb=idb)


def S5(t):
    t = np.clip(t, 0, 1); return t ** 3 * (10 - 15 * t + 6 * t * t)


def observables(res, uc=8.0, du=0.5):
    sig = S5((res['ub'] - uc + 2 * du) / (4 * du))
    w = res['Wb'] * sig; S0 = w.sum(); D = S0 + 1e-6
    xi = res['xb'] - res['xd']
    mu = (w * res['ub']).sum() / D; mx = (w * xi).sum() / D
    su = np.sqrt(max((w * res['ub'] ** 2).sum() / D - mu ** 2, 0)); sx = np.sqrt(max((w * xi ** 2).sum() / D - mx ** 2, 0))
    return np.array([np.log(S0 + 1e-6), mu, mx, np.log(su + 1e-3), np.log(sx + 1e-3), np.log((su + 1e-3) / (mu + 1e-2))])


def twin(a0, h, tf, seed=None, rec_every=40):
    r0 = run(a0, seed=seed, tf=tf, rec_every=rec_every); r1 = run(a0 + h, seed=seed, tf=tf, rec_every=rec_every)
    out = []
    for H0, H1 in zip(r0['hist'], r1['hist']):
        row = dict(t=H0['t'])
        for sp in ('e', 'b'):
            i0, i1 = H0['id' + sp], H1['id' + sp]
            com, a, b = np.intersect1d(i0, i1, return_indices=True)
            dx = np.abs(H1['x' + sp][b] - H0['x' + sp][a]) / h; du = np.abs(H1['u' + sp][b] - H0['u' + sp][a]) / h
            row['n' + sp] = int(com.size); row['nmis' + sp] = int(i0.size + i1.size - 2 * com.size)
            if com.size:
                row['dx_mean' + sp] = float(dx.mean()); row['du_mean' + sp] = float(du.mean())
                row['du_max' + sp] = float(du.max())
                if sp == 'b':
                    tr = H0['ub'][a] > 3.0
                    row['du_mean_trapped'] = float(du[tr].mean()) if tr.any() else 0.0
                    row['ntrapped'] = int(tr.sum())
        o0 = observables(dict(ub=H0['ub'], Wb=H0['Wb'], xb=H0['xb'], xd=H0['xd']), uc=3.0, du=0.3)
        o1 = observables(dict(ub=H1['ub'], Wb=H1['Wb'], xb=H1['xb'], xd=H1['xd']), uc=3.0, du=0.3)
        row['dobs'] = ((o1 - o0) / h).tolist(); row['obs'] = o0.tolist()
        out.append(row)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "twin":
        out = twin(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
        json.dump(out, open(sys.argv[5], "w"))
    if mode == "obs":
        a0, tf = float(sys.argv[2]), float(sys.argv[3]); seed = None if sys.argv[4] == "none" else int(sys.argv[4])
        r = run(a0, seed=seed, tf=tf, rec_every=10 ** 9)
        print(json.dumps(dict(a0=a0, seed=seed, tf=tf, obs=observables(r, uc=3.0, du=0.3).tolist())))
    if mode == "scan":
        a0 = float(sys.argv[2]); t0 = time.time()
        r = run(a0, tf=float(sys.argv[3]))
        ub = r['ub']; Wb = r['Wb']
        print(json.dumps(dict(a0=a0, time=time.time() - t0, nb=int(ub.size), Wtot=float(Wb.sum()),
                              umax=float(ub.max()) if ub.size else 0, ue_max=float(r['ue'].max()), ue_min=float(r['ue'].min()),
                              Emax=float(np.abs(r['E']).max()),
                              q_u_gt=[float(Wb[ub > c].sum()) for c in (2, 5, 8, 15, 30)],
                              obs=observables(r).tolist())))
