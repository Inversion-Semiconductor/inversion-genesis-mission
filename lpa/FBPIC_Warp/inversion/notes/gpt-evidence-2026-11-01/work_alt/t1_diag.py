"""T1 diagnostic: where the label-IBP (libp) summand is large. Cold sheet, T = 10 and 15."""
import sys
sys.path.insert(0, "inversion/notes/gpt-evidence-2026-11-01/work_alt")
sys.dont_write_bytecode = True
import numpy as np
import t1_sheet as M

for T in (5.0, 10.0, 15.0):
    nstep = int(round(T / M.DT))
    a = 2 * np.pi * np.random.default_rng(5).random(200000)
    x, v, t, ta, q, qa = M.run_tangents(a, M.A0, nstep)
    tx, tv = t; tax, tav = ta; qx, qv = q
    t2 = tx * tx + tv * tv
    tn = np.sqrt(t2)
    kap = np.abs(tx * tav - tv * tax) / tn ** 3
    e = M.estimators(a, M.A0, nstep)
    L = np.abs(e["libp"][0]); G = np.abs(e["lag"][0])
    nu = np.abs(e["nu"])
    top = L > np.quantile(L, 0.999)
    rk = lambda z: (np.argsort(np.argsort(z)) / z.size)
    print(f"T={T}: |t| median {np.median(tn):.3g}, p99 {np.quantile(tn,0.99):.3g}; kappa median {np.median(kap):.3g}, p99.9 {np.quantile(kap,0.999):.3g}")
    print(f"  per-particle SD: lag {e['lag'][0].std():.3g}, libp {e['libp'][0].std():.3g}; |nu| median {np.median(nu):.3g} p99 {np.quantile(nu,0.99):.3g}")
    print(f"  top 0.1% |libp|: median kappa-rank {np.median(rk(kap)[top]):.4f}, median |t|-rank {np.median(rk(tn)[top]):.3f}, median |xi|-rank {np.median(rk(e['qn'])[top]):.3f}")
    print(f"  share of sum of squares from top 0.1%: libp {np.sum(L[top]**2)/np.sum(L**2):.3f}, lag {np.sum(G[G>np.quantile(G,0.999)]**2)/np.sum(G**2):.3f}")
    print(f"  corr(log|libp|, log kappa*|nu|) = {np.corrcoef(np.log(L+1e-30), np.log(kap*nu+1e-30))[0,1]:.3f}")
