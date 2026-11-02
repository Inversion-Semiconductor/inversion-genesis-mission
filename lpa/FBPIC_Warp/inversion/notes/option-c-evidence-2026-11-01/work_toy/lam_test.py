"""Measure the weight-variance growth rate of the mean-field LR in a STABLE cold plasma (V0 = 0).

Prediction (derived): d Var(omega)/dt = <|dF_phys|^2>/sigma^2 + lam Var(omega),
lam = <dE_noise^2>/(sigma^2 Var(omega)) ~ L^2/(12 N sigma^2) for a periodic 1D box (random weights).
Usage: ES_V0=0 python -I lam_test.py <out.json>
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import es1d_lr as M
from multiprocessing import Pool

def job(a):
    sigma, N = a
    r = M.run(M.TH0, sigma, N, 11, do_mf=True, do_tan=True, series=True, tmax=60)
    ser = r["series"]
    t = np.array([o["t"] for o in ser]); om = np.array([o["omrms"] for o in ser]); lam = np.array([o["lam"] for o in ser])
    tr = np.array([o["tanrms"] for o in ser])
    m = t >= 20
    slope = np.polyfit(t[m], np.log(om[m] ** 2), 1)[0]
    pred = M.L ** 2 / (12 * N * sigma ** 2)
    return dict(sigma=sigma, N=N, slope_var=float(slope), lam_meas=float(np.median(lam[m])),
                lam_disc=float(np.log1p(np.median(lam[m]) * M.DT) / M.DT), lam_pred=float(pred),
                omrms_t=dict(zip([f"{x:g}" for x in t[::50]], om[::50].tolist())),
                tanrms_t=dict(zip([f"{x:g}" for x in t[::50]], tr[::50].tolist())))

if __name__ == "__main__":
    cases = [(s, N) for s in (0.03, 0.1) for N in (32768, 131072, 524288)]
    with Pool(6) as p:
        res = p.map(job, cases)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    for r in res:
        print(f"sigma {r['sigma']} N {r['N']}: Var(omega) growth {r['slope_var']:.4f}  lam_meas {r['lam_meas']:.4f} "
              f"(discrete {r['lam_disc']:.4f})  L^2/(12 N sigma^2) {r['lam_pred']:.4f}  omrms {r['omrms_t']}  tanrms {r['tanrms_t']}")
