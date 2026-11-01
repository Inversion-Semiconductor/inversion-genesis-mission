"""T1b: hybrid label-IBP. chi(log|t|) switches label IBP on only where the sheet is stretched.

Per particle: (1-chi) tau grad(phi).t - phi d_a(chi tau) + nu d_n phi,
d_a(chi tau) = chi d_a tau + tau chi'(s) d_a s, s = log|t|, d_a s = t.t_a/|t|^2.
Exact in the continuum for any smooth chi (periodic labels).
"""
import json, sys
sys.path.insert(0, "inversion/notes/gpt-evidence-2026-11-01/work_alt")
sys.dont_write_bytecode = True
from multiprocessing import Pool
import numpy as np
import t1_sheet as M


def smooth(s, lo, hi):
    y = np.clip((s - lo) / (hi - lo), 0, 1)
    return y * y * (3 - 2 * y), np.where((y > 0) & (y < 1), 6 * y * (1 - y) / (hi - lo), 0.0)


def est(a, nstep, cuts):
    x, v, t, ta, q, qa = M.run_tangents(a, M.A0, nstep)
    tx, tv = t; tax, tav = ta; qx, qv = q; qax, qav = qa
    phi = M.phi_all(v); dphi = M.dphi_all(v)
    t2 = tx * tx + tv * tv; tn = np.sqrt(t2)
    nvv = tx / tn  # v-component of the unit normal
    nu = (-tv * qx + tx * qv) / tn
    qt = qx * tx + qv * tv
    tau = qt / t2
    dtau = ((qax * tx + qav * tv + qx * tax + qv * tav) * t2 - 2 * qt * (tx * tax + tv * tav)) / (t2 * t2)
    s = np.log(tn); ds = (tx * tax + tv * tav) / t2
    out = {"lag": (dphi * qv).mean(axis=1)}
    for lo, hi in cuts:
        chi, dchi = smooth(s, np.log(lo), np.log(hi))
        tang = tau * dphi * tv  # grad(phi).t = phi'(v) t_v
        c = (1 - chi) * tang - phi * (chi * dtau + tau * dchi * ds) + nu * nvv * dphi
        out[f"hyb_{lo}_{hi}"] = c.mean(axis=1)
        out[f"hyb_{lo}_{hi}_psd"] = c.std(axis=1)
    out["lag_psd"] = (dphi * qv).std(axis=1)
    return {k: v.tolist() for k, v in out.items()}


CUTS = [(3, 10), (10, 30), (30, 100)]


def job(args):
    n, seed, nstep = args
    rng = np.random.default_rng(seed)
    return est(2 * np.pi * rng.random(n), nstep, CUTS)


def lat(args):
    n, j, nch, nstep = args
    return est(M.lattice(n, j, nch), nstep, CUTS)


if __name__ == "__main__":
    res = {}
    with Pool(16) as pool:
        for T in (10.0, 15.0, 20.0):
            nstep = int(round(T / M.DT))
            r = {}
            for n in (10**6, 4 * 10**6):
                outs = pool.map(lat, [(n, j, 32, nstep) for j in range(32)])
                r[f"lattice_{n}"] = {k: np.mean([o[k] for o in outs], axis=0).tolist() for k in outs[0] if not k.endswith("psd")}
                print(T, n, "lattice", json.dumps(r[f"lattice_{n}"]), flush=True)
            outs = pool.map(job, [(10**5, 500 + s, nstep) for s in range(16)])
            d = {}
            for k in outs[0]:
                arr = np.array([o[k] for o in outs])
                d[k] = dict(mean=arr.mean(0).tolist(), se=(arr.std(0, ddof=1) / 4).tolist())
            r["random_1e5"] = d
            print(T, "random", json.dumps(d), flush=True)
            res[str(T)] = r
            json.dump(res, open("t1_hybrid.json", "w"), indent=1)
