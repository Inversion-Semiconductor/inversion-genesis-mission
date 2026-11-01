"""Regular regime (short T): Lagrangian is the truth; test pair and mesh bias."""
import json, sys
from multiprocessing import Pool
sys.path.insert(0, sys.argv[1])
import numpy as np
import cold_beam_bias as m

T = float(sys.argv[3])

def run(task):
    m.T = T; m.NSTEP = int(round(T / m.DT))
    kind, vT, arg = task
    if kind == "lag":
        c, g = m.lagrangian(200000, 1, vT)
        mu, se = m.summarize(c)
        return task, {"mean": mu.tolist(), "se": se.tolist(), "growth": g}
    if kind == "pair":
        c = m.pairs(4000, 2, vT, [arg])
        mu, se = m.summarize(c[0])
        return task, {"mean": mu.tolist(), "se": se.tolist()}
    if kind == "coll":
        nx, dv = arg
        c, lm, lx = m.collocated(100000, 3, vT, nx, dv)
        mu, se = m.summarize(c)
        return task, {"mean": mu.tolist(), "se": se.tolist(), "lost_mean": lm, "lost_max": lx}

if __name__ == "__main__":
    tasks = []
    for vT in (0.0, 0.3):
        tasks.append(("lag", vT, None))
        for s in (1e-4, 0.01, 0.03, 0.1, 0.2, 0.4):
            tasks.append(("pair", vT, s))
        for nx, dv in ((128, 0.025), (128, 0.05), (64, 0.1), (64, 0.2)):
            tasks.append(("coll", vT, (nx, dv)))
    with Pool(len(tasks)) as p:
        out = p.map(run, tasks)
    res = {"T": T}
    for task, r in out:
        key = f"{task[0]}_vT{task[1]}_{task[2]}"
        res[key] = r
        print(key, np.round(r["mean"], 4), np.round(r["se"], 4), r.get("lost_mean", ""), flush=True)
    json.dump(res, open(sys.argv[2], "w"), indent=1)
