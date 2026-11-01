"""Cold beam, T = 10: does the pair estimate converge as sigma -> 0 (more particles)?"""
import json, sys
from multiprocessing import Pool
sys.path.insert(0, sys.argv[1])
import numpy as np
import cold_beam_bias as m

def run(task):
    sigma, seed = task
    m.T = 10.0; m.NSTEP = 200
    c = m.pairs(10000, seed, 0.0, [sigma])
    return task, c[0]

if __name__ == "__main__":
    tasks = [(s, seed) for s in (0.001, 0.003, 0.01) for seed in range(100, 108)]
    with Pool(24) as p:
        out = p.map(run, tasks)
    res = {}
    for s in (0.001, 0.003, 0.01):
        c = np.concatenate([o for (t, o) in out if t[0] == s], axis=1)
        mu, se = m.summarize(c)
        res[str(s)] = {"mean": mu.tolist(), "se": se.tolist(), "n_base": c.shape[1]}
        print(s, c.shape[1], np.round(mu, 3), np.round(se, 3), flush=True)
    json.dump(res, open(sys.argv[2], "w"), indent=1)
