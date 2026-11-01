"""Fine scan of J(A) for the cold and warm beams, two seeds, N = 4e5 each."""
import json, sys
from multiprocessing import Pool
sys.path.insert(0, sys.argv[1])
import numpy as np
import cold_beam_bias as m

As = np.round(np.arange(0.94, 1.0601, 0.005), 4)

def job(args):
    vT, seed, A = args
    return m.J_of_A(A, 400_000, seed, vT).tolist()

if __name__ == "__main__":
    tasks = [(vT, s, a) for vT in (0.0, 0.3) for s in (11, 12) for a in As]
    with Pool(26) as p:
        out = p.map(job, tasks)
    res = {"A": As.tolist(), "N": 400000}
    i = 0
    for vT in (0.0, 0.3):
        for s in (11, 12):
            res[f"vT{vT}_seed{s}"] = out[i:i + len(As)]
            i += len(As)
    json.dump(res, open(sys.argv[2], "w"), indent=1)
