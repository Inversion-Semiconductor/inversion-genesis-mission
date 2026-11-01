"""T1c: lattice secants at T = 15 (N = 4e6 labels) to check the scale-free derivative."""
import json, sys
sys.path.insert(0, "inversion/notes/gpt-evidence-2026-11-01/work_alt")
sys.dont_write_bytecode = True
from multiprocessing import Pool
import numpy as np
import t1_sheet as M
if __name__ == "__main__":
    out = {}
    with Pool(16) as pool:
        for T in (15.0,):
            nstep = int(round(T / M.DT))
            for h in (0.1, 0.03, 0.01, 0.003, 0.001):
                Jp = M.J_lattice(4 * 10**6, nstep, M.A0 + h, pool, 32)
                Jm = M.J_lattice(4 * 10**6, nstep, M.A0 - h, pool, 32)
                out[f"{T}_{h}"] = ((Jp - Jm) / (2 * h)).tolist()
                print(T, h, out[f"{T}_{h}"], flush=True)
    json.dump(out, open("t1_sec15.json", "w"), indent=1)
