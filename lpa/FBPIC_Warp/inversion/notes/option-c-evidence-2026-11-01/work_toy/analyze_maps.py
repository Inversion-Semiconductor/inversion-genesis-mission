"""Tables for maps_lr.py runs. Doubling: z against the noisy transfer-operator reference.
Standard map: z against CRN FD at h = 0.03 of the same noisy model (h = 0.1 has visible O(h^2) bias). Usage: python -I analyze_maps.py
"""
import json
import os

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
NREC = (5, 10, 20, 40, 80)
pf = json.load(open(f"{D}/dbl_pf.json"))


def load(p):
    return json.load(open(p))["summary"] if os.path.exists(p) else None


def row(S, key, ref, refse=0.0):
    if key not in S:
        return "      -      "
    m, s = S[key]
    z = (m - ref) / np.sqrt(s ** 2 + refse ** 2) if (s > 0 or refse > 0) else 0.0
    return f"{m:+.4f}({s:.4f},z{z:+.1f})"


for mode in ("dbl", "std"):
    for sig in ("0.3", "0.1", "0.1_rep", "0.1_rep2", "0.03", "0.01", "0"):
        S = load(f"{D}/{mode}_s{sig}.json")
        if S is None:
            continue
        print(f"\n== {mode} sigma {sig}")
        print("  n   ref               lr                  lrW5                lrW10               lrW20"
              "               fd0.1               fd0.01              tan(mean)   tanrms   omrms")
        for n in NREC:
            if mode == "dbl":
                ref, rse = pf[sig.split("_")[0]][f"n{n}"][0], 0.0
            else:
                ref, rse = S[f"fd0.03_n{n}"]
            cells = [row(S, f"{k}_n{n}", ref, rse)
                     for k in ("lr", "lrW5", "lrW10", "lrW20", "fd0.1", "fd0.01")]
            print(f"  {n:<3d} {ref:+.4f}({rse:.4f})  " + "  ".join(cells) +
                  f"  {S[f'tan_n{n}'][0]:+.2e}  {S[f'tanrms_n{n}'][0]:.2e}  {S.get(f'omrms_n{n}', (0, 0))[0]:.2f}")
print("\n== doubling map: noisy-model derivative (transfer operator) vs sigma, n = 80")
for s, v in pf.items():
    print(f"  sigma {s:6s} dA/dtheta {v['n80'][0]:+.6f}  bias vs sigma=0 {v['n80'][0] - pf['0']['n80'][0]:+.2e}")
