"""Growth rates of the plasma weight rms, plain (c0) vs leave-one-out charge-neutral (c2), seed 1."""
import json, os
import numpy as np
print("sigma N | rate ln om_pl t[10,30]: c0 / c2 | rate t[30,75]: c0 / c2 | om_pl(30) c0/c2 | om_pl(75) c0/c2 | max|dE|(75) c0/c2")
for s in ('0.01', '0.03', '0.1'):
    for n in (1, 4, 16, 64):
        row = []
        for c in (0, 2):
            f = f'out/grow/c{c}_{s}_{n}.json'
            if not os.path.exists(f): row.append(None); continue
            L = np.array(json.load(open(f))['tlog'])
            r = []
            for a, b in ((10, 30), (30, 75)):
                sel = (L[:, 0] >= a) & (L[:, 0] <= b)
                r.append(np.polyfit(L[sel, 0], np.log(L[sel, 1]), 1)[0] if sel.sum() > 3 and L[sel, 0].max() >= b - 1 else np.nan)
            at = lambda t, k: L[np.argmin(np.abs(L[:, 0] - t)), k]
            row.append((r[0], r[1], at(30, 1), at(75, 1) if L[-1, 0] >= 74 else np.nan, at(75, 3) if L[-1, 0] >= 74 else np.nan))
        if all(x is None for x in row): continue
        g = lambda k: " / ".join("%8.3g" % (x[k] if x else np.nan) for x in row)
        print(f"{s:5s} {n:2d} | {g(0)} | {g(1)} | {g(2)} | {g(3)} | {g(4)}")
