"""Paired test of the centering control variate: centered minus plain LR on the same seed and noise."""
import json, os, sys
import numpy as np
s, n = sys.argv[1], sys.argv[2]
for T in (30.0, 75.0):
    D = []; P = []; C = []
    for sd in range(1001, 1040):
        fp = f"../work_wake/out/lr/lr_{s}_{n}_{sd}.json"; fc = f"out/ens/c2_{s}_{n}_{sd}.json"
        fc30 = f"out/ens30/c2_{s}_{n}_{sd}.json"
        if not os.path.exists(fc): fc = fc30
        if not (os.path.exists(fp) and os.path.exists(fc)): continue
        g = lambda f: [r for r in json.load(open(f))["recs"] if abs(r["t"] - T) < 1e-9]
        a, b = g(fp), g(fc)
        if not a or not b: continue
        P.append(a[0]["dobs"]); C.append(b[0]["dobs"])
    if not P: continue
    P, C = np.array(P), np.array(C); D = C - P
    f = lambda X: " ".join("%9.3g+-%-8.2g" % (u, v) for u, v in zip(X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))))
    print(f"sigma={s} N={n} t={T} pairs={len(D)}\n  plain    {f(P)}\n  centered {f(C)}\n  diff     {f(D)}\n  z(diff)  " + " ".join("%9.2f          " % v for v in D.mean(0) / (D.std(0, ddof=1) / np.sqrt(len(D)))))
