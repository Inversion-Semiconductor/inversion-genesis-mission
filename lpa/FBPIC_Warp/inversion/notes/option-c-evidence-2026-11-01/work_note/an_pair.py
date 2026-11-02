"""Paired check on the same seeds: late-window mixed LR (LOO centered) minus exact pathwise tangent, sigma=0.3, N=4."""
import glob, json
import numpy as np
def get(tag, T):
    out = {}
    for f in glob.glob(f"out/lr/{tag}_*.json"):
        r = json.load(open(f)); out[r["seed"]] = np.array(next(c for c in r["recs"] if repr(c["t"]) == T)["dobs"])
    return out
for T in ["45.0", "75.0"]:
    a = get("m30k1_0.3_4", T); b = get("p30_0.3_4", T); s = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in s]); m = d.mean(0); se = d.std(0, ddof=1) / np.sqrt(len(d))
    print(f"t={T} n={len(s)} LR-path:", " ".join("%8.3g±%-6.2g(z%4.1f)" % (x, y, x / y) for x, y in zip(m, se)))
