"""Diagnose the t=45 deviation of the late-window mixed estimator at sigma=0.3: pathwise, plain, centered N=4 and N=16."""
import glob, json
import numpy as np

OFFS = [0.0, 0.02, -0.02, 0.04, -0.04]
for T in ["45.0", "75.0"]:
    rs = [json.load(open(f)) for f in glob.glob("out/fd/w30_0.3_4_*.json")]
    O = {o: np.array([r["res"][repr(o)][T]["obs"] for r in rs]) for o in OFFS}
    fit = np.array([[np.polyfit(OFFS, [O[o][i, j] for o in OFFS], 3)[-2] for j in range(6)] for i in range(len(rs))])
    fm, fse = fit.mean(0), fit.std(0, ddof=1) / np.sqrt(len(fit))
    print(f"t={T} FD fit3 N=4:", " ".join("%8.3g±%-6.2g" % (a, b) for a, b in zip(fm, fse)))
    for tag in ["p30_0.3_4", "m30k1c0_0.3_4", "m30k1_0.3_4", "m30k1_0.3_16"]:
        fl = glob.glob(f"out/lr/{tag}_*.json")
        if not fl: continue
        D = np.array([next(c for c in json.load(open(f))["recs"] if repr(c["t"]) == T)["dobs"] for f in fl])
        m, se = D.mean(0), D.std(0, ddof=1) / np.sqrt(len(D))
        z = (m - fm) / np.hypot(se, fse)
        print(f"  {tag:14s} n={len(D)}", " ".join("%8.3g±%-6.2g" % (a, b) for a, b in zip(m, se)), " z:", " ".join("%5.1f" % v for v in z))
