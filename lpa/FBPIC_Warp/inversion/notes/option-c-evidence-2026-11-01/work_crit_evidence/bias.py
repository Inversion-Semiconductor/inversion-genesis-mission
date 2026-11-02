"""Noisy-model gradient minus deterministic gradient, with SE, pooled over critic and original FD seeds (dedup by seed)."""
import glob, json, sys
import numpy as np
T = sys.argv[1] if len(sys.argv) > 1 else "75.0"
NAMES = ["logQ", "<u>", "<xi>", "lsu", "lsxi", "lsu/u"]
def load(sig):
    rs = {}
    for d in ["../work_wake/out/fd", "out/fd"]:
        for f in glob.glob(f"{d}/fd_{sig}_1_*.json"):
            r = json.load(open(f)); rs[r["seed"]] = r
    return rs
def sec(rs, h):
    return np.array([(np.array(r["res"][repr(h)][T]["obs"]) - np.array(r["res"][repr(-h)][T]["obs"])) / (2 * h) for r in rs.values()])
B = load("0.0")
print("t =", T, NAMES)
for h in (0.04, 0.02):
    gb = sec(B, h); mb = gb.mean(0); sb = gb.std(0, ddof=1) / np.sqrt(len(gb))
    print(f"\nh={h}: sigma=0 n={len(gb)} g0 =", " ".join("%8.3g±%-6.2g" % (a, b) for a, b in zip(mb, sb)))
    for sig in ["0.01", "0.03", "0.1"]:
        rs = load(sig); g = sec(rs, h); m = g.mean(0); s = g.std(0, ddof=1) / np.sqrt(len(g))
        rel = np.abs(m - mb) / np.abs(mb); rse = np.hypot(s, sb) / np.abs(mb); z = (m - mb) / np.hypot(s, sb)
        print(f"  sigma={sig:4s} n={len(g):3d} g =", " ".join("%8.3g±%-6.2g" % (a, b) for a, b in zip(m, s)))
        print("      rel bias ", " ".join("%5.2f±%-5.2f(z%4.1f)" % (a, b, c) for a, b, c in zip(rel, rse, z)))
gb4 = sec(B, 0.04); gb2 = sec(B, 0.02); d = gb4 - gb2
print("\nsigma=0 h=0.04 vs h=0.02 (paired):", " ".join("%5.2f±%-5.2f" % (a, b) for a, b in zip(np.abs(d.mean(0)) / np.abs(gb4.mean(0)), d.std(0, ddof=1) / np.sqrt(len(d)) / np.abs(gb4.mean(0)))))
for sig in ["0.0", "0.01", "0.03", "0.1"]:
    rs = load(sig); Q = np.array([r["res"]["0.0"][T]["Q_trap"] for r in rs.values()])
    print(f"Q_trap sigma={sig}: {Q.mean():.5f} ± {Q.std(ddof=1)/np.sqrt(len(Q)):.5f} (n={len(Q)})")
