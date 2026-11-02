"""Critic analysis: centered LR vs CRN FD (cubic fit and h=0.04 secant), bias vs sigma=0 with SE."""
import glob, json, sys
import numpy as np
NAMES = ["logQ", "<u>", "<xi>", "lsu", "lsxi", "lsu/u"]
T = sys.argv[1] if len(sys.argv) > 1 else "75.0"

def ms(D):
    D = np.asarray(D, float); n = len(D)
    return D.mean(0), D.std(0, ddof=1) / np.sqrt(n), D.std(0, ddof=1), n

def load_fd(sig, N):
    dd = {}
    for dr in ["../work_wake/out/fd", "out/fd"]:
        for fn in glob.glob(f"{dr}/fd_{sig}_{N}_*.json"):
            r = json.load(open(fn)); dd[r["seed"]] = r
    rs = list(dd.values())
    if not rs: return None
    offs = [0.0, 0.02, -0.02, 0.04, -0.04]
    O = {o: np.array([r["res"][repr(o)][T]["obs"] for r in rs]) for o in offs}
    Q = np.array([r["res"]["0.0"][T]["Q_trap"] for r in rs])
    sec = {h: (O[h] - O[-h]) / (2 * h) for h in offs if h > 0}
    fit = np.array([[np.polyfit(offs, [O[o][i, j] for o in offs], 3)[-2] for j in range(6)] for i in range(len(rs))])
    return dict(sec=sec, fit=fit, Q=Q, obs=O[0.0], n=len(rs))

def load_lr(tag):
    rs = [json.load(open(f)) for f in glob.glob(f"out/lr/{tag}_*.json")]
    rec = [next(c for c in r["recs"] if repr(c["t"]) == T) for r in rs]
    return np.array([c["dobs"] for c in rec]), np.array([c.get("om_rms_plasma", np.nan) for c in rec])

f = lambda a, b: " ".join("%8.3g±%-7.2g" % (x, y) for x, y in zip(a, b))
print("t =", T, " columns:", NAMES)
for tag, sig, N in [("c2_0.3_1", 0.3, 1), ("c1_0.3_1", 0.3, 1), ("c2_0.1_4", 0.1, 4), ("c2_0.03_4", 0.03, 4)]:
    D, om = load_lr(tag)
    if len(D) < 2: continue
    fd = load_fd(sig, N)
    m, se, sd, n = ms(D)
    print(f"\n{tag}: LR n={n}, median om_pl={np.median(om):.3g}")
    print("  LR     ", f(m, se))
    if fd:
        for key, arr in [("fit3", fd["fit"]), ("h.04", fd["sec"][0.04]), ("h.02", fd["sec"][0.02])]:
            fm, fse, fsd, fn = ms(arr)
            z = (m - fm) / np.sqrt(se ** 2 + fse ** 2)
            print(f"  FD {key} n={fn}", f(fm, fse))
            print("     z   ", " ".join("%17.2f" % v for v in z), "  power(|g|/SEc):", " ".join("%.1f" % v for v in np.abs(fm) / np.sqrt(se ** 2 + fse ** 2)))
        fm, fse, fsd, fn = ms(fd["fit"])
        print("  SD/|g| LR ", " ".join("%8.3g" % v for v in sd / np.abs(fm)), "   FD(fit) per seed", " ".join("%8.3g" % v for v in fsd / np.abs(fm)))
print("\n--- noisy-model gradient vs deterministic (N=1, h=0.04 and fit3), difference ± SE, and Q_trap")
base = load_fd("0.0", 1)
if base:
    for sig in ["0.0", "0.01", "0.03", "0.1"]:
        fd = load_fd(sig, 1)
        if not fd: continue
        for key in ["fit", 0.04, 0.02]:
            a = fd["fit"] if key == "fit" else fd["sec"][key]; b = base["fit"] if key == "fit" else base["sec"][key]
            ma, sa, _, na = ms(a); mb, sb, _, nb = ms(b)
            print(f"sigma={sig:5s} {str(key):5s} n={na:3d} g=", f(ma, sa), " rel.diff=", " ".join("%6.2f±%-5.2f" % (abs(x - y) / abs(y), np.hypot(sx, sy) / abs(y)) for x, y, sx, sy in zip(ma, mb, sa, sb)))
        q = fd["Q"]; print(f"   Q_trap = {q.mean():.5f} ± {q.std(ddof=1)/np.sqrt(len(q)):.5f}")
