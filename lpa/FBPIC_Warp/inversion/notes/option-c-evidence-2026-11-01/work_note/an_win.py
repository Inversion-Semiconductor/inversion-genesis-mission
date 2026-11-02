"""Late-window option C test (noise only for t >= 30): bias vs deterministic (paired), LR vs FD, growth."""
import glob, json, sys
import numpy as np

NAMES = ["logQ", "<u>", "<xi>", "lsu", "lsxi", "lsu/u"]
OFFS = [0.0, 0.02, -0.02, 0.04, -0.04]


def load_fd(sig, N):
    rs = {}
    for f in glob.glob(f"out/fd/w30_{sig}_{N}_*.json"):
        r = json.load(open(f)); rs[r["seed"]] = r
    return rs


def grads(rs, T, seeds):
    O = {o: np.array([rs[s]["res"][repr(o)][T]["obs"] for s in seeds]) for o in OFFS}
    sec4 = (O[0.04] - O[-0.04]) / 0.08
    fit = np.array([[np.polyfit(OFFS, [O[o][i, j] for o in OFFS], 3)[-2] for j in range(6)] for i in range(len(seeds))])
    Q = np.array([rs[s]["res"]["0.0"][T]["Q_trap"] for s in seeds])
    return sec4, fit, Q


def ms(a):
    a = np.asarray(a, float); return a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a)), a.std(0, ddof=1)


f = lambda a, b: " ".join("%9.3g±%-7.2g" % (x, y) for x, y in zip(a, b))
for T in ["45.0", "75.0"]:
    print(f"\n===== t = {T}  columns {NAMES}")
    for N in (1, 4):
        B = load_fd("0.0", N)
        print(f"-- bias, N={N}, noise only t>=30, paired with deterministic seeds")
        for sig in ["0.01", "0.03", "0.1", "0.3"]:
            S = load_fd(sig, N)
            seeds = sorted(set(S) & set(B))
            if len(seeds) < 3: continue
            for key in ("h.04", "fit"):
                gs = grads(S, T, seeds); gb = grads(B, T, seeds)
                a = gs[0] if key == "h.04" else gs[1]; b = gb[0] if key == "h.04" else gb[1]
                d = a - b; md, sd_, _ = ms(d); mb, _, _ = ms(b)
                print(f"  sig={sig:4s} n={len(seeds):2d} {key}: rel bias", " ".join("%6.2f±%-5.2f" % (x / abs(y), s / abs(y)) for x, s, y in zip(md, sd_, mb)))
            q = gs[2] / gb[2] - 1; mq, sq, _ = ms(q)
            print(f"      trapped charge change {100*mq:.1f} ± {100*sq:.1f} %")
    print("-- LR (mixed, LOO centered) vs windowed FD of the same noisy model")
    for tag in ["m30k1_0.03_1", "m30k1_0.03_4", "m30k1_0.1_1", "m30k1_0.1_4", "m30k03_0.1_4", "m30k1_0.3_1", "m30k1_0.3_4"]:
        sig, N = tag.split("_")[1], int(tag.split("_")[2])
        files = glob.glob(f"out/lr/{tag}_*.json")
        recs = [next(c for c in json.load(open(fl))["recs"] if repr(c["t"]) == T) for fl in files]
        D = np.array([c["dobs"] for c in recs])
        om = np.median([c["om_rms_plasma"] for c in recs]); omt = np.median([c["om_rms_trap"] for c in recs])
        S = load_fd(sig, N); seeds = sorted(S)
        sec4, fit, _ = grads(S, T, seeds)
        m, se, sd = ms(D); fm, fse, fsd = ms(fit)
        z = (m - fm) / np.hypot(se, fse); pw = np.abs(fm) / np.hypot(se, fse)
        print(f"\n  {tag}: n_LR={len(D)} n_FD={len(seeds)} median om_rms plasma {om:.3g} trapped {omt:.3g}")
        print("    LR      ", f(m, se)); print("    FD fit3 ", f(fm, fse))
        print("    z       ", " ".join("%17.2f" % v for v in z)); print("    power   ", " ".join("%17.1f" % v for v in pw))
        print("    SD/|g| LR one run ", " ".join("%8.3g" % v for v in sd / np.abs(fm)))
        print("    SD/|g| FD one seed", " ".join("%8.3g" % v for v in fsd / np.abs(fm)))

print("\n===== growth of om_rms (median over seeds), tlog columns t, plasma, bunch, max|dE|, median plasma tangent")
for tag in ["m30k1_0.03_1", "m30k1_0.03_4", "m30k1_0.1_1", "m30k1_0.1_4", "m30k03_0.1_4", "m30k1_0.3_1", "m30k1_0.3_4"]:
    L = np.array([json.load(open(fl))["tlog"] for fl in glob.glob(f"out/lr/{tag}_*.json")])
    med = np.median(L, axis=0); t = med[:, 0]
    sel = lambda a, b: (t >= a) & (t <= b)
    rate = lambda col, a, b: np.polyfit(t[sel(a, b)], np.log(med[sel(a, b), col]), 1)[0]
    pick = [np.argmin(abs(t - x)) for x in (32, 45, 60, 75)]
    print(f"  {tag}: bunch om at t=32,45,60,75:", " ".join("%.3g" % med[k, 2] for k in pick),
          "| plasma:", " ".join("%.3g" % med[k, 1] for k in pick),
          "| max dE:", " ".join("%.3g" % med[k, 3] for k in pick),
          "| rate bunch [35,55] %.3f [55,75] %.3f, plasma [35,55] %.3f [55,75] %.3f" % (rate(2, 35, 55), rate(2, 55, 75), rate(1, 35, 55), rate(1, 55, 75)))
