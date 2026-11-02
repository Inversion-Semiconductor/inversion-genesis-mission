"""Analyze option C on the 1D wake: CRN FD of the noisy model, mean-field LR, joint LR, pathwise.

python -I analyze.py [t]   (t = record time, default 75)
Writes out/summary_t<t>.json.
"""
import glob, json, re, sys
import numpy as np

NAMES = ["logQ", "<u>", "<xi>", "log su", "log sxi", "log su/<u>"]
T = sys.argv[1] if len(sys.argv) > 1 else "75.0"
od = "out"


def ms(D):
    D = np.asarray(D)
    n = len(D)
    return D.mean(0), D.std(0, ddof=1) / np.sqrt(n), D.std(0, ddof=1), n


def fmt(m, se):
    return "".join("%9.3f+-%-7.3f" % (a, b) for a, b in zip(m, se))


fd = {}
for f in glob.glob(f"{od}/fd/fd_*.json"):
    r = json.load(open(f))
    key = (r["sigma"], r["nmult"])
    fd.setdefault(key, []).append(r)
summary = dict(t=T, fd={}, lr={}, joint={}, path={}, obs={})
print(f"=== record time t = {T}")
print("--- CRN FD of the noisy model (central, slope per seed, mean +- SE); obs mean; Q_trap")
for key in sorted(fd):
    rs = fd[key]
    offs = sorted(float(o) for o in rs[0]["res"])
    O = {o: np.array([r["res"][repr(o)][T]["obs"] for r in rs]) for o in offs}
    Q = np.array([r["res"]["0.0"][T]["Q_trap"] for r in rs])
    m0, se0, sd0, n = ms(O[0.0])
    summary["obs"][str(key)] = dict(mean=m0.tolist(), se=se0.tolist(), sd=sd0.tolist(), Q=float(Q.mean()), Qse=float(Q.std(ddof=1) / np.sqrt(n)))
    print("sigma=%-5s N=%-2d n=%-3d obs   " % (key[0], key[1], n) + fmt(m0, se0) + "  Q=%.4f" % Q.mean())
    for h in [o for o in offs if o > 0]:
        D = (O[h] - O[-h]) / (2 * h)
        m, se, sd, n = ms(D)
        summary["fd"][str(key + (h,))] = dict(mean=m.tolist(), se=se.tolist(), sd=sd.tolist(), n=n)
        print("sigma=%-5s N=%-2d h=%-5s FD  " % (key[0], key[1], h) + fmt(m, se))
    # pooled cubic fit of the ensemble-mean curve over |h| <= 0.04 (curvature aware)
    sel = [o for o in offs if abs(o) <= 0.04 + 1e-9]
    if len(sel) >= 5:
        S = np.array([[np.polyfit(sel, [O[o][i, j] for o in sel], 3)[-2] for j in range(6)] for i in range(len(rs))])
        m, se, sd, n = ms(S)
        summary["fd"][str(key + ("fit3",))] = dict(mean=m.tolist(), se=se.tolist(), sd=sd.tolist(), n=n)
        print("sigma=%-5s N=%-2d fit3       " % (key[0], key[1]) + fmt(m, se))

lr = {}
for f in glob.glob(f"{od}/lr/lr_*.json"):
    r = json.load(open(f)); lr.setdefault((r["sigma"], r["nmult"]), []).append(r)
print("--- mean-field LR (mean +- SE; single-run SD in brackets), joint LR with leave-one-out baseline")
for key in sorted(lr):
    rs = lr[key]
    rec = [next(c for c in r["recs"] if repr(c["t"]) == T) for r in rs]
    D = np.array([c["dobs"] for c in rec]); m, se, sd, n = ms(D)
    ok = np.all(np.isfinite(D), 1)
    om = np.median([c["om_rms_trap"] for c in rec]); omp = np.median([c["om_rms_plasma"] for c in rec])
    tm = np.mean([r["time"] for r in rs])
    summary["lr"][str(key)] = dict(mean=m.tolist(), se=se.tolist(), sd=sd.tolist(), n=n, om_trap=om, om_pl=omp, time=tm)
    print("sigma=%-5s N=%-2d n=%-3d LR  " % (key[0], key[1], n) + fmt(m, se) + " om_trap=%.3g om_pl=%.3g t=%.0fs" % (om, omp, tm))
    print("                       SD  " + "".join("%18.3g" % s for s in sd))
    if "Sj" in rec[0]:
        J = np.array([c["obs"] for c in rec]); S = np.array([c["Sj"] for c in rec]); GW = np.array([c["dobs_W"] for c in rec])
        b = (J.sum(0)[None] - J) / (n - 1)
        G = (J - b) * S[:, None] + GW
        m2, se2, sd2, _ = ms(G)
        summary["joint"][str(key)] = dict(mean=m2.tolist(), se=se2.tolist(), sd=sd2.tolist(), n=n, sdS=float(S.std()))
        print("                    joint  " + fmt(m2, se2) + " sd(S)=%.3g" % S.std())

pa = {}
for f in glob.glob(f"{od}/path/p_*.json"):
    r = json.load(open(f)); pa.setdefault((r["sigma"], r["nmult"]), []).append(r)
print("--- pathwise tangent of the noisy model (per-realization exact derivative)")
for key in sorted(pa):
    rs = pa[key]
    rec = [next(c for c in r["recs"] if repr(c["t"]) == T) for r in rs]
    D = np.array([c["dobs"] for c in rec]); m, se, sd, n = ms(D)
    med = np.median([c["tan_med_trap"] for c in rec]); medp = np.median([c["tan_med_pl"] for c in rec])
    summary["path"][str(key)] = dict(mean=m.tolist(), se=se.tolist(), sd=sd.tolist(), n=n, tan_med_trap=med, tan_med_pl=medp)
    print("sigma=%-5s N=%-2d n=%-3d path" % (key[0], key[1], n) + fmt(m, se) + " |xi|med trap=%.3g pl=%.3g" % (med, medp))
json.dump(summary, open(f"{od}/summary_t{T}.json", "w"), indent=1)
