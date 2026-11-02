"""Centered (leave-one-out) mean-field LR ensembles vs CRN FD of the noisy model (work_wake/out/fd)."""
import glob, json
import numpy as np
NAMES = ["logQ", "<u>", "<xi>", "log su", "log sxi", "log su/<u>"]
FD = "../work_wake/out/fd"
for T in ("30.0", "75.0"):
    print(f"=== t = {T}")
    for s, n, fdd, fn in (("0.01", 1, FD, 1), ("0.03", 1, FD, 1), ("0.1", 1, FD, 1), ("0.1", 4, FD, 4),
                          ("0.03", 4, "out/fd4", 4), ("0.03", 16, "out/fd4", 4)):
        lr = [json.load(open(f)) for f in glob.glob(f"out/ens/c2_{s}_{n}_*.json")]
        fd = [json.load(open(f)) for f in glob.glob(f"{fdd}/fd_{s}_{fn}_*.json")]
        if not lr or not fd: continue
        L = np.array([[r for r in d["recs"] if abs(r["t"] - float(T)) < 1e-9][0]["dobs"] for d in lr])
        for h in (0.02, 0.04):
            D = np.array([(np.array(d["res"][repr(h)][T]["obs"]) - np.array(d["res"][repr(-h)][T]["obs"])) / (2 * h) for d in fd])
            fm, fse = D.mean(0), D.std(0, ddof=1) / np.sqrt(len(D))
            print(f"  sigma={s} N={n}  FD(N={fn}) h={h} (n={len(D)}): " + " ".join("%9.3g+-%-8.2g" % (a, b) for a, b in zip(fm, fse)))
        lm, lse, lsd = L.mean(0), L.std(0, ddof=1) / np.sqrt(len(L)), L.std(0, ddof=1)
        print(f"  sigma={s} N={n}  LRc (n={len(L)}):         " + " ".join("%9.3g+-%-8.2g" % (a, b) for a, b in zip(lm, lse)))
        print(f"  sigma={s} N={n}  LRc SD/|FD h=0.04|:     " + " ".join("%9.3g          " % v for v in lsd / np.abs(fm)))
        print(f"  sigma={s} N={n}  z vs FD h=0.04:         " + " ".join("%9.2f          " % v for v in (lm - fm) / np.sqrt(lse ** 2 + fse ** 2)))
