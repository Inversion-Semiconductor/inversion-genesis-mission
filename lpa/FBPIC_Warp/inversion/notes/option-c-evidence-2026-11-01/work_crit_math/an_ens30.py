"""All centered sigma=0.03, N=4 runs at t=30 (out/ens + out/ens30) vs CRN FD at N=4 (out/fd4) and N=1."""
import glob, json
import numpy as np
T = "30.0"
L = []
for f in glob.glob("out/ens/c2_0.03_4_*.json") + glob.glob("out/ens30/c2_0.03_4_*.json"):
    d = json.load(open(f)); L.append([r for r in d["recs"] if abs(r["t"] - 30.0) < 1e-9][0]["dobs"])
L = np.array(L)
for fdd, fn in (("out/fd4", 4), ("../work_wake/out/fd", 1)):
    fd = [json.load(open(f)) for f in glob.glob(f"{fdd}/fd_0.03_{fn}_*.json")]
    D = np.array([(np.array(d["res"]["0.04"][T]["obs"]) - np.array(d["res"]["-0.04"][T]["obs"])) / 0.08 for d in fd])
    print(f"FD N={fn} h=0.04 n={len(D)}: " + " ".join("%8.3g+-%-7.2g" % (a, b) for a, b in zip(D.mean(0), D.std(0, ddof=1) / np.sqrt(len(D)))))
m, se = L.mean(0), L.std(0, ddof=1) / np.sqrt(len(L))
med = np.median(L, 0)
print(f"LRc N=4 n={len(L)}:      " + " ".join("%8.3g+-%-7.2g" % (a, b) for a, b in zip(m, se)))
print("LRc median:         " + " ".join("%8.3g         " % v for v in med))
print("z vs FD N=4:        " + " ".join("%8.2f         " % v for v in (m - D.mean(0)) / se))
# bootstrap-free robustness: fraction of seeds below the FD value
fd4 = [json.load(open(f)) for f in glob.glob("out/fd4/fd_0.03_4_*.json")]
G = np.array([(np.array(d["res"]["0.04"][T]["obs"]) - np.array(d["res"]["-0.04"][T]["obs"])) / 0.08 for d in fd4]).mean(0)
print("frac LR < FD(N=4):  " + " ".join("%8.2f         " % v for v in (L < G).mean(0)))
