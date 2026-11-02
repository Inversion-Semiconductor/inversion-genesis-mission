"""Median over seeds of the weight rms (plasma, bunch) and max|dE| versus time for each (sigma, N) LR ensemble."""
import glob, json
import numpy as np
runs = {}
for f in glob.glob("out/lr/lr_*.json"):
    r = json.load(open(f)); runs.setdefault((r["sigma"], r["nmult"]), []).append(np.array(r["tlog"]))
ts = (10, 20, 30, 45, 60, 75)
print("sigma N  n | median om_rms plasma at t=" + ",".join(map(str, ts)) + " | bunch | median max|dE|")
for k in sorted(runs):
    L = runs[k]
    g = lambda c: [np.median([l[np.argmin(np.abs(l[:, 0] - t)), c] for l in L]) for t in ts]
    print("%-5s %-2d %-2d | %s | %s | %s" % (k[0], k[1], len(L), " ".join("%8.3g" % v for v in g(1)),
          " ".join("%8.3g" % v for v in g(2)), " ".join("%8.3g" % v for v in g(3))))
