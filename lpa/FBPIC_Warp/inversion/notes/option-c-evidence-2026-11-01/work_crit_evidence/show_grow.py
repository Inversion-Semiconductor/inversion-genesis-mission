import json, glob, sys
import numpy as np
for f in sorted(glob.glob(sys.argv[1])):
    r = json.load(open(f)); tl = np.array(r["tlog"])
    t = tl[:, 0]; om = tl[:, 1]; ob = tl[:, 2]
    pick = [10, 20, 30, 45, 60, 75]
    vals = [om[np.argmin(abs(t - p))] for p in pick]
    def rate(a, b):
        m = (t >= a) & (t <= b); return np.polyfit(t[m], np.log(np.maximum(om[m], 1e-300)), 1)[0]
    rec = {repr(c["t"]): c for c in r["recs"]}
    d75 = rec["75.0"]["dobs"]; d30 = rec["30.0"]["dobs"]
    print(f.split("/")[-1].ljust(24), " ".join("%9.3g" % v for v in vals), " r10-30=%.3f r30-75=%.3f" % (rate(10, 30), rate(30, 75)), " bunch75=%.3g" % ob[-1], " d30=", " ".join("%.3g" % v for v in d30[:3]), " d75=", " ".join("%.3g" % v for v in d75[:3]))
