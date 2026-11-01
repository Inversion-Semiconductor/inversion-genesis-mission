"""Verify the r2_tan exact tangent against central FD (h=1e-6) at short horizons."""
import sys, time, json
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from r2_tan import run
tf = float(sys.argv[1]); ppc = int(sys.argv[2])
t0 = time.time(); r = run(2.5, ppc=ppc, tf=tf, mode="exact"); te = time.time() - t0
t0 = time.time(); rp = run(2.5 + 1e-6, ppc=ppc, tf=tf, mode="none"); tp = time.time() - t0
rm = run(2.5 - 1e-6, ppc=ppc, tf=tf, mode="none")
fd = (np.array(rp["obs"]) - np.array(rm["obs"])) / 2e-6
print("tf", tf, "ppc", ppc, "nb", r["nb"], "ne", r["ne"], "time exact %.1fs primal %.1fs" % (te, tp))
print("obs  ", np.round(r["obs"], 5))
print("exact", np.round(r["dobs"], 6))
print("FD   ", np.round(fd, 6))
print("relerr", np.abs(np.array(r["dobs"]) - fd) / (np.abs(fd) + 1e-12))
print("log", json.dumps(r["log"][-1]))
