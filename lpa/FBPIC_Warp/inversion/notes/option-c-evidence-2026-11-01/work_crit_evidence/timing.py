import os, time, sys
os.environ["NOJOINT"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wake_lr_c as W
res = {}
for rep in range(3):
    for name, mode, c in [("primal", "primal", 0), ("path", "path", 0), ("lr", "lr", 0), ("lr_c2", "lr", 2)]:
        W.CENTER = c
        t0 = time.process_time(); W.run(2.5, 7, 0.3, mode, "all", 1, 75.0); dt = time.process_time() - t0
        res.setdefault(name, []).append(dt)
import numpy as np
p = np.median(res["primal"])
for k, v in res.items(): print(k, "cpu s", np.round(v, 2), "median/primal %.2f" % (np.median(v) / p))
