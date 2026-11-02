"""CRN finite differences of the noisy model: one seed, all a0 offsets, obs at REC_T.

python -I fd_batch.py <sigma> <noise> <nmult> <seed> <out.json> [offsets comma list]
"""
import sys, json
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import wake_lr as W

sigma = float(sys.argv[1]); noise = sys.argv[2]; nmult = int(sys.argv[3]); seed = int(sys.argv[4]); outp = sys.argv[5]
import os
if os.path.exists(outp):
    sys.exit(0)
offs = [float(v) for v in sys.argv[6].split(",")] if len(sys.argv) > 6 else [0.0, 0.01, -0.01, 0.02, -0.02, 0.04, -0.04, 0.08, -0.08]
res = {}
for o in offs:
    r = W.run(2.5 + o, seed, sigma, "primal", noise, nmult)
    res[repr(o)] = {repr(c["t"]): dict(obs=c["obs"], Q_trap=c["Q_trap"]) for c in r["recs"]}
json.dump(dict(sigma=sigma, noise=noise, nmult=nmult, seed=seed, res=res), open(outp, "w"))
