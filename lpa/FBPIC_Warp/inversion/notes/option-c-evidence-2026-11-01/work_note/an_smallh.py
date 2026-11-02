"""t=45 reference check: CRN secants of the late-window model (sigma=0.3, N=4) at h = 0.005 to 0.04."""
import glob, json
import numpy as np
for T in ["45.0", "75.0"]:
    A = {json.load(open(f))["seed"]: json.load(open(f)) for f in glob.glob("out/fd/w30_0.3_4_*.json")}
    B = {json.load(open(f))["seed"]: json.load(open(f)) for f in glob.glob("out/fd/w30s_0.3_4_*.json")}
    seeds = sorted(set(A) & set(B))
    print(f"t={T} n={len(seeds)}")
    for h, src in [(0.005, B), (0.01, B), (0.02, A), (0.04, A)]:
        s = np.array([(np.array(src[k]["res"][repr(h)][T]["obs"]) - np.array(src[k]["res"][repr(-h)][T]["obs"])) / (2 * h) for k in seeds])
        print(f"  h={h:5}", " ".join("%8.3g±%-6.2g" % (m, e) for m, e in zip(s.mean(0), s.std(0, ddof=1) / np.sqrt(len(s)))))
