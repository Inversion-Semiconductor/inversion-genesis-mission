"""Analyze Test B: CRN ensemble FD reference vs exact tangent vs GPT columns (R2 pilot, tf = 75).

Inputs: out/r2fd_s<seed>_a<a0>.json (primal obs), out/r2t_<mode>_s<seed>_p<ppc>.json (tangent runs).
"""
import glob, json, re, sys
import numpy as np

NAMES = ["logQ", "<u>", "<xi>", "log su", "log sxi", "log su/<u>"]
A0 = 2.5
od = sys.argv[1] if len(sys.argv) > 1 else "out"

# ---------------------------------------------------------------- FD reference
fd = {}
for f in glob.glob(f"{od}/r2fd_s*_a*.json"):
    m = re.search(r"r2fd_s(\d+)_a([0-9.]+)\.json", f)
    r = json.load(open(f))
    fd.setdefault(int(m.group(1)), {})[round(float(m.group(2)) - A0, 4)] = np.array(r["obs"])
seeds = sorted(s for s, d in fd.items() if len(d) == 9)
print("FD seeds complete:", len(seeds))
HS = (0.005, 0.01, 0.02, 0.04)
center = np.array([fd[s][0.0] for s in seeds])
sd_obs = center.std(0, ddof=1)
ref = {}
for h in HS:
    D = np.array([(fd[s][h] - fd[s][-h]) / (2 * h) for s in seeds])
    ref[f"FD h={h}"] = (D.mean(0), D.std(0, ddof=1) / np.sqrt(len(seeds)), D.std(0, ddof=1))
dA = np.array([-0.04, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04])
for deg, hmax in ((3, 0.04), (3, 0.02), (2, 0.02), (5, 0.04)):
    sel = np.abs(dA) <= hmax + 1e-9
    D = np.array([[np.polyfit(dA[sel], [fd[s][round(a, 4)][j] for a in dA[sel]], deg)[-2] for j in range(6)] for s in seeds])
    ref[f"fit deg{deg} |h|<={hmax}"] = (D.mean(0), D.std(0, ddof=1) / np.sqrt(len(seeds)), D.std(0, ddof=1))
# pooled fit to the ensemble mean (curvature-aware): cubic on the mean slice
mean_slice = np.array([np.mean([fd[s][round(a, 4)] for s in seeds], 0) for a in dA])

print("\nseed-to-seed SD of obs at a0=2.5:", np.round(sd_obs, 4))
print("\n%-22s" % "reference" + "".join("%16s" % n for n in NAMES))
for k, (m, se, sd) in ref.items():
    print("%-22s" % k + "".join("%8.3f+-%-6.3f" % (a, b) for a, b in zip(m, se)))

# ---------------------------------------------------------------- tangent runs
runs = {}
for f in glob.glob(f"{od}/r2t_*.json"):
    r = json.load(open(f))
    key = (r["mode"], r["ppc"])
    runs.setdefault(key, []).append(r)


def metrics(JA, JR):
    JA = JA / sd_obs; JR = JR / sd_obs
    cos = JA @ JR / np.linalg.norm(JA) / np.linalg.norm(JR)
    rho = np.linalg.norm(JA) / np.linalg.norm(JR)
    kap = abs(1 - (JA @ JR) / (JA @ JA))
    e = np.linalg.norm(JA - JR) / np.linalg.norm(JR)
    return cos, rho, kap, e


REF = sys.argv[2] if len(sys.argv) > 2 else "fit deg3 |h|<=0.04"
JR, JRse, _ = ref[REF]
print(f"\nColumns vs {REF}; z = |mean - ref| / sqrt(SE^2 + SEref^2) per observable; ok = within 2 SE + 20% |ref|")
print("%-34s %3s %5s" % ("mode", "ppc", "n") + "".join("%16s" % n for n in NAMES) + "   cos   rho  kappa  ecol  ok/6  t[s]")
rows = []
for (mode, ppc), rs in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    D = np.array([r["dobs"] for r in rs])
    m = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else np.full(6, np.nan)
    ok = np.abs(m - JR) <= 2 * np.sqrt(np.nan_to_num(se) ** 2 + JRse ** 2) + 0.2 * np.abs(JR)
    cos, rho, kap, e = metrics(m, JR)
    tt = np.mean([r["time"] for r in rs])
    print("%-34s %3d %5d" % (mode, ppc, len(rs)) + "".join("%8.3f+-%-6.3f" % (a, b) for a, b in zip(m, se))
          + "  %5.2f %5.2f %6.2f %5.2f  %d/6  %5.0f" % (cos, rho, kap, e, ok.sum(), tt))
    rows.append(dict(mode=mode, ppc=ppc, n=len(rs), mean=m.tolist(), se=se.tolist(), cos=cos, rho=rho, kappa=kap,
                     ecol=e, ok=int(ok.sum()), time=tt,
                     per_seed_cos=[metrics(np.array(r["dobs"]), JR)[0] for r in rs]))
json.dump(dict(ref={k: [v[0].tolist(), v[1].tolist()] for k, v in ref.items()}, sd_obs=sd_obs.tolist(), refkey=REF,
               rows=rows), open(f"{od}/r2_summary.json", "w"), indent=1)
