"""Summarize Test A outputs (doubling map and standard map) into plain-text tables."""
import glob, json, os
import numpy as np

od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

print("=== Doubling map (A1) ===")
for case in ("U", "G"):
    for nf in (20, 30):
        print(f"\ncase {case}, n_f = {nf}")
        Ns = []
        for N in ("1e4", "1e5", "1e6"):
            f = f"{od}/dbl_{case}_{nf}_{N}.json"
            if os.path.exists(f):
                Ns.append((N, json.load(open(f))))
        if not Ns:
            continue
        print("  reference:", Ns[0][1]["ref"])
        keys = ["lag", "lmean_M64", "gpt_K2_T1", "gpt_K2_T3", "gpt_K2_T5", "gpt_K8_T1", "gpt_K8_T3", "gpt_K8_T5",
                "gpt_K16_T1", "fd_h0.01", "fd_h0.001"]
        print("  %-12s" % "estimator" + "".join("%30s" % f"N={N} mean+-SE (sd)" for N, _ in Ns))
        for k in keys:
            line = "  %-12s" % k
            for N, d in Ns:
                if k in d["summary"]:
                    m, se = d["summary"][k]; sd = se * np.sqrt(d["nseed"])
                    line += "%30s" % f"{m:+.4f}+-{se:.4f} ({sd:.3g})"
                else:
                    line += "%30s" % "-"
            print(line)

print("\n=== Standard map (A2) ===")
for case, nf in (("blob", 25), ("blob", 50), ("unif", 25)):
    rf = f"{od}/sm_ref_{case}_{nf}.json"
    if os.path.exists(rf):
        r = json.load(open(rf))
        print(f"\n{case} n_f={nf} reference CRN FD N=4e7:", {h: f"{v[0]:+.3f}+-{v[1]:.3f}" for h, v in r["fd"].items()})
    else:
        print(f"\n{case} n_f={nf} reference:", "exact 0" if case == "unif" else "missing")
    for N in ("1e5", "1e6"):
        rows = []
        for f in sorted(glob.glob(f"{od}/sm_{case}_{nf}_{N}_*.json")):
            d = json.load(open(f))
            for k, v in d["res"].items():
                rows.append((k, v))
        for k, v in rows:
            extra = f" rms {v['rms']:.3g}  {v['sec_per_run']:.1f}s/run" if "rms" in v else ""
            print(f"  N={N} {k:12s} {v['mean']:+9.4f} +- {v['se']:.4f} (sd {v['sd']:.4f}){extra}")
