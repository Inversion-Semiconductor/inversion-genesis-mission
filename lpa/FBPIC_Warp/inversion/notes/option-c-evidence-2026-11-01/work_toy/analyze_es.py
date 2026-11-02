"""Summarize es1d_lr.py ensembles: mean +- SE of mf, jnt, tan, fd(h); z-scores against FD.

Usage: python -I analyze_es.py out/es_*.json
"""
import json
import os
import sys

import numpy as np

OBS = ("K", "m1", "s1")


def ms(a):
    a = np.asarray(a, float)
    return a.mean(axis=0), a.std(axis=0, ddof=1) / np.sqrt(a.shape[0])


for path in sys.argv[1:]:
    d = json.load(open(path))
    rows = d["rows"]; S = len(rows)
    hs = d["hs"]
    print(f"\n=== {path}: sigma {d['sigma']} N {d['N']} seeds {S} wall {d['time']:.0f}s")
    for tk in rows[0]["base"]:
        b = [r["base"][tk] for r in rows]
        J = np.array([o["J"] for o in b])
        line = [f"{tk:4s} J={np.round(J.mean(0), 4)} sdJ={np.round(J.std(0, ddof=1), 4)}"]
        if "omrms" in b[0]:
            line.append(f"omrms med {np.median([o['omrms'] for o in b]):.3g}")
        if "tanrms" in b[0]:
            line.append(f"tanrms med {np.median([o['tanrms'] for o in b]):.3g}")
        print("  " + "  ".join(line))
        est = {}
        if "mf" in b[0]:
            est["mf"] = np.array([o["mf"] for o in b])
            Om = np.array([o["Om"] for o in b])
            # joint LR with leave-one-out baseline
            Jl = (J.sum(0)[None, :] - J) / (S - 1)
            est["jnt"] = (J - Jl) * Om[:, None]
        if "tan" in b[0]:
            est["tan"] = np.array([o["tan"] for o in b])
        for h in hs:
            est[f"fd{h:g}"] = np.array([r[f"fd{h:g}"][tk] for r in rows])
        # reference: FD at the h with smallest SE among the two largest h values, report all
        refk = min([f"fd{h:g}" for h in sorted(hs)[-2:]], key=lambda k: np.mean(ms(est[k])[1]))
        if os.environ.get("REF_H"):
            refk = f"fd{float(os.environ['REF_H']):g}"
        rm, rs = ms(est[refk])
        for kk, a in est.items():
            m, s = ms(a)
            z = (m - rm) / np.sqrt(s ** 2 + rs ** 2) if kk != refk else np.zeros(3)
            sd = a.std(axis=0, ddof=1)
            print(f"    {kk:7s} " + "  ".join(f"{o}: {m[i]:+.4e} +- {s[i]:.1e} (sd {sd[i]:.1e}, z {z[i]:+.1f})"
                                           for i, o in enumerate(OBS)) + ("  [ref]" if kk == refk else ""))
