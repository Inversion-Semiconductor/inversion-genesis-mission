"""Markdown tables for sec_wake.md from out/summary_t<t>.json (run analyze.py first).

python -I tables.py <t>
"""
import json, sys
import numpy as np
T = sys.argv[1]
S = json.load(open(f"out/summary_t{T}.json"))
NAMES = ["logQ", "<u>", "<xi>", "log su", "log sxi", "log su/<u>"]
fd = {eval(k): v for k, v in S["fd"].items()}; lr = {eval(k): v for k, v in S["lr"].items()}
jt = {eval(k): v for k, v in S["joint"].items()}; ob = {eval(k): v for k, v in S["obs"].items()}
ref0 = fd[(0.0, 1, 0.04)]
f = lambda m, s: "%.3g ± %.2g" % (m, s)
print(f"\n#### t = {T}: noisy-model CRN FD (reference i), h = 0.04, and deterministic FD (reference ii)\n")
print("| sigma | N | " + " | ".join(NAMES) + " |"); print("|" + "---|" * (len(NAMES) + 2))
for k in sorted(fd, key=str):
    if k[2] in (0.04,):
        v = fd[k]; print(f"| {k[0]} | {k[1]} | " + " | ".join(f(a, b) for a, b in zip(v["mean"], v["se"])) + " |")
print(f"\n#### t = {T}: relative gradient bias |g_sigma - g_0| / |g_0| (both FD h = 0.04, N = 1); g_0 spread between h = 0.02 and 0.04 in the last row\n")
print("| sigma | " + " | ".join(NAMES) + " |"); print("|" + "---|" * (len(NAMES) + 1))
g0 = np.array(ref0["mean"])
for s in sorted({k[0] for k in fd if k[1] == 1}):
    if s == 0.0: continue
    g = np.array(fd[(s, 1, 0.04)]["mean"])
    print(f"| {s} | " + " | ".join("%.2f" % v for v in np.abs(g - g0) / np.abs(g0)) + " |")
g02 = np.array(fd[(0.0, 1, 0.02)]["mean"])
print("| 0 (h=0.02 vs 0.04) | " + " | ".join("%.2f" % v for v in np.abs(g02 - g0) / np.abs(g0)) + " |")
print(f"\n#### t = {T}: observables (ensemble mean at a0 = 2.5), N = 1\n")
print("| sigma | Q_trap | " + " | ".join(NAMES) + " |"); print("|" + "---|" * (len(NAMES) + 2))
for k in sorted(ob):
    if k[1] != 1: continue
    v = ob[k]; print(f"| {k[0]} | {v['Q']:.4f} | " + " | ".join("%.3g" % a for a in v["mean"]) + " |")
print(f"\n#### t = {T}: mean-field LR vs noisy FD (fit3, same sigma and N if present, else N = 1); z = |LR - FD| / sqrt(SE_LR^2 + SE_FD^2); single-run SD / |g|\n")
print("| sigma | N | runs | quantity | " + " | ".join(NAMES) + " |"); print("|" + "---|" * (len(NAMES) + 4))
for k in sorted(lr):
    v = lr[k]; key = (k[0], k[1], "fit3") if (k[0], k[1], "fit3") in fd else (k[0], 1, "fit3")
    if key not in fd: continue
    r = fd[key]; m, se, sd = map(np.array, (v["mean"], v["se"], v["sd"])); rm, rse = np.array(r["mean"]), np.array(r["se"])
    z = np.abs(m - rm) / np.sqrt(se ** 2 + rse ** 2)
    print(f"| {k[0]} | {k[1]} | {v['n']} | LR mean ± SE | " + " | ".join(f(a, b) for a, b in zip(m, se)) + " |")
    print(f"| | | | FD (N={key[1]}) | " + " | ".join(f(a, b) for a, b in zip(rm, rse)) + " |")
    print(f"| | | | z | " + " | ".join("%.1f" % a for a in z) + " |")
    print(f"| | | | SD/abs(g) | " + " | ".join("%.2g" % a for a in sd / np.abs(rm)) + " |")
    if k in jt:
        j = jt[k]; print(f"| | | | joint SD/abs(g) | " + " | ".join("%.2g" % a for a in np.array(j["sd"]) / np.abs(rm)) + " |")
