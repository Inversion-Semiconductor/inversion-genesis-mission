import json, sys
import numpy as np
d = json.load(open(sys.argv[1]))
A = np.array(d["A"])
for vT in ("0.0", "0.3"):
    J = np.array([d[f"vT{vT}_seed11"], d[f"vT{vT}_seed12"]])  # (2, nA, 3)
    print("vT", vT, "T", d.get("T", 30))
    Jm = J.mean(0)
    noise = np.abs(J[0] - J[1]).mean(0) / np.sqrt(2)  # per-point std of one seed approx
    print("  per-point seed noise (one run)", noise.round(5))
    for half in (0.06, 0.03, 0.015):
        sel = np.abs(A - 1.0) <= half + 1e-9
        for s in range(2):
            sl = [np.polyfit(A[sel] - 1, J[s, sel, j], 3)[2] for j in range(3)]
            print(f"  cubic fit +-{half} seed{s}", np.round(sl, 3))
    # local central differences on the seed mean
    i0 = np.argmin(np.abs(A - 1.0))
    for k in (1, 2, 4, 8, 12):
        print(f"  CD h={k*0.005:.3f}", np.round((Jm[i0 + k] - Jm[i0 - k]) / (2 * k * 0.005), 3))
    print("  J(A) seed-mean, J1:", np.round(Jm[:, 0], 4).tolist())
    print("  J(A) seed-mean, J3:", np.round(Jm[:, 2], 4).tolist())
