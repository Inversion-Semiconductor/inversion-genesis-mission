"""Per-dogbox-iteration and per-solve cost of Jacobian options on Twin-S (estimates).

All costs in primal Twin-S runs (86 s, plan 12.10 step 3). Measured inputs:
K=1 Lagrangian gradient 3.47 (c1 = 2.47); P5 without EV 1.96 (c1 = 0.96);
FD recovery 5 to 8 Jacobians per 1-D solve (plan 12.5).
"""
import numpy as np
from scipy.special import betainc

Ks = [1, 2, 5, 10, 25]
opts = {
    "central FD (h=0.02)": lambda K: 1 + 2 * K,
    "one-sided FD": lambda K: 1 + K,
    "R-v now (c1=2.47, reset 0.05-0.9)": lambda K: (1 + 2.47 * K + 0.05, 1 + 2.47 * K + 0.9),
    "R-v, deposit fixed (c1=1.0-1.3)": lambda K: (1 + 1.0 * K + 0.05, 1 + 1.3 * K + 0.9),
    "pure weights (c0=2-8, c1=0.7-0.8)": lambda K: (1 + 2 + 0.7 * K, 1 + 8 + 0.8 * K),
    "dipole pairs (gradf: 3+K*(1-2))": lambda K: (1 + 2 + 1 * K, 1 + 3 + 2 * K),
}
print("Run-equivalents per Jacobian (including the residual run at the iterate)")
print(f"{'option':38s}" + "".join(f"K={K:<9d}" for K in Ks))
for name, f in opts.items():
    cells = []
    for K in Ks:
        v = f(K)
        cells.append(f"{v:<11.1f}" if np.isscalar(v) else f"{v[0]:.1f}-{v[1]:<6.1f}")
    print(f"{name:38s}" + "".join(cells))
print("\nBreak-even with one-sided FD needs c0 + c1*K < K, i.e. c1 < 1 - c0/K.")
print("Lagrangian-carrying variants have c1 >= 0.96 (P5, no EV): never below one-sided FD.")
print("\nWall time per 1-D-style solve (5-8 Jacobians), K=25, hours, 1 GPU / 2 GPUs:")
for name, c in (("one-sided FD", 26), ("central FD", 51), ("pure weights best", 20.5), ("R-v now", 63.7)):
    lo, hi = 5 * c * 86 / 3600, 8 * c * 86 / 3600
    print(f"  {name:18s} {lo:5.1f}-{hi:5.1f} h  /  {lo/2:5.1f}-{hi/2:5.1f} h")
# null test of the cosine bar in d=11: P(cos >= c) = 0.5*I_{1-c^2}((d-1)/2, 1/2)
d = 11
for c in (0.5, 0.9):
    print(f"P(cos >= {c}) for a random direction in {d}-D: {0.5*betainc((d-1)/2, 0.5, 1-c*c):.2e}")
