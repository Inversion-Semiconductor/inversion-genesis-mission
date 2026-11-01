"""Error of collocated particle-pdf against the Lagrangian reference (integrable pendulum),
normalised by the RMS of the reference over T = 5, 10, 20, 40. Also bounce-phase spread."""
import json, numpy as np
from scipy.special import ellipk
d=json.load(open("pendulum_pdf.json"))
T=["5","10","20","40"]; lag=np.array([d["lag"][t] for t in T])
rms=np.sqrt((lag**2).mean(0))
print("reference RMS over T (v^2, cos x, S):",np.round(rms,2))
for key,val in d["pdf"].items():
    dv,nx=key.split("_"); p=np.array([val[t] for t in T])
    e=np.abs(p-lag)/rms
    print(f"dv={float(dv):5.3f} Nx={nx:>3}: |err|/RMS at T=20 {np.round(e[2],2)}, T=40 {np.round(e[3],2)}")
# bounce periods across the blob (x amplitude 1.2 +- 0.25, v=0): T_b = 4K(k^2), k = sin(a/2)
for a in (0.95,1.2,1.45):
    print(f"amplitude {a}: bounce period {4*ellipk(np.sin(a/2)**2):.2f}")
Tb=lambda a:4*ellipk(np.sin(a/2)**2)
for t in (5,10,20,40):
    print(f"T={t}: rotations {t/Tb(1.2):.1f}, bounce-phase spread across +-1 sigma {2*np.pi*t*(1/Tb(0.95)-1/Tb(1.45)):.2f} rad")
