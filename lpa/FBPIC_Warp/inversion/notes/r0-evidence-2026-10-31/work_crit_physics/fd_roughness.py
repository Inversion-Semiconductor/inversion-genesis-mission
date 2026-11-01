"""Structure of the P7 FD secants at N_t=2000 (plan 12.10 step 4 table, theta-hat units).
Rough-part amplitude A(h) = h*|D(h)-D(1e-2)| per residual; fit A ~ h^alpha (Hoelder exponent)."""
import numpy as np
h=np.array([1e-2,1e-3,1e-4,1e-5,1e-6,1e-7])
D=np.array([
[6.8,4.9,20.1,33.4,-8.9,77.4],
[24.1,24.0,12.6,30.1,-159,33.6],
[5.8,14.1,-18.4,-32.0,-230,409],
[-75.6,-75.4,-99.0,-147,-13.4,-92.7],
[5.4,4.0,-22.7,-42.8,-13.8,-149],
[38.3,39.3,30.7,98.6,-23.1,-62.9],
[37.5,43.1,35.8,78.6,-80.1,-140],
[38.9,46.6,59.3,37.4,21.2,38.4],
[38.9,46.2,57.5,13.5,-17.0,-15.4],
[41.6,51.0,60.4,112,60.9,64.1],
[22.2,33.8,72.3,190,126,494]])
ref=D[:,0]
A=np.abs(D-ref[:,None])*h[None,:]
med=np.median(A[:,1:],axis=0)
rms_col=np.sqrt(np.mean((D-ref[:,None])**2,axis=0))/np.sqrt(np.mean(ref**2))
for hh,m,r in zip(h[1:],med,rms_col[1:]):
    print(f"h={hh:.0e}: median rough amplitude h|D(h)-D(1e-2)| = {m:.2e}; rel. column deviation from h=1e-2: {r:.2f}")
p=np.polyfit(np.log(h[1:]),np.log(med),1)
print(f"fit A(h) ~ h^{p[0]:.2f} over h=1e-7..1e-3 (alpha=1: smooth; alpha=0: white noise)")
p2=np.polyfit(np.log(h[2:]),np.log(med[1:]),1)
print(f"fit over 1e-7..1e-4: alpha={p2[0]:.2f}")
# implied secant error at h=1e-2 if the power law continues
a=np.exp(np.polyval(p,np.log(1e-2)))
print(f"extrapolated rough amplitude at h=1e-2: {a:.2e}; implied secant noise {a/1e-2:.1f} vs median |D(1e-2)| {np.median(np.abs(ref)):.1f}")
