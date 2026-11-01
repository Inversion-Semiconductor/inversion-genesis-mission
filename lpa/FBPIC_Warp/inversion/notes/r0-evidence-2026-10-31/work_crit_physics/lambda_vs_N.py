"""Sensitivity of the Lyapunov rate to N in an N-body-chaotic case (sec_tests R1 pilot, Debye model)
and the phase advance of betatron motion during the Twin-S growth window."""
import numpy as np
N=np.array([1e3,1e4,1e5]); lam=np.array([0.18,0.11,0.077])   # sec_tests.md Section 2.2
p=np.polyfit(np.log(N),np.log(lam),1)[0]
print(f"lambda ~ N^{p:.3f}; doubling N changes lambda by factor {2**p:.3f} ({(1-2**p)*100:.0f}% drop)")
print(f"quadrupling: factor {4**p:.3f}")
# betatron phase advance during steps 1200-1800 (16 wp0^-1), local density 1 -> 0
wp0dt=0.0266
t=np.arange(1200,1800)*1.0
zb=-10+0.05*t-12.0           # bunch position, um (12 um behind the laser)
S=np.where((zb>=10)&(zb<=66),np.exp(-np.log(2)*4*(zb-38)**2/28**2),0.0)
for g in (2,4,8,12):
    adv=np.sum(np.sqrt(S)/np.sqrt(2*g))*wp0dt
    print(f"gamma={g:2d}: betatron phase advance over steps 1200-1800 = {adv:.2f} rad = {adv/2/np.pi:.2f} periods")
