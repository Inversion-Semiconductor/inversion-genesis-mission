"""Twin-S timeline: laser and bunch position vs step, local density, local plasma
and betatron frequencies, and the P7 tangent growth rates per interval.
Inputs: inversion/configs/twin_s.json values and the P7 tangent table (plan 12.10 step 4)."""
import numpy as np
c=2.99792458e8; e=1.602176634e-19; me=9.1093837e-31; eps0=8.8541878128e-12
n0=8e24; dz=0.05e-6; dt=dz/c
wp0=np.sqrt(n0*e**2/(eps0*me))
def S(z):
    zc,F,b=38e-6,28e-6,2.0
    s=np.exp(-np.log(2)*(4*(z-zc)**2/F**2)**(b/2))
    return np.where((z>=10e-6)&(z<=66e-6),s,0.0)
print(f"wp0={wp0:.3e} 1/s, wp0*dt={wp0*dt:.4f}")
steps=np.array([450,800,960,1000,1200,1350,1500,1520,1650,1800,2000])
for n in steps:
    zl=-10e-6+n*dz         # laser centroid (vg ~ c)
    for lag in (0.0,12e-6):
        z=zl-lag; s=float(S(z))
        print(f"step {n:5d} {'laser' if lag==0 else 'bunch(-12um)'}: z={z*1e6:6.1f} um, n/n0={s:.3f}, wp_loc/wp0={np.sqrt(s):.3f}")
# P7 infinity-norm table
st=np.array([1200,1500,1800,2000]); dux=np.array([2.3e6,1.5e8,8.7e10,1.7e11]); dx=np.array([22,53,1.6e4,1.3e5]); dE=np.array([5.6e3,2.3e4,2.0e6,1.6e7])
for a,b in zip(range(3),range(1,4)):
    L=st[b]-st[a]
    r=lambda q: np.log(q[b]/q[a])/L
    print(f"{st[a]}-{st[b]}: rate dux {r(dux):.4f}/step = {r(dux)/(wp0*dt):.2f} wp0 ; dx {r(dx):.4f}; dE {r(dE):.4f}")
# ballistic check of dx growth after 1800: dx(2000)-dx(1800) ~ c*dux/gamma*T
T=200*dt
for g in (6,12):
    print(f"ballistic dx increment with gamma={g}: {c*1.7e11/g*T:.2e} m/J (observed {1.3e5-1.6e4:.2e})")
# betatron frequency for bunch gamma at local density
for s in (0.6,0.3,0.1):
    for g in (2,6,12):
        print(f"n/n0={s}, gamma={g}: w_beta/wp0={np.sqrt(s)/np.sqrt(2*g):.3f}")
print("lambda (P7 headline) = ln10/120 per step =", np.log(10)/120, "=",np.log(10)/120/(wp0*dt),"wp0")
