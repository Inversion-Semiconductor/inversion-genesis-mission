"""Integrable phase mixing (no chaos) vs collocated particle-pdf (Chung eqs. 36, 41, 42, 44).
Pendulum in a prescribed field: dx/dt=v, dv/dt=-theta*sin(x), theta=1, x periodic.
f0: warm Gaussian blob inside the island (x0=1.2, v0=0, sigma=0.25 in x and v).
Source: delta a = -sin(x) (exact). Observables at T: <v^2>, <cos x>, <S(v)> with S a smooth step.
Reference: Lagrangian tangent (integrable: grows ~t, unbiased) at large N, checked by CRN FD.
Usage: python3 -I pendulum_pdf.py"""
import numpy as np, sys, json
rng=np.random.default_rng(1)
N=int(sys.argv[1]) if len(sys.argv)>1 else 200000
dt=0.05; Tchk=[5,10,20,40]; nsteps=int(Tchk[-1]/dt+0.5)
L=2*np.pi
def wrap(x): return (x+np.pi)%L-np.pi
def obs(x,v):
    s=1/(1+np.exp(-(v-0.8)/0.1))
    return np.stack([v*v,np.cos(x),s])
def obs_grad(x,v):
    s=1/(1+np.exp(-(v-0.8)/0.1)); ds=s*(1-s)/0.1
    gx=np.stack([0*x,-np.sin(x),0*x]); gv=np.stack([2*v,0*v,ds]); return gx,gv
def init(n,seed):
    r=np.random.default_rng(seed)
    return 1.2+0.25*r.standard_normal(n), 0.25*r.standard_normal(n)
def primal(theta,x,v):
    out={}
    x=x.copy(); v=v.copy()
    for k in range(1,nsteps+1):
        v+= -theta*np.sin(x)*dt*0.5; x=x+v*dt; v+= -theta*np.sin(x)*dt*0.5
        t=k*dt
        for T in Tchk:
            if abs(t-T)<1e-9: out[T]=obs(wrap(x),v).mean(1)
    return out
def lagrangian(x,v):
    x=x.copy(); v=v.copy(); xx=np.zeros_like(x); xv=np.zeros_like(x); out={}
    for k in range(1,nsteps+1):
        # half kick tangent
        xv+= (-np.cos(x)*xx - np.sin(x))*dt*0.5; v+= -np.sin(x)*dt*0.5
        x=x+v*dt; xx=xx+xv*dt
        xv+= (-np.cos(x)*xx - np.sin(x))*dt*0.5; v+= -np.sin(x)*dt*0.5
        t=k*dt
        for T in Tchk:
            if abs(t-T)<1e-9:
                gx,gv=obs_grad(wrap(x),v); c=gx*xx+gv*xv
                out[T]=(c.mean(1), c.std(1)/np.sqrt(len(x)), np.median(np.hypot(xx,xv)))
    return out
def b2(z):  # quadratic B-spline, support |z|<1.5
    a=np.abs(z); return np.where(a<0.5,0.75-a*a,np.where(a<1.5,0.5*(1.5-a)**2,0.0))
def db2(z):
    a=np.abs(z); s=np.sign(z); return np.where(a<0.5,-2*z,np.where(a<1.5,-s*(1.5-a),0.0))
def pdf_colloc(x,v,Nx,dv,vmax=3.0):
    """Collocated particle-pdf: H from d/dv of quadratic shape (x: linear), n_hat with linear x and
    quadratic v shapes (same v support as f), transfer with same shapes. Weights W_hat in units of W."""
    Nv=int(round(2*vmax/dv)); dx=L/Nx
    x=x.copy(); v=v.copy(); Wh=np.zeros_like(x); out={}; lost=[]
    for k in range(1,nsteps+1):
        v+= -np.sin(x)*dt*0.5; x=wrap(x+v*dt)
        # weight update at integer time (between half kicks), source delta a = -sin x
        gx=(x+np.pi)/dx-0.5; ix0=np.floor(gx).astype(int); fx=gx-ix0
        gvv=(v+vmax)/dv-0.5; iv0=np.round(gvv).astype(int)
        H=np.zeros(Nx*Nv); nh=np.zeros(Nx*Nv)
        da=-np.sin(x)
        for ox,wx in ((0,1-fx),(1,fx)):
            ixx=(ix0+ox)%Nx
            for ov in (-1,0,1):
                ivv=iv0+ov; ok=(ivv>=0)&(ivv<Nv)
                z=gvv-ivv
                key=ixx*Nv+np.clip(ivv,0,Nv-1)
                # dv f contribution: d/dv b2((v-v_c)/dv) = db2(z)/dv ; note H = da * d_v f
                H+=np.bincount(key[ok],weights=(-wx*da*db2(z)/dv)[ok],minlength=Nx*Nv)  # d/dv_node of b2((v_p-v_node)/dv) = -db2/dv
                nh+=np.bincount(key[ok],weights=(wx*b2(z))[ok],minlength=Nx*Nv)
        # NOTE: H computed with the particle's own da (delta a evaluated at particle x); mesh x-smoothing small.
        h_mesh=np.where(nh>1e-12,H/np.maximum(nh,1e-12),0.0)
        lost.append(np.abs(H[nh<=1e-12]).sum()/max(np.abs(H).sum(),1e-300))
        hs=np.zeros_like(x)
        for ox,wx in ((0,1-fx),(1,fx)):
            ixx=(ix0+ox)%Nx
            for ov in (-1,0,1):
                ivv=iv0+ov; ok=(ivv>=0)&(ivv<Nv); z=gvv-ivv
                key=ixx*Nv+np.clip(ivv,0,Nv-1)
                hs+=np.where(ok,wx*b2(z)*h_mesh[key],0.0)
        Wh+= -hs*dt
        v+= -np.sin(x)*dt*0.5
        t=k*dt
        for T in Tchk:
            if abs(t-T)<1e-9:
                c=obs(x,v)*Wh
                out[T]=(c.mean(1), c.std(1)/np.sqrt(len(x)), float(np.mean(lost)))
    return out
res={}
x0,v0=init(N,11)
lag=lagrangian(x0,v0)
# CRN FD reference
xr,vr=init(1000000,99)
fd={}
for hh in (0.01,0.003):
    p=primal(1+hh,xr,vr); m=primal(1-hh,xr,vr)
    fd[hh]={T:((p[T]-m[T])/(2*hh)).tolist() for T in Tchk}
lagbig=lagrangian(xr,vr)
print("observables: <v^2>, <cos x>, <S(v)> ; derivative wrt theta")
for T in Tchk:
    print(f"T={T:3d}: Lagrangian(N=1e6) {np.round(lagbig[T][0],4)} +- {np.round(lagbig[T][1],4)}  median|xi|={lagbig[T][2]:.2f};  CRN FD h=0.01 {np.round(fd[0.01][T],4)}, h=0.003 {np.round(fd[0.003][T],4)}")
for dv in (0.1,0.05,0.025):
    for Nx in (64,128):
        o=pdf_colloc(x0,v0,Nx,dv)
        for T in Tchk:
            ref=lagbig[T][0]
            print(f"pdf dv={dv:4.2f} Nx={Nx} T={T:3d}: {np.round(o[T][0],4)} +- {np.round(o[T][1],4)} ; rel.err {np.round((o[T][0]-ref)/np.abs(ref),2)} ; lost-source frac {o[T][2]:.3f}")
        res[f'{dv}_{Nx}']={T:o[T][0].tolist() for T in Tchk}
json.dump({"lag":{T:lagbig[T][0].tolist() for T in Tchk},"fd":fd,"pdf":res},open("pendulum_pdf.json","w"),indent=1)
