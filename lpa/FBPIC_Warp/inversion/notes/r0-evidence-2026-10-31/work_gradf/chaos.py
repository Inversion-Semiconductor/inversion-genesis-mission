import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import cold_beam_bias as m
for B,OM in ((0.12,2.6),(0.2,2.2),(0.3,2.2)):
    m.B=B; m.OM=OM
    rng=np.random.default_rng(0)
    n=20000
    x,v=m.init(n,rng,0.0)
    xx,xv=np.zeros(n),np.zeros(n)
    out=[]
    for k in range(m.NSTEP):
        t=k*m.DT
        xv=xv+0.5*m.DT*(m.dforce_dx(x,t,1.0)*xx-np.sin(x)); v=v+0.5*m.DT*m.force(x,t,1.0)
        x=x+m.DT*v; xx=xx+m.DT*xv
        xv=xv+0.5*m.DT*(m.dforce_dx(x,t+m.DT,1.0)*xx-np.sin(x)); v=v+0.5*m.DT*m.force(x,t+m.DT,1.0)
        if (k+1)%150==0:
            a=np.hypot(xx,xv); out.append(np.percentile(a,[50,90,99,100]).round(1).tolist())
    h,_=np.histogram(v,bins=np.arange(-3,4.01,0.5))
    print(B,OM,out); print('  vhist',(h/n).round(3).tolist())
