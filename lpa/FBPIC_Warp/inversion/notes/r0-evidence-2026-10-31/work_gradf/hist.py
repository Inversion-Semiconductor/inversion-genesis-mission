import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import cold_beam_bias as m
rng=np.random.default_rng(0)
x,v=m.init(50000,rng,0.0,True)
for A in (0.95,1.0,1.05):
    xx,vv=m.push(x,v,0,m.NSTEP,A)
    h,e=np.histogram(vv,bins=np.arange(-2.5,4.01,0.25))
    print(A, ' '.join(f'{c/len(vv):.3f}' for c in h))
print(e)
