import sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import cold_beam_bias as m
t=time.time(); print(m.J_of_A(1.0, 400000, 1, 0.0), time.time()-t)
t=time.time(); c,g=m.lagrangian(2000,2,0.0); print('lag', c.mean(1), g, time.time()-t)
t=time.time(); cp=m.pairs(2000,2,0.0,[1e-6, 1e-3], every=1); print('pair', cp.mean(2), time.time()-t)
t=time.time(); cc,lm,lx=m.collocated(2000,2,0.0,64,0.1); print('coll', cc.mean(1), lm, lx, time.time()-t)
