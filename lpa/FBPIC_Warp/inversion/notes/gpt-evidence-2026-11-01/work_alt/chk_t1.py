import sys; sys.path.insert(0, "inversion/notes/gpt-evidence-2026-11-01/work_alt")
import numpy as np, t1_sheet as M
a=np.array([0.3,1.7,4.0]); n=200
x,v,t,ta,q,qa=M.run_tangents(a,M.A0,n)
h=1e-6
xp,vp,*_=M.run_tangents(a+h,M.A0,n); xm,vm,*_=M.run_tangents(a-h,M.A0,n)
print('t',t[0],(xp-xm)/2/h, t[1],(vp-vm)/2/h)
_,_,tp,_,qp,_=M.run_tangents(a+h,M.A0,n); _,_,tm,_,qm,_=M.run_tangents(a-h,M.A0,n)
print('ta',ta[0],(tp[0]-tm[0])/2/h, ta[1],(tp[1]-tm[1])/2/h)
print('qa',qa[0],(qp[0]-qm[0])/2/h, qa[1],(qp[1]-qm[1])/2/h)
xA,vA,*_=M.run_tangents(a,M.A0+h,n); xB,vB,*_=M.run_tangents(a,M.A0-h,n)
print('q',q[0],(xA-xB)/2/h,q[1],(vA-vB)/2/h)
# identity check per particle
e=M.estimators(a,M.A0,n)
