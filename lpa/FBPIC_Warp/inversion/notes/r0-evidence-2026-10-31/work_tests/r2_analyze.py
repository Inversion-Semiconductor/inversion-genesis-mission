import json, numpy as np
rows=[json.loads(l) for l in open('results_all.jsonl')]
names=['logQ','<u>','<xi>','log su','log sxi','log su/<u>']
sl=sorted([r for r in rows if 'a0' in r and r['seed'] is None], key=lambda r:r['a0'])
A=np.array([r['a0'] for r in sl]); O=np.array([r['obs'] for r in sl])
grid=np.isclose((A-2.45)/0.005, np.round((A-2.45)/0.005)) & (A<=2.55001)
Ag, Og = A[grid], O[grid]
fit=np.array([np.polyval(np.polyder(np.polyfit(Ag-2.5, Og[:,j], 3)), 0) for j in range(6)])
resid=np.array([np.std(Og[:,j]-np.polyval(np.polyfit(Ag-2.5, Og[:,j], 3), Ag-2.5)) for j in range(6)])
c=O[np.isclose(A,2.5,rtol=0,atol=1e-9)][0]
def fd(h):
    p=O[np.isclose(A,2.5+h,rtol=0,atol=1e-9)]; m=O[np.isclose(A,2.5-h,rtol=0,atol=1e-9)]
    return (p[0]-m[0])/(2*h) if len(p) and len(m) else None
tw=[r for r in json.load(open('twin_a25_h1e-9.json')) if abs(r['t']-58)<1e-9][0]; tw7=[r for r in json.load(open('twin_a25_h1e-7.json')) if abs(r['t']-58)<1e-9][0]
print('t twin', tw['t'])
print('%-12s %9s %9s %9s %9s %9s %9s %9s %9s'%('obs','cubic','fitres','FD1e-2','FD5e-3','FD1e-3','FD1e-4','FD1e-5','exact'))
for j in range(6):
    vals=[fit[j],resid[j]]+[ (fd(h)[j] if fd(h) is not None else np.nan) for h in (0.01,0.005,1e-3,1e-4,1e-5)]+[tw['dobs'][j]]
    print('%-12s'%names[j]+' '.join('%9.3f'%v for v in vals))
print('exact h=1e-7 twin', np.round(tw7['dobs'],3))
seeds=[r for r in rows if 'a0' in r and r['seed'] is not None]
if seeds:
    S={}
    for r in seeds: S.setdefault(r['seed'],{})[round(r['a0'],4)]=np.array(r['obs'])
    sec=np.array([(d[2.52]-d[2.48])/0.04 for d in S.values() if 2.52 in d and 2.48 in d])
    lev=np.array([d[2.48] for d in S.values() if 2.48 in d])
    print('n seeds', len(sec))
    print('seed-to-seed sd of obs at 2.48', np.round(lev.std(0,ddof=1),4))
    print('per-seed secant h=0.02: mean', np.round(sec.mean(0),3), 'sd', np.round(sec.std(0,ddof=1),3))
    se_target=0.05*np.abs(fit)
    print('seeds needed for SE<=5% of |cubic slope|:', np.ceil((sec.std(0,ddof=1)/se_target)**2))
    np.save('r2_whiten_sd.npy', lev.std(0,ddof=1))

# column metrics, whitened by seed sd
sd=np.load('r2_whiten_sd.npy')
def met(JA,JR,r):
    delta=abs((JA-JR)@r)/abs(JA@JR); cos=JA@JR/np.linalg.norm(JA)/np.linalg.norm(JR)
    rho=np.linalg.norm(JA)/np.linalg.norm(JR); kap=abs(1-(JA@JR)/(JA@JA)); e=np.linalg.norm(JA-JR)/np.linalg.norm(JR)
    return 'delta=%.3g cos=%.3f rho=%.2f ecol=%.2f kappa=%.3f'%(delta,cos,rho,e,kap)
JR=fit/sd
# synthetic target at a0*=2.53 from the cubic fit
vt=np.array([np.polyval(np.polyfit(Ag-2.5, Og[:,j], 3), 0.03) for j in range(6)])
r=(c-vt)/sd
print('whitened |r| at 2.5:', np.linalg.norm(r).round(2), ' |J_ref|:', np.linalg.norm(JR).round(1))
print('exact (twin 1e-9) vs cubic:', met(np.array(tw['dobs'])/sd, JR, r))
for h in (0.01,0.005,1e-3,1e-4):
    v=fd(h)
    if v is not None: print('FD h=%g vs cubic:'%h, met(v/sd, JR, r))
