import json, numpy as np, collections
rows=[json.loads(l) for l in open('results_all.jsonl')]
g=collections.defaultdict(list)
for x in rows:
    if 'col' in x: g[(x['N'],x['M'],x['col'])].append((x['J'],x['dJ'],x['time']))
    elif 'NX' in x: print('vlasov', x)
for k,v in sorted(g.items()):
    v=np.array(v); n=len(v)
    print('N=%d M=%d col=%s n=%d  J %.3f  dJ mean %.3f sd %.3f se %.3f  time %.0f s'%(k[0],k[1],k[2],n,v[:,0].mean(),v[:,1].mean(),v[:,1].std(ddof=1),v[:,1].std(ddof=1)/n**.5,v[:,2].mean()))
