import json, sys
for f in sys.argv[1:]:
    for l in open(f):
        d = json.loads(l); vw = d['var_omega_t']
        print(d['variant'], d['N'], d['sigma'], 'LR', [f'{m:.3g}+-{s:.2g}' for m, s in zip(d['lr_mean'], d['lr_se'])],
              'FD', [f'{m:.3g}+-{s:.2g}' for m, s in zip(d['fd_mean'], d['fd_se'])], 'Varw t=3,9,15,30:', [f'{vw[i]:.3g}' for i in (0, 2, 4, 9)])
