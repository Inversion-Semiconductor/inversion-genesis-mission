import json, sys
import numpy as np
for f in sys.argv[1:]:
    r = json.load(open(f)); print(f, r['mode'], r['sigma'], r['noise'], 'drop', r['drop'])
    for c in r['recs']:
        keys = [k for k in ('Q_trap','plasma_du_rms','om_rms_trap','om_rms_plasma','tan_med_trap','tan_p90_trap','tan_med_pl','tan_p90_pl','Sj') if k in c]
        print(' t=%4.0f' % c['t'], 'obs', np.round(c['obs'], 3), 'dobs', np.round(c.get('dobs', []), 3),
              ' '.join('%s=%.3g' % (k, c[k]) for k in keys), 'max|dE|=%.3g' % np.max(np.abs(c.get('dE_s', [0]))))
