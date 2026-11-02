"""Growth of the LR weight rms with and without field feedback (out/grow, t <= 30, seed 1)."""
import json
import numpy as np
print("sigma N | rate of ln om_rms(plasma), t in [10,30]: feedback / no-feedback | om_pl(30) fb / nofb | sigma*om_nofb(30)")
for s in ('0.01', '0.03', '0.1', '0.3', '1.0'):
    for m in ((1, 4) if s == '0.01' else (1, 4, 16)):
        row = []
        for fb in ('g', 'n'):
            L = np.array(json.load(open(f'out/grow/{fb}_{s}_{m}.json'))['tlog'])
            sel = (L[:, 0] >= 10) & (L[:, 0] <= 30)
            row.append((np.polyfit(L[sel, 0], np.log(L[sel, 1]), 1)[0], L[-1, 1]))
        print("%-5s %-2d | %6.3f / %6.3f | %9.3g / %9.3g | %.3g" % (s, m, row[0][0], row[1][0], row[0][1], row[1][1], float(s) * row[1][1]))
