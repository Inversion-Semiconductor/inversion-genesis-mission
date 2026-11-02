#!/bin/bash
# Key ensemble: sigma = 0.03, N = 524288, stronger driver, long horizon (tangent explodes; LR weights grow slower).
cd "$(dirname "$0")"
ES_EPS=1e-2 ES_TREC=20,40,60,80,100 python -I es1d_lr.py ens 0.03 524288 32 out/es_long_s0.03_N524k.json 0.3,0.1 > out/es_long_s0.03_N524k.log 2>&1
echo done
