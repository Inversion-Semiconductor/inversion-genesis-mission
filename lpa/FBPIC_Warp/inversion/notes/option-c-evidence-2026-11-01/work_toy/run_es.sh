#!/bin/bash
cd "$(dirname "$0")"
python -I es1d_lr.py ens 0.1 32768 96 out/es_s0.1_N32k.json > out/es_s0.1_N32k.log 2>&1
python -I es1d_lr.py ens 0.1 131072 96 out/es_s0.1_N131k.json > out/es_s0.1_N131k.log 2>&1
python -I es1d_lr.py ens 0.03 131072 96 out/es_s0.03_N131k.json > out/es_s0.03_N131k.log 2>&1
python -I es1d_lr.py ens 0.01 131072 96 out/es_s0.01_N131k_fd.json 0.3,0.1 fd_only > out/es_s0.01_fd.log 2>&1
python -I es1d_lr.py ens 0.003 131072 96 out/es_s0.003_N131k_fd.json 0.3,0.1 fd_only > out/es_s0.003_fd.log 2>&1
python -I es1d_lr.py ens 0.1 524288 48 out/es_s0.1_N524k.json > out/es_s0.1_N524k.log 2>&1
echo done
