#!/bin/bash
cd "$(dirname "$0")"
for s in 0.3 0.1 0.03 0.01; do python -I maps_lr.py dbl $s 500000 40 out/dbl_s$s.json > out/dbl_s$s.log 2>&1; done
for s in 0.3 0.1 0.03 0; do python -I maps_lr.py std $s 500000 40 out/std_s$s.json > out/std_s$s.log 2>&1; done
echo done
