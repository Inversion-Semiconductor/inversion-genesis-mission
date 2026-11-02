#!/bin/bash
cd "$(dirname "$0")"
python -I maps_lr.py dbl 0.1 500000 40 out/dbl_s0.1_rep.json 9000 > out/dbl_s0.1_rep.log 2>&1
