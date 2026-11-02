#!/bin/bash
cd "$(dirname "$0")"
python -I maps_lr.py dbl 0.1 500000 40 out/dbl_s0.1_rep2.json 11000 > out/dbl_s0.1_rep2.log 2>&1
