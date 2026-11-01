#!/bin/sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P 30 -L1 sh -c 'python3 -I "$0" "$@"' < jobs.txt > results_all.jsonl 2> errors.log
echo done
