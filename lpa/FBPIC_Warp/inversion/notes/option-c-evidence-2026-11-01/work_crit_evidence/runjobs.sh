#!/bin/sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -P 16 -I{} sh -c '{} > /dev/null 2>>out/err.log' < jobs.txt
echo done > out/DONE
