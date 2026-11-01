#!/bin/sh
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
xargs -P 24 -L1 sh -c 'python3 -I "$0" "$@"' < jobs75.txt > results_t75.jsonl 2> errors75.log
echo done
