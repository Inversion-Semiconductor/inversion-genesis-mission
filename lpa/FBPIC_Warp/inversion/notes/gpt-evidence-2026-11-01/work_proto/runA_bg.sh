cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for c in U G; do
 for nf in 20 30; do
  python3 -I testA_doubling.py $c $nf 1e5 16 out/dbl_${c}_${nf}_1e5.json > out/dbl_${c}_${nf}_1e5.log 2>&1 &
  python3 -I testA_doubling.py $c $nf 1e6 8 out/dbl_${c}_${nf}_1e6.json 1,3,5 2,8 > out/dbl_${c}_${nf}_1e6.log 2>&1 &
 done
done
for nf in 25 50; do python3 -I testA_stdmap.py ref blob $nf 4e7 out/sm_ref_blob_$nf.json > out/sm_ref_blob_$nf.log 2>&1 & done
wait
