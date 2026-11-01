cd "$(dirname "$0")"
for c in U G; do
 for nf in 20 30; do
  python3 -I testA_doubling.py $c $nf 1e4 16 out/dbl_${c}_${nf}_1e4.json > out/dbl_${c}_${nf}_1e4.log 2>&1 &
  python3 -I testA_doubling.py $c $nf 1e5 16 out/dbl_${c}_${nf}_1e5.json > out/dbl_${c}_${nf}_1e5.log 2>&1 &
  python3 -I testA_doubling.py $c $nf 1e6 8 out/dbl_${c}_${nf}_1e6.json 1,3,5 2,8 > out/dbl_${c}_${nf}_1e6.log 2>&1 &
 done
done
wait
